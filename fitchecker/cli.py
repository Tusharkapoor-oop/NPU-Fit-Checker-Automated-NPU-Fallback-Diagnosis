"""
fitchecker/cli.py

Entry point for NPU Fit Checker.

Usage:
  python -m fitchecker run --model <path> --device "<device name>"
  python -m fitchecker run --from-profile <json>
  python -m fitchecker list-devices
  python -m fitchecker inspect-profile <json>
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

from fitchecker.profile_parser import load_profile, describe_schema
from fitchecker.rules import apply_rules
from fitchecker.batch import run_batch
from fitchecker.report import (
    build_markdown_report,
    print_terminal_report,
    save_report,
)


def _results_path(label: str) -> pathlib.Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return pathlib.Path("results") / f"report_{label}_{ts}.md"


# ---------------------------------------------------------------------------
# list-devices
# ---------------------------------------------------------------------------

def cmd_list_devices(args: argparse.Namespace) -> None:
    try:
        from fitchecker.hub_client import list_devices, QAI_HUB_AVAILABLE
    except ImportError:
        print("[ERROR] qai_hub not installed. Run: pip install qai-hub")
        sys.exit(1)

    if not QAI_HUB_AVAILABLE:
        print("[ERROR] qai_hub not installed. Run: pip install qai-hub")
        sys.exit(1)

    devices = list_devices()
    print(f"Available AI Hub devices ({len(devices)}):")
    for d in sorted(devices):
        print(f"  {d}")

    print(f"\nNote: To update devices.txt correctly, run scripts/phase0_setup.py")
    print(f"      (list-devices does not apply the Windows/Compute filter)")


# ---------------------------------------------------------------------------
# inspect-profile
# ---------------------------------------------------------------------------

def cmd_inspect_profile(args: argparse.Namespace) -> None:
    path = args.json_path
    if not pathlib.Path(path).exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    print(describe_schema(path))


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    baseline_profile = None

    if args.from_profile:
        print(f"[fitchecker] Offline mode: loading {args.from_profile}")
        profile = load_profile(args.from_profile)
        if args.device:
            profile.device_name = args.device
        if getattr(args, "model_name", None):
            profile.model_name = args.model_name
        if getattr(args, "job_id", None):
            profile.job_id = args.job_id
        label = pathlib.Path(args.from_profile).stem

    elif args.model:
        try:
            from fitchecker.hub_client import run_full_pipeline, QAI_HUB_AVAILABLE
        except ImportError:
            print("[ERROR] qai_hub not installed. Run: pip install qai-hub")
            sys.exit(1)

        if not QAI_HUB_AVAILABLE:
            print("[ERROR] qai_hub not installed.")
            sys.exit(1)

        if not args.device:
            print("[ERROR] --device required with --model.")
            sys.exit(1)

        # Enforce devices.txt (JSON lines format)
        try:
            from fitchecker.hub_client import validate_device_name
            validate_device_name(args.device)
        except FileNotFoundError:
            print("[WARNING] experiments/devices.txt not found. Run phase0_setup.py first.")
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_stem = pathlib.Path(args.model).stem
        save_path  = pathlib.Path("experiments") / f"{model_stem}_{ts}_profile.json"

        print(f"[fitchecker] Submitting '{args.model}' -> '{args.device}' ...")
        run_full_pipeline(
            model_path=args.model,
            device_name=args.device,
            save_path=str(save_path),
            compile_options=args.compile_options or "--target_runtime onnx",
            profile_options=args.options or "",
        )
        profile = load_profile(str(save_path), device_name=args.device, model_name=model_stem)
        label = model_stem

    else:
        print("[ERROR] Provide --model or --from-profile.")
        sys.exit(1)

    if args.baseline:
        if not pathlib.Path(args.baseline).exists():
            print(f"[WARNING] Baseline not found: {args.baseline}. Skipping comparison.")
        else:
            baseline_profile = load_profile(args.baseline)
            if args.device and not baseline_profile.device_name:
                baseline_profile.device_name = args.device

    findings = apply_rules(profile)
    print_terminal_report(profile, findings, baseline_profile=baseline_profile)

    # Determine labels from actual model/precision (not hardcoded)
    baseline_label  = args.baseline_label  if args.baseline_label  else "Baseline"
    optimized_label = args.optimized_label if args.optimized_label else "Optimized"

    report_md = build_markdown_report(
        profile, findings, baseline_profile=baseline_profile,
        title=f"NPU Fit Checker - {label}",
        baseline_label=baseline_label,
        optimized_label=optimized_label,
    )
    out_path = pathlib.Path(args.output) if args.output else _results_path(label)
    saved = save_report(report_md, str(out_path))
    print(f"\n[fitchecker] Report saved: {saved}")


# ---------------------------------------------------------------------------
# autofix
# ---------------------------------------------------------------------------

def cmd_autofix(args: argparse.Namespace) -> None:
    try:
        from fitchecker.autofix import run_autofix_loop
    except ImportError as e:
        print(f"[ERROR] Could not import autofix pipeline: {e}")
        sys.exit(1)

    if not args.model or not args.device:
        print("[ERROR] --model and --device are required for autofix.")
        sys.exit(1)

    try:
        from fitchecker.hub_client import validate_device_name
        validate_device_name(args.device)
    except FileNotFoundError:
        print("[WARNING] experiments/devices.txt not found. Run phase0_setup.py first.")
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    run_autofix_loop(
        onnx_model_path=args.model,
        device_name=args.device,
        output_dir=args.output_dir or "experiments/autofix_result"
    )

def cmd_batch(args: argparse.Namespace) -> None:
    if args.models:
        model_pairs = []
        for item in args.models:
            if ":" in item:
                path, name = item.split(":", 1)
            else:
                path = item
                import pathlib
                name = pathlib.Path(item).stem
            model_pairs.append((path, name))
    else:
        model_pairs = [
            ("mobilenet_v2.onnx", "MobileNetV2"),
            ("squeezenet1_0.onnx", "SqueezeNet"),
            ("resnet18.onnx", "ResNet18"),
        ]

    from fitchecker.batch import run_batch
    run_batch(
        models=model_pairs,
        device_name=args.device,
        compile_options=args.compile_options,
        profile_options=args.profile_options,
    )

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fitchecker",
        description="NPU Fit Checker - profile AI models on Qualcomm AI Hub.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    run_p = sub.add_parser("run", help="Profile a model and generate a report.")
    grp = run_p.add_mutually_exclusive_group()
    grp.add_argument("--model", "-m", help="Path to .onnx model file.")
    grp.add_argument("--from-profile", metavar="JSON",
                     help="Analyse a saved profile JSON (offline mode).")
    run_p.add_argument("--device", "-d", default="Snapdragon X2 Elite CRD",
                       help='AI Hub device name. Must be in experiments/devices.txt.')
    run_p.add_argument("--model-name", default="",
                       help="Model name to display in reports.")
    run_p.add_argument("--job-id", default="",
                       help="AI Hub job ID to display in reports.")
    run_p.add_argument("--options", default="",
                       help="AI Hub profile job options string.")
    run_p.add_argument("--compile-options", default="--target_runtime onnx",
                       help='AI Hub compile job options (default: "--target_runtime onnx").')
    run_p.add_argument("--baseline", metavar="JSON",
                       help="Baseline profile JSON for before/after comparison.")
    run_p.add_argument("--baseline-label", default="",
                       help="Label for baseline column (e.g. 'FP32 baseline').")
    run_p.add_argument("--optimized-label", default="",
                       help="Label for optimized column (e.g. 'INT8 quantized').")
    run_p.add_argument("--output", "-o",
                       help="Output path for Markdown report.")

    # list-devices
    sub.add_parser("list-devices", help="List available AI Hub devices.")

    # inspect-profile
    ip = sub.add_parser("inspect-profile", help="Print schema of a saved profile JSON.")
    ip.add_argument("json_path", help="Path to profile JSON.")
    
    # autofix
    af = sub.add_parser("autofix", help="Auto-quantize and re-profile a completely FP32-bound model.")
    af.add_argument("--model", "-m", required=True, help="Path to .onnx model file.")
    af.add_argument("--device", "-d", required=True, help='AI Hub device name.')
    af.add_argument("--output-dir", default="", help="Directory to save autofix reports and profiles.")

    # batch
    bp = sub.add_parser("batch", help="Run batch profiling across multiple models.")
    bp.add_argument("--device", "-d", default="Snapdragon X2 Elite CRD", help="Target device name.")
    bp.add_argument("--models", nargs="+", help="List of model paths or model_path:model_name pairs.")
    bp.add_argument("--compile-options", default="--target_runtime onnx", help="Compile options.")
    bp.add_argument("--profile-options", default="", help="Profile options.")

    return parser


class MasterLogger:
    def __init__(self, original_stream, log_file):
        self.original_stream = original_stream
        self.log_file = log_file

    def write(self, message):
        self.original_stream.write(message)
        self.original_stream.flush()
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(message)
        except Exception:
            pass

    def flush(self):
        self.original_stream.flush()


def main() -> None:
    import sys
    import pathlib
    import datetime
    
    log_dir = pathlib.Path("experiments")
    log_dir.mkdir(exist_ok=True)
    master_log = log_dir / "master_run_log.txt"
    
    with open(master_log, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"--- NEW RUN STARTED AT {datetime.datetime.now().isoformat()} ---\n")
        f.write(f"--- COMMAND: {' '.join(sys.argv)} ---\n")
        f.write(f"{'='*80}\n")
        
    sys.stdout = MasterLogger(sys.stdout, master_log)
    sys.stderr = MasterLogger(sys.stderr, master_log)
    
    try:
        parser = build_parser()
        args = parser.parse_args()

        if args.command == "run":
            cmd_run(args)
        elif args.command == "list-devices":
            cmd_list_devices(args)
        elif args.command == "inspect-profile":
            cmd_inspect_profile(args)
        elif args.command == "autofix":
            cmd_autofix(args)
        elif args.command == "batch":
            cmd_batch(args)
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
