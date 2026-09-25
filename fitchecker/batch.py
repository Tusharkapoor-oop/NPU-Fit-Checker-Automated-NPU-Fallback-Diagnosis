"""
fitchecker/batch.py

Batch profiling runner for NPU Fit Checker.
Runs the complete compilation and profiling pipeline across multiple ONNX models
to evaluate Qualcomm Hexagon NPU fit and generalizability.

Reuses run_full_pipeline() from hub_client.py UNCHANGED.
Never invents numbers; records exact job URLs or failure messages.
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import pathlib
import re
import sys
from typing import Optional

from fitchecker.hub_client import DEFAULT_PRIMARY_DEVICE, run_full_pipeline
from fitchecker.profile_parser import load_profile
from fitchecker.report import render_batch_summary_markdown


class OutputTee(io.StringIO):
    """Duplicates stdout to both terminal and an in-memory buffer."""

    def __init__(self, original_stdout):
        super().__init__()
        self.original_stdout = original_stdout

    def write(self, s):
        self.original_stdout.write(s)
        self.original_stdout.flush()
        return super().write(s)

    def flush(self):
        self.original_stdout.flush()
        super().flush()


def _append_to_evidence_log(
    model_name: str,
    model_path: str,
    status: str,
    compile_job_url: Optional[str] = None,
    profile_job_url: Optional[str] = None,
    stage_failed: Optional[str] = None,
    error_message: Optional[str] = None,
    metrics: Optional[dict] = None,
) -> None:
    """Appends batch run evidence to docs/EVIDENCE_LOG.md."""
    log_path = pathlib.Path("docs/EVIDENCE_LOG.md")
    if not log_path.exists():
        return

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = ["", f"### Batch Run: {model_name} ({ts})", f"- **Model Path**: `{model_path}`", f"- **Status**: `{status}`"]

    if compile_job_url:
        lines.append(f"- **Compile Job**: {compile_job_url}")
    if profile_job_url:
        lines.append(f"- **Profile Job**: {profile_job_url}")

    if status == "SUCCESS" and metrics:
        lines.append(f"- **Total Layers**: {metrics.get('total_layers')}")
        lines.append(f"- **NPU Layers**: {metrics.get('npu_layers')} ({metrics.get('npu_pct')})")
        lines.append(f"- **CPU Fallback Layers**: {metrics.get('cpu_layers')}")
        lines.append(f"- **Latency**: {metrics.get('latency')}")
        lines.append(f"- **Saved Profile**: `{metrics.get('profile_path')}`")
    elif status == "FAILED":
        lines.append(f"- **Failed Stage**: `{stage_failed}`")
        lines.append(f"- **Exact Error**: `{error_message}`")

    lines.append("")

    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def run_batch(
    models: list[tuple[str, str]],
    device_name: str = DEFAULT_PRIMARY_DEVICE,
    compile_options: str = "--target_runtime onnx",
    profile_options: str = "",
) -> list[dict]:
    """
    Executes batch profiling for a list of (model_path, model_name) pairs.
    Reuses run_full_pipeline() from hub_client.py UNCHANGED.

    Returns a list of result dictionaries suitable for render_batch_summary_markdown.
    """
    results: list[dict] = []
    batch_dir = pathlib.Path("experiments/batch")
    batch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"Starting NPU Fit Checker Batch Runner ({len(models)} models)")
    print(f"Target Device: {device_name}")
    print(f"Compile Options: {compile_options}")
    print(f"=======================================================\n")

    for idx, (m_path, m_name) in enumerate(models, 1):
        print(f"\n[{idx}/{len(models)}] Processing {m_name} ({m_path}) ...")
        m_file = pathlib.Path(m_path)

        if not m_file.exists():
            err_msg = f"Model file not found on disk: {m_path}"
            print(f"[ERROR] {err_msg}")
            res = {
                "model_name": m_name,
                "model_path": m_path,
                "status": "FAILED",
                "stage_failed": "file_check",
                "error_message": err_msg,
                "total_layers": "N/A",
                "npu_pct": "N/A",
                "cpu_layers": "N/A",
                "latency": "N/A",
                "link": "Local file missing",
            }
            results.append(res)
            _append_to_evidence_log(
                model_name=m_name,
                model_path=m_path,
                status="FAILED",
                stage_failed="file_check",
                error_message=err_msg,
            )
            continue

        model_out_dir = batch_dir / m_name
        model_out_dir.mkdir(parents=True, exist_ok=True)
        save_path = model_out_dir / "profile.json"

        # Capture output while teeing to stdout
        orig_stdout = sys.stdout
        tee = OutputTee(orig_stdout)
        sys.stdout = tee

        c_job_url: Optional[str] = None
        p_job_url: Optional[str] = None
        stage_failed: Optional[str] = None
        error_message: Optional[str] = None

        try:
            # Reuses run_full_pipeline UNCHANGED
            run_full_pipeline(
                model_path=str(m_file),
                device_name=device_name,
                save_path=str(save_path),
                compile_options=compile_options,
                profile_options=profile_options,
            )
        except Exception as exc:
            captured = tee.getvalue()
            # Determine stage of failure from captured trace
            if "Profile job:" in captured:
                stage_failed = "profile"
            elif "Compile job:" in captured:
                stage_failed = "compile"
            elif "upload" in str(exc).lower():
                stage_failed = "model_upload"
            else:
                stage_failed = "compile_submission"

            error_message = f"{type(exc).__name__}: {str(exc)}"
            print(f"\n[batch] FAILED at stage '{stage_failed}': {error_message}")
        finally:
            sys.stdout = orig_stdout

        captured_text = tee.getvalue()
        c_match = re.search(r"Compile job:\s+(https?://\S+)", captured_text)
        if c_match:
            c_job_url = c_match.group(1).rstrip("/.,")

        p_match = re.search(r"Profile job:\s+(https?://\S+)", captured_text)
        if p_match:
            p_job_url = p_match.group(1).rstrip("/.,")

        if stage_failed is not None:
            # Failure handling: record exact error and stage, no silent retries
            link_str = p_job_url or c_job_url or f"Failed at {stage_failed}"
            res = {
                "model_name": m_name,
                "model_path": m_path,
                "status": f"FAILED ({stage_failed})",
                "stage_failed": stage_failed,
                "error_message": error_message,
                "total_layers": "N/A",
                "npu_pct": "N/A",
                "cpu_layers": "N/A",
                "latency": "N/A",
                "link": f"[{stage_failed} error]({c_job_url})" if c_job_url else stage_failed,
            }
            results.append(res)
            _append_to_evidence_log(
                model_name=m_name,
                model_path=m_path,
                status="FAILED",
                compile_job_url=c_job_url,
                profile_job_url=p_job_url,
                stage_failed=stage_failed,
                error_message=error_message,
            )
        else:
            # Success handling: parse downloaded profile
            profile = load_profile(str(save_path), device_name=device_name, model_name=m_name)
            total = profile.total_layer_count
            npu = profile.npu_layer_count
            cpu = profile.cpu_layer_count
            npu_pct_val = f"{profile.npu_fraction * 100:.1f}%" if total > 0 else "0.0%"

            lat_us = profile.total_inference_time_us
            if lat_us is not None:
                lat_str = f"{lat_us / 1000.0:.3f} ms ({lat_us:.0f} µs)"
            else:
                lat_str = "N/A"

            job_link = f"[Profile Job]({p_job_url})" if p_job_url else f"`{save_path.name}`"

            res = {
                "model_name": m_name,
                "model_path": m_path,
                "status": "SUCCESS",
                "stage_failed": None,
                "error_message": None,
                "total_layers": total,
                "npu_layers": npu,
                "npu_pct": npu_pct_val,
                "cpu_layers": cpu,
                "latency": lat_str,
                "link": job_link,
                "profile_path": str(save_path).replace("\\", "/"),
            }
            results.append(res)
            _append_to_evidence_log(
                model_name=m_name,
                model_path=m_path,
                status="SUCCESS",
                compile_job_url=c_job_url,
                profile_job_url=p_job_url,
                metrics={
                    "total_layers": total,
                    "npu_layers": npu,
                    "npu_pct": npu_pct_val,
                    "cpu_layers": cpu,
                    "latency": lat_str,
                    "profile_path": str(save_path).replace("\\", "/"),
                },
            )
            print(f"[batch] SUCCESS: {m_name} -> {npu}/{total} NPU layers ({npu_pct_val}), latency: {lat_str}")

    # Generate BATCH_SUMMARY.md
    summary_md = render_batch_summary_markdown(results, device_name=device_name)
    summary_file = batch_dir / "BATCH_SUMMARY.md"
    summary_file.write_text(summary_md, encoding="utf-8")
    print(f"\n[batch] Batch summary written to: {summary_file}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NPU Fit Checker batch profiling across models.")
    parser.add_argument("--device", default=DEFAULT_PRIMARY_DEVICE, help="Target device name from devices.txt")
    parser.add_argument(
        "--models",
        nargs="+",
        help="List of model paths or model_path:model_name pairs",
    )
    parser.add_argument("--compile-options", default="--target_runtime onnx", help="Options for compilation")
    parser.add_argument("--profile-options", default="", help="Options for profiling")

    args = parser.parse_args()

    # Default candidate list if none specified
    if args.models:
        model_pairs = []
        for item in args.models:
            if ":" in item:
                path, name = item.split(":", 1)
            else:
                path = item
                name = pathlib.Path(item).stem
            model_pairs.append((path, name))
    else:
        model_pairs = [
            ("mobilenet_v2.onnx", "MobileNetV2"),
            ("squeezenet1_0.onnx", "SqueezeNet"),
            ("resnet18.onnx", "ResNet18"),
        ]

    run_batch(
        models=model_pairs,
        device_name=args.device,
        compile_options=args.compile_options,
        profile_options=args.profile_options,
    )


if __name__ == "__main__":
    main()
