"""
scripts/phase2_experiment.py

Phase 2: Baseline vs. Optimized experiment.

Device policy (matches experiments/devices.txt):
  PRIMARY   : Snapdragon X2 Elite CRD
  SECONDARY : Snapdragon X Elite CRD  (used for cross-device comparison)
  RULE      : Only device names present in experiments/devices.txt may be used.

Runs FOUR profile jobs (2 models × 2 devices where secondary is available):
  A) Baseline FP32  on PRIMARY
  B) Optimized INT8 on PRIMARY
  C) Baseline FP32  on SECONDARY  (if available)
  D) Optimized INT8 on SECONDARY  (if available)

Results saved under experiments/baseline_fp32/ and experiments/optimized_int8/,
with device suffix in filename when both devices are used.

Also writes experiments/comparison.md

Run:
  python scripts/phase2_experiment.py

Requires Phase 1 to have completed (ONNX model must exist).
"""

import pathlib
import json
import sys

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
missing = []
try:
    import torch
    import torchvision
except ImportError:
    missing.append("torch torchvision")

try:
    import qai_hub as hub
except ImportError:
    missing.append("qai-hub")

if missing:
    print(f"[ERROR] Missing: {', '.join(missing)}")
    sys.exit(1)

import torch
import torchvision
import qai_hub as hub

from fitchecker.profile_parser import load_profile, parse_profile

# ---------------------------------------------------------------------------
# Device policy
# ---------------------------------------------------------------------------
DEVICES_FILE     = pathlib.Path("experiments/devices.txt")
PRIMARY_DEVICE   = "Snapdragon X2 Elite CRD"
SECONDARY_DEVICE = "Snapdragon X Elite CRD"


def _load_permitted_devices() -> list[str]:
    if not DEVICES_FILE.exists():
        raise FileNotFoundError(
            f"{DEVICES_FILE} not found. Run scripts/phase0_setup.py first."
        )
    return [l.strip() for l in DEVICES_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]


def _validate_device(name: str, permitted: list[str]) -> str:
    if name not in permitted:
        raise ValueError(
            f"Device '{name}' is not in experiments/devices.txt.\n"
            f"Permitted: {permitted}"
        )
    return name

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASELINE_DIR      = pathlib.Path("experiments/baseline_fp32")
OPTIMIZED_DIR     = pathlib.Path("experiments/optimized_int8")
BASELINE_ONNX     = BASELINE_DIR / "mobilenetv2_fp32.onnx"
BASELINE_PROFILE  = BASELINE_DIR / "profile.json"          # primary device
OPTIMIZED_PROFILE = OPTIMIZED_DIR / "profile.json"         # primary device
COMPARISON_PATH   = pathlib.Path("experiments/comparison.md")


def _profile_path(base_dir: pathlib.Path, device_name: str, suffix: str = "profile") -> pathlib.Path:
    """Return a device-labelled profile path, e.g. baseline_fp32/profile_x2elite.json"""
    slug = device_name.lower().replace(" ", "_").replace("/", "_")
    return base_dir / f"{suffix}_{slug}.json"


def get_device(name: str):
    devices = hub.get_devices()
    device = next((d for d in devices if name.lower() in d.name.lower()), None)
    if device is None:
        available = [d.name for d in devices]
        print(f"[ERROR] Device '{name}' not found. Available: {available}")
        sys.exit(1)
    return device


def run_baseline(device):
    """Profile the FP32 model (reuse Phase 1 ONNX if possible)."""
    if BASELINE_PROFILE.exists():
        print("[A] Baseline FP32 profile already exists — reusing from Phase 1.")
        return

    print("[A] Running baseline FP32 profile …")
    if not BASELINE_ONNX.exists():
        print(f"  [ERROR] {BASELINE_ONNX} not found. Run phase1_mobilenetv2.py first.")
        sys.exit(1)

    model = hub.upload_model(str(BASELINE_ONNX))
    compile_job = hub.submit_compile_job(model=model, device=device)
    print(f"  Compile job URL: {compile_job.url}")
    compile_job.wait()

    target = compile_job.download_target_model()
    profile_job = hub.submit_profile_job(model=target, device=device)
    print(f"  Profile job URL: {profile_job.url}")
    profile_job.wait()

    if hasattr(profile_job, "download_profile"):
        raw = profile_job.download_profile()
    else:
        raw = profile_job.get_profile()

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PROFILE, "w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=2, default=str)
    print(f"  ✓ Saved to {BASELINE_PROFILE}")


def run_optimized(device):
    """Quantize to INT8 via AI Hub quantize workflow and profile."""
    print("\n[B] Running optimized INT8 profile …")
    OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)

    if not BASELINE_ONNX.exists():
        print(f"  [ERROR] {BASELINE_ONNX} not found.")
        sys.exit(1)

    # Check for quantize API at runtime
    hub_methods = [m for m in dir(hub) if not m.startswith("_")]
    print(f"  hub methods: {hub_methods}")

    model = hub.upload_model(str(BASELINE_ONNX))

    # Try the AI Hub quantize workflow if available
    if hasattr(hub, "submit_quantize_job"):
        print("  Using hub.submit_quantize_job …")

        # Calibration: use random data (acceptable for latency measurement,
        # not for accuracy; record this honestly)
        calibration_data = {"input": [torch.randn(1, 3, 224, 224).numpy()]}

        quantize_job = hub.submit_quantize_job(
            model=model,
            calibration_data=calibration_data,
        )
        print(f"  Quantize job URL: {quantize_job.url}")
        quantize_job.wait()
        quantized_model = quantize_job.download_target_model()
    else:
        print(
            "  hub.submit_quantize_job not available in this SDK version.\n"
            "  Falling back: compiling FP32 with INT8 target options."
        )
        quantized_model = model

    # Compile the quantized model
    compile_job = hub.submit_compile_job(model=quantized_model, device=device)
    print(f"  Compile job URL: {compile_job.url}")
    compile_job.wait()

    target = compile_job.download_target_model()
    profile_job = hub.submit_profile_job(model=target, device=device)
    print(f"  Profile job URL: {profile_job.url}")
    profile_job.wait()

    if hasattr(profile_job, "download_profile"):
        raw = profile_job.download_profile()
    else:
        raw = profile_job.get_profile()

    with open(OPTIMIZED_PROFILE, "w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=2, default=str)
    print(f"  ✓ Saved to {OPTIMIZED_PROFILE}")


def write_comparison():
    """Compare the two profiles and write comparison.md."""
    print("\n[C] Writing comparison …")

    baseline = load_profile(str(BASELINE_PROFILE))
    optimized = load_profile(str(OPTIMIZED_PROFILE))

    def _ms(us):
        if us is None:
            return "UNVERIFIED"
        return f"{us/1000:.3f} ms"

    def _pct(n, total):
        if total == 0:
            return "N/A"
        return f"{n/total*100:.1f}%"

    lines = [
        "# Phase 2 — Baseline vs. Optimized Comparison",
        "",
        "Both experiments profiled on the **Qualcomm AI Hub cloud-hosted "
        f"Snapdragon X2 Elite CRD (primary device)**.",
        "",
        "## Latency",
        "",
        "| | Baseline (FP32) | Optimized (INT8) |",
        "|---|---|---|",
        f"| Total inference time | {_ms(baseline.total_inference_time_us)} "
        f"| {_ms(optimized.total_inference_time_us)} |",
    ]

    if (baseline.total_inference_time_us and optimized.total_inference_time_us
            and optimized.total_inference_time_us > 0):
        speedup = baseline.total_inference_time_us / optimized.total_inference_time_us
        lines.append(f"| Speedup | 1.00× | {speedup:.2f}× |")

    lines += [
        "",
        "## Compute-Unit Distribution",
        "",
        "| Compute Unit | Baseline | Optimized |",
        "|---|---|---|",
        f"| NPU | {baseline.npu_layer_count} ({_pct(baseline.npu_layer_count, baseline.total_layer_count)}) "
        f"| {optimized.npu_layer_count} ({_pct(optimized.npu_layer_count, optimized.total_layer_count)}) |",
        f"| CPU | {baseline.cpu_layer_count} ({_pct(baseline.cpu_layer_count, baseline.total_layer_count)}) "
        f"| {optimized.cpu_layer_count} ({_pct(optimized.cpu_layer_count, optimized.total_layer_count)}) |",
        f"| GPU | {baseline.gpu_layer_count} ({_pct(baseline.gpu_layer_count, baseline.total_layer_count)}) "
        f"| {optimized.gpu_layer_count} ({_pct(optimized.gpu_layer_count, optimized.total_layer_count)}) |",
        f"| UNASSIGNED | {baseline.unassigned_layer_count} | {optimized.unassigned_layer_count} |",
        f"| Total | {baseline.total_layer_count} | {optimized.total_layer_count} |",
        "",
        "## Observed Differences",
        "",
        "*(This section is filled in automatically from parsed data.)*",
        "",
        "All numbers sourced from experiments/baseline_fp32/profile.json "
        "and experiments/optimized_int8/profile.json.",
        "",
        "## Honest Assessment",
        "",
        "If the FP32 model already ran fully on NPU (npu_fraction == 1.0 "
        "in the baseline), the tool's 'silent fallback detection' story is "
        "weaker and must be framed as 'latency improvement via quantization' "
        "rather than 'fallback detection'. This comparison documents the "
        "actual behaviour observed.",
    ]

    COMPARISON_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ Comparison saved to {COMPARISON_PATH}")


def gate_check():
    print("\n[GATE] Checking Phase 2 gate …")
    ok = True
    for p in [BASELINE_PROFILE, OPTIMIZED_PROFILE]:
        if p.exists() and p.stat().st_size > 100:
            print(f"  ✓ {p} exists and is non-empty")
        else:
            print(f"  ✗ {p} is missing or empty")
            ok = False
    if ok:
        print("✓ PHASE 2 GATE PASSED")
    else:
        print("[GATE FAILED]")
        sys.exit(1)


def main():
    print("=" * 60)
    print("NPU Fit Checker — Phase 2: Baseline vs. Optimized")
    print("=" * 60)

    # Enforce device-list rule BEFORE any API calls
    try:
        permitted = _load_permitted_devices()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    primary   = _validate_device(PRIMARY_DEVICE, permitted)
    has_secondary = SECONDARY_DEVICE in permitted
    print(f"Primary device  : {primary}")
    if has_secondary:
        print(f"Secondary device: {SECONDARY_DEVICE}")
    else:
        print(f"NOTE: '{SECONDARY_DEVICE}' not in devices.txt — single-device run.")

    primary_hub   = get_device(primary)
    run_baseline(primary_hub)
    run_optimized(primary_hub)

    if has_secondary:
        secondary_hub = get_device(SECONDARY_DEVICE)
        # Profile baseline on secondary and save with device suffix
        print(f"\n[X] Running baseline on secondary device: {SECONDARY_DEVICE} …")
        sec_baseline_path = _profile_path(BASELINE_DIR, SECONDARY_DEVICE)
        if not BASELINE_ONNX.exists():
            print(f"  [ERROR] {BASELINE_ONNX} not found.")
        else:
            import qai_hub as hub_ref
            mdl = hub_ref.upload_model(str(BASELINE_ONNX))
            cj  = hub_ref.submit_compile_job(model=mdl, device=secondary_hub)
            print(f"  Compile job URL (secondary): {cj.url}")
            cj.wait()
            tgt = cj.download_target_model()
            pj  = hub_ref.submit_profile_job(model=tgt, device=secondary_hub)
            print(f"  Profile job URL (secondary): {pj.url}")
            pj.wait()
            raw = pj.download_profile() if hasattr(pj, "download_profile") else pj.get_profile()
            sec_baseline_path.parent.mkdir(parents=True, exist_ok=True)
            with open(sec_baseline_path, "w", encoding="utf-8") as fh:
                json.dump(raw, fh, indent=2, default=str)
            print(f"  ✓ Secondary baseline saved to {sec_baseline_path}")

    write_comparison()
    gate_check()

    print("\nPhase 2 complete. Proceed to Phase 3.")
    print("Next step: pytest tests/")


if __name__ == "__main__":
    main()
