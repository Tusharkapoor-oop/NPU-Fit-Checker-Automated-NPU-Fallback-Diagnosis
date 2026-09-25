"""
scripts/run_task4_optimized.py

Task 4: Runs the optimized INT8 quantize, compile, and profile pipeline on AI Hub
for MobileNetV2 on Snapdragon X2 Elite CRD.

Steps:
  1. Load permitted devices from experiments/devices.txt (exact matching).
  2. Upload mobilenet_v2.onnx.
  3. Submit quantize job (INT8 weights and activations).
  4. Wait for quantize job, get target model via get_target_model().
  5. Submit compile job with options="--target_runtime onnx" on Snapdragon X2 Elite CRD.
  6. Wait for compile job, get target model via get_target_model().
  7. Submit profile job with options="--onnx_execution_providers=qnn" on Snapdragon X2 Elite CRD.
  8. Wait for profile job, download raw profile JSON via download_profile().
  9. Save profile UNTOUCHED to experiments/optimized_int8/profile.json and experiments/optimized/profile.json.
  10. Update docs/EVIDENCE_LOG.md and write experiments/optimized_int8/notes.md.
"""

import io
import json
import os
import pathlib
import sys
import numpy as np

# Force UTF-8 output to prevent Windows cp1252 emoji crash during .wait()
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    import qai_hub as hub
except ImportError:
    print("[ERROR] qai_hub not installed.")
    sys.exit(1)

DEVICES_FILE = pathlib.Path("experiments/devices.txt")
TARGET_DEVICE_NAME = "Snapdragon X2 Elite CRD"
ONNX_PATH = pathlib.Path("mobilenet_v2.onnx")
EVIDENCE_LOG = pathlib.Path("docs/EVIDENCE_LOG.md")


def load_permitted_devices() -> list[str]:
    if not DEVICES_FILE.exists():
        raise FileNotFoundError(f"{DEVICES_FILE} not found. Run phase0_setup.py first.")
    devices = []
    for line in DEVICES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            devices.append(entry.get("name"))
        except Exception:
            devices.append(line)
    return devices


def main():
    print("=" * 60, flush=True)
    print("Task 4: Optimized INT8 Quantize + Compile + Profile Pipeline", flush=True)
    print(f"Device: {TARGET_DEVICE_NAME}", flush=True)
    print("=" * 60, flush=True)

    # Validate device
    permitted = load_permitted_devices()
    print(f"Permitted devices: {permitted}", flush=True)
    if TARGET_DEVICE_NAME not in permitted:
        print(f"[ERROR] '{TARGET_DEVICE_NAME}' is not in permitted devices list!", flush=True)
        sys.exit(1)

    all_hub_devices = hub.get_devices()
    device_obj = next((d for d in all_hub_devices if d.name == TARGET_DEVICE_NAME), None)
    if device_obj is None:
        print(f"[ERROR] Device '{TARGET_DEVICE_NAME}' not found on AI Hub.", flush=True)
        sys.exit(1)
    print(f"Device found on AI Hub: {device_obj.name}", flush=True)

    if not ONNX_PATH.exists():
        print(f"[ERROR] {ONNX_PATH} not found.", flush=True)
        sys.exit(1)

    # 1. Upload Model
    print(f"\n[1] Uploading {ONNX_PATH} ({ONNX_PATH.stat().st_size:,} bytes) ...", flush=True)
    model = hub.upload_model(str(ONNX_PATH))
    print(f"    Uploaded model: {model}", flush=True)

    # 2. Calibration data & Quantize Job
    print("\n[2] Submitting quantize job (INT8 weights & activations) ...", flush=True)
    calibration_data = {"input": [np.random.randn(1, 3, 224, 224).astype(np.float32) for _ in range(5)]}
    quantize_job = hub.submit_quantize_job(
        model=model,
        calibration_data=calibration_data,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
        name="mobilenet_v2_int8_quantize",
    )
    quantize_url = getattr(quantize_job, "url", "URL_UNAVAILABLE")
    print(f"    Quantize job ID : {quantize_job.job_id}", flush=True)
    print(f"    Quantize job URL: {quantize_url}", flush=True)

    print("    Waiting for quantize job ...", flush=True)
    quantize_job.wait()
    print("    Quantize job complete.", flush=True)

    quantized_model = quantize_job.get_target_model()
    print(f"    Quantized target model: {quantized_model}", flush=True)

    # 3. Compile Job
    print(f"\n[3] Submitting compile job on '{device_obj.name}' with options='--target_runtime onnx' ...", flush=True)
    compile_job = hub.submit_compile_job(
        model=quantized_model,
        device=device_obj,
        options="--target_runtime onnx",
        name="mobilenet_v2_int8_compile",
    )
    compile_url = getattr(compile_job, "url", "URL_UNAVAILABLE")
    print(f"    Compile job ID : {compile_job.job_id}", flush=True)
    print(f"    Compile job URL: {compile_url}", flush=True)

    print("    Waiting for compile job ...", flush=True)
    compile_job.wait()
    print("    Compile job complete.", flush=True)

    compiled_model = compile_job.get_target_model()
    print(f"    Compiled target model: {compiled_model}", flush=True)

    # 4. Profile Job
    print(f"\n[4] Submitting profile job on '{device_obj.name}' with options='--onnx_execution_providers=qnn' ...", flush=True)
    profile_job = hub.submit_profile_job(
        model=compiled_model,
        device=device_obj,
        options="--onnx_execution_providers=qnn",
        name="mobilenet_v2_int8_profile",
    )
    profile_url = getattr(profile_job, "url", "URL_UNAVAILABLE")
    print(f"    Profile job ID : {profile_job.job_id}", flush=True)
    print(f"    Profile job URL: {profile_url}", flush=True)

    print("    Waiting for profile job ...", flush=True)
    profile_job.wait()
    print("    Profile job complete.", flush=True)

    # 5. Download Profile JSON
    print("\n[5] Downloading profile JSON ...", flush=True)
    profile_data = profile_job.download_profile()

    # Save to both target directories
    for target_dir in [pathlib.Path("experiments/optimized_int8"), pathlib.Path("experiments/optimized")]:
        target_dir.mkdir(parents=True, exist_ok=True)
        out_file = target_dir / "profile.json"
        with out_file.open("w", encoding="utf-8") as fh:
            json.dump(profile_data, fh, indent=2, default=str)
        print(f"    Saved: {out_file} ({out_file.stat().st_size:,} bytes)", flush=True)

    # Save notes
    notes_content = f"""# Optimized (INT8) Profile Notes

- Device: {device_obj.name}
- Model: MobileNetV2 (INT8 Quantized QDQ)
- Quantize Job URL: {quantize_url}
- Compile Job URL: {compile_url}
- Profile Job URL: {profile_url}
- Compile Options: --target_runtime onnx
- Profile Options: --onnx_execution_providers=qnn
- Raw Profile: experiments/optimized_int8/profile.json
"""
    for target_dir in [pathlib.Path("experiments/optimized_int8"), pathlib.Path("experiments/optimized")]:
        (target_dir / "notes.md").write_text(notes_content, encoding="utf-8")

    # Update evidence log
    evidence_entry = f"""
## Phase 2 — MobileNetV2 Optimized (INT8) on {device_obj.name}

- Quantize job URL: {quantize_url}
- Compile job URL : {compile_url}
- Profile job URL : {profile_url}
- Raw profile     : experiments/optimized_int8/profile.json
"""
    if EVIDENCE_LOG.exists():
        existing = EVIDENCE_LOG.read_text(encoding="utf-8")
        if profile_url not in existing:
            EVIDENCE_LOG.write_text(existing + evidence_entry, encoding="utf-8")
            print("    Updated EVIDENCE_LOG.md", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("TASK 4 OPTIMIZED RUN COMPLETE", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
