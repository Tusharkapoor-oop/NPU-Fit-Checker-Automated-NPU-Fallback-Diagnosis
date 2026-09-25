"""
scripts/phase1_mobilenetv2.py

Phase 1: Run the official MobileNetV2 profile example on AI Hub.

Device policy (matches experiments/devices.txt):
  PRIMARY   : Snapdragon X2 Elite CRD
  SECONDARY : Snapdragon X Elite CRD
  RULE      : Only device names present in experiments/devices.txt may be used.

Steps:
  1. Read permitted device names from experiments/devices.txt.
  2. Export MobileNetV2 (torchvision) to ONNX.
  3. Upload and compile for the primary device.
  4. Submit a profile job.
  5. Download the raw profile JSON and save it untouched.
  6. Print job URLs.
  7. Gate: profile JSON must exist and be non-empty.

Run:
  python scripts/phase1_mobilenetv2.py

Requires:
  pip install torch torchvision onnx qai-hub
"""

import pathlib
import json
import sys

# ---------------------------------------------------------------------------
# Check dependencies
# ---------------------------------------------------------------------------
missing = []
try:
    import torch
    import torchvision
except ImportError:
    missing.append("torch torchvision")

try:
    import onnx
except ImportError:
    missing.append("onnx")

try:
    import qai_hub as hub
except ImportError:
    missing.append("qai-hub")

if missing:
    print(f"[ERROR] Missing packages: {', '.join(missing)}")
    print("Install with: pip install torch torchvision onnx qai-hub")
    sys.exit(1)

import torch
import torchvision
import qai_hub as hub

# ---------------------------------------------------------------------------
# Device policy
# ---------------------------------------------------------------------------
DEVICES_FILE     = pathlib.Path("experiments/devices.txt")
PRIMARY_DEVICE   = "Snapdragon X2 Elite CRD"
SECONDARY_DEVICE = "Snapdragon X Elite CRD"


def _load_permitted_devices() -> list[str]:
    """Load the device names saved by phase0_setup.py. Never use others."""
    if not DEVICES_FILE.exists():
        raise FileNotFoundError(
            f"{DEVICES_FILE} not found. Run scripts/phase0_setup.py first."
        )
    names = [l.strip() for l in DEVICES_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    return names


def _validate_device(name: str, permitted: list[str]) -> str:
    """Raise ValueError if name is not in the permitted list (devices.txt rule)."""
    if name not in permitted:
        raise ValueError(
            f"Device '{name}' is not in experiments/devices.txt.\n"
            f"Permitted devices: {permitted}"
        )
    return name


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR   = pathlib.Path("experiments/baseline_fp32")
ONNX_PATH    = OUTPUT_DIR / "mobilenetv2_fp32.onnx"
PROFILE_PATH = OUTPUT_DIR / "profile.json"
NOTES_PATH   = OUTPUT_DIR / "notes.md"


def export_onnx():
    print("[1/6] Exporting MobileNetV2 FP32 to ONNX …")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
    print(f"  ✓ Saved to {ONNX_PATH} ({size_mb:.1f} MB)")
    return ONNX_PATH


def upload_and_compile(onnx_path: pathlib.Path, device_name: str):
    print(f"\n[2/6] Uploading model to AI Hub …")
    model = hub.upload_model(str(onnx_path))
    print(f"  ✓ Model uploaded: {model}")

    print(f"\n[3/6] Compiling for {device_name} …")
    # Look up available submit methods before calling
    print(f"  hub public API: {[x for x in dir(hub) if not x.startswith('_')]}")

    devices = hub.get_devices()
    device = next((d for d in devices if device_name.lower() in d.name.lower()), None)
    if device is None:
        available = [d.name for d in devices]
        print(f"  [ERROR] Device '{device_name}' not found. Available: {available}")
        sys.exit(1)

    compile_job = hub.submit_compile_job(model=model, device=device)
    print(f"  ✓ Compile job submitted")
    print(f"  → Job URL: {compile_job.url}")

    print("  Waiting for compile job …")
    compile_job.wait()
    print("  ✓ Compile complete")
    return compile_job, device


def profile(compile_job, device):
    print("\n[4/6] Submitting profile job …")
    target_model = compile_job.download_target_model()
    profile_job = hub.submit_profile_job(model=target_model, device=device)
    print(f"  ✓ Profile job submitted")
    print(f"  → Job URL: {profile_job.url}")

    # Save URLs to notes
    urls = {
        "compile_job_url": str(compile_job.url),
        "profile_job_url": str(profile_job.url),
    }

    print("  Waiting for profile job (this may take several minutes) …")
    profile_job.wait()
    print("  ✓ Profile complete")

    # Discover download method at runtime
    dl_methods = [m for m in dir(profile_job) if "download" in m.lower() or "profile" in m.lower()]
    print(f"  Available download/profile methods: {dl_methods}")

    if hasattr(profile_job, "download_profile"):
        raw_profile = profile_job.download_profile()
    elif hasattr(profile_job, "get_profile"):
        raw_profile = profile_job.get_profile()
    else:
        print(f"  [ERROR] No profile download method found. Methods: {dl_methods}")
        sys.exit(1)

    return raw_profile, urls


def save_results(raw_profile: dict, urls: dict):
    print("\n[5/5] Saving results …")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save raw profile JSON — untouched
    with open(PROFILE_PATH, "w", encoding="utf-8") as fh:
        json.dump(raw_profile, fh, indent=2, default=str)
    print(f"  ✓ Profile JSON saved to: {PROFILE_PATH}")

    # Write notes
    notes = f"""# Phase 1 — MobileNetV2 FP32 Baseline

## Job URLs
- Compile job: {urls.get('compile_job_url', 'UNVERIFIED')}
- Profile job: {urls.get('profile_job_url', 'UNVERIFIED')}

## Model
- Architecture: MobileNetV2 (torchvision, ImageNet1K V1 weights)
- Precision: FP32
- ONNX opset: 17
- Input: [1, 3, 224, 224]

## Device
- Qualcomm AI Hub cloud-hosted Snapdragon X Elite device

## Profile JSON top-level keys
{list(raw_profile.keys())}

## Notes
Raw profile JSON saved untouched as profile.json.
See docs/PROFILE_SCHEMA.md for schema description.
"""
    NOTES_PATH.write_text(notes, encoding="utf-8")
    print(f"  ✓ Notes saved to: {NOTES_PATH}")

    # Gate check
    if PROFILE_PATH.exists() and PROFILE_PATH.stat().st_size > 100:
        print("\n✓ PHASE 1 GATE PASSED: profile JSON exists and is non-empty.")
    else:
        print("\n[GATE FAILED] profile.json is missing or empty!")
        sys.exit(1)


def main():
    print("=" * 60)
    print("NPU Fit Checker — Phase 1: MobileNetV2 Baseline Profile")
    print("=" * 60)

    # Enforce device-list rule BEFORE any API calls
    try:
        permitted = _load_permitted_devices()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    primary = _validate_device(PRIMARY_DEVICE, permitted)
    print(f"Primary device  : {primary}")

    # Check secondary (warn only, not a hard error)
    if SECONDARY_DEVICE in permitted:
        print(f"Secondary device: {SECONDARY_DEVICE}")
    else:
        print(f"NOTE: Secondary device '{SECONDARY_DEVICE}' not in devices.txt — skipping secondary profile.")

    onnx_path = export_onnx()
    compile_job, device = upload_and_compile(onnx_path, primary)
    raw_profile, urls = profile(compile_job, device)
    save_results(raw_profile, urls)

    print("\nPhase 1 complete. Proceed to Phase 2.")
    print(f"Profile saved to: {PROFILE_PATH}")
    print("Next step: python scripts/phase2_experiment.py")


if __name__ == "__main__":
    main()
