"""
scripts/phase1_mobilenet.py

Task 1: Prove the path with a real profile.

Steps:
  1. Export MobileNetV2 from torchvision to ONNX.
  2. Submit compile job on Snapdragon X2 Elite CRD
     with options="--target_runtime onnx" (per official docs).
  3. Wait, call get_target_model(), submit profile job.
  4. Wait, call download_profile(), save raw JSON to
     experiments/profile_mobilenet_x2.json UNTOUCHED.
  5. Record both job URLs in docs/EVIDENCE_LOG.md.
  6. GATE: profile JSON must exist and be > 100 bytes.
  7. Print top-level keys and one sample layer for schema review.

All methods confirmed via experiments/api_methods.txt before use.
Run from repo root:
  python scripts/phase1_mobilenet.py
"""

import json
import pathlib
import sys
import io
import os

# Force UTF-8 output so hub SDK emoji (e.g. U+23F3 hourglass in .wait())
# does not crash on Windows cp1252 terminals.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Method verification guard
# ---------------------------------------------------------------------------
API_METHODS_FILE = pathlib.Path("experiments/api_methods.txt")

def _verify_method(class_name: str, method_name: str) -> None:
    """Abort if method_name is not listed in api_methods.txt for class_name."""
    if not API_METHODS_FILE.exists():
        print(f"[ERROR] {API_METHODS_FILE} not found. Run phase0_setup.py first.")
        sys.exit(1)
    text = API_METHODS_FILE.read_text(encoding="utf-8")
    section_marker = f"# hub.{class_name}"
    if section_marker not in text:
        print(f"[ERROR] {class_name} section missing from {API_METHODS_FILE}")
        sys.exit(1)
    # Extract the section lines
    lines = text.split(section_marker, 1)[1].split("\n# hub.", 1)[0].split()
    if method_name not in lines:
        print(
            f"[ERROR] Method '{method_name}' not found under {section_marker} "
            f"in {API_METHODS_FILE}.\n"
            "Do NOT use this method — it is unverified."
        )
        sys.exit(1)
    print(f"  [verify] hub.{class_name}.{method_name}() -> CONFIRMED in api_methods.txt")


PRIMARY_DEVICE = "Snapdragon X2 Elite CRD"
FALLBACK_DEVICE = "Snapdragon X Elite CRD"
PROFILE_OUT    = pathlib.Path("experiments/profile_mobilenet_x2.json")
EVIDENCE_LOG   = pathlib.Path("docs/EVIDENCE_LOG.md")


def export_onnx() -> pathlib.Path:
    """Export MobileNetV2 to ONNX if not already present."""
    onnx_path = pathlib.Path("mobilenet_v2.onnx")
    if onnx_path.exists():
        print(f"[phase1] ONNX already exists: {onnx_path}  ({onnx_path.stat().st_size:,} bytes)")
        return onnx_path

    print("[phase1] Exporting MobileNetV2 to ONNX ...")
    try:
        import torch
        import torchvision.models as models
    except ImportError as e:
        print(f"[ERROR] {e}. Run: pip install torch torchvision")
        sys.exit(1)

    model = models.mobilenet_v2(weights=None)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    print(f"[phase1] Exported: {onnx_path}  ({onnx_path.stat().st_size:,} bytes)")
    return onnx_path


def update_evidence_log(compile_url: str, profile_url: str, device: str) -> None:
    EVIDENCE_LOG.parent.mkdir(exist_ok=True)
    header = "# Evidence Log\n\n"
    entry = (
        f"## Phase 1 — MobileNetV2 profile on {device}\n\n"
        f"- Compile job URL : {compile_url}\n"
        f"- Profile job URL : {profile_url}\n"
        f"- Raw profile     : experiments/profile_mobilenet_x2.json\n\n"
    )
    existing = EVIDENCE_LOG.read_text(encoding="utf-8") if EVIDENCE_LOG.exists() else header
    if compile_url not in existing:
        EVIDENCE_LOG.write_text(existing + entry, encoding="utf-8")
        print(f"[phase1] Evidence log updated: {EVIDENCE_LOG}")
    else:
        print(f"[phase1] Evidence log already contains this entry.")


def main() -> None:
    print("=" * 60)
    print("Phase 1 — MobileNetV2 proof-of-path profile")
    print("=" * 60)

    # Verify required methods exist before any hub calls
    print("\n[0] Verifying API methods against api_methods.txt ...")
    _verify_method("CompileJob", "wait")
    _verify_method("CompileJob", "get_target_model")
    _verify_method("CompileJob", "url")
    _verify_method("ProfileJob", "wait")
    _verify_method("ProfileJob", "download_profile")
    _verify_method("ProfileJob", "url")
    print("  All methods verified.")

    # Import hub
    try:
        import qai_hub as hub
    except ImportError:
        print("[ERROR] qai_hub not installed. Run: pip install qai-hub")
        sys.exit(1)

    # Load permitted devices
    devices_file = pathlib.Path("experiments/devices.txt")
    if not devices_file.exists():
        print("[ERROR] experiments/devices.txt not found. Run phase0_setup.py first.")
        sys.exit(1)
    permitted = [json.loads(l)["name"] for l in devices_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\n[1] Permitted devices: {permitted}")
    if PRIMARY_DEVICE not in permitted:
        print(f"[ERROR] {PRIMARY_DEVICE!r} not in devices.txt. Abort.")
        sys.exit(1)

    # Export ONNX
    print("\n[2] Export / locate model ...")
    onnx_path = export_onnx()

    # Upload model
    print("\n[3] Upload model to AI Hub ...")
    model = hub.upload_model(str(onnx_path))
    print(f"  Uploaded model: {model}")

    # Submit compile job — primary device first
    device_used = PRIMARY_DEVICE
    print(f"\n[4] Submit compile job on '{device_used}' ...")
    all_hub_devices = hub.get_devices()
    device_obj = next((d for d in all_hub_devices if d.name == device_used), None)
    if device_obj is None:
        print(f"  [ERROR] Device {device_used!r} not found on hub. Aborting.")
        sys.exit(1)

    compile_job = hub.submit_compile_job(
        model=model,
        device=device_obj,
        options="--target_runtime onnx",
    )
    compile_url = getattr(compile_job, "url", "URL_UNAVAILABLE")
    print(f"  Compile job URL: {compile_url}")

    # Wait for compile
    print("\n[5] Waiting for compile job ...")
    compile_job.wait()
    print("  Compile complete.")

    # get_target_model (confirmed in api_methods.txt)
    print("\n[6] get_target_model() ...")
    target_model = compile_job.get_target_model()
    print(f"  target_model: {target_model}")

    # Submit profile job
    print(f"\n[7] Submit profile job on '{device_used}' ...")
    profile_job = hub.submit_profile_job(
        model=target_model,
        device=device_obj,
    )
    profile_url = getattr(profile_job, "url", "URL_UNAVAILABLE")
    print(f"  Profile job URL: {profile_url}")

    # Wait for profile
    print("\n[8] Waiting for profile job ...")
    profile_job.wait()
    print("  Profile complete.")

    # Download profile (confirmed in api_methods.txt)
    print("\n[9] download_profile() ...")
    profile_data = profile_job.download_profile()

    # Save UNTOUCHED
    PROFILE_OUT.parent.mkdir(exist_ok=True)
    with PROFILE_OUT.open("w", encoding="utf-8") as fh:
        json.dump(profile_data, fh, indent=2, default=str)
    size = PROFILE_OUT.stat().st_size
    print(f"  Saved: {PROFILE_OUT}  ({size:,} bytes)")

    # Gate
    if size < 100:
        print(f"[GATE FAILED] Profile file is only {size} bytes — likely empty.")
        sys.exit(1)
    print(f"  GATE PASSED: profile file exists and is {size:,} bytes.")

    # Update evidence log
    update_evidence_log(compile_url, profile_url, device_used)

    # Print schema for user review
    print("\n" + "=" * 60)
    print("SCHEMA PREVIEW — top-level keys:")
    for k, v in profile_data.items():
        vtype = type(v).__name__
        if isinstance(v, list):
            sample = f"list[{len(v)} items]"
            if v and isinstance(v[0], dict):
                sample += f"  first-item keys: {list(v[0].keys())}"
        elif isinstance(v, dict):
            sample = f"dict  keys: {list(v.keys())}"
        elif isinstance(v, str) and len(v) > 100:
            sample = repr(v[:100]) + "..."
        else:
            sample = repr(v)
        print(f"  {k!r} ({vtype}): {sample}")

    # Print one sample layer if available
    layer_data = None
    for key in profile_data:
        val = profile_data[key]
        if isinstance(val, list) and val and isinstance(val[0], dict):
            print(f"\nSample entry from {key!r}[0]:")
            import pprint
            pprint.pprint(val[0], indent=4, width=100)
            layer_data = val[0]
            break

    print("=" * 60)
    print("Phase 1 complete.")
    print(f"STOP HERE — review the schema above and confirm keys with the user")
    print(f"before proceeding to Task 2 (PROFILE_SCHEMA.md) and Task 3 (parser).")
    print("=" * 60)


if __name__ == "__main__":
    main()
