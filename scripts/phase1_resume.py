"""
Resumes Phase 1 from the already-submitted compile job jgzl4r6x5.
Waits for compile, gets target model, submits profile, downloads result.
Run from repo root with UTF-8 encoding set:
  set PYTHONIOENCODING=utf-8 && .venv\Scripts\python.exe scripts\phase1_resume.py
"""

import json
import pathlib
import sys
import pprint
import os

# Force UTF-8 stdout so the hub SDK's emoji progress bars don't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

COMPILE_JOB_ID = "jgzl4r6x5"          # submitted in the previous run
PRIMARY_DEVICE = "Snapdragon X2 Elite CRD"
PROFILE_OUT    = pathlib.Path("experiments/profile_mobilenet_x2.json")
EVIDENCE_LOG   = pathlib.Path("docs/EVIDENCE_LOG.md")

try:
    import qai_hub as hub
except ImportError:
    print("[ERROR] qai_hub not installed.")
    sys.exit(1)


def main():
    print("=" * 60)
    print("Phase 1 Resume — compile job already submitted")
    print(f"Compile job ID: {COMPILE_JOB_ID}")
    print("=" * 60)

    # Retrieve the compile job object
    print(f"\n[1] Retrieving compile job {COMPILE_JOB_ID} ...")
    compile_job = hub.get_job(COMPILE_JOB_ID)
    compile_url = getattr(compile_job, "url", "URL_UNAVAILABLE")
    print(f"  URL: {compile_url}")
    status = compile_job.get_status()
    print(f"  Status: {status}")

    # Wait for it to finish (with UTF-8 stdout already set)
    print("\n[2] Waiting for compile job to finish ...")
    compile_job.wait()
    print("  Compile done.")

    # get_target_model — confirmed in api_methods.txt
    print("\n[3] get_target_model() ...")
    target_model = compile_job.get_target_model()
    print(f"  target_model: {target_model}")

    # Find device object
    print(f"\n[4] Locating device '{PRIMARY_DEVICE}' ...")
    all_devices = hub.get_devices()
    device_obj = next((d for d in all_devices if d.name == PRIMARY_DEVICE), None)
    if device_obj is None:
        print(f"  [ERROR] Device not found: {PRIMARY_DEVICE}")
        sys.exit(1)
    print(f"  Found: {device_obj.name}")

    # Submit profile job
    print("\n[5] Submit profile job ...")
    profile_job = hub.submit_profile_job(
        model=target_model,
        device=device_obj,
    )
    profile_url = getattr(profile_job, "url", "URL_UNAVAILABLE")
    print(f"  Profile job URL: {profile_url}")

    # Wait for profile
    print("\n[6] Waiting for profile job ...")
    profile_job.wait()
    print("  Profile done.")

    # Download profile — confirmed in api_methods.txt
    print("\n[7] download_profile() ...")
    profile_data = profile_job.download_profile()

    # Save UNTOUCHED
    PROFILE_OUT.parent.mkdir(exist_ok=True)
    with PROFILE_OUT.open("w", encoding="utf-8") as fh:
        json.dump(profile_data, fh, indent=2, default=str)
    size = PROFILE_OUT.stat().st_size
    print(f"  Saved: {PROFILE_OUT}  ({size:,} bytes)")

    if size < 100:
        print(f"[GATE FAILED] Profile file is only {size} bytes.")
        sys.exit(1)
    print(f"  GATE PASSED: {size:,} bytes")

    # Update evidence log
    EVIDENCE_LOG.parent.mkdir(exist_ok=True)
    entry = (
        f"\n## Phase 1 — MobileNetV2 profile on {PRIMARY_DEVICE}\n\n"
        f"- Compile job URL : {compile_url}\n"
        f"- Profile job URL : {profile_url}\n"
        f"- Raw profile     : {PROFILE_OUT}\n"
    )
    existing = EVIDENCE_LOG.read_text(encoding="utf-8") if EVIDENCE_LOG.exists() else "# Evidence Log\n"
    if COMPILE_JOB_ID not in existing:
        EVIDENCE_LOG.write_text(existing + entry, encoding="utf-8")
    print(f"  Evidence log: {EVIDENCE_LOG}")

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

    print("\nSample entry from first list field (likely layers/nodes):")
    for key in profile_data:
        val = profile_data[key]
        if isinstance(val, list) and val and isinstance(val[0], dict):
            print(f"  From key {key!r}[0]:")
            pprint.pprint(val[0], indent=4, width=120)
            break

    print("=" * 60)
    print("Phase 1 complete. STOP — review schema above before Tasks 2-6.")
    print("=" * 60)


if __name__ == "__main__":
    main()

