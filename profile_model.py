"""Profile MobileNetV2 on AI Hub and save raw profile JSON - using X Elite CRD."""
from __future__ import annotations

import json
import time
import pathlib
from typing import Optional

from fitchecker.hub_client import submit_compile_job, submit_profile_job, _require_hub


def poll_job_job(job, timeout_sec=900, poll_interval=15):
    """Poll a job until it's done, without using the buggy .wait() method."""
    _require_hub()
    start = time.time()
    while time.time() - start < timeout_sec:
        status = job.get_status()
        print(f"  Job status: code={status.code} ({int(time.time()-start)}s)")
        if status.code in ("completed", "failed", "error"):
            return status.code
        time.sleep(poll_interval)
    raise TimeoutError(f"Job did not complete within {timeout_sec}s")


def run_profile():
    """Compile → profile → download, saving raw profile JSON."""
    # Compile the model
    print("Compiling mobilenet_v2.onnx for Snapdragon X Elite CRD...")
    compile_job = submit_compile_job('mobilenet_v2.onnx', 'Snapdragon X Elite CRD')
    print(f"Compile job submitted: {compile_job.url}")

    print("Waiting for compile job to finish...")
    poll_job_job(compile_job, timeout_sec=900)
    print("Compile job finished!")

    # Use get_target_model() instead of download_target_model()
    print("Getting target model from compile job...")
    target_model = compile_job.get_target_model()
    print("Target model obtained")

    # Submit profile job
    print("Submitting profile job...")
    profile_job = submit_profile_job(target_model, 'Snapdragon X Elite CRD')
    print(f"Profile job submitted: {profile_job.url}")

    print("Waiting for profile job to finish...")
    poll_job_job(profile_job, timeout_sec=900)

    # Download profile
    print("Downloading profile...")
    profile = profile_job.download_profile()
    print("Profile downloaded!")

    # Save raw profile
    save_path = pathlib.Path('experiments/profile_mobilenet_x2.json')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as fh:
        json.dump(profile, fh, indent=2, default=str)
    print(f"Profile saved to: {save_path}")

    return profile


if __name__ == "__main__":
    profile = run_profile()
    print(f"\nProfile keys: {list(profile.keys())}")