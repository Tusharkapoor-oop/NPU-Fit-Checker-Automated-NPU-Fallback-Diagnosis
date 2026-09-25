"""Profile MobileNetV2 on AI Hub with INT8+QNN optimization and save raw profile JSON."""
from __future__ import annotations

import json
import time
import pathlib
from typing import Optional

from fitchecker.hub_client import submit_compile_job, submit_profile_job, _require_hub


def poll_job_job(job, timeout_sec=900, poll_interval=15):
    """Poll a job until it's done."""
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
    """Compile → profile with QNN options → download, saving raw profile JSON."""
    # Compile the model with --target_runtime onnx
    print("Compiling mobilenet_v2.onnx for Snapdragon X Elite CRD (optimized)...")
    compile_job = submit_compile_job('mobilenet_v2.onnx', 'Snapdragon X Elite CRD')
    print(f"Compile job submitted: {compile_job.url}")

    print("Waiting for compile job to finish...")
    poll_job_job(compile_job, timeout_sec=900)
    print("Compile job finished!")

    # Get target model
    print("Getting target model from compile job...")
    target_model = compile_job.get_target_model()
    print("Target model obtained")

    # Submit profile job with QNN execution provider option
    # Per AI Hub docs: options="--onnx_execution_providers=qnn"
    print("Submitting profile job with --onnx_execution_providers=qnn...")
    profile_job = submit_profile_job(target_model, 'Snapdragon X Elite CRD', options="--onnx_execution_providers=qnn")
    print(f"Profile job submitted: {profile_job.url}")

    print("Waiting for profile job to finish...")
    poll_job_job(profile_job, timeout_sec=900)

    # Download profile
    print("Downloading profile...")
    profile = profile_job.download_profile()
    print("Profile downloaded!")

    # Save raw profile
    save_path = pathlib.Path('experiments/optimized_int8/profile.json')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as fh:
        json.dump(profile, fh, indent=2, default=str)
    print(f"Profile saved to: {save_path}")

    return profile


if __name__ == "__main__":
    profile = run_profile()
    print(f"\nProfile keys: {list(profile.keys())}")
    # Quick parse to show results
    from fitchecker.profile_parser import load_profile as lp
    result = lp('experiments/optimized_int8/profile.json')
    print(f"\nParsed: {result.total_layer_count} layers, {result.npu_layer_count} NPU, {result.cpu_layer_count} CPU")
    print(f"Fallback layers: {len(result.fallback_layers)}")
    print(f"Total time ms: {result.total_inference_time_ms}")