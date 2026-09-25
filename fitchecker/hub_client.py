"""
fitchecker/hub_client.py

Wraps Qualcomm AI Hub SDK calls.

Verified against qai_hub v0.55.0 on 2025-09-21 by running
dir() on each class.  See experiments/api_methods.txt.

API facts used here (all confirmed via dir()):
  ProfileJob : .wait(), .url, .download_profile()
  CompileJob  : .wait(), .url, .get_target_model()
  hub.submit_compile_job(model, device, options=...)
  hub.submit_profile_job(model, device, options=...)
  hub.upload_model(path)

Device filter:
  device.attributes must contain 'os:windows' AND 'format:compute'.
  device.os is a numeric string (e.g. '11'), NOT 'Windows'.

Device name matching: EXACT only.  No substring matching allowed.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Device policy
# ---------------------------------------------------------------------------
DEFAULT_PRIMARY_DEVICE   = "Snapdragon X2 Elite CRD"
DEFAULT_SECONDARY_DEVICE = "Snapdragon X Elite CRD"
_DEVICES_FILE            = pathlib.Path("experiments/devices.txt")

_ATTR_OS      = "os:windows"
_ATTR_COMPUTE = "format:compute"

# ---------------------------------------------------------------------------
# Lazy import (offline --from-profile mode works without qai_hub)
# ---------------------------------------------------------------------------
try:
    import qai_hub as hub
    QAI_HUB_AVAILABLE = True
except ImportError:
    QAI_HUB_AVAILABLE = False


def _require_hub() -> None:
    if not QAI_HUB_AVAILABLE:
        raise RuntimeError("qai_hub not installed. Run: pip install qai-hub")


# ---------------------------------------------------------------------------
# devices.txt helpers
# ---------------------------------------------------------------------------

def load_permitted_devices() -> list[str]:
    """
    Return the list of allowed device names from experiments/devices.txt.
    Raises FileNotFoundError if Phase 0 has not been run.
    """
    if not _DEVICES_FILE.exists():
        raise FileNotFoundError(
            f"{_DEVICES_FILE} not found. Run scripts/phase0_setup.py first."
        )
    names: list[str] = []
    for line in _DEVICES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            names.append(entry["name"])
        except Exception:
            # Warn about malformed JSON lines and fall back to plain-name format
            print(f"[hub_client] WARNING: could not parse JSON line, treating as plain name: {line!r}")
            names.append(line)
    return names


def validate_device_name(name: str) -> str:
    """
    Raise ValueError if name is not an exact match to a permitted device.
    Returns name unchanged if valid.
    """
    permitted = load_permitted_devices()
    if name not in permitted:
        raise ValueError(
            f"Device {name!r} is not in experiments/devices.txt.\n"
            f"Permitted: {permitted}\n"
            "Never use a device name not in that file."
        )
    return name


# ---------------------------------------------------------------------------
# Device lookup — EXACT name matching only
# ---------------------------------------------------------------------------

def find_device(device_name: str):
    """
    Return the hub.Device whose name exactly matches device_name.
    Raises ValueError if not found or not in devices.txt.
    """
    _require_hub()
    validate_device_name(device_name)   # enforces devices.txt rule
    devices = hub.get_devices()
    exact = [d for d in devices if d.name == device_name]
    if not exact:
        available = [d.name for d in devices]
        raise ValueError(
            f"Device {device_name!r} not found on AI Hub.\n"
            f"Available: {available}"
        )
    return exact[0]


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

def submit_compile_job(model_path: str, device_name: str,
                       compile_options: str = "--target_runtime onnx"):
    """
    Upload model_path (must be .onnx) and submit a compile job.
    compile_options defaults to '--target_runtime onnx' per the official
    Compute example in the AI Hub documentation.

    Returns the compile job object (hub.CompileJob).
    Prints the job URL (confirmed: CompileJob.url exists).
    """
    _require_hub()
    p = pathlib.Path(model_path)
    if p.suffix.lower() != ".onnx":
        raise ValueError(
            f"Only .onnx models are supported. Got: {p.suffix}\n"
            "Export your model to ONNX first (torch.onnx.export or torch.export)."
        )
    device = find_device(device_name)
    model  = hub.upload_model(str(p))

    job = hub.submit_compile_job(
        model=model,
        device=device,
        options=compile_options,
    )
    # .url confirmed via dir(hub.CompileJob)
    job_url = getattr(job, "url", "URL_UNAVAILABLE")
    print(f"[hub_client] Compile job: {job_url}")
    return job


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

def submit_profile_job(compile_job, device_name: str, options: str = ""):
    """
    Wait for compile_job to finish using polling (not .wait()), call get_target_model()
    (confirmed via dir(hub.CompileJob)), then submit a profile job.

    Returns the profile job object (hub.ProfileJob).
    """
    _require_hub()
    device = find_device(device_name)

    print("[hub_client] Waiting for compile job ...")
    # Use polling instead of .wait() to avoid Windows cp1252 codec error
    _poll_job_job(compile_job, timeout_sec=600)

    # get_target_model() confirmed via dir(hub.CompileJob)
    target_model = compile_job.get_target_model()

    kwargs: dict = dict(model=target_model, device=device)
    if options:
        kwargs["options"] = options

    profile_job = hub.submit_profile_job(**kwargs)
    # .url confirmed via dir(hub.ProfileJob)
    job_url = getattr(profile_job, "url", "URL_UNAVAILABLE")
    print(f"[hub_client] Profile job: {job_url}")
    return profile_job


# ---------------------------------------------------------------------------
# Download profile
# ---------------------------------------------------------------------------

def _poll_job_job(job, timeout_sec=600, poll_interval=15):
    """Poll a job until it's done, without using the buggy .wait() method."""
    _require_hub()
    start = time.time()
    last_code = None
    while time.time() - start < timeout_sec:
        status = job.get_status()
        code = status.code
        elapsed = int(time.time() - start)
        # Log the exact type and value on first poll so we know what we're comparing
        if last_code is None:
            print(f"  [poll] status.code type={type(code).__name__}, repr={code!r}")
        print(f"  Job status: code={code} ({elapsed}s)")
        last_code = code
        # Check all known terminal status strings from AI Hub
        # (different job types may use different casing)
        if code in ("completed", "failed", "error",
                     "SUCCESS", "FAILED", "ERROR"):
            return code
        time.sleep(poll_interval)
    # Safeguard: print what we were stuck on so future unknown codes are visible
    print(f"  [poll] TIMEOUT — last status.code was: {last_code!r} "
          f"(type={type(last_code).__name__}). Not in terminal set.")
    raise TimeoutError(f"Job did not complete within {timeout_sec}s")


def wait_and_download_profile(
    profile_job,
    save_path: Optional[str] = None,
) -> dict:
    """
    Wait for profile_job to complete then download the raw profile dict.
    Saves to save_path (JSON) if provided.

    Uses _poll_job_job() instead of .wait() to avoid Windows cp1252
    codec error when status codes contain emoji-like characters.
    Methods used (all confirmed via dir(hub.ProfileJob)):
      .get_status(), .download_profile()
    """
    _require_hub()
    print("[hub_client] Waiting for profile job ...")
    poll_code = _poll_job_job(profile_job, timeout_sec=600)

    if poll_code not in ("completed", "SUCCESS"):
        raise RuntimeError(
            f"Profile job ended with code {poll_code}; expected 'completed' or 'SUCCESS'"
        )

    # .download_profile() confirmed
    profile = profile_job.download_profile()
    profile["job_id"] = getattr(profile_job, "job_id", None) or profile_job.id

    if save_path:
        dest = pathlib.Path(save_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as fh:
            json.dump(profile, fh, indent=2, default=str)
        print(f"[hub_client] Raw profile saved: {dest}")

    return profile


# ---------------------------------------------------------------------------
# Quantize
# ---------------------------------------------------------------------------

def submit_quantize_job(
    model,
    calibration_data,
    weights_dtype="INT8",
    activations_dtype="INT8",
    name=None,
    options="",
    project=None,
):
    """
    Submit a quantize job to AI Hub.

    Real API signature (verified via inspect.signature):
    submit_quantize_job(model, calibration_data, weights_dtype, activations_dtype, name, options, project)
    Returns QuantizeJob.

    Methods confirmed via dir():
      .get_target_model(), .url, .wait()
    """
    _require_hub()
    model = hub.upload_model(str(model)) if hasattr(model, 'suffix') and str(model).suffix.lower() == '.onnx' else hub.upload_model(model)

    job = hub.submit_quantize_job(
        model=model,
        calibration_data=calibration_data,
        weights_dtype=weights_dtype,
        activations_dtype=activations_dtype,
        name=name,
        options=options,
        project=project,
    )
    # .url confirmed via dir(hub.QuantizeJob)
    job_url = getattr(job, "url", "URL_UNAVAILABLE")
    print(f"[hub_client] Quantize job: {job_url}")
    return job


# ---------------------------------------------------------------------------
# Convenience pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    model_path: str,
    device_name: str = DEFAULT_PRIMARY_DEVICE,
    save_path: Optional[str] = None,
    compile_options: str = "--target_runtime onnx",
    profile_options: str = "",
) -> dict:
    """compile -> profile -> download.  Returns raw profile dict."""
    c_job   = submit_compile_job(model_path, device_name, compile_options)
    p_job   = submit_profile_job(c_job, device_name, options=profile_options)
    profile = wait_and_download_profile(p_job, save_path=save_path)
    return profile


# ---------------------------------------------------------------------------
# List devices (for CLI)
# ---------------------------------------------------------------------------

def list_devices() -> list[str]:
    _require_hub()
    return [d.name for d in hub.get_devices()]
