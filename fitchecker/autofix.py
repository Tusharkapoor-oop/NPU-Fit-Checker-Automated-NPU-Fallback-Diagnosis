"""
fitchecker/autofix.py

Closed-loop auto-quantization and re-profiling.

API signature for submit_quantize_job (verified from official docs at
https://workbench.aihub.qualcomm.com/docs/hub/quantize_examples.html):

    client.submit_quantize_job(
        model=<hub.Model>,              # ONNX model (from compile job)
        calibration_data=<dict>,         # {input_name: [np.ndarray, ...]}
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
    )

QuantizeJob methods (from experiments/api_methods.txt):
    get_status, get_target_model, url, wait, download_target_model,
    download_job_logs, download_results, etc.
"""

from __future__ import annotations

import pathlib
import datetime
import sys
import traceback
import numpy as np

try:
    import qai_hub as hub
    QAI_HUB_AVAILABLE = True
except ImportError:
    QAI_HUB_AVAILABLE = False

from fitchecker.hub_client import (
    find_device,
    submit_compile_job,
    submit_profile_job,
    wait_and_download_profile,
    _poll_job_job,
    _require_hub,
)
from fitchecker.profile_parser import ProfileResult, load_profile
from fitchecker.rules import apply_rules
from fitchecker.report import build_markdown_report, print_terminal_report, save_report


def detect_optimization_opportunity(profile: ProfileResult) -> bool:
    """
    Returns True if the profile shows 100% NPU execution but the model
    was NOT run with --onnx_execution_providers=qnn on a quantized model
    (i.e., FP32, already fully on NPU, but INT8 hasn't been tried yet).
    """
    if profile.total_layer_count > 0 and profile.npu_layer_count == profile.total_layer_count:
        return True
    if profile.npu_layer_count > 0 and profile.cpu_layer_count == 0:
        return True
    return False


def run_autofix_loop(onnx_model_path: str, device_name: str, output_dir: str = "experiments/autofix_result"):
    """
    a. Loads the original FP32 model
    b. Calls submit_quantize_job using the REAL API signature
    c. Compiles the quantized model (options "--target_runtime onnx")
    d. Profiles it on the SAME device, with options "--onnx_execution_providers=qnn"
    e. Saves the new raw profile JSON to experiments/autofix_result/
    f. Loads both and produces a before/after comparison
    """
    _require_hub()
    p = pathlib.Path(onnx_model_path)
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== AUTOFIX PIPELINE: {p.name} on {device_name} ===")
    print(f"    Output dir: {out_dir.resolve()}")
    sys.stdout.flush()

    device = find_device(device_name)

    # ===================================================================
    # STEP 1: Compile FP32 model to ONNX (baseline)
    # ===================================================================
    print("\n[STEP 1/6] Compiling unquantized ONNX model...")
    sys.stdout.flush()
    c_job = submit_compile_job(onnx_model_path, device_name, compile_options="--target_runtime onnx")
    print(f"  [DIAG] Compile job submitted. Polling with 1800s timeout...")
    sys.stdout.flush()
    compile_status = _poll_job_job(c_job, timeout_sec=1800)
    print(f"  [DIAG] Compile job finished with status: {compile_status}")
    sys.stdout.flush()

    if compile_status not in ("completed", "SUCCESS"):
        print(f"  [FATAL] Compile job did not complete. Status: {compile_status}. Aborting.")
        return None

    print("  [DIAG] Calling c_job.get_target_model()...")
    sys.stdout.flush()
    unquantized_onnx_model = c_job.get_target_model()
    print(f"  [DIAG] Got target model: {type(unquantized_onnx_model)} = {unquantized_onnx_model}")
    sys.stdout.flush()

    # ===================================================================
    # STEP 2: Profile FP32 model (baseline)
    # ===================================================================
    print("\n[STEP 2/6] Profiling unquantized model (baseline)...")
    sys.stdout.flush()

    # NOTE: submit_profile_job in hub_client.py internally re-polls compile_job
    # (already completed, should be instant) then calls hub.submit_profile_job.
    # wait_and_download_profile has its own 600s timeout internally.
    p_job = submit_profile_job(c_job, device_name)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_stem = p.stem
    baseline_path = out_dir / f"{model_stem}_{ts}_baseline_profile.json"

    print(f"  [DIAG] Waiting for profile job to finish (600s internal timeout)...")
    sys.stdout.flush()
    wait_and_download_profile(p_job, save_path=str(baseline_path))
    print(f"  [DIAG] Baseline profile saved to: {baseline_path}")
    sys.stdout.flush()

    baseline_profile = load_profile(str(baseline_path), device_name=device_name, model_name=model_stem)
    print(f"  [DIAG] Baseline parsed: {baseline_profile.total_layer_count} layers, "
          f"NPU={baseline_profile.npu_layer_count}, CPU={baseline_profile.cpu_layer_count}, "
          f"GPU={baseline_profile.gpu_layer_count}")
    sys.stdout.flush()

    # ===================================================================
    # STEP 3: Detect optimization opportunity
    # ===================================================================
    print("\n[STEP 3/6] Checking optimization opportunity...")
    sys.stdout.flush()
    is_opportunity = detect_optimization_opportunity(baseline_profile)
    print(f"  [DIAG] detect_optimization_opportunity() returned: {is_opportunity}")
    print(f"  [DIAG]   total_layer_count={baseline_profile.total_layer_count}")
    print(f"  [DIAG]   npu_layer_count={baseline_profile.npu_layer_count}")
    print(f"  [DIAG]   cpu_layer_count={baseline_profile.cpu_layer_count}")
    sys.stdout.flush()

    if not is_opportunity:
        print(f"  [EXIT] Model does not meet autofix criteria. Returning baseline profile.")
        return baseline_profile

    print(f"  [DIAG] Optimization opportunity confirmed. Proceeding to quantize.")
    sys.stdout.flush()

    # ===================================================================
    # STEP 4: Build calibration data and submit quantize job
    # ===================================================================
    print("\n[STEP 4/6] Preparing calibration data and submitting quantize job...")
    sys.stdout.flush()

    # Get target shapes from compile job to build calibration data
    print("  [DIAG] Calling c_job.get_target_shapes()...")
    sys.stdout.flush()
    try:
        shapes = c_job.get_target_shapes()
    except Exception as e:
        print(f"  [ERROR] get_target_shapes() failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        return None
    print(f"  [DIAG] get_target_shapes() returned: {shapes}")
    print(f"  [DIAG]   type={type(shapes)}, keys={list(shapes.keys()) if isinstance(shapes, dict) else 'NOT A DICT'}")
    sys.stdout.flush()

    # Build calibration data dict
    calibration_data = {}
    for input_name, shape_data in shapes.items():
        if isinstance(shape_data, tuple) and len(shape_data) == 2:
            shape_tuple, _ = shape_data
        else:
            shape_tuple = shape_data
        
        arr = np.random.uniform(0, 1, size=shape_tuple).astype(np.float32)
        calibration_data[input_name] = [arr]
        print(f"  [DIAG] calibration_data['{input_name}'] = [array shape={arr.shape}, dtype={arr.dtype}]")
    sys.stdout.flush()

    # CRITICAL CHECK: verify input name matches what the model actually expects
    # From AI Hub job pages, MobileNetV2 input is named "input" (lowercase).
    print(f"  [DIAG] calibration_data keys: {list(calibration_data.keys())}")
    print(f"  [DIAG] Expected input name from AI Hub job page: 'input'")
    if "input" not in calibration_data:
        print(f"  [WARNING] 'input' not found in calibration_data keys! "
              f"Keys are: {list(calibration_data.keys())}. This may cause a mismatch.")
    sys.stdout.flush()

    # Submit quantize job with explicit try/except
    print(f"\n  [DIAG] About to call hub.submit_quantize_job() with:")
    print(f"  [DIAG]   model={type(unquantized_onnx_model)} = {unquantized_onnx_model}")
    print(f"  [DIAG]   calibration_data keys={list(calibration_data.keys())}")
    for k, v in calibration_data.items():
        print(f"  [DIAG]   calibration_data['{k}']: {len(v)} sample(s), shape={v[0].shape}, dtype={v[0].dtype}")
    print(f"  [DIAG]   weights_dtype=hub.QuantizeDtype.INT8")
    print(f"  [DIAG]   activations_dtype=hub.QuantizeDtype.INT8")
    sys.stdout.flush()

    try:
        quantize_job = hub.submit_quantize_job(
            model=unquantized_onnx_model,
            calibration_data=calibration_data,
            weights_dtype=hub.QuantizeDtype.INT8,
            activations_dtype=hub.QuantizeDtype.INT8,
        )
    except Exception as e:
        print(f"\n  [FATAL] hub.submit_quantize_job() raised an exception!")
        print(f"  [FATAL] Exception type: {type(e).__name__}")
        print(f"  [FATAL] Exception message: {e}")
        print(f"  [FATAL] Full traceback:")
        traceback.print_exc()
        sys.stdout.flush()
        return None

    quant_url = getattr(quantize_job, 'url', 'URL_UNAVAILABLE')
    print(f"  [DIAG] Quantize job submitted! URL: {quant_url}")
    print(f"  [DIAG] Polling quantize job with 1800s timeout...")
    sys.stdout.flush()

    quant_status = _poll_job_job(quantize_job, timeout_sec=1800)
    print(f"  [DIAG] Quantize job finished with status: {quant_status}")
    sys.stdout.flush()

    if quant_status not in ("completed", "SUCCESS"):
        print(f"  [FATAL] Quantize job did not complete. Status: {quant_status}. Aborting.")
        return None

    quantized_onnx_model = quantize_job.get_target_model()
    print(f"  [DIAG] Got quantized model: {type(quantized_onnx_model)} = {quantized_onnx_model}")
    sys.stdout.flush()

    # ===================================================================
    # STEP 5: Compile quantized model
    # ===================================================================
    print("\n[STEP 5/6] Compiling quantized model...")
    sys.stdout.flush()
    c_quant_job = hub.submit_compile_job(
        model=quantized_onnx_model,
        device=device,
        options="--target_runtime onnx",
    )
    c_quant_url = getattr(c_quant_job, 'url', 'URL_UNAVAILABLE')
    print(f"  [DIAG] Compile (quantized) job URL: {c_quant_url}")
    sys.stdout.flush()
    _poll_job_job(c_quant_job, timeout_sec=1800)

    # ===================================================================
    # STEP 6: Profile quantized model with QNN
    # ===================================================================
    print("\n[STEP 6/6] Profiling quantized model with --onnx_execution_providers=qnn...")
    sys.stdout.flush()
    p_quant_job = submit_profile_job(c_quant_job, device_name, options="--onnx_execution_providers=qnn")

    quant_path = out_dir / f"{model_stem}_{ts}_quantized_profile.json"
    wait_and_download_profile(p_quant_job, save_path=str(quant_path))
    quant_profile = load_profile(str(quant_path), device_name=device_name, model_name=model_stem)

    # ===================================================================
    # Report results using existing report.py comparison logic
    # ===================================================================
    print("\n[REPORT] Generating comparison report...")
    sys.stdout.flush()
    findings = apply_rules(quant_profile)
    print_terminal_report(quant_profile, findings, baseline_profile=baseline_profile)

    report_md = build_markdown_report(
        quant_profile,
        findings,
        baseline_profile=baseline_profile,
        title=f"Autofix Report: {model_stem}",
        baseline_label="FP32 Baseline",
        optimized_label="INT8 Auto-Quantized",
    )
    
    # Append the mandatory caveat about synthetic calibration data
    caveat = (
        "\n\n## Caveats and Limits\n\n"
        "- Calibration data is synthetic (uniform random noise), used to measure real latency on real hardware. "
        "This validates timing, not model accuracy. A production system would calibrate against representative input samples.\n"
    )
    report_md += caveat
    
    report_path = out_dir / f"{model_stem}_{ts}_autofix_report.md"
    save_report(report_md, str(report_path))

    print(f"\n[Autofix] Pipeline complete! Report saved to {report_path}")
    sys.stdout.flush()
    return quant_profile