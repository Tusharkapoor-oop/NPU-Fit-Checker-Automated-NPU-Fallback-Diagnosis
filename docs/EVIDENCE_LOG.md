# NPU Fit Checker — Evidence Log

This file maps every metric, claim, and rule in the project directly to its verified source file or Qualcomm AI Hub job URL.

---

## 1. Hardware & Environment
- **Platform**: Python 3.13 on Windows x64
- **SDK**: `qai-hub` 0.55.0 with configured API token
- **Device List**: Extracted from `qai-hub list-devices` and saved to `experiments/devices.txt`:
  - `Snapdragon X2 Elite CRD` (OS: Windows 11, Format: Compute, Chipset: `sc8480xp`, Hexagon DSP: `v81`)
  - `Snapdragon X Elite CRD` (OS: Windows 11, Format: Compute, Chipset: `sc8380xp`, Hexagon DSP: `v73`)
  - `Snapdragon X Plus 8-Core CRD` (OS: Windows 11, Format: Compute, Chipset: `sc8380xp`, Hexagon DSP: `v73`)
- **API Surface**: Confirmed via `experiments/api_methods.txt` (dir inspection for `ProfileJob`, `CompileJob`, `Device`, `Client`)

---

## 2. Baseline Experiment (FP32)
- **Model**: MobileNetV2 (torchvision weights `DEFAULT`, 224×224, batch 1, exported to `mobilenet_v2.onnx`)
- **Compile Job ID**: `jgzl4r6x5`
  - URL: https://workbench.aihub.qualcomm.com/jobs/jgzl4r6x5/
  - Options: `--target_runtime onnx`
- **Profile Job ID**: `j5m0d379g`
  - URL: https://workbench.aihub.qualcomm.com/jobs/j5m0d379g/
  - Device: Qualcomm AI Hub cloud-hosted Snapdragon X2 Elite CRD
- **Local Profile JSON**: `experiments/baseline_fp32/profile.json` (21,762 bytes)
- **Verified Metrics**:
  - `len(execution_detail)` = **104 layers**
  - Compute unit breakdown: **104 NPU (100.0%)**, **0 CPU (0.0%)**, **0 GPU (0.0%)**
  - `execution_summary.estimated_inference_time` = **281 µs** (0.281 ms)
  - `execution_summary.estimated_inference_peak_memory` = **32,792,576 bytes** (~31.3 MB total process peak memory)
  - `execution_summary.inference_memory_peak_range` = **[1,708,032, 1,708,032] bytes** (~1.63 MB working set delta, shown as "2 MB" on AI Hub web dashboard)
  - `execution_summary.warm_load_time` = **331,363 µs** (~331.4 ms)

---

## 3. Optimized Experiment (INT8 QDQ)
- **Quantize Job ID**: `jgjrwld7p`
  - URL: https://workbench.aihub.qualcomm.com/jobs/jgjrwld7p/
- **Compile Job ID**: `jgzl47oz5`
  - URL: https://workbench.aihub.qualcomm.com/jobs/jgzl47oz5/
  - Options: `--target_runtime onnx`
- **Profile Job ID**: `j57e7d0qp`
  - URL: https://workbench.aihub.qualcomm.com/jobs/j57e7d0qp/
  - Device: Qualcomm AI Hub cloud-hosted Snapdragon X2 Elite CRD
  - Options: `--onnx_execution_providers=qnn`
- **Local Profile JSON**: `experiments/optimized_int8/profile.json` (31,598 bytes)
- **Verified Metrics**:
  - `len(execution_detail)` = **141 layers** (104 base + 37 QDQ adaptation ops)
  - Compute unit breakdown: **139 NPU (98.58%)**, **2 CPU (1.42%)**, **0 GPU (0.0%)**
  - `execution_summary.estimated_inference_time` = **154 µs** (0.154 ms) → **1.82× speedup / 45.2% faster**
  - `execution_summary.estimated_inference_peak_memory` = **32,915,456 bytes** (~31.4 MB total process peak memory)
  - `execution_summary.inference_memory_peak_range` = **[1,716,224, 1,716,224] bytes** (~1.64 MB working set delta, shown as "2 MB" on AI Hub web dashboard)
  - `execution_summary.warm_load_time` = **328,921 µs** (~328.9 ms)
  - CPU Layers:
    - `QcQuantizeOp_input_q` (type `QuantizeLinear`, execution_time: 116 µs)
    - `QcQuantizeOp_output_dq` (type `DequantizeLinear`, execution_time: 53 µs)
  - Explanation: Expected boundary QDQ adapters transitioning FP32 host data into and out of the INT8 Hexagon NPU execution graph.

---

## 4. Rules & Evidence Traceability
- **`full_npu_execution`**: Traces to `experiments/baseline_fp32/profile.json` (104/104 NPU layers).
- **`cpu_fallback_detected`**: Traces to `experiments/optimized_int8/profile.json` (2 boundary QDQ ops on CPU).
- **`no_latency_data`**: Structural guard verified against `experiments/baseline_fp32/profile.json`.

---

## 5. Evidence Artifacts
- Baseline Profile: `experiments/baseline_fp32/profile.json`
- Optimized Profile: `experiments/optimized_int8/profile.json`
- Hardware Devices: `experiments/devices.txt`
- Schema Document: `docs/PROFILE_SCHEMA.md`
- Side-by-Side Comparison: `experiments/COMPARISON.md`
- Limitations & Scope: `docs/LIMITS.md`
- Evidence Screenshots: `docs/evidence_screenshots/`