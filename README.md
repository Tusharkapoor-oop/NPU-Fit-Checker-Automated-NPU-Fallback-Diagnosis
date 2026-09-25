# NPU Fit Checker

Qualcomm AI Hub's dashboard tells you *how many* layers ran on the NPU versus CPU. It doesn't tell you **which** layer fell back, **why** it fell back, or **how to fix it**.

**NPU Fit Checker explains WHICH layer fell back and WHY, automatically — AI Hub's dashboard only shows you a count.**

Validated on one model family, designed to generalize.

---

## The Problem

When deploying AI models on Snapdragon X Elite and X2 Elite NPUs, developers often encounter silent CPU fallbacks. A model might show 98% NPU execution on Qualcomm AI Hub's dashboard, but the developer is left guessing:
- Which specific layers were rejected by the Hexagon NPU?
- Are those CPU layers critical architectural incompatibilities, or harmless I/O adapters?
- What exact compiler flag, graph transformation, or quantization step restores acceleration?

Without automated diagnostics, developers waste engineering hours manually inspecting hundreds of graph nodes.

## The Solution

NPU Fit Checker automates the diagnosis and optimization pipeline:
1. **Profiles** the model directly on Qualcomm AI Hub's cloud-hosted Snapdragon hardware.
2. **Parses** the empirical execution summary and per-layer hardware telemetry.
3. **Classifies** layer behavior using deterministic diagnostic rules, distinguishing between genuine unsupported operator fallbacks and expected QDQ boundary adapters.
4. **Reports** plain-language diagnoses with actionable remediation steps and verified hardware before/after metrics.

---

## Real Measured Results (Snapdragon X2 Elite CRD)

All numbers below are extracted directly from real Qualcomm AI Hub execution profiles ([`experiments/baseline_fp32/profile.json`](experiments/baseline_fp32/profile.json) and [`experiments/optimized_int8/profile.json`](experiments/optimized_int8/profile.json)) on a cloud Snapdragon X2 Elite CRD (`sc8480xp`, Hexagon v81 DSP, Windows 11).

| Metric | Baseline (FP32) | Optimized (INT8 QDQ) | Delta / Improvement | Source Field |
|---|---|---|---|---|
| **AI Hub Compile Job** | [`jgzl4r6x5`](https://workbench.aihub.qualcomm.com/jobs/jgzl4r6x5/) | [`jgzl47oz5`](https://workbench.aihub.qualcomm.com/jobs/jgzl47oz5/) | — | Job ID |
| **AI Hub Profile Job** | [`j5m0d379g`](https://workbench.aihub.qualcomm.com/jobs/j5m0d379g/) | [`j57e7d0qp`](https://workbench.aihub.qualcomm.com/jobs/j57e7d0qp/) | — | Job ID |
| **Inference Latency** | **281 µs** (0.281 ms) | **154 µs** (0.154 ms) | **1.82× faster (−45.2%)** | `execution_summary.estimated_inference_time` |
| **Total Layers** | 104 | 141 | +37 QDQ operators | `len(execution_detail)` |
| **NPU Layers** | 104 (100.0%) | 139 (98.58%) | +35 layers | `compute_unit == "NPU"` |
| **CPU Fallback Layers** | 0 (0.0%) | 2 (1.42%) | Expected boundary ops | `compute_unit == "CPU"` |
| **Total Process Peak Memory** | 32,792,576 bytes (~31.3 MB) | 32,915,456 bytes (~31.4 MB) | +122,880 bytes (+0.37%) | `execution_summary.estimated_inference_peak_memory` |
| **Runtime Working Set (Dashboard "2 MB")** | 1,708,032 bytes (~1.63 MB) | 1,716,224 bytes (~1.64 MB) | +8,192 bytes | `execution_summary.inference_memory_peak_range[0]` |
| **Warm Load Time** | 331,363 µs (~331.4 ms) | 328,921 µs (~328.9 ms) | −2,442 µs | `execution_summary.warm_load_time` |

### Memory Metric Reconciliation: 32.8 MB vs Dashboard "2 MB"
In the profile JSON, Qualcomm AI Hub reports two distinct memory metrics:
- **`estimated_inference_peak_memory` (32,792,576 bytes ≈ 31.3 MB)**: Measures the overall process peak memory footprint, including graph allocations, model weights, and runtime context buffers.
- **`inference_memory_peak_range` (1,708,032 bytes ≈ 1.63 MB)**: Measures the transient runtime working set per inference invocation. This is the exact metric rounded and displayed on the AI Hub web dashboard as **"Estimated Peak Memory Usage: 2 MB"**.

Both metrics are real and non-contradictory: one measures total process footprint, the other measures inference working set delta.

---

## Technical Contribution: Boundary QDQ Adapters Are Expected, Not Bugs

When profiling the INT8 QDQ model, the profile reports 2 layers executing on CPU:
- `QcQuantizeOp_input_q` (type `QuantizeLinear`, execution time: 116 µs)
- `QcQuantizeOp_output_dq` (type `DequantizeLinear`, execution time: 53 µs)

### Why They Run on CPU
When an ONNX model is quantized to INT8 with QDQ format, the host environment supplies FP32 tensors at the model input and expects FP32 predictions at the model output. The input `QuantizeLinear` converts FP32 to INT8, and the output `DequantizeLinear` converts INT8 back to FP32. Because they interface with host-side application memory, they execute on the host CPU execution provider. Every single one of the 139 internal neural network layers (Conv, BatchNorm/Clip, Gemm, Reshape, Transpose, GlobalAveragePool) executes 100% on the Hexagon NPU.

### The Harder Problem We Solve
A naive diagnostic tool simply counts "2 CPU fallbacks" and alerts the user that the model has failed compilation or contains unsupported operators.

**`QcQuantizeOp_input_q` and `QcQuantizeOp_output_dq` are expected QDQ boundary adapters, not bugs.**

Correctly classifying them as expected boundary transitions (rather than flagging them as erroneous fallback failures) is the real technical contribution of intelligent NPU fit analysis. NPU Fit Checker distinguishes harmless graph-boundary data marshalling from genuine internal unsupported operators that fracture NPU execution, preventing developers from wasting time debugging expected behavior.

---

## Diagnostic Rules Engine

The tool evaluates profiles using deterministic rules backed by empirical evidence:

| Rule Name | Trigger Condition | Severity | Plain-Language Diagnosis | Evidence Source |
|---|---|---|---|---|
| `full_npu_execution` | 100% layers on NPU, 0 CPU/GPU | INFO | All layers fit the Hexagon NPU completely. | [`experiments/baseline_fp32/profile.json`](experiments/baseline_fp32/profile.json) |
| `cpu_fallback_detected` | Any CPU layer detected | WARNING | Explains whether CPU ops are boundary QDQ adapters or internal unsupported ops. | [`experiments/optimized_int8/profile.json`](experiments/optimized_int8/profile.json) |
| `no_latency_data` | Missing `estimated_inference_time` | WARNING | Structural schema guard when profile execution data is incomplete. | [`experiments/baseline_fp32/profile.json`](experiments/baseline_fp32/profile.json) |

---

## Quick Start

### Installation

```bash
git clone https://github.com/<your-username>/npu-fit-checker.git
cd npu-fit-checker
pip install -r requirements.txt
```

### Configure AI Hub Token (Once)

```bash
qai-hub configure --api_token <your_token>
```

### Run on a Model (Submits to AI Hub Cloud)

```bash
python -m fitchecker run --model mobilenet_v2.onnx --device "Snapdragon X2 Elite CRD"
```

### Run Offline on a Downloaded Profile

```bash
python -m fitchecker run --from-profile experiments/baseline_fp32/profile.json
```

---

## Evidence & Verification

Every result is backed by verifiable artifacts and jobs on Qualcomm AI Hub:

- **Baseline Jobs**:
  - Compile: [`jgzl4r6x5`](https://workbench.aihub.qualcomm.com/jobs/jgzl4r6x5/)
  - Profile: [`j5m0d379g`](https://workbench.aihub.qualcomm.com/jobs/j5m0d379g/)
- **Optimized Jobs**:
  - Quantize: [`jgjrwld7p`](https://workbench.aihub.qualcomm.com/jobs/jgjrwld7p/)
  - Compile: [`jgzl47oz5`](https://workbench.aihub.qualcomm.com/jobs/jgzl47oz5/)
  - Profile: [`j57e7d0qp`](https://workbench.aihub.qualcomm.com/jobs/j57e7d0qp/)
- **Evidence Screenshots**:
  - Baseline Compile: [`docs/evidence_screenshots/baseline_compile_jgzl4r6x5.png`](docs/evidence_screenshots/baseline_compile_jgzl4r6x5.png)
  - Baseline Profile: [`docs/evidence_screenshots/baseline_profile_j5m0d379g.png`](docs/evidence_screenshots/baseline_profile_j5m0d379g.png)
  - Optimized Profile: [`docs/evidence_screenshots/optimized_profile_j57e7d0qp.png`](docs/evidence_screenshots/optimized_profile_j57e7d0qp.png)
- **Detailed Documentation**:
  - [EVIDENCE_LOG.md](docs/EVIDENCE_LOG.md) — Comprehensive claim-to-file traceability log
  - [COMPARISON.md](experiments/COMPARISON.md) — Side-by-side empirical performance comparison
  - [PROFILE_SCHEMA.md](docs/PROFILE_SCHEMA.md) — Verified schema documentation for AI Hub telemetry
  - [PRIOR_ART.md](docs/PRIOR_ART.md) — Comparison against existing tools (Windows ML, Workbench, Task Manager)
  - [LIMITS.md](docs/LIMITS.md) — Complete disclosure of tool constraints and assumptions

---

## Honest Device Statement

> **Hardware Environment**: All profiling jobs were executed on Qualcomm AI Hub's cloud-hosted Snapdragon compute devices (`Snapdragon X2 Elite CRD` / `Snapdragon X Elite CRD`). No physical HP hardware was accessed or used for these experiments.

---

## Known Limitations

- **Model Scope**: Rules validated on one model family (MobileNetV2, FP32 and INT8 variants). Not yet tested against a model with a genuinely unsupported operator. Designed to generalize across standard ONNX architectures.
- **Cloud Device Only**: Operates via Qualcomm AI Hub cloud Workbench. Does not perform on-device profiling on local laptops.
- **Pattern-Based Rules**: Diagnoses are produced via deterministic structural heuristics over AI Hub profiles, not machine learning classifiers.
- **Boundary Classification**: QDQ boundary classification tested on MobileNetV2 architecture.
- Full details in [docs/LIMITS.md](docs/LIMITS.md).

---

## License

MIT