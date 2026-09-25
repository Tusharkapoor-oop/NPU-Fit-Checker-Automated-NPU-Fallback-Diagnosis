# Competition Submission Draft: NPU Fit Checker

**Competition**: Snapdragon AI Lab Build & Present Challenge (2026)  
**Track**: Developer Tools & PC-Class AI Enablement  
**Target Hardware**: Snapdragon X2 Elite CRD (Primary), Snapdragon X Elite CRD (Secondary)  

---

## 1. Project Title & Tagline

- **Project Title**: NPU Fit Checker
- **Tagline**: Automated fallback diagnosis and plain-language optimization intelligence for AI models on Snapdragon Compute NPUs.

---

## 2. Executive Summary (150 words)

Deploying AI models onto Snapdragon X Elite and X2 Elite Windows devices often suffers from "silent CPU fallback" — when unsupported graph operators are evicted from the Hexagon NPU to the host CPU, degrading latency and battery efficiency. While Qualcomm AI Hub provides cloud profiling, its output is a dense, multi-thousand-line JSON payload that leaves developers guessing why operators fell back and how to fix them.

NPU Fit Checker is an open-source developer CLI that ingests ONNX models, executes compile and profile jobs on Qualcomm AI Hub's cloud-hosted Snapdragon devices, and translates raw hardware metrics into clear, actionable diagnosis. In our empirical benchmarks on the **Snapdragon X2 Elite CRD**, NPU Fit Checker demonstrated:
1. Complete 100% NPU mapping for MobileNetV2 FP32 (281 µs).
2. A **1.82× speedup** (154 µs) under INT8 quantization.
3. Pinpoint detection of boundary QDQ adapter fallbacks (`QuantizeLinear` / `DequantizeLinear`) on the CPU with actionable developer fixes.

---

## 3. Problem Statement & Novelty

### The Problem
Developers migrating models to Windows on Snapdragon face a high barrier to understanding NPU compilation behavior:
- Compilers provide little insight when an op falls back to CPU.
- Profiling JSONs require manual scripting to parse `execution_detail` cycles and times.
- Quantized models often incur hidden boundary adaptation overhead at the host interface.

### Novelty & Technical Contribution
NPU Fit Checker bridges the gap between raw hardware telemetry and developer action:
- **Strict Confirmed Schema Parser**: Validates `execution_summary` and `execution_detail` with microsecond-level timing precision, raising explicit schema errors rather than hallucinating compute units.
- **Evidence-Backed Rule Engine**: Rules are strictly tied to reproducible empirical files on disk (`experiments/`).
- **Differential Before/After Benchmarking**: Measures exact speedup and layer transitions between baseline FP32 and optimized INT8 variants.
- **Zero Fabrication Guarantee**: Every number presented is audited and cross-referenced in an immutable `EVIDENCE_LOG.md`.

---

## 4. Empirical Benchmark Data (Traceability Matrix)

All data obtained from Qualcomm AI Hub cloud profiling on **Snapdragon X2 Elite CRD**:

| Metric | Baseline (FP32) | Optimized (INT8 QDQ) | Impact | Evidence Reference |
|---|---|---|---|---|
| **Inference Latency** | 281 µs (0.281 ms) | 154 µs (0.154 ms) | **1.82× speedup** (45.2% latency reduction) | `experiments/baseline_fp32/profile.json` vs `experiments/optimized_int8/profile.json` |
| **Total Layers** | 104 | 141 | +37 QDQ adaptation ops | `experiments/COMPARISON.md` |
| **NPU Placement** | 104 / 104 (100.0%) | 139 / 141 (98.6%) | 139 layers accelerated | `experiments/COMPARISON.md` |
| **CPU Fallback** | 0 (0.0%) | 2 (1.4%) | `QcQuantizeOp_input_q` (116 µs), `QcQuantizeOp_output_dq` (53 µs) | `experiments/COMPARISON.md` |
| **Compile Options** | `--target_runtime onnx` | `--target_runtime onnx` | Target ONNX execution runtime | Qualcomm AI Hub Compile Job `jgzl4r6x5`, `jgzl47oz5` |
| **Profile Options** | (default) | `--onnx_execution_providers=qnn` | Target QNN Execution Provider | Qualcomm AI Hub Profile Job `j5m0d379g`, `j57e7d0qp` |

---

## 5. Technology Stack & Implementation

- **Language**: Python 3.13 (Windows x64)
- **SDK**: Qualcomm AI Hub Python SDK (`qai_hub` v0.55.0)
- **Model Framework**: ONNX (Opset 17), PyTorch / TorchVision (MobileNetV2)
- **Terminal UI**: Rich library (custom styling, tables, panels)
- **Testing**: PyTest (30 unit tests covering device filtering, schema validation, and rule triggers)

---

## 6. Self-Audit & Integrity Verification

Every single claim, layer count, and latency value in our submission has been audited against `docs/EVIDENCE_LOG.md`:
- Primary Device: Snapdragon X2 Elite CRD (Chipset `sc8480xp`, Hexagon DSP `v81`).
- Secondary Device: Snapdragon X Elite CRD (Chipset `sc8380xp`, Hexagon DSP `v73`).
- No physical laptop testing claimed; all hardware was accessed via Qualcomm AI Hub cloud infrastructure.
- Zero guessed or estimated numbers.
