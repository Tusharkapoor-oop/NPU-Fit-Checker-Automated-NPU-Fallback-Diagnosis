# NPU Fit Checker — demo_profile
*Generated: 2026-09-21 20:37:02*

> **Device**: Qualcomm AI Hub cloud-hosted Snapdragon X Elite device  
> **Model**: MobileNetV2  
> **Job ID**: demo-001  

## Latency

| Metric | Value |
|--------|-------|
| Total inference time | 12.500 ms (12500 µs) |

## Compute-Unit Breakdown

| Compute Unit | Layer Count | Fraction |
|---|---|---|
| NPU | 2 | 50.0% |
| CPU | 2 | 50.0% |
| GPU | 0 | 0.0% |
| UNASSIGNED | 0 | 0.0% |
| **Total** | 4 | 100.0% |

## Findings

### Finding 1: [WARN] cpu_fallback_detected

**Affected layers** (2): `custom_gelu`, `fc`

**What happened**: Qualcomm's QNN runtime moves a layer to CPU when it cannot compile that op for the NPU. Common causes: the operator is not in QNN's supported op list for this SoC, or the layer's tensor rank / shape is not supported.

**Fix**: 1. Check the QNN supported-ops list for Snapdragon X Elite at https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/introduction.html.
2. Replace unsupported ops with QNN-compatible alternatives (e.g., replace custom activations with ReLU/GELU).
3. If the op cannot be replaced, wrap it in a CPU-only sub-graph and keep the rest on NPU.

**Evidence**: `UNVERIFIED – will be set after Phase 2`

### Finding 2: [WARN] low_npu_utilisation

**Affected layers** (2): `custom_gelu`, `fc`

**What happened**: When most layers fall back to CPU the model will not benefit from the HTP (Hexagon Tensor Processor) hardware. This can happen with FP32 models on devices where the NPU only accepts quantized (INT8/FP16) inputs, or with unusual op combinations.

**Fix**: 1. Quantize the model to INT8 using AI Hub's quantize workflow (hub.submit_quantize_job) or ONNX Runtime's QNN EP.
2. Check for dynamic shapes: replace dynamic axes with fixed batch / sequence sizes where possible.
3. Inspect the per-layer compute_unit column in the report and target the heaviest CPU-resident layers first.

**Evidence**: `UNVERIFIED – will be set after Phase 2`

## CPU Fallback Layer Detail

| Layer Name | Op Type | Compute Unit | Time (µs) |
|---|---|---|---|
| `custom_gelu` | Gelu | CPU | 4200.0 |
| `fc` | Gemm | CPU | 3100.0 |

---
## Caveats and Limits

- Profiling was performed on a **cloud-hosted** Snapdragon X Elite device via Qualcomm AI Hub. Results may differ on physical hardware.
- Layer-level compute-unit assignment depends on the QNN SDK version used during compilation. Results may differ with different SDK versions.
- Rules are based on observed patterns from experiments in `experiments/`. See `docs/LIMITS.md` for a full list of limitations.

*All latency values come from the downloaded AI Hub profile JSON.*
*No values in this report are estimated or fabricated.*