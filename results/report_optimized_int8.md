# NPU Fit Checker - profile
*Generated: 2026-09-23 10:07:33*

> **Device**: Profiled on Qualcomm AI Hub cloud-hosted Snapdragon device: Snapdragon X2 Elite CRD  
> **Model**: MobileNetV2 (INT8 QDQ)  
> **Job ID**: j57e7d0qp  

## Latency

| Metric | Value |
|--------|-------|
| Total inference time | 0.154 ms (154 µs) |
| FP32 Baseline inference time | 0.281 ms (281 µs) |
| Speedup | 1.82x |

## Compute-Unit Breakdown

| Compute Unit | Layer Count | Fraction |
|---|---|---|
| NPU | 139 | 98.6% |
| CPU | 2 | 1.4% |
| GPU | 0 | 0.0% |
| UNASSIGNED | 0 | 0.0% |
| **Total** | 141 | 100.0% |

## Findings

### Finding 1: [WARN] cpu_fallback_detected

**Affected layers** (2): `QcQuantizeOp_input_q`, `QcQuantizeOp_output_dq`

**What happened**: The model contains operations that fell back to CPU execution. In quantized models, this commonly occurs at input/output boundaries where QuantizeLinear and DequantizeLinear adapt FP32 host data to INT8.

**Fix**: To eliminate boundary QDQ fallbacks, configure the model compilation with native INT8 input/output tensor types, or keep quantization adapters on the host preprocessing pipeline.

**Evidence**: `experiments/optimized_int8/profile.json`

## CPU Fallback Layer Detail

| Layer Name | Op Type | Compute Unit | Time (us) |
|---|---|---|---|
| `QcQuantizeOp_input_q` | QuantizeLinear | CPU | 116.0 |
| `QcQuantizeOp_output_dq` | DequantizeLinear | CPU | 53.0 |

## Before / After Comparison

| Metric | FP32 Baseline | INT8 Optimized |
|---|---|---|
| Total inference time | 0.281 ms (281 µs) | 0.154 ms (154 µs) |
| NPU layers | 104 | 139 |
| CPU layers | 0 | 2 |

---
## Caveats and Limits

- Profiled on Qualcomm AI Hub cloud-hosted Snapdragon device: Snapdragon X2 Elite CRD. Results may differ on physical hardware.
- Layer-level compute-unit assignment depends on the QNN SDK version used during compilation. Results may differ with different SDK versions.
- Rules are based on observed patterns from experiments in `experiments/`. See `docs/LIMITS.md` for limitations.

*All latency values come from the downloaded AI Hub profile JSON.*