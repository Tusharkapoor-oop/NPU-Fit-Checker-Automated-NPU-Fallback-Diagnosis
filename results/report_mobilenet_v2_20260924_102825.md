# NPU Fit Checker - mobilenet_v2
*Generated: 2026-09-24 10:28:25*

> **Device**: Profiled on Qualcomm AI Hub cloud-hosted Snapdragon device: Snapdragon X2 Elite CRD  
> **Model**: mobilenet_v2  
> **Job ID**: UNVERIFIED  

## Latency

| Metric | Value |
|--------|-------|
| Total inference time | 0.273 ms (273 µs) |

## Compute-Unit Breakdown

| Compute Unit | Layer Count | Fraction |
|---|---|---|
| NPU | 104 | 100.0% |
| CPU | 0 | 0.0% |
| GPU | 0 | 0.0% |
| UNASSIGNED | 0 | 0.0% |
| **Total** | 104 | 100.0% |

## Findings

### Finding 1: [INFO] full_npu_execution

**Affected layers** (1): `(all layers)`

**What happened**: Every operator in the model was placed on the Qualcomm Hexagon NPU. Zero CPU fallback was observed.

**Fix**: No action needed. Model fits the NPU completely.

**Evidence**: `experiments/baseline_fp32/profile.json`

---
## Caveats and Limits

- Profiled on Qualcomm AI Hub cloud-hosted Snapdragon device: Snapdragon X2 Elite CRD. Results may differ on physical hardware.
- Layer-level compute-unit assignment depends on the QNN SDK version used during compilation. Results may differ with different SDK versions.
- Rules are based on observed patterns from experiments in `experiments/`. See `docs/LIMITS.md` for limitations.

*All latency values come from the downloaded AI Hub profile JSON.*