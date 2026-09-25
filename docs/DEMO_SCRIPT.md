# Video Demonstration Script: NPU Fit Checker

**Duration**: ~2 minutes 30 seconds  
**Target Audience**: Judges, ML Systems Engineers, Snapdragon AI Challenge Reviewers  
**Core Message**: "Turn dense Qualcomm AI Hub profiling JSON into instant, actionable NPU diagnostic intelligence."

---

## [0:00 - 0:30] Introduction & The Problem

**Visual**: Screen recording showing a raw 30,000-byte AI Hub `profile.json` in VS Code. Dense nested JSON, thousands of lines.

**Voiceover**:
> "Qualcomm's Snapdragon X Elite and X2 Elite NPUs offer groundbreaking on-device AI performance. But when you compile and profile an AI model on Qualcomm AI Hub, what you get back is a dense, multi-thousand-line JSON payload. 
> 
> Did your Conv layers actually run on the Hexagon NPU? Did an unsupported operator silently fall back to the CPU? Where is latency being lost? Finding answers requires digging through raw op logs.
> 
> That's why we built **NPU Fit Checker**."

---

## [0:30 - 1:15] Demo 1: The Baseline FP32 Run

**Visual**: Open terminal. Run:
```bash
python -m fitchecker run --from-profile experiments/baseline_fp32/profile.json --device "Snapdragon X2 Elite CRD"
```

**Voiceover**:
> "Let's run NPU Fit Checker on a MobileNetV2 FP32 baseline profiled directly on a cloud-hosted Snapdragon X2 Elite CRD.
> 
> In less than a second, NPU Fit Checker parses the profile, validates the schema, and renders a clean terminal report.
> 
> Look at the compute breakdown: 104 out of 104 layers ran 100% on the Hexagon NPU with 0 CPU fallbacks. Total latency is 281 microseconds. The tool triggers our verified `full_npu_execution` rule, confirming complete NPU mapping."

---

## [1:15 - 2:00] Demo 2: The INT8 Optimization & Boundary Fallback Diagnosis

**Visual**: Run:
```bash
python -m fitchecker run \
  --from-profile experiments/optimized_int8/profile.json \
  --device "Snapdragon X2 Elite CRD" \
  --baseline experiments/baseline_fp32/profile.json \
  --baseline-label "FP32 Baseline" \
  --optimized-label "INT8 Optimized"
```

**Voiceover**:
> "Now let's compare that against the INT8 quantized model compiled with the QNN execution provider.
> 
> First, the good news: latency drops from 281 µs down to 154 µs — a **1.82× speedup**!
> 
> But look at the warning: `cpu_fallback_detected`. NPU Fit Checker pinpointed exactly two operators that fell back to CPU: `QcQuantizeOp_input_q` and `QcQuantizeOp_output_dq`.
> 
> Why did this happen? Because the host runtime supplies FP32 inputs at the model boundary, forcing CPU adapter conversions. NPU Fit Checker doesn't just show the issue — it provides the exact fix: compile with native INT8 tensor IO types or handle quantization in host preprocessing."

---

## [2:00 - 2:30] Architecture, Strict Honesty & Conclusion

**Visual**: Show `docs/EVIDENCE_LOG.md` and run `pytest tests/ -v` (all 30 tests passing).

**Voiceover**:
> "Every single number reported by NPU Fit Checker traces to a real file in our repository or an active Qualcomm AI Hub job URL. 
> 
> We test against 30 automated test cases, enforce strict schema validation, and filter only genuine PC-class Snapdragon compute devices.
> 
> NPU Fit Checker gives Windows AI developers the clarity they need to maximize throughput on Snapdragon X Elite and X2 Elite. Thank you."
