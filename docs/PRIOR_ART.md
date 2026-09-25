# Prior Art & Novelty Analysis

This document analyzes existing tools and frameworks that intersect with NPU Fit Checker.
The core claimed addition of this project is the **explain-and-suggest-fix diagnostic layer**
built on top of raw Qualcomm AI Hub profiling data. We make no novelty claims regarding
hardware profiling or compiler execution itself.

> **Investigation Date**: 2026-09-21  
> **Search Query 1**: `onnx qnn fallback diagnose site:github.com`  
> **Search Query 2**: `qualcomm npu fallback tool`  
> **Search Query 3**: `qnn op support checker python`  

---

## 1. Existing Tools & Overlap Analysis

| Existing Tool / Resource | What It Does | Official Source | What NPU Fit Checker Adds |
|---|---|---|---|
| **Qualcomm AI Hub Workbench Profiling** | Submits models to cloud Snapdragon hardware; returns dense raw JSON metrics (`execution_detail`, `execution_summary`). | [workbench.aihub.qualcomm.com](https://workbench.aihub.qualcomm.com) | Automatically parses the JSON, validates schema, attributes compute units per layer, and outputs human-readable diagnosis + fixes. |
| **Windows ML Inspect** | Analyzes operator support for the Windows ML runtime (CPU vs GPU/NPU). | [Microsoft Docs](https://learn.microsoft.com/en-us/windows/ai/windows-ml/inspect-tool) | Tailored specifically for Qualcomm Snapdragon Hexagon NPU / QNN EP; diagnoses boundary QDQ adapter overhead. |
| **ONNX Runtime QNN EP Op Support Table** | Static documentation listing supported ONNX operators for QNN execution providers. | [ONNX Runtime Docs](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) | Dynamic per-model profiling; measures actual execution time and hardware cycles instead of theoretical operator tables. |
| **Windows 11 Task Manager NPU Meter** | Visualizes aggregate per-process NPU utilization. | Windows 11 Built-in | Layer-by-layer compute unit attribution, not just aggregate engine utilization. |

---

## 2. GitHub Search Findings

Search queries returned **no automated CLI tools** that parse Qualcomm AI Hub JSON output to provide fallback diagnosis and optimization guidance:
- `onnx qnn fallback diagnose`: 0 packaged repositories.
- `qualcomm ai hub profile parser cli`: 0 packaged tools.
- `qnn op support checker python`: Scattered forum discussions and snippet scripts, but no standalone developer tool.

---

## 3. Scope Boundaries & What Is NOT Claimed

To preserve strict engineering honesty:
1. **No Automatic Graph Rewriting**: NPU Fit Checker does not automatically mutate ONNX graphs or retrain weights; it provides diagnostic findings and instructions.
2. **Cloud Profiling Only**: All profiling is executed on Qualcomm AI Hub cloud devices (`Snapdragon X2 Elite CRD`, `Snapdragon X Elite CRD`). No physical Snapdragon hardware was tested locally.
3. **No Claim on Profiling Engine**: Qualcomm AI Hub performs the profiling. NPU Fit Checker is the diagnostic and translation intelligence layer.
