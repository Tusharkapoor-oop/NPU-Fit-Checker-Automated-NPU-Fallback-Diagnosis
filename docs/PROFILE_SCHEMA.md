Schema description of: experiments/profile_mobilenet_x2.json
=============================================================

Top-level keys (2):
  execution_summary (dict): Summary metrics for the overall inference.
  execution_detail (list[dict]): Per-layer/per-op profiling data (104 items for MobileNetV2).

execution_summary:
  Key: estimated_inference_time
    Type: int
    Unit: microseconds (µs)  [verified from QAI Hub Python SDK docs]
    Description: Estimated total inference time across all runs.
    Example value: 281  (i.e. 281 µs = 0.281 ms)

  Key: estimated_inference_peak_memory
    Type: int
    Unit: bytes
    Description: Peak memory observed during inference.
    Example value: 32792576

  Key: first_load_time
    Type: int
    Unit: microseconds (µs)
    Description: Time for first load (model + initialization).
    Example value: 2495537  (i.e. ~2.50 seconds)

  Key: first_load_peak_memory
    Type: int
    Unit: bytes
    Description: Peak memory during first load.
    Example value: 358952960

  Key: warm_load_time
    Type: int
    Unit: microseconds (µs)
    Description: Time for warmed (cached) load.
    Example value: 331363  (i.e. ~331 ms)

  Key: warm_load_peak_memory
    Type: int
    Unit: bytes
    Description: Peak memory during warm load.
    Example value: 31326208

  Key: compile_time
    Type: int
    Unit: microseconds (µs)
    Description: Compilation time.
    Example value: 0

  Key: compile_peak_memory
    Type: int
    Unit: bytes
    Description: Peak memory during compilation.
    Example value: 0

  Key: compile_memory_increase_range
    Type: list[int, int]
    Description: Minimum and maximum memory increase during compilation.
    Example value: [0, 0]

  Key: compile_memory_peak_range
    Type: list[int, int]
    Description: Minimum and maximum peak memory during compilation.
    Example value: [0, 0]

  Key: first_load_memory_increase_range
    Type: list[int, int]
    Description: Minimum and maximum memory increase during first load.
    Example value: [290738176, 290738176]

  Key: first_load_memory_peak_range
    Type: list[int, int]
    Description: Minimum and maximum peak memory increase during first load.
    Example value: [353968128, 353968128]

  Key: warm_load_memory_increase_range
    Type: list[int, int]
    Description: Minimum and maximum memory increase during warm load.
    Example value: [26279936, 26279936]

  Key: warm_load_memory_peak_range
    Type: list[int, int]
    Description: Minimum and maximum peak memory increase during warm load.
    Example value: [26341376, 26341376]

  Key: inference_memory_increase_range
    Type: list[int, int]
    Description: Minimum and maximum memory increase during inference.
    Example value: [1708032, 1708032]

  Key: inference_memory_peak_range
    Type: list[int, int]
    Description: Minimum and maximum peak memory increase during inference.
    Example value: [1708032, 1708032]

  Key: all_compile_times
    Type: list[int]
    Description: List of compile times across multiple runs.
    Example value: []

  Key: all_first_load_times
    Type: list[int]
    Description: List of first-load times across multiple runs.
    Example value: [2495537]

  Key: all_warm_load_times
    Type: list[int]
    Description: List of warm-load times across multiple runs.
    Example value: [331363]

  Key: all_inference_times
    Type: list[int]
    Description: List of inference times (µs) across runs.
    Example value: [1995, 350, 321, 281, ...] (100 values)

execution_detail:
  Per-layer (or per-op-graph-node) information (104 items). Each entry has:
    - name (str): Layer/op name, e.g. "Input", "Transpose", "/features/features.0/features.0.0/Conv"
    - type (str): Operation type, e.g. "Input", "Conv", "Clip", "Add", "GlobalAveragePool", "Gemm", "Output"
    - compute_unit (str): Compute unit running the op, e.g. "NPU", "CPU", "GPU"
    - execution_time (int): Time executed on the compute unit in microseconds (µs)
    - execution_cycles (int): Number of CPU/NPU cycles consumed

  Example entry:
  {
    "name": "/features/features.0/features.0.0/Conv",
    "type": "Conv",
    "compute_unit": "NPU",
    "execution_time": 0,
    "execution_cycles": 0
  }

Key observations:
  - All 104 per-layer entries in this profile have compute_unit="NPU", meaning the entire MobileNetV2 model ran on the NPU on Snapdragon X2 Elite.
  - No layers fell back to CPU or GPU in this baseline run.
  - The total latency is captured in execution_summary.estimated_inference_time (µs).
  - execution_cycles provides hardware execution cycles.

Schema usage for fit-checker:
  - Total NPU fraction: 100% (104 / 104 execution_detail entries have compute_unit="NPU")
  - Total CPU fraction: 0% (0 entries with compute_unit="CPU")
  - Fallback layers: none (0 entries with compute_unit != "NPU")
  - Total estimated latency: execution_summary.estimated_inference_time (µs) = 281 µs (0.281 ms)