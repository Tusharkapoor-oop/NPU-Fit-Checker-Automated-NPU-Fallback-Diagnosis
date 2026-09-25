"""Test the rules engine against the real MobileNetV2 profile."""
from fitchecker.profile_parser import load_profile
from fitchecker.rules import full_npu_execution, no_fallback_layers, high_latency_threshold

result = load_profile('experiments/baseline_fp32/profile.json')
print("=== Baseline Profile ===")
print(f"Total layers: {result.total_layer_count}")
print(f"NPU layers: {result.npu_layer_count}")
print(f"CPU layers: {result.cpu_layer_count}")
print(f"GPU layers: {result.gpu_layer_count}")
print(f"Fallback layers: {len(result.fallback_layers)}")
print(f"Total time ms: {result.total_inference_time_ms}")
print()

print("=== full_npu_execution rule ===")
r = full_npu_execution(result)
print(f"Detected: {r['detected']}")
print(f"Explanation: {r['explanation']}")
print(f"Evidence: {r['evidence']}")
print()

print("=== no_fallback_layers rule ===")
r = no_fallback_layers(result)
print(f"Detected: {r['detected']}")
print(f"Explanation: {r['explanation']}")
print(f"Evidence: {r['evidence']}")
print()

print("=== high_latency_threshold rule (200ms) ===")
r = high_latency_threshold(result, threshold_ms=200.0)
print(f"Detected: {r['detected']}")
print(f"Explanation: {r['explanation']}")
print(f"Evidence: {r['evidence']}")