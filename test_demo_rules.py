"""Test rules against demo profile that has CPU fallbacks."""
from fitchecker.profile_parser import load_profile
from fitchecker.rules import full_npu_execution, no_fallback_layers

result = load_profile('experiments/demo_profile.json')
print('=== Demo Profile ===')
print(f'Total layers: {result.total_layer_count}')
print(f'NPU layers: {result.npu_layer_count}')
print(f'CPU layers: {result.cpu_layer_count}')
print(f'GPU layers: {result.gpu_layer_count}')
print(f'Fallback layers: {len(result.fallback_layers)}')
print(f'Total time ms: {result.total_inference_time_ms}')
print()

print('=== full_npu_execution ===')
r = full_npu_execution(result)
print(f'Detected: {r["detected"]}')
print(f'Explanation: {r["explanation"]}')
print()

print('=== no_fallback_layers ===')
r = no_fallback_layers(result)
print(f'Detected: {r["detected"]}')
print(f'Explanation: {r["explanation"]}')