import json
data = open('experiments/demo_profile.json', 'r', encoding='utf-8-sig').read()
profile = json.loads(data)
print('Layers:')
for i, layer in enumerate(profile.get('layers', [])):
    if i >= 5:
        print(f'  ... ({i+1} total layers)')
        break
    print(f'  {layer.get("name")}: {layer.get("type")} on {layer.get("compute_unit")}, time={layer.get("execution_time")}')

print()
print('Top-level inference_time:', profile.get('inference_time'))
print('Device:', profile.get('device'))
print('Model:', profile.get('model_name'))
print('Job ID:', profile.get('job_id'))