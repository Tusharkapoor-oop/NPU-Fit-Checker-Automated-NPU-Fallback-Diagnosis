import qai_hub as hub

# Get public method names for ProfileJob, CompileJob, Device, Client
profile_methods = [m for m in dir(hub.ProfileJob) if not m.startswith('_')]
compile_methods = [m for m in dir(hub.CompileJob) if not m.startswith('_')]
device_methods = [m for m in dir(hub.Device) if not m.startswith('_')]
client_methods = [m for m in dir(hub.Client) if not m.startswith('_')]

output = {
    "ProfileJob": sorted(profile_methods),
    "CompileJob": sorted(compile_methods),
    "Device": sorted(device_methods),
    "Client": sorted(client_methods)
}

import json
with open('experiments/api_methods.txt', 'w') as f:
    json.dump(output, f, indent=2)

print("Saved experiments/api_methods.txt")
print(f"ProfileJob methods ({len(profile_methods)}): {profile_methods[:20]}...")
print(f"CompileJob methods ({len(compile_methods)}): {compile_methods[:20]}...")
print(f"Device methods ({len(device_methods)}): {device_methods[:20]}...")
print(f"Client methods ({len(client_methods)}): {client_methods[:20]}...")