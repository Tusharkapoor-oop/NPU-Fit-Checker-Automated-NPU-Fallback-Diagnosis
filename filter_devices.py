data = open('experiments/devices_raw.txt', 'rb').read()
text = data.decode('utf-16-le')
lines = text.splitlines()

# Keep only lines where OS starts with "Windows" and type is "Compute"
kept = []
for line in lines:
    stripped = line.strip()
    if stripped and '|' in stripped:
        parts = [p.strip() for p in stripped.split('|')]
        # parts[0]=empty, parts[1]=device, parts[2]=OS, parts[3]=Vendor, parts[4]=Type
        if len(parts) >= 5:
            os_val = parts[2]  # OS is third column (index 2)
            type_val = parts[4]  # Type is fifth column (index 4)
            if os_val.startswith('Windows') and type_val == 'Compute':
                kept.append(line)

print(f"Kept {len(kept)} devices:")
for line in kept:
    print(line)

# Now create the new devices.txt file
header_lines = []
footer_lines = []
in_header = True
in_devices = False
in_footer = False

for line in lines:
    stripped = line.strip()
    if stripped == '+---------------------------------+--------------+----------+---------+-----------------------------------------------------------+------------------------------------------------------------+':
        in_header = False
        in_devices = True
    elif stripped.startswith('Don'):
        in_devices = False
        in_footer = True

    if in_header:
        header_lines.append(line)
    elif in_devices and line.strip() in kept:
        pass  # will add manually
    elif in_devices:
        pass  # skip
    elif in_footer:
        footer_lines.append(line)
    else:
        pass

# Rebuild: header + kept devices + footer
new_text = '\n'.join(header_lines) + '\n'
for line in kept:
    new_text += line + '\n'
new_text += '\n'.join(footer_lines)

open('experiments/devices.txt', 'w', encoding='utf-8').write(new_text)
print(f"\nWrote experiments/devices.txt with {len(kept)} devices")