text = open('experiments/devices.txt', 'rb').read().decode('utf-8-sig')
lines = text.splitlines()
print('Total lines:', len(lines))
print('\\nAll lines:')
for i, line in enumerate(lines[:10]):
    print(f'  Line {i}: {line[:80]}')
print('  ...')
for i in range(max(0, len(lines)-5), len(lines)):
    print(f'  Line {i}: {lines[i][:80]}')