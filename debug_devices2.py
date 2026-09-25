"""
debug_devices2.py

Debug script: dumps all device lines from experiments/devices_raw.txt (UTF-16-LE).
Run from the project root:
    python debug_devices2.py
"""

import pathlib

if __name__ == "__main__":
    raw_path = pathlib.Path(__file__).parent / "experiments" / "devices_raw.txt"

    with open(raw_path, "rb") as fh:
        data = fh.read()

    text = data.decode("utf-16-le")
    lines = text.splitlines()

    # Print all device info for debugging
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped:
            try:
                print(f"Line {i}: {stripped[:200]}")
            except Exception as exc:
                print(f"Line {i}: <unprintable, len={len(stripped)}, error={exc}>")