"""
scripts/phase0_setup.py

Phase 0 automation:
  1. Python 3.13 / Windows x64 check.
  2. pip install requirements.txt.
  3. Import qai_hub, print version and top-level public names.
  4. Enumerate devices.  Filter: attributes must contain BOTH
       "os:windows"     (meaning Windows OS)
       "format:compute" (meaning PC-class Compute device)
     Save name, os, attributes as JSON lines -> experiments/devices.txt.
     GATE: exactly 3 entries with exact names matching EXPECTED_DEVICES.
  5. Dump public names of ProfileJob/CompileJob/QuantizeJob/Device/Client
     -> experiments/api_methods.txt.
     Rule: hub_client.py may ONLY call methods listed in that file.

Run from repo root:
  python scripts/phase0_setup.py

Configure token first (never pass it here):
  qai-hub configure --api_token YOUR_TOKEN
"""

import json
import pathlib
import subprocess
import sys

# ---------------------------------------------------------------------------
# Known PC-class Windows Compute devices (exact names, verified 2025-09-21)
# ---------------------------------------------------------------------------
EXPECTED_DEVICES: list[str] = [
    "Snapdragon X Elite CRD",
    "Snapdragon X Plus 8-Core CRD",
    "Snapdragon X2 Elite CRD",
]
PRIMARY_DEVICE   = "Snapdragon X2 Elite CRD"
SECONDARY_DEVICE = "Snapdragon X Elite CRD"

# Attribute tags that must both be present on a hub.Device for it to qualify
_ATTR_OS      = "os:windows"
_ATTR_COMPUTE = "format:compute"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result


def _is_pc_class(device) -> bool:
    """
    True only when device.attributes contains BOTH 'os:windows' and
    'format:compute'.  Verified against live hub.get_devices() on
    2025-09-21: this selects exactly 3 devices (X Elite CRD,
    X Plus 8-Core CRD, X2 Elite CRD) out of 78.
    Note: device.os holds a numeric string (e.g. '11') — not 'Windows' —
    so we must use device.attributes, not device.os, for this filter.
    """
    attrs = getattr(device, "attributes", []) or []
    return _ATTR_OS in attrs and _ATTR_COMPUTE in attrs


def _public_names(obj) -> list[str]:
    return sorted(n for n in dir(obj) if not n.startswith("_"))


def main() -> None:
    sep = "=" * 60
    print(sep)
    print("NPU Fit Checker - Phase 0 Setup")
    print(sep)

    # 1. Python version
    print("\n[1/5] Python version ...")
    v = sys.version_info
    print(f"  Python {v.major}.{v.minor}.{v.micro} ({sys.platform})")
    if v.major == 3 and v.minor == 13:
        print("  OK Python 3.13 (expected)")
    elif v.major == 3 and v.minor >= 10:
        print(f"  OK Python {v.major}.{v.minor}")
    else:
        print(f"  WARNING: need 3.10+, got {v.major}.{v.minor}")

    # 2. Install requirements
    print("\n[2/5] pip install requirements ...")
    r = run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    if r.returncode != 0:
        print("[ERROR] pip install failed")
        sys.exit(1)
    print("  OK requirements installed")

    # 3. Import qai_hub
    print("\n[3/5] Importing qai_hub ...")
    try:
        import qai_hub as hub
        ver = getattr(hub, "__version__", "?")
        print(f"  OK qai_hub version: {ver}")
        print(f"  Top-level public names: {_public_names(hub)}")
    except ImportError as e:
        print(f"  [ERROR] Cannot import qai_hub: {e}")
        print("  Run: pip install qai-hub")
        sys.exit(1)

    # 4. Filter devices
    print("\n[4/5] Listing and filtering AI Hub devices ...")
    print(f"  Filter: '{_ATTR_OS}' AND '{_ATTR_COMPUTE}' in device.attributes")
    print("  Note: device.os is a numeric string (e.g. '11'), not 'Windows'.")
    print("        The OS is identified via the 'os:windows' attribute tag.")
    try:
        all_devices = hub.get_devices()
    except Exception as e:
        print(f"  [ERROR] hub.get_devices() failed: {e}")
        print("  Run: qai-hub configure --api_token YOUR_TOKEN")
        sys.exit(1)

    print(f"  Total hub devices: {len(all_devices)}")
    print("  First 5 raw entries:")
    for d in all_devices[:5]:
        print(f"    name={d.name!r}  os={d.os!r}  "
              f"attributes={getattr(d, 'attributes', [])}")

    pc = [d for d in all_devices if _is_pc_class(d)]
    print(f"\n  PC-class kept: {len(pc)} (expected 3)")
    for d in pc:
        tag = ""
        if d.name == PRIMARY_DEVICE:
            tag = "  <- PRIMARY"
        elif d.name == SECONDARY_DEVICE:
            tag = "  <- SECONDARY"
        print(f"    {d.name!r}  attributes={getattr(d,'attributes',[])}{tag}")

    # Gate: count
    if len(pc) != 3:
        print(f"\n[GATE FAILED] Expected exactly 3, found {len(pc)}.")
        sys.exit(1)

    # Gate: exact names
    found = {d.name for d in pc}
    for exp in EXPECTED_DEVICES:
        if exp not in found:
            print(f"[GATE FAILED] Expected device missing: {exp!r}")
            print(f"  Found: {sorted(found)}")
            sys.exit(1)

    # Save JSON lines: name, os, attributes
    out = pathlib.Path("experiments") / "devices.txt"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for d in pc:
            entry = {
                "name":       d.name,
                "os":         d.os,
                "attributes": getattr(d, "attributes", []),
            }
            fh.write(json.dumps(entry) + "\n")
    print(f"\n  OK saved {len(pc)} entries to {out}  (JSON lines)")

    # 5. Dump API method names
    print("\n[5/5] Dumping public API method names ...")
    api_out = pathlib.Path("experiments") / "api_methods.txt"
    sections: list[str] = []
    for cname in ("ProfileJob", "CompileJob", "QuantizeJob", "Device", "Client"):
        cls = getattr(hub, cname, None)
        if cls is None:
            sections.append(f"# hub.{cname}: NOT FOUND in this SDK version\n")
            print(f"  WARNING: hub.{cname} not found")
            continue
        names = _public_names(cls)
        sections.append(f"# hub.{cname}\n" + "\n".join(names) + "\n")
        print(f"  hub.{cname}: {names}")
    api_out.write_text("\n".join(sections), encoding="utf-8")
    print(f"  OK api_methods.txt saved to {api_out}")

    print(f"\n{sep}")
    print("Phase 0 PASSED")
    print(f"  PRIMARY   : {PRIMARY_DEVICE}")
    print(f"  SECONDARY : {SECONDARY_DEVICE}")
    print(f"  devices.txt     -> {out}")
    print(f"  api_methods.txt -> {api_out}")
    print("Next: python scripts/phase1_mobilenet.py")
    print(sep)


if __name__ == "__main__":
    main()
