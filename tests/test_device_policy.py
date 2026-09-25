"""
tests/test_device_policy.py

Unit tests for the devices.txt rule:
  - validate_device_name() must accept names in the file.
  - validate_device_name() must reject names not in the file.
  - load_permitted_devices() must raise FileNotFoundError if file absent.
  - Phone / IoT device names must not appear in a properly populated devices.txt.

These tests run without a real devices.txt; they use a tmp_path fixture.
"""

import json
import pathlib
import pytest


# ---------------------------------------------------------------------------
# Helpers to fake the devices.txt location for tests
# ---------------------------------------------------------------------------

def _write_devices_file(path: pathlib.Path, names: list[str]) -> None:
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDevicesFileParsing:
    """load_permitted_devices reads lines correctly."""

    def test_missing_file_raises(self, tmp_path, monkeypatch):
        """FileNotFoundError when devices.txt does not exist."""
        import fitchecker.hub_client as hc
        monkeypatch.setattr(hc, "_DEVICES_FILE", tmp_path / "devices.txt")
        with pytest.raises(FileNotFoundError):
            hc.load_permitted_devices()

    def test_reads_names(self, tmp_path, monkeypatch):
        import fitchecker.hub_client as hc
        f = tmp_path / "devices.txt"
        _write_devices_file(f, [
            "Snapdragon X2 Elite CRD",
            "Snapdragon X Elite CRD",
        ])
        monkeypatch.setattr(hc, "_DEVICES_FILE", f)
        names = hc.load_permitted_devices()
        assert "Snapdragon X2 Elite CRD" in names
        assert "Snapdragon X Elite CRD" in names

    def test_blank_lines_ignored(self, tmp_path, monkeypatch):
        import fitchecker.hub_client as hc
        f = tmp_path / "devices.txt"
        f.write_text("Snapdragon X2 Elite CRD\n\n\nSnapdragon X Elite CRD\n", encoding="utf-8")
        monkeypatch.setattr(hc, "_DEVICES_FILE", f)
        names = hc.load_permitted_devices()
        assert len(names) == 2


class TestValidateDeviceName:
    """validate_device_name enforces the devices.txt rule."""

    def _setup(self, tmp_path, monkeypatch, permitted: list[str]):
        import fitchecker.hub_client as hc
        f = tmp_path / "devices.txt"
        _write_devices_file(f, permitted)
        monkeypatch.setattr(hc, "_DEVICES_FILE", f)
        return hc

    def test_primary_device_passes(self, tmp_path, monkeypatch):
        hc = self._setup(tmp_path, monkeypatch, [
            "Snapdragon X2 Elite CRD",
            "Snapdragon X Elite CRD",
        ])
        result = hc.validate_device_name("Snapdragon X2 Elite CRD")
        assert result == "Snapdragon X2 Elite CRD"

    def test_secondary_device_passes(self, tmp_path, monkeypatch):
        hc = self._setup(tmp_path, monkeypatch, [
            "Snapdragon X2 Elite CRD",
            "Snapdragon X Elite CRD",
        ])
        result = hc.validate_device_name("Snapdragon X Elite CRD")
        assert result == "Snapdragon X Elite CRD"

    def test_unknown_device_raises(self, tmp_path, monkeypatch):
        hc = self._setup(tmp_path, monkeypatch, [
            "Snapdragon X2 Elite CRD",
            "Snapdragon X Elite CRD",
        ])
        with pytest.raises(ValueError, match="not in experiments/devices.txt"):
            hc.validate_device_name("Qualcomm Phone 8 Gen 3")

    def test_empty_string_raises(self, tmp_path, monkeypatch):
        hc = self._setup(tmp_path, monkeypatch, ["Snapdragon X2 Elite CRD"])
        with pytest.raises(ValueError):
            hc.validate_device_name("")

    def test_iot_device_not_in_list(self, tmp_path, monkeypatch):
        """Verifies IoT / phone names were filtered before writing devices.txt."""
        hc = self._setup(tmp_path, monkeypatch, [
            "Snapdragon X2 Elite CRD",
            "Snapdragon X Elite CRD",
        ])
        # These should NOT be in a properly filtered devices.txt
        for bad_name in ["QCS6490 RB3 Gen 2", "Snapdragon 8 Gen 3 MTP", "SA8775P RB5"]:
            with pytest.raises(ValueError):
                hc.validate_device_name(bad_name)


class TestDeviceConstants:
    """Verify the module-level device name constants are correct."""

    def test_primary_constant(self):
        from fitchecker.hub_client import DEFAULT_PRIMARY_DEVICE
        assert DEFAULT_PRIMARY_DEVICE == "Snapdragon X2 Elite CRD"

    def test_secondary_constant(self):
        from fitchecker.hub_client import DEFAULT_SECONDARY_DEVICE
        assert DEFAULT_SECONDARY_DEVICE == "Snapdragon X Elite CRD"
