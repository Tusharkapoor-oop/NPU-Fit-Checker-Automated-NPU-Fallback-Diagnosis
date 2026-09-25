"""
tests/test_profile_parser.py

Unit tests for fitchecker/profile_parser.py based on confirmed schema.
Tests run against:
  1. Synthetic dicts conforming to the confirmed schema.
  2. The real profile JSON from experiments/profile_mobilenet_x2.json.
  3. SchemaError validation on invalid/missing keys.
"""

import json
import pathlib
import pytest

from fitchecker.profile_parser import (
    parse_profile,
    load_profile,
    describe_schema,
    ProfileResult,
    SchemaError,
)

REAL_PROFILE_PATH = pathlib.Path("experiments/profile_mobilenet_x2.json")


def _synthetic_profile() -> dict:
    """A minimal synthetic profile adhering strictly to the confirmed schema."""
    return {
        "execution_summary": {
            "estimated_inference_time": 4500,  # 4500 µs = 4.5 ms
            "estimated_inference_peak_memory": 1048576,
        },
        "execution_detail": [
            {
                "name": "conv1",
                "type": "Conv",
                "compute_unit": "NPU",
                "execution_time": 120,
                "execution_cycles": 1000,
            },
            {
                "name": "depthwise_conv2",
                "type": "Conv",
                "compute_unit": "NPU",
                "execution_time": 85,
                "execution_cycles": 800,
            },
            {
                "name": "custom_gelu",
                "type": "Gelu",
                "compute_unit": "CPU",
                "execution_time": 800,
                "execution_cycles": 5000,
            },
        ],
    }


def _all_npu_profile() -> dict:
    p = _synthetic_profile()
    for layer in p["execution_detail"]:
        layer["compute_unit"] = "NPU"
    return p


# ---------------------------------------------------------------------------
# Tests: parse_profile
# ---------------------------------------------------------------------------

class TestParseProfile:
    def test_basic_parsing(self):
        result = parse_profile(
            _synthetic_profile(),
            job_id="test-job-001",
            device_name="Snapdragon X2 Elite CRD",
            model_name="MobileNetV2",
        )
        assert isinstance(result, ProfileResult)
        assert result.job_id == "test-job-001"
        assert result.device_name == "Snapdragon X2 Elite CRD"
        assert result.model_name == "MobileNetV2"
        assert result.schema_verified is True

    def test_latency_parsed(self):
        result = parse_profile(_synthetic_profile())
        assert result.total_inference_time_us == 4500.0
        assert abs(result.total_inference_time_ms - 4.5) < 0.001

    def test_layer_counts(self):
        result = parse_profile(_synthetic_profile())
        assert result.total_layer_count == 3
        assert result.npu_layer_count == 2
        assert result.cpu_layer_count == 1
        assert result.gpu_layer_count == 0
        assert result.unassigned_layer_count == 0

    def test_compute_unit_case_insensitive(self):
        profile = _synthetic_profile()
        profile["execution_detail"][0]["compute_unit"] = "npu"
        profile["execution_detail"][1]["compute_unit"] = "Npu"
        result = parse_profile(profile)
        assert result.npu_layer_count == 2

    def test_fallback_layers(self):
        result = parse_profile(_synthetic_profile())
        fallbacks = result.fallback_layers
        assert len(fallbacks) == 1
        assert fallbacks[0].name == "custom_gelu"
        assert fallbacks[0].compute_unit == "CPU"

    def test_npu_fraction(self):
        result = parse_profile(_synthetic_profile())
        assert abs(result.npu_fraction - 2 / 3) < 0.001
        assert abs(result.cpu_fraction - 1 / 3) < 0.001

    def test_all_npu(self):
        result = parse_profile(_all_npu_profile())
        assert result.cpu_layer_count == 0
        assert result.npu_fraction == 1.0
        assert result.fallback_layers == []


# ---------------------------------------------------------------------------
# Tests: SchemaError on invalid schema
# ---------------------------------------------------------------------------

class TestSchemaError:
    def test_missing_execution_summary(self):
        with pytest.raises(SchemaError, match="execution_summary"):
            parse_profile({"execution_detail": []})

    def test_missing_execution_detail(self):
        with pytest.raises(SchemaError, match="execution_detail"):
            parse_profile({"execution_summary": {"estimated_inference_time": 100}})

    def test_missing_estimated_inference_time(self):
        with pytest.raises(SchemaError, match="estimated_inference_time"):
            parse_profile({"execution_summary": {}, "execution_detail": []})

    def test_missing_layer_key(self):
        data = {
            "execution_summary": {"estimated_inference_time": 100},
            "execution_detail": [
                {"name": "conv1", "type": "Conv", "compute_unit": "NPU"}  # missing execution_time/cycles
            ],
        }
        with pytest.raises(SchemaError, match="Missing required key"):
            parse_profile(data)

    def test_empty_compute_unit(self):
        data = {
            "execution_summary": {"estimated_inference_time": 100},
            "execution_detail": [
                {
                    "name": "conv1",
                    "type": "Conv",
                    "compute_unit": "   ",
                    "execution_time": 10,
                    "execution_cycles": 100,
                }
            ],
        }
        with pytest.raises(SchemaError, match="Empty compute_unit"):
            parse_profile(data)


# ---------------------------------------------------------------------------
# Tests: load_profile and describe_schema
# ---------------------------------------------------------------------------

class TestLoadProfile:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_profile("nonexistent/path/profile.json")

    def test_load_temp_json(self, tmp_path):
        p = tmp_path / "profile.json"
        p.write_text(json.dumps(_synthetic_profile()), encoding="utf-8")
        result = load_profile(str(p), job_id="job-123")
        assert result.job_id == "job-123"
        assert result.total_inference_time_us == 4500.0

    def test_describe_temp_file(self, tmp_path):
        p = tmp_path / "profile.json"
        p.write_text(json.dumps(_synthetic_profile()), encoding="utf-8")
        output = describe_schema(str(p))
        assert "execution_summary" in output
        assert "execution_detail" in output
        assert "total_time_us" in output


# ---------------------------------------------------------------------------
# Real Profile Test (Gate validation)
# ---------------------------------------------------------------------------

class TestRealProfile:
    @pytest.mark.skipif(not REAL_PROFILE_PATH.exists(), reason="Real profile not yet downloaded")
    def test_real_mobilenet_profile(self):
        result = load_profile(str(REAL_PROFILE_PATH), device_name="Snapdragon X2 Elite CRD")
        assert result.schema_verified is True
        assert result.total_layer_count == 104
        assert result.npu_layer_count == 104
        assert result.cpu_layer_count == 0
        assert result.npu_fraction == 1.0
        assert result.fallback_layers == []
        assert result.total_inference_time_us == 281.0
        assert abs(result.total_inference_time_ms - 0.281) < 1e-4
