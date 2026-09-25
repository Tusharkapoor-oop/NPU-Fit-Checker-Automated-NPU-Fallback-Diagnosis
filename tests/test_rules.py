"""
tests/test_rules.py

Unit tests for fitchecker/rules.py.
Validates that:
  - full_npu_execution fires on baseline FP32 (104 NPU layers) and stays quiet on optimized INT8.
  - cpu_fallback_detected fires on optimized INT8 (2 QDQ layers on CPU) and stays quiet on baseline FP32.
  - Unverified schema outputs 'cannot analyse profile' and returns [].
  - Every finding points to a real evidence file that exists on disk.
"""

import pathlib
import pytest

from fitchecker.profile_parser import load_profile, ProfileResult
from fitchecker.rules import apply_rules, RuleFinding

BASELINE_PATH = pathlib.Path("experiments/baseline_fp32/profile.json")
OPTIMIZED_PATH = pathlib.Path("experiments/optimized_int8/profile.json")


class TestRulesOnRealFiles:
    def test_baseline_triggers_full_npu_only(self):
        profile = load_profile(str(BASELINE_PATH))
        findings = apply_rules(profile)
        names = [f.rule_name for f in findings]

        assert "full_npu_execution" in names
        assert "cpu_fallback_detected" not in names
        assert "no_latency_data" not in names

    def test_optimized_triggers_cpu_fallback(self):
        profile = load_profile(str(OPTIMIZED_PATH))
        findings = apply_rules(profile)
        names = [f.rule_name for f in findings]

        assert "cpu_fallback_detected" in names
        assert "full_npu_execution" not in names

        # Check affected layers
        finding = next(f for f in findings if f.rule_name == "cpu_fallback_detected")
        affected_names = [l.name for l in finding.affected_layers]
        assert "QcQuantizeOp_input_q" in affected_names
        assert "QcQuantizeOp_output_dq" in affected_names
        assert len(affected_names) == 2


class TestSchemaGuard:
    def test_unverified_schema_blocks_rules(self, capsys):
        profile = load_profile(str(BASELINE_PATH))
        profile.schema_verified = False

        findings = apply_rules(profile)
        assert findings == []

        captured = capsys.readouterr()
        assert "cannot analyse profile" in captured.out


class TestFindingStructure:
    def test_all_findings_have_valid_fields_and_real_evidence(self):
        for path in (BASELINE_PATH, OPTIMIZED_PATH):
            profile = load_profile(str(path))
            findings = apply_rules(profile)
            for f in findings:
                assert f.rule_name
                assert f.explanation
                assert f.fix
                assert f.evidence_file
                assert pathlib.Path(f.evidence_file).exists(), f"Evidence file missing: {f.evidence_file}"
                assert f.severity in ("CRITICAL", "WARNING", "INFO")
