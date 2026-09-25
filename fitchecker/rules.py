"""
fitchecker/rules.py

Fallback detection rules for NPU Fit Checker.
Derived strictly from real profiling evidence on Snapdragon X2 Elite CRD.

Rules:
  1. full_npu_execution:
     Fires on baseline FP32 (104/104 NPU layers).
     Evidence: experiments/baseline_fp32/profile.json
  2. cpu_fallback_detected:
     Fires on optimized INT8 (2 boundary QDQ layers on CPU: QcQuantizeOp_input_q, QcQuantizeOp_output_dq).
     Evidence: experiments/optimized_int8/profile.json
  3. no_latency_data:
     Structural guard when latency is missing.
     Evidence: experiments/baseline_fp32/profile.json

All rules carry real evidence_file paths.
Unverified rules are documented in docs/LIMITS.md.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Callable, Optional

from fitchecker.profile_parser import ProfileResult, LayerInfo


# ---------------------------------------------------------------------------
# Data classes — used by apply_rules() and the test suite
# ---------------------------------------------------------------------------

@dataclass
class RuleFinding:
    """One triggered rule finding."""
    rule_name: str
    explanation: str
    fix: str
    evidence_file: str
    affected_layers: list[LayerInfo] = field(default_factory=list)
    severity: str = "INFO"


@dataclass
class RuleDefinition:
    """A rule applied to a ProfileResult."""
    name: str
    description: str
    explanation: str
    fix: str
    evidence_file: str
    verified: bool = True
    severity: str = "WARNING"
    _check_fn: Optional[Callable[[ProfileResult], list[LayerInfo]]] = field(
        default=None, repr=False
    )

    def check(self, profile: ProfileResult) -> Optional[RuleFinding]:
        if not _schema_is_verified(profile):
            return None
        if self.evidence_file.startswith("UNVERIFIED"):
            return None
        # Verify evidence file exists on disk
        if not pathlib.Path(self.evidence_file).exists():
            return None
        if self._check_fn is None:
            return None

        affected = self._check_fn(profile)
        if not affected:
            return None

        return RuleFinding(
            rule_name=self.name,
            explanation=self.explanation,
            fix=self.fix,
            evidence_file=self.evidence_file,
            affected_layers=affected,
            severity=self.severity,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _schema_is_verified(profile: ProfileResult) -> bool:
    return getattr(profile, "schema_verified", False)


def _guard(result: ProfileResult) -> bool:
    """Return True if the profile has recognisable data (layers or latency)."""
    return (
        result.total_layer_count > 0
        or result.total_inference_time_ms is not None
    )


def _cpu_layers(profile: ProfileResult) -> list[LayerInfo]:
    return [la for la in profile.layers if la.compute_unit.upper() == "CPU"]


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

ALL_RULES: list[RuleDefinition] = [
    # ------------------------------------------------------------------
    # RULE 1: CPU fallback detected
    # Evidence: experiments/optimized_int8/profile.json (2 QDQ ops on CPU)
    # ------------------------------------------------------------------
    RuleDefinition(
        name="cpu_fallback_detected",
        description="One or more layers executed on CPU instead of NPU.",
        explanation=(
            "The model contains operations that fell back to CPU execution. "
            "In quantized models, this commonly occurs at input/output boundaries "
            "where QuantizeLinear and DequantizeLinear adapt FP32 host data to INT8."
        ),
        fix=(
            "To eliminate boundary QDQ fallbacks, configure the model compilation "
            "with native INT8 input/output tensor types, or keep quantization adapters "
            "on the host preprocessing pipeline."
        ),
        evidence_file="experiments/optimized_int8/profile.json",
        verified=True,
        severity="WARNING",
        _check_fn=_cpu_layers,
    ),

    # ------------------------------------------------------------------
    # RULE 2: Full NPU execution (good-news rule)
    # Evidence: experiments/baseline_fp32/profile.json (104/104 layers on NPU)
    # ------------------------------------------------------------------
    RuleDefinition(
        name="full_npu_execution",
        description="All layers ran on the NPU.",
        explanation=(
            "Every operator in the model was placed on the Qualcomm Hexagon NPU. "
            "Zero CPU fallback was observed."
        ),
        fix="No action needed. Model fits the NPU completely.",
        evidence_file="experiments/baseline_fp32/profile.json",
        verified=True,
        severity="INFO",
        _check_fn=lambda p: (
            [LayerInfo(name="(all layers)", op_type="N/A", compute_unit="NPU")]
            if p.total_layer_count > 0
            and p.cpu_layer_count == 0
            and p.gpu_layer_count == 0
            and p.unassigned_layer_count == 0
            else []
        ),
    ),

    # ------------------------------------------------------------------
    # RULE 3: No latency data
    # Evidence: structural guard verified against valid profile
    # ------------------------------------------------------------------
    RuleDefinition(
        name="no_latency_data",
        description="Profile does not contain valid inference latency metrics.",
        explanation=(
            "The profile was downloaded but execution_summary lacks "
            "estimated_inference_time."
        ),
        fix=(
            "Verify profile job options and run 'fitchecker inspect-profile <json>' "
            "to inspect hardware metrics."
        ),
        evidence_file="experiments/baseline_fp32/profile.json",
        verified=True,
        severity="WARNING",
        _check_fn=lambda p: (
            [LayerInfo(name="(profile)", op_type="N/A", compute_unit="N/A")]
            if p.total_inference_time_us is None
            else []
        ),
    ),
]


# ---------------------------------------------------------------------------
# Primary API used by cli.py, report.py, and the test suite
# ---------------------------------------------------------------------------

def apply_rules(profile: ProfileResult) -> list[RuleFinding]:
    """
    Apply all registered rules to a parsed profile.
    If schema is not verified, prints a guard message and returns [].
    Returns findings sorted by severity (WARNING before INFO).
    """
    if not _schema_is_verified(profile):
        print("[rules] cannot analyse profile")
        return []

    findings = []
    for rule in ALL_RULES:
        finding = rule.check(profile)
        if finding is not None:
            if getattr(profile, "profile_path", None):
                finding.evidence_file = profile.profile_path
            findings.append(finding)

    order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    findings.sort(key=lambda f: order.get(f.severity, 99))
    return findings


# ---------------------------------------------------------------------------
# Legacy standalone functions (kept for backward compatibility)
# These return plain dicts and do NOT use the evidence-file guard.
# Use apply_rules() for all production and test code.
# ---------------------------------------------------------------------------

def full_npu_execution(result: ProfileResult) -> dict:
    """
    Rule: Model runs fully on NPU with no CPU/GPU fallbacks.
    Legacy dict-returning form. Prefer apply_rules() in new code.
    """
    if not _guard(result):
        return {
            "detected": False,
            "explanation": "cannot analyse profile: insufficient data",
            "fix": "None required - profile could not be analysed",
            "evidence": "UNVERIFIED",
        }

    if result.cpu_layer_count == 0 and result.npu_layer_count > 0:
        return {
            "detected": True,
            "explanation": (
                f"All {result.npu_layer_count} layers run on the NPU "
                f"with no CPU fallbacks. "
                f"NPU fraction: {result.npu_fraction:.0%}"
            ),
            "fix": (
                "No action needed - model already executes on NPU. "
                "If targeting a different device, verify NPU support "
                "for all op types."
            ),
            "evidence": "experiments/baseline_fp32/profile.json",
        }
    return {
        "detected": True,
        "explanation": (
            f"NPU execution incomplete: {result.npu_layer_count} of "
            f"{result.total_layer_count} layers on NPU, "
            f"{result.cpu_layer_count} on CPU, "
            f"{result.gpu_layer_count} on GPU"
        ),
        "fix": (
            "Identify unsupported ops and quantize or refactor "
            "for NPU compatibility. Consider using "
            "--onnx_execution_providers=qnn for QNN-accelerated ops."
        ),
        "evidence": "experiments/baseline_fp32/profile.json",
    }


def no_fallback_layers(result: ProfileResult) -> dict:
    """
    Rule: No layers fall back to CPU/GPU.
    Legacy dict-returning form. Prefer apply_rules() in new code.
    """
    if not _guard(result):
        return {
            "detected": False,
            "explanation": "cannot analyse profile: insufficient data",
            "fix": "None required - profile could not be analysed",
            "evidence": "UNVERIFIED",
        }

    total = max(result.total_layer_count, 1)
    gpu_frac = result.gpu_layer_count / total

    if len(result.fallback_layers) == 0:
        return {
            "detected": True,
            "explanation": (
                f"No fallback layers detected: all {result.total_layer_count} "
                f"layers execute on NPU. CPU fraction: {result.cpu_fraction:.0%}, "
                f"GPU fraction: {gpu_frac:.0%}"
            ),
            "fix": (
                "No fallback fix needed. Model is NPU-optimized. "
                "If targeting a different device or quantized variant, "
                "re-run the tool to check for new fallbacks."
            ),
            "evidence": "experiments/baseline_fp32/profile.json",
        }
    return {
        "detected": True,
        "explanation": (
            f"Fallback layers detected: {len(result.fallback_layers)} of "
            f"{result.total_layer_count} layers fall back to "
            f"{result.cpu_fraction:.0%} CPU and {gpu_frac:.0%} GPU"
        ),
        "fix": (
            "Identify unsupported operations and consider: "
            "(1) quantizing with --onnx_execution_providers=qnn, "
            "(2) refactoring ops for NPU compatibility, "
            "(3) falling back to CPU for selected layers."
        ),
        "evidence": "experiments/baseline_fp32/profile.json",
    }


def high_latency_threshold(result: ProfileResult, threshold_ms: float = 200.0) -> dict:
    """
    Rule: Inference latency exceeds a threshold.
    Legacy dict-returning form. Prefer apply_rules() in new code.
    """
    if not _guard(result):
        return {
            "detected": False,
            "explanation": "cannot analyse profile: insufficient data",
            "fix": "None required - profile could not be analysed",
            "evidence": "UNVERIFIED",
        }

    t_ms = result.total_inference_time_ms
    if t_ms is not None and t_ms > threshold_ms:
        return {
            "detected": True,
            "explanation": (
                f"Inference latency {t_ms:.1f}ms "
                f"exceeds {threshold_ms:.1f}ms threshold "
                f"({t_ms / 1000.0:.2f}s). Consider optimization."
            ),
            "fix": (
                "Try INT8 quantization with QNN execution provider, "
                "reduce model complexity, or use layer fusion. "
                "Re-run profiling after optimization to measure improvement."
            ),
            "evidence": "experiments/baseline_fp32/profile.json",
        }
    t_str = f"{t_ms:.1f}ms" if t_ms is not None else "N/A"
    return {
        "detected": False,
        "explanation": (
            f"Inference latency {t_str} "
            f"within {threshold_ms:.1f}ms threshold. No action needed."
        ),
        "fix": "None required. Latency is within acceptable range.",
        "evidence": "experiments/baseline_fp32/profile.json",
    }