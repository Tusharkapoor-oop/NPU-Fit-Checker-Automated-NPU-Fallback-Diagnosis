"""
fitchecker/profile_parser.py

Reads a Qualcomm AI Hub profile JSON and returns a structured
ProfileResult dataclass.

Confirmed schema (from docs/PROFILE_SCHEMA.md):
  - Top-level keys:
      execution_summary (dict)
      execution_detail  (list[dict])
  - execution_summary keys:
      estimated_inference_time (int/float, in microseconds µs)
  - execution_detail item keys:
      name (str)
      type (str)
      compute_unit (str: "NPU", "CPU", "GPU", etc.)
      execution_time (int/float)
      execution_cycles (int/float)

Every field access uses ONLY confirmed key names.
If any required key is missing, SchemaError is raised (never default to UNASSIGNED or 0).
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Optional


class SchemaError(ValueError):
    """Raised when profile JSON does not conform to the confirmed AI Hub schema."""
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LayerInfo:
    """Per-layer (or per-op-graph-node) information from the profile."""
    name: str
    op_type: str                                # e.g. "Conv", "Clip", "Gemm"
    compute_unit: str                           # e.g. "NPU", "CPU", "GPU"
    execution_time_us: Optional[float] = None   # microseconds
    execution_time_ms: Optional[float] = None   # milliseconds
    execution_cycles: Optional[int] = None
    raw: dict = field(default_factory=dict)     # full raw dict for debugging


@dataclass
class ProfileResult:
    """Top-level result of parsing a profile JSON."""
    job_id: Optional[str] = None
    device_name: Optional[str] = None
    model_name: Optional[str] = None
    profile_path: Optional[str] = None

    # Latency
    total_inference_time_us: Optional[float] = None   # microseconds
    total_inference_time_ms: Optional[float] = None   # milliseconds

    # Compute unit summary counts
    npu_layer_count: int = 0
    cpu_layer_count: int = 0
    gpu_layer_count: int = 0
    unassigned_layer_count: int = 0

    # Per-layer data
    layers: list[LayerInfo] = field(default_factory=list)

    # Schema verification flag
    schema_verified: bool = True

    # Raw top-level keys for schema exploration
    raw_keys: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def total_layer_count(self) -> int:
        return len(self.layers)

    @property
    def npu_fraction(self) -> float:
        if self.total_layer_count == 0:
            return 0.0
        return self.npu_layer_count / self.total_layer_count

    @property
    def cpu_fraction(self) -> float:
        if self.total_layer_count == 0:
            return 0.0
        return self.cpu_layer_count / self.total_layer_count

    @property
    def fallback_layers(self) -> list[LayerInfo]:
        """Layers not running on NPU."""
        return [layer for layer in self.layers if layer.compute_unit.upper() != "NPU"]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

REQUIRED_LAYER_KEYS = ("name", "type", "compute_unit", "execution_time", "execution_cycles")


def parse_profile(
    data: dict,
    job_id: Optional[str] = None,
    device_name: Optional[str] = None,
    model_name: Optional[str] = None,
) -> ProfileResult:
    """
    Parse a raw profile dict into a ProfileResult.
    Raises SchemaError if confirmed keys are missing.
    """
    if not isinstance(data, dict):
        raise SchemaError(f"Profile data must be a dict, got {type(data).__name__}")

    # Top-level schema validation
    if "execution_summary" not in data:
        raise SchemaError("Missing required top-level key: 'execution_summary'")
    if not isinstance(data["execution_summary"], dict):
        raise SchemaError("'execution_summary' must be a dict")

    if "execution_detail" not in data:
        raise SchemaError("Missing required top-level key: 'execution_detail'")
    if not isinstance(data["execution_detail"], list):
        raise SchemaError("'execution_detail' must be a list")

    summary = data["execution_summary"]
    if "estimated_inference_time" not in summary:
        raise SchemaError("Missing required key in execution_summary: 'estimated_inference_time'")

    try:
        est_time_us = float(summary["estimated_inference_time"])
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"Invalid 'estimated_inference_time' value: {summary['estimated_inference_time']}") from exc

    result = ProfileResult()
    result.raw = data
    result.raw_keys = list(data.keys())
    result.total_inference_time_us = est_time_us
    result.total_inference_time_ms = est_time_us / 1000.0
    result.schema_verified = True

    # Job / device / model metadata: check explicit args, then data dict if present
    result.job_id = job_id or data.get("job_id") or data.get("id")
    result.device_name = device_name or data.get("device") or data.get("device_name")
    result.model_name = model_name or data.get("model_name") or data.get("name")

    # Parse layers
    for idx, raw_layer in enumerate(data["execution_detail"]):
        if not isinstance(raw_layer, dict):
            raise SchemaError(f"Layer {idx} in execution_detail must be a dict")
        layer = _parse_layer(raw_layer, idx)
        result.layers.append(layer)

        cu = layer.compute_unit.upper()
        if cu == "NPU":
            result.npu_layer_count += 1
        elif cu == "CPU":
            result.cpu_layer_count += 1
        elif cu == "GPU":
            result.gpu_layer_count += 1
        else:
            result.unassigned_layer_count += 1

    return result


def _parse_layer(raw_layer: dict, index: int) -> LayerInfo:
    """Parse a single layer dict into LayerInfo. Raises SchemaError if keys missing."""
    for k in REQUIRED_LAYER_KEYS:
        if k not in raw_layer:
            raise SchemaError(f"Missing required key '{k}' in layer {index}: {raw_layer}")

    name = str(raw_layer["name"])
    op_type = str(raw_layer["type"])
    compute_unit = str(raw_layer["compute_unit"]).strip()
    if not compute_unit:
        raise SchemaError(f"Empty compute_unit in layer {index}: {raw_layer}")

    try:
        raw_time = float(raw_layer["execution_time"])
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"Invalid execution_time in layer {index}: {raw_layer['execution_time']}") from exc

    try:
        cycles = int(raw_layer["execution_cycles"])
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"Invalid execution_cycles in layer {index}: {raw_layer['execution_cycles']}") from exc

    return LayerInfo(
        name=name,
        op_type=op_type,
        compute_unit=compute_unit.upper(),
        execution_time_us=raw_time,
        execution_time_ms=raw_time / 1000.0,
        execution_cycles=cycles,
        raw=raw_layer,
    )


def load_profile(
    json_path: str,
    job_id: Optional[str] = None,
    device_name: Optional[str] = None,
    model_name: Optional[str] = None,
) -> ProfileResult:
    """Load a profile from a JSON file path."""
    p = pathlib.Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Profile file not found: {json_path}")
    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    result = parse_profile(data, job_id=job_id, device_name=device_name, model_name=model_name)
    result.profile_path = str(p).replace("\\", "/")
    return result


def describe_schema(json_path: str) -> str:
    """Print a human-readable schema description of a profile JSON file."""
    p = pathlib.Path(json_path)
    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    lines = [f"Schema description of: {p.name}", "=" * 60]
    lines.append(f"Top-level keys ({len(data)}):")
    for k, v in data.items():
        vtype = type(v).__name__
        if isinstance(v, list):
            sample = f"list[{len(v)} items]"
            if v and isinstance(v[0], dict):
                sample += f", item keys: {list(v[0].keys())}"
        elif isinstance(v, dict):
            sample = f"dict, keys: {list(v.keys())}"
        elif isinstance(v, str) and len(v) > 80:
            sample = repr(v[:80]) + "…"
        else:
            sample = repr(v)
        lines.append(f"  {k!r} ({vtype}): {sample}")

    result = parse_profile(data)
    lines.append("")
    lines.append("Parsed summary:")
    lines.append(f"  job_id           : {result.job_id}")
    lines.append(f"  device_name      : {result.device_name}")
    lines.append(f"  model_name       : {result.model_name}")
    lines.append(f"  total_time_us    : {result.total_inference_time_us} µs")
    lines.append(f"  total_time_ms    : {result.total_inference_time_ms:.3f} ms")
    lines.append(f"  total_layers     : {result.total_layer_count}")
    lines.append(f"  NPU layers       : {result.npu_layer_count}")
    lines.append(f"  CPU layers       : {result.cpu_layer_count}")
    lines.append(f"  GPU layers       : {result.gpu_layer_count}")
    lines.append(f"  Unassigned layers: {result.unassigned_layer_count}")

    return "\n".join(lines)