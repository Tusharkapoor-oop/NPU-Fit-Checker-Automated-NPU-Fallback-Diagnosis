"""
fitchecker/report.py

Generates plain-language + Markdown reports from profile findings.

Key changes from earlier version:
  - Device name read from profile.device_name (not hard-coded).
  - Baseline/optimised labels come from what was actually run.
  - Removed the "No values are fabricated" footer until the parser
    has passed real-file tests (Task 3).
  - "cannot analyse profile" message when schema is not verified.
"""

from __future__ import annotations

import datetime
import pathlib
from typing import Optional

from fitchecker.profile_parser import ProfileResult
from fitchecker.rules import RuleFinding

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def _fmt_latency(us: Optional[float]) -> str:
    if us is None:
        return "UNVERIFIED"
    ms = us / 1000.0
    return f"{ms:.3f} ms ({us:.0f} µs)"


def _severity_icon(severity: str) -> str:
    return {"CRITICAL": "[CRIT]", "WARNING": "[WARN]", "INFO": "[INFO]"}.get(severity, "[????]")


def _device_label(profile: ProfileResult) -> str:
    """Return the device name as it appeared in the job; never hard-code."""
    if profile.device_name:
        return profile.device_name
    return "Snapdragon device (name not found in profile JSON)"


def build_markdown_report(
    profile: ProfileResult,
    findings: list[RuleFinding],
    baseline_profile: Optional[ProfileResult] = None,
    title: str = "NPU Fit Checker Report",
    baseline_label: str = "Baseline",
    optimized_label: str = "Optimized",
) -> str:
    lines: list[str] = []
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Header
    lines.append(f"# {title}")
    lines.append(f"*Generated: {ts}*")
    lines.append("")
    device = _device_label(profile)
    lines.append(f"> **Device**: Profiled on Qualcomm AI Hub cloud-hosted Snapdragon device: {device}  ")
    lines.append(f"> **Model**: {profile.model_name or 'UNVERIFIED'}  ")
    lines.append(f"> **Job ID**: {profile.job_id or 'UNVERIFIED'}  ")
    lines.append("")

    # Schema warning
    if not getattr(profile, "schema_verified", False):
        lines.append(
            "> **WARNING**: The profile schema has not been verified against a real file.\n"
            "> Rules have not been applied. Run Task 1 + Task 3 to obtain a verified report."
        )
        lines.append("")

    # Latency
    lines.append("## Latency")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total inference time | {_fmt_latency(profile.total_inference_time_us)} |")
    if baseline_profile and baseline_profile is not profile:
        lines.append(
            f"| {baseline_label} inference time | "
            f"{_fmt_latency(baseline_profile.total_inference_time_us)} |"
        )
        if (profile.total_inference_time_us is not None
                and baseline_profile.total_inference_time_us is not None
                and baseline_profile.total_inference_time_us > 0):
            speedup = baseline_profile.total_inference_time_us / profile.total_inference_time_us
            lines.append(f"| Speedup | {speedup:.2f}x |")
    lines.append("")

    # Compute-unit breakdown
    lines.append("## Compute-Unit Breakdown")
    lines.append("")
    lines.append("| Compute Unit | Layer Count | Fraction |")
    lines.append("|---|---|---|")
    total = profile.total_layer_count
    def _row(unit: str, count: int) -> str:
        frac = f"{count/total*100:.1f}%" if total > 0 else "N/A"
        return f"| {unit} | {count} | {frac} |"
    lines.append(_row("NPU", profile.npu_layer_count))
    lines.append(_row("CPU", profile.cpu_layer_count))
    lines.append(_row("GPU", profile.gpu_layer_count))
    lines.append(_row("UNASSIGNED", profile.unassigned_layer_count))
    lines.append(_row("**Total**", total))
    lines.append("")

    # Findings
    lines.append("## Findings")
    lines.append("")
    if not findings:
        if not getattr(profile, "schema_verified", False):
            lines.append("*Cannot analyse profile: schema not verified.*")
        else:
            lines.append("*No issues detected.*")
    else:
        for idx, f in enumerate(findings, 1):
            icon = _severity_icon(f.severity)
            lines.append(f"### Finding {idx}: {icon} {f.rule_name}")
            lines.append("")
            if f.affected_layers:
                names = [la.name for la in f.affected_layers[:10]]
                extra = len(f.affected_layers) - 10
                summary = ", ".join(f"`{n}`" for n in names)
                if extra > 0:
                    summary += f" ... and {extra} more"
                lines.append(f"**Affected layers** ({len(f.affected_layers)}): {summary}")
                lines.append("")
            lines.append(f"**What happened**: {f.explanation}")
            lines.append("")
            lines.append(f"**Fix**: {f.fix}")
            lines.append("")
            lines.append(f"**Evidence**: `{f.evidence_file}`")
            lines.append("")

    # CPU fallback detail
    fallback = profile.fallback_layers
    if fallback:
        lines.append("## CPU Fallback Layer Detail")
        lines.append("")
        lines.append("| Layer Name | Op Type | Compute Unit | Time (us) |")
        lines.append("|---|---|---|---|")
        for layer in fallback[:50]:
            t = f"{layer.execution_time_us:.1f}" if layer.execution_time_us is not None else "N/A"
            lines.append(f"| `{layer.name}` | {layer.op_type} | {layer.compute_unit} | {t} |")
        if len(fallback) > 50:
            lines.append(f"| ... | {len(fallback) - 50} more rows omitted | | |")
        lines.append("")

    # Before/after comparison
    if baseline_profile and baseline_profile is not profile:
        lines.append("## Before / After Comparison")
        lines.append("")
        lines.append(f"| Metric | {baseline_label} | {optimized_label} |")
        lines.append("|---|---|---|")
        lines.append(
            f"| Total inference time | {_fmt_latency(baseline_profile.total_inference_time_us)} "
            f"| {_fmt_latency(profile.total_inference_time_us)} |"
        )
        lines.append(
            f"| NPU layers | {baseline_profile.npu_layer_count} "
            f"| {profile.npu_layer_count} |"
        )
        lines.append(
            f"| CPU layers | {baseline_profile.cpu_layer_count} "
            f"| {profile.cpu_layer_count} |"
        )
        lines.append("")

    # Caveats
    lines.append("---")
    lines.append("## Caveats and Limits")
    lines.append("")
    lines.append(
        f"- Profiled on Qualcomm AI Hub cloud-hosted Snapdragon device: {device}. "
        "Results may differ on physical hardware."
    )
    lines.append(
        "- Layer-level compute-unit assignment depends on the QNN SDK version used "
        "during compilation. Results may differ with different SDK versions."
    )
    lines.append(
        "- Rules are based on observed patterns from experiments in `experiments/`. "
        "See `docs/LIMITS.md` for limitations."
    )
    lines.append("")
    lines.append("*All latency values come from the downloaded AI Hub profile JSON.*")
    # NOTE: footer "No values fabricated" removed until parser passes real-file tests.

    return "\n".join(lines)


def print_terminal_report(
    profile: ProfileResult,
    findings: list[RuleFinding],
    baseline_profile: Optional[ProfileResult] = None,
) -> None:
    if not RICH_AVAILABLE:
        print(build_markdown_report(profile, findings, baseline_profile))
        return

    console = Console()
    model   = profile.model_name or "UNVERIFIED"
    device  = _device_label(profile)
    latency = _fmt_latency(profile.total_inference_time_us)

    console.print(Panel(
        f"[bold cyan]Model[/]: {model}\n"
        f"[bold cyan]Device[/]: {device}\n"
        f"[bold cyan]Total inference time[/]: {latency}\n"
        f"[bold cyan]Findings[/]: {len(findings)}",
        title="[bold]NPU Fit Checker[/]",
        border_style="cyan",
    ))

    table = Table(title="Compute-Unit Breakdown", box=box.SIMPLE_HEAVY)
    table.add_column("Unit", style="bold")
    table.add_column("Layers", justify="right")
    table.add_column("Fraction", justify="right")
    total = profile.total_layer_count
    def _pct(n: int) -> str:
        return f"{n/total*100:.1f}%" if total > 0 else "N/A"
    table.add_row("NPU",        str(profile.npu_layer_count),        _pct(profile.npu_layer_count),        style="green")
    table.add_row("CPU",        str(profile.cpu_layer_count),        _pct(profile.cpu_layer_count),        style="red" if profile.cpu_layer_count > 0 else "")
    table.add_row("GPU",        str(profile.gpu_layer_count),        _pct(profile.gpu_layer_count))
    table.add_row("UNASSIGNED", str(profile.unassigned_layer_count), _pct(profile.unassigned_layer_count), style="yellow" if profile.unassigned_layer_count > 0 else "")
    table.add_row("Total",      str(total),                          "100%",                               style="bold")
    console.print(table)

    if not getattr(profile, "schema_verified", False):
        console.print("[yellow]WARNING: Schema not verified. No rules applied.[/]")

    for finding in findings:
        icon   = _severity_icon(finding.severity)
        colour = {"CRITICAL": "red", "WARNING": "yellow", "INFO": "green"}.get(finding.severity, "white")
        console.print(Panel(
            f"[bold {colour}]{icon} {finding.rule_name}[/]\n\n"
            f"[bold]What happened:[/] {finding.explanation}\n\n"
            f"[bold]Fix:[/] {finding.fix}\n\n"
            f"[bold]Evidence:[/] {finding.evidence_file}",
            border_style=colour,
        ))


def save_report(report_md: str, output_path: str) -> pathlib.Path:
    p = pathlib.Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report_md, encoding="utf-8")
    return p


def render_batch_summary_markdown(
    results: list[dict],
    device_name: str = "Snapdragon X2 Elite CRD",
) -> str:
    """
    Renders batch profiling results into a markdown comparison table.
    Columns: Model | Total Layers | NPU % | CPU Fallback Layers | Latency | Status
    Every cell traces to a real file or job URL.
    """
    lines: list[str] = []
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("# Batch Profiling Summary")
    lines.append("")
    lines.append(f"*Generated: {ts}*  ")
    lines.append(f"> **Target Device**: Qualcomm AI Hub cloud-hosted {device_name}")
    lines.append("")
    lines.append("| Model | Total Layers | NPU % | CPU Fallback Layers | Latency | Status | Profile / Job |")
    lines.append("|---|---|---|---|---|---|---|")

    for r in results:
        model = r.get("model_name", "Unknown")
        status = r.get("status", "Unknown")
        total_layers = r.get("total_layers", "N/A")
        npu_pct = r.get("npu_pct", "N/A")
        cpu_layers = r.get("cpu_layers", "N/A")
        latency = r.get("latency", "N/A")
        link = r.get("link", "N/A")

        lines.append(
            f"| **{model}** | {total_layers} | {npu_pct} | {cpu_layers} | {latency} | {status} | {link} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("### Batch Execution Notes")
    for r in results:
        model = r.get("model_name", "Unknown")
        if r.get("status") == "SUCCESS":
            p_path = r.get("profile_path", "")
            lines.append(f"- **{model}**: Profiled successfully. Telemetry saved to `{p_path}`.")
        else:
            stage = r.get("stage_failed", "unknown stage")
            err = r.get("error_message", "unknown error")
            lines.append(f"- **{model}**: FAILED at stage `{stage}`. Error: `{err}`.")

    return "\n".join(lines)
