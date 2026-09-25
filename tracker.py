"""
NPU Fit Checker — Offline Web Tracker (stdlib only, no pip install needed).

Run:
    python tracker.py [--port 8080]

Then open in a live browser:
    http://localhost:8080

Tracks everything in one page:
  - Profile selector (all experiments/*.json) + baseline selector
  - KPI cards: latency, layers, NPU %, CPU fallbacks, speedup
  - Compute-unit breakdown (inline SVG bars, no CDN)
  - Findings tracker (rules engine output with fix + evidence)
  - Layer explorer (search / compute-unit filter / sort, vanilla JS)
  - Before/After comparison table
  - Run history (results/*.md) + Markdown export / save

Uses only the Python standard library + the local fitchecker package,
so it runs on this machine with no internet access.
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import pathlib
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fitchecker.profile_parser import SchemaError, load_profile
from fitchecker.report import build_markdown_report, save_report
from fitchecker.rules import apply_rules

ROOT = pathlib.Path(__file__).parent
EXPERIMENTS = ROOT / "experiments"
RESULTS = ROOT / "results"

META = {
    "experiments/baseline_fp32/profile.json": ("MobileNetV2 (FP32)", "j5m0d379g"),
    "experiments/optimized_int8/profile.json": ("MobileNetV2 (INT8 QDQ)", "j57e7d0qp"),
}

DEFAULT_PROFILE = "experiments/optimized_int8/profile.json"
DEFAULT_BASELINE = "experiments/baseline_fp32/profile.json"
DEFAULT_DEVICE = "Snapdragon X2 Elite CRD"


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

def bundled_profiles() -> list[str]:
    if not EXPERIMENTS.exists():
        return []
    out = []
    for p in sorted(EXPERIMENTS.rglob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if "execution_summary" in d and "execution_detail" in d:
                out.append(str(p.relative_to(ROOT)).replace("\\", "/"))
        except Exception:
            continue
    return out


def load_view(profile_rel: str, device: str):
    """Return (profile, findings, err). Never raises."""
    try:
        prof = load_profile(str(ROOT / profile_rel))
    except (SchemaError, FileNotFoundError) as exc:
        return None, [], str(exc)
    model, job = META.get(profile_rel, ("", ""))
    if device:
        prof.device_name = device
    if model:
        prof.model_name = model
    if job:
        prof.job_id = job
    return prof, apply_rules(prof), ""


def history_reports() -> list[pathlib.Path]:
    if not RESULTS.exists():
        return []
    return sorted(RESULTS.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)


# ---------------------------------------------------------------------------
# HTML rendering (self-contained: inline CSS + vanilla JS, zero CDN)
# ---------------------------------------------------------------------------

CSS = """
*{box-sizing:border-box}body{font-family:'Segoe UI',Arial,sans-serif;margin:0;
background:#0f172a;color:#e2e8f0}header{background:#1e293b;padding:18px 28px;
border-bottom:3px solid #06b6d4}header h1{margin:0;font-size:22px}
header p{margin:4px 0 0;color:#94a3b8;font-size:13px}.wrap{padding:20px 28px;
max-width:1200px;margin:auto}form.bar{background:#1e293b;padding:14px 18px;
border-radius:10px;display:flex;gap:14px;flex-wrap:wrap;align-items:end;margin-bottom:18px}
label{font-size:12px;color:#94a3b8;display:block;margin-bottom:4px}
select,input{background:#0f172a;color:#e2e8f0;border:1px solid #334155;
border-radius:6px;padding:7px 10px;font-size:14px}
button{background:#06b6d4;border:none;color:#04222b;font-weight:700;
border-radius:6px;padding:8px 16px;cursor:pointer;font-size:14px}
button:hover{background:#22d3ee}.kpis{display:grid;grid-template-columns:
repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:18px}
.kpi{background:#1e293b;border-radius:10px;padding:14px;border-left:4px solid #06b6d4}
.kpi small{color:#94a3b8;font-size:12px}.kpi b{font-size:22px;display:block;margin-top:4px}
.kpi.warn{border-color:#f59e0b}.kpi.ok{border-color:#22c55e}
.card{background:#1e293b;border-radius:10px;padding:18px;margin-bottom:18px}
.card h2{margin:0 0 12px;font-size:17px;color:#67e8f9}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid #334155}
th{color:#67e8f9;position:sticky;top:0;background:#1e293b}
tr:hover td{background:#0f172a}.finding{border:1px solid #334155;border-radius:8px;
padding:12px 14px;margin-bottom:10px}.finding.WARN{border-left:5px solid #f59e0b}
.finding.INFO{border-left:5px solid #22c55e}.finding.CRIT{border-left:5px solid #ef4444}
.bar-row{display:flex;align-items:center;gap:10px;margin:7px 0;font-size:13px}
.bar-lbl{width:110px;color:#94a3b8}.bar-track{flex:1;background:#0f172a;border-radius:6px;height:22px}
.bar-fill{height:22px;border-radius:6px;line-height:22px;font-size:12px;font-weight:700;
color:#04222b;padding-left:8px;white-space:nowrap}.npu{background:#22c55e}.cpu{background:#f59e0b}
.gpu{background:#60a5fa}.una{background:#64748b}
#layerSearch{width:100%;margin-bottom:10px}.scroll{max-height:430px;overflow:auto}
a{color:#67e8f9}.foot{color:#64748b;font-size:12px;text-align:center;padding:16px}
.hist{font-size:13px}.hist li{margin:5px 0}
"""

JS = """
function filterLayers(){
  var q=document.getElementById('layerSearch').value.toLowerCase();
  var cu=document.getElementById('cuFilter').value;
  var rows=document.querySelectorAll('#layerTable tbody tr');
  var n=0;
  rows.forEach(function(r){
    var name=r.cells[0].textContent.toLowerCase();
    var op=r.cells[1].textContent.toLowerCase();
    var unit=r.cells[2].textContent;
    var ok=(q===''||name.includes(q)||op.includes(q))&&(cu==='ALL'||unit===cu);
    r.style.display=ok?'':'none'; if(ok)n++;
  });
  document.getElementById('layerCount').textContent=n+' shown';
}
function sortLayers(col){
  var tb=document.querySelector('#layerTable tbody');
  var rows=Array.from(tb.rows);
  var asc=tb.dataset.sortCol!=col||tb.dataset.sortDir==='desc';
  rows.sort(function(a,b){
    var x=a.cells[col].textContent, y=b.cells[col].textContent;
    if(col===3){x=parseFloat(x)||0;y=parseFloat(y)||0;return asc?x-y:y-x;}
    return asc?x.localeCompare(y):y.localeCompare(x);
  });
  rows.forEach(function(r){tb.appendChild(r);});
  tb.dataset.sortCol=col;tb.dataset.sortDir=asc?'asc':'desc';
}
"""


def esc(s) -> str:
    return html.escape("" if s is None else str(s))


def option_list(items: list[str], current: str) -> str:
    return "\n".join(
        f'<option value="{esc(v)}"{(" selected" if v == current else "")}>{esc(v)}</option>'
        for v in items
    )


def render(profile_rel: str, baseline_rel: str, device: str) -> str:
    profiles = bundled_profiles()
    if profile_rel not in profiles:
        profile_rel = DEFAULT_PROFILE if DEFAULT_PROFILE in profiles else (profiles[0] if profiles else "")
    if baseline_rel not in ("(none)", *profiles):
        baseline_rel = DEFAULT_BASELINE if DEFAULT_BASELINE in profiles else "(none)"

    prof, findings, err = load_view(profile_rel, device) if profile_rel else (None, [], "no profiles found")
    base = None
    if baseline_rel != "(none)":
        base, _, _ = load_view(baseline_rel, device)

    speedup = None
    if (base and prof and base.total_inference_time_us and prof.total_inference_time_us
            and prof.total_inference_time_us > 0):
        speedup = base.total_inference_time_us / prof.total_inference_time_us

    # ---- KPI cards ----
    if err or prof is None:
        body_top = f'<div class="card"><h2>Error</h2><p>{esc(err)}</p></div>'
        layer_json = "[]"
    else:
        total = prof.total_layer_count or 1
        npu_pct = prof.npu_layer_count / total * 100
        lat = f"{prof.total_inference_time_us / 1000.0:.3f} ms ({prof.total_inference_time_us:.0f} us)" \
            if prof.total_inference_time_us is not None else "UNVERIFIED"
        body_top = f"""
        <div class="kpis">
          <div class="kpi"><small>Inference latency</small><b>{lat}</b></div>
          <div class="kpi"><small>Total layers</small><b>{prof.total_layer_count}</b></div>
          <div class="kpi ok"><small>NPU fit</small><b>{npu_pct:.1f}%</b>{prof.npu_layer_count} layers</div>
          <div class="kpi {'warn' if prof.cpu_layer_count else 'ok'}"><small>CPU fallbacks</small>
            <b>{prof.cpu_layer_count}</b>{'needs fix' if prof.cpu_layer_count else 'clean'}</div>
          <div class="kpi"><small>Speedup vs baseline</small>
            <b>{f'{speedup:.2f}x' if speedup else '&mdash;'}</b>{esc(baseline_rel) if speedup else 'no baseline'}</div>
        </div>
        <div class="card"><h2>Run identity</h2>
          <table><tr><th>Field</th><th>Value</th></tr>
          <tr><td>Device</td><td>{esc(prof.device_name or 'UNVERIFIED')}</td></tr>
          <tr><td>Model</td><td>{esc(prof.model_name or 'UNVERIFIED')}</td></tr>
          <tr><td>Job ID</td><td>{esc(prof.job_id or 'UNVERIFIED')}</td></tr>
          <tr><td>Profile file</td><td>{esc(profile_rel)}</td></tr>
          <tr><td>Schema verified</td><td>{'yes' if prof.schema_verified else 'NO'}</td></tr>
          </table></div>
        <div class="card"><h2>Compute-unit breakdown</h2>
          {svg_bar('NPU', prof.npu_layer_count, total, 'npu')}
          {svg_bar('CPU', prof.cpu_layer_count, total, 'cpu')}
          {svg_bar('GPU', prof.gpu_layer_count, total, 'gpu')}
          {svg_bar('UNASSIGNED', prof.unassigned_layer_count, total, 'una')}
        </div>
        <div class="card"><h2>Findings ({len(findings)})</h2>{render_findings(findings)}</div>
        <div class="card"><h2>Before / After comparison</h2>{render_compare(base, prof, baseline_rel, profile_rel)}</div>
        <div class="card"><h2>Layer explorer ({len(prof.layers)} layers, <span id="layerCount"></span>)</h2>
          <input id="layerSearch" placeholder="Search name or op type..." oninput="filterLayers()"/>
          <label>Compute unit</label>
          <select id="cuFilter" onchange="filterLayers()">
            <option value="ALL">ALL</option><option value="NPU">NPU</option>
            <option value="CPU">CPU</option><option value="GPU">GPU</option>
          </select>
          <div class="scroll"><table id="layerTable"><thead><tr>
            <th onclick="sortLayers(0)" style="cursor:pointer">Layer &#8597;</th>
            <th onclick="sortLayers(1)" style="cursor:pointer">Op &#8597;</th>
            <th onclick="sortLayers(2)" style="cursor:pointer">Unit &#8597;</th>
            <th onclick="sortLayers(3)" style="cursor:pointer">Time (us) &#8597;</th>
          </tr></thead><tbody>
          {render_layer_rows(prof)}
          </tbody></table></div></div>
        """
        layer_json = json.dumps([
            {"name": l.name, "op_type": l.op_type, "compute_unit": l.compute_unit,
             "time_us": l.execution_time_us} for l in prof.layers
        ])

    hist = "".join(
        f'<li><a href="/report?name={esc(r.name)}">{esc(r.name)}</a></li>'
        for r in history_reports()[:15]
    ) or "<li>No saved reports yet — use Export below.</li>"

    qs = urllib.parse.urlencode({"profile": profile_rel, "baseline": baseline_rel, "device": device})
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>NPU Fit Checker — Tracker</title><style>{CSS}</style></head><body>
<header><h1>NPU Fit Checker — Tracking Dashboard</h1>
<p>Qualcomm AI Hub cloud-hosted Snapdragon devices · offline mode · every number traces to a raw profile JSON</p></header>
<div class="wrap">
<form class="bar" method="get" action="/">
  <div><label>Profile</label><select name="profile">{option_list(profiles, profile_rel)}</select></div>
  <div><label>Baseline</label><select name="baseline">{option_list(['(none)'] + profiles, baseline_rel)}</select></div>
  <div><label>Device</label><input name="device" value="{esc(device)}" size="28"/></div>
  <div><button type="submit">Load</button></div>
</form>
{body_top}
<div class="card"><h2>History &amp; Export</h2>
<p><a href="/export?{esc(qs)}"><button>Download Markdown report</button></a>
<a href="/save?{esc(qs)}"><button>Save report to results/</button></a></p>
<ul class="hist">{hist}</ul></div>
</div><div class="foot">NPU Fit Checker · CLI: python -m fitchecker run --from-profile &lt;json&gt; --baseline &lt;json&gt; ·
Tracker: python tracker.py</div>
<script>{JS}</script>
<script>filterLayers();</script>
</body></html>"""


def svg_bar(label: str, count: int, total: int, cls: str) -> str:
    pct = (count / total * 100) if total else 0
    return (f'<div class="bar-row"><div class="bar-lbl">{label}</div>'
            f'<div class="bar-track"><div class="bar-fill {cls}" style="width:{max(pct, 2 if count else 0):.1f}%">'
            f'{count} ({pct:.1f}%)</div></div></div>')


def render_findings(findings) -> str:
    if not findings:
        return "<p>No issues detected. Model fits the NPU completely.</p>"
    out = []
    for i, f in enumerate(findings, 1):
        layers = ", ".join(f"<code>{esc(la.name)}</code>" for la in f.affected_layers[:10])
        if len(f.affected_layers) > 10:
            layers += " ..."
        out.append(f"""<div class="finding {esc(f.severity)}">
        <b>Finding {i}: [{esc(f.severity)}] {esc(f.rule_name)}</b><br/>
        <b>Affected ({len(f.affected_layers)}):</b> {layers}<br/>
        <b>What happened:</b> {esc(f.explanation)}<br/>
        <b>Fix:</b> {esc(f.fix)}<br/>
        <b>Evidence:</b> <code>{esc(f.evidence_file)}</code></div>""")
    return "\n".join(out)


def render_layer_rows(prof) -> str:
    rows = []
    for l in prof.layers:
        t = f"{l.execution_time_us:.1f}" if l.execution_time_us is not None else "N/A"
        rows.append(f"<tr><td><code>{esc(l.name)}</code></td><td>{esc(l.op_type)}</td>"
                    f"<td>{esc(l.compute_unit)}</td><td>{t}</td></tr>")
    return "\n".join(rows)


def render_compare(base, prof, baseline_rel: str, profile_rel: str) -> str:
    if base is None:
        return "<p>Select a baseline profile above to enable comparison tracking.</p>"
    speedup = ""
    if base.total_inference_time_us and prof.total_inference_time_us and prof.total_inference_time_us > 0:
        speedup = f"{base.total_inference_time_us / prof.total_inference_time_us:.2f}x"
    return f"""<table><tr><th>Metric</th><th>{esc(baseline_rel)}</th><th>{esc(profile_rel)}</th></tr>
<tr><td>Inference time (us)</td><td>{base.total_inference_time_us}</td><td>{prof.total_inference_time_us}</td></tr>
<tr><td>NPU layers</td><td>{base.npu_layer_count}</td><td>{prof.npu_layer_count}</td></tr>
<tr><td>CPU layers</td><td>{base.cpu_layer_count}</td><td>{prof.cpu_layer_count}</td></tr>
<tr><td>Total layers</td><td>{base.total_layer_count}</td><td>{prof.total_layer_count}</td></tr>
<tr><td><b>Speedup</b></td><td colspan="2"><b>{speedup}</b></td></tr></table>"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "NPUFitTracker/1.0"

    def log_message(self, fmt, *args):  # quieter logging
        print(f"[tracker] {self.address_string()} {fmt % args}")

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        get = lambda k, d="": q.get(k, [d])[0]

        if url.path == "/api/data":
            profile_rel = get("profile", DEFAULT_PROFILE)
            prof, findings, err = load_view(profile_rel, get("device", DEFAULT_DEVICE))
            if prof is None:
                self._send(404, json.dumps({"error": err}).encode(), "application/json")
                return
            payload = {
                "profile": profile_rel,
                "device": prof.device_name,
                "model": prof.model_name,
                "job_id": prof.job_id,
                "latency_us": prof.total_inference_time_us,
                "total_layers": prof.total_layer_count,
                "npu": prof.npu_layer_count,
                "cpu": prof.cpu_layer_count,
                "gpu": prof.gpu_layer_count,
                "findings": [{"rule": f.rule_name, "severity": f.severity,
                              "layers": [la.name for la in f.affected_layers]} for f in findings],
            }
            self._send(200, json.dumps(payload).encode(), "application/json")
            return

        if url.path == "/export":
            profile_rel = get("profile", DEFAULT_PROFILE)
            baseline_rel = get("baseline", DEFAULT_BASELINE)
            prof, findings, err = load_view(profile_rel, get("device", DEFAULT_DEVICE))
            if prof is None:
                self._send(404, f"Error: {err}".encode(), "text/plain")
                return
            base = None
            if baseline_rel != "(none)":
                base, _, _ = load_view(baseline_rel, get("device", DEFAULT_DEVICE))
            md = build_markdown_report(prof, findings, baseline_profile=base,
                                       title=f"NPU Fit Checker - {pathlib.Path(profile_rel).stem}",
                                       baseline_label="FP32 Baseline", optimized_label="INT8 Optimized")
            fname = f"report_{pathlib.Path(profile_rel).stem}.md"
            self._send(200, md.encode("utf-8"), "text/markdown",
                       {"Content-Disposition": f'attachment; filename="{fname}"'})
            return

        if url.path == "/save":
            profile_rel = get("profile", DEFAULT_PROFILE)
            baseline_rel = get("baseline", DEFAULT_BASELINE)
            prof, findings, err = load_view(profile_rel, get("device", DEFAULT_DEVICE))
            if prof is None:
                self._send(404, f"Error: {err}".encode(), "text/plain")
                return
            base = None
            if baseline_rel != "(none)":
                base, _, _ = load_view(baseline_rel, get("device", DEFAULT_DEVICE))
            md = build_markdown_report(prof, findings, baseline_profile=base,
                                       title=f"NPU Fit Checker - {pathlib.Path(profile_rel).stem}",
                                       baseline_label="FP32 Baseline", optimized_label="INT8 Optimized")
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = RESULTS / f"report_{pathlib.Path(profile_rel).stem}_{ts}.md"
            saved = save_report(md, str(out))
            self.send_response(303)
            self.send_header("Location", "/?profile=" + urllib.parse.quote(profile_rel)
                             + "&baseline=" + urllib.parse.quote(baseline_rel))
            self.end_headers()
            print(f"[tracker] Report saved: {saved}")
            return

        if url.path == "/report":
            name = get("name", "")
            target = (RESULTS / name).resolve()
            if not name or RESULTS.resolve() not in target.parents or not target.exists():
                self._send(404, b"Report not found", "text/plain")
                return
            body = ("<html><head><meta charset='utf-8'><title>" + html.escape(name)
                    + "</title></head><body style='background:#0f172a;color:#e2e8f0;"
                      "font-family:monospace;white-space:pre-wrap;padding:24px'>"
                    + html.escape(target.read_text(encoding="utf-8"))
                    + "<p><a href='/' style='color:#67e8f9'>&larr; back</a></p></body></html>")
            self._send(200, body.encode("utf-8"), "text/html; charset=utf-8")
            return

        # default: main page
        page = render(get("profile", DEFAULT_PROFILE),
                      get("baseline", DEFAULT_BASELINE),
                      get("device", DEFAULT_DEVICE))
        self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="NPU Fit Checker offline web tracker (stdlib only).")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[tracker] NPU Fit Checker tracking dashboard live at http://{args.host}:{args.port}")
    print("[tracker] Press Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[tracker] Stopped.")


if __name__ == "__main__":
    main()
