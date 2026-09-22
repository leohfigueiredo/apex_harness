"""
Apex Harness — Observability & Tracing Engine (trace.py)
Captures execution telemetry (tool calls, latencies, tokens, errors)
and generates standalone offline HTML dashboards.
"""

import os
import json
import time
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class TraceEvent:
    session_id: str
    turn: int
    tool: str
    args_summary: str
    result_summary: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    model: str
    ts: str
    status: str = "success"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceEvent":
        return cls(
            session_id=data.get("session_id", "default"),
            turn=data.get("turn", 0),
            tool=data.get("tool", "unknown"),
            args_summary=data.get("args_summary", ""),
            result_summary=data.get("result_summary", ""),
            latency_ms=data.get("latency_ms", 0.0),
            tokens_in=data.get("tokens_in", 0),
            tokens_out=data.get("tokens_out", 0),
            model=data.get("model", "local-llm"),
            ts=data.get("ts", datetime.now(timezone.utc).isoformat()),
            status=data.get("status", "success")
        )


class TraceLogger:
    """Manages telemetry events and formats interactive HTML reports."""

    def __init__(self, log_dir: Optional[str] = None):
        if log_dir:
            self.log_dir = Path(log_dir).resolve()
        else:
            base = os.environ.get("APEX_TRACE_DIR", str(Path.home() / ".apex_traces"))
            self.log_dir = Path(base).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "traces.jsonl"

    def log_event(self, event: TraceEvent) -> None:
        """Append an execution event to the traces JSON Lines file."""
        try:
            line = json.dumps(event.to_dict())
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            # Observability logging should never crash the main application
            print(f"[TraceLogger Warning] Failed to write trace event: {e}")

    def get_events(self, session_id: Optional[str] = None) -> List[TraceEvent]:
        """Read and return logged events, optionally filtered by session_id."""
        if not self.log_file.exists():
            return []

        events = []
        try:
            with open(self.log_file, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        ev = TraceEvent.from_dict(data)
                        if session_id is None or ev.session_id == session_id:
                            events.append(ev)
                    except Exception:
                        continue
        except Exception:
            return []
        return events

    def export_html(self, output_path: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """Generate a zero-dependency offline HTML telemetry dashboard."""
        events = self.get_events(session_id=session_id)
        out_file = Path(output_path).resolve() if output_path else self.log_dir / "dashboard.html"

        total_calls = len(events)
        total_latency = sum(e.latency_ms for e in events)
        avg_latency = (total_latency / total_calls) if total_calls > 0 else 0.0
        total_tokens_in = sum(e.tokens_in for e in events)
        total_tokens_out = sum(e.tokens_out for e in events)
        errors = sum(1 for e in events if e.status == "error" or "error" in e.result_summary.lower()[:30])

        # Find max latency for SVG bar chart scaling
        max_lat = max([e.latency_ms for e in events] or [1.0])

        rows_html = []
        for idx, ev in enumerate(reversed(events), 1):
            bar_width = int(min(100, (ev.latency_ms / max_lat) * 100)) if max_lat > 0 else 0
            is_err = ev.status == "error" or "error" in ev.result_summary.lower()[:30]
            status_badge = '<span class="badge badge-err">ERR</span>' if is_err else '<span class="badge badge-ok">OK</span>'

            row = f"""
            <tr>
                <td>{ev.ts[11:19]}</td>
                <td><code>{ev.session_id[:10]}</code></td>
                <td><b>{ev.tool}</b></td>
                <td>
                    <div class="lat-container">
                        <span>{ev.latency_ms:.1f}ms</span>
                        <div class="lat-bar" style="width: {bar_width}%;"></div>
                    </div>
                </td>
                <td>{ev.tokens_in} / {ev.tokens_out}</td>
                <td>{status_badge}</td>
                <td>
                    <details>
                        <summary>View Payload</summary>
                        <p><b>Args:</b></p>
                        <pre>{ev.args_summary}</pre>
                        <p><b>Result:</b></p>
                        <pre>{ev.result_summary}</pre>
                    </details>
                </td>
            </tr>
            """
            rows_html.append(row)

        table_body = "\n".join(rows_html) if rows_html else "<tr><td colspan='7' style='text-align:center;'>No trace events found.</td></tr>"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Apex Harness — Telemetry Dashboard</title>
    <style>
        :root {{
            --bg: #0d1117;
            --surface: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --text-muted: #8b949e;
            --accent: #58a6ff;
            --accent-glow: rgba(88, 166, 255, 0.15);
            --ok: #3fb950;
            --err: #f85149;
            --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
            --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: var(--font-sans);
            margin: 0;
            padding: 24px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        h1 {{
            margin: 0;
            font-size: 24px;
            color: #f0f6fc;
            letter-spacing: -0.5px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }}
        .stat-label {{
            font-size: 12px;
            text-transform: uppercase;
            color: var(--text-muted);
            letter-spacing: 0.5px;
        }}
        .stat-value {{
            font-size: 26px;
            font-weight: 700;
            color: #f0f6fc;
            margin-top: 6px;
            font-family: var(--font-mono);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border);
            font-size: 13px;
        }}
        th {{
            background: #1f242c;
            color: var(--text-muted);
            font-weight: 600;
        }}
        code, pre {{
            font-family: var(--font-mono);
            font-size: 12px;
        }}
        pre {{
            background: #090d13;
            padding: 8px;
            border-radius: 4px;
            overflow-x: auto;
            max-height: 150px;
            white-space: pre-wrap;
        }}
        .badge {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge-ok {{ background: rgba(63, 185, 80, 0.2); color: var(--ok); }}
        .badge-err {{ background: rgba(248, 81, 73, 0.2); color: var(--err); }}
        .lat-container {{
            display: flex;
            align-items: center;
            gap: 8px;
            min-width: 120px;
        }}
        .lat-bar {{
            height: 6px;
            background: var(--accent);
            border-radius: 3px;
            opacity: 0.8;
        }}
        details summary {{
            cursor: pointer;
            color: var(--accent);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚡ Apex Harness Telemetry</h1>
        <div style="font-size: 12px; color: var(--text-muted);">
            Generated at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
        </div>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Total Tool Calls</div>
            <div class="stat-value">{total_calls}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Avg Latency</div>
            <div class="stat-value">{avg_latency:.1f}ms</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Est Tokens (In / Out)</div>
            <div class="stat-value">{total_tokens_in} / {total_tokens_out}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Errors / Warnings</div>
            <div class="stat-value" style="color: {'var(--err)' if errors > 0 else 'var(--ok)'};">{errors}</div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th>Time</th>
                <th>Session</th>
                <th>Tool</th>
                <th>Latency</th>
                <th>Tokens (In/Out)</th>
                <th>Status</th>
                <th>Payload & Result</th>
            </tr>
        </thead>
        <tbody>
            {table_body}
        </tbody>
    </table>
</body>
</html>
"""
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        return str(out_file)


# Global singleton logger
_GLOBAL_TRACE_LOGGER: Optional[TraceLogger] = None

def get_trace_logger() -> TraceLogger:
    global _GLOBAL_TRACE_LOGGER
    if _GLOBAL_TRACE_LOGGER is None:
        _GLOBAL_TRACE_LOGGER = TraceLogger()
    return _GLOBAL_TRACE_LOGGER
