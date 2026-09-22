import tempfile
from pathlib import Path

from apex_harness.trace import TraceEvent, TraceLogger


def test_trace_logging_and_retrieval():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TraceLogger(log_dir=tmpdir)

        ev1 = TraceEvent(
            session_id="session_alpha",
            turn=1,
            tool="read_file",
            args_summary="{'path': 'foo.py'}",
            result_summary="print('hello')",
            latency_ms=12.5,
            tokens_in=50,
            tokens_out=20,
            model="local-test",
            ts="2026-09-17T16:00:00Z"
        )
        ev2 = TraceEvent(
            session_id="session_beta",
            turn=1,
            tool="bash_exec",
            args_summary="{'command': 'ls'}",
            result_summary="file1 file2",
            latency_ms=25.0,
            tokens_in=30,
            tokens_out=15,
            model="local-test",
            ts="2026-09-17T16:01:00Z"
        )

        logger.log_event(ev1)
        logger.log_event(ev2)

        # All events
        all_events = logger.get_events()
        assert len(all_events) == 2
        assert all_events[0].tool == "read_file"
        assert all_events[1].tool == "bash_exec"

        # Filtered events
        alpha_events = logger.get_events(session_id="session_alpha")
        assert len(alpha_events) == 1
        assert alpha_events[0].session_id == "session_alpha"


def test_trace_html_export():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TraceLogger(log_dir=tmpdir)

        ev = TraceEvent(
            session_id="sess_123",
            turn=1,
            tool="web_search",
            args_summary="{'query': 'rust vs python'}",
            result_summary="Rust is fast, Python is simple.",
            latency_ms=150.2,
            tokens_in=100,
            tokens_out=300,
            model="qwen-coder",
            ts="2026-09-17T16:05:00Z"
        )
        logger.log_event(ev)

        out_path = Path(tmpdir) / "test_dashboard.html"
        generated_file = logger.export_html(output_path=str(out_path))

        assert Path(generated_file).exists()
        content = Path(generated_file).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "Apex Harness Telemetry" in content
        assert "web_search" in content
        assert "150.2ms" in content
