import json
import os
import tempfile
from pathlib import Path

from apex_harness.mcp_client import MCPManager
from apex_harness.session_memory import SessionMemory


def test_session_memory_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_sessions.db"
        mem = SessionMemory(str(db_path))

        # Open session
        sess_id = "sess_arch_001"
        mem.open_session(sess_id, model="local-qwen")

        # Log decisions and files
        mem.log_decision(sess_id, "Use SQLite WAL mode for session tracking")
        mem.log_file_touched(sess_id, "/home/leonardo/test.py")
        mem.add_note(sess_id, "Check performance under high concurrency")

        # Add TODO and close it
        todo1 = mem.add_todo(sess_id, "Implement tests for session memory")
        todo2 = mem.add_todo(sess_id, "Document CLI /sessions commands")
        assert todo1 > 0
        assert todo2 > 0

        # Close first todo
        ok = mem.close_todo(sess_id, todo1)
        assert ok is True

        # Check list_sessions
        sessions = mem.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["id"] == sess_id
        assert sessions[0]["event_count"] >= 5

        # Check resume text
        summary = mem.resume_session(sess_id)
        assert "RESUMED SESSION: sess_arch_001" in summary
        assert "SQLite WAL mode" in summary
        assert "test.py" in summary
        assert f"(#{todo2})" in summary
        assert "[✓] Implement tests for session memory" in summary


def test_session_memory_env_session_id_and_active_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "active_sessions.db"
        os.environ.pop("APEX_SESSION_ID", None)
        mem = SessionMemory(str(db_path))
        session_id = mem.ensure_session_id()

        assert session_id
        assert os.environ["APEX_SESSION_ID"] == session_id

        mem.open_session(session_id, model="local-qwen")
        mem.add_note(session_id, "Session should be persisted")

        active = mem.get_active_sessions(limit=10)
        assert len(active) >= 1
        assert active[0]["id"] == session_id


def test_mcp_manager_respects_enabled_servers_from_env(monkeypatch, tmp_path):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({
        "mcpServers": {
            "alpha": {"command": "echo", "args": ["alpha"]},
            "beta": {"command": "echo", "args": ["beta"]},
        }
    }), encoding="utf-8")

    monkeypatch.setenv("APEX_MCP_CONFIG", str(cfg))
    monkeypatch.setenv("APEX_MCP_SERVERS", "beta")

    mgr = MCPManager()
    assert list(mgr.servers.keys()) == ["beta"]
    assert mgr.enabled_servers == ["beta"]
    assert "alpha" not in mgr.servers


def test_estimate_token_throughput_fallback_when_server_has_no_timings():
    from apex_harness.core import estimate_token_throughput

    info = estimate_token_throughput(
        prompt_tokens=160,
        completion_tokens=80,
        elapsed_seconds=2.0,
        cached_tokens=40,
        timings={},
    )

    assert info["prefill_tps"] > 0
    assert info["decode_tps"] > 0
    assert info["cached_tokens"] == 40


def test_smooth_live_usage_metric_is_stable():
    from apex_harness.cli import smooth_live_metric

    first = smooth_live_metric("decode_tps", 8.0)
    second = smooth_live_metric("decode_tps", 20.0)
    third = smooth_live_metric("decode_tps", 12.0)

    assert first == 8.0
    assert second > first
    assert third > 0


def test_format_live_usage_renders_dashboard_metrics():
    from apex_harness.cli import format_live_usage

    rendered = format_live_usage({
        "prompt_tokens": 128,
        "completion_tokens": 64,
        "prefill_tps": 55.3,
        "decode_tps": 22.7,
    })

    assert "Prompt" in rendered
    assert "Completion" in rendered
    assert "Prefill" in rendered
    assert "Decode" in rendered
    assert "55.3" in rendered
    assert "22.7" in rendered
