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


def test_session_memory_disk_persistence_and_deletion():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_sessions.db"
        mem = SessionMemory(str(db_path))

        sess_id = "sess_disk_001"
        mem.open_session(sess_id, model="test-qwen")

        # Log user message (should trigger auto-summary and disk JSON file creation)
        mem.log_event(sess_id, "user_message", "Como otimizar a latência KV cache no AMD Ryzen AI 9?")
        mem.log_event(sess_id, "assistant_message", "Para otimizar o KV cache, mantenha o prefixo de ferramentas estável.")

        # Check JSON file exists on disk
        json_file = mem.sessions_dir / f"{sess_id}.json"
        assert json_file.exists()

        # Check content in JSON file
        with open(json_file, "r", encoding="utf-8") as f:
            disk_data = json.load(f)
        assert disk_data["id"] == sess_id
        assert len(disk_data["messages"]) == 2
        assert "Como otimizar" in disk_data["summary"]

        # Check get_session returns messages and summary
        sess = mem.get_session(sess_id)
        assert sess is not None
        assert sess["id"] == sess_id
        assert len(sess["messages"]) == 2
        assert sess["messages"][0]["role"] == "user"
        assert sess["messages"][1]["role"] == "assistant"

        # Check delete_session removes SQLite records and disk file
        deleted = mem.delete_session(sess_id)
        assert deleted is True
        assert not json_file.exists()
        assert mem.get_session(sess_id) is None


def test_mcp_turbo_toggle(monkeypatch, tmp_path):
    monkeypatch.delenv("APEX_MCP_SERVERS", raising=False)
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({
        "mcpServers": {
            "notebooks": {"command": "echo", "args": ["nb"]},
            "memory": {"command": "echo", "args": ["mem"]},
        }
    }), encoding="utf-8")

    monkeypatch.setenv("APEX_MCP_CONFIG", str(cfg))
    mgr = MCPManager()
    assert len(mgr.servers) == 2

    # Switch to Turbo (unload all)
    mgr.set_enabled_servers([])
    assert mgr.servers == {}
    assert os.environ.get("APEX_MCP_SERVERS") == "none"

    # Select specific server
    mgr.set_enabled_servers(["notebooks"])
    assert list(mgr.servers.keys()) == ["notebooks"]
