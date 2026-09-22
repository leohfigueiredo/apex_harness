"""
Tests for specialized subagents in apex_harness.subagents
"""

import json
from unittest.mock import patch, MagicMock
from apex_harness.subagents import CommitSubagent, TriageSubagent, CriticSubagent, get_subagent_registry


def test_subagent_registry():
    reg = get_subagent_registry()
    assert len(reg) == 3
    names = [s["name"] for s in reg]
    assert "CriticSubagent" in names
    assert "CommitSubagent" in names
    assert "TriageSubagent" in names


def test_commit_subagent_npu_and_fallback():
    sub = CommitSubagent()

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": "feat(core): add subagent support"}}]
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = sub.generate_commit_message("M core.py", "1 file changed, 10 insertions(+)")
        assert res.success is True
        assert "feat(core)" in res.content
        assert res.latency_ms >= 0.0


def test_triage_subagent_runs():
    sub = TriageSubagent()

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": "1. Must parse JSON safely\n2. Must return 0 on error"}}]
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = sub.triage_task("Implement a JSON parser with safety limits")
        assert res.success is True
        assert "JSON" in res.content


def test_critic_subagent_wrapper():
    sub = CriticSubagent()
    with patch("apex_harness.subagents.run_critic") as mock_critic:
        mock_critic.return_value = MagicMock(approved=True, feedback="APPROVED", latency_ms=10.0, model="m")
        res = sub.review_diff("+++ some diff")
        assert res.approved is True
        mock_critic.assert_called_once()
