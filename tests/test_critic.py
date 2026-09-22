import json
from unittest.mock import patch, MagicMock
import urllib.error

from apex_harness.critic import CriticConfig, CriticResult, run_critic


def test_critic_empty_diff():
    res = run_critic("")
    assert res.approved is True
    assert "empty" in res.feedback.lower()


def test_critic_approved_mock():
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": "APPROVED"}}]
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = run_critic("+++ test line", CriticConfig())
        assert res.approved is True
        assert res.feedback == "APPROVED"
        assert res.latency_ms >= 0


def test_critic_rejection_mock():
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": "SyntaxError: Unexpected indent on line 4"}}]
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = run_critic("+++ bad code", CriticConfig())
        assert res.approved is False
        assert "SyntaxError" in res.feedback


def test_critic_endpoint_unreachable_fail_open():
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        res = run_critic("+++ test line", CriticConfig())
        assert res.approved is True
        assert "unavailable" in res.feedback.lower()


def test_critic_npu_routing_fallback_to_secondary():
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": "APPROVED"}}]
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    calls = []
    def mock_urlopen(req, timeout=20.0):
        url = req.full_url
        calls.append(url)
        if "8090" in url:
            raise urllib.error.URLError("NPU service down")
        return mock_resp

    cfg = CriticConfig(
        api_base="http://127.0.0.1:8090/v1",
        fallback_api_base="http://127.0.0.1:8000/v1",
        use_npu=True
    )

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        res = run_critic("+++ test line", cfg)
        assert res.approved is True
        assert res.feedback == "APPROVED"
        chat_calls = [c for c in calls if "chat/completions" in c]
        assert len(chat_calls) == 2
        assert "8090" in chat_calls[0]
        assert "8000" in chat_calls[1]


def test_critic_npu_model_resolution():
    cfg = CriticConfig(
        model="qwen2.5-coder:1.5b",
        npu_model="qwen2.5-coder-1.5b-int4",
        api_base="http://127.0.0.1:8090/v1"
    )
    mock_models_resp = MagicMock()
    mock_models_resp.read.return_value = json.dumps({
        "data": [{"id": "qwen2.5-coder-1.5b-xdna2-custom"}]
    }).encode("utf-8")
    mock_models_resp.__enter__.return_value = mock_models_resp

    mock_chat_resp = MagicMock()
    mock_chat_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": "APPROVED"}}]
    }).encode("utf-8")
    mock_chat_resp.__enter__.return_value = mock_chat_resp

    def mock_urlopen(req, timeout=20.0):
        if "models" in req.full_url:
            return mock_models_resp
        return mock_chat_resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        res = run_critic("+++ test line", cfg)
        assert res.approved is True
        assert res.model == "qwen2.5-coder-1.5b-xdna2-custom"
