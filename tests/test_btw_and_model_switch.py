import os
import json
import urllib.request
import threading
from apex_harness.core import ApexAgent
from apex_harness.server import create_web_server


def test_agent_set_model():
    agent = ApexAgent(model_name="initial-model", enable_mcp=False)
    assert agent.model_name == "initial-model"

    res = agent.set_model("qwen2.5-coder-7b-instruct")
    assert agent.model_name == "qwen2.5-coder-7b-instruct"
    assert "qwen2.5-coder-7b-instruct" in res

    # Check that setting empty model keeps old model
    res_empty = agent.set_model("   ")
    assert agent.model_name == "qwen2.5-coder-7b-instruct"
    assert "Modelo inválido" in res_empty


def test_agent_inject_btw():
    agent = ApexAgent(model_name="test-model", enable_mcp=False)
    initial_len = len(agent.history)

    res = agent.inject_btw("Lembre-se de utilizar sintaxe assíncrona")
    assert len(agent.history) == initial_len + 1
    last_msg = agent.history[-1]
    assert last_msg["role"] == "user"
    assert "[BY-THE-WAY / NOTA LATERAL DO USUÁRIO]: Lembre-se de utilizar sintaxe assíncrona" in last_msg["content"]
    assert "com sucesso" in res


def test_web_server_endpoints():
    agent = ApexAgent(model_name="test-model", enable_mcp=False)
    server = create_web_server(agent=agent, port=7899)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        # GET /api/models
        req = urllib.request.Request("http://127.0.0.1:7899/api/models")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            assert data["active"] == "test-model"

        # POST /api/model
        req_model = urllib.request.Request(
            "http://127.0.0.1:7899/api/model",
            data=json.dumps({"model": "deepseek-r1-distill"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req_model) as resp:
            data = json.loads(resp.read().decode())
            assert data["status"] == "ok"
            assert data["active"] == "deepseek-r1-distill"
            assert agent.model_name == "deepseek-r1-distill"

        # POST /api/btw
        req_btw = urllib.request.Request(
            "http://127.0.0.1:7899/api/btw",
            data=json.dumps({"note": "Foque no desempenho"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req_btw) as resp:
            data = json.loads(resp.read().decode())
            assert data["status"] == "ok"
            assert any("[BY-THE-WAY" in m.get("content", "") for m in agent.history)

        # GET /api/wiki
        req_wiki = urllib.request.Request("http://127.0.0.1:7899/api/wiki")
        with urllib.request.urlopen(req_wiki) as resp:
            data = json.loads(resp.read().decode())
            assert data["status"] == "ok"
            assert "articles" in data

    finally:
        server.shutdown()


def test_consecutive_chat_requests():
    """Verify that sending a second chat request completes cleanly without deadlock."""
    agent = ApexAgent(model_name="test-model", enable_mcp=False)

    call_count = {"count": 0}
    def fake_step(user_input, on_chunk=None, on_usage=None, images=None):
        call_count["count"] += 1
        resp = f"Resposta {call_count['count']}: {user_input}"
        if on_chunk:
            on_chunk(resp)
        if on_usage:
            on_usage({"prompt_tokens": 12, "completion_tokens": 8})
        return resp

    agent.step = fake_step
    server = create_web_server(agent=agent, port=7897)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        # Request 1
        req1 = urllib.request.Request(
            "http://127.0.0.1:7897/api/chat",
            data=json.dumps({"message": "Pergunta 1"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req1) as resp:
            content1 = resp.read().decode("utf-8")
            assert 'data: {"type": "text_chunk", "content": "Resposta 1: Pergunta 1"}' in content1
            assert 'data: {"type": "done", "content": "Resposta 1: Pergunta 1"}' in content1

        # Request 2 (consecutive request that previously had issues)
        req2 = urllib.request.Request(
            "http://127.0.0.1:7897/api/chat",
            data=json.dumps({"message": "Pergunta 2"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req2) as resp:
            content2 = resp.read().decode("utf-8")
            assert 'data: {"type": "text_chunk", "content": "Resposta 2: Pergunta 2"}' in content2
            assert 'data: {"type": "done", "content": "Resposta 2: Pergunta 2"}' in content2

        assert call_count["count"] == 2
    finally:
        server.shutdown()


def test_web_server_session_and_mcp_endpoints():
    agent = ApexAgent(model_name="test-model", enable_mcp=False)
    server = create_web_server(agent=agent, port=7896)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        # 1. POST /api/session/new
        req_new = urllib.request.Request(
            "http://127.0.0.1:7896/api/session/new",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req_new) as resp:
            data = json.loads(resp.read().decode())
            assert data["status"] == "ok"
            sess_id = data["session_id"]
            assert sess_id.startswith("apex-")

        # 2. GET /api/sessions
        req_sess = urllib.request.Request("http://127.0.0.1:7896/api/sessions")
        with urllib.request.urlopen(req_sess) as resp:
            data = json.loads(resp.read().decode())
            assert "sessions" in data
            assert any(s["id"] == sess_id for s in data["sessions"])

        # 3. GET /api/session?id=...
        req_get = urllib.request.Request(f"http://127.0.0.1:7896/api/session?id={sess_id}")
        with urllib.request.urlopen(req_get) as resp:
            data = json.loads(resp.read().decode())
            assert data["status"] == "ok"
            assert data["session"]["id"] == sess_id

        # 4. POST /api/session/select
        req_sel = urllib.request.Request(
            "http://127.0.0.1:7896/api/session/select",
            data=json.dumps({"id": sess_id}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req_sel) as resp:
            data = json.loads(resp.read().decode())
            assert data["status"] == "ok"
            assert data["session"]["id"] == sess_id

        # 5. GET /api/mcp
        req_mcp = urllib.request.Request("http://127.0.0.1:7896/api/mcp")
        with urllib.request.urlopen(req_mcp) as resp:
            data = json.loads(resp.read().decode())
            assert "mode" in data
            assert "available_servers" in data

        # 6. POST /api/mcp (turbo mode)
        req_mcp_turbo = urllib.request.Request(
            "http://127.0.0.1:7896/api/mcp",
            data=json.dumps({"mode": "turbo"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req_mcp_turbo) as resp:
            data = json.loads(resp.read().decode())
            assert data["status"] == "ok"
            assert data["mode"] == "turbo"
            assert data["mcp_tools_count"] == 0

        # 7. DELETE /api/session
        req_del = urllib.request.Request(
            f"http://127.0.0.1:7896/api/session?id={sess_id}",
            method="DELETE"
        )
        with urllib.request.urlopen(req_del) as resp:
            data = json.loads(resp.read().decode())
            assert data["status"] == "ok"
            assert data["deleted"] == sess_id

    finally:
        os.environ.pop("APEX_MCP_SERVERS", None)
        server.shutdown()
