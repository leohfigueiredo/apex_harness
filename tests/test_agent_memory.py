"""
Testes para agent_memory.py (Memória Simbólica e em Camadas L0-L3).
"""

import json
from pathlib import Path
from apex_harness.agent_memory import (
    ContextOffloader,
    LongTermMemory,
    AgentMemory,
    OFFLOAD_PREFIX,
)


def test_offloader_records_and_creates_mermaid_canvas(tmp_path):
    off = ContextOffloader(session_id="test_sess", base_dir=tmp_path)
    
    n1 = off.record("bash_exec", {"command": "ls -la"}, "file1.txt\nfile2.txt", tool_call_id="call_1")
    assert n1 == "N1"
    
    n2 = off.record("read_file", {"path": "main.py"}, "Error: File not found", tool_call_id="call_2")
    assert n2 == "N2"
    
    # Verifica steps gravados
    assert len(off.steps) == 2
    assert off.steps[0].ok is True
    assert off.steps[1].ok is False  # Detectou "Error"
    
    # Verifica ficheiros salvos em disco (refs)
    ref1 = Path(off.steps[0].ref)
    assert ref1.exists()
    assert "file1.txt" in ref1.read_text(encoding="utf-8")
    
    # Verifica Mermaid Canvas
    canvas = off.canvas()
    assert "graph LR" in canvas
    assert "N1" in canvas
    assert "N2" in canvas
    assert "classDef err" in canvas  # N2 deu erro
    
    # Drilldown
    drill = off.drilldown("N1")
    assert "file1.txt" in drill
    
    # Drilldown em nó inexistente
    missing = off.drilldown("N999")
    assert "não existe" in missing or "nao existe" in missing


def test_offloader_offload_history(tmp_path):
    off = ContextOffloader(session_id="test_sess_hist", base_dir=tmp_path)
    
    long_output = "Line " + "\nLine ".join(str(i) for i in range(100))  # > 600 chars
    off.record("bash_exec", "cmd1", long_output, tool_call_id="call_1")
    off.record("bash_exec", "cmd2", long_output, tool_call_id="call_2")
    off.record("bash_exec", "cmd3", long_output, tool_call_id="call_3")
    off.record("bash_exec", "cmd4", long_output, tool_call_id="call_4")
    
    history = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "run task"},
        {"role": "tool", "name": "bash_exec", "tool_call_id": "call_1", "content": long_output},
        {"role": "tool", "name": "bash_exec", "tool_call_id": "call_2", "content": long_output},
        {"role": "tool", "name": "bash_exec", "tool_call_id": "call_3", "content": long_output},
        {"role": "tool", "name": "bash_exec", "tool_call_id": "call_4", "content": long_output},
    ]
    
    # Mantém os últimos 2 intactos, faz offload dos primeiros 2
    changed = off.offload_history(history, keep_last=2, min_chars=100)
    assert changed == 2
    
    assert history[2]["content"].startswith(OFFLOAD_PREFIX)
    assert "N1" in history[2]["content"]
    assert history[3]["content"].startswith(OFFLOAD_PREFIX)
    assert "N2" in history[3]["content"]
    
    # Os últimos 2 continuam com o texto original
    assert not history[4]["content"].startswith(OFFLOAD_PREFIX)
    assert not history[5]["content"].startswith(OFFLOAD_PREFIX)


def test_long_term_memory_l0_to_l3(tmp_path):
    # Desativa embeddings remotas (usa FTS5 BM25 puro)
    ltm = LongTermMemory(base_dir=tmp_path, embedder=None)
    
    # L0
    ltm.add_turn("sess1", "user", "O meu projeto usa Python 3.12 e SQLite com FTS5.")
    ltm.add_turn("sess1", "assistant", "Entendido, vou focar em compatibilidade com SQLite.")
    
    assert len(ltm.pending_turns()) == 2
    search_l0 = ltm.search_conversations("Python SQLite")
    assert len(search_l0) >= 1
    assert "SQLite" in search_l0[0]["content"]
    
    # L1
    ok, msg = ltm.add_atom("O utilizador prefere funções puras em vez de POO.", category="preference")
    assert ok is True
    ok2, _ = ltm.add_atom("O utilizador prefere funções puras em vez de POO.", category="preference")
    assert ok2 is False  # Duplicado rejeitado
    
    ok3, _ = ltm.add_atom("O ambiente usa kernel Linux com módulo amdxdna ativo.", category="environment")
    assert ok3 is True
    
    # L2 Scenario gerado em Markdown
    scen_pref = tmp_path / "memory" / "scenarios" / "preference.md"
    assert scen_pref.exists()
    assert "funções puras" in scen_pref.read_text(encoding="utf-8")
    
    # L1 Busca BM25
    hits = ltm.search("amdxdna kernel")
    assert len(hits) >= 1
    assert "amdxdna" in hits[0]["content"]
    
    # L3 Persona (simulado com função mock)
    def mock_llm(prompt: str) -> str:
        return "# Perfil\n- Desenvolvedor Linux focado em alta performance."
    
    updated = ltm.update_persona(mock_llm)
    assert updated is True
    persona_txt = ltm.persona_text()
    assert "Desenvolvedor Linux" in persona_txt
    
    # Recall block formatado
    block = ltm.recall_block("funções puras")
    assert block is not None
    assert "(preference)" in block


def test_server_memory_endpoints(tmp_path, monkeypatch):
    """Testa os endpoints GET e POST /api/memory do ApexWebHandler."""
    import json
    import io
    from apex_harness.server import ApexWebHandler, _GLOBAL_AGENT
    from apex_harness.core import ApexAgent
    import apex_harness.server as srv_mod

    monkeypatch.setenv("APEX_SESSION_DIR", str(tmp_path))
    agent = ApexAgent(enable_mcp=False)
    srv_mod._GLOBAL_AGENT = agent

    class DummyRequest:
        def __init__(self, method="GET", path="/api/memory", body=b""):
            self.method = method
            self.path = path
            self.body = body

        def makefile(self, mode, *args, **kwargs):
            if "r" in mode:
                return io.BytesIO(self.body)
            return io.BytesIO()

    # 1. Teste POST /api/memory (remember)
    body = json.dumps({"action": "remember", "fact": "Preferir Python moderno", "category": "preference"}).encode("utf-8")
    handler = ApexWebHandler.__new__(ApexWebHandler)
    handler.path = "/api/memory"
    handler.requestline = "POST /api/memory HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    handler._headers_buffer = []
    handler.log_message = lambda *args, **kwargs: None
    handler.log_request = lambda *args, **kwargs: None
    
    handler.do_POST()
    resp_raw = handler.wfile.getvalue().decode("utf-8")
    # pega o payload json após os headers HTTP
    json_part = resp_raw.split("\r\n\r\n")[-1]
    data = json.loads(json_part)
    assert data.get("status") == "ok"
    assert data.get("inserted") is True

    # 2. Teste GET /api/memory
    handler_get = ApexWebHandler.__new__(ApexWebHandler)
    handler_get.path = "/api/memory"
    handler_get.requestline = "GET /api/memory HTTP/1.1"
    handler_get.request_version = "HTTP/1.1"
    handler_get.headers = {}
    handler_get.rfile = io.BytesIO()
    handler_get.wfile = io.BytesIO()
    handler_get._headers_buffer = []
    handler_get.log_message = lambda *args, **kwargs: None
    handler_get.log_request = lambda *args, **kwargs: None

    handler_get.do_GET()
    get_raw = handler_get.wfile.getvalue().decode("utf-8")
    get_data = json.loads(get_raw.split("\r\n\r\n")[-1])
    assert get_data.get("enabled") is True
    assert get_data.get("atoms_count") >= 1
    assert any("Preferir Python moderno" in a.get("fact", "") for a in get_data.get("recent_atoms", []))

    # 3. Teste POST /api/memory (search)
    search_body = json.dumps({"action": "search", "query": "Python moderno"}).encode("utf-8")
    handler_search = ApexWebHandler.__new__(ApexWebHandler)
    handler_search.path = "/api/memory"
    handler_search.requestline = "POST /api/memory HTTP/1.1"
    handler_search.request_version = "HTTP/1.1"
    handler_search.headers = {"Content-Length": str(len(search_body))}
    handler_search.rfile = io.BytesIO(search_body)
    handler_search.wfile = io.BytesIO()
    handler_search._headers_buffer = []
    handler_search.log_message = lambda *args, **kwargs: None
    handler_search.log_request = lambda *args, **kwargs: None

    handler_search.do_POST()
    search_raw = handler_search.wfile.getvalue().decode("utf-8")
    search_data = json.loads(search_raw.split("\r\n\r\n")[-1])
    assert len(search_data.get("results", [])) >= 1

