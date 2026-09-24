"""
Apex Harness — Web Server & REST/SSE API (server.py)
Provides a DeepSeek Harness style browser interface with real-time streaming,
reasoning traces (<think>), model hot-swapping, /btw side-notes, and session history.
"""

import os
import json
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Optional, Dict, Any, List, Tuple

# CORRIGIDO: `OpenAI` era usado nas linhas ~128 e ~348 sem NUNCA ter sido
# importado. Na primeira dessas linhas isso rebentava /api/models com
# NameError -> HTTP 500; na segunda era silenciosamente engolido por um
# `except Exception: pass`. `httpx` era importado localmente em dois sítios.
import httpx
from openai import OpenAI

from apex_harness.core import ApexAgent
from apex_harness.session_memory import get_session_memory

_GLOBAL_AGENT: Optional[ApexAgent] = None
_SERVER_THREAD: Optional[threading.Thread] = None
_RUNNING_SERVER: Optional[ThreadingHTTPServer] = None

_LMS_BIN = shutil.which("lms") or "/home/leonardo/.lmstudio/bin/lms"
#: Cache de `lms ls --json`: {basename_minusculo: modelKey} e o conjunto de keys.
_LMS_CACHE: Dict[str, Any] = {"ts": 0.0, "by_basename": {}, "keys": ()}
_LMS_TIMEOUT = 15.0


def _lms_env():
    """`lms` precisa de HOME; alguns contextos de servico perdem-no."""
    env = os.environ.copy()
    env.setdefault("HOME", os.path.expanduser("~"))
    return env


def _lms(*args: str, timeout: float = _LMS_TIMEOUT) -> Tuple[int, str, str]:
    """Corre o CLI do LM Studio e devolve (rc, stdout, stderr) sem rebentar."""
    try:
        r = subprocess.run([_LMS_BIN, *args], capture_output=True, text=True,
                           timeout=timeout, env=_lms_env())
        return r.returncode, r.stdout or "", r.stderr or ""
    except FileNotFoundError:
        return 127, "", f"binario 'lms' nao encontrado em {_LMS_BIN}"
    except subprocess.TimeoutExpired:
        return 124, "", f"'{_LMS_BIN} {' '.join(args)}' excedeu {timeout}s"
    except Exception as e:
        return 1, "", str(e)


def _lms_models(force: bool = False) -> Tuple[Dict[str, str], Tuple[str, ...]]:
    """
    Mapa {basename_do_ficheiro_em_minusculas -> modelKey} + todas as modelKeys.

    Isto e o que faltava para a troca de modelo funcionar: o /api/models oferece
    nomes de ficheiro ('Swift-Qwen3.8-27B-Q4_K_M.gguf') mas o `lms load` so
    aceita modelKeys ('ukisai/Swift-Qwen3.8-27B-GGUF/swift-qwen3.8-27b-q4_k_m.gguf').
    Sem esta traducao, escolher um GGUF pelo nome nao carregava NADA e o servidor
    respondia na mesma "Modelo alterado com sucesso".
    """
    now = time.time()
    if not force and (now - _LMS_CACHE["ts"]) < 10.0 and _LMS_CACHE["keys"]:
        return _LMS_CACHE["by_basename"], _LMS_CACHE["keys"]

    rc, out, _err = _lms("ls", "--json")
    if rc != 0 or not out.strip():
        return _LMS_CACHE["by_basename"], _LMS_CACHE["keys"]

    try:
        models = json.loads(out)
    except Exception:
        return _LMS_CACHE["by_basename"], _LMS_CACHE["keys"]

    by_base: Dict[str, str] = {}
    keys: List[str] = []
    for m in models if isinstance(models, list) else []:
        key = m.get("modelKey") or m.get("path")
        if not key:
            continue
        keys.append(key)
        rel = m.get("path") or key
        base = os.path.basename(rel).lower()
        by_base.setdefault(base, key)
        by_base.setdefault(str(key).lower(), key)

    _LMS_CACHE.update({"ts": now, "by_basename": by_base, "keys": tuple(keys)})
    return by_base, tuple(keys)


def _lms_loaded() -> Optional[str]:
    """modelKey atualmente carregado no LM Studio, ou None."""
    lst = _lms_ps_list()
    return lst[0].get("identifier") or lst[0].get("modelKey") if lst else None


def _lms_ps_list(force: bool = False) -> List[Dict[str, Any]]:
    """
    TODAS as instancias carregadas no LM Studio.

    Importa porque o `lms load` NAO substitui o modelo carregado -- cria uma
    instancia NOVA. Carregar o mesmo modelo duas vezes com outro ainda em
    memoria deixa ambas presas em RAM; foi assim que 63 GB ficaram carregados
    numa maquina de 45 GiB e o `lms ps` continuou a listar o modelo antigo
    primeiro, parecendo que a troca nao tinha funcionado.
    """
    rc, out, _err = _lms("ps", "--json", timeout=20)
    if rc != 0 or not out.strip():
        return []
    try:
        data = json.loads(out)
    except Exception:
        return []
    return [m for m in data if isinstance(m, dict)] if isinstance(data, list) else []



def _resolve_lm_model(requested: str) -> Optional[str]:
    """Traduz o que o utilizador escolheu para uma modelKey do LM Studio."""
    req = (requested or "").strip()
    if not req:
        return None
    by_base, keys = _lms_models()
    if req in keys:
        return req
    hit = by_base.get(req.lower()) or by_base.get(os.path.basename(req).lower())
    if hit:
        return hit
    # Nome de ficheiro GGUF: procurar por basename em qualquer pasta conhecida.
    stem = os.path.basename(req).lower()
    for k in keys:
        if os.path.basename(k).lower() == stem:
            return k
    return None



class ApexWebHandler(BaseHTTPRequestHandler):
    """HTTP & SSE Handler for Apex Harness Web UI."""

    def log_message(self, format, *args):
        # Suppress standard HTTP request logging to keep terminal clean
        pass

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_DELETE(self):
        url_path = urllib.parse.urlparse(self.path).path
        if url_path in ("/api/session", "/api/sessions"):
            query_params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            sess_id = query_params.get("id", [""])[0].strip()
            self._handle_delete_session({"id": sess_id})
        else:
            self.send_error(404, "Endpoint Not Found")

    def do_GET(self):
        url_path = urllib.parse.urlparse(self.path).path

        if url_path in ["/", "/index.html"]:
            self._serve_static_file("index.html", "text/html")
        elif url_path.startswith("/web/"):
            rel_file = url_path[5:]
            ext = os.path.splitext(rel_file)[1]
            mime_type = "text/html"
            if ext == ".css":
                mime_type = "text/css"
            elif ext == ".js":
                mime_type = "application/javascript"
            elif ext in [".png", ".jpg", ".svg", ".ico"]:
                mime_type = f"image/{ext[1:]}"
            self._serve_static_file(rel_file, mime_type)
        elif url_path == "/api/vision":
            self._handle_get_vision()
        elif url_path == "/api/models":
            self._handle_get_models()
        elif url_path == "/api/history":
            self._handle_get_history()
        elif url_path == "/api/session":
            self._handle_get_session()
        elif url_path == "/api/sessions":
            self._handle_get_sessions()
        elif url_path == "/api/mcp":
            self._handle_get_mcp()
        elif url_path == "/api/project":
            self._handle_get_project()
        elif url_path == "/api/status":
            self._handle_get_status()
        elif url_path == "/api/wiki":
            self._handle_get_wiki()
        elif url_path == "/api/ping":
            self._send_json({"ok": True})
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        url_path = urllib.parse.urlparse(self.path).path

        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len) if content_len > 0 else b""
        data = {}
        if post_body:
            try:
                data = json.loads(post_body.decode("utf-8"))
            except Exception:
                pass

        if url_path == "/api/chat":
            self._handle_post_chat(data)
        elif url_path == "/api/tokenize":
            self._handle_post_tokenize(data)
        elif url_path == "/api/model":
            self._handle_post_model(data)
        elif url_path == "/api/session/select":
            self._handle_post_session_select(data)
        elif url_path == "/api/session/new":
            self._handle_post_session_new()
        elif url_path in ("/api/session/delete", "/api/sessions/delete"):
            self._handle_delete_session(data)
        elif url_path == "/api/mcp":
            self._handle_post_mcp(data)
        elif url_path == "/api/btw":
            self._handle_post_btw(data)
        elif url_path == "/api/clear":
            self._handle_post_clear()
        elif url_path == "/api/effort":
            self._handle_post_effort(data)
        elif url_path == "/api/project":
            self._handle_post_project(data)
        elif url_path == "/api/wiki/compile":
            self._handle_post_wiki_compile(data)
        else:
            self.send_error(404, "Endpoint Not Found")

    def _serve_static_file(self, rel_filename: str, content_type: str):
        web_dir = Path(__file__).parent / "web"
        target_path = (web_dir / rel_filename).resolve()
        if not str(target_path).startswith(str(web_dir.resolve())):
            self.send_error(403, "Access Denied")
            return

        if not target_path.exists() or not target_path.is_file():
            self.send_error(404, f"File {rel_filename} not found")
            return

        try:
            content = target_path.read_bytes()
            self.send_response(200)
            self._set_cors_headers()
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            # Sem isto o browser guarda app.js/style.css em cache e continua a
            # correr JavaScript antigo depois de o ficheiro em disco ser
            # corrigido -- o que faz uma correcao parecer que nao funcionou. Numa
            # interface servida de localhost o custo de revalidar e nulo.
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, str(e))

    def _handle_get_wiki(self):
        """Devolve lista de artigos da WikiSkill ou um artigo específico."""
        try:
            from apex_harness.wikiskill import get_wiki_store
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            slug = qs.get("slug", [""])[0] or qs.get("title", [""])[0]
            category = qs.get("category", [""])[0]

            store = get_wiki_store()
            if slug:
                art = store.get_article(slug)
                if art:
                    self._send_json({"status": "ok", "article": art})
                else:
                    self._send_json({"status": "error", "message": f"Artigo '{slug}' não encontrado."}, status=404)
            else:
                articles = store.list_articles(category=category if category else None)
                self._send_json({
                    "status": "ok",
                    "count": len(articles),
                    "wiki_dir": str(store.wiki_dir),
                    "articles": articles
                })
        except Exception as e:
            self._send_json({"status": "error", "message": str(e)}, status=500)

    def _handle_post_wiki_compile(self, data: Dict[str, Any]):
        """Compila experiência de sessões recentes em conhecimento persistente na Wiki (arXiv:2608.27454)."""
        try:
            from apex_harness.wikiskill import WikiSkillCompiler
            limit = int(data.get("limit", 5)) if isinstance(data, dict) else 5
            compiler = WikiSkillCompiler()
            res = compiler.compile_from_sessions(limit_sessions=limit)
            self._send_json(res)
        except Exception as e:
            self._send_json({"status": "error", "message": str(e)}, status=500)

    def _handle_get_models(self):
        """Lista de modelos NORMALIZADA.

        Antes isto misturava QUATRO esquemas de nomes diferentes -- modelKeys do
        LM Studio ("swift-qwen3.8-27b@q4_k_m"), nomes de ficheiro GGUF
        ("Swift-Qwen3.8-27B-Q4_K_M.gguf"), caminhos com barras
        ("prism-ml/.../x.gguf") -- sem deduplicar. O mesmo modelo aparecia duas
        vezes, e como a troca de modelo so aceitava modelKeys, metade das opcoes
        da lista nao carregavam nada.

        Agora: UMA entrada por modelo, sempre com a modelKey como valor, o nome
        amigavel no label, e os metadados que o `lms ls --json` ja oferece.
        """
        agent = _GLOBAL_AGENT or ApexAgent()

        # 1. Backend ativo (llama-server 8080 ou LM Studio 1234).
        from apex_harness.core import detect_active_api_base
        active_url = detect_active_api_base(agent.base_url)
        if active_url != agent.base_url:
            agent.base_url = active_url
            # CORRIGIDO: `OpenAI` nao estava importado -> NameError aqui -> HTTP 500.
            agent.client = OpenAI(
                base_url=agent.base_url, api_key=agent.api_key,
                timeout=httpx.Timeout(5.0, read=600.0, write=60.0, pool=60.0),
            )

        entries: Dict[str, Dict[str, Any]] = {}

        def _add(key: str, label: Optional[str] = None, **meta):
            if not key:
                return
            if key not in entries:
                entries[key] = {"key": key, "label": label or key}
            if label:
                entries[key]["label"] = label
            entries[key].update({k: v for k, v in meta.items() if v is not None})

        # 2. Modelos do LM Studio (fonte principal: traz metadados ricos).
        by_base, keys = _lms_models()
        rc, out, _err = _lms("ls", "--json")
        if rc == 0 and out.strip():
            try:
                for m in json.loads(out):
                    key = m.get("modelKey") or m.get("path")
                    name = m.get("displayName") or os.path.basename(m.get("path") or key or "")
                    size = m.get("sizeBytes") or 0
                    _add(
                        key,
                        label=name,
                        arch=m.get("architecture"),
                        ctx=m.get("maxContextLength"),
                        vision=bool(m.get("vision")),
                        tools=bool(m.get("trainedForToolUse")),
                        size_gb=round(size / 1024 ** 3, 2) if size else None,
                    )
            except Exception:
                pass

        # 3. O que o endpoint OpenAI do backend anuncia (llama-server, ou LM
        #    Studio se o `lms ls` falhou).
        try:
            raw = urllib.request.urlopen(f"{agent.base_url}/models", timeout=3).read()
            for m in json.loads(raw).get("data", []):
                mid = m.get("id")
                if not mid:
                    continue
                # Traduzir nome de ficheiro -> modelKey, para nao duplicar.
                _add(_resolve_lm_model(mid) or mid, label=os.path.basename(mid))
        except Exception:
            pass

        # 4. O modelo ativo tem de estar sempre presente.
        if agent.model_name:
            _add(_resolve_lm_model(agent.model_name) or agent.model_name)

        available = list(entries.keys())
        # Ativo primeiro, para o <select> o mostrar no topo.
        active = _resolve_lm_model(agent.model_name) or agent.model_name
        if active in available:
            available.remove(active)
            available.insert(0, active)

        self._send_json({
            "active": active,
            "available": available,
            "models": [entries[k] for k in available],
            "base_url": agent.base_url,
            "reasoning_effort": agent.reasoning_effort,
        })

    def _handle_get_project(self):
        curr = Path.cwd().resolve()
        self._send_json({
            "project_dir": str(curr),
            "folder_name": curr.name
        })

    def _handle_post_project(self, data: Dict[str, Any]):
        new_dir = data.get("project_dir", "").strip()
        if not new_dir:
            self._send_json({"status": "error", "message": "Caminho da pasta vazio."}, status=400)
            return

        target_path = Path(os.path.expanduser(new_dir)).resolve()
        if not target_path.exists():
            self._send_json({"status": "error", "message": f"Pasta não encontrada: {target_path}"}, status=404)
            return
        if not target_path.is_dir():
            self._send_json({"status": "error", "message": f"O caminho não é uma pasta: {target_path}"}, status=400)
            return

        try:
            os.chdir(str(target_path))
            self._send_json({
                "status": "ok",
                "message": f"Pasta de trabalho alterada para: {target_path}",
                "project_dir": str(target_path),
                "folder_name": target_path.name
            })
        except Exception as e:
            self._send_json({"status": "error", "message": str(e)}, status=500)

    def _handle_get_status(self):
        """Estado do backend — corrigido.

        Antes: `lms ps --json` com timeout=1.0s (falhava muito), dois `print` de
        DEBUG esquecidos, e a verificacao de fallback so olhava para o
        llama-server na 8080. Resultado: reportava "offline" mesmo com o LM
        Studio a servir na 1234.
        """
        agent = _GLOBAL_AGENT or ApexAgent()

        # CONTAGEM DE TOKENS
        #
        # Era `total_chars // 4`: uma estimativa de caracteres. Para ingles
        # corrente da ~4 caracteres por token, mas para portugues anda perto de 3
        # e para codigo pior ainda -- a barra mostrava um numero que nao
        # correspondia a nada.
        #
        # Agora preferimos SEMPRE o que o servidor reportou (`usage.prompt_tokens`
        # do ultimo pedido). A estimativa fica so como recurso, quando ainda nao
        # houve nenhum pedido nesta sessao.
        from apex_harness.core import message_text
        chars_est = sum(len(message_text(m.get("content"))) for m in agent.history) // 4
        last = getattr(agent, "last_usage", None) or {}
        real_prompt = int(last.get("prompt_tokens") or 0)
        est_tokens = real_prompt or chars_est

        loaded_model = None
        status = "offline"
        load_pct = 0

        # 1. LM Studio.
        lm = _lms_loaded()
        if lm:
            loaded_model = lm
            status = "ready"
            load_pct = 100

        # 2. llama-server na 8080 (ou no backend que o agente estiver a usar).
        if not loaded_model:
            for probe in ("http://127.0.0.1:8080", agent.base_url.rstrip("/").rsplit("/v1", 1)[0]):
                try:
                    raw = urllib.request.urlopen(f"{probe}/health", timeout=2.5).read()
                    health = json.loads(raw)
                    hstatus = health.get("status")
                    if hstatus == "ok":
                        status, load_pct = "ready", 100
                        loaded_model = agent.model_name
                        break
                    if hstatus == "loading model":
                        status, load_pct = "loading", 50
                        loaded_model = agent.model_name
                        break
                except Exception:
                    continue

        # 3. Ultimo recurso: o proprio endpoint OpenAI responde?
        if not loaded_model:
            try:
                raw = urllib.request.urlopen(f"{agent.base_url}/models", timeout=2.5).read()
                data = json.loads(raw).get("data", [])
                if data:
                    loaded_model = data[0].get("id") or agent.model_name
                    status, load_pct = "ready", 100
            except Exception:
                pass

        ram_pct = 0.0
        try:
            import psutil
            ram_pct = psutil.virtual_memory().percent
        except Exception:
            pass

        # Contexto real do modelo carregado, quando o LM Studio o sabe.
        max_ctx = 65536
        try:
            by_base, keys = _lms_models()
            key = loaded_model or agent.model_name
            rc, out, _e = _lms("ls", "--json")
            if rc == 0 and out.strip():
                for m in json.loads(out):
                    if (m.get("modelKey") or "") == key or (m.get("path") or "") == key:
                        max_ctx = m.get("maxContextLength") or max_ctx
                        break
        except Exception:
            pass

        self._send_json({
            "model": loaded_model or agent.model_name,
            "model_loaded_pct": load_pct,
            "status": status,
            "context_tokens": est_tokens,
            "context_tokens_estimated": not bool(real_prompt),
            "max_context": max_ctx,
            "context_pct": min(100.0, round((est_tokens / max_ctx) * 100, 1)),
            "ram_pct": ram_pct,
            "messages_count": len(agent.history),
            # Para a barra de contagem: leitura (o que o modelo le) e escrita
            # (o que escreve). `cached_tokens` mostra quanto do prompt veio do KV
            # cache -- e o que explica os 15,8 s -> 0,4 s entre turnos.
            "completion_tokens": int(last.get("completion_tokens") or 0),
            "cached_tokens": int(last.get("cached_tokens") or 0),
            "prefill_tps": last.get("prefill_tps") or 0.0,
            "decode_tps": last.get("decode_tps") or 0.0,
            "session_prompt_tokens": int(getattr(agent, "session_prompt_tokens", 0) or 0),
            "session_completion_tokens": int(getattr(agent, "session_completion_tokens", 0) or 0),
            "session_cached_tokens": int(getattr(agent, "session_cached_tokens", 0) or 0),
        })

    def _handle_get_history(self):
        agent = _GLOBAL_AGENT or ApexAgent()
        clean_history = []
        for m in agent.history:
            if m.get("role") != "system":
                clean_history.append({
                    "role": m.get("role"),
                    "content": m.get("content")
                })
        self._send_json({"history": clean_history, "model": agent.model_name})

    def _handle_get_sessions(self):
        try:
            mem = get_session_memory()
            sessions = mem.list_sessions(limit=50)
            current_id = os.environ.get("APEX_SESSION_ID") or (sessions[0]["id"] if sessions else "")
            self._send_json({
                "sessions": sessions,
                "current_session_id": current_id
            })
        except Exception as e:
            self._send_json({"sessions": [], "error": str(e)})

    def _handle_get_session(self):
        query_params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        sess_id = query_params.get("id", [""])[0].strip()
        if not sess_id:
            self._send_json({"error": "Parametro id obrigatorio"}, status=400)
            return
        mem = get_session_memory()
        sess = mem.get_session(sess_id)
        if not sess:
            self._send_json({"error": f"Sessao '{sess_id}' nao encontrada"}, status=404)
            return
        self._send_json({"status": "ok", "session": sess})

    def _handle_post_session_select(self, data: Dict[str, Any]):
        sess_id = (data.get("id") or "").strip()
        if not sess_id:
            self._send_json({"error": "id obrigatorio"}, status=400)
            return
        mem = get_session_memory()
        sess = mem.get_session(sess_id)
        if not sess:
            self._send_json({"error": f"Sessao '{sess_id}' nao encontrada"}, status=404)
            return
        os.environ["APEX_SESSION_ID"] = sess_id
        agent = _GLOBAL_AGENT or ApexAgent()
        messages = sess.get("messages", [])
        if hasattr(agent, "load_history"):
            agent.load_history(messages)
        self._send_json({"status": "ok", "session": sess})

    def _handle_post_session_new(self):
        agent = _GLOBAL_AGENT or ApexAgent()
        mem = get_session_memory()
        os.environ.pop("APEX_SESSION_ID", None)
        new_id = mem.ensure_session_id(model=agent.model_name)
        agent.reset()
        self._send_json({"status": "ok", "session_id": new_id})

    def _handle_delete_session(self, data: Optional[Dict[str, Any]] = None):
        data = data or {}
        sess_id = (data.get("id") or "").strip()
        if not sess_id:
            query_params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            sess_id = query_params.get("id", [""])[0].strip()
        if not sess_id:
            self._send_json({"error": "id obrigatorio"}, status=400)
            return
        mem = get_session_memory()
        mem.delete_session(sess_id)
        current = os.environ.get("APEX_SESSION_ID")
        active_id = current
        if current == sess_id or not current:
            agent = _GLOBAL_AGENT or ApexAgent()
            os.environ.pop("APEX_SESSION_ID", None)
            active_id = mem.ensure_session_id(model=agent.model_name)
            agent.reset()
        self._send_json({
            "status": "ok",
            "deleted": sess_id,
            "active_session_id": active_id
        })

    def _handle_get_mcp(self):
        try:
            from apex_harness.mcp_client import get_mcp_manager
            agent = _GLOBAL_AGENT or ApexAgent()
            mgr = get_mcp_manager()

            all_servers = []
            if mgr.config_path and mgr.config_path.exists():
                try:
                    with open(mgr.config_path, "r", encoding="utf-8") as f:
                        cdata = json.load(f)
                        all_servers = list(cdata.get("mcpServers", {}).keys())
                except Exception:
                    pass
            if not all_servers:
                all_servers = list(mgr.servers.keys())

            mcp_tool_names = [t["function"]["name"] for t in agent.tools if t["function"]["name"].startswith("mcp_")]
            is_turbo = len(mcp_tool_names) == 0
            enabled = mgr.enabled_servers

            if is_turbo:
                mode = "turbo"
            elif set(all_servers) and set(enabled) >= set(all_servers):
                mode = "all"
            else:
                mode = "custom"

            self._send_json({
                "mode": mode,
                "available_servers": all_servers,
                "enabled_servers": enabled,
                "total_tools": len(agent.tools),
                "mcp_tools_count": len(mcp_tool_names),
                "mcp_tools": mcp_tool_names
            })
        except Exception as e:
            self._send_json({"error": str(e), "mode": "unknown", "available_servers": []}, status=500)

    def _handle_post_mcp(self, data: Dict[str, Any]):
        try:
            from apex_harness.mcp_client import get_mcp_manager
            agent = _GLOBAL_AGENT or ApexAgent()
            mgr = get_mcp_manager()

            mode = (data.get("mode") or "").lower().strip()
            servers = data.get("servers") or []

            all_servers = []
            if mgr.config_path and mgr.config_path.exists():
                try:
                    with open(mgr.config_path, "r", encoding="utf-8") as f:
                        cdata = json.load(f)
                        all_servers = list(cdata.get("mcpServers", {}).keys())
                except Exception:
                    pass
            if not all_servers:
                all_servers = list(mgr.servers.keys())

            if mode in ("turbo", "unload", "off", "disable"):
                agent.tools = [t for t in agent.tools if not t["function"]["name"].startswith("mcp_")]
                mgr.set_enabled_servers([])
                msg = f"⚡ Modo Turbo ativado (Sem MCPs, ~4.2 t/s). {len(agent.tools)} ferramentas base ativas."
                final_mode = "turbo"
            elif mode in ("all", "load", "on", "enable"):
                mgr.set_enabled_servers(all_servers)
                agent.tools = [t for t in agent.tools if not t["function"]["name"].startswith("mcp_")]
                agent._init_mcp()
                msg = f"🔌 Todos os MCPs carregados ({len(agent.tools)} ferramentas ativas)."
                final_mode = "all"
            elif mode == "custom" or servers:
                chosen = [s.strip() for s in servers if s.strip()]
                mgr.set_enabled_servers(chosen)
                agent.tools = [t for t in agent.tools if not t["function"]["name"].startswith("mcp_")]
                agent._init_mcp()
                msg = f"🔌 MCPs configurados: {', '.join(chosen)} ({len(agent.tools)} ferramentas ativas)."
                final_mode = "custom"
            else:
                self._send_json({"error": "Modo inválido. Escolha 'turbo', 'all' ou 'custom'."}, status=400)
                return

            mcp_tool_names = [t["function"]["name"] for t in agent.tools if t["function"]["name"].startswith("mcp_")]
            self._send_json({
                "status": "ok",
                "message": msg,
                "mode": final_mode,
                "available_servers": all_servers,
                "enabled_servers": mgr.enabled_servers,
                "total_tools": len(agent.tools),
                "mcp_tools_count": len(mcp_tool_names)
            })
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_post_model(self, data: Dict[str, Any]):
        """Troca de modelo — agora VERIFICADA.

        O bug grave que isto corrige: `ApexAgent.set_model()` apenas muda uma
        string; nao carrega nada. E o `lms load` so era tentado quando o nome
        coincidia exatamente com uma modelKey do LM Studio. Como o /api/models
        oferecia maioritariamente NOMES DE FICHEIRO, escolher um modelo na
        pratica nao fazia nada -- e mesmo assim o servidor respondia
        `{"status": "ok", "message": "Modelo alterado com sucesso"}`.

        Agora: resolve o nome -> modelKey, carrega, e CONFIRMA com `lms ps`.
        So devolve ok se o modelo pedido estiver mesmo carregado.
        """
        requested = (data.get("model", "") or "").strip()
        agent = _GLOBAL_AGENT or ApexAgent()
        if not requested:
            self._send_json({"status": "error", "message": "Nome de modelo invalido"}, status=400)
            return

        if "mmproj" in requested.lower():
            self._send_json({
                "status": "error",
                "message": (f"'{requested}' e um projetor de visao multimodal (mmproj), "
                            "nao um modelo de linguagem. Escolhe o modelo principal.")
            }, status=400)
            return

        _lms_models(force=True)          # refresh, pode ter entrado modelo novo
        model_key = _resolve_lm_model(requested)
        currently = _lms_loaded()

        # Se ja e o modelo carregado, nao vale a pena recarregar.
        if model_key and currently and model_key == currently:
            agent.set_model(model_key)
            self._send_json({"status": "ok", "message": f"'{model_key}' ja estava carregado.",
                             "active": agent.model_name, "base_url": agent.base_url,
                             "verified": True})
            return

        if not model_key:
            # Se nao esta no catalogo do LM Studio, permite definir como modelo do agente (llama-server externo / testes)
            agent.set_model(requested)
            self._send_json({
                "status": "ok",
                "message": f"Modelo alterado para '{requested}'.",
                "active": agent.model_name,
                "base_url": agent.base_url,
                "verified": False
            })
            return

        # CRITICO: `lms load` NAO substitui o modelo carregado -- cria uma
        # INSTANCIA NOVA. Sem descarregar primeiro, ficam varios modelos presos
        # em RAM ao mesmo tempo e o carregamento acaba a falhar por pressao de
        # memoria. A UI apresenta UM modelo ativo, logo: descarregar tudo, carregar um.
        loaded_ids = [m.get("identifier") or m.get("modelKey") for m in _lms_ps_list()]
        loaded_ids = [i for i in loaded_ids if i]
        if loaded_ids:
            rc_u, out_u, err_u = _lms("unload", "--all", timeout=180)
            if rc_u != 0:
                self._send_json({
                    "status": "error",
                    "message": (f"Nao consegui descarregar o modelo atual "
                                f"({', '.join(loaded_ids)}); nao vou carregar outro para "
                                f"nao esgotar a RAM. Detalhe: {((err_u or out_u) or '').strip()[:200]}"),
                }, status=400)
                return

        # Carregar de facto.
        rc, out, err = _lms("load", model_key, timeout=600)
        if rc != 0:
            detail = (err or out).strip()
            if "invalid ggml type 142" in detail:
                msg = (f"'{model_key}' usa quantizacao ternaria (ggml type 142), nao suportada "
                       "pelo runtime do LM Studio. Usa o fork da PrismML (github.com/PrismML-Eng/llama.cpp).")
            elif "exited before becoming healthy" in detail:
                msg = f"'{model_key}' nao arrancou saudavel. Detalhe: {detail[:300]}"
            elif "memory" in detail.lower() or "ram" in detail.lower():
                msg = (f"'{model_key}' precisa de mais RAM do que a disponivel. "
                       f"Fecha outros programas ou usa um quant menor. Detalhe: {detail[:200]}")
            else:
                msg = f"Falha ao carregar '{model_key}': {detail[:300]}"
            self._send_json({"status": "error", "message": msg}, status=400)
            return

        # VERIFICAR. Sem isto, voltariamos a mentir ao utilizador.
        after = [m.get("identifier") or m.get("modelKey") for m in _lms_ps_list()]
        after = [i for i in after if i]
        if model_key not in after:
            self._send_json({
                "status": "error",
                "message": (f"O LM Studio nao confirmou a troca: pedido '{model_key}', "
                            f"carregado '{', '.join(after) or 'nada'}'."),
            }, status=400)
            return

        agent.set_model(model_key)
        agent.base_url = "http://127.0.0.1:1234/v1"
        agent.client = OpenAI(
            base_url=agent.base_url, api_key=agent.api_key,
            timeout=httpx.Timeout(5.0, read=600.0, write=60.0, pool=60.0),
        )
        self._send_json({
            "status": "ok",
            "message": f"Modelo carregado e confirmado: {model_key}",
            "active": agent.model_name,
            "base_url": agent.base_url,
            "verified": True,
        })

    def _handle_post_btw(self, data: Dict[str, Any]):
        note = data.get("note", "").strip()
        agent = _GLOBAL_AGENT or ApexAgent()
        if note:
            msg = agent.inject_btw(note)
            self._send_json({"status": "ok", "message": msg, "history_len": len(agent.history)})
        else:
            self._send_json({"status": "error", "message": "Nota lateral /btw vazia"}, status=400)

    def _handle_post_effort(self, data: Dict[str, Any]):
        effort = data.get("effort", "").strip()
        agent = _GLOBAL_AGENT or ApexAgent()
        if effort:
            msg = agent.set_reasoning_effort(effort)
            self._send_json({"status": "ok", "message": msg, "effort": agent.reasoning_effort})
        else:
            self._send_json({"status": "error", "message": "Effort inválido"}, status=400)

    def _handle_post_clear(self):
        agent = _GLOBAL_AGENT or ApexAgent()
        agent.reset()
        self._send_json({"status": "ok", "message": "Histórico e contexto reiniciados com sucesso."})

    def _backend_vision(self) -> Dict[str, Any]:
        """
        O backend carregado sabe ver imagens?

        O llama-server responde a isso no /props:
            modalities: {"vision": false, "video": false, "audio": false}
        e passa a `true` quando arranca com `-mm/--mmproj FILE`.

        Isto e o que impede a interface de oferecer um botao de imagem que
        falharia a seguir: sem projetor multimodal o modelo e texto apenas, e
        mandar-lhe uma imagem so produz um erro do servidor.
        """
        agent = _GLOBAL_AGENT or ApexAgent()
        base = (agent.base_url or "http://127.0.0.1:8080/v1").rstrip("/").rsplit("/v1", 1)[0]
        info: Dict[str, Any] = {"vision": False, "modalities": {}, "base": base}
        try:
            with urllib.request.urlopen(f"{base}/props", timeout=3) as resp:
                props = json.loads(resp.read().decode("utf-8"))
            mods = props.get("modalities") or {}
            info["modalities"] = mods
            info["vision"] = bool(mods.get("vision"))
        except Exception:
            pass
        return info

    def _handle_get_vision(self):
        self._send_json(self._backend_vision())

    def _handle_post_tokenize(self, data: Dict[str, Any]):
        """
        Contagem EXACTA de tokens do texto que esta a ser escrito.

        O llama-server tem `/tokenize`, que usa o tokenizer real do modelo
        carregado. Pedir-lho e muito melhor do que `len(texto) // 4`: em
        portugues a divisao por 4 subestima, em codigo erra mais, e o numero
        salta de forma visivel quando se cola um bloco.

        Se o backend nao tiver `/tokenize` (nem todos tem), devolve a estimativa
        com `estimated: true` -- a interface mostra-a na mesma, marcada.
        """
        texto = str(data.get("text") or "")
        if not texto:
            self._send_json({"tokens": 0, "estimated": False})
            return

        base = None
        try:
            from apex_harness.core import detect_active_api_base
            base = detect_active_api_base().rstrip("/")
        except Exception:
            base = None
        if not base:
            base = "http://127.0.0.1:8080/v1"
        root = base.rsplit("/v1", 1)[0]

        try:
            req = urllib.request.Request(
                f"{root}/tokenize",
                data=json.dumps({"content": texto}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            toks = payload.get("tokens")
            if isinstance(toks, list):
                self._send_json({"tokens": len(toks), "estimated": False})
                return
        except Exception:
            pass

        self._send_json({"tokens": max(1, len(texto) // 4), "estimated": True})

    def _handle_post_chat(self, data: Dict[str, Any]):
        user_message = (data.get("message") or "").strip()
        images = data.get("images") or []
        if not isinstance(images, list):
            images = []
        # Uma imagem sem texto e uma pergunta legitima ("o que e isto?"). Sem
        # nenhum dos dois e que nao ha nada para fazer.
        if not user_message and not images:
            self._send_json({"error": "Mensagem do usuário vazia"}, status=400)
            return

        agent = _GLOBAL_AGENT or ApexAgent()

        # As imagens so fazem sentido se o backend tiver projetor multimodal. Sem
        # isto o llama-server devolve um erro incompreensivel a meio do stream;
        # melhor recusar aqui, com uma frase que se percebe.
        if images:
            vis = self._backend_vision()
            if not vis.get("vision"):
                self._send_json({
                    "error": ("O backend carregado nao tem visao. O modelo e texto "
                              "apenas, ou o llama-server nao arrancou com o projetor "
                              "multimodal (-mm/--mmproj). Carrega um modelo "
                              "multimodal pelo atalho, que o projetor ao lado do "
                              "modelo e ligado automaticamente.")
                }, status=400)
                return
            validos = []
            for img in images[:6]:
                if isinstance(img, str) and img.startswith("data:image/"):
                    if len(img) > 12 * 1024 * 1024:
                        continue
                    validos.append(img)
            images = validos
            if not images:
                self._send_json({"error": "Nenhuma imagem valida (esperado data:image/...)"},
                                status=400)
                return

        # Send SSE Headers
        #
        # `Connection: close`, NAO `keep-alive`. Esta era a razao de o botao de
        # enviar morrer depois da primeira pergunta.
        #
        # O BaseHTTPRequestHandler trata o cabecalho Connection de forma
        # especial: ao ver `keep-alive` poe self.close_connection = False. Mas
        # uma resposta SSE nao tem Content-Length, portanto o unico terminador
        # que o browser conhece e o EOF do socket. Com o socket aberto, o
        #   while (true) { const { value, done } = await reader.read(); if (done) break; }
        # do web/app.js nunca ve `done: true`. O `finally` que faz
        # `isGenerating = false; btnSend.disabled = false` nunca corre, e a
        # partir dai o clique no botao E o Enter sao no-ops (app.js:265,
        # `if (!message || isGenerating) return;`).
        #
        # Medido: /api/chat pendurava 170 s MESMO quando a resposta era um
        # sucesso completo com o evento `done` ja enviado. Nao era um problema
        # só do caminho de erro -- era de todos os caminhos.
        self.send_response(200)
        self._set_cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        terminal_enviado = {"v": False}

        def send_sse_event(event_type: str, content: str):
            if event_type in ("done", "error"):
                terminal_enviado["v"] = True
            payload = json.dumps({"type": event_type, "content": content})
            msg = f"data: {payload}\n\n"
            try:
                self.wfile.write(msg.encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass

        def on_chunk(chunk_str: str):
            if chunk_str.startswith("<think>"):
                send_sse_event("think_start", "")
                chunk_str = chunk_str[7:]
            if chunk_str.endswith("</think>"):
                chunk_str = chunk_str[:-8]
                send_sse_event("think_chunk", chunk_str)
                send_sse_event("think_end", "")
                return

            if agent._in_reasoning:
                send_sse_event("think_chunk", chunk_str)
            else:
                send_sse_event("text_chunk", chunk_str)

        def on_usage(info: Dict[str, Any]):
            # `usage` so com `content` (o cliente le sempre esse campo). Vai
            # antes do `done` para a barra de baixo actualizar no fim do turno.
            send_sse_event("usage", json.dumps({
                **info,
                "session_completion_tokens": agent.session_completion_tokens,
                "session_prompt_tokens": agent.session_prompt_tokens,
            }))

        try:
            # ------------------------------------------------------------------ #
            # Persistência de sessão — CORRIGIDO.                                 #
            # O `get_session_memory` estava importado desde sempre mas nunca era  #
            # chamado neste handler. Resultado: a lista de sessões ficava vazia,  #
            # e o painel lateral mostrava "Nenhuma sessão salva no banco" mesmo   #
            # com horas de conversa. Agora cada turno abre (ou reabre) a sessão  #
            # e regista os dois lados da conversa.                                #
            # ------------------------------------------------------------------ #
            mem = get_session_memory()
            client_sid = (data.get("session_id") or "").strip()
            if client_sid:
                sid = mem.ensure_session_id(session_id=client_sid, model=agent.model_name)
            else:
                sid = os.environ.get("APEX_SESSION_ID") or mem.ensure_session_id(
                    model=agent.model_name
                )
            send_sse_event("session_id", sid)
            if user_message:
                mem.log_event(sid, "user_message", user_message)

            full_response = agent.step(user_input=user_message, on_chunk=on_chunk,
                                        on_usage=on_usage, images=images)

            if full_response:
                if full_response.startswith("API Error:") or full_response.startswith("⚠️"):
                    send_sse_event("error", full_response)
                else:
                    mem.log_event(sid, "assistant_message", full_response)
                    send_sse_event("done", full_response)
            else:
                send_sse_event("done", full_response or "")
        except Exception as e:
            send_sse_event("error", str(e))
        finally:
            # Um cliente SSE so sabe que o turno acabou pelo EOF. Em QUALQUER
            # caminho -- incluindo se o agent.step() rebentar a meio -- temos de
            # garantir as duas coisas, senao a interface fica presa:
            #   1. um evento terminal, para o cliente parar de renderizar;
            #   2. o fecho do socket, para o loop de leitura terminar.
            if not terminal_enviado["v"]:
                send_sse_event("done", "")
            try:
                self.wfile.flush()
            except Exception:
                pass
            self.close_connection = True


    def _send_json(self, data: Dict[str, Any], status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self._set_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def create_web_server(agent: Optional[ApexAgent] = None, host: str = "127.0.0.1", port: int = 7860) -> ThreadingHTTPServer:
    global _GLOBAL_AGENT
    if agent:
        _GLOBAL_AGENT = agent
    else:
        _GLOBAL_AGENT = ApexAgent(enable_mcp=False)
        def _bg_mcp():
            try:
                _GLOBAL_AGENT._init_mcp()
            except BaseException:
                pass
        threading.Thread(target=_bg_mcp, daemon=True).start()

    server = ThreadingHTTPServer((host, port), ApexWebHandler)
    return server


def start_server_in_thread(agent: Optional[ApexAgent] = None, host: str = "127.0.0.1", port: int = 7860) -> str:
    global _SERVER_THREAD, _RUNNING_SERVER
    if _RUNNING_SERVER is not None:
        return f"http://{host}:{port}"

    server = create_web_server(agent=agent, host=host, port=port)
    _RUNNING_SERVER = server

    def _run():
        server.serve_forever()

    _SERVER_THREAD = threading.Thread(target=_run, daemon=True)
    _SERVER_THREAD.start()

    return f"http://{host}:{port}"


if __name__ == "__main__":
    port = int(os.environ.get("APEX_WEB_PORT", "7860"))
    print(f"🚀 Iniciando Apex Harness Web UI na porta {port}...")
    srv = create_web_server(port=port)
    print(f"✓ Web Harness ativo em http://localhost:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando servidor web.")
