import os
import re
import json
import httpx
import urllib.request
from typing import List, Dict, Any, Callable, Optional, Tuple
from openai import OpenAI
from apex_harness.tools import (
    TOOLS_DEFINITION,
    TOOLS_SNIPPETS,
    compress_tool_output,
    execute_tool,
    set_current_agent,
)
from apex_harness.jev_decider import get_jev, JevPermit

DEFAULT_SYSTEM_PROMPT = """You are Apex Harness, an autonomous, highly capable AI engineer and researcher running locally on an AMD Ryzen AI 9 HX 370 (Zen 5, 24 threads) with Radeon 890M GPU and 96GB Unified Memory.

Capabilities & Rules:
1. LIVE INTERNET ACCESS: You have active, real-time web search capabilities through `web_search` and `fetch_url`. You MUST NEVER claim you cannot access the internet or that your knowledge is cut off. If the user asks about recent topics, external documentation, or repos, actively call `web_search` and fetch information.
2. OPERATING SYSTEM & CODING ACCESS: You have full access to inspect files (`list_dir`, `read_file`), make changes (`write_file`, `edit_file`), and execute terminal commands (`bash_exec`).
3. HARDWARE AWARENESS: You are running on high-end hardware with 96GB RAM and ROCm/Vulkan acceleration. Be fast, precise, and practical.
4. RIGOROUS EXECUTION: Always inspect files and verify directory contents before modifying them. When writing or editing code, ensure clean syntax and test your work with `bash_exec` whenever appropriate.
5. CONTINUITY: You keep the full conversation history. Always continue from what you have already done instead of restarting, and refer back to earlier steps, tool results and decisions even if they were many turns ago. If you are unsure whether you already did something, check rather than redo it.

{tools_snippets}
"""

# ---------------------------------------------------------------------------
# Declarative Attention Protocol (DA) -- REMOVIDO, com o porque registado.
#
# O prompt de sistema instruia o modelo a declarar o seu "ambito de atencao"
# no inicio de cada passo, com <global> / <focus:N> / <local>, alegando que
# isso "significantly reduces latency on long contexts".
#
# Problema: NADA no codigo alguma vez leu essas marcas. Nao havia parser, nao
# havia truncagem de historico, nao havia mecanismo nenhum -- era texto morto
# na saida do modelo. Verificado com `grep -rn "focus:\\|<global>\\|<local>"`.
#
# E pior do que inutil: a instrucao diz literalmente
#     "<local> -- you only need the immediate prior message and your current output"
#     "Use <local> by default for tool execution, code writing..."
# ou seja, pede ao modelo para AGIR como se nao tivesse acesso ao historico.
# Num agente, isso produz exactamente o sintoma de "o modelo perde-se e nao
# completa as tarefas": declara <local>, ignora o que leu e escreveu nos passos
# anteriores, e recomeca em vez de continuar.
#
# A latencia que ele dizia resolver e hoje tratada onde realmente importa: a
# cache KV do servidor, mantendo o prefixo do prompt estavel. Medido nesta
# maquina: 46,2 s na 1a chamada contra 2,2 s nas seguintes (21x), e a
# instabilidade do prefixo custava ~42 s POR TURNO.
#
# Se algum dia se quiser a funcionalidade a serio, tem de ser implementada:
# ler o scope declarado na resposta e enviar de facto um historico truncado.
# Nao basta pedi-lo no prompt.
# ---------------------------------------------------------------------------
DA_PROTOCOL = ""   # mantido vazio por compatibilidade; nao e injetado no prompt


def estimate_token_throughput(
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    elapsed_seconds: float = 0.0,
    cached_tokens: int = 0,
    timings: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Return prefill and decode throughput numbers.

    Some local servers send `prompt_per_second` / `predicted_per_second` in the
    final usage chunk, but others omit timings entirely. In that case, estimate
    throughput from the actual elapsed time so the live terminal and status panel
    keep updating during generation instead of staying at zero.
    """
    timings = timings or {}
    if not isinstance(timings, dict):
        timings = {}

    def _float(name: str, default: float = 0.0) -> float:
        try:
            value = timings.get(name)
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError, AttributeError):
            return default

    prefill = _float("prompt_per_second", 0.0)
    decode = _float("predicted_per_second", 0.0)

    # If the upstream server did not include timings, fall back to a crude but
    # useful estimate based on the last observed prompt/completion payload.
    if elapsed_seconds <= 0:
        elapsed_seconds = 0.001

    if prefill <= 0 and prompt_tokens > 0:
        effective_prompt = max(1, prompt_tokens - cached_tokens)
        prefill = effective_prompt / elapsed_seconds

    if decode <= 0 and completion_tokens > 0:
        decode = completion_tokens / elapsed_seconds

    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "cached_tokens": int(cached_tokens),
        "prefill_tps": round(max(0.0, prefill), 2),
        "decode_tps": round(max(0.0, decode), 2),
    }


def extract_text_tool_calls(text: str) -> List[Tuple[str, dict]]:
    """Extract tool calls emitted as text/tags when local models don't use structured OpenAI output."""
    if not text:
        return []

    calls = []

    # 1. Pattern: <tool_call> ... </tool_call>
    tag_matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    for block in tag_matches:
        try:
            data = json.loads(block.strip())
            name = data.get("name") or data.get("function") or data.get("tool")
            args = data.get("arguments") or data.get("parameters") or data.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            if name and isinstance(args, dict):
                calls.append((str(name), args))
        except Exception:
            pass

    # 2. Pattern: ```tool_call ... ``` or ```json ... ``` with tool format
    code_matches = re.findall(r'```(?:tool_call|json)?\s*(\{[^`]*?\})\s*```', text, re.DOTALL)
    for block in code_matches:
        try:
            data = json.loads(block.strip())
            name = data.get("name") or data.get("function") or data.get("tool")
            args = data.get("arguments") or data.get("parameters") or data.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            if name and isinstance(args, dict) and (str(name), args) not in calls:
                calls.append((str(name), args))
        except Exception:
            pass

    return calls


REASONING_PROMPTS = {
    "low": "[Reasoning Effort: LOW] Keep your internal thinking brief, highly focused, and direct. Move quickly to the solution without unnecessary elaboration.",
    "medium": "[Reasoning Effort: MEDIUM] Balance analytical depth and efficiency in your reasoning before providing the solution.",
    "high": "[Reasoning Effort: HIGH] Think carefully through the problem, analyze edge cases, validate assumptions, and prioritize technical accuracy and elegance.",
    "off": "[Reasoning Effort: OFF] Disable internal thinking trace. Do NOT output <think> tags. Provide the final response directly."
}

def message_text(content: Any) -> str:
    """
    Achata o `content` de uma mensagem em texto simples.

    Uma mensagem com imagem tem `content` como LISTA de blocos:
        [{"type": "text", "text": "..."},
         {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]

    Quem so quer o texto (contagem de tokens, arquivo para RAG, compactacao)
    tem de passar por aqui -- `str(lista)` daria a representacao Python, com o
    base64 inteiro dentro, que e lixo e enche o contexto.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        partes = []
        for bloco in content:
            if isinstance(bloco, dict):
                if bloco.get("type") == "text":
                    partes.append(str(bloco.get("text") or ""))
                elif bloco.get("type") == "image_url":
                    # Marcador curto: a imagem conta como presenca, nao como
                    # milhares de caracteres de base64.
                    partes.append("[imagem]")
            elif isinstance(bloco, str):
                partes.append(bloco)
        return "\n".join(p for p in partes if p)
    if content is None:
        return ""
    return str(content)


def detect_active_api_base(preferred: Optional[str] = None) -> str:
    """Detect if llama-server (8080) or LM Studio (1234) is currently running."""
    if preferred and "8080" not in preferred:
        return preferred
    env_base = os.environ.get("APEX_API_BASE")
    if env_base and "8080" not in env_base:
        return env_base

    # Check 8080 first, then 1234
    for port in [8080, 1234]:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models", headers={"User-Agent": "ApexHarness"})
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                if resp.status == 200:
                    return f"http://127.0.0.1:{port}/v1"
        except Exception:
            pass

    return env_base or "http://127.0.0.1:8080/v1"

class ApexAgent:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.2,
        #: Penalizacao de repeticao enviada ao llama-server.
        #:
        #: Existe por causa de um loop degenerativo real, observado em 18/09: o
        #: modelo repetia "pode ajustar o curriculo..." seguido do mesmo plano em
        #: ingles, indefinidamente, sem nunca produzir resposta.
        #:
        #: A causa era a combinacao de dois defaults:
        #:   temperature 0.2  (o harness impoe isto; o servidor usa 0.8)
        #:   repeat_penalty 1.0 = DESLIGADO, presence/frequency_penalty 0.0
        #: Com quase-greedy e zero penalizacao, a sequencia repetida tem sempre a
        #: probabilidade mais alta e nada empurra o modelo para fora do ciclo.
        #:
        #: Medido neste servidor, num prompt feito para induzir repeticao:
        #:   sem penalizacao        2 repeticoes da mesma linha
        #:   repeat_penalty=1.15    1 repeticao
        #: 1.1 e suave de proposito: castiga o suficiente para quebrar o ciclo sem
        #: estragar a repeticao legitima (codigo, nomes, listas).
        repeat_penalty: float = 1.1,
        enable_mcp: bool = True,
        reasoning_effort: str = "medium",
        # 15 era um teto baixo: tarefas reais de engenharia (ler -> editar ->
        # testar -> corrigir) rebentavam com "Limite maximo de execucao de
        # ferramentas atingido" a meio, deixando o trabalho por acabar.
        # Configuravel por APEX_MAX_TURNS.
        max_turns: Optional[int] = None,
        # Manter o conjunto de ferramentas ESTAVEL entre turnos para a cache KV
        # do servidor funcionar (ver _get_relevant_tools). Vale ~20x por turno.
        stable_tools: bool = True,
    ):
        self.base_url = detect_active_api_base(base_url)
        self.api_key = api_key or os.environ.get("APEX_API_KEY", "no-key-required")
        if not model_name:
            if "1234" in self.base_url:
                default_model = "swift-qwen3.8-27b@q4_k_m"
            elif "8080" in self.base_url:
                default_model = "llama-local-model"
            else:
                default_model = "local-model"
            self.model_name = os.environ.get("APEX_MODEL", default_model)
        else:
            self.model_name = model_name
        self._base_system_prompt = system_prompt
        self.reasoning_effort = (reasoning_effort or "medium").lower()
        self.system_prompt = self._compose_system_prompt()
        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
        self.max_turns = max_turns if max_turns is not None else int(
            os.environ.get("APEX_MAX_TURNS", "40")
        )
        self.stable_tools = stable_tools
        self._in_reasoning = False

        # ------------------------------------------------------ contagem de tokens
        # `stream_options={"include_usage": True}` faz o llama-server enviar um
        # chunk final com `choices` VAZIO e `usage`/`timings` preenchidos. O loop
        # de leitura descartava-o com `if not chunk.choices: continue`, e por isso
        # a interface nunca soube os tokens reais -- mostrava `total_chars // 4`,
        # que para portugues e para codigo erra bastante.
        #
        # Medido nesta maquina, num pedido de 8 tokens:
        #   usage.prompt_tokens                        = 19
        #   usage.completion_tokens                    =  8
        #   usage.prompt_tokens_details.cached_tokens  = 14   <- veio do KV cache
        #   model_extra['timings'] = {cache_n: 14, prompt_n: 5,
        #                             prompt_per_second: 6.78, predicted_per_second: 2.67}
        self.last_usage: Dict[str, Any] = {}
        self.session_prompt_tokens = 0        # lidos pelo modelo, somados
        self.session_completion_tokens = 0    # escritos pelo modelo, somados
        self.session_cached_tokens = 0        # servidos do KV cache

        
        # Connect timeout: 5s is ample for a local 127.0.0.1 server; 15s just delays
        # error feedback when the server isn't up yet. Read stays long for prefill.
        timeout_config = httpx.Timeout(connect=5.0, read=600.0, write=60.0, pool=60.0)
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout_config
        )
        self.history: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.tools = list(TOOLS_DEFINITION)
        self.enable_mcp = enable_mcp

        # Auto-load MCP tools if enabled
        if self.enable_mcp:
            self._init_mcp()

        set_current_agent(self)

        # Warm-up Jev in background so the first real ask_permit() hits a warm
        # model. The future is intentionally discarded — we don't care about
        # the result, only that the model loads its weights.
        try:
            _jev_warmup = get_jev().ask_permit("echo warmup")
            # Don't block: let it run in the background thread pool
        except Exception:
            pass

    def _init_mcp(self):
        """Discover and append tools from configured MCP servers."""
        try:
            from apex_harness.mcp_client import get_mcp_manager
            mcp_mgr = get_mcp_manager()
            # Discover tools from active servers
            for s_name in mcp_mgr.servers.keys():
                # Discover with short timeout to ensure fast startup
                mcp_tools = mcp_mgr.discover_tools_for_server(s_name, timeout=4.0)
                for t in mcp_tools:
                    if not any(existing["function"]["name"] == t["function"]["name"] for existing in self.tools):
                        self.tools.append(t)
        except BaseException:
            pass

    def _compose_system_prompt(self) -> str:
        # Tier-1: inject the snippet menu into the system prompt (stable, always
        # cached after the first turn). If the template has a {tools_snippets}
        # placeholder we fill it; otherwise we append the block at the end so
        # existing custom prompts that don't have the placeholder still get it.
        base = self._base_system_prompt
        if "{tools_snippets}" in base:
            base = base.format(tools_snippets=TOOLS_SNIPPETS)
        else:
            base = f"{base}\n\n{TOOLS_SNIPPETS}"
        effort_instr = REASONING_PROMPTS.get(self.reasoning_effort, "")
        if effort_instr:
            base = f"{base}\n\n{effort_instr}"
        return base

    def set_reasoning_effort(self, effort: str) -> str:
        clean = effort.lower().strip()
        if clean in ["low", "baixo", "1"]:
            self.reasoning_effort = "low"
        elif clean in ["medium", "medio", "médio", "2"]:
            self.reasoning_effort = "medium"
        elif clean in ["high", "alto", "xhigh", "3"]:
            self.reasoning_effort = "high"
        elif clean in ["off", "desligado", "none", "0"]:
            self.reasoning_effort = "off"
        else:
            return f"Nível de effort inválido: '{effort}'. Escolha: low (baixo), medium (médio), high (alto) ou off (desligado)."
        
        self.system_prompt = self._compose_system_prompt()
        if self.history and self.history[0].get("role") == "system":
            self.history[0]["content"] = self.system_prompt
        return f"Nível de raciocínio (effort) ajustado para: [bold green]{self.reasoning_effort.upper()}[/bold green]"

    def set_model(self, new_model_name: str) -> str:
        """Dynamically switch active LLM model mid-task without clearing history."""
        old_model = self.model_name
        clean_model = new_model_name.strip()
        if not clean_model:
            return f"Modelo inválido. Modelo atual permanece: '{old_model}'."
        self.model_name = clean_model
        return f"Modelo alterado com sucesso: de '{old_model}' ➔ '{self.model_name}'"

    def inject_btw(self, note: str) -> str:
        """Inject a high-priority side note (/btw) into conversation context mid-task."""
        clean_note = note.strip()
        if not clean_note:
            return "Nota lateral vazia. Nenhuma alteração feita."
        formatted_entry = f"[BY-THE-WAY / NOTA LATERAL DO USUÁRIO]: {clean_note}"
        self.history.append({"role": "user", "content": formatted_entry})
        return f"Nota lateral /btw injetada no contexto com sucesso: '{clean_note}'"

    def reset(self):
        """Reset conversation history back to the base system prompt."""
        self.history = [{"role": "system", "content": self.system_prompt}]

    def load_history(self, messages: List[Dict[str, Any]]):
        """Replace active conversation history with a provided message sequence."""
        self.history = [{"role": "system", "content": self.system_prompt}]
        for m in messages:
            role = m.get("role")
            if role in ("user", "assistant"):
                self.history.append({"role": role, "content": m.get("content", "")})

    def _rag_db_path(self) -> str:
        """Caminho da memoria vetorial de sessoes (o MESMO que o arquivo usa)."""
        from pathlib import Path as _Path
        return os.environ.get("APEX_RAG_DB", str(_Path.home() / ".apex_sessions" / "apex_rag.db"))

    def _recall_from_rag(self, query: str, top_k: int = 4, max_chars: int = 2500) -> Optional[str]:
        """
        Recupera da memoria vetorial os trechos mais relevantes para `query`.

        Isto e a metade que FALTAVA. O compact() arquivava o historico antigo na
        RAG, mas `step()` nunca a consultava -- a memoria era so de escrita, o que
        na pratica e o mesmo que nao existir: depois de uma compactacao o agente
        ficava sem qualquer acesso ao que tinha feito antes. Era esta a razao de
        "o modelo perder-se e nao completar as tarefas".

        Devolve um bloco de texto pronto a injetar no contexto, ou None.
        """
        q = (query or "").strip()
        if len(q) < 8:                      # saudacoes nao precisam de memoria
            return None
        try:
            from pathlib import Path as _Path
            db = self._rag_db_path()
            if not _Path(db).exists() or _Path(db).stat().st_size < 8192:
                return None                 # nada arquivado ainda

            from apex_harness.rag.engine import ApexRAG
            rag = ApexRAG(db_path=db)       # MESMA base que o _archive_messages_to_rag
            hits = rag.search(query=q, mode="hybrid", top_k=top_k)
            if not hits:
                return None

            blocks, total = [], 0
            for h in hits:
                txt = (h.get("content") or h.get("text") or "").strip()
                if not txt:
                    continue
                if total + len(txt) > max_chars:
                    txt = txt[: max(0, max_chars - total)]
                if not txt:
                    break
                blocks.append(txt)
                total += len(txt)
                if total >= max_chars:
                    break
            if not blocks:
                return None
            return "\n\n---\n\n".join(blocks)
        except Exception:
            # Falha de recuperacao nunca deve partir a geracao.
            return None

    def _archive_messages_to_rag(self, messages_to_archive: List[Dict[str, Any]]):
        """Archive compacted conversation messages to persistent RAG memory store."""
        try:
            from apex_harness.rag.store import VectorStore
            from apex_harness.rag.embeddings import OllamaEmbedder
            
            # CORRIGIDO: era o caminho RELATIVO "apex_rag.db", o que fazia com que
            # cada pasta de trabalho criasse a sua propria memoria vetorial -- o
            # agente "esquecia-se" ao mudar de projeto. Passa a viver num sitio
            # estavel, ao lado da base de dados de sessoes.
            from pathlib import Path as _Path
            _default_db = str(_Path.home() / ".apex_sessions" / "apex_rag.db")
            db_path = os.environ.get("APEX_RAG_DB", _default_db)
            _Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            store = VectorStore(db_path)
            
            text_blocks = []
            for msg in messages_to_archive:
                role = msg.get("role", "unknown")
                # message_text, nao str(): uma mensagem com imagem tem `content`
                # como lista, e str(lista) enfiava o base64 inteiro no RAG.
                cnt = message_text(msg.get("content"))
                if cnt and isinstance(cnt, str):
                    text_blocks.append(f"[{role.upper()}]: {cnt[:2000]}")
                    
            if not text_blocks:
                return
                
            full_text = "\n".join(text_blocks)
            embedder = OllamaEmbedder()
            embeddings = embedder.get_embeddings_batch([full_text])
            if embeddings and len(embeddings) > 0 and len(embeddings[0]) > 0:
                # CORRIGIDO: `VectorStore.add_chunks(doc_id, chunks, embeddings)` espera
                # `chunks` como lista de DICIONARIOS com a chave "text", e NAO aceita
                # um kwarg `metadatas`. O codigo antigo passava `chunks=[full_text]`
                # (lista de strings) e `metadatas=[...]`, o que levantava TypeError.
                # Como tudo isto estava dentro de `except Exception: pass`, o arquivo
                # falhava EM SILENCIO: o compact() dizia "Historico arquivado em
                # memoria RAG" enquanto gravava ZERO chunks, e as mensagens antigas
                # eram descartadas de vez. Era esta a causa do agente "perder-se".
                store.add_chunks(
                    doc_id="session_history",
                    chunks=[{
                        "text": full_text,
                        "metadata": {"source": "compact_archive", "doc_id": "session_history"},
                    }],
                    embeddings=embeddings,
                )
        except Exception as e:
            # Ja nao se engole em silencio: se o arquivo falhar, tem de se saber,
            # porque o efeito e perda definitiva de contexto.
            import sys as _sys
            print(f"[APEX] AVISO: falha ao arquivar historico na memoria RAG: "
                  f"{type(e).__name__}: {e}", file=_sys.stderr)

    def compact(self) -> str:
        """
        Compact conversation history to save context window tokens.

        Strategy (DA & RAG-aware):
          1. Truncate any single oversized message to 3000 chars.
          2. Archive old conversation messages to SQLite VectorStore for long-term memory retrieval.
          3. Build a 20-line narrative summary of the older history.
          4. Preserve the last 3 tool result messages verbatim so the agent doesn't lose current execution context.
          5. Keep the last user message so the current task is not lost.
        """
        # Step 1 — truncate individual giants
        for msg in self.history:
            cnt = message_text(msg.get("content"))
            if isinstance(cnt, str) and len(cnt) > 3000:
                msg["content"] = cnt[:3000] + "\n... [Conteúdo longo truncado para preservar contexto]"

        if len(self.history) <= 2:
            return "Histórico já está no estado inicial."

        # Step 2 — identify anchor messages to keep verbatim at the tail
        last_user = None
        for m in reversed(self.history):
            if m.get("role") == "user":
                last_user = m
                break

        # Collect the last 3 tool result messages to preserve as live context
        recent_tool_results = []
        for m in reversed(self.history[1:]):
            if m.get("role") == "tool" and len(recent_tool_results) < 3:
                recent_tool_results.insert(0, m)

        # Step 3 — summarise everything except the tail anchors and archive them to RAG
        tail_ids = {id(m) for m in recent_tool_results}
        if last_user:
            tail_ids.add(id(last_user))

        pruned_messages = [msg for msg in self.history[1:] if id(msg) not in tail_ids]
        if pruned_messages:
            self._archive_messages_to_rag(pruned_messages)

        summary_lines = []
        for msg in self.history[1:]:
            if id(msg) in tail_ids:
                continue
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content:
                summary_lines.append(f"- [{role.upper()}]: {str(content)[:150]}...")
            elif "tool_calls" in msg:
                calls = [tc["function"]["name"] for tc in msg.get("tool_calls", [])]
                summary_lines.append(f"- [TOOL CALLS]: {', '.join(calls)}")

        summary_text = "Resumo do contexto anterior compactado (detalhes arquivados em memória RAG):\n" + "\n".join(summary_lines[:20])

        # Step 4 — assemble the new, compact history
        new_hist = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": summary_text},
            {"role": "assistant", "content": "Entendido. Contexto anterior resumido e arquivado em memória RAG. Resultados de tools recentes preservados abaixo."},
        ]
        # Re-inject the preserved tool results so the agent has live execution context
        new_hist.extend(recent_tool_results)
        # Always end with the current user request
        if last_user and last_user.get("content") != summary_text:
            new_hist.append({"role": "user", "content": last_user.get("content", "")})

        self.history = new_hist
        return f"Contexto compactado com sucesso! Histórico arquivado em memória RAG e reduzido para {len(self.history)} blocos essenciais."

    def auto_compact_if_needed(self, max_history_len: int = 16, max_estimated_tokens: int = 40000):
        """Prevent memory bloat by auto-compacting if history gets too long or large in token size."""
        total_chars = sum(len(message_text(m.get("content"))) for m in self.history)
        if len(self.history) >= max_history_len or (total_chars // 3.5) >= max_estimated_tokens:
            self.compact()

    def _get_relevant_tools(self, user_text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Escolhe o conjunto de ferramentas a enviar ao modelo.

        ATENCAO -- historico desta funcao, porque a versao anterior era uma
        ANTI-OTIMIZACAO medida:

        A ideia era podar os esquemas de ferramentas por intencao, para encolher
        o prompt (~4000 tokens). O problema e que o conjunto mudava em TODOS os
        turnos -- incluindo a lista de MCP, que era sempre anexada mas cujo
        subconjunto variava. Como o prefixo do prompt (sistema + ferramentas)
        mudava, o servidor nao conseguia reutilizar a cache KV e tinha de reler
        os ~4000 tokens de raiz em cada turno.

        Medido nesta maquina, contra o LM Studio, mesmo prompt e mesmo modelo:

            prefixo ESTAVEL  1a chamada ............ 46,2 s
            prefixo ESTAVEL  2a e 3a chamadas ......  2,2 s   (21x mais rapido)
            prefixo MUDA     20 -> 16 ferramentas .. 44,4 s
            prefixo MUDA     16 -> 12 ferramentas .. 41,2 s

        Ou seja: poupar ~300 tokens de prompt custava ~42 s POR TURNO. A
        estabilidade do prefixo vale muito mais do que o tamanho do prompt,
        porque a cache torna o tamanho irrelevante a partir do 2o turno.

        Solucao: por omissao devolvemos um conjunto ESTAVEL e deterministico
        (todas as ferramentas, ordenadas por nome). O 1o turno paga o prompt
        maior; todos os seguintes sao praticamente instantaneos.

        `self.stable_tools = False` repoe o comportamento antigo, caso se queira
        comparar.
        """
        if not self.tools:
            return None

        if self.stable_tools:
            # Ordem deterministica: se a ordem mudasse, o prefixo mudava outra vez.
            return sorted(self.tools, key=lambda t: t["function"]["name"])

        text_lower = user_text.lower().strip()
        
        # Conversational / Greetings / Short conceptual queries don't need tools
        conversational_starters = {"olá", "ola", "oi", "hello", "hi", "bom dia", "boa tarde", "boa noite", "obrigado", "valeu", "valeu!", "ok", "beleza"}
        if text_lower in conversational_starters or (len(text_lower) < 15 and not any(c in text_lower for c in ["/", ".", "-", "_", "run", "cat", "ls", "git"])):
            return None

        # Categorize tools by intent
        web_keywords = ["pesquise", "search", "busque", "procure", "google", "web", "url", "http", "https", "site"]
        file_keywords = ["leia", "read", "veja", "liste", "edite", "edit", "crie", "create", "escreva", "write", "arquivo", "file", "dir", "pasta", ".py", ".md", ".json", ".txt", ".sh", ".yaml"]
        exec_keywords = ["execute", "rode", "bash", "terminal", "comando", "run", "teste", "test", "corrija", "fix", "pip", "python", "pytest", "build", "make"]
        git_keywords = ["git", "commit", "diff", "branch", "checkout", "status", "staged", "repo", "repositório"]
        session_keywords = ["session", "sessão", "sessao", "todo", "tarefa", "decisão", "decisao", "note", "anotação"]
        rag_keywords = ["rag", "memory", "memoria", "lembrar", "recuperar", "historico", "recall", "busca no banco"]
        tdp_keywords = ["tdp", "pipeline", "arquitetura", "complex task", "tarefa complexa"]
        wiki_keywords = ["wiki", "skill", "runbook", "conhecimento", "aprenda", "artigo", "documente"]

        matching_names = set()

        if any(kw in text_lower for kw in web_keywords):
            matching_names.update(["web_search", "fetch_url"])
        if any(kw in text_lower for kw in file_keywords):
            matching_names.update(["read_file", "write_file", "edit_file", "list_dir"])
        if any(kw in text_lower for kw in exec_keywords):
            matching_names.update(["bash_exec", "read_file", "list_dir"])
        if any(kw in text_lower for kw in git_keywords):
            matching_names.update(["git_status", "git_diff", "git_log", "git_branch", "git_commit"])
        if any(kw in text_lower for kw in session_keywords):
            matching_names.update(["session_log", "recall_memory"])
        if any(kw in text_lower for kw in rag_keywords):
            matching_names.update(["rag_search", "rag_ingest", "recall_memory"])
        if any(kw in text_lower for kw in tdp_keywords):
            matching_names.update(["run_tdp_pipeline"])
        if any(kw in text_lower for kw in wiki_keywords):
            matching_names.update(["consult_wiki", "record_wiki_skill"])

        # Always include custom MCP tools if registered
        for tool in self.tools:
            t_name = tool["function"]["name"]
            if t_name.startswith("mcp_"):
                matching_names.add(t_name)

        if not matching_names:
            # Fallback: if user prompt has action intent but didn't hit specific categories, return all tools
            action_triggers = ["pesquise", "search", "leia", "edite", "crie", "execute", "run", "bash", "rag", "/", ".", "diff"]
            if any(kw in text_lower for kw in action_triggers):
                return self.tools
            return None

        # Filter self.tools list to matching names
        filtered = [t for t in self.tools if t["function"]["name"] in matching_names]
        return filtered if filtered else self.tools

    def _capture_usage(self, chunk: Any, on_usage: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        """
        Le o `usage`/`timings` que o llama-server manda no chunk final.

        Tem de ser chamado ANTES do `if not chunk.choices: continue` do loop de
        leitura -- esse chunk e exactamente o que traz a contagem, e era ele que
        estava a ser descartado.
        """
        usage = getattr(chunk, "usage", None)
        if usage is None:
            return

        def _i(obj: Any, nome: str) -> int:
            try:
                return int(getattr(obj, nome, 0) or 0)
            except (TypeError, ValueError):
                return 0

        prompt = _i(usage, "prompt_tokens")
        completion = _i(usage, "completion_tokens")

        cached = 0
        det = getattr(usage, "prompt_tokens_details", None)
        if det is not None:
            cached = _i(det, "cached_tokens")

        # `timings` nao faz parte do esquema OpenAI; o SDK guarda-o em model_extra.
        timings = {}
        extra = getattr(chunk, "model_extra", None) or {}
        if isinstance(extra, dict):
            timings = extra.get("timings") or {}
        if not timings:
            timings = getattr(chunk, "timings", None) or {}

        if not cached:
            cached = _i(timings, "cache_n") if isinstance(timings, dict) else 0

        # Estimate throughput from the server timing payload when present; otherwise
        # compute it from the elapsed generation window. This keeps the live token
        # speed visible even if the upstream backend omits timings.
        elapsed_seconds = 0.0
        if isinstance(timings, dict):
            for key in ("elapsed_time", "total_time", "time_seconds"):
                value = timings.get(key)
                if value is not None:
                    try:
                        elapsed_seconds = max(float(value), 0.0)
                        break
                    except (TypeError, ValueError):
                        pass

        info = estimate_token_throughput(
            prompt_tokens=prompt,
            completion_tokens=completion,
            elapsed_seconds=elapsed_seconds or 1.0,
            cached_tokens=cached,
            timings=timings if isinstance(timings, dict) else {},
        )

        # Prompt_n = tokens que tiveram mesmo de ser processados agora; cache_n =
        # tokens reaproveitados do KV cache. A soma da o prompt todo.
        info["new_prompt_tokens"] = _i(timings, "prompt_n") if isinstance(timings, dict) else max(0, prompt - cached)

        self.last_usage = info
        self.session_prompt_tokens += prompt
        self.session_completion_tokens += completion
        self.session_cached_tokens += cached

        if on_usage:
            try:
                on_usage(dict(info))
            except Exception:
                pass

    def step(
        self,
        user_input: str,
        on_tool_start: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_tool_finish: Optional[Callable[[str, str], None]] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        on_usage: Optional[Callable[[Dict[str, Any]], None]] = None,
        images: Optional[List[str]] = None
    ) -> str:
        self._in_reasoning = False

        # Recuperar da memoria vetorial o que foi arquivado em compactacoes
        # anteriores. Inserido ANTES da mensagem do utilizador, para garantir
        # que a conversa termina sempre com role: "user" (evita quebrar templates Jinja/OpenAI).
        recalled = self._recall_from_rag(user_input)
        if recalled:
            self.history.append({
                "role": "system",
                "content": ("[MEMORIA DE SESSOES ANTERIORES - trechos recuperados do "
                            "historico arquivado. Usa-os se forem relevantes para a "
                            "tarefa atual; se nao forem, ignora-os.]\n\n" + recalled),
            })

        # Com imagens, o `content` passa a ser uma lista de blocos -- e o formato
        # multimodal que o llama-server (mtmd) entende. Sem imagens fica string,
        # para nao mexer em nada do que ja funciona.
        if images:
            blocos: List[Dict[str, Any]] = []
            if user_input:
                blocos.append({"type": "text", "text": user_input})
            for img in images:
                blocos.append({"type": "image_url", "image_url": {"url": img}})
            self.history.append({"role": "user", "content": blocos})
        else:
            self.history.append({"role": "user", "content": user_input})
        self.auto_compact_if_needed()

        # Tier-2 tool routing: launch Jev ask_tools in parallel NOW while we
        # set up the rest of the turn. We resolve it just before the first LLM
        # call — giving Jev the full turn latency to answer without adding
        # any serial delay.
        _tool_names = [t["function"]["name"] for t in self.tools]
        if self.stable_tools:
            # stable_tools=True: send all tools sorted (KV cache stability)
            active_tools = self._get_relevant_tools(user_input)
            _jev_tools_future = None
        else:
            # stable_tools=False: use Jev to pick the relevant subset
            try:
                _jev_tools_future = get_jev().ask_tools(user_input, _tool_names)
            except Exception:
                _jev_tools_future = None
            active_tools = self._get_relevant_tools(user_input)  # keyword fallback

        turns = 0

        # Resolve Jev tool-routing answer (it had the full setup time to run)
        if not self.stable_tools and _jev_tools_future is not None:
            try:
                jev_names = _jev_tools_future.result(timeout=2.0)
                if jev_names:
                    from apex_harness.tools import get_tool_schemas_for
                    active_tools = get_tool_schemas_for(jev_names, self.tools)
            except Exception:
                pass  # keep keyword fallback

        while turns < self.max_turns:
            turns += 1
            request_kwargs = {
                "model": self.model_name,
                "messages": self.history,
                "temperature": self.temperature,
                "stream": True,
                # Sem isto o servidor nao manda contagem nenhuma. Ver _capture_usage.
                "stream_options": {"include_usage": True},
            }
            # `repeat_penalty` nao faz parte do esquema OpenAI -- e uma extensao do
            # llama-server, portanto vai em `extra_body`. Ver a nota no __init__
            # sobre o loop degenerativo que isto resolve.
            if self.repeat_penalty and self.repeat_penalty > 1.0:
                request_kwargs["extra_body"] = {"repeat_penalty": self.repeat_penalty}
            if active_tools:
                request_kwargs["tools"] = active_tools
                request_kwargs["tool_choice"] = "auto"

            try:
                stream = self.client.chat.completions.create(**request_kwargs)
            except Exception as e:
                err_str = str(e).lower()

                # Um backend que nao conheca `stream_options` recusaria o pedido
                # TODO. A contagem de tokens e um extra, nao pode custar a
                # resposta: tira-se o campo e tenta-se de novo, uma vez.
                if "stream_options" in err_str or "include_usage" in err_str:
                    request_kwargs.pop("stream_options", None)
                    stream = self.client.chat.completions.create(**request_kwargs)
                    err_str = ""

                # Mesma logica para o `repeat_penalty`: e uma extensao do
                # llama-server. Num backend que a recuse, cai-se para o pedido
                # sem ela em vez de perder a resposta.
                if err_str and ("repeat_penalty" in err_str or "extra_body" in err_str):
                    request_kwargs.pop("extra_body", None)
                    stream = self.client.chat.completions.create(**request_kwargs)
                    err_str = ""

                # Se excedeu o tamanho de contexto disponível no servidor, compacta e tenta de novo
                if "exceeds the available context size" in err_str or ("context size" in err_str and "exceed" in err_str):
                    self.compact()
                    try:
                        if active_tools:
                            request_kwargs["messages"] = self.history
                            stream = self.client.chat.completions.create(**request_kwargs)
                        else:
                            stream = self.client.chat.completions.create(
                                model=self.model_name,
                                messages=self.history,
                                temperature=self.temperature,
                                stream=True
                            )
                    except Exception as retry_err:
                        if len(self.history) > 2:
                            self.history = [self.history[0], self.history[-1]]
                            try:
                                stream = self.client.chat.completions.create(
                                    model=self.model_name,
                                    messages=self.history,
                                    tools=self.tools,
                                    tool_choice="auto",
                                    temperature=self.temperature,
                                    stream=True
                                )
                            except Exception as final_err:
                                return f"⚠️ Limite de contexto do servidor excedido ({final_err}). Use /clear para reiniciar."
                        else:
                            return f"⚠️ Limite de contexto do servidor excedido ({retry_err}). Use /clear para reiniciar."
                else:
                    # Fallback to non-streaming if server rejects streaming tools
                    try:
                        resp = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=self.history,
                            temperature=self.temperature
                        )
                        content = resp.choices[0].message.content or ""
                        if not content.strip():
                            if self.history and self.history[-1].get("role") == "user":
                                self.history.pop()
                            return (
                                "⚠️  O modelo retornou uma resposta vazia (fallback). "
                                "Tente reformular ou use /compact para liberar contexto."
                            )
                        self.history.append({"role": "assistant", "content": content})
                        if on_chunk:
                            on_chunk(content)
                        return content
                    except Exception as inner_err:
                        err_msg = f"API Error: Failed to communicate with model at {self.base_url}: {str(e)}"
                        return err_msg

            accumulated_content = []
            # Guardado a parte: o raciocinio NAO entra no `full_text` normal (nao
            # e resposta), mas tem de sobreviver para o caso de o modelo nao
            # chegar a produzir resposta nenhuma. Ver o fim do loop.
            accumulated_reasoning = []
            tool_calls_map = {}

            for chunk in stream:
                # ANTES do `continue`: e este o chunk que traz `usage` e `timings`
                # (vem com `choices` vazio). Descartado aqui, a contagem de tokens
                # nunca chegava a interface.
                self._capture_usage(chunk, on_usage)

                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # Reasoning content support (DeepSeek / llama-server)
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    if not self._in_reasoning:
                        self._in_reasoning = True
                        if on_chunk:
                            on_chunk("<think>")
                    accumulated_reasoning.append(reasoning)
                    if on_chunk:
                        on_chunk(reasoning)
                
                # Stream assistant text chunks
                if delta.content:
                    if self._in_reasoning:
                        self._in_reasoning = False
                        if on_chunk:
                            on_chunk("</think>")
                    accumulated_content.append(delta.content)
                    if on_chunk:
                        on_chunk(delta.content)

                # Collect streamed tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index if tc.index is not None else 0
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                "id": tc.id or f"call_{idx}",
                                "name": tc.function.name or "",
                                "arguments": tc.function.arguments or ""
                            }
                        else:
                            if tc.id and not tool_calls_map[idx]["id"]:
                                tool_calls_map[idx]["id"] = tc.id
                            if tc.function.name:
                                tool_calls_map[idx]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls_map[idx]["arguments"] += tc.function.arguments

            if self._in_reasoning:
                self._in_reasoning = False
                if on_chunk:
                    on_chunk("</think>")
            full_text = "".join(accumulated_content)
            reasoning_text = "".join(accumulated_reasoning).strip()

            # Se o modelo produziu SO raciocinio, o `full_text` fica vazio.
            #
            # Um modelo de "thinking" servido pelo llama-server manda o pensamento
            # em `delta.reasoning_content` e a resposta final em `delta.content`.
            # Quando o modelo termina sem chegar a escrever `content` -- porque
            # bateu num stop, porque se enganou no formato, ou porque gastou o
            # orcamento de tokens a pensar -- o acumulador fica vazio.
            #
            # Antes isto deitava fora o raciocinio TODO e devolvia "resposta
            # vazia": o utilizador perdia o trabalho e o turno morria, sem
            # sequer ficar no historico. Agora o raciocinio e aproveitado como
            # resposta, que e o melhor que ha para mostrar.
            if not full_text.strip() and reasoning_text and not tool_calls_map:
                full_text = f"<think>\n{reasoning_text}\n</think>"

            # If no structured tool calls were emitted, check for text-formatted tool calls
            if not tool_calls_map and full_text:
                fallback_calls = extract_text_tool_calls(full_text)
                if fallback_calls:
                    for i, (fname, fargs) in enumerate(fallback_calls):
                        tool_calls_map[i] = {
                            "id": f"call_text_{turns}_{i}",
                            "name": fname,
                            "arguments": json.dumps(fargs)
                        }

            # If no tools were invoked at all, this turn is completed
            if not tool_calls_map:
                if not full_text.strip():
                    if self.history and self.history[-1].get("role") == "user":
                        self.history.pop()
                    return (
                        "⚠️  O modelo retornou uma resposta vazia (empty output). "
                        "Isso pode acontecer quando o contexto está cheio ou o modelo "
                        "teve dificuldade com a requisição. "
                        "Tente reformular sua pergunta ou use /compact para liberar contexto."
                    )
                self.history.append({"role": "assistant", "content": full_text})
                return full_text

            # Model requested tool calls
            parsed_tool_calls = list(tool_calls_map.values())
            assistant_msg = {
                "role": "assistant",
                "content": full_text or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"]
                        }
                    }
                    for tc in parsed_tool_calls
                ]
            }
            self.history.append(assistant_msg)

            for tc in parsed_tool_calls:
                name = tc["name"]
                raw_args = tc["arguments"]
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except Exception:
                    args = {}

                if on_tool_start:
                    on_tool_start(name, args)

                # Jev permission guard for bash_exec
                # Runs synchronously (fast — Jev already warm) before execution.
                # Behaviour:
                #   allow → execute normally
                #   ask   → proceed (interactive approval is out of scope here;
                #            the on_tool_start callback can surface this to UI)
                #   deny  → block and return a refusal string
                if name == "bash_exec":
                    cmd = args.get("command", "")
                    try:
                        permit_future = get_jev().ask_permit(cmd)
                        permit: JevPermit = permit_future.result(timeout=6.0)
                        if permit.verdict == "deny":
                            block_msg = (
                                f"[Jev DENY] Comando bloqueado por política de segurança.\n"
                                f"Razão: {permit.reason}\n"
                                f"Comando: {cmd[:200]}"
                            )
                            if on_tool_finish:
                                on_tool_finish(name, block_msg)
                            self.history.append({
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "name": name,
                                "content": block_msg,
                            })
                            continue
                    except Exception:
                        pass  # Jev unavailable → proceed normally

                set_current_agent(self)
                result_str = execute_tool(name, args)
                # Tier-2 / Visibility Ladder: compress verbose tool outputs
                # before they enter the history (and thus get re-sent every turn).
                # compress_tool_output applies per-tool line limits:
                #   bash_exec → 80 lines, read_file → 120 lines, etc.
                # A truncation marker is appended so the model can ask for more.
                result_str = compress_tool_output(name, result_str)

                if on_tool_finish:
                    on_tool_finish(name, result_str)

                self.history.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": name,
                    "content": result_str
                })

        timeout_msg = "Limite máximo de execução de ferramentas atingido (15 turnos)."
        self.history.append({"role": "assistant", "content": timeout_msg})
        return timeout_msg
