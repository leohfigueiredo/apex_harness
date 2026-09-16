import os
import re
import json
import httpx
from typing import List, Dict, Any, Callable, Optional, Tuple
from openai import OpenAI
from apex_harness.tools import TOOLS_DEFINITION, execute_tool, set_current_agent

DEFAULT_SYSTEM_PROMPT = """You are Apex Harness, an autonomous, highly capable AI engineer and researcher running locally on an AMD Ryzen AI 9 HX 370 (Zen 5, 24 threads) with Radeon 890M GPU and 96GB Unified Memory.

Capabilities & Rules:
1. LIVE INTERNET ACCESS: You have active, real-time web search capabilities through `web_search` and `fetch_url`. You MUST NEVER claim you cannot access the internet or that your knowledge is cut off. If the user asks about recent topics, external documentation, or repos, actively call `web_search` and fetch information.
2. OPERATING SYSTEM & CODING ACCESS: You have full access to inspect files (`list_dir`, `read_file`), make changes (`write_file`, `edit_file`), and execute terminal commands (`bash_exec`).
3. HARDWARE AWARENESS: You are running on high-end hardware with 96GB RAM and ROCm/Vulkan acceleration. Be fast, precise, and practical.
4. RIGOROUS EXECUTION: Always inspect files and verify directory contents before modifying them. When writing or editing code, ensure clean syntax and test your work with `bash_exec` whenever appropriate.

Declarative Attention Protocol (DA):
To maximise inference speed and reduce KV-cache pressure, declare your attention scope at the start of each reasoning step:
- <global> — you need to reference earlier context (full attention required)
- <focus:N> — you need only the last N messages (e.g. <focus:3> for the last 3 turns)
- <local> — you only need the immediate prior message and your current output
Use <local> by default for tool execution, code writing, and step-by-step tasks where prior conversation is not needed. Use <global> only when you must recall something from early in the session. This significantly reduces latency on long contexts.
"""

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

class ApexAgent:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.2,
        max_turns: int = 15,
        enable_mcp: bool = True,
        reasoning_effort: str = "medium"
    ):
        self.base_url = base_url or os.environ.get("APEX_API_BASE", "http://127.0.0.1:8080/v1")
        self.api_key = api_key or os.environ.get("APEX_API_KEY", "no-key-required")
        default_model = "llama-local-model" if "8080" in self.base_url else "local-model"
        self.model_name = model_name or os.environ.get("APEX_MODEL", default_model)
        self._base_system_prompt = system_prompt
        self.reasoning_effort = (reasoning_effort or "medium").lower()
        self.system_prompt = self._compose_system_prompt()
        self.temperature = temperature
        self.max_turns = max_turns
        self._in_reasoning = False
        
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
        except Exception:
            pass

    def _compose_system_prompt(self) -> str:
        prompt = self._base_system_prompt
        effort_instr = REASONING_PROMPTS.get(self.reasoning_effort, "")
        if effort_instr:
            prompt = f"{prompt}\n\n{effort_instr}"
        return prompt

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

    def reset(self):
        """Reset conversation history back to the base system prompt."""
        self.history = [{"role": "system", "content": self.system_prompt}]

    def _archive_messages_to_rag(self, messages_to_archive: List[Dict[str, Any]]):
        """Archive compacted conversation messages to persistent RAG memory store."""
        try:
            from apex_harness.rag.store import VectorStore
            from apex_harness.rag.embeddings import OllamaEmbedder
            
            db_path = os.environ.get("APEX_RAG_DB", "apex_rag.db")
            store = VectorStore(db_path)
            
            text_blocks = []
            for msg in messages_to_archive:
                role = msg.get("role", "unknown")
                cnt = msg.get("content") or ""
                if cnt and isinstance(cnt, str):
                    text_blocks.append(f"[{role.upper()}]: {cnt[:2000]}")
                    
            if not text_blocks:
                return
                
            full_text = "\n".join(text_blocks)
            embedder = OllamaEmbedder()
            embeddings = embedder.get_embeddings_batch([full_text])
            if embeddings and len(embeddings) > 0 and len(embeddings[0]) > 0:
                store.add_chunks(
                    doc_id="session_history",
                    chunks=[full_text],
                    embeddings=embeddings,
                    metadatas=[{"source": "compact_archive", "doc_id": "session_history"}]
                )
        except Exception:
            pass

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
            cnt = msg.get("content")
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
        total_chars = sum(len(str(m.get("content") or "")) for m in self.history)
        if len(self.history) >= max_history_len or (total_chars // 3.5) >= max_estimated_tokens:
            self.compact()

    def _get_relevant_tools(self, user_text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Pi-Harness Optimization: Sub-set Tool Pruning & Category Scoping.
        Instead of sending all tool schemas (which inflates the KV cache by ~4000 tokens),
        we dynamically filter to only the tool subset required for the user's intent.
        """
        if not self.tools:
            return None
        
        text_lower = user_text.lower().strip()
        
        # Conversational / Greetings / Short conceptual queries don't need tools
        conversational_starters = {"olá", "ola", "oi", "hello", "hi", "bom dia", "boa tarde", "boa noite", "obrigado", "valeu", "valeu!", "ok", "beleza"}
        if text_lower in conversational_starters or (len(text_lower) < 15 and not any(c in text_lower for c in ["/", ".", "-", "_", "run", "cat", "ls", "git"])):
            return None

        # Categorize tools by intent
        web_keywords = ["pesquise", "search", "busque", "procure", "google", "web", "url", "http", "https", "site"]
        file_keywords = ["leia", "read", "veja", "liste", "edite", "edit", "crie", "create", "escreva", "write", "arquivo", "file", "dir", "pasta", ".py", ".md", ".json", ".txt", ".sh", ".yaml"]
        exec_keywords = ["execute", "rode", "bash", "terminal", "comando", "run", "teste", "test", "corrija", "fix", "git", "pip", "python", "pytest", "build", "make"]
        rag_keywords = ["rag", "memory", "memoria", "lembrar", "recuperar", "historico", "recall", "busca no banco"]
        tdp_keywords = ["tdp", "pipeline", "arquitetura", "complex task", "tarefa complexa"]

        matching_names = set()

        if any(kw in text_lower for kw in web_keywords):
            matching_names.update(["web_search", "fetch_url"])
        if any(kw in text_lower for kw in file_keywords):
            matching_names.update(["read_file", "write_file", "edit_file", "list_dir"])
        if any(kw in text_lower for kw in exec_keywords):
            matching_names.update(["bash_exec", "read_file", "list_dir"])
        if any(kw in text_lower for kw in rag_keywords):
            matching_names.update(["rag_search", "rag_ingest", "recall_memory"])
        if any(kw in text_lower for kw in tdp_keywords):
            matching_names.update(["run_tdp_pipeline"])

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

    def step(
        self,
        user_input: str,
        on_tool_start: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_tool_finish: Optional[Callable[[str, str], None]] = None,
        on_chunk: Optional[Callable[[str], None]] = None
    ) -> str:
        """Run a complete multi-turn tool resolution cycle with real-time token streaming and text-fallback tool calling."""
        self.history.append({"role": "user", "content": user_input})
        self.auto_compact_if_needed()
        
        # Pi-Harness: Sub-set tool pruning to accelerate prefill and maximize KV-cache hits
        active_tools = self._get_relevant_tools(user_input)
        
        turns = 0

        while turns < self.max_turns:
            turns += 1
            request_kwargs = {
                "model": self.model_name,
                "messages": self.history,
                "temperature": self.temperature,
                "stream": True
            }
            if active_tools:
                request_kwargs["tools"] = active_tools
                request_kwargs["tool_choice"] = "auto"

            try:
                stream = self.client.chat.completions.create(**request_kwargs)
            except Exception as e:
                err_str = str(e).lower()
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
            tool_calls_map = {}

            for chunk in stream:
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

                set_current_agent(self)
                result_str = execute_tool(name, args)
                # Pi-Harness: Head & Tail Scratchpad Truncation to protect KV-cache bandwidth while preserving stack traces
                max_chars = 12000
                if len(result_str) > max_chars:
                    head_len = 5000
                    tail_len = 5000
                    truncated_count = len(result_str) - (head_len + tail_len)
                    result_str = (
                        result_str[:head_len] +
                        f"\n\n... [{truncated_count} caracteres intermediários truncados pelo Apex Pi-Optimizer para preservar KV Cache] ...\n\n" +
                        result_str[-tail_len:]
                    )

                if on_tool_finish:
                    on_tool_finish(name, result_str)

                self.history.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": name,
                    "content": result_str
                })

        return "Limite máximo de execução de ferramentas atingido (15 turnos)."
