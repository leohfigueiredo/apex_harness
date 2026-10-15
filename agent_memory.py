"""
Apex Harness — Agent Memory (agent_memory.py)

Porta para Python das duas ideias centrais do TencentDB-Agent-Memory
(https://github.com/TencentCloud/TencentDB-Agent-Memory), adaptada a um harness
LOCAL onde a cache KV do servidor vale mais do que o tamanho do prompt.

1. MEMORIA SIMBOLICA DE CURTO PRAZO (ContextOffloader)
   Cada resultado de ferramenta e gravado INTEGRALMENTE em disco
   (refs/<node_id>.md) com um passo resumido em steps.jsonl. Quando o contexto
   enche, os resultados ANTIGOS no historico sao trocados por um stub de uma
   linha com o node_id, e o estado da tarefa passa a ser um canvas Mermaid
   compacto. O modelo recupera o texto bruto com memory_drilldown(node_id).

       output gigante ──► refs/N7.md (sem perda)
                     └──► stub "[offload N7 · bash_exec · ✗] ..." no historico
                     └──► no N7 no canvas Mermaid

2. MEMORIA DE LONGO PRAZO EM CAMADAS (LongTermMemory)
       L0 Conversation  turnos crus                      (SQLite)
       L1 Atom          factos atomicos + embedding       (SQLite + FTS5)
       L2 Scenario      um Markdown por categoria          (scenarios/*.md)
       L3 Persona       perfil consolidado                 (persona.md)
   Recall hibrido: BM25 (FTS5) + vetores (Ollama nomic-embed-text) fundidos
   por Reciprocal Rank Fusion. Sem Ollama, cai para BM25 sozinho.

Diferencas deliberadas em relacao ao original (e porque):
  * O offload NAO corre a cada turno. Trocar mensagens antigas muda o prefixo
    do prompt e invalida a cache KV (medido aqui: 2 s -> 45 s por turno). So
    corre quando o contexto passa do limiar, tal como o compact() ja fazia.
  * A extracao automatica de L1 por LLM so corre se houver um endpoint
    DEDICADO (APEX_MEMORY_LLM_URL, ou a NPU com o Lemonade ativo). Usar o
    mesmo llama-server do agente expulsaria a conversa da cache KV. Sem
    endpoint dedicado, o proprio agente grava factos com memory_remember().
  * Tudo o que e caro e opcional e falha em silencio: memoria nunca parte a
    geracao.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

LLMFn = Callable[[str], str]

ATOM_CATEGORIES = ("preference", "project", "environment", "workflow", "gotcha", "general")
OFFLOAD_PREFIX = "[offload "


def memory_base_dir() -> Path:
    """Raiz da memoria. Segue APEX_SESSION_DIR para os testes ficarem hermeticos."""
    base = (os.environ.get("APEX_MEMORY_DIR")
            or os.environ.get("APEX_SESSION_DIR")
            or str(Path.home() / ".apex_sessions"))
    p = Path(base).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _mmd_escape(text: str, limit: int = 48) -> str:
    """Rotulo seguro para Mermaid: sem aspas, colchetes, chavetas nem quebras."""
    t = re.sub(r"[\"'`\[\]{}()<>|#;]", " ", text or "")
    t = re.sub(r"\s+", " ", t).strip()
    return (t[: limit - 1] + "…") if len(t) > limit else t


# =============================================================================
# 1. Curto prazo: offload simbolico
# =============================================================================

_ERROR_MARKERS = re.compile(
    r"(Traceback \(most recent call last\)|\bError\b|\bERROR\b|\bFAILED\b|"
    r"Exit Code: [1-9]|command not found|No such file|Permission denied|\[Jev DENY\])"
)


@dataclass
class OffloadStep:
    node_id: str
    tool: str
    args: str
    ok: bool
    summary: str
    lines: int
    chars: int
    ref: str
    ts: float

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


class ContextOffloader:
    """Grava cada resultado de ferramenta em disco e mantem o canvas Mermaid."""

    def __init__(self, session_id: str, base_dir: Optional[Path] = None):
        safe = re.sub(r"[^\w.-]", "_", session_id or "default")[:80]
        self.session_id = safe
        self.dir = (base_dir or memory_base_dir()) / "offload" / safe
        self.refs_dir = self.dir / "refs"
        self.refs_dir.mkdir(parents=True, exist_ok=True)
        self.steps_path = self.dir / "steps.jsonl"
        self.canvas_path = self.dir / "canvas.mmd"
        self._lock = threading.Lock()
        self.steps: List[OffloadStep] = self._load_steps()
        self.by_call_id: Dict[str, str] = {}

    # ------------------------------------------------------------------ io
    def _load_steps(self) -> List[OffloadStep]:
        out: List[OffloadStep] = []
        if self.steps_path.exists():
            for line in self.steps_path.read_text(encoding="utf-8").splitlines():
                try:
                    out.append(OffloadStep(**json.loads(line)))
                except Exception:
                    continue
        return out

    @staticmethod
    def _summarize(output: str) -> str:
        """Resumo deterministico (sem LLM): 1a linha de erro, senao 1a linha util."""
        lines = [l.strip() for l in (output or "").splitlines() if l.strip()]
        if not lines:
            return "(sem output)"
        for l in lines:
            if _ERROR_MARKERS.search(l):
                return l[:160]
        for l in lines:
            if not l.startswith(("STDOUT:", "STDERR:", "---", "===")):
                return l[:160]
        return lines[0][:160]

    # ---------------------------------------------------------------- api
    def record(self, tool: str, args: Any, raw_output: str,
               tool_call_id: Optional[str] = None) -> str:
        """Guarda o output COMPLETO (antes de qualquer truncagem). Devolve o node_id."""
        raw_output = raw_output if isinstance(raw_output, str) else str(raw_output)
        args_repr = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False, default=str)
        with self._lock:
            node_id = f"N{len(self.steps) + 1}"
            ref = self.refs_dir / f"{node_id}.md"
            ref.write_text(
                f"# {node_id} · {tool}\n\n## args\n```\n{args_repr[:4000]}\n```\n\n"
                f"## output\n```\n{raw_output}\n```\n",
                encoding="utf-8",
            )
            step = OffloadStep(
                node_id=node_id, tool=tool, args=args_repr[:200],
                ok=not bool(_ERROR_MARKERS.search(raw_output[:20000])),
                summary=self._summarize(raw_output),
                lines=raw_output.count("\n") + 1, chars=len(raw_output),
                ref=str(ref), ts=time.time(),
            )
            self.steps.append(step)
            with self.steps_path.open("a", encoding="utf-8") as f:
                f.write(step.to_json() + "\n")
            if tool_call_id:
                self.by_call_id[tool_call_id] = node_id
            self.canvas_path.write_text(self.canvas(), encoding="utf-8")
        return node_id

    def get(self, node_id: str) -> Optional[OffloadStep]:
        nid = (node_id or "").strip().upper()
        return next((s for s in self.steps if s.node_id == nid), None)

    def stub(self, node_id: str) -> str:
        s = self.get(node_id)
        if not s:
            return f"{OFFLOAD_PREFIX}{node_id}] (passo desconhecido)"
        mark = "✓" if s.ok else "✗"
        return (f"{OFFLOAD_PREFIX}{s.node_id} · {s.tool} · {mark}] {s.summary} "
                f"({s.lines} linhas offloaded — memory_drilldown('{s.node_id}') devolve o texto completo)")

    def canvas(self, max_nodes: int = 40) -> str:
        """Canvas Mermaid da tarefa: um no por passo, ligados em sequencia."""
        steps = self.steps[-max_nodes:]
        if not steps:
            return "graph LR\n    START((inicio))"
        out = ["graph LR"]
        if len(self.steps) > max_nodes:
            out.append(f'    OLD["… {len(self.steps) - max_nodes} passos anteriores"]')
            prev = "OLD"
        else:
            out.append("    START((inicio))")
            prev = "START"
        for s in steps:
            mark = "ok" if s.ok else "ERRO"
            label = _mmd_escape(f"{s.node_id} {s.tool} {mark}: {s.summary}")
            out.append(f'    {s.node_id}["{label}"]')
            out.append(f"    {prev} --> {s.node_id}")
            prev = s.node_id
        failed = [s.node_id for s in steps if not s.ok]
        if failed:
            out.append("    classDef err fill:#5b1a1a,stroke:#ff6b6b")
            out.append(f"    class {','.join(failed)} err")
        return "\n".join(out)

    def drilldown(self, node_id: str, start: int = 0, max_chars: int = 12000) -> str:
        s = self.get(node_id)
        if not s:
            known = ", ".join(x.node_id for x in self.steps[-10:]) or "nenhum"
            return f"node_id '{node_id}' nao existe nesta sessao. Ultimos: {known}"
        try:
            text = Path(s.ref).read_text(encoding="utf-8")
        except Exception as e:
            return f"Falha a ler {s.ref}: {e}"
        start = max(0, int(start or 0))
        chunk = text[start:start + int(max_chars or 12000)]
        rest = len(text) - (start + len(chunk))
        if rest > 0:
            chunk += f"\n… [{rest} caracteres restantes — memory_drilldown('{s.node_id}', start={start + len(chunk)})]"
        return chunk

    def offload_history(self, history: List[Dict[str, Any]], keep_last: int = 3,
                        min_chars: int = 600) -> int:
        """
        Troca o content de resultados de ferramenta ANTIGOS por stubs (in-place).
        Mantem os `keep_last` mais recentes intactos. Devolve quantos trocou.
        """
        tool_idx = [i for i, m in enumerate(history) if m.get("role") == "tool"]
        candidates = tool_idx[:-keep_last] if keep_last > 0 else tool_idx
        changed = 0
        for i in candidates:
            m = history[i]
            content = m.get("content")
            if not isinstance(content, str) or content.startswith(OFFLOAD_PREFIX):
                continue
            if len(content) < min_chars:
                continue
            nid = self.by_call_id.get(m.get("tool_call_id") or "")
            if not nid:
                # Resultado que nao passou pelo record() (ex.: sessao carregada):
                # grava-o agora para nao perder nada.
                nid = self.record(m.get("name") or "tool", "(desconhecido)", content,
                                  tool_call_id=m.get("tool_call_id"))
            m["content"] = self.stub(nid)
            changed += 1
        return changed


# =============================================================================
# 2. Longo prazo: L0 -> L3
# =============================================================================

_EXTRACT_PROMPT = """Extrai factos DURADOUROS sobre o utilizador, o ambiente e os projetos desta conversa.
Ignora detalhes efemeros (outputs de comandos, erros ja resolvidos, cumprimentos).
Cada facto: uma frase curta, autonoma, verificavel. Maximo {max_n} factos.
Categorias: preference | project | environment | workflow | gotcha | general

Responde APENAS com JSON: [{{"fact": "...", "category": "..."}}]
Se nao houver nada duradouro, responde [].

CONVERSA:
{conversation}
"""

_PERSONA_PROMPT = """Com base nestes factos sobre o utilizador e o seu trabalho, escreve um perfil
conciso em Markdown (maximo 20 linhas) com secoes: Preferencias, Ambiente, Projetos ativos,
Regras de trabalho. Nao inventes nada que nao esteja nos factos.

FACTOS:
{facts}
"""


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"\w+", (text or "").lower()) if len(t) > 1]


class LongTermMemory:
    def __init__(self, base_dir: Optional[Path] = None,
                 embedder: Optional[Any] = "auto",
                 extract_every: int = 5,
                 persona_every: int = 50):
        self.base = (base_dir or memory_base_dir())
        self.mem_dir = self.base / "memory"
        self.scen_dir = self.mem_dir / "scenarios"
        self.scen_dir.mkdir(parents=True, exist_ok=True)
        self.persona_path = self.mem_dir / "persona.md"
        self.db_path = self.base / "agent_memory.db"
        self.extract_every = max(1, int(os.environ.get("APEX_MEMORY_EXTRACT_EVERY", extract_every)))
        self.persona_every = max(1, int(persona_every))
        self._lock = threading.RLock()
        self._embedder = self._make_embedder() if embedder == "auto" else embedder
        self._embed_ok = self._embedder is not None
        self._init_db()

    # ----------------------------------------------------------------- db
    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.db_path), timeout=10.0, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL;")
        return c

    def _init_db(self) -> None:
        with self._lock, self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS l0_turns(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT, role TEXT, content TEXT, ts REAL,
                    extracted INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS l1_atoms(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL, category TEXT, source TEXT,
                    hash TEXT UNIQUE, embedding TEXT, ts REAL);
                CREATE VIRTUAL TABLE IF NOT EXISTS l1_fts USING fts5(
                    content, category, content='l1_atoms', content_rowid='id');
                CREATE TRIGGER IF NOT EXISTS l1_ai AFTER INSERT ON l1_atoms BEGIN
                    INSERT INTO l1_fts(rowid, content, category)
                    VALUES (new.id, new.content, new.category); END;
                CREATE TRIGGER IF NOT EXISTS l1_ad AFTER DELETE ON l1_atoms BEGIN
                    INSERT INTO l1_fts(l1_fts, rowid, content, category)
                    VALUES ('delete', old.id, old.content, old.category); END;
                CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
            """)

    def _meta(self, k: str, default: str = "0") -> str:
        with self._conn() as c:
            r = c.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
            return r["v"] if r else default

    def _set_meta(self, k: str, v: str) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO meta(k, v) VALUES (?, ?)", (k, v))

    # ---------------------------------------------------------- embeddings
    @staticmethod
    def _make_embedder() -> Optional[Any]:
        if os.environ.get("APEX_MEMORY_EMBED", "1").lower() in ("0", "false", "no"):
            return None
        try:
            from apex_harness.rag.embeddings import OllamaEmbedder
            return OllamaEmbedder()
        except Exception:
            return None

    def _embed(self, text: str) -> Optional[List[float]]:
        if not self._embed_ok:
            return None
        try:
            vec = self._embedder.get_embedding(text)
            return vec or None
        except Exception:
            # Ollama em baixo: desliga vetores nesta instancia em vez de pagar
            # o timeout em cada pedido. O BM25 continua a funcionar.
            self._embed_ok = False
            return None

    @staticmethod
    def _cos(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    # ------------------------------------------------------------------ L0
    def add_turn(self, session_id: str, role: str, content: str) -> None:
        text = (content or "").strip()
        if not text:
            return
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO l0_turns(session_id, role, content, ts) VALUES (?,?,?,?)",
                      (session_id, role, text[:8000], time.time()))

    def pending_turns(self, limit: int = 40) -> List[sqlite3.Row]:
        with self._conn() as c:
            return c.execute(
                "SELECT * FROM l0_turns WHERE extracted=0 ORDER BY id LIMIT ?", (limit,)
            ).fetchall()

    def search_conversations(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        toks = _tokenize(query)[:8]
        if not toks:
            return []
        where = " OR ".join("lower(content) LIKE ?" for _ in toks)
        with self._conn() as c:
            rows = c.execute(
                f"SELECT session_id, role, content, ts FROM l0_turns WHERE {where} "
                f"ORDER BY id DESC LIMIT ?", [f"%{t}%" for t in toks] + [top_k]
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ L1
    def add_atom(self, fact: str, category: str = "general", source: str = "agent") -> Tuple[bool, str]:
        fact = re.sub(r"\s+", " ", (fact or "")).strip()
        if len(fact) < 6:
            return False, "facto demasiado curto"
        cat = category if category in ATOM_CATEGORIES else "general"
        h = hashlib.sha1(fact.lower().encode("utf-8")).hexdigest()
        emb = self._embed(fact)
        with self._lock, self._conn() as c:
            if c.execute("SELECT 1 FROM l1_atoms WHERE hash=?", (h,)).fetchone():
                return False, "duplicado exato"
            if emb:
                for r in c.execute("SELECT content, embedding FROM l1_atoms "
                                   "WHERE embedding IS NOT NULL ORDER BY id DESC LIMIT 500"):
                    try:
                        if self._cos(emb, json.loads(r["embedding"])) >= 0.93:
                            return False, f"quase duplicado de: {r['content'][:80]}"
                    except Exception:
                        continue
            c.execute("INSERT INTO l1_atoms(content, category, source, hash, embedding, ts) "
                      "VALUES (?,?,?,?,?,?)",
                      (fact, cat, source, h, json.dumps(emb) if emb else None, time.time()))
        self._write_scenario(cat)
        return True, "gravado"

    def count_atoms(self) -> int:
        with self._conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM l1_atoms").fetchone()[0])

    def search(self, query: str, top_k: int = 6, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """Recall hibrido BM25 + vetor, fundido por Reciprocal Rank Fusion."""
        toks = _tokenize(query)[:16]
        if not toks:
            return []
        fts_q = " OR ".join(f'"{t}"' for t in toks)
        scores: Dict[int, float] = {}
        rows_by_id: Dict[int, Dict[str, Any]] = {}
        with self._conn() as c:
            try:
                bm = c.execute(
                    "SELECT a.id, a.content, a.category, a.embedding FROM l1_fts f "
                    "JOIN l1_atoms a ON a.id=f.rowid WHERE l1_fts MATCH ? "
                    "ORDER BY bm25(l1_fts) LIMIT 30", (fts_q,)
                ).fetchall()
            except sqlite3.OperationalError:
                bm = []
            for rank, r in enumerate(bm):
                rows_by_id[r["id"]] = dict(r)
                scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (rrf_k + rank + 1)

            qv = self._embed(query)
            if qv:
                cand = c.execute("SELECT id, content, category, embedding FROM l1_atoms "
                                 "WHERE embedding IS NOT NULL ORDER BY id DESC LIMIT 2000").fetchall()
                sims = []
                for r in cand:
                    try:
                        sims.append((self._cos(qv, json.loads(r["embedding"])), r))
                    except Exception:
                        continue
                sims.sort(key=lambda x: x[0], reverse=True)
                for rank, (sim, r) in enumerate(sims[:30]):
                    if sim < 0.35:
                        break
                    rows_by_id.setdefault(r["id"], dict(r))
                    scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (rrf_k + rank + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"id": i, "content": rows_by_id[i]["content"],
                 "category": rows_by_id[i]["category"], "score": round(s, 5)}
                for i, s in ranked]

    # ------------------------------------------------------------------ L2
    def _write_scenario(self, category: str) -> None:
        with self._conn() as c:
            rows = c.execute("SELECT content, ts FROM l1_atoms WHERE category=? ORDER BY id",
                             (category,)).fetchall()
        lines = [f"# Cenario: {category}", "",
                 f"_Gerado automaticamente a partir de {len(rows)} factos L1._", ""]
        lines += [f"- {r['content']}" for r in rows]
        (self.scen_dir / f"{category}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ L3
    def persona_text(self, max_chars: int = 1500) -> str:
        try:
            t = self.persona_path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
        return t[:max_chars]

    def update_persona(self, llm: LLMFn) -> bool:
        with self._conn() as c:
            facts = [r["content"] for r in c.execute(
                "SELECT content FROM l1_atoms ORDER BY id DESC LIMIT 200")]
        if not facts:
            return False
        try:
            text = _strip_think(llm(_PERSONA_PROMPT.format(facts="\n".join(f"- {f}" for f in facts))))
        except Exception:
            return False
        if len(text) < 20:
            return False
        self.persona_path.write_text(text + "\n", encoding="utf-8")
        self._set_meta("persona_at_count", str(self.count_atoms()))
        return True

    # ------------------------------------------------------- extracao L0->L1
    @staticmethod
    def _parse_atoms(raw: str) -> List[Dict[str, str]]:
        txt = _strip_think(raw)
        m = re.search(r"\[.*\]", txt, flags=re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:
            return []
        out = []
        for d in data if isinstance(data, list) else []:
            if isinstance(d, dict) and isinstance(d.get("fact"), str):
                out.append({"fact": d["fact"], "category": str(d.get("category", "general")).lower()})
        return out

    def extract(self, llm: LLMFn, max_per_pass: int = 20) -> int:
        """L0 -> L1. Marca os turnos como processados mesmo sem factos."""
        rows = self.pending_turns()
        if not rows:
            return 0
        convo = "\n".join(f"[{r['role'].upper()}] {r['content'][:1500]}" for r in rows)
        try:
            raw = llm(_EXTRACT_PROMPT.format(max_n=max_per_pass, conversation=convo[-12000:]))
        except Exception:
            return 0  # nao marca: tenta outra vez na proxima janela
        added = 0
        for a in self._parse_atoms(raw)[:max_per_pass]:
            ok, _ = self.add_atom(a["fact"], a["category"], source="extract")
            added += int(ok)
        with self._conn() as c:
            c.executemany("UPDATE l0_turns SET extracted=1 WHERE id=?", [(r["id"],) for r in rows])
        if self.count_atoms() - int(self._meta("persona_at_count", "0")) >= self.persona_every:
            self.update_persona(llm)
        return added

    def should_extract(self) -> bool:
        return len(self.pending_turns(limit=self.extract_every * 2)) >= self.extract_every * 2

    # -------------------------------------------------------------- recall
    def recall_block(self, query: str, top_k: int = 5, max_chars: int = 1200) -> Optional[str]:
        if len((query or "").strip()) < 8:
            return None
        hits = self.search(query, top_k=top_k)
        if not hits:
            return None
        out, total = [], 0
        for h in hits:
            line = f"- ({h['category']}) {h['content']}"
            if total + len(line) > max_chars:
                break
            out.append(line)
            total += len(line)
        return "\n".join(out) if out else None


# =============================================================================
# LLM auxiliar para extracao (NUNCA o mesmo servidor do agente por omissao)
# =============================================================================

def resolve_memory_llm() -> Optional[LLMFn]:
    """
    Endpoint dedicado para extracao/persona, ou None.

    Ordem: APEX_MEMORY_LLM_URL (+ APEX_MEMORY_LLM_MODEL) -> NPU XDNA2 se o
    Lemonade ja estiver a correr. Nao arranca servidores nem usa o llama-server
    principal: isso expulsaria a conversa da cache KV.
    """
    if os.environ.get("APEX_MEMORY_EXTRACT", "1").lower() in ("0", "false", "no"):
        return None
    url = os.environ.get("APEX_MEMORY_LLM_URL")
    model = os.environ.get("APEX_MEMORY_LLM_MODEL")
    if not url:
        try:
            from apex_harness.npu_backend import get_npu_manager
            mgr = get_npu_manager()
            if mgr.is_running():
                url = mgr.api_base
                model = model or os.environ.get("APEX_CLEF_NPU_MODEL", "qwen3.5-9b-npu2")
        except Exception:
            url = None
    if not url:
        return None

    def _call(prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(base_url=url, api_key=os.environ.get("APEX_MEMORY_LLM_KEY", "none"), timeout=120.0)
        resp = client.chat.completions.create(
            model=model or "local-model",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=900,
        )
        return resp.choices[0].message.content or ""

    return _call


# =============================================================================
# Fachada usada pelo ApexAgent
# =============================================================================

class AgentMemory:
    def __init__(self, session_id: str, base_dir: Optional[Path] = None,
                 llm: Optional[LLMFn] = "auto", embedder: Optional[Any] = "auto"):
        self.base = base_dir or memory_base_dir()
        self.offloader = ContextOffloader(session_id, self.base)
        self.ltm = LongTermMemory(self.base, embedder=embedder)
        self._llm = resolve_memory_llm() if llm == "auto" else llm
        self._bg: Optional[threading.Thread] = None

    def new_session(self, session_id: str) -> None:
        self.offloader = ContextOffloader(session_id, self.base)

    def after_turn(self, user_text: str, assistant_text: str) -> None:
        sid = self.offloader.session_id
        self.ltm.add_turn(sid, "user", user_text)
        self.ltm.add_turn(sid, "assistant", assistant_text)
        if self._llm and self.ltm.should_extract() and not (self._bg and self._bg.is_alive()):
            self._bg = threading.Thread(target=self.ltm.extract, args=(self._llm,), daemon=True)
            self._bg.start()

    def persona_block(self) -> str:
        p = self.ltm.persona_text()
        return f"\n\n## Memoria de longo prazo (perfil do utilizador)\n{p}" if p else ""


def memory_enabled() -> bool:
    return os.environ.get("APEX_MEMORY", "1").lower() not in ("0", "false", "no")
