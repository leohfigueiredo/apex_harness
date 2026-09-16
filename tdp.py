"""
apex_harness.tdp — Task-Decoupled Planning (TDP) pipeline.

WHAT IT DOES
------------
Decomposes a complex user task into three isolated, single-purpose agent calls:

  Agent 1 — Researcher
    Reads the raw task and distils ONLY the structural rules, hard constraints,
    and input/output contracts. Strips up to 90% of prose/context noise.
    Output: a compact, numbered spec (≤ researcher_budget tokens).

  Agent 2 — Planner  (PTCF framework)
    Acts as a Staff Engineer. Receives only the distilled spec and produces a
    deterministic implementation blueprint (Persona / Task / Context / Format /
    Steps / Risks). Does NOT write code.

  Agent 3 — Executor + Self-Healing
    Receives only the PTCF plan and generates the final solution.
    If a code block is produced AND execute_code=True, the code is run inside
    a bash_exec sandbox. On failure, a self-healing loop retries up to
    max_heal_attempts times, feeding the exact error back to the agent.

WHY THIS SAVES TOKENS
---------------------
Each agent operates on a *fresh two-message context* (system + one user
message). No chat history accumulates between agents. The Researcher strips the
input; the Planner strips the spec; the Executor only sees the plan.
A typical 2400-token prompt reaches the Executor as ~300-400 tokens.

USAGE
-----
    from apex_harness.tdp import run_tdp, TDPConfig

    result = run_tdp(
        task="Write a Python function that ...",
        agent=apex_agent_instance,
        cfg=TDPConfig(stream=True),
        on_event=lambda stage, chunk: print(chunk, end="", flush=True),
    )
    print(result.solution)
"""

from __future__ import annotations

import re
import tempfile
import os
import subprocess
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Any, Dict

# ─────────────────────────────────────────────────────────────────────────────
#  Config & Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TDPConfig:
    """Runtime configuration for the TDP pipeline."""
    #: Maximum self-healing iterations for the Executor agent.
    max_heal_attempts: int = 3
    #: Stream text chunks to on_event as they arrive.
    stream: bool = True
    #: Soft token budget hint passed to the Researcher system prompt.
    researcher_budget: int = 800
    #: Soft token budget hint passed to the Planner system prompt.
    planner_budget: int = 1200
    #: Whether to run extracted code blocks in a bash sandbox for validation.
    execute_code: bool = True
    #: Print internal prompts to stderr for debugging.
    verbose: bool = False


@dataclass
class TDPResult:
    """Output of a complete TDP pipeline run."""
    spec: str              # Researcher output
    plan: str              # Planner output
    solution: str          # Executor final output
    healed: bool = False   # True if at least one self-healing iteration ran
    heal_attempts: int = 0
    #: Estimated token savings: 1 - (researcher_tokens / raw_task_tokens)
    tokens_saved_pct: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Agent system prompts
# ─────────────────────────────────────────────────────────────────────────────

_RESEARCHER_SYSTEM = """\
You are a requirements distiller. Your only job is to extract the essential \
engineering signal from the user's task description.

Output ONLY:
1. Hard constraints (must / must not)
2. Input/output contract (types, shapes, edge cases)
3. Performance or correctness requirements
4. Dependencies or environment assumptions

Rules:
- Discard: motivation, backstory, prose explanation, redundant examples.
- Keep: examples ONLY if they define a non-obvious edge case.
- Use a numbered bullet list. Be terse. No headers. No preamble.
- Target ≤{budget} tokens. If forced to choose, keep constraints over examples.
""".strip()

_PLANNER_SYSTEM = """\
You are a Staff Engineer writing a precise implementation blueprint.
You do NOT write code. You architect the perfect solution.

Use EXACTLY this format:

PERSONA: [role and seniority level best suited to implement this]
TASK: [one sentence, fully specific — no ambiguity]
CONTEXT: [paste only the constraints from the spec that directly affect \
implementation choices — nothing else]
FORMAT: [exact output format: file structure, function signatures, \
naming conventions, return types]
STEPS:
  1. [first concrete action, ≤2 lines]
  2. ...
RISKS:
  - [top failure mode and its mitigation]
  - [second failure mode and its mitigation]
""".strip()

_EXECUTOR_SYSTEM = """\
You are a senior engineer implementing exactly the plan given.
Rules:
- Write production-quality code. No placeholders, no TODOs, no ellipsis.
- Match the FORMAT section of the plan exactly.
- Wrap all code in a fenced code block with the correct language tag.
- After the code block, add a one-paragraph explanation of the key decisions.
""".strip()

_HEALER_PREFIX = """\
The previous implementation attempt failed with this error:

{error}

Fix ONLY the error above. Do not rewrite unrelated parts. \
Keep the same structure, naming, and format from the plan.

Plan:
{plan}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _call_agent_clean(
    agent: Any,
    system: str,
    user: str,
    on_chunk: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Run a single inference call on a FRESH two-message context.

    Deliberately bypasses agent.history so no chat state leaks between
    TDP stages. Uses the agent's existing authenticated client.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    accumulated: List[str] = []
    try:
        stream = agent.client.chat.completions.create(
            model=agent.model_name,
            messages=messages,
            temperature=0.15,   # low temperature for deterministic planning
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                accumulated.append(delta.content)
                if on_chunk:
                    on_chunk(delta.content)
    except Exception as exc:
        # Non-streaming fallback
        try:
            resp = agent.client.chat.completions.create(
                model=agent.model_name,
                messages=messages,
                temperature=0.15,
            )
            text = resp.choices[0].message.content or ""
            accumulated = [text]
            if on_chunk:
                on_chunk(text)
        except Exception as inner:
            return f"[TDP agent error: {exc} / {inner}]"

    return "".join(accumulated)


def _extract_code_blocks(text: str) -> List[tuple[str, str]]:
    """
    Return a list of (language, code) tuples from fenced code blocks.

    Supports ```python, ```javascript, ```js, ```bash, ```sh, ```typescript.
    """
    pattern = re.compile(
        r"```(python|javascript|js|typescript|bash|sh)?\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    results = []
    for m in pattern.finditer(text):
        lang = (m.group(1) or "python").lower().strip()
        code = m.group(2).strip()
        results.append((lang, code))
    return results


def _run_in_sandbox(lang: str, code: str, timeout: int = 30) -> Optional[str]:
    """
    Execute code in a temporary file via subprocess.

    Returns None on success, or the error string on failure.
    Only runs Python and JS — other languages are skipped (no error).
    """
    ext_map = {
        "python": (".py", ["python3"]),
        "javascript": (".js", ["node"]),
        "js": (".js", ["node"]),
        "typescript": (".ts", ["npx", "ts-node", "--transpile-only"]),
        "bash": (".sh", ["bash"]),
        "sh": (".sh", ["bash"]),
    }
    entry = ext_map.get(lang)
    if not entry:
        return None  # unsupported lang → skip validation

    ext, runner = entry
    with tempfile.NamedTemporaryFile(suffix=ext, mode="w", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            runner + [tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            return err[:2000]  # cap to avoid flooding context
        return None
    except subprocess.TimeoutExpired:
        return f"Execution timed out after {timeout}s."
    except FileNotFoundError:
        return None  # runner not installed → skip validation
    finally:
        os.unlink(tmp_path)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per 3.5 chars."""
    return max(1, int(len(text) / 3.5))


# ─────────────────────────────────────────────────────────────────────────────
#  Complexity heuristic — used by the auto-trigger from the main agent
# ─────────────────────────────────────────────────────────────────────────────

_COMPLEX_KEYWORDS = frozenset([
    "implement", "implementa", "cria", "create", "build", "constrói",
    "system", "sistema", "pipeline", "framework", "refactor", "refatora",
    "architecture", "arquitectura", "design", "optimize", "optimiza",
    "algorithm", "algoritmo", "parser", "compiler", "scheduler",
    "distributed", "distribuído", "concurrent", "concorrente",
    "database", "api", "microservice", "deploy", "migrate",
])

def is_complex_task(text: str, min_words: int = 30) -> bool:
    """
    Heuristic: return True if the task is long AND contains complexity keywords.

    Used by the auto-trigger in the tools layer to decide whether to
    escalate to the TDP pipeline instead of a direct agent reply.
    """
    words = text.lower().split()
    if len(words) < min_words:
        return False
    return any(w in _COMPLEX_KEYWORDS for w in words)


# ─────────────────────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_tdp(
    task: str,
    agent: Any,
    cfg: Optional[TDPConfig] = None,
    on_event: Optional[Callable[[str, str], None]] = None,
) -> TDPResult:
    """
    Execute the full TDP pipeline for a given task.

    Parameters
    ----------
    task:
        Raw user task description.
    agent:
        A live ApexAgent instance. Its `.client` and `.model_name` are reused;
        its `.history` is NOT modified.
    cfg:
        TDPConfig. Defaults are used if None.
    on_event:
        Callback ``on_event(stage: str, chunk: str)`` called for every
        streamed token. ``stage`` is one of:
        "researcher", "planner", "executor", "healer", "status".

    Returns
    -------
    TDPResult
    """
    cfg = cfg or TDPConfig()
    raw_tokens = _estimate_tokens(task)

    def _emit(stage: str, chunk: str) -> None:
        if on_event:
            on_event(stage, chunk)

    # ── STAGE 1: Researcher ──────────────────────────────────────────────────
    _emit("status", "\n🔬 [RESEARCHER] Analisando e destilando requisitos...\n")

    researcher_sys = _RESEARCHER_SYSTEM.format(budget=cfg.researcher_budget)
    if cfg.verbose:
        import sys
        print(f"\n[TDP:researcher prompt]\n{researcher_sys}\n---\n{task}\n", file=sys.stderr)

    spec_chunks: List[str] = []
    def _on_spec(chunk: str):
        spec_chunks.append(chunk)
        if cfg.stream:
            _emit("researcher", chunk)

    spec = _call_agent_clean(agent, researcher_sys, task,
                             on_chunk=_on_spec if cfg.stream else None)
    if not cfg.stream:
        spec = "".join(spec_chunks) or spec

    spec_tokens = _estimate_tokens(spec)
    saved_pct = max(0.0, 1.0 - spec_tokens / raw_tokens)
    _emit("status",
          f"\n   → Spec: ~{spec_tokens} tokens "
          f"(vs ~{raw_tokens} originais ≈ {saved_pct:.0%} poupado)\n")

    # ── STAGE 2: Planner ─────────────────────────────────────────────────────
    _emit("status", "\n📐 [PLANNER] Elaborando plano PTCF...\n")

    planner_user = f"Spec:\n{spec}"
    if cfg.verbose:
        import sys
        print(f"\n[TDP:planner user]\n{planner_user}\n", file=sys.stderr)

    plan_chunks: List[str] = []
    def _on_plan(chunk: str):
        plan_chunks.append(chunk)
        if cfg.stream:
            _emit("planner", chunk)

    plan = _call_agent_clean(agent, _PLANNER_SYSTEM, planner_user,
                             on_chunk=_on_plan if cfg.stream else None)
    if not cfg.stream:
        plan = "".join(plan_chunks) or plan

    _emit("status", f"\n   → Plano: ~{_estimate_tokens(plan)} tokens\n")

    # ── STAGE 3: Executor + Self-Healing ─────────────────────────────────────
    _emit("status", "\n⚡ [EXECUTOR] Gerando e validando solução...\n")

    solution = ""
    healed = False
    heal_attempts = 0
    previous_error: Optional[str] = None
    current_prompt = f"Plan:\n{plan}"

    for attempt in range(cfg.max_heal_attempts + 1):
        # First call: executor. Subsequent calls: healer.
        if attempt == 0:
            sys_prompt = _EXECUTOR_SYSTEM
            user_prompt = current_prompt
            stage_label = "executor"
        else:
            sys_prompt = _EXECUTOR_SYSTEM
            user_prompt = _HEALER_PREFIX.format(
                error=previous_error, plan=plan
            )
            stage_label = "healer"
            heal_attempts = attempt
            _emit("status",
                  f"\n   ✗ Erro detectado. Auto-corrigindo (tentativa {attempt}/{cfg.max_heal_attempts})...\n")
            _emit("status", f"   Erro: {previous_error[:200]}...\n" if len(previous_error) > 200 else f"   Erro: {previous_error}\n")

        sol_chunks: List[str] = []
        def _on_sol(chunk: str, _label=stage_label):
            sol_chunks.append(chunk)
            if cfg.stream:
                _emit(_label, chunk)

        solution = _call_agent_clean(agent, sys_prompt, user_prompt,
                                     on_chunk=_on_sol if cfg.stream else None)
        if not cfg.stream:
            solution = "".join(sol_chunks) or solution

        # Sandbox validation
        if not cfg.execute_code:
            _emit("status", "\n   ✔ Código gerado (validação desactivada).\n")
            break

        code_blocks = _extract_code_blocks(solution)
        if not code_blocks:
            _emit("status", "\n   ✔ Resposta sem bloco de código — entrega directa.\n")
            break

        # Run the first code block
        lang, code = code_blocks[0]
        _emit("status", f"\n   ⏳ Executando sandbox ({lang})...\n")
        error = _run_in_sandbox(lang, code)

        if error is None:
            if attempt > 0:
                healed = True
                _emit("status", f"\n   ✔ Corrigido na tentativa {attempt}.\n")
            else:
                _emit("status", "\n   ✔ Validação passou (0 erros).\n")
            break

        previous_error = error
        if attempt == cfg.max_heal_attempts:
            _emit("status",
                  f"\n   ⚠ Máximo de tentativas atingido ({cfg.max_heal_attempts}). "
                  f"Entregando melhor versão disponível.\n")

    return TDPResult(
        spec=spec,
        plan=plan,
        solution=solution,
        healed=healed,
        heal_attempts=heal_attempts,
        tokens_saved_pct=saved_pct,
    )
