"""
apex_harness.jev_decider — Typed Decision Layer (Jev pattern)

Based on: "Jev Engineering for Coding Agents" (Diogo Almeida / TypeSafe, 2026)

Jev is NOT the model that writes the code.
It is the System 1 decision layer beside the System 2 coding model.

Three typed decision primitives:
  1. ask_choice(state, question, options) → JevChoice (selected option, probabilities, confidence)
  2. ask_score(state, question, scale)    → JevScore  (value, probabilities, confidence)
  3. ask_noul(state, question)            → JevNoul   (probability_yes, confidence, is_yes)

Convenience wrappers for harness integration:
  - ask_permit(command)                   → JevPermit ("allow" | "ask" | "deny")
  - ask_tools(user_message, tools)        → List[str]
  - ask_guardrail(model_output)           → JevAnswer (pass bool)

Backend priority:
  1. DecisionForge / Clef System 1 Sidecar @ 127.0.0.1:8501 (/v1/systemone)
  2. Dedicated Clef Sidecar                @ 127.0.0.1:8091 (/v1/systemone)
  3. NPU XDNA2 (Lemonade / FastFlowLM)     @ 127.0.0.1:8090
  4. Local Ollama                          @ 127.0.0.1:11434 (fallback)
  5. LM Studio                             @ 127.0.0.1:1234

All calls are non-blocking: the caller gets a Future and can .result() when ready.
The main generation loop is never stalled.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration: Clef-Flash-9B & NPU XDNA2 Architecture
# ---------------------------------------------------------------------------

CLEF_FLASH_LOCAL_PATH = os.environ.get(
    "APEX_CLEF_PATH",
    "/run/media/leonardo/C69ADBDF9ADBCA57/AIModels/bartowski/Cloudflare_clef-flash-GGUF/Cloudflare_clef-flash-Q4_K_M.gguf"
)
DEFAULT_CLEF_MODEL = os.environ.get("APEX_CLEF_MODEL", "clef-flash")
DEFAULT_CLEF_NPU_MODEL = os.environ.get("APEX_CLEF_NPU_MODEL", "qwen3.5-9b-npu2")


def get_jev_endpoints() -> List[Dict[str, str]]:
    """
    Build dynamic prioritized list of decision backends:
      1. DecisionForge / Clef System 1 sidecar (@ port 8501 or APEX_DECISION_URL).
      2. NPU XDNA2 (Lemonade / FastFlowLM @ port 8090) if usable.
      3. Dedicated Clef sidecar (llama-server / systemone @ port 8091).
      4. Local Ollama (clef-flash @ port 11434).
      5. LM Studio (clef-flash / Qwen @ port 1234).
      6. Primary local LLM server fallback (port 8080/8000).
    """
    endpoints: List[Dict[str, str]] = []

    # 1. DecisionForge Clef System 1 Sidecar (/v1/systemone)
    decision_url = os.environ.get("APEX_DECISION_URL", "http://127.0.0.1:8501").rstrip("/")
    decision_model = os.environ.get("APEX_DECISION_MODEL", "models/apex_guard_05b")
    endpoints.append({
        "name": "DecisionForge-System1-8501",
        "api": decision_url,
        "model": decision_model,
        "format": "systemone",
    })
    if decision_url != "http://127.0.0.1:8091":
        endpoints.append({
            "name": "DecisionForge-System1-8091",
            "api": "http://127.0.0.1:8091",
            "model": decision_model,
            "format": "systemone",
        })

    use_npu = os.environ.get("APEX_USE_NPU_JEV", "1").lower() in ("1", "true", "yes")

    # 1. NPU XDNA2 check (port 8090)
    if use_npu:
        try:
            from apex_harness.npu_detect import npu_available
            from apex_harness.npu_backend import route_request, get_npu_manager
            status = npu_available()
            if status.usable:
                endpoints.append({
                    "name": "NPU-XDNA2-Lemonade-8090",
                    "api": "http://127.0.0.1:8090/v1",
                    "model": DEFAULT_CLEF_NPU_MODEL,
                    "format": "openai",
                })
        except Exception:
            pass

    # 2. Dedicated Clef-9B sidecar (port 8091)
    endpoints.append({
        "name": "llama-server-clef-8091",
        "api": "http://127.0.0.1:8091/v1",
        "model": DEFAULT_CLEF_MODEL,
        "format": "openai",
    })

    # 3. Local Ollama with clef-flash (port 11434)
    endpoints.append({
        "name": "Ollama-clef-flash",
        "api": "http://127.0.0.1:11434/v1",
        "model": "clef-flash",
        "format": "openai",
    })

    # 4. LM Studio with clef-flash (port 1234)
    endpoints.append({
        "name": "LMStudio-clef-flash",
        "api": "http://127.0.0.1:1234/v1",
        "model": DEFAULT_CLEF_MODEL,
        "format": "openai",
    })

    # 5. Primary server fallback
    fallback_base = os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1").rstrip("/")
    endpoints.append({
        "name": "primary-server-fallback",
        "api": fallback_base,
        "model": os.environ.get("APEX_PRIMARY_MODEL", "local-llm"),
        "format": "openai",
    })

    return endpoints


# Backward-compatible endpoint list
_JEV_ENDPOINTS: List[Dict[str, str]] = get_jev_endpoints()

_JEV_SYSTEM_PROMPT = (
    "Você é um motor de decisão, não um assistente de chat. Você NUNCA explica, "
    "NUNCA pensa em voz alta, NUNCA usa tags <think>. Responda IMEDIATAMENTE "
    "com um único objeto JSON, nada mais — sem markdown, sem texto antes ou depois.\n\n"
    "Existem 3 tipos de pergunta. Responda no formato exato do tipo pedido:\n\n"
    "1) CHOICE — escolher uma opção de uma lista fixa\n"
    '{"type":"choice","selected":"<opção exata>","probabilities":{"<opção1>":0.0,"<opção2>":0.0},"confidence":0.0}\n\n'
    "2) SCORE — nota numa escala descrita\n"
    '{"type":"score","value":<número>,"probabilities":{"<nível1>":0.0,"<nível2>":0.0},"confidence":0.0}\n\n'
    "3) NOUL — pergunta sim/não com probabilidade\n"
    '{"type":"noul","probability_yes":0.0,"confidence":0.0}\n\n'
    "Regras:\n"
    '- "confidence" reflete o quão certo você está da resposta (0 a 1), não a probabilidade em si.\n'
    "- Nunca invente uma opção fora da lista fornecida.\n"
    "- Se o estado (state) não tiver informação suficiente, ainda responda com a "
    "melhor estimativa e confidence baixa — nunca recuse, nunca peça mais contexto."
)

_JEV_MAX_TOKENS  = 256
_JEV_TEMPERATURE = 0.0
_JEV_TIMEOUT_SEC = 12.0


def _detect_hardware_label(backend_name: str) -> str:
    """Classify backend into human-readable hardware source."""
    low = backend_name.lower()
    if "decisionforge" in low or "system1" in low:
        return "Clef Joint Schema Head (System 1)"
    if "npu" in low or "8090" in low or "xdna" in low:
        return "NPU AMD XDNA2 (Lemonade 8090)"
    if "ollama" in low:
        return "Ollama (Clef-Flash 9B)"
    if "1234" in low or "lmstudio" in low:
        return "LM Studio (Clef-Flash 9B)"
    if "clef" in low or "8091" in low:
        return "Clef-Flash 9B (Dedicated Sidecar)"
    return "GPU/CPU Fallback"

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class JevChoice:
    selected:      str
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence:    float = 1.0
    latency_ms:    float = 0.0
    backend:       str   = ""
    hardware:      str   = ""
    ok:            bool  = True
    raw:           str   = ""
    error:         Optional[str] = None


@dataclass
class JevScore:
    value:         float
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence:    float = 1.0
    latency_ms:    float = 0.0
    backend:       str   = ""
    hardware:      str   = ""
    ok:            bool  = True
    raw:           str   = ""
    error:         Optional[str] = None


@dataclass
class JevNoul:
    probability_yes: float
    confidence:      float = 1.0
    latency_ms:      float = 0.0
    backend:         str   = ""
    hardware:        str   = ""
    ok:              bool  = True
    raw:             str   = ""
    error:           Optional[str] = None

    @property
    def is_yes(self) -> bool:
        return self.probability_yes >= 0.5


@dataclass
class JevPermit:
    verdict:    str             # "allow" | "ask" | "deny"
    reason:     str   = ""
    confidence: float = 1.0
    latency_ms: float = 0.0
    backend:    str   = ""
    hardware:   str   = ""
    ok:         bool  = True


@dataclass
class JevAnswer:
    raw:        str
    value:      Any
    confidence: float = 1.0
    latency_ms: float = 0.0
    backend:    str   = ""
    hardware:   str   = ""
    ok:         bool  = True
    error:      Optional[str] = None

# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------

def _clean_thinking(text: str) -> str:
    """Remove <think>...</think> blocks if any slip through."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> Optional[Dict]:
    """Extract first JSON object from model output."""
    clean = _clean_thinking(text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*?\}", clean, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _resolve_target_model(ep: Dict[str, str]) -> str:
    """Probe /models to detect active model or fallback to configured model."""
    model = ep.get("model", DEFAULT_CLEF_MODEL)
    try:
        url = f"{ep['api'].rstrip('/')}/models"
        req = urllib.request.Request(url, headers={"User-Agent": "ApexHarness-Jev"})
        with urllib.request.urlopen(req, timeout=0.8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models_list = data.get("data", [])
            if models_list and isinstance(models_list, list):
                mid = models_list[0].get("id")
                if mid:
                    return str(mid)
    except Exception:
        pass
    return model


def _call_openai(api: str, model: str, system: str, user: str, timeout: float) -> str:
    payload = {
        "model":       model,
        "messages":    [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens":  _JEV_MAX_TOKENS,
        "temperature": _JEV_TEMPERATURE,
        "stream":      False,
    }
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        f"{api.rstrip('/')}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"].strip()


def _call_ollama(api: str, model: str, system: str, user: str, timeout: float) -> str:
    payload = {
        "model":   model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "format":  "json",
        "options": {"temperature": _JEV_TEMPERATURE, "num_predict": _JEV_MAX_TOKENS},
        "stream":  False,
    }
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        f"{api.rstrip('/')}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["message"]["content"].strip()


def _call_systemone(
    api: str,
    model: str,
    req_type: str,
    state: str,
    question: str,
    criteria: Any,
    timeout: float,
) -> str:
    """Send structured request to DecisionForge Clef /v1/systemone endpoint."""
    if req_type == "choice":
        crit_dict = {opt: opt for opt in criteria} if isinstance(criteria, list) else (criteria or {})
        q_def = {"type": "choice", "instructions": question, "criteria": crit_dict}
    elif req_type == "score":
        q_def = {"type": "score", "instructions": question, "criteria": criteria or {}}
    elif req_type == "noul":
        q_def = {"type": "noul", "instructions": question}
    else:
        q_def = {"type": "choice", "instructions": question, "criteria": {}}

    payload = {
        "model": model,
        "state": state,
        "questions": {"q": q_def},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{api.rstrip('/')}/v1/systemone",
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "ApexHarness-Jev"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    ans = body.get("answers", {}).get("q", {})
    ans_type = ans.get("type", req_type)

    if ans_type == "choice":
        return json.dumps({
            "type": "choice",
            "selected": ans.get("choice", ""),
            "confidence": ans.get("confidence", 1.0),
            "probabilities": ans.get("probabilities", {}),
        })
    elif ans_type == "score":
        return json.dumps({
            "type": "score",
            "value": ans.get("score", 0.0),
            "confidence": ans.get("confidence", 1.0),
            "probabilities": ans.get("probabilities", {}),
        })
    elif ans_type == "noul":
        return json.dumps({
            "type": "noul",
            "probability_yes": ans.get("noul", 0.0),
            "confidence": ans.get("confidence", 1.0),
        })

    return json.dumps(ans)


def _jev_call(
    system: str,
    user: str,
    req_type: str = "choice",
    state: str = "",
    question: str = "",
    criteria: Any = None,
) -> Tuple[str, str, float]:
    """Try each backend in order. Returns (raw_output, backend_name, latency_ms)."""
    endpoints = get_jev_endpoints()
    last_err: Exception = RuntimeError("No backends configured")
    for ep in endpoints:
        t0 = time.perf_counter()
        try:
            if ep["format"] == "systemone":
                target_model = ep.get("model", "models/apex_guard_05b")
                st = state if state else user
                q = question if question else "Avaliação de decisão"
                raw = _call_systemone(ep["api"], target_model, req_type, st, q, criteria, timeout=3.0)
            else:
                target_model = _resolve_target_model(ep)
                if ep["format"] == "openai":
                    raw = _call_openai(ep["api"], target_model, system, user, _JEV_TIMEOUT_SEC)
                else:
                    raw = _call_ollama(ep["api"], target_model, system, user, _JEV_TIMEOUT_SEC)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return raw, ep["name"], latency_ms
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(f"All Jev backends unavailable: {last_err}") from last_err

# ---------------------------------------------------------------------------
# JevDecider Core Class
# ---------------------------------------------------------------------------

class JevDecider:
    """
    Lightweight typed System 1 decision layer running in a background thread pool.
    """

    def __init__(self, workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="jev")

    # ------------------------------------------------------------------
    # 1. Canonical System 1: CHOICE
    # ------------------------------------------------------------------

    def ask_choice(self, state: str, question: str, options: List[str]) -> "Future[JevChoice]":
        """
        Pick one of N options with probabilities and confidence.
        Returns Future[JevChoice].
        """
        opt_json = json.dumps(options, ensure_ascii=False)
        user_prompt = (
            f"STATE:\n{state}\n\n"
            f"QUESTION (choice):\n{question}\n\n"
            f"OPTIONS: {opt_json}"
        )
        fallback_option = options[0] if options else ""

        def _run() -> JevChoice:
            t0 = time.perf_counter()
            try:
                raw, backend, latency = _jev_call(_JEV_SYSTEM_PROMPT, user_prompt, req_type="choice", state=state, question=question, criteria=options)
                parsed = _extract_json(raw)
                if parsed and parsed.get("type") == "choice":
                    selected = str(parsed.get("selected", fallback_option))
                    if selected not in options and fallback_option in options:
                        selected = fallback_option
                    probs = parsed.get("probabilities", {})
                    confidence = float(parsed.get("confidence", 1.0))
                    return JevChoice(
                        selected=selected,
                        probabilities=probs if isinstance(probs, dict) else {},
                        confidence=confidence,
                        latency_ms=latency,
                        backend=backend,
                        hardware=_detect_hardware_label(backend),
                        ok=True,
                        raw=raw,
                    )
            except Exception as exc:
                latency = (time.perf_counter() - t0) * 1000.0
                return JevChoice(
                    selected=fallback_option,
                    confidence=0.0,
                    latency_ms=latency,
                    backend="fallback",
                    hardware="Fallback",
                    ok=False,
                    error=str(exc),
                )
            return JevChoice(selected=fallback_option, confidence=0.0, ok=False)

        return self._pool.submit(_run)

    # ------------------------------------------------------------------
    # 2. Canonical System 1: SCORE
    # ------------------------------------------------------------------

    def ask_score(self, state: str, question: str, scale: Dict[str, str]) -> "Future[JevScore]":
        """
        Score a state on a defined rubric scale with confidence.
        Returns Future[JevScore].
        """
        scale_json = json.dumps(scale, ensure_ascii=False)
        user_prompt = (
            f"STATE:\n{state}\n\n"
            f"QUESTION (score):\n{question}\n\n"
            f"SCALE: {scale_json}"
        )

        def _run() -> JevScore:
            t0 = time.perf_counter()
            try:
                raw, backend, latency = _jev_call(_JEV_SYSTEM_PROMPT, user_prompt, req_type="score", state=state, question=question, criteria=scale)
                parsed = _extract_json(raw)
                if parsed and parsed.get("type") == "score":
                    val = float(parsed.get("value", 0.0))
                    probs = parsed.get("probabilities", {})
                    confidence = float(parsed.get("confidence", 1.0))
                    return JevScore(
                        value=val,
                        probabilities=probs if isinstance(probs, dict) else {},
                        confidence=confidence,
                        latency_ms=latency,
                        backend=backend,
                        hardware=_detect_hardware_label(backend),
                        ok=True,
                        raw=raw,
                    )
            except Exception as exc:
                latency = (time.perf_counter() - t0) * 1000.0
                return JevScore(
                    value=0.0,
                    confidence=0.0,
                    latency_ms=latency,
                    backend="fallback",
                    hardware="Fallback",
                    ok=False,
                    error=str(exc),
                )
            return JevScore(value=0.0, confidence=0.0, ok=False)

        return self._pool.submit(_run)

    # ------------------------------------------------------------------
    # 3. Canonical System 1: NOUL (Yes/No + probability)
    # ------------------------------------------------------------------

    def ask_noul(self, state: str, question: str) -> "Future[JevNoul]":
        """
        Evaluate binary proposition (Sim / Não) with probability and confidence.
        Returns Future[JevNoul].
        """
        user_prompt = (
            f"STATE:\n{state}\n\n"
            f"QUESTION (noul):\n{question}"
        )

        def _run() -> JevNoul:
            t0 = time.perf_counter()
            try:
                raw, backend, latency = _jev_call(_JEV_SYSTEM_PROMPT, user_prompt, req_type="noul", state=state, question=question, criteria=None)
                parsed = _extract_json(raw)
                if parsed and parsed.get("type") == "noul":
                    prob = float(parsed.get("probability_yes", 0.0))
                    confidence = float(parsed.get("confidence", 1.0))
                    return JevNoul(
                        probability_yes=prob,
                        confidence=confidence,
                        latency_ms=latency,
                        backend=backend,
                        hardware=_detect_hardware_label(backend),
                        ok=True,
                        raw=raw,
                    )
            except Exception as exc:
                latency = (time.perf_counter() - t0) * 1000.0
                return JevNoul(
                    probability_yes=0.0,
                    confidence=0.0,
                    latency_ms=latency,
                    backend="fallback",
                    hardware="Fallback",
                    ok=False,
                    error=str(exc),
                )
            return JevNoul(probability_yes=0.0, confidence=0.0, ok=False)

        return self._pool.submit(_run)

    # ------------------------------------------------------------------
    # Convenience Harness Wrappers
    # ------------------------------------------------------------------

    def ask_permit(self, command: str) -> "Future[JevPermit]":
        """Evaluate whether a bash command is safe: allow | ask | deny."""
        state = f"Comando bash a ser executado pelo agente:\n{command[:400]}"
        question = "Esse comando deve ser permitido automaticamente (allow), requerer confirmação humana (ask) ou ser bloqueado (deny)?"
        options = ["allow", "ask", "deny"]

        choice_fut = self.ask_choice(state=state, question=question, options=options)

        def _transform() -> JevPermit:
            res = choice_fut.result()
            verdict = res.selected if res.ok else "ask"
            if verdict not in ("allow", "ask", "deny"):
                verdict = "ask"
            return JevPermit(
                verdict=verdict,
                reason=f"Jev confidence: {res.confidence:.2f}",
                confidence=res.confidence,
                latency_ms=res.latency_ms,
                backend=res.backend,
                hardware=res.hardware,
                ok=res.ok,
            )

        return self._pool.submit(_transform)

    def ask_guardrail(self, model_output: str) -> "Future[JevAnswer]":
        """Verify if model output passed safety guardrails."""
        state = f"Output gerado pelo modelo:\n{model_output[:800]}"
        question = "A saída do modelo é segura, livre de comandos destrutivos e atende aos guardrails?"
        noul_fut = self.ask_noul(state=state, question=question)

        def _transform() -> JevAnswer:
            res = noul_fut.result()
            return JevAnswer(
                raw=res.raw,
                value=res.is_yes if res.ok else True,  # fail-open
                confidence=res.confidence,
                latency_ms=res.latency_ms,
                backend=res.backend,
                hardware=res.hardware,
                ok=res.ok,
                error=res.error,
            )

        return self._pool.submit(_transform)

    def ask_tools(self, user_message: str, available_tools: List[str]) -> "Future[List[str]]":
        """Select relevant tools for this turn."""
        state = f"Mensagem do usuário: {user_message[:500]}\nFerramentas disponíveis: {', '.join(available_tools)}"
        question = "Qual a ferramenta primária mais adequada para responder a este turno?"
        choice_fut = self.ask_choice(state=state, question=question, options=available_tools)

        def _transform() -> List[str]:
            res = choice_fut.result()
            if res.ok and res.selected in available_tools:
                return [res.selected]
            return list(available_tools)

        return self._pool.submit(_transform)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self, wait: bool = False) -> None:
        self._pool.shutdown(wait=wait)

    def __repr__(self) -> str:
        backends = ", ".join(ep["name"] for ep in get_jev_endpoints())
        return f"JevDecider(backends=[{backends}])"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_singleton: Optional[JevDecider] = None


def get_jev() -> JevDecider:
    """Return the module-level JevDecider singleton (lazy init)."""
    global _singleton
    if _singleton is None:
        _singleton = JevDecider()
    return _singleton
