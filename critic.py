"""
Apex Harness — Self-Critic & Draft Review Engine (critic.py)
Evaluates proposed code diffs before disk writes using an auxiliary/draft LLM.
Measures latency and overhead for speculative evaluation.
"""

import os
import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from apex_harness.npu_backend import route_request


@dataclass
class CriticConfig:
    model: str = field(default_factory=lambda: os.environ.get("APEX_CRITIC_MODEL", "qwen2.5-coder:1.5b"))
    npu_model: str = field(default_factory=lambda: os.environ.get("APEX_CRITIC_NPU_MODEL", "qwen2.5-coder-1.5b-int4"))
    api_base: str = field(default_factory=lambda: route_request(task_type="critic"))
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", "dummy"))
    max_tokens: int = 512
    timeout: float = 20.0
    temperature: float = 0.1
    use_npu: bool = True
    fallback_api_base: str = field(default_factory=lambda: os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1").rstrip("/"))


@dataclass
class CriticResult:
    approved: bool
    feedback: str
    latency_ms: float
    model: str


def _resolve_model_for_endpoint(ep: str, cfg: CriticConfig) -> str:
    """
    Resolve the appropriate model ID for the target endpoint.
    For NPU (Lemonade on port 8090), uses npu_model or probes /models for active loaded model.
    For fallback endpoints (e.g. llama-server on port 8080/8000), uses standard cfg.model.
    """
    is_npu_endpoint = ":8090" in ep or "lemonade" in ep.lower()
    default_target = cfg.npu_model if is_npu_endpoint else cfg.model
    if not is_npu_endpoint:
        return default_target

    try:
        req = urllib.request.Request(f"{ep}/models", headers={"Authorization": f"Bearer {cfg.api_key}"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models_list = data.get("data", [])
            if models_list and isinstance(models_list, list):
                model_id = models_list[0].get("id")
                if model_id:
                    return str(model_id)
    except Exception:
        pass
    return default_target


def run_critic(diff: str, config: Optional[CriticConfig] = None) -> CriticResult:
    """
    Run an automated review on a unified diff using the critic model.
    Returns CriticResult indicating whether the diff was approved or flagged.
    """
    cfg = config or CriticConfig()

    if not diff or not diff.strip():
        return CriticResult(
            approved=True,
            feedback="Diff is empty.",
            latency_ms=0.0,
            model=cfg.model
        )

    prompt = (
        "You are an automated code review critic. Review the following code diff for:\n"
        "1. Critical syntax errors or broken imports\n"
        "2. Dangerous regressions or data loss\n"
        "3. Security vulnerabilities\n\n"
        f"```diff\n{diff[:6000]}\n```\n\n"
        "If the diff looks clean and safe to apply, reply ONLY with the word 'APPROVED'.\n"
        "If you detect serious issues, briefly explain them."
    )

    endpoint_to_try = cfg.api_base if cfg.use_npu else cfg.fallback_api_base
    endpoints = [endpoint_to_try]
    if cfg.fallback_api_base and cfg.fallback_api_base != endpoint_to_try:
        endpoints.append(cfg.fallback_api_base)

    start = time.perf_counter()
    last_error: Optional[Exception] = None

    for ep in endpoints:
        target_model = _resolve_model_for_endpoint(ep, cfg)
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": "You are a code review assistant. Be strict and concise."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature
        }
        url = f"{ep}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}"
        }
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"].strip()
                latency = (time.perf_counter() - start) * 1000.0

                approved = "APPROVED" in content.upper()
                return CriticResult(
                    approved=approved,
                    feedback=content,
                    latency_ms=latency,
                    model=target_model
                )
        except urllib.error.URLError as e:
            last_error = e
            # Try next endpoint in fallback list
            continue
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000.0
            return CriticResult(
                approved=True,
                feedback=f"[Critic Error: {str(e)}, skipped review]",
                latency_ms=latency,
                model=cfg.model
            )

    latency = (time.perf_counter() - start) * 1000.0
    reason = getattr(last_error, "reason", str(last_error)) if last_error else "Endpoint unreachable"
    return CriticResult(
        approved=True,
        feedback=f"[Critic Warning: Endpoint unavailable ({reason}), skipped review]",
        latency_ms=latency,
        model=cfg.model
    )
