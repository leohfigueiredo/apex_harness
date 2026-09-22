"""
apex_harness.subagents — Specialized Sub-Agents with NPU Offload

Decouples lightweight, repetitive cognitive tasks (code review, commit message
generation, task specification/triage) from the primary 30B orchestration model.

Sub-agents:
  1. CriticSubagent  — Reviews unified diffs for syntax/security/regressions (NPU / port 8090)
  2. CommitSubagent  — Generates Conventional Commits from git status & diff
  3. TriageSubagent  — Distills noisy user tasks into compact engineering specs

Architecture:
  - Operates on ephemeral 2-message contexts (no state pollution of agent.history).
  - Routes to AMD XDNA2 NPU via route_request() when available.
  - Transparently fails over to primary server (port 8000/8080) without user interruption.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from apex_harness.npu_backend import route_request
from apex_harness.critic import run_critic, CriticConfig, CriticResult, _resolve_model_for_endpoint


@dataclass
class SubagentConfig:
    name: str
    task_type: str
    description: str
    system_prompt: str
    model: str = field(default_factory=lambda: os.environ.get("APEX_SUBAGENT_MODEL", "qwen2.5-coder:1.5b"))
    npu_model: str = field(default_factory=lambda: os.environ.get("APEX_SUBAGENT_NPU_MODEL", "qwen2.5-coder-1.5b-int4"))
    use_npu: bool = True
    timeout: float = 15.0
    temperature: float = 0.2
    max_tokens: int = 512
    fallback_api_base: str = field(default_factory=lambda: os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1").rstrip("/"))


@dataclass
class SubagentResponse:
    content: str
    hardware: str  # "NPU XDNA2" or "GPU/CPU Fallback"
    endpoint: str
    model: str
    latency_ms: float
    success: bool
    error: Optional[str] = None


class BaseSubagent:
    """Base class for ephemeral subagents executing specialized tasks."""

    def __init__(self, config: SubagentConfig):
        self.cfg = config

    def get_target_endpoint(self) -> str:
        if self.cfg.use_npu:
            return route_request(task_type=self.cfg.task_type)
        return self.cfg.fallback_api_base

    def run(self, user_prompt: str) -> SubagentResponse:
        """
        Execute an ephemeral inference call without polluting main agent history.
        Tries NPU endpoint first, failing over to fallback endpoint if needed.
        """
        primary_endpoint = self.get_target_endpoint()
        endpoints = [primary_endpoint]
        if self.cfg.fallback_api_base and self.cfg.fallback_api_base != primary_endpoint:
            endpoints.append(self.cfg.fallback_api_base)

        start = time.perf_counter()
        last_error = None

        for ep in endpoints:
            is_npu = ":8090" in ep or "lemonade" in ep.lower()
            hw_label = "NPU XDNA2" if is_npu else "GPU/CPU (Local Server)"
            target_model = _resolve_model_for_endpoint(ep, CriticConfig(model=self.cfg.model, npu_model=self.cfg.npu_model))

            payload = {
                "model": target_model,
                "messages": [
                    {"role": "system", "content": self.cfg.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": self.cfg.max_tokens,
                "temperature": self.cfg.temperature
            }

            url = f"{ep}/chat/completions"
            headers = {"Content-Type": "application/json"}
            try:
                req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
                with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    content = data["choices"][0]["message"]["content"].strip()
                    latency = (time.perf_counter() - start) * 1000.0
                    return SubagentResponse(
                        content=content,
                        hardware=hw_label,
                        endpoint=ep,
                        model=target_model,
                        latency_ms=latency,
                        success=True
                    )
            except urllib.error.URLError as e:
                last_error = e
                continue
            except Exception as e:
                last_error = e
                continue

        latency = (time.perf_counter() - start) * 1000.0
        return SubagentResponse(
            content="",
            hardware="None",
            endpoint=primary_endpoint,
            model=self.cfg.model,
            latency_ms=latency,
            success=False,
            error=str(last_error)
        )


class CommitSubagent(BaseSubagent):
    """Generates Conventional Commits based on git changes."""

    def __init__(self):
        super().__init__(SubagentConfig(
            name="CommitSubagent",
            task_type="commit",
            description="Gera mensagens de commit convencionais a partir do git diff",
            system_prompt=(
                "You are an expert Git commit assistant. Generate a clean, conventional commit message "
                "(e.g., feat:, fix:, refactor:, docs:, chore:, test:) summarizing the provided changes.\n"
                "Format:\n"
                "<type>(<scope>): <short imperative description (50 chars max)>\n\n"
                "- Bullet point explaining key change\n"
                "- Another bullet point if needed\n\n"
                "Do not output markdown fences. Return ONLY the raw commit message text."
            ),
            max_tokens=256,
            temperature=0.2
        ))

    def generate_commit_message(self, git_status: str, git_diff_stat: str, git_diff_sample: str = "") -> SubagentResponse:
        prompt = (
            f"Git Status:\n{git_status}\n\n"
            f"Diff Stats:\n{git_diff_stat}\n"
        )
        if git_diff_sample:
            prompt += f"\nDiff Sample:\n{git_diff_sample[:3000]}\n"
        return self.run(prompt)


class TriageSubagent(BaseSubagent):
    """Distills noisy tasks and extracts structured requirements."""

    def __init__(self):
        super().__init__(SubagentConfig(
            name="TriageSubagent",
            task_type="triage",
            description="Destila requisitos técnicos e restrições sem poluir o histórico principal",
            system_prompt=(
                "You are a requirements analyst. Extract strictly the essential technical constraints, "
                "input/output formats, and risks from the task description.\n"
                "Be ultra-concise, numbering the core requirements."
            ),
            max_tokens=384,
            temperature=0.1
        ))

    def triage_task(self, raw_task: str) -> SubagentResponse:
        return self.run(f"Task Description:\n{raw_task[:4000]}")


class CriticSubagent:
    """Wrapper around critic.py for unified subagent management."""

    name = "CriticSubagent"
    task_type = "critic"
    description = "Revisa e valida diffs de código antes da escrita em disco"

    def __init__(self, config: Optional[CriticConfig] = None):
        self.config = config or CriticConfig()

    def review_diff(self, diff_text: str) -> CriticResult:
        return run_critic(diff_text, self.config)


def get_subagent_registry() -> List[Dict[str, Any]]:
    """Returns metadata for all available subagents."""
    from apex_harness.npu_detect import npu_available
    npu_status = npu_available()
    npu_hw_status = "NPU XDNA2 Ativa" if npu_status.usable else "Fallback GPU/CPU"

    return [
        {
            "name": "CriticSubagent",
            "role": "Segurança & Revisão de Diffs",
            "model_default": "qwen2.5-coder:1.5b",
            "model_npu": "qwen2.5-coder-1.5b-int4",
            "target_hw": "NPU XDNA2 (Porta 8090)",
            "status": npu_hw_status,
            "ephemeral": True,
        },
        {
            "name": "CommitSubagent",
            "role": "Geração de Mensagens Git Commit",
            "model_default": "qwen2.5-coder:1.5b",
            "model_npu": "qwen2.5-coder-1.5b-int4",
            "target_hw": "NPU XDNA2 (Porta 8090)",
            "status": npu_hw_status,
            "ephemeral": True,
        },
        {
            "name": "TriageSubagent",
            "role": "Destilação de Requisitos & Resumo Rápido",
            "model_default": "qwen2.5-coder:1.5b",
            "model_npu": "qwen2.5-coder-1.5b-int4",
            "target_hw": "NPU XDNA2 (Porta 8090)",
            "status": npu_hw_status,
            "ephemeral": True,
        },
    ]
