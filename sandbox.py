"""
Apex Harness — Sandbox Execution Engine (sandbox.py)
Provides isolation for shell commands and filesystem operations.
Supports:
1. Bubblewrap (bwrap) - Linux namespace isolation (mount, pid, net)
2. Firejail - Fallback containerization
3. Restricted subprocess - In-process allowlist / directory boundary enforcement
"""

import os
import shutil
import subprocess
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict


@dataclass
class SandboxConfig:
    workspace_dir: str = field(default_factory=lambda: os.environ.get("APEX_WORKSPACE", os.getcwd()))
    allow_network: bool = True
    allowed_envs: List[str] = field(default_factory=lambda: [
        "PATH", "HOME", "TERM", "LANG", "LC_ALL", "USER", "SHELL", "TMPDIR", "PYTHONPATH"
    ])
    dry_run: bool = field(default_factory=lambda: os.environ.get("APEX_DRY_RUN", "0").lower() in ("1", "true", "yes"))
    timeout: int = 60
    backend: Optional[str] = None  # 'auto', 'bwrap', 'firejail', 'restricted', 'none'


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    backend: str
    dry_run: bool = False
    diff: Optional[str] = None

    def to_tool_output(self) -> str:
        parts = []
        if self.dry_run:
            parts.append(f"[SANDBOX DRY-RUN ({self.backend})] Command simulated, no execution occurred.")
            if self.diff:
                parts.append(f"DIFF PREVIEW:\n{self.diff}")
            return "\n\n".join(parts)

        if self.stdout:
            parts.append(f"STDOUT:\n{self.stdout.strip()}")
        if self.stderr:
            parts.append(f"STDERR:\n{self.stderr.strip()}")
        parts.append(f"Exit Code: {self.exit_code} (Backend: {self.backend})")
        return "\n\n".join(parts)


class SandboxExecutor:
    """Executes commands inside an isolated environment using bwrap, firejail, or restricted subproc."""

    DANGEROUS_PATTERNS = [
        "rm -rf /", "rm -rf /*", "mkfs", "dd if=/dev", ":(){ :|:& };:",
        "> /dev/sda", "> /dev/nvme", "chmod -R 777 /", "chown -R"
    ]

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self.workspace_dir = str(Path(self.config.workspace_dir).resolve())
        self.backend = self._probe_backend()

    def _probe_backend(self) -> str:
        env_backend = os.environ.get("APEX_SANDBOX_BACKEND", "").lower().strip()
        requested = (self.config.backend or env_backend or "auto").lower()

        if requested in ("bwrap", "bubblewrap") and shutil.which("bwrap"):
            return "bwrap"
        if requested == "firejail" and shutil.which("firejail"):
            return "firejail"
        if requested == "restricted":
            return "restricted"
        if requested == "none":
            return "none"

        # Auto detection priority: bwrap -> firejail -> restricted
        if shutil.which("bwrap"):
            return "bwrap"
        if shutil.which("firejail"):
            return "firejail"
        return "restricted"

    def is_path_safe(self, path: str) -> bool:
        """Verify if a path is located within the designated workspace directory."""
        try:
            target = Path(path).resolve()
            ws = Path(self.workspace_dir).resolve()
            return target == ws or ws in target.parents
        except Exception:
            return False

    def generate_diff(self, file_path: str, new_content: str) -> str:
        """Generate unified diff between existing file content and proposed new content."""
        p = Path(file_path).resolve()
        if p.exists() and p.is_file():
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    old_lines = f.readlines()
            except Exception:
                old_lines = []
        else:
            old_lines = []

        new_lines = new_content.splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{p.name}" if p.exists() else "/dev/null",
            tofile=f"b/{p.name}",
            lineterm=""
        )
        return "\n".join(diff)

    def filter_environment(self) -> Dict[str, str]:
        """Return safe subset of environment variables."""
        return {
            k: v for k, v in os.environ.items()
            if k in self.config.allowed_envs
        }

    def execute(self, command: str, cwd: Optional[str] = None, timeout: Optional[int] = None) -> SandboxResult:
        """Execute command in selected sandbox backend."""
        target_cwd = str(Path(cwd or self.workspace_dir).resolve())
        timeout_sec = timeout if timeout is not None else self.config.timeout

        # Safety check for path containment if restricted
        if not self.is_path_safe(target_cwd):
            return SandboxResult(
                stdout="",
                stderr=f"Security Error: Working directory '{target_cwd}' is outside sandbox workspace '{self.workspace_dir}'.",
                exit_code=1,
                backend=self.backend
            )

        if self.config.dry_run:
            return SandboxResult(
                stdout="",
                stderr="",
                exit_code=0,
                backend=self.backend,
                dry_run=True,
                diff=f"[Command preview]: {command} (in {target_cwd})"
            )

        if self.backend == "bwrap":
            return self._run_bwrap(command, target_cwd, timeout_sec)
        elif self.backend == "firejail":
            return self._run_firejail(command, target_cwd, timeout_sec)
        elif self.backend == "none":
            return self._run_unrestricted(command, target_cwd, timeout_sec)
        else:
            return self._run_restricted(command, target_cwd, timeout_sec)

    def _run_bwrap(self, command: str, cwd: str, timeout: int) -> SandboxResult:
        """Execute command using bubblewrap (bwrap). System is ro-bind, only workspace is rw bind."""
        bwrap_cmd = [
            "bwrap",
            "--ro-bind", "/", "/",
            "--dev", "/dev",
            "--proc", "/proc",
            "--tmpfs", "/tmp",
            "--bind", self.workspace_dir, self.workspace_dir,
            "--chdir", cwd
        ]

        if not self.config.allow_network:
            bwrap_cmd.append("--unshare-net")

        bwrap_cmd.extend(["/bin/bash", "-c", command])

        try:
            res = subprocess.run(
                bwrap_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self.filter_environment()
            )
            return SandboxResult(
                stdout=res.stdout,
                stderr=res.stderr,
                exit_code=res.returncode,
                backend="bwrap"
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                stdout="",
                stderr=f"Error: Command timed out after {timeout} seconds inside bwrap.",
                exit_code=124,
                backend="bwrap"
            )
        except Exception as e:
            return SandboxResult(
                stdout="",
                stderr=f"Error running bwrap: {str(e)}",
                exit_code=1,
                backend="bwrap"
            )

    def _run_firejail(self, command: str, cwd: str, timeout: int) -> SandboxResult:
        """Execute command using firejail sandbox."""
        fj_cmd = [
            "firejail",
            "--quiet",
            f"--whitelist={self.workspace_dir}",
            f"--private-cwd={cwd}"
        ]
        if not self.config.allow_network:
            fj_cmd.append("--net=none")

        fj_cmd.extend(["/bin/bash", "-c", command])

        try:
            res = subprocess.run(
                fj_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self.filter_environment()
            )
            return SandboxResult(
                stdout=res.stdout,
                stderr=res.stderr,
                exit_code=res.returncode,
                backend="firejail"
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                stdout="",
                stderr=f"Error: Command timed out after {timeout} seconds inside firejail.",
                exit_code=124,
                backend="firejail"
            )
        except Exception as e:
            return SandboxResult(
                stdout="",
                stderr=f"Error running firejail: {str(e)}",
                exit_code=1,
                backend="firejail"
            )

    def _run_restricted(self, command: str, cwd: str, timeout: int) -> SandboxResult:
        """Fallback restricted executor: disallows dangerous commands and cleans env."""
        cmd_lower = command.lower()
        for dangerous in self.DANGEROUS_PATTERNS:
            if dangerous in cmd_lower:
                return SandboxResult(
                    stdout="",
                    stderr=f"Security Error: Command contains forbidden dangerous pattern '{dangerous}'.",
                    exit_code=1,
                    backend="restricted"
                )

        try:
            res = subprocess.run(
                ["/bin/bash", "-c", command],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self.filter_environment()
            )
            return SandboxResult(
                stdout=res.stdout,
                stderr=res.stderr,
                exit_code=res.returncode,
                backend="restricted"
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                stdout="",
                stderr=f"Error: Command timed out after {timeout} seconds.",
                exit_code=124,
                backend="restricted"
            )
        except Exception as e:
            return SandboxResult(
                stdout="",
                stderr=f"Error running restricted execution: {str(e)}",
                exit_code=1,
                backend="restricted"
            )

    def _run_unrestricted(self, command: str, cwd: str, timeout: int) -> SandboxResult:
        """Unrestricted execution (when APEX_SANDBOX_BACKEND=none)."""
        try:
            res = subprocess.run(
                ["/bin/bash", "-c", command],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return SandboxResult(
                stdout=res.stdout,
                stderr=res.stderr,
                exit_code=res.returncode,
                backend="none"
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                stdout="",
                stderr=f"Error: Command timed out after {timeout} seconds.",
                exit_code=124,
                backend="none"
            )
        except Exception as e:
            return SandboxResult(
                stdout="",
                stderr=f"Error executing command: {str(e)}",
                exit_code=1,
                backend="none"
            )


# Global singleton instance for easy tool access
_GLOBAL_SANDBOX: Optional[SandboxExecutor] = None

def get_sandbox_executor(config: Optional[SandboxConfig] = None) -> SandboxExecutor:
    global _GLOBAL_SANDBOX
    if _GLOBAL_SANDBOX is None or config is not None:
        _GLOBAL_SANDBOX = SandboxExecutor(config)
    return _GLOBAL_SANDBOX
