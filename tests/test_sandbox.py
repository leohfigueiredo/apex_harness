import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from apex_harness.sandbox import SandboxConfig, SandboxExecutor, SandboxResult


def test_sandbox_probe_backend():
    # If bwrap is on machine, it should probe bwrap
    executor = SandboxExecutor(SandboxConfig(backend="auto"))
    if shutil.which("bwrap"):
        assert executor.backend == "bwrap"
    else:
        assert executor.backend in ("firejail", "restricted")

    # Mock no bwrap or firejail
    with patch("shutil.which", return_value=None):
        mock_executor = SandboxExecutor(SandboxConfig(backend="auto"))
        assert mock_executor.backend == "restricted"


def test_sandbox_path_safety():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SandboxConfig(workspace_dir=tmpdir)
        executor = SandboxExecutor(config)

        # Inside workspace
        inside_file = Path(tmpdir) / "sub" / "file.txt"
        assert executor.is_path_safe(str(inside_file)) is True

        # Outside workspace
        outside_file = Path("/etc/passwd")
        assert executor.is_path_safe(str(outside_file)) is False


def test_sandbox_dry_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SandboxConfig(workspace_dir=tmpdir, dry_run=True)
        executor = SandboxExecutor(config)

        res = executor.execute("rm -rf myfile.txt", cwd=tmpdir)
        assert res.dry_run is True
        assert res.exit_code == 0
        assert "preview" in res.diff.lower()


def test_sandbox_restricted_dangerous_command():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SandboxConfig(workspace_dir=tmpdir, backend="restricted")
        executor = SandboxExecutor(config)

        res = executor.execute("rm -rf /")
        assert res.exit_code == 1
        assert "forbidden dangerous pattern" in res.stderr.lower()


def test_sandbox_diff_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "sample.py"
        test_file.write_text("print('hello world')\n", encoding="utf-8")

        config = SandboxConfig(workspace_dir=tmpdir)
        executor = SandboxExecutor(config)

        diff = executor.generate_diff(str(test_file), "print('hello universe')\n")
        assert "-print('hello world')" in diff
        assert "+print('hello universe')" in diff


def test_sandbox_execution_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SandboxConfig(workspace_dir=tmpdir)
        executor = SandboxExecutor(config)

        res = executor.execute("echo 'apex_sandbox_test'", cwd=tmpdir)
        assert res.exit_code == 0
        assert "apex_sandbox_test" in res.stdout
