import subprocess
import tempfile
from pathlib import Path

from apex_harness.tools import (
    git_status,
    git_diff,
    git_log,
    git_branch,
    git_commit,
)


def _init_temp_repo(repo_dir: str):
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Apex Tester"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "tester@apex.local"], cwd=repo_dir, check=True)


def test_git_tools_workflow():
    with tempfile.TemporaryDirectory() as tmpdir:
        _init_temp_repo(tmpdir)

        # 1. Initial status in empty repo
        status = git_status(tmpdir)
        assert "clean" in status.lower() or "no commits" in status.lower() or "master" in status or "main" in status

        # 2. Create file and commit
        f = Path(tmpdir) / "hello.py"
        f.write_text("print('hello')", encoding="utf-8")

        # Status should show untracked
        status_untracked = git_status(tmpdir)
        assert "hello.py" in status_untracked

        # 3. Commit with add_all=True
        commit_res = git_commit(tmpdir, message="Initial test commit", add_all=True)
        assert "Initial test commit" in commit_res or "root-commit" in commit_res

        # 4. Check log
        log_res = git_log(tmpdir, n=5)
        assert "Initial test commit" in log_res

        # 5. Modify file and check diff
        f.write_text("print('hello world')", encoding="utf-8")
        diff_res = git_diff(tmpdir)
        assert "+print('hello world')" in diff_res

        # 6. Branch operations
        branch_create = git_branch(tmpdir, create="feature-1")
        assert "feature-1" in branch_create

        branch_list = git_branch(tmpdir)
        assert "feature-1" in branch_list
