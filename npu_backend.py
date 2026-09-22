"""
Apex Harness — NPU Backend & Server Manager (npu_backend.py)
Manages the Lemonade + FastFlowLM runtime lifecycle for AMD XDNA2 NPU.
Provides OpenAI-compatible routing with fail-open fallback to Vulkan/llama-server.
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from apex_harness.npu_detect import NPUStatus, npu_available

DEFAULT_NPU_PORT = 8090
DEFAULT_NPU_HOST = "127.0.0.1"


class NPUServerManager:
    """
    Manages the lifecycle of a local Lemonade + FastFlowLM NPU server process.
    Follows the same Popen + logfile pattern used in optimized_launcher.py.
    """

    def __init__(
        self,
        port: int = DEFAULT_NPU_PORT,
        host: str = DEFAULT_NPU_HOST,
        model: Optional[str] = None,
        extra_args: Optional[List[str]] = None
    ):
        self.port = port
        self.host = host
        self.model = model
        self.extra_args = extra_args or []
        self.process: Optional[subprocess.Popen] = None
        self.log_file: Optional[Any] = None
        self.log_path: str = f"/tmp/lemonade-npu-{self.port}.log"

    @property
    def api_base(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def is_running(self) -> bool:
        """Check if server process is alive."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def check_health(self, timeout: float = 1.0) -> bool:
        """Ping the server /v1/models endpoint."""
        if not self.is_running():
            return False
        try:
            url = f"{self.api_base}/models"
            req = urllib.request.Request(url, headers={"User-Agent": "ApexHarness-NPU"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.getcode() == 200
        except Exception:
            return False

    def start(
        self,
        model: Optional[str] = None,
        wait_ready: bool = True,
        ready_timeout: float = 10.0
    ) -> bool:
        """
        Start the Lemonade FastFlowLM server process.
        Returns True if successfully started and ready; False on any failure (silent fallback).
        """
        # Safety gate: verify hardware and software prerequisites
        status = npu_available()
        if not status.usable or not status.lemonade_bin:
            return False

        if self.is_running():
            return True

        lemond_bin = shutil.which("lemond") or os.path.expanduser("~/.local/bin/lemond")
        if os.path.isfile(lemond_bin) and os.access(lemond_bin, os.X_OK):
            argv = [
                lemond_bin,
                "--host", self.host,
                "--port", str(self.port)
            ]
        else:
            argv = [
                status.lemonade_bin,
                "serve",
                "--host", self.host,
                "--port", str(self.port)
            ]
        if self.extra_args:
            argv.extend(self.extra_args)

        try:
            self.log_file = open(self.log_path, "w")
            env = os.environ.copy()
            # Ensure AMD NPU drivers and libraries are recognized
            env["AMDNPU_ENABLE_SVA"] = "1"

            self.process = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )

            if wait_ready:
                start_t = time.time()
                while time.time() - start_t < ready_timeout:
                    if self.process.poll() is not None:
                        # Process died prematurely
                        return False
                    if self.check_health(timeout=0.5):
                        return True
                    time.sleep(0.3)
                # Timed out waiting for ready
                return False

            return True
        except Exception:
            self.stop()
            return False

    def stop(self) -> None:
        """Gracefully stop the server process and close log file."""
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        self.process = None

        if self.log_file is not None:
            try:
                self.log_file.close()
            except Exception:
                pass
            self.log_file = None


# Global singleton manager for the session
_GLOBAL_NPU_MANAGER: Optional[NPUServerManager] = None


def get_npu_manager() -> NPUServerManager:
    global _GLOBAL_NPU_MANAGER
    if _GLOBAL_NPU_MANAGER is None:
        _GLOBAL_NPU_MANAGER = NPUServerManager()
        atexit.register(_GLOBAL_NPU_MANAGER.stop)
    return _GLOBAL_NPU_MANAGER


def route_request(task_type: str = "critic", default_url: Optional[str] = None) -> str:
    """
    Route request endpoint based on NPU availability.
    Returns:
      - NPU endpoint (http://127.0.0.1:8090/v1) if NPU is usable and running/available.
      - Fallback endpoint (default_url or OPENAI_API_BASE / local llama-server) otherwise.

    Always silent: never raises an error if NPU is not available.
    """
    fallback = (
        default_url
        or os.environ.get("OPENAI_API_BASE")
        or "http://localhost:8000/v1"
    ).rstrip("/")

    try:
        status = npu_available()
        if not status.usable:
            return fallback

        mgr = get_npu_manager()
        # If not already running, try non-blocking start
        if not mgr.is_running():
            ok = mgr.start(wait_ready=False)
            if not ok:
                return fallback

        return mgr.api_base
    except Exception:
        return fallback
