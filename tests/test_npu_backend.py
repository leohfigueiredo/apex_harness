import os
from unittest.mock import patch, MagicMock
import pytest

from apex_harness.npu_detect import NPUStatus
from apex_harness.npu_backend import (
    NPUServerManager,
    route_request,
    get_npu_manager
)


def test_manager_init():
    mgr = NPUServerManager(port=8099, host="127.0.0.1", model="test-model")
    assert mgr.port == 8099
    assert mgr.api_base == "http://127.0.0.1:8099/v1"
    assert mgr.is_running() is False
    assert mgr.log_path == "/tmp/lemonade-npu-8099.log"


def test_manager_start_not_usable():
    mock_status = NPUStatus(
        driver_loaded=True,
        kernel_version="7.0.0",
        kernel_ok=True,
        lemonade_installed=False,
        usable=False,
        lemonade_bin=None
    )

    with patch("apex_harness.npu_backend.npu_available", return_value=mock_status):
        mgr = NPUServerManager()
        started = mgr.start()
        assert started is False
        assert mgr.is_running() is False


def test_manager_start_and_stop_mocked():
    mock_status = NPUStatus(
        driver_loaded=True,
        kernel_version="7.0.0",
        kernel_ok=True,
        lemonade_installed=True,
        usable=True,
        lemonade_bin="/usr/bin/lemonade"
    )

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # Process is running

    with patch("apex_harness.npu_backend.npu_available", return_value=mock_status), \
         patch("subprocess.Popen", return_value=mock_proc), \
         patch("apex_harness.npu_backend.NPUServerManager.check_health", return_value=True), \
         patch("builtins.open", MagicMock()):
        
        mgr = NPUServerManager(port=8095)
        started = mgr.start(wait_ready=True)
        assert started is True
        assert mgr.is_running() is True
        assert mgr.api_base == "http://127.0.0.1:8095/v1"

        mgr.stop()
        assert mock_proc.terminate.called
        assert mgr.is_running() is False


def test_route_request_fallback_when_unavailable():
    mock_status = NPUStatus(
        driver_loaded=True,
        kernel_version="7.0.0",
        kernel_ok=True,
        lemonade_installed=False,
        usable=False
    )

    with patch("apex_harness.npu_backend.npu_available", return_value=mock_status):
        url = route_request(task_type="critic", default_url="http://custom-server:8080/v1")
        assert url == "http://custom-server:8080/v1"

        # Default fallback without explicit url
        url_default = route_request(task_type="critic")
        assert "http://" in url_default
        assert "8090" not in url_default


def test_route_request_npu_when_available():
    mock_status = NPUStatus(
        driver_loaded=True,
        kernel_version="7.0.0",
        kernel_ok=True,
        lemonade_installed=True,
        usable=True,
        lemonade_bin="/usr/bin/lemonade"
    )

    mock_mgr = MagicMock()
    mock_mgr.is_running.return_value = True
    mock_mgr.api_base = "http://127.0.0.1:8090/v1"

    with patch("apex_harness.npu_backend.npu_available", return_value=mock_status), \
         patch("apex_harness.npu_backend.get_npu_manager", return_value=mock_mgr):
        url = route_request(task_type="critic", default_url="http://custom-server:8080/v1")
        assert url == "http://127.0.0.1:8090/v1"


def test_route_request_silent_exception_handling():
    with patch("apex_harness.npu_backend.npu_available", side_effect=RuntimeError("Hardware error")):
        url = route_request(default_url="http://fallback:8000/v1")
        assert url == "http://fallback:8000/v1"
