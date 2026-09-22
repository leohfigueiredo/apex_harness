import os
from unittest.mock import patch, MagicMock
import subprocess

from apex_harness.npu_detect import (
    NPUStatus,
    _parse_version,
    check_kernel,
    check_driver,
    check_lemonade,
    npu_available
)


def test_parse_version():
    assert _parse_version("7.0.0-31-generic") == (7, 0, 0)
    assert _parse_version("6.10.1") == (6, 10, 1)
    assert _parse_version("6.9.5") == (6, 9, 5)
    assert _parse_version("invalid") == (0, 0, 0)


def test_check_kernel():
    ver, ok = check_kernel(min_version="6.10.0")
    assert isinstance(ver, str)
    # The machine runs 7.0.0, so this should pass
    assert ok is True

    ver_fail, ok_fail = check_kernel(min_version="99.0.0")
    assert ok_fail is False


def test_check_driver_mock_loaded():
    mock_run = MagicMock()
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "kernel: [drm] Initialized amdxdna_accel_driver 0.7.0 for 0000:c7:00.1"

    with patch("subprocess.run", mock_run):
        loaded, found, info = check_driver()
        assert loaded is True
        assert found is True
        assert "amdxdna" in info


def test_check_driver_mock_not_loaded():
    mock_run = MagicMock()
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "kernel: unrelated log line"

    with patch("subprocess.run", mock_run), patch("os.path.exists", return_value=False):
        loaded, found, info = check_driver()
        assert loaded is False
        assert found is False
        assert info is None


def test_check_lemonade_not_installed():
    with patch("shutil.which", return_value=None), patch("os.path.isfile", return_value=False):
        installed, bin_path, ver, fastflow = check_lemonade()
        assert installed is False
        assert bin_path is None
        assert ver is None
        assert fastflow is False


def test_check_lemonade_mock_installed():
    def mock_subprocess(cmd, **kwargs):
        res = MagicMock()
        res.returncode = 0
        if "--version" in cmd:
            res.stdout = "lemonade 10.0.1"
            res.stderr = ""
        elif "--help" in cmd:
            res.stdout = "Lemonade CLI with FastFlowLM NPU accelerator"
            res.stderr = ""
        return res

    with patch("shutil.which", return_value="/usr/bin/lemonade"), \
         patch("subprocess.run", side_effect=mock_subprocess):
        installed, bin_path, ver, fastflow = check_lemonade()
        assert installed is True
        assert bin_path == "/usr/bin/lemonade"
        assert ver == "lemonade 10.0.1"
        assert fastflow is True


def test_npu_available_full_mock():
    with patch("apex_harness.npu_detect.check_kernel", return_value=("7.0.0", True)), \
         patch("apex_harness.npu_detect.check_driver", return_value=(True, True, "amdxdna 0000:c7:00.1")), \
         patch("apex_harness.npu_detect.check_lemonade", return_value=(True, "/usr/bin/lemonade", "10.0.0", True)):
        status = npu_available()
        assert status.usable is True
        assert status.driver_loaded is True
        assert status.kernel_ok is True
        assert status.lemonade_installed is True
        assert status.fastflowlm_available is True
        assert len(status.missing_requirements) == 0
        assert "PRONTO PARA USO" in status.report


def test_npu_available_missing_lemonade():
    with patch("apex_harness.npu_detect.check_kernel", return_value=("7.0.0", True)), \
         patch("apex_harness.npu_detect.check_driver", return_value=(True, True, "amdxdna 0000:c7:00.1")), \
         patch("apex_harness.npu_detect.check_lemonade", return_value=(False, None, None, False)):
        status = npu_available()
        assert status.usable is False
        assert status.driver_loaded is True
        assert status.lemonade_installed is False
        assert any("Lemonade" in m for m in status.missing_requirements)
        assert "NÃO DISPONÍVEL (Fallback Ativo)" in status.report


def test_npu_available_live():
    """Verify live discovery on the host executes cleanly without raising any exception."""
    status = npu_available()
    assert isinstance(status, NPUStatus)
    assert isinstance(status.kernel_version, str)
    assert isinstance(status.driver_loaded, bool)
    assert isinstance(status.usable, bool)
    assert isinstance(status.report, str)
    assert len(status.report) > 0
