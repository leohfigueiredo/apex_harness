"""
Apex Harness — NPU Detection Module (npu_detect.py)
Detects AMD XDNA2 NPU hardware, kernel driver status (amdxdna),
and Lemonade / FastFlowLM runtime availability on Linux.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class NPUStatus:
    driver_loaded: bool
    kernel_version: str
    kernel_ok: bool
    lemonade_installed: bool
    usable: bool
    device_found: bool = False
    device_info: Optional[str] = None
    lemonade_bin: Optional[str] = None
    lemonade_version: Optional[str] = None
    fastflowlm_available: bool = False
    missing_requirements: List[str] = field(default_factory=list)
    report: str = ""


def _parse_version(v_str: str) -> Tuple[int, ...]:
    """Extract numeric components from version string (e.g. '7.0.0-31-generic' -> (7, 0, 0))."""
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", v_str.strip())
    if not match:
        return (0, 0, 0)
    return tuple(int(x) if x is not None else 0 for x in match.groups())


def check_kernel(min_version: str = "6.10.0") -> Tuple[str, bool]:
    """Check running kernel version against minimum required for AMD XDNA2."""
    try:
        ver = os.uname().release
    except Exception:
        ver = "0.0.0"

    current = _parse_version(ver)
    required = _parse_version(min_version)
    is_ok = current >= required
    return ver, is_ok


def check_driver() -> Tuple[bool, bool, Optional[str]]:
    """
    Check if the amdxdna driver is loaded and NPU hardware device is detected.
    Runs non-privileged commands (dmesg with fallback to journalctl -kg xdna).
    Also inspects /sys/class/accel/ and /dev/accel/.
    Returns (driver_loaded, device_found, device_info).
    """
    driver_loaded = False
    device_found = False
    device_info = None

    # 1. Check direct accel device nodes in sysfs
    accel_dir = "/sys/class/accel"
    if os.path.exists(accel_dir):
        try:
            for entry in os.listdir(accel_dir):
                driver_link = os.path.join(accel_dir, entry, "device", "driver")
                if os.path.exists(driver_link):
                    real_drv = os.path.realpath(driver_link)
                    if "amdxdna" in real_drv.lower():
                        driver_loaded = True
                        device_found = True
                        device_info = f"Found via sysfs: {entry} ({os.path.basename(real_drv)})"
                        break
        except Exception:
            pass

    # 2. Check kernel logs via dmesg or journalctl
    log_output = ""
    try:
        p = subprocess.run(
            ["dmesg"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3
        )
        if p.returncode == 0 and p.stdout:
            log_output = p.stdout
    except Exception:
        pass

    if not log_output:
        try:
            p = subprocess.run(
                ["journalctl", "-kg", "xdna", "-n", "50", "--no-pager"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3
            )
            if p.returncode == 0 and p.stdout:
                log_output = p.stdout
        except Exception:
            pass

    if log_output:
        for line in log_output.splitlines():
            if "amdxdna" in line.lower() or "amdnpu" in line.lower():
                driver_loaded = True
                if "0000:" in line or "firmware" in line.lower() or "initialized" in line.lower():
                    device_found = True
                    device_info = line.strip()

    # 3. Check /dev/accel node
    if os.path.exists("/dev/accel"):
        device_found = True

    return driver_loaded, device_found, device_info


def check_lemonade() -> Tuple[bool, Optional[str], Optional[str], bool]:
    """
    Check if Lemonade binary/service is installed and supports FastFlowLM.
    Returns (installed, bin_path, version, fastflowlm_available).
    """
    bin_path = shutil.which("lemonade")
    if not bin_path:
        # Check standard user and local paths
        candidates = [
            os.path.expanduser("~/.local/bin/lemonade"),
            "/usr/local/bin/lemonade",
            "/usr/bin/lemonade",
            os.path.expanduser("~/.lemonade/bin/lemonade"),
        ]
        for c in candidates:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                bin_path = c
                break

    if not bin_path:
        return False, None, None, False

    version_str = None
    has_fastflow = False

    try:
        p = subprocess.run(
            [bin_path, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3
        )
        version_str = (p.stdout or p.stderr).strip()
    except Exception:
        version_str = "unknown"

    try:
        p_help = subprocess.run(
            [bin_path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3
        )
        help_text = (p_help.stdout or "") + (p_help.stderr or "")
        if "fastflow" in help_text.lower() or "npu" in help_text.lower():
            has_fastflow = True
    except Exception:
        pass

    if not has_fastflow:
        flm_bin = shutil.which("flm") or os.path.expanduser("~/.local/bin/flm")
        if os.path.isfile(flm_bin) and os.access(flm_bin, os.X_OK):
            has_fastflow = True

    return True, bin_path, version_str, has_fastflow


def npu_available(min_kernel: str = "6.10.0") -> NPUStatus:
    """
    Public entry point for NPU discovery.
    Evaluates hardware, kernel driver, and runtime stack.
    Safe and non-destructive: never attempts to install or modify system state.
    """
    k_ver, k_ok = check_kernel(min_version=min_kernel)
    drv_loaded, dev_found, dev_info = check_driver()
    lem_installed, lem_bin, lem_ver, fastflow = check_lemonade()

    missing = []
    if not drv_loaded:
        missing.append("Driver amdxdna não carregado (verifique firmware amdnpu e suporte do kernel)")
    if not k_ok:
        missing.append(f"Kernel {k_ver} incompatível (mínimo exigido: {min_kernel})")
    if not lem_installed:
        missing.append("Serviço/Binário Lemonade 10.0+ não encontrado no PATH")
    elif not fastflow:
        missing.append("Runtime FastFlowLM não detectado na instalação do Lemonade")

    usable = drv_loaded and k_ok and lem_installed

    # Generate user-friendly diagnostic report
    lines = [
        "=== Diagnóstico de Hardware NPU (AMD XDNA2) ===",
        f"  • Versão do Kernel: {k_ver} ({'OK' if k_ok else 'INCOMPATÍVEL'})",
        f"  • Driver amdxdna:   {'CARREGADO' if drv_loaded else 'NÃO DETECTADO'}",
    ]
    if dev_info:
        lines.append(f"  • Dispositivo NPU:   {dev_info}")
    lines.append(f"  • Lemonade 10.0+:    {'INSTALADO (' + str(lem_bin) + ')' if lem_installed else 'NÃO INSTALADO'}")
    lines.append(f"  • Runtime FastFlowLM:{'DISPONÍVEL' if fastflow else 'NÃO DISPONÍVEL'}")
    lines.append(f"  • Status Operacional:{'PRONTO PARA USO' if usable else 'NÃO DISPONÍVEL (Fallback Ativo)'}")
    lines.append("\nCompatibilidade de Modelos na NPU:")
    lines.append("  • Formato requerido: ONNX Runtime GenAI / Vitis AI (AWQ int4 / FastFlowLM compiled)")
    lines.append("  • Modelos ideais:    Dense compactos (0.5B a 7B, ex: Qwen2.5-Coder-1.5B/7B, DeepSeek-R1-Distill-1.5B)")
    lines.append("  • Limitação:         GGUF e MoEs gigantes (ex: 131B / DeepSeek 671B) rodam na GPU/CPU via llama-server")

    if missing:
        lines.append("\nPendências para Habilitação do NPU:")
        for m in missing:
            lines.append(f"  [!] {m}")

    report = "\n".join(lines)

    return NPUStatus(
        driver_loaded=drv_loaded,
        kernel_version=k_ver,
        kernel_ok=k_ok,
        lemonade_installed=lem_installed,
        usable=usable,
        device_found=dev_found,
        device_info=dev_info,
        lemonade_bin=lem_bin,
        lemonade_version=lem_ver,
        fastflowlm_available=fastflow,
        missing_requirements=missing,
        report=report
    )


if __name__ == "__main__":
    status = npu_available()
    print(status.report)
    sys.exit(0 if status.usable else 1)
