"""
apex_harness.optimized_launcher — drop-in replacement for
`launcher_common.launch_llama_server()`.

The original function is at ~/.local/lib/launcher_common.py:482. It hard-codes a
flag set that is wrong for this machine in four measurable ways (see hwtune.py for
the full analysis): a `taskset -c 0-7` that straddles both core types and both L3
domains, an inert ROCm environment block on a Vulkan binary, an `LD_PRELOAD` that
shadows the system libdrm, and no use of the speculative-decoding support the
installed build actually has.

HOW TO USE
----------
Option A - monkeypatch (no edit to launcher_common.py, keeps the old code as a
fallback)::

    # in launcher.py, immediately after `from launcher_common import (...)`
    import launcher_common
    from apex_harness.optimized_launcher import install
    install(launcher_common)

Option B - permanent patch: copy the body of `launch_llama_server` below over
launcher_common.py:482-559.

Both keep the original signature `(model_path, port, context, extra_args)` and the
original return value `(proc, log_file, log_path)` so nothing else needs changing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Make `apex_harness` importable even when this is loaded from ~/.local/lib.
_APEX_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APEX_ROOT not in sys.path:
    sys.path.insert(0, _APEX_ROOT)

from apex_harness.hwtune import (  # noqa: E402
    TuneOptions,
    detect_topology,
    format_plan,
    plan_server_command,
    profile_model,
    theoretical_decode_tps,
)

#: The launcher binary that ships with the Apex Harness install.
DEFAULT_SERVER = os.path.expanduser("~/.local/bin/llama-server")


def build_command(
    model_path: str,
    port: int = 8080,
    context: int = 32768,
    extra_args: Optional[List[str]] = None,
    server_bin: Optional[str] = None,
    backend: str = "auto",
    speculative: bool = True,
    verbose: bool = True,
) -> Tuple[List[str], dict]:
    """Plan an optimised llama-server invocation. Returns (argv, env)."""
    bin_path = server_bin or DEFAULT_SERVER

    # Thin models that live on a slow spinning disk (ntfs3 on /dev/sda) should be
    # read fully into RAM rather than mmap'd, otherwise a page eviction during
    # decode turns into a disk read in the middle of the token loop.
    on_slow_disk = False
    try:
        real = os.path.realpath(model_path)
        mounts = Path("/proc/mounts").read_text(errors="replace").splitlines()
        best = ""
        for line in mounts:
            parts = line.split()
            if len(parts) >= 3 and real.startswith(parts[1]) and len(parts[1]) > len(best):
                best, dev, fstype = parts[1], parts[0], parts[2]
        if best:
            # mmap of a file that fits in RAM is fine; the danger is eviction.
            size_gib = os.path.getsize(real) / 1024**3
            avail_gib = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3
            on_slow_disk = fstype in ("ntfs3", "ntfs", "fuseblk") and size_gib > avail_gib * 0.5
    except Exception:
        pass

    opts = TuneOptions(
        context=context,
        port=port,
        backend=backend,
        speculative=speculative,
        # Slow-disk detection: models on NTFS/spinning disks should be read
        # fully into RAM (mlock) rather than mmap'd, to avoid decode stalls
        # when the OS evicts pages back to the slow medium mid-generation.
        # On fast disks (ext4/btrfs/NVMe), mmap+mlock is strictly better:
        # fast startup and still pinned against swap.
        #
        # NA PRATICA o mlock nao acontece: o hard limit de RLIMIT_MEMLOCK e
        # 8192 KB e nao sobe sem root, entao o llama-server falha no primeiro
        # buffer ("Cannot allocate memory / Try increasing RLIMIT_MEMLOCK") e
        # segue em mmap normal. plan_server_command() deteta o limite e emite
        # -lm mmap, para o plano nao anunciar uma fixacao que nao existe.
        load_mode="mlock" if on_slow_disk else "mmap+mlock",
        # prio=2 nao chega a ser usado sem CAP_SYS_NICE: plan_server_command()
        # verifica CapPrm bit 23 e omite o --prio quando a capacidade falta.
        # Medido como utilizador normal, forcar o flag produzia
        #   "failed to set thread priority 2 : Operation not permitted"
        # 6180 vezes num unico log, sem alterar o escalonamento. Fica pedido aqui
        # para o caso de o binario ter setcap; o hwtune decide se sai.
        prio=2,
        prio_batch=2,
    )
    argv, env, notes = plan_server_command(model_path, bin_path, opts, detect_topology())

    if extra_args:
        argv += list(extra_args)

    if verbose:
        try:
            prof = profile_model(model_path)
            print(format_plan(argv, notes))
            print(f"  model: {os.path.basename(model_path)} "
                  f"({prof.size_gib:.1f} GiB, arch={prof.arch or '?'}, "
                  f"moe={prof.is_moe}, mtp={prof.has_mtp})")
            print(f"  bandwidth-bound decode ceiling: {theoretical_decode_tps(prof):.1f} t/s "
                  f"(CPU; @120 GB/s x 0.55. The Vulkan path measured ~2x higher on a dense 7B.)\n")
        except Exception:
            print(f"[apex] launching: {' '.join(argv)}")

    return argv, env


def launch_llama_server(
    model_path: str,
    port: int = 8080,
    context: int = 32768,
    extra_args: Optional[List[str]] = None,
    server_bin: Optional[str] = None,
    backend: str = "auto",
    speculative: bool = True,
    verbose: bool = True,
    log_to_file: bool = False,
) -> Tuple[subprocess.Popen, Any, str]:
    """
    Start llama-server with hardware-aware flags.

    Drop-in compatible with launcher_common.launch_llama_server: same positional
    signature (model_path, port, context, extra_args) and the same
    (proc, log_file, log_path) return tuple.

    `log_to_file=True` sends the child's stdout straight to `log_path` instead of
    to an OS pipe. Use it for callers that do NOT run a drain loop -- see below.
    """
    argv, env = build_command(
        model_path, port=port, context=context, extra_args=extra_args,
        server_bin=server_bin, backend=backend, speculative=speculative, verbose=verbose,
    )

    log_path = f"/tmp/llama-server-{port}.log"

    if log_to_file:
        # Ficheiro, NAO PIPE.
        #
        # Com stdout=PIPE e sem ninguem a ler, o llama-server enche os ~64 KiB do
        # buffer do pipe durante o carregamento e a partir dai BLOQUEIA na
        # escrita. Fica "a carregar" para sempre, o wait_for_server_ready
        # desiste, o launcher abre o browser e sai -- e ao sair fecha o lado de
        # leitura do pipe, portanto a linha de log seguinte da EPIPE e o
        # servidor morre sem deixar rasto.
        #
        # Foi isto que fez o atalho "Apex Web Harness" parar o
        # apex-backend.service as 15:50 e deixar a porta 8080 morta, com o
        # harness a responder "offline" e o botao de enviar inutil. O atalho de
        # terminal nao sofria disto porque chama stream_server_output() e drena
        # o pipe; o do web nao chamava.
        log_file = open(log_path, "ab", buffering=0)
        proc = subprocess.Popen(
            argv,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    else:
        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

    # Warm start NPU manager in background if NPU is usable (parallel warm-up)
    try:
        from apex_harness.npu_detect import npu_available
        from apex_harness.npu_backend import get_npu_manager
        if npu_available().usable:
            mgr = get_npu_manager()
            if not mgr.is_running():
                mgr.start(wait_ready=False)
    except Exception:
        pass

    return proc, log_file, log_path


def install(module: Any = None, **kwargs: Any) -> None:
    """
    Monkeypatch `launcher_common` (or any module with the same attribute names)
    so its existing call sites pick up the optimised launcher.

    Keeps `get_llama_server` / `needs_flash_server` intact so the flash-fork
    routing for Flash-Next architectures still works.
    """
    if module is None:
        import launcher_common as module  # type: ignore

    original = getattr(module, "launch_llama_server", None)

    def _patched(model_path, port=8080, context=32768, extra_args=None, **call_kwargs):
        # Preserve the flash-fork routing decision made by launcher_common.
        bin_path = DEFAULT_SERVER
        try:
            if hasattr(module, "get_llama_server"):
                bin_path = module.get_llama_server(model_path)
        except Exception:
            pass
        # `**call_kwargs` (e nao so `**kwargs`) para que o chamador possa pedir
        # log_to_file=True. Sem isto, um argumento novo passado pelo launcher
        # rebentava com TypeError, o atalho engolia a excecao num `except
        # Exception: print(...)` e o servidor simplesmente nao subia.
        merged = {**kwargs, **call_kwargs}
        return launch_llama_server(
            model_path, port=port, context=context, extra_args=extra_args,
            server_bin=bin_path, **merged,
        )

    _patched.__wrapped_original__ = original      # type: ignore[attr-defined]
    _patched.__doc__ = (original.__doc__ or "") + "\n\n[replaced by apex_harness.optimized_launcher]"
    module.launch_llama_server = _patched


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Plan (or run) an optimised llama-server")
    ap.add_argument("model")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--backend", default="auto", choices=["auto", "vulkan", "cpu"])
    ap.add_argument("--server-bin", default=DEFAULT_SERVER)
    ap.add_argument("--no-speculative", action="store_true")
    ap.add_argument("--exec", action="store_true")
    a = ap.parse_args()

    argv, env = build_command(
        a.model, port=a.port, context=a.context, server_bin=a.server_bin,
        backend=a.backend, speculative=not a.no_speculative,
    )
    if a.exec:
        os.execve(argv[0], argv, env)
