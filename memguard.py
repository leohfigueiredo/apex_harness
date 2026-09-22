"""
Pre-flight memory guard for this APU.

Why this file exists
--------------------
On this machine a model that is too large does not fail with a tidy
"out of memory" error from llama.cpp. It fails in one of two ways, both of
which take the desktop down with them:

  2026-09-17 21:47:32  kernel BUG at amdgpu_vm_pt_free+0x61/0xd0 [amdgpu]
                       -> "Fixing recursive fault but reboot is needed!"
                       -> hard reboot. Real kernel use-after-free in the amdgpu
                          VM page-table teardown, reached because the GPU was
                          driven to exhaustion and then every DRM client
                          (Xwayland, nautilus, gnome-shell) closed its fd.

  2026-09-18 14:44:26  kernel global OOM, 19 seconds of kill sweeps.
                       8 processes killed including lm-studio,
                       antigravity-ide, chrome-devtools AND
                       org.gnome.Shell@ubuntu.service -> the session died and
                       gdm restarted the desktop at 14:45:09. Looks exactly
                       like a reboot; uptime proves it was not one.

The kernel OOM killer has no idea which process is the pig, so it kills by
oom_score across the whole system. `systemd-oomd` is enabled but only acts on
cgroup memory pressure, which never gets a chance to build before the kernel
sweeps. Nothing in the current setup protects the session.

So the guard has to run BEFORE the model process starts, which is what this
module does. It is cheap (reads /proc/meminfo + two sysfs files) and safe.

The two pools, and why both matter
----------------------------------
    GPU pool = mem_info_vram_total x gpu_frac   (BIOS UMA carve-out)
    CPU pool = MemAvailable - reserve            (swap NOT counted, see below)

Swap is excluded on purpose. Model weights are streamed once per token, so any
weight byte that lands in swap is re-read from disk on every decode step: the
machine does not slow down, it stops responding. Swap is for idle desktop
pages. Counting it as model headroom is precisely the mistake that makes a
"fits on paper" plan die in practice.

GTT is deliberately NOT counted. Measured on this box:

    48 GiB carve-out:  Vulkan -ngl 999  13.69 t/s decode  |  CPU -ngl 0   6.78 t/s
    0.5 GiB carve-out: Vulkan -ngl 999   5.62 t/s decode  |  CPU -ngl 0  13.81 t/s

Weights that spill out of the carve-out into GTT are read over a much slower
path, so a plan that only "fits" because of GTT is a plan that is slower than
just using the CPU. The guard therefore reports spill explicitly instead of
quietly counting GTT as free space.

The GPU fraction defaults to 0.80, not 1.00. Measured: pushing close to 100%
of the carve-out produced

    amdgpu: The CS has been rejected (-12)
    amdgpu_vm_validate() failed
    Not enough memory for command submission!

and killed gnome-shell. The command-submission ring and the compositor need
room, so 20% is held back.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .hwtune import ModelProfile, profile_model

GIB = 1024 ** 3

#: Default share of the BIOS carve-out we are willing to fill with weights.
DEFAULT_GPU_FRAC = 0.80

#: System RAM held back for the desktop, browser and kernel page cache.
#: Measured idle footprint of the GNOME session is ~3.4 GiB, but Chromium with
#: a few tabs plus an IDE spike well past that, and once the guard has passed
#: the model still needs page cache to keep the GGUF hot.
DEFAULT_RESERVE_GIB = 10.0

#: Bytes per KV-cache element, by llama.cpp -ctk/-ctv type.
KV_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q5_1": 24 / 32,
    "q5_0": 22 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,
    "iq4_nl": 18 / 32,
}

VERDICT_OK = "OK"
VERDICT_TIGHT = "APERTADO"
VERDICT_REFUSE = "RECUSAR"


@dataclass
class Pools:
    """Live memory picture, in GiB, in the two pools llama.cpp can use."""

    gpu_total_gib: float
    gpu_used_gib: float
    ram_total_gib: float
    ram_available_gib: float
    swap_free_gib: float
    swap_total_gib: float = 0.0

    @property
    def gpu_free_gib(self) -> float:
        return max(0.0, self.gpu_total_gib - self.gpu_used_gib)


@dataclass
class Budget:
    gpu_usable_gib: float
    cpu_usable_gib: float
    reserve_gib: float
    gpu_frac: float
    notes: List[str] = field(default_factory=list)

    @property
    def ceiling_gib(self) -> float:
        return self.gpu_usable_gib + self.cpu_usable_gib


def _meminfo() -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.split()
        if not parts:
            continue
        try:
            out[key.strip()] = int(parts[0]) / 1024 ** 2  # kB -> GiB
        except ValueError:
            continue
    return out


def _vram() -> Tuple[float, float]:
    """(total, used) of the BIOS UMA carve-out, in GiB."""
    total = used = 0.0
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        dev = card / "device"
        try:
            total = int((dev / "mem_info_vram_total").read_text().strip()) / GIB
        except Exception:
            continue
        try:
            used = int((dev / "mem_info_vram_used").read_text().strip()) / GIB
        except Exception:
            used = 0.0
        break
    return total, used


def read_pools() -> Pools:
    mi = _meminfo()
    total, used = _vram()
    return Pools(
        gpu_total_gib=round(total, 2),
        gpu_used_gib=round(used, 2),
        ram_total_gib=round(mi.get("MemTotal", 0.0), 2),
        ram_available_gib=round(mi.get("MemAvailable", 0.0), 2),
        swap_free_gib=round(mi.get("SwapFree", 0.0), 2),
        swap_total_gib=round(mi.get("SwapTotal", 0.0), 2),
    )


def compute_budget(
    pools: Pools,
    gpu_frac: float = DEFAULT_GPU_FRAC,
    reserve_gib: float = DEFAULT_RESERVE_GIB,
) -> Budget:
    notes: List[str] = []

    gpu_usable = pools.gpu_free_gib * gpu_frac
    if pools.gpu_total_gib < 1.0:
        notes.append(
            f"Sem carve-out UMA ({pools.gpu_total_gib:.2f} GiB): o iGPU so tem GTT, "
            "que foi MEDIDO 2.5x mais lento que a CPU nesta maquina. Nao compensa "
            "offload; rode -ngl 0."
        )
        gpu_usable = 0.0

    # Swap is deliberately EXCLUDED from the model ceiling. Model weights are
    # touched once per token, so any byte of them that lands in swap is re-read
    # from disk on every single decode step: the machine does not run slowly, it
    # stops. Swap exists to absorb idle desktop pages, not weights. Counting it
    # as headroom is what makes a "fits on paper" plan die in practice.
    cpu_usable = max(0.0, pools.ram_available_gib - reserve_gib)
    if cpu_usable <= 0:
        notes.append("RAM disponivel ja abaixo da reserva: nao lance nada agora.")
    if pools.swap_free_gib < pools.swap_total_gib * 0.25:
        notes.append(
            f"Swap com apenas {pools.swap_free_gib:.1f} GiB livres de "
            f"{pools.swap_total_gib:.1f}: o desktop ja esta pressionado."
        )

    return Budget(
        gpu_usable_gib=round(gpu_usable, 2),
        cpu_usable_gib=round(cpu_usable, 2),
        reserve_gib=reserve_gib,
        gpu_frac=gpu_frac,
        notes=notes,
    )


def model_size_gib(path: str) -> Tuple[float, int]:
    """
    Total GiB of a model, following multi-part GGUF splits.

    A sharded model is one model, so checking only the first shard would let a
    60 GiB 3-way split sail through a 20 GiB check.
    """
    p = os.path.realpath(path)
    if os.path.isdir(p):
        parts = sorted(glob(os.path.join(p, "*.gguf")))
    else:
        parts = [p]
        stem = os.path.basename(p)
        # `...-00001-of-00003.gguf` -> collect the whole set.
        if "-of-" in stem:
            prefix = stem.split("-of-")[0].rsplit("-", 1)[0]
            parts = sorted(glob(os.path.join(os.path.dirname(p), prefix + "-*.gguf"))) or [p]

    total = 0
    for f in parts:
        try:
            total += os.path.getsize(os.path.realpath(f))
        except OSError:
            continue
    return total / GIB, len(parts)


def kv_cache_gib(profile: ModelProfile, ctx: int, kv_type: str = "f16") -> float:
    """KV cache size for `ctx` tokens. Returns 0.0 when the metadata is missing."""
    if not profile.n_layer or not profile.n_head or not profile.n_embd:
        return 0.0
    head_dim = profile.n_embd / profile.n_head
    n_kv = profile.n_head_kv or profile.n_head
    per_token = profile.n_layer * n_kv * head_dim * 2  # K and V
    return per_token * ctx * KV_BYTES.get(kv_type.lower(), 2.0) / GIB


@dataclass
class Plan:
    model: str
    size_gib: float
    shards: int
    arch: str
    n_layer: int
    is_moe: bool
    active_weight_gib: float
    kv_gib: float
    ctx: int
    kv_type: str
    required_gib: float
    gpu_usable_gib: float
    cpu_usable_gib: float
    ceiling_gib: float
    gpu_placed_gib: float
    cpu_placed_gib: float
    spill_gib: float
    n_gpu_layers: int
    verdict: str
    reasons: List[str]
    notes: List[str]

    def as_text(self) -> str:
        W = 74
        line = "-" * W
        ok = self.verdict == VERDICT_OK
        tight = self.verdict == VERDICT_TIGHT
        mark = "OK  " if ok else ("ATENCAO" if tight else "RECUSADO")
        out = [
            line,
            f" memguard: {mark}",
            line,
            f" modelo            {os.path.basename(self.model)}",
            f"   arquivo         {self.size_gib:.2f} GiB"
            + (f" em {self.shards} partes" if self.shards > 1 else ""),
            f"   arch            {self.arch or '?'}   camadas {self.n_layer or '?'}"
            + ("   MoE" if self.is_moe else "   denso"),
            f"   pesos ativos/tk {self.active_weight_gib:.2f} GiB",
            f"   KV cache        {self.kv_gib:.2f} GiB  (ctx {self.ctx}, {self.kv_type})",
            f"   PRECISA         {self.required_gib:.2f} GiB",
            "",
            f" orcamento",
            f"   GPU (carve-out) {self.gpu_usable_gib:.2f} GiB  (80% do UMA, resto p/ compositor+CS)",
            f"   CPU (RAM)       {self.cpu_usable_gib:.2f} GiB  (RAM disponivel menos reserva; swap nao conta)",
            f"   TETO            {self.ceiling_gib:.2f} GiB",
            "",
            f" plano",
            f"   na GPU          {self.gpu_placed_gib:.2f} GiB",
            f"   na CPU          {self.cpu_placed_gib:.2f} GiB",
            f"   transbordo GTT  {self.spill_gib:.2f} GiB"
            + ("   <-- LENTO (medido 2.5x pior que CPU)" if self.spill_gib > 0.5 else ""),
            f"   -ngl            {self.n_gpu_layers}"
            + ("" if self.n_gpu_layers < self.n_layer else "  (todas)"),
        ]
        if self.reasons:
            out += ["", " motivos"]
            out += [f"   - {r}" for r in self.reasons]
        if self.notes:
            out += ["", " avisos"]
            out += [f"   - {n}" for n in self.notes]
        out.append(line)
        return "\n".join(out)


def plan_model(
    path: str,
    ctx: int = 8192,
    kv_type: str = "f16",
    gpu_frac: float = DEFAULT_GPU_FRAC,
    reserve_gib: float = DEFAULT_RESERVE_GIB,
    pools: Optional[Pools] = None,
    mmap: bool = True,
) -> Plan:
    pools = pools or read_pools()
    budget = compute_budget(pools, gpu_frac=gpu_frac, reserve_gib=reserve_gib)
    size_gib, shards = model_size_gib(path)
    profile = profile_model(path)

    kv = kv_cache_gib(profile, ctx, kv_type)
    reasons: List[str] = []
    notes: List[str] = list(budget.notes)

    # With mmap the weights are file-backed and reclaimable, so the CPU side is
    # charged for what actually has to be resident. Without mmap every byte is
    # anonymous and unreclaimable, which is strictly worse on a tight machine.
    if not mmap:
        notes.append("--no-mmap: todos os pesos ficam residentes e nao reclamaveis.")

    required = size_gib + kv + 1.0  # +1 GiB for compute buffers / scratch

    gpu_placed = min(required, budget.gpu_usable_gib)
    cpu_placed = max(0.0, required - gpu_placed)
    spill = 0.0
    if cpu_placed > budget.cpu_usable_gib:
        spill = cpu_placed - budget.cpu_usable_gib

    ngl = profile.n_layer
    if profile.n_layer and size_gib > 0:
        per_layer = size_gib / profile.n_layer
        if per_layer > 0:
            ngl = int(min(profile.n_layer, budget.gpu_usable_gib / per_layer))

    # Verdict rules, calibrated against the three configurations we have MEASURED
    # on this exact machine -- not against theory:
    #
    #   Qwen3-Coder-30B-A3B Q4_K_S  18.1 GiB, fits entirely in the 38 GiB GPU
    #                               budget -> 37.0 t/s, rock solid.  => OK
    #   DeepSeek-V4-Flash reap-200b 57.5 GiB, needs the GPU budget filled to
    #                               100% AND 19.4 GiB of system RAM -> global
    #                               OOM at 14:44, session killed.     => must REFUSE
    #   Qwen3.8-Flash-Next-131B     62.0 GiB -> same shape, worse.     => must REFUSE
    #
    # The rule that separates them is NOT "does it fit in the ceiling". A plan
    # that fits only by filling the carve-out to the brim AND taking a large
    # slice of system RAM has already failed here twice, because partial offload
    # is the one configuration whose failure mode is not an exception: it is the
    # amdgpu command-submission rejection, the amdgpu_vm_pt_free kernel BUG, or
    # a global OOM sweep that takes org.gnome.Shell with it.
    cpu_needed = max(0.0, required - budget.gpu_usable_gib)
    fits_on_gpu = required <= budget.gpu_usable_gib

    if required > budget.ceiling_gib:
        verdict = VERDICT_REFUSE
        reasons.append(
            f"Precisa de {required:.2f} GiB e o teto e {budget.ceiling_gib:.2f} GiB: "
            f"faltam {required - budget.ceiling_gib:.2f} GiB."
        )
        reasons.append(
            "Nao existe flag que resolva. O desfecho medido nesta maquina nao e um "
            "erro limpo do llama.cpp: e OOM global do kernel (leva a sessao GNOME) "
            "ou BUG em amdgpu_vm_pt_free (reboot)."
        )
    elif not fits_on_gpu and budget.gpu_usable_gib > 1.0:
        share = cpu_needed / budget.cpu_usable_gib if budget.cpu_usable_gib > 0 else 1.0
        if share > 0.5:
            verdict = VERDICT_REFUSE
            reasons.append(
                f"Nao cabe no iGPU: sobram {cpu_needed:.2f} GiB para a CPU, que e "
                f"{share:.0%} do orcamento de RAM do sistema."
            )
            reasons.append(
                "Offload parcial e exatamente a configuracao que ja derrubou esta "
                "maquina. Quando o carve-out fica cheio, o amdgpu rejeita o command "
                "submission (-12) e o compositor morre junto."
            )
        else:
            verdict = VERDICT_TIGHT
            reasons.append(
                f"Nao cabe no iGPU: {cpu_needed:.2f} GiB teriam de rodar na CPU "
                f"({share:.0%} do orcamento de RAM)."
            )
            reasons.append(
                "Offload parcial: o desempenho cai para o do componente mais lento e "
                "a margem some assim que o navegador abrir."
            )
    elif required > budget.gpu_usable_gib * 0.85:
        verdict = VERDICT_TIGHT
        reasons.append(
            f"Cabe no iGPU, mas com so {budget.gpu_usable_gib - required:.2f} GiB "
            "de folga no carve-out."
        )
    else:
        verdict = VERDICT_OK

    if gpu_placed >= budget.gpu_usable_gib and cpu_placed > 0.5 and gpu_usable_positive(budget):
        notes.append(
            "Carve-out cheio: tudo o que sobrar vai para GTT/GTT-shared, que e o "
            "caminho lento. Reduza o modelo ou aumente o UMA no BIOS."
        )
    if budget.gpu_usable_gib > 0 and required > budget.gpu_usable_gib:
        notes.append(
            f"Nao cabe inteiro no carve-out ({budget.gpu_usable_gib:.2f} GiB). "
            "Parte roda na CPU; o desempenho cai para o do componente mais lento."
        )

    return Plan(
        model=os.path.realpath(path),
        size_gib=round(size_gib, 2),
        shards=shards,
        arch=profile.arch,
        n_layer=profile.n_layer,
        is_moe=profile.is_moe,
        active_weight_gib=round(profile.active_weight_gib, 2),
        kv_gib=round(kv, 2),
        ctx=ctx,
        kv_type=kv_type,
        required_gib=round(required, 2),
        gpu_usable_gib=budget.gpu_usable_gib,
        cpu_usable_gib=budget.cpu_usable_gib,
        ceiling_gib=round(budget.ceiling_gib, 2),
        gpu_placed_gib=round(gpu_placed, 2),
        cpu_placed_gib=round(cpu_placed, 2),
        spill_gib=round(spill, 2),
        n_gpu_layers=ngl,
        verdict=verdict,
        reasons=reasons,
        notes=notes,
    )


def gpu_usable_positive(budget: Budget) -> bool:
    return budget.gpu_usable_gib > 1.0


def check(
    path: str,
    ctx: int = 8192,
    kv_type: str = "f16",
    gpu_frac: float = DEFAULT_GPU_FRAC,
    reserve_gib: float = DEFAULT_RESERVE_GIB,
) -> Plan:
    return plan_model(
        path, ctx=ctx, kv_type=kv_type, gpu_frac=gpu_frac, reserve_gib=reserve_gib
    )


def _main(argv: Optional[Sequence[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)

    # Split on a literal `--` BEFORE argparse sees it. argparse's REMAINDER is
    # unreliable once positionals and optionals are interleaved -- it rejects
    # the very `--` it is meant to stop at -- and this is a safety tool, so the
    # command boundary must be unambiguous.
    run_cmd: List[str] = []
    if "--" in raw:
        cut = raw.index("--")
        raw, run_cmd = raw[:cut], raw[cut + 1:]

    ap = argparse.ArgumentParser(
        prog="memguard",
        description="Confere se um modelo cabe nesta APU antes de carrega-lo. "
                    "Use `-- <comando>` para executa-lo so se passar.",
    )
    ap.add_argument("model", nargs="?", help="caminho do .gguf (ou pasta)")
    ap.add_argument("--ctx", type=int, default=8192, help="contexto que sera alocado")
    ap.add_argument("--kv-type", default="f16", help="tipo do KV cache (f16/q8_0/q4_0)")
    ap.add_argument("--gpu-frac", type=float, default=DEFAULT_GPU_FRAC,
                    help="fracao do carve-out UMA utilizavel (padrao 0.80)")
    ap.add_argument("--reserve", type=float, default=DEFAULT_RESERVE_GIB,
                    help="GiB de RAM reservados para o desktop (padrao 10)")
    ap.add_argument("--json", action="store_true", help="saida em JSON")
    ap.add_argument("--json-pools", action="store_true", help="so as pools, em JSON")
    args = ap.parse_args(raw)

    if args.json_pools:
        print(json.dumps(asdict(read_pools()), indent=2))
        return 0

    if not args.model:
        pools = read_pools()
        budget = compute_budget(pools, gpu_frac=args.gpu_frac, reserve_gib=args.reserve)
        print(f"RAM visivel      {pools.ram_total_gib:.2f} GiB")
        print(f"RAM disponivel   {pools.ram_available_gib:.2f} GiB")
        print(f"Swap livre       {pools.swap_free_gib:.2f} GiB")
        print(f"Carve-out UMA    {pools.gpu_total_gib:.2f} GiB (em uso {pools.gpu_used_gib:.2f})")
        print(f" -> teto modelo  {budget.ceiling_gib:.2f} GiB"
              f"  (GPU {budget.gpu_usable_gib:.2f} + CPU {budget.cpu_usable_gib:.2f})")
        for n in budget.notes:
            print(f" !  {n}")
        return 0

    if not os.path.exists(args.model):
        print(f"memguard: nao encontrei {args.model}", file=sys.stderr)
        return 2

    plan = check(
        args.model,
        ctx=args.ctx,
        kv_type=args.kv_type,
        gpu_frac=args.gpu_frac,
        reserve_gib=args.reserve,
    )

    if args.json:
        print(json.dumps(asdict(plan), indent=2))
    else:
        print(plan.as_text())

    if run_cmd:
        cmd = list(run_cmd)
        if plan.verdict == VERDICT_REFUSE:
            print("\nmemguard: comando NAO executado (veredito RECUSAR).", file=sys.stderr)
            if shutil.which("notify-send"):
                subprocess.run(
                    ["notify-send", "-u", "critical", "memguard",
                     "Modelo grande demais: carga bloqueada para nao derrubar a sessao."],
                    check=False,
                )
            return 3
        os.execvp(cmd[0], cmd)

    return 0 if plan.verdict != VERDICT_REFUSE else 3


if __name__ == "__main__":
    raise SystemExit(_main())
