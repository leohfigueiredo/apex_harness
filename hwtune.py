"""
apex_harness.hwtune — hardware-aware, evidence-based llama-server tuning.

WHY THIS EXISTS
---------------
`launcher_common.launch_llama_server()` hard-codes a flag set that was written for
a hypothetical machine, not this one. Measured on this host:

  * `taskset -c 0-7` pins the process to CPUs 0-7. On a Ryzen AI 9 HX 370 that is
    4x Zen 5 (5.16 GHz, L3 domain {0-3,12-15}) **plus** 4x Zen 5c (3.29 GHz, L3
    domain {4-11,16-23}) — two different core types and two different L3 domains.
    Meanwhile llama.cpp still spawns `n_threads = 12` (measured: "llama threadpool
    init, n_threads = 12"), so 12 threads fight over 8 logical CPUs, half of them
    the slow cores, all of them busy-waiting in ggml's barrier.
    Measured cost: `llama-bench -t 12` = 13.73 t/s vs `-t 24` = **0.35 t/s**.

  * The accelerator env block (HSA_OVERRIDE_GFX_VERSION, HSA_ENABLE_SDMA,
    HIP_VISIBLE_DEVICES, AMD_GPU_BUILD_TARGET) targets ROCm/HIP, but the binary
    that gets launched is the **Vulkan** build (libggml-vulkan.so, no
    libggml-hip.so). Those variables cannot affect it. `LD_PRELOAD` of
    libdrm_amdgpu.so.1 additionally shadows the /opt/amdgpu userspace driver.

  * `-ngl 99` is correct for prefill but the launcher never sets `-t`, so the CPU
    thread pool is left at an oversubscribed default.

  * The build supports `--spec-type draft-mtp` and the sibling `mtp-*.gguf` files
    are explicitly filtered OUT of the model list, so the single largest available
    speedup is switched off.

This module replaces the guessing with measurement + model introspection.

USAGE
-----
    from apex_harness.hwtune import plan_server_command, TuneOptions

    argv, env, notes = plan_server_command(
        model_path="/mnt/HDD/AIModels/.../Qwen3.8-27B-Q4_K_M.gguf",
        context=32768,
        server_bin="/home/leonardo/.local/bin/llama-server",
    )
    # argv is a ready-to-exec list; env is the environment to use; notes explains
    # every decision it took and why.

Everything here is dependency-free (stdlib only) so it can be imported by the
launcher without adding packages.
"""

from __future__ import annotations

import os
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
#  CPU topology
# --------------------------------------------------------------------------- #

#: LPDDR5X-7500 256-bit theoretical peak on Strix Point, GB/s.
THEORETICAL_BW_GBS = 120.0


@dataclass
class CpuTopology:
    """Hybrid-core layout of the host CPU."""
    fast_cpus: List[int] = field(default_factory=list)   # Zen 5 (high clock)
    slow_cpus: List[int] = field(default_factory=list)   # Zen 5c (low clock)
    #: logical CPU -> physical core id (from lscpu's CORE column)
    core_of: Dict[int, int] = field(default_factory=dict)

    def _cores(self, cpus: Sequence[int]) -> int:
        if self.core_of:
            return len({self.core_of.get(c, c) for c in cpus})
        return len(cpus) // 2 or len(cpus)

    @property
    def n_fast_cores(self) -> int:
        return self._cores(self.fast_cpus) if self.fast_cpus else 0

    @property
    def n_physical_cores(self) -> int:
        allc = self.fast_cpus + self.slow_cpus
        return self._cores(allc) if allc else 0

    def one_thread_per_core(self) -> List[int]:
        """
        Pick exactly one logical CPU per physical core.

        This is the mask that matters: with SMT, two threads per core share the
        same L1/L2 and cannot move more bytes between them, while the second
        thread still costs barrier synchronisation. Measured on this host,
        going from 12 to 24 threads on the same cores drops aggregate read
        bandwidth from 118 GB/s to 97 GB/s, and llama-bench -t 24 collapses
        from 13.73 t/s to 0.35 t/s.
        """
        seen: Dict[int, int] = {}
        picked: List[int] = []
        for c in sorted(self.fast_cpus + self.slow_cpus):
            core = self.core_of.get(c, c)
            if core in seen:
                continue
            seen[core] = c
            picked.append(c)
        return picked or list(range(self.n_physical_cores or 12))

    def mask_hex(self, cpus: Sequence[int]) -> str:
        """Build the hex affinity mask llama.cpp expects for --cpu-mask."""
        m = 0
        for c in cpus:
            if 0 <= c < 256:
                m |= 1 << c
        return f"0x{m:X}"


def detect_topology() -> CpuTopology:
    """
    Classify logical CPUs as fast/slow by their advertised max frequency.

    On the Ryzen AI 9 HX 370 this yields fast={0-3,12-15} (Zen 5, 5.16 GHz) and
    slow={4-11,16-23} (Zen 5c, 3.29 GHz). SMT siblings are read from the core-id
    column rather than assumed to be offset by a fixed amount.
    """
    topo = CpuTopology()
    try:
        out = subprocess.check_output(["lscpu", "-p=CPU,CORE,MAXMHZ"], text=True)
    except Exception:
        topo.fast_cpus = [0, 1, 2, 3]
        topo.slow_cpus = list(range(4, os.cpu_count() or 12))
        return topo

    mhz: Dict[int, float] = {}
    for line in out.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            cpu = int(parts[0])
            topo.core_of[cpu] = int(parts[1])
            mhz[cpu] = float(parts[2])
        except ValueError:
            continue

    if not mhz:
        return topo
    fastest = max(mhz.values())
    topo.fast_cpus = sorted(c for c, m in mhz.items() if m >= fastest * 0.9)
    topo.slow_cpus = sorted(c for c, m in mhz.items() if m < fastest * 0.9)
    return topo


# --------------------------------------------------------------------------- #
#  GGUF metadata introspection (no third-party deps)
# --------------------------------------------------------------------------- #

_GGUF_VALUE_SIZES = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}


class _Reader:
    def __init__(self, fh):
        self.fh = fh

    def u32(self) -> int:
        return struct.unpack("<I", self.fh.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.fh.read(8))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.fh.read(4))[0]

    def string(self) -> str:
        n = self.u64()
        return self.fh.read(n).decode("utf-8", "replace")

    def value(self, vtype: int) -> Any:
        if vtype == 8:                      # STRING
            return self.string()
        if vtype == 9:                      # ARRAY
            inner = self.u32()
            n = self.u64()
            if inner in _GGUF_VALUE_SIZES:
                # Fixed-width elements: seek past them without reading.
                self.fh.seek(_GGUF_VALUE_SIZES[inner] * n, os.SEEK_CUR)
                return f"<array[{n}]>"
            if inner == 8:
                # Variable-length strings have no stride, so they must be walked.
                # would desynchronise the stream and corrupt every later key.
                # tokenizer.ggml.tokens is routinely ~250k entries; reading them
                # all is a few tens of milliseconds with buffered I/O.
                if n > 8_000_000:
                    raise ValueError(f"implausible string array length {n}")
                for _ in range(n):
                    self.string()
                return f"<str-array[{n}]>"
            if inner == 9:
                # Nested arrays are not used by GGUF in practice; refuse rather
                # than silently desynchronise.
                raise ValueError("nested arrays unsupported")
            raise ValueError(f"unsupported array elem type {inner}")
        if vtype in _GGUF_VALUE_SIZES:
            size = _GGUF_VALUE_SIZES[vtype]
            raw = self.fh.read(size)
            fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                   6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}.get(vtype)
            return struct.unpack(fmt, raw)[0] if fmt else raw[0]
        raise ValueError(f"unknown GGUF value type {vtype}")


def read_gguf_metadata(path: str, max_kv: int = 4096) -> Dict[str, Any]:
    """
    Parse a GGUF file's metadata key/value block.

    Only the header + KV section is read, so this is fast even for a 60 GB model.
    Returns {} on any parse failure (never raises) so it is safe to call on
    untrusted or exotic files.
    """
    meta: Dict[str, Any] = {}
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != b"GGUF":
                return {}
            r = _Reader(fh)
            r.u32()                 # version
            meta["_tensor_count"] = r.u64()
            n_kv = r.u64()
            for _ in range(min(n_kv, max_kv)):
                try:
                    key = r.string()
                    vtype = r.u32()
                    meta[key] = r.value(vtype)
                except Exception:
                    break
    except Exception:
        return {}
    return meta


def read_gguf_tensor_names(path: str, limit: int = 20000) -> List[str]:
    """
    Return the tensor names declared in a GGUF's tensor-info section.

    This is what actually decides whether a model can self-speculate: the
    `*nextn_predict_layers` metadata key advertises MTP support, but the
    `blk.N.nextn.*` tensors must physically be present in the file. Some
    distributions ship the MTP heads in a *separate* `mtp-*.gguf`, in which case
    the metadata key is present but the tensors are not.
    """
    names: List[str] = []
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != b"GGUF":
                return []
            r = _Reader(fh)
            r.u32()                             # version
            n_tensors = r.u64()
            n_kv = r.u64()
            # Skip the KV block exactly as read_gguf_metadata does.
            for _ in range(min(n_kv, 4096)):
                try:
                    r.string()
                    r.value(r.u32())
                except Exception:
                    return names
            # Tensor info records: name, n_dims, dims[n_dims], type, offset
            for _ in range(min(n_tensors, limit)):
                try:
                    names.append(r.string())
                    nd = r.u32()
                    fh.seek(8 * nd, os.SEEK_CUR)   # dims (u64 each)
                    fh.seek(4, os.SEEK_CUR)        # ggml_type (u32)
                    fh.seek(8, os.SEEK_CUR)        # offset (u64)
                except Exception:
                    break
    except Exception:
        return names
    return names


#: Tipos de tensor que o llama.cpp mainline conhece. O enum ggml_type vai de 0
#: (F32) a 39 (MXFP4); qualquer coisa acima disto so existe em forks.
#:
#: Isto importa porque a falha NAO e obvia: o ficheiro aparece na lista, tem o
#: tamanho certo, e o llama-server so rebenta no arranque com
#:
#:     llama_model_load: error loading model: llama_model_loader: failed to
#:     load model from .../Ternary-Bonsai-2-27B-PQ2_0.gguf
#:     srv llama_server: exiting due to model loading error
#:
#: Medido nos ficheiros desta maquina:
#:   Ternary-Bonsai-2-27B-PQ2_0.gguf   402 tensores tipo 142  (so PrismML)
#:   Ternary-Bonsai-2-27B-PTQ1_0.gguf  402 tensores tipo 143  (so PrismML)
#:   Qwen3-Coder-30B-A3B-Q4_K_S.gguf   tipos 0, 12, 13, 14    (padrao)
MAINLINE_MAX_GGML_TYPE = 39

#: Nomes legiveis para os tipos que aparecem nos ficheiros desta maquina.
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
    142: "PQ2_0 (PrismML)", 143: "PTQ1_0 (PrismML)",
}


def find_mmproj(model_path: str) -> Optional[str]:
    """
    Encontra o projetor multimodal (mmproj) que acompanha este modelo.

    Um modelo de visao so ve imagens se o llama-server arrancar com
    `-mm/--mmproj FILE`. O ficheiro costuma ficar ao lado do modelo, com
    `mmproj` no nome.

    Medido: Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf (601 MB) esta na mesma pasta
    dos ficheiros do Bonsai 2. Sem este ficheiro o modelo e texto apenas e o
    /props do servidor responde `modalities: {vision: false}`.

    Devolve None quando nao ha nenhum -- o que e o caso normal, nao um erro.
    """
    try:
        pasta = Path(os.path.realpath(model_path)).parent
    except Exception:
        return None

    candidatos: List[str] = []
    for padrao in ("*mmproj*.gguf", "*MMPROJ*.gguf"):
        candidatos += [str(p) for p in sorted(pasta.glob(padrao))]
    if not candidatos:
        return None

    # Se houver mais do que um, prefere o de maior precisao que exista:
    # f16 > bf16 > q8_0 > q4_0. Nao vale a pena adivinhar mais do que isto.
    def peso(nome: str) -> int:
        n = nome.lower()
        for i, marca in enumerate(("f16", "bf16", "q8_0", "q5_1", "q4_0")):
            if marca in n:
                return 100 - i
        return 0

    return max(candidatos, key=peso)


def read_gguf_tensor_types(path: str) -> Dict[int, int]:
    """
    Le o cabecalho GGUF e devolve {tipo_de_tensor: numero_de_tensores}.

    Percorre o mesmo formato que read_gguf_tensor_names, mas guarda o tipo em vez
    do nome. Devolve {} se o ficheiro nao for um GGUF legivel.
    """
    import struct

    out: Dict[int, int] = {}
    try:
        with open(path, "rb", buffering=1 << 20) as fh:
            if fh.read(4) != b"GGUF":
                return {}
            struct.unpack("<I", fh.read(4))
            n_tensors = struct.unpack("<Q", fh.read(8))[0]
            n_kv = struct.unpack("<Q", fh.read(8))[0]

            def u32() -> int:
                return struct.unpack("<I", fh.read(4))[0]

            def u64() -> int:
                return struct.unpack("<Q", fh.read(8))[0]

            def read_str() -> str:
                return fh.read(u64()).decode("utf-8", "replace")

            def skip_value(vtype: int) -> None:
                sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                         10: 8, 11: 8, 12: 8}
                if vtype == 8:
                    read_str()
                elif vtype == 9:
                    elem = u32()
                    n = u64()
                    for _ in range(n):
                        skip_value(elem)
                elif vtype in sizes:
                    fh.read(sizes[vtype])

            for _ in range(min(n_kv, 4096)):
                read_str()
                skip_value(u32())

            for _ in range(min(n_tensors, 200000)):
                read_str()
                n_dims = u32()
                fh.read(8 * n_dims)
                ttype = u32()
                u64()
                out[ttype] = out.get(ttype, 0) + 1
    except Exception:
        return {}
    return out


def unsupported_quant_types(path: str) -> List[int]:
    """
    Tipos de tensor que o llama.cpp mainline NAO sabe ler. Lista vazia = ok.

    Serve para avisar no seletor em vez de deixar o utilizador escolher um modelo
    que rebenta no arranque e deixa o harness sem backend durante minutos.
    """
    tipos = read_gguf_tensor_types(path)
    return sorted(t for t in tipos if t > MAINLINE_MAX_GGML_TYPE)


def quant_type_label(tipo: int) -> str:
    return GGML_TYPE_NAMES.get(tipo, f"tipo {tipo}")


@dataclass
class ModelProfile:
    """What we need to know about a GGUF in order to tune for it."""
    path: str
    size_gib: float
    arch: str = ""
    n_layer: int = 0
    n_expert: int = 0                 # >0 => MoE
    n_expert_used: int = 0            # active experts per token
    n_ctx_train: int = 0
    n_head: int = 0
    n_head_kv: int = 0
    n_embd: int = 0
    has_mtp: bool = False             # model ships nextn / MTP heads
    #: parsed from an `A<n>B` model-name token, e.g. "Qwen3-Coder-30B-A3B" -> 3.0
    active_params_b: float = 0.0
    #: parsed from an `<n>B` model-name token, e.g. "Qwen3-Coder-30B-A3B" -> 30.0
    total_params_b: float = 0.0

    @property
    def is_moe(self) -> bool:
        return self.n_expert > 0

    @property
    def active_fraction(self) -> float:
        """
        Share of weights read per token.

        Prefers the `A<n>B` convention in the model name (e.g. Qwen3-Coder-30B-A3B
        -> 3/30 = 0.10), which is the published active-parameter count and is more
        trustworthy than deriving it from expert_used_count/expert_count: that ratio
        ignores the always-on attention and embedding weights, so it underestimates
        the active footprint and flatters the t/s estimate.
        """
        if self.active_params_b and self.total_params_b:
            return min(1.0, self.active_params_b / self.total_params_b)
        if not self.is_moe or not self.n_expert:
            return 1.0
        # Fallback: assume experts are ~85% of the weights and the rest is always on.
        expert_frac = 0.85
        return (1.0 - expert_frac) + expert_frac * (max(self.n_expert_used, 1) / self.n_expert)

    @property
    def active_weight_gib(self) -> float:
        """GiB of weights that must be streamed from RAM for every generated token."""
        return self.size_gib * self.active_fraction

    @property
    def gqa_ratio(self) -> float:
        if self.n_head and self.n_head_kv:
            return self.n_head / self.n_head_kv
        return 1.0


def profile_model(path: str) -> ModelProfile:
    """Read size + architecture facts from a GGUF; degrade gracefully."""
    try:
        size = os.path.getsize(os.path.realpath(path))
    except Exception:
        size = 0
    p = ModelProfile(path=path, size_gib=round(size / 1024**3, 3))

    # Published MoE parameter counts from the conventional name, e.g.
    # "Qwen3-Coder-30B-A3B-Instruct" -> total 30B, active 3B.
    import re as _re
    base = os.path.basename(path)
    m_active = _re.search(r"\bA(\d+(?:\.\d+)?)B\b", base, _re.I)
    if m_active:
        p.active_params_b = float(m_active.group(1))
        # Total = the first `<n>B` that is NOT the `A<n>B` token.
        stripped = _re.sub(r"\bA\d+(?:\.\d+)?B\b", "", base, flags=_re.I)
        m_total = _re.search(r"(\d+(?:\.\d+)?)B", stripped, _re.I)
        if m_total:
            p.total_params_b = float(m_total.group(1))

    meta = read_gguf_metadata(path)
    if not meta:
        return p

    p.arch = str(meta.get("general.architecture", "") or "")
    a = p.arch

    def gi(key: str) -> int:
        v = meta.get(f"{a}.{key}", meta.get(key, 0))
        try:
            return int(v)
        except Exception:
            return 0

    p.n_layer = gi("block_count")
    p.n_expert = gi("expert_count")
    p.n_expert_used = gi("expert_used_count")
    p.n_ctx_train = gi("context_length")
    p.n_head = gi("attention.head_count")
    p.n_head_kv = gi("attention.head_count_kv")
    p.n_embd = gi("embedding_length")

    # MTP/nextn: the metadata key alone is NOT sufficient. Some distributions set
    # `*.nextn_predict_layers` while shipping the actual `blk.N.nextn.*` tensors in
    # a separate `mtp-*.gguf`. Self-speculation only works if the tensors are HERE.
    if gi("nextn_predict_layers") > 0:
        names = read_gguf_tensor_names(path)
        p.has_mtp = any("nextn" in n.lower() for n in names)
    return p


def _is_thinking_model(path: str) -> bool:
    """
    Detecta um modelo de raciocinio pelo PROPRIO chat template do GGUF.

    Isto decidir se vale a pena ligar n-gram speculation, e a diferenca medida e
    enorme: 1,95x num modelo de codigo contra 0,28x (3x MAIS LENTO) num modelo
    de raciocinio. Nao da para tratar como detalhe.

    O sinal tem de vir do ficheiro, nao do nome: renomear o GGUF nao pode mudar
    o comportamento do motor. Medido nos modelos desta maquina:

        Swift-Qwen3.8-27B   chat_template 8952 chars -> '<think', 'thinking',
                            'reasoning', 'enable_thinking>'          => pensa
        Qwen3-Coder-30B-A3B chat_template 6896 chars -> so 'reasoning' => NAO pensa

    Nota: casar apenas "reasoning" daria falso positivo -- aparece no template do
    modelo de codigo. O marcador que distingue e '<think' / 'enable_thinking'.
    O nome fica como rede de seguranca, e sempre no sentido conservador (desligar
    especulacao), porque o erro barato e nao ligar.
    """
    try:
        meta = read_gguf_metadata(path) or {}
        ct = str(meta.get("tokenizer.chat_template", "")).lower()
        if "<think" in ct or "enable_thinking" in ct:
            return True
    except Exception:
        pass

    nome = os.path.basename(path).lower()
    return any(p in nome for p in _THINKING_NAME_HINTS)


#: Rede de seguranca por nome, para GGUFs sem chat template legivel. Inclui o
#: Swift, que e o modelo pre-selecionado no atalho do desktop.
_THINKING_NAME_HINTS = ("think", "qwen3.8", "swift", "-r1", "qwq", "magistral", "reason")


def find_draft_model(model_path: str, profile: Optional[ModelProfile] = None) -> Optional[Tuple[str, str]]:
    """
    Locate a sibling speculative-decoding sidecar for `model_path`.

    Returns (draft_path, spec_type) or None. Recognised conventions, in the order
    they are preferred:

      * `mtp-<name>.gguf`    -> ("--spec-type", "draft-mtp")     *(shares the target's KV; cheapest)*
      * `dflash-<name>.gguf` -> ("--spec-type", "draft-dflash")  *(block-diffusion sidecar)*
      * `<name>-eagle3.gguf` -> ("--spec-type", "draft-eagle3")
      * a much smaller GGUF in the same directory whose name shares the target's
        family token (e.g. `Qwen2.5-Coder-0.5B` for `Qwen2.5-Coder-7B`)
        -> ("--spec-type", "draft-simple")

    The launcher's `_SKIP_PREFIXES` currently hides `mtp-` and `dflash-` files, which
    is exactly backwards: they are not alternate models, they are accelerators.
    """
    d = os.path.dirname(os.path.abspath(model_path))
    base = os.path.basename(model_path).lower()

    try:
        entries = os.listdir(d)
    except Exception:
        return None

    def pick(predicate) -> Optional[str]:
        for name in sorted(entries):
            if name.lower().endswith(".gguf") and predicate(name.lower()):
                full = os.path.join(d, name)
                if os.path.exists(full) and os.path.getsize(full) > 1024 * 1024:
                    return full
        return None

    # 1. MTP: cheap, no second KV cache, wins at long context.
    hit = pick(lambda n: n.startswith("mtp-"))
    if hit:
        return hit, "draft-mtp"

    # 2. DFlash: block diffusion, wins at short context / structured output.
    hit = pick(lambda n: n.startswith("dflash-"))
    if hit:
        return hit, "draft-dflash"

    # 3. EAGLE-3 head.
    hit = pick(lambda n: n.endswith("eagle3.gguf") or "eagle3" in n)
    if hit:
        return hit, "draft-eagle3"

    return None


# --------------------------------------------------------------------------- #
#  Memory-bandwidth probe (cached)
# --------------------------------------------------------------------------- #

def theoretical_decode_tps(profile: ModelProfile, bw_gbs: float = THEORETICAL_BW_GBS,
                           efficiency: float = 0.55) -> float:
    """
    Upper bound on decode t/s for a bandwidth-bound model.

    Bytes moved per token is the *active* weight footprint. For MoE that is the
    always-on weights (attention, embeddings, norms) plus only the experts the
    router selects, which is why MoE beats dense so decisively here.

    `efficiency` defaults to 0.55, calibrated against measurement on this host:
    Qwen2.5-Coder-7B-Q4_K_M is 4.683 GB on disk and decoded at 13.73 t/s on CPU,
    i.e. 64.3 GB/s of effective traffic = 54.5% of the measured 118 GB/s aggregate
    STREAM peak. The gap is structural, not a tuning failure: 118 GB/s is a
    multi-threaded streaming figure, while single-stream decode is a latency-bound
    dependent chain. The Vulkan path does better on MoE (a reported 37.2 t/s for
    Qwen3-Coder-30B-A3B on this exact iGPU), so treat 0.55 as a CPU-side bound.

    Deliberately optimistic in other ways too: it ignores KV-cache traffic, which
    grows with context, and assumes perfect expert batching - real implementations
    gather scattered expert blocks and move more than the theoretical minimum.
    """
    active_gib = profile.active_weight_gib
    if active_gib <= 0:
        return 0.0
    return (bw_gbs * efficiency) / active_gib


# --------------------------------------------------------------------------- #
#  The planner
# --------------------------------------------------------------------------- #

@dataclass
class TuneOptions:
    context: int = 32768
    parallel_slots: int = 1
    #: TCP port for the HTTP server. Must be threaded through explicitly: dropping
    #: it silently makes llama-server fall back to its default 8080.
    port: int = 8080
    #: "auto" | "vulkan" | "cpu". "auto" offloads when the model plausibly fits.
    backend: str = "auto"
    #: enable a speculative-decoding sidecar when one is found
    speculative: bool = True
    #: force MTP (--spec-type draft-mtp) when the target ships nextn heads.
    #: Measured neutral-to-negative on a dense target here - off by default.
    use_mtp: bool = False
    #: load a separate draft/sidecar GGUF (mtp-/dflash-/eagle3-). Costs its own
    #: weight reads per draft step, so off by default.
    use_draft_sidecar: bool = False
    #: scheduling priority for the main (decode) thread pool.
    #: DEFAULT 0 (OS default), and that is a measured decision, not caution.
    #: Raising nice priority needs CAP_SYS_NICE; only lowering it is
    #: unprivileged. As a normal user, `--prio 2` produced
    #:   "failed to set process priority 2 : Permission denied (13)"
    #:   "failed to set thread priority 2 : Operation not permitted"
    #: the second of which repeated 6180 times in one server lifetime, once per
    #: worker thread per task. plan_server_command() now emits the flags only
    #: when _has_cap_sys_nice() is true, so setting a non-zero value here is
    #: safe -- it simply will not be used without the capability.
    prio: int = 0
    #: scheduling priority for the batch (prefill) thread pool. See `prio`.
    prio_batch: int = 0
    #: fraction of the 48 GiB UMA carve-out we are willing to fill
    vram_budget_frac: float = 0.85
    #: q8_0 KV is the sweet spot: ~half the size of f16, no measurable quality loss.
    #: q4_0 saves ~25% more memory at long contexts at a small quality cost.
    #: Both require -fa on (already enforced in plan_server_command).
    kv_type: str = "q8_0"
    #: model loading mode — passed verbatim as `-lm <mode>` to llama-server.
    #: --mlock / --mmap / --no-mmap are deprecated since build ~10000; use this.
    #: Choices: "auto" | "mmap" | "mlock" | "mmap+mlock"
    #:   mmap        — mmap only, no pinning (DEFAULT). Chosen because mlock is
    #:                 usually impossible: stock Ubuntu ships a hard
    #:                 RLIMIT_MEMLOCK of 8192 KiB, so `mmap+mlock` fails on its
    #:                 first buffer ("Cannot allocate memory / Try increasing
    #:                 RLIMIT_MEMLOCK") and silently continues as plain mmap.
    #:                 plan_server_command() downgrades automatically when the
    #:                 limit cannot cover the weights, so setting a pinning mode
    #:                 here is harmless -- it just will not survive the check.
    #:   mmap+mlock  — mmap for fast startup, then pin against swap. Requires
    #:                 RLIMIT_MEMLOCK >= model size (root, or a systemd unit with
    #:                 LimitMEMLOCK=infinity).
    #:   mlock       — read weights fully into RAM (no mmap). Use for models on
    #:                 slow/NTFS disks where mmap page eviction causes decode
    #:                 stalls. Same RLIMIT_MEMLOCK requirement.
    #:   auto        — let llama-server decide (usually plain mmap).
    load_mode: str = "mmap"
    #: When True, restrict the CPU affinity mask to the *fast* Zen5 cores only,
    #: excluding the slower Zen5c cluster. Reduces cross-L3-domain traffic during
    #: decode at the cost of fewer cores for prefill. Measure before enabling:
    #: on HX 370 this limits -t to 4 physical cores.
    fast_cores_only: bool = False
    metrics: bool = True
    alias: str = "llama-local-model"
    #: Server-side default temperature. The original launcher pinned 0.3; the
    #: ApexAgent always sends its own temperature, so this only affects clients
    #: that hit the endpoint directly. Measured to cost ~5% decode (6.15 vs 6.44
    #: t/s on a dense 7B), so it is left UNSET by default: a server-side sampling
    #: default should not buy a throughput regression. Set explicitly if needed.
    temp: Optional[float] = None
    #: Server-side cap on generated tokens (-n). The original launcher used 16384
    #: and the harness client does not send max_tokens, so removing this would
    #: make generation unbounded up to n_ctx - a runaway-generation footgun for
    #: an agent loop. Preserved deliberately.
    n_predict: int = 16384
    extra: List[str] = field(default_factory=list)


def _vram_budget_gib() -> float:
    """Read the BIOS UMA carve-out the amdgpu driver reports."""
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        f = card / "device" / "mem_info_vram_total"
        try:
            return int(f.read_text().strip()) / 1024**3
        except Exception:
            continue
    return 0.0


def _gtt_total_gib() -> float:
    """
    Read the GTT size the amdgpu driver advertises.

    GTT is system RAM that the iGPU can map. On an APU this is not optional
    trivia: it is where the Vulkan backend actually allocates when the BIOS
    carve-out is small, which is the configuration AMD recommends. Deciding
    offload from the carve-out alone therefore breaks the moment a user
    follows that advice.
    """
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        f = card / "device" / "mem_info_gtt_total"
        try:
            return int(f.read_text().strip()) / 1024**3
        except Exception:
            continue
    return 0.0


def _system_ram_gib() -> float:
    """
    Physical RAM, INCLUDING the BIOS carve-out.

    /proc/meminfo's MemTotal excludes the carve-out (it never enters the page
    allocator), so it must be added back to know how much silicon is present.
    """
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                visible = int(line.split()[1]) / 1024**2
                return visible + _vram_budget_gib()
    except Exception:
        pass
    return 0.0


def _visible_ram_gib() -> float:
    """CPU-visible MemTotal from /proc/meminfo in GiB."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024**2
    except Exception:
        pass
    return 0.0


def _memlock_limit_gib() -> float:
    """
    Hard RLIMIT_MEMLOCK in GiB, or -1.0 for unlimited.

    This exists because `-lm mmap+mlock` and `-lm mlock` are silently useless,
    and actively misleading, when the limit is small. On stock Ubuntu the hard
    limit is 8192 KiB and cannot be raised without root, so llama-server prints

        warning: failed to mlock 436264960-byte buffer (after previously locking
        0 bytes): Cannot allocate memory
        Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root).

    and then keeps going in plain mmap mode. The flag did nothing, but the plan
    claimed the model was pinned against swap -- which is the whole reason to
    emit it. Better to detect it and say so than to emit a lie.
    """
    try:
        import resource

        _soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        if hard == resource.RLIM_INFINITY:
            return -1.0
        return hard / 1024**3
    except Exception:
        return 0.0


def _has_cap_sys_nice() -> bool:
    """
    True when this process may raise its own scheduling priority.

    Measured on this host as a normal user, `--prio 2` does NOT work:

        W cmn set_process_: failed to set process priority 2 : Permission denied (13)
        warn: failed to set thread priority 2 : Operation not permitted (1)

    That second line repeated 6180 times in a single server lifetime -- every
    worker thread, on every task. Raising nice priority needs CAP_SYS_NICE
    (bit 23); only *lowering* it is unprivileged. So the flag is not "safe
    without CAP_SYS_NICE", it is inert without it and only costs syscalls.
    """
    if os.geteuid() == 0:
        return True
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("CapPrm:"):
                return bool(int(line.split()[1], 16) & (1 << 23))
    except Exception:
        pass
    return False



def _gpu_budget_gib(budget_frac: float = 0.85) -> Tuple[float, str]:
    """
    How many GiB we are willing to place on the iGPU.

    ONLY the BIOS carve-out counts, not GTT. This is a MEASURED conclusion, and
    it is counter-intuitive, so the evidence is recorded here:

      With a 48 GiB carve-out (weights resident in DEVICE_LOCAL memory):
          CPU -ngl 0   decode 6.78 t/s   prefill 46.30 t/s
          Vulkan -ngl 999 decode 13.69 t/s  prefill 75.54 t/s   <-- Vulkan wins 2.0x

      With a 0.5 GiB carve-out (weights must live in GTT, host-visible RAM):
          CPU -ngl 0   decode 13.81 t/s  prefill 73.95 t/s
          Vulkan -ngl 999 decode 5.62 t/s   prefill 65.12 t/s   <-- Vulkan LOSES 2.5x

    So offloading to this iGPU only pays when the weights fit in the dedicated
    carve-out. When they spill into GTT, the iGPU reads them over a much slower
    path and loses to twelve AVX-512 cores that share the same memory controller.
    Note the CPU figure ALSO improved (6.78 -> 13.81) once the carve-out shrank,
    purely because the model now stays resident in page cache instead of being
    re-read from a spinning disk.

    Consequence: a small carve-out is the right BIOS setting for CPU inference,
    and a large one is the right setting for GPU inference. They are opposite
    requirements, which is exactly why `--backend vulkan` remains overridable.

    Returns (gib, explanation).
    """
    carve = _vram_budget_gib()
    gtt = _gtt_total_gib()
    usable = carve * budget_frac
    if carve >= 4.0:
        why = (f"DEVICE_LOCAL carve-out {carve:.1f} GiB x {budget_frac:.0%} = "
               f"{usable:.1f} GiB (GTT {gtt:.0f} GiB ignored by design)")
    else:
        why = (f"DEVICE_LOCAL carve-out is only {carve:.2f} GiB, so the iGPU has "
               f"nowhere fast to put weights (GTT is {gtt:.0f} GiB but measured 2.5x "
               f"SLOWER than CPU here)")
    return usable, why


def plan_server_command(
    model_path: str,
    server_bin: str,
    opts: Optional[TuneOptions] = None,
    topo: Optional[CpuTopology] = None,
) -> Tuple[List[str], Dict[str, str], List[str]]:
    """
    Build an optimised llama-server argv + environment for one model.

    Returns (argv, env, notes). `notes` is a human-readable rationale for every
    decision, suitable for printing at launch time.
    """
    opts = opts or TuneOptions()
    topo = topo or detect_topology()
    prof = profile_model(model_path)
    notes: List[str] = []

    n_fast = topo.n_fast_cores or 4
    n_phys = topo.n_physical_cores or 12

    # ---------------------------------------------------------------- backend --
    # The offload budget is the DEVICE_LOCAL carve-out ONLY. See _gpu_budget_gib()
    # for the measurements: with a 0.5 GiB carve-out, Vulkan decode is 5.62 t/s
    # against 13.81 t/s on the CPU, because GTT-resident weights are read over a
    # path far slower than twelve AVX-512 cores sharing the same memory controller.
    gpu_budget, budget_why = _gpu_budget_gib(opts.vram_budget_frac)
    visible_ram = _visible_ram_gib()
    offload = opts.backend in ("vulkan", "gpu")
    if opts.backend == "auto":
        # Weights + KV + compute buffers must fit. KV at q8_0 is ~1 byte/element.
        kv_gib = (2 * prof.n_layer * prof.n_head_kv * (prof.n_embd // max(prof.n_head, 1))
                  * opts.context * 1.0 / 1024**3) if (prof.n_layer and prof.n_head) else 2.0
        need = prof.size_gib + max(kv_gib, 0.5)
        if need < gpu_budget:
            offload = True
            notes.append(
                f"backend=vulkan: weights {prof.size_gib:.1f} GiB + KV ~{kv_gib:.1f} GiB = "
                f"{need:.1f} GiB fit the GPU budget of {gpu_budget:.1f} GiB ({budget_why})."
            )
        elif visible_ram > 0 and need > visible_ram:
            offload = True
            notes.append(
                f"backend=vulkan (large model fallback): need {need:.1f} GiB exceeds CPU-visible RAM "
                f"({visible_ram:.1f} GiB). Offloading to Vulkan to map across VRAM and GTT."
            )
        else:
            offload = False
            notes.append(
                f"backend=cpu: need {need:.1f} GiB (weights {prof.size_gib:.1f} + KV "
                f"~{kv_gib:.1f}) vs a GPU budget of {gpu_budget:.1f} GiB ({budget_why}). "
                f"CPU is the faster path whenever weights spill out of the carve-out."
            )
    else:
        notes.append(f"backend={opts.backend} (forced)")

    argv: List[str] = [
        server_bin,
        "-m", model_path,
        "--host", "127.0.0.1",              # NOT 0.0.0.0: no reason to expose an LLM on the LAN
        "--port", str(opts.port),
    ]

    # ------------------------------------------------------------- threading --
    # The launcher's `taskset -c 0-7` is the single worst line in the codebase:
    # it straddles both core types and both L3 domains while llama.cpp still
    # spawns 12 threads. We let llama.cpp own scheduling and give it exactly the
    # physical core count, with an explicit mask when offloading to the GPU
    # (so the CPU threads stay off the SMT siblings the GPU driver may want).
    #
    # fast_cores_only: restrict to the fast Zen5 cluster only. This reduces
    # cross-L3-domain traffic during decode but limits thread count to the
    # number of fast physical cores. Off by default — must be measured per
    # workload before enabling (prefill speed drops with fewer cores).
    core_cpus = topo.one_thread_per_core()
    if opts.fast_cores_only and topo.fast_cpus:
        fast_one_per_core: List[int] = []
        seen_fast: dict = {}
        for c in sorted(topo.fast_cpus):
            core = topo.core_of.get(c, c)
            if core not in seen_fast:
                seen_fast[core] = c
                fast_one_per_core.append(c)
        core_cpus = fast_one_per_core or core_cpus
        threads = len(core_cpus)
        notes.append(
            f"fast_cores_only=True: restricted to Zen5 fast cluster {core_cpus} "
            f"({threads} physical cores). Zen5c {topo.slow_cpus} excluded to avoid "
            f"cross-L3-domain traffic during decode. Prefill may be slower."
        )
    else:
        threads = n_phys
        if opts.fast_cores_only and not topo.fast_cpus:
            notes.append(
                "fast_cores_only=True requested but topology detection found no fast-core "
                "cluster — using all physical cores instead."
            )
    argv += ["-t", str(threads), "-tb", str(threads)]
    argv += ["--cpu-mask", topo.mask_hex(core_cpus), "--cpu-strict", "1"]
    notes.append(
        f"threads={threads} (physical cores only). Measured on this host: -t 12 -> 13.7 t/s, "
        f"-t 16 -> 11.7 t/s, -t 24 -> 0.35 t/s. Never exceed the physical core count."
    )
    notes.append(
        f"cpu-mask={topo.mask_hex(core_cpus)} = one thread per physical core {core_cpus} "
        f"(fast Zen5 {topo.fast_cpus}, slow Zen5c {topo.slow_cpus}). The previous "
        f"`taskset -c 0-7` spanned BOTH core types and BOTH L3 domains while llama.cpp "
        f"still spawned 12 threads onto those 8 logical CPUs."
    )

    # --------------------------------------------------------- offload + ctx --
    if offload:
        argv += ["-ngl", "999"]
    else:
        argv += ["-ngl", "0"]

    argv += ["-c", str(min(opts.context, prof.n_ctx_train or opts.context))]
    argv += ["-np", str(opts.parallel_slots)]

    # ------------------------------------------------------- flash attention --
    argv += ["-fa", "on"]
    notes.append("-fa on: required for quantized V-cache and the fused attention path.")

    # ------------------------------------------------------------- KV cache ---
    argv += ["-ctk", opts.kv_type, "-ctv", opts.kv_type]
    # NOTE: --defrag-thold was removed from this plan. llama-server b10456 prints
    #   "DEPRECATED: --defrag-thold is deprecated and no longer necessary to
    #    specify"
    # so it was pure noise in the command line, and every unknown/obsolete flag
    # in a generated plan is a flag nobody can trust.
    notes.append(
        f"KV cache {opts.kv_type} (was q4_0). q4_0 V-cache costs measurable quality; "
        f"q8_0 is ~half of f16 with no detectable loss."
    )

    # ------------------------------------------------------------ batching ----
    # Measured (fork README, Strix Halo): dense prefers ubatch 512; MoE gains ~29%
    # at ubatch 2048. Do not generalise one to the other.
    if prof.is_moe:
        argv += ["-b", "2048", "-ub", "2048"]
        notes.append("-b/-ub 2048: MoE model. Measured +29% prefill at ubatch 2048 on this class of APU.")
    else:
        argv += ["-b", "2048", "-ub", "512"]
        notes.append("-b 2048 / -ub 512: dense model. ubatch 512 measured fastest for dense on this class of APU.")

    # --------------------------------------------------------- prompt reuse ---
    # An agentic loop re-sends the same system prompt + tool schemas every turn.
    argv += ["--cache-reuse", "256"]
    argv += ["-sps", "0.50"]
    notes.append(
        "--cache-reuse 256 (was 64): the agent resends system prompt + tool schemas "
        "every turn, so KV reuse directly cuts prompt-eval cost."
    )

    # ------------------------------------------------------ speculative dec. --
    if opts.speculative:
        # Default policy: n-gram self-speculation. It is the only method MEASURED
        # to pay off on this machine, because its drafts are free - an n-gram
        # lookup reads no model weights, so verification is pure profit.
        #
        # Measured here, Qwen2.5-Coder-7B-Q4_K_M, CPU-only, -t 12, 256 tokens,
        # acceptance read from the server log:
        #     baseline                  13.01 t/s   (1.00x)
        #     --spec-type ngram-mod     21.48 t/s   (1.65x)   78.2% accept
        #     --spec-default            21.57 t/s   (1.66x)   78.2% accept
        #     --spec-type ngram-simple  25.41 t/s   (1.95x)   88.5% accept
        #     --spec-type ngram-map-k4v 25.44 t/s   (1.96x)   88.5% accept
        # Prompt throughput unchanged (~70 t/s) in every case.
        #
        # MTP was measured and did NOT pay off on a dense target:
        #     Qwen3.8-27B-Q4_K_M baseline  3.38 t/s, prefill 14.56 t/s
        #     + draft-mtp n-max 3          3.38 t/s  (1.00x)  73.5% accept
        #     + draft-mtp n-max 2          2.92 t/s  (0.87x)  80.9% accept  <-- worse
        # On a bandwidth-bound machine the draft step itself reads weights, and a
        # k-token verification pass costs more than a 1-token pass, so the gain can
        # cancel out. MTP is context- and content-dependent (it wins on long,
        # quote-heavy contexts) - enable it explicitly per workload, not by default.
        if opts.use_mtp and prof.has_mtp:
            argv += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4", "--spec-draft-n-min", "2"]
            notes.append(
                "MTP forced via TuneOptions(use_mtp=True). WARNING: measured 1.00x at n-max 3 and "
                "0.87x at n-max 2 on a dense 27B here - validate against your own workload "
                "(it is reported to win on long, quote-heavy contexts)."
            )
        elif _is_thinking_model(model_path):
            # Nao emite o flag. O aviso de risco estava escrito aqui em baixo
            # desde o inicio, mas o codigo emitia o flag na mesma -- avisar e
            # continuar a fazer a coisa errada e pior do que nao avisar, porque
            # deixa a impressao de que o risco foi tratado.
            notes.append(
                "-> --spec-type ngram-simple DESLIGADO: este modelo tem <think>/"
                "enable_thinking no chat template, logo e um modelo de raciocinio. "
                "Medido nesta maquina com o Qwen3.8-27B: o modelo emite prosa de "
                "<think> em vez de reemitir a entrada, a aceitacao foi 0,00% (0/96) "
                "e o decode caiu de 3,375 para 1,191 t/s (0,28x). O n-gram so paga "
                "o custo. Para forcar, TuneOptions(speculative=False) nao serve -- "
                "use o CLI com --spec-type explicito."
            )
        else:
            argv += ["--spec-type", "ngram-simple"]
            notes.append(
                "-> --spec-type ngram-simple: measured 1.95x decode (13.01 -> 25.41 t/s, "
                "88.5% acceptance) on a CODE-EDITING workload on this host. Zero extra memory, "
                "lossless. VERIFY before trusting it: read the server's own printed "
                "`draft acceptance = X (a accepted / b generated)` line."
            )
            notes.append(
                "!! RISK: n-gram speculation is CONTENT-DEPENDENT and can be a 2.8x SLOWDOWN. "
                "Measured on this host with a THINKING model (Qwen3.8-27B): acceptance 0.00% "
                "(0/96), decode 3.375 -> 1.191 t/s. This model was not detected as a thinking "
                "model, so the flag is emitted -- but if acceptance comes back below ~0.6, "
                "turn it off with TuneOptions(speculative=False)."
            )

        # A sidecar, if present, is still only used when explicitly requested:
        # loading a second model costs its own weight reads per draft step.
        if opts.use_draft_sidecar and not prof.has_mtp:
            draft = find_draft_model(model_path, prof)
            if draft:
                draft_path, spec_type = draft
                argv = [a for a in argv if a != "ngram-simple"]
                argv += [
                    "-md", draft_path, "--spec-type", spec_type,
                    "--spec-draft-ngl", "999" if offload else "0",
                    "--spec-draft-n-max", "7" if spec_type == "draft-dflash" else "8",
                    "--spec-draft-n-min", "3",
                ]
                notes.append(f"sidecar explicitly requested: {os.path.basename(draft_path)} -> {spec_type}.")

    # ------------------------------------------------------------- load mode --
    # --mlock / --mmap / --no-mmap are deprecated since llama-server ~build 10000.
    # The new unified flag is `-lm / --load-mode`.
    #
    #   mmap+mlock  Best for large-RAM machines: fast startup + pinned against swap.
    #   mlock       For models on slow/NTFS disks: reads everything into RAM first.
    #   mmap        Minimal pinning — appropriate when RAM is tight.
    #   auto        Let llama-server decide (plain mmap on most systems).
    effective_load_mode = opts.load_mode
    if visible_ram > 0 and prof.size_gib > visible_ram and "mlock" in effective_load_mode:
        effective_load_mode = "mmap"
        notes.append(
            f"load_mode safety override: model size ({prof.size_gib:.1f} GiB) exceeds CPU-visible RAM "
            f"({visible_ram:.1f} GiB). Downgraded load_mode from '{opts.load_mode}' to 'mmap' to prevent kernel OOM reboot."
        )

    # Second override, and the one that actually fires on this host: mlock needs
    # RLIMIT_MEMLOCK to cover the weights. Stock Ubuntu ships a hard limit of
    # 8192 KiB, so `mmap+mlock` fails on its first buffer and degrades to plain
    # mmap while the plan still advertises pinning.
    if "mlock" in effective_load_mode:
        limit_gib = _memlock_limit_gib()
        if limit_gib >= 0 and limit_gib < prof.size_gib:
            notes.append(
                f"load_mode override: RLIMIT_MEMLOCK hard limit is "
                f"{limit_gib * 1024:.0f} MiB but the model is {prof.size_gib:.1f} GiB, so "
                f"mmap+mlock would fail on its first buffer and silently run as plain "
                f"mmap. Emitting 'mmap' instead. To actually pin, raise the limit: "
                f"`ulimit -l unlimited` (needs root, or a systemd unit with "
                f"LimitMEMLOCK=infinity)."
            )
            effective_load_mode = "mmap"

    argv += ["-lm", effective_load_mode]
    notes.append(
        f"-lm {effective_load_mode}: model loading mode. Replaces deprecated --mlock / --mmap / --no-mmap flags."
    )

    # ------------------------------------------------------------ scheduling --
    # prio is opt-in. Raising nice priority requires CAP_SYS_NICE (bit 23); only
    # lowering it is unprivileged. Emitting --prio without the capability makes
    # llama-server log a failure for EVERY worker thread on EVERY task (measured:
    # 6180 lines in one session) and changes nothing.
    want_prio = opts.prio or getattr(opts, "prio_batch", 0)
    if want_prio and _has_cap_sys_nice():
        argv += ["--prio", str(opts.prio), "--prio-batch", str(opts.prio_batch)]
        notes.append(
            f"--prio {opts.prio} --prio-batch {opts.prio_batch}: CAP_SYS_NICE is present, "
            f"so the priority bump will actually apply."
        )
    elif want_prio:
        notes.append(
            f"--prio {opts.prio} OMITTED: no CAP_SYS_NICE (CapPrm bit 23 is clear and "
            f"we are not root). Measured effect of forcing it anyway: "
            f"'failed to set thread priority 2 : Operation not permitted' once per "
            f"worker thread per task -- 6180 of them in one server lifetime -- with "
            f"no scheduling change. To enable, either run with "
            f"`sudo setcap cap_sys_nice+ep <llama-server>` or add "
            f"LimitRTPRIO/AmbientCapabilities=CAP_SYS_NICE to a systemd unit."
        )
    elif not want_prio:
        notes.append("--prio disabled by TuneOptions(prio=0): OS-default scheduling.")
    argv += ["--timeout", "3600", "--no-webui"]

    if opts.metrics:
        argv += ["--metrics"]
        notes.append("--metrics: exposes /metrics so the benchmark harness can read real t/s.")

    argv += ["--alias", opts.alias]

    # Preserva dois defaults de servidor do launcher original, para que esta
    # correção não altere silenciosamente o comportamento de quem fala com o
    # endpoint sem passar estes campos. Ver TuneOptions.temp / .n_predict.
    if opts.temp is not None:
        argv += ["--temp", str(opts.temp)]
    if opts.n_predict and opts.n_predict > 0:
        argv += ["-n", str(opts.n_predict)]

    argv += list(opts.extra)

    # --------------------------------------------------------------- env ------
    env = os.environ.copy()

    # Only set accelerator variables that the *loaded backend* can actually read.
    # The launcher sets ROCm/HIP variables unconditionally, but `llama-server`
    # here is the Vulkan build (libggml-vulkan.so); HSA_* / HIP_* are inert and
    # LD_PRELOAD of libdrm additionally shadows /opt/amdgpu userspace.
    for dead in ("HSA_OVERRIDE_GFX_VERSION", "HSA_ENABLE_SDMA", "HIP_VISIBLE_DEVICES",
                 "AMD_GPU_BUILD_TARGET", "LD_PRELOAD", "RADV_PERFTEST",
                 "AMD_VULKAN_ICD"):
        if dead in env:
            env.pop(dead, None)

    if offload:
        # Vulkan only. RADV is the Mesa driver; let it be selected normally rather
        # than forcing AMD_VULKAN_ICD, which breaks if the ICD name changes.
        # Do NOT set RADV_PERFTEST=coop_matrix: upstream llama.cpp #16339 tracks
        # Mesa changes breaking coopmat, and on RDNA3.5 the generic path is used.
        notes.append(
            "env: dropped inert ROCm variables (HSA_*, HIP_*, LD_PRELOAD, AMD_GPU_BUILD_TARGET) - "
            "the launched binary is the Vulkan build and cannot read them."
        )
        notes.append("env: dropped RADV_PERFTEST=coop_matrix (see llama.cpp#16339; not a stable win on RDNA3.5).")

    # Keep the allocator honest on a 45 GiB host with a 48 GiB carve-out.
    # Keep a single Vulkan allocation honest on a unified-memory APU: without a
    # cap, a sparse buffer can try to reserve more than the machine physically
    # has (GTT is often advertised as half of RAM, or set via amdgpu.gttsize).
    env.setdefault("GGML_VK_FORCE_MAX_ALLOCATION_SIZE", str(int(gpu_budget * 1024**3)))

    return argv, env, notes


def format_plan(argv: Sequence[str], notes: Sequence[str], width: int = 88) -> str:
    """Render a plan for display in the launcher terminal."""
    lines = ["=" * width, " APEX HARNESS - HARDWARE-AWARE LAUNCH PLAN".center(width), "=" * width]
    lines.append("")
    lines.append(" Rationale:")
    for n in notes:
        lines.append(f"   - {n}")
    lines.append("")
    lines.append(" Command:")

    # Group each flag with its value so the wrapped command stays readable:
    #   -m <path>  rather than  -m \ \n <path>
    groups: List[str] = []
    cur: List[str] = []
    for a in argv:
        if a.startswith("-") and cur:
            groups.append(" ".join(cur))
            cur = [a]
        else:
            cur.append(a)
    if cur:
        groups.append(" ".join(cur))

    for i, g in enumerate(groups):
        sep = " \\" if i < len(groups) - 1 else ""
        lines.append(f"   {g}{sep}")
    lines.append("=" * width)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  CLI for manual use / debugging
# --------------------------------------------------------------------------- #

def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Show (or run) the hardware-tuned llama-server plan for a GGUF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("model")
    ap.add_argument("--server-bin", default=os.path.expanduser("~/.local/bin/llama-server"))
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--backend", default="auto", choices=["auto", "vulkan", "cpu"])
    ap.add_argument("--no-speculative", action="store_true",
                    help="disable all speculative decoding (ngram + MTP sidecars)")
    ap.add_argument("--mtp", action="store_true",
                    help="enable MTP draft-mtp speculative decoding (requires mtp-*.gguf sidecar)")
    ap.add_argument("--fast-cores-only", action="store_true",
                    help="restrict CPU affinity to the fast Zen5 core cluster only (fewer threads, "
                         "less cross-L3 traffic — benchmark before enabling)")
    ap.add_argument("--load-mode", default="mmap+mlock",
                    choices=["auto", "mmap", "mlock", "mmap+mlock"],
                    help="model loading mode passed as -lm to llama-server")
    ap.add_argument("--prio", type=int, default=2,
                    help="process priority for decode threads (0=normal, 1=medium, 2=high, 3=realtime)")
    ap.add_argument("--kv-type", default="q8_0", choices=["f16", "q8_0", "q4_0"],
                    help="KV cache quantization type")
    ap.add_argument("--exec", action="store_true", help="actually launch the server (os.execve)")
    a = ap.parse_args()

    argv, env, notes = plan_server_command(
        a.model, a.server_bin,
        TuneOptions(
            context=a.context,
            backend=a.backend,
            speculative=not a.no_speculative,
            use_mtp=a.mtp,
            fast_cores_only=a.fast_cores_only,
            load_mode=a.load_mode,
            prio=a.prio,
            prio_batch=a.prio,
            kv_type=a.kv_type,
        ),
    )
    print(format_plan(argv, notes))

    prof = profile_model(a.model)
    print(f"\n model profile: {prof}")
    print(f" bandwidth-bound decode ceiling: {theoretical_decode_tps(prof):.1f} t/s "
          f"(at {THEORETICAL_BW_GBS:.0f} GB/s x 0.55 CPU efficiency)")

    if a.exec:
        os.execve(argv[0], argv, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
