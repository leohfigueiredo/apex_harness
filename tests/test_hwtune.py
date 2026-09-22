"""
Tests for hardware tuning and model planning safety in apex_harness.hwtune
"""

from unittest.mock import patch
from apex_harness.hwtune import plan_server_command, TuneOptions, ModelProfile, CpuTopology


def test_hwtune_large_model_safety():
    """Verify that models larger than CPU-visible RAM downgrade mlock and force Vulkan offload."""
    mock_prof = ModelProfile(
        path="/dummy/deepseek-v4.gguf",
        size_gib=80.76,
        n_layer=60,
        n_expert=8,
        n_expert_used=2,
        n_head=32,
        n_head_kv=8,
        n_embd=4096,
        n_ctx_train=32768,
        has_mtp=False,
    )
    topo = CpuTopology(fast_cpus=[0, 1], slow_cpus=[2, 3])

    with patch("apex_harness.hwtune.profile_model", return_value=mock_prof), \
         patch("apex_harness.hwtune._visible_ram_gib", return_value=45.6), \
         patch("apex_harness.hwtune._gpu_budget_gib", return_value=(40.8, "carveout 48 GiB")):

        opts = TuneOptions(backend="auto", load_mode="mmap+mlock")
        argv, env, notes = plan_server_command("/dummy/deepseek-v4.gguf", "/usr/bin/llama-server", opts, topo)

        # Check load_mode was downgraded to mmap
        assert "-lm" in argv
        lm_idx = argv.index("-lm")
        assert argv[lm_idx + 1] == "mmap"

        # Check offload was set to Vulkan (-ngl 999)
        assert "-ngl" in argv
        ngl_idx = argv.index("-ngl")
        assert argv[ngl_idx + 1] == "999"

        # Verify notes mention the safety override
        notes_text = " ".join(notes)
        assert "exceeds cpu-visible ram" in notes_text.lower()


def _argv_for(model_path, meta, **opts):
    """Build a plan for a synthetic model without touching the disk."""
    prof = ModelProfile(path=model_path, size_gib=4.0, n_layer=32, n_head=32,
                        n_head_kv=8, n_embd=4096, n_ctx_train=32768)
    with patch("apex_harness.hwtune.profile_model", return_value=prof), \
         patch("apex_harness.hwtune.read_gguf_metadata", return_value=meta), \
         patch("apex_harness.hwtune.find_draft_model", return_value=None), \
         patch("apex_harness.hwtune.detect_topology", return_value=CpuTopology(
             fast_cpus=[0, 1, 2, 3, 12, 13, 14, 15],
             slow_cpus=[4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23])):
        argv, env, notes = plan_server_command(model_path, "/bin/llama-server",
                                                TuneOptions(**opts))
    return argv, env, notes


def test_ngram_speculation_is_skipped_for_thinking_models():
    """
    A thinking model must NOT get n-gram speculation.

    Measured on this host: Qwen3.8-27B emits <think> prose instead of re-emitting
    the input, acceptance was 0.00% (0/96), and decode fell 3.375 -> 1.191 t/s
    (0.28x). The code used to warn about exactly this and then emit the flag
    anyway, which is worse than not warning at all.
    """
    argv, _, notes = _argv_for(
        "/models/Swift-Qwen3.8-27B-Q4_K_M.gguf",
        {"tokenizer.chat_template": "...<think>...</think>...enable_thinking..."},
    )
    assert "ngram-simple" not in argv
    assert any("DESLIGADO" in n for n in notes)


def test_ngram_speculation_is_kept_for_non_thinking_models():
    """
    A plain coder model keeps the flag: measured 1.95x (13.01 -> 25.41 t/s) with
    88.5% acceptance on a code-editing workload here.

    Note the template deliberately contains "reasoning" alone -- that word appears
    in the real Qwen3-Coder template too and must NOT be enough to classify a
    model as thinking.
    """
    argv, _, _ = _argv_for(
        "/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_S.gguf",
        {"tokenizer.chat_template": "You are a helpful assistant. reasoning effort: medium"},
    )
    assert "ngram-simple" in argv


def test_thinking_detection_falls_back_to_the_name():
    """A GGUF with no readable chat template still gets classified by name."""
    argv, _, _ = _argv_for("/models/Swift-Qwen3.8-27B-Q4_K_M.gguf", {})
    assert "ngram-simple" not in argv


def test_prio_is_omitted_without_cap_sys_nice():
    """
    Raising scheduling priority needs CAP_SYS_NICE.

    Measured as a normal user, `--prio 2` produced "failed to set thread priority
    2 : Operation not permitted" 6180 times in one server lifetime with no
    scheduling change.
    """
    with patch("apex_harness.hwtune._has_cap_sys_nice", return_value=False):
        argv, _, notes = _argv_for("/models/x.gguf", {}, prio=2, prio_batch=2)
    assert "--prio" not in argv
    assert any("CAP_SYS_NICE" in n for n in notes)

    with patch("apex_harness.hwtune._has_cap_sys_nice", return_value=True):
        argv, _, _ = _argv_for("/models/x.gguf", {}, prio=2, prio_batch=2)
    assert "--prio" in argv


def test_mlock_is_downgraded_when_rlimit_memlock_is_too_small():
    """
    Stock Ubuntu ships an 8192 KiB hard RLIMIT_MEMLOCK, so `-lm mmap+mlock` fails
    on its first buffer and silently runs as plain mmap while the plan claims the
    weights are pinned.
    """
    with patch("apex_harness.hwtune._has_cap_sys_nice", return_value=False), \
         patch("apex_harness.hwtune._memlock_limit_gib", return_value=8192 / 1024**2):
        argv, _, notes = _argv_for("/models/x.gguf", {}, load_mode="mmap+mlock")
    assert argv[argv.index("-lm") + 1] == "mmap"
    assert any("RLIMIT_MEMLOCK" in n for n in notes)

    with patch("apex_harness.hwtune._has_cap_sys_nice", return_value=False), \
         patch("apex_harness.hwtune._memlock_limit_gib", return_value=-1.0):
        argv, _, _ = _argv_for("/models/x.gguf", {}, load_mode="mmap+mlock")
    assert argv[argv.index("-lm") + 1] == "mmap+mlock"


def test_deprecated_defrag_thold_is_not_emitted():
    """llama-server b10456 prints 'DEPRECATED: --defrag-thold' for it."""
    argv, _, _ = _argv_for("/models/x.gguf", {})
    assert "--defrag-thold" not in argv


def _gguf_com_tipos(tmp_path, tipos):
    """Escreve um GGUF minimo com os tipos de tensor indicados."""
    import struct

    p = tmp_path / "m.gguf"
    with open(p, "wb") as fh:
        fh.write(b"GGUF")
        fh.write(struct.pack("<I", 3))          # version
        fh.write(struct.pack("<Q", len(tipos)))  # n_tensors
        fh.write(struct.pack("<Q", 0))           # n_kv
        for i, t in enumerate(tipos):
            nome = f"blk.{i}.weight".encode()
            fh.write(struct.pack("<Q", len(nome)))
            fh.write(nome)
            fh.write(struct.pack("<I", 1))       # n_dims
            fh.write(struct.pack("<Q", 4))       # dim
            fh.write(struct.pack("<I", t))       # ggml type
            fh.write(struct.pack("<Q", 0))       # offset
    return str(p)


def test_prism_only_quant_types_are_detected(tmp_path):
    """
    Tipos 142/143 sao da PrismML; o llama.cpp mainline vai ate ao 39.

    Medido: Ternary-Bonsai-2-27B-PQ2_0.gguf tem 402 tensores do tipo 142 e o
    PTQ1_0 tem 402 do tipo 143. O llama-server so rebenta no ARRANQUE --
    'failed to load model ... exiting due to model loading error' -- e o harness
    fica sem backend. Detetar no seletor evita a espera.
    """
    from apex_harness.hwtune import unsupported_quant_types

    assert unsupported_quant_types(_gguf_com_tipos(tmp_path, [0, 12, 142, 142])) == [142]

    tmp2 = tmp_path / "sub"
    tmp2.mkdir()
    assert unsupported_quant_types(_gguf_com_tipos(tmp2, [0, 30, 143])) == [143]


def test_standard_quant_types_pass(tmp_path):
    from apex_harness.hwtune import unsupported_quant_types

    # 0 F32, 12 Q4_K, 13 Q5_K, 14 Q6_K, 30 BF16 -- tudo mainline.
    assert unsupported_quant_types(_gguf_com_tipos(tmp_path, [0, 12, 13, 14, 30])) == []


def test_unreadable_file_is_not_flagged(tmp_path):
    """Um ficheiro ilegivel devolve lista vazia, nao uma acusacao falsa."""
    from apex_harness.hwtune import unsupported_quant_types

    p = tmp_path / "nao_e_gguf.bin"
    p.write_bytes(b"isto nao e um gguf")
    assert unsupported_quant_types(str(p)) == []
    assert unsupported_quant_types(str(tmp_path / "inexistente.gguf")) == []
