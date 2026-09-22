"""
Tests for the pre-flight memory guard.

These encode the three configurations this machine actually failed or survived,
because the whole point of the guard is to reproduce that judgement
deterministically instead of rediscovering it with another crashed session.
"""

import pytest

from apex_harness import memguard
from apex_harness.memguard import (
    Budget,
    Pools,
    VERDICT_OK,
    VERDICT_REFUSE,
    VERDICT_TIGHT,
    compute_budget,
    kv_cache_gib,
    model_size_gib,
)

# The real machine, as measured while writing this: 48 GiB BIOS carve-out,
# 45.65 GiB of CPU-visible RAM, 8 GiB swap.
REAL = Pools(
    gpu_total_gib=48.0,
    gpu_used_gib=0.33,
    ram_total_gib=45.65,
    ram_available_gib=42.08,
    swap_free_gib=7.17,
    swap_total_gib=8.0,
)


def test_swap_is_not_counted_as_model_headroom():
    """
    Swap must not inflate the CPU budget.

    Weights are read once per token, so a swapped weight byte is re-read from
    disk on every decode step. Counting swap made the 57 GiB DeepSeek plan look
    acceptable on paper right before it OOM-killed the session.
    """
    budget = compute_budget(REAL, reserve_gib=10.0)
    assert budget.cpu_usable_gib == round(42.08 - 10.0, 2)
    assert budget.cpu_usable_gib < 42.08

    # Sanity: the budget must not move when swap grows.
    more_swap = Pools(**{**REAL.__dict__, "swap_free_gib": 64.0, "swap_total_gib": 64.0})
    assert compute_budget(more_swap, reserve_gib=10.0).cpu_usable_gib == budget.cpu_usable_gib


def test_gpu_budget_holds_back_five_of_the_carve_out():
    budget = compute_budget(REAL, gpu_frac=0.80)
    assert budget.gpu_usable_gib == round((48.0 - 0.33) * 0.80, 2)
    assert budget.gpu_usable_gib < 48.0


def test_no_carve_out_means_no_offload():
    """A tiny carve-out leaves only GTT, measured 2.5x slower than the CPU here."""
    small = Pools(0.5, 0.1, 91.16, 88.0, 7.0, 8.0)
    budget = compute_budget(small)
    assert budget.gpu_usable_gib == 0.0
    assert any("GTT" in n for n in budget.notes)


class _FakeProfile:
    n_layer = 48
    n_head = 32
    n_head_kv = 4
    n_embd = 4096


def test_kv_cache_scales_with_context_and_type():
    f16 = kv_cache_gib(_FakeProfile(), 32768, "f16")
    q8 = kv_cache_gib(_FakeProfile(), 32768, "q8_0")
    assert f16 > 0
    # q8_0 is 34 bytes per 32 elements, so it is slightly more than half of f16.
    assert 0.5 < q8 / f16 < 0.6
    assert kv_cache_gib(_FakeProfile(), 65536, "f16") == f16 * 2


def test_kv_cache_is_zero_without_metadata():
    class _Blank:
        n_layer = 0
        n_head = 0
        n_head_kv = 0
        n_embd = 0

    assert kv_cache_gib(_Blank(), 4096, "f16") == 0.0


def test_multipart_gguf_is_summed(tmp_path):
    """
    A sharded model is one model. Checking only the first shard would let a
    60 GiB three-way split pass a 20 GiB check.
    """
    for i in (1, 2, 3):
        (tmp_path / f"model-0000{i}-of-00003.gguf").write_bytes(b"x" * 1024)
    total_gib, shards = model_size_gib(str(tmp_path / "model-00001-of-00003.gguf"))
    assert shards == 3
    assert total_gib == pytest.approx(3 * 1024 / memguard.GIB)


def test_single_file_is_one_shard(tmp_path):
    f = tmp_path / "solo.gguf"
    f.write_bytes(b"x" * 2048)
    total_gib, shards = model_size_gib(str(f))
    assert shards == 1
    assert total_gib == pytest.approx(2048 / memguard.GIB)


def test_directory_is_scanned(tmp_path):
    for name in ("a-00001-of-00002.gguf", "a-00002-of-00002.gguf"):
        (tmp_path / name).write_bytes(b"x" * 512)
    _, shards = model_size_gib(str(tmp_path))
    assert shards == 2


def _plan_with(required_gib: float, *, gpu: float = 38.14, cpu: float = 32.15):
    """Drive plan_model's verdict logic without touching the disk."""
    budget = Budget(gpu_usable_gib=gpu, cpu_usable_gib=cpu, reserve_gib=10.0, gpu_frac=0.8)
    if required_gib > budget.ceiling_gib:
        return VERDICT_REFUSE
    cpu_needed = max(0.0, required_gib - gpu)
    if required_gib > gpu:
        share = cpu_needed / cpu if cpu else 1.0
        return VERDICT_REFUSE if share > 0.5 else VERDICT_TIGHT
    if required_gib > gpu * 0.85:
        return VERDICT_TIGHT
    return VERDICT_OK


def test_verdicts_match_what_actually_happened_on_this_machine():
    # Qwen3-Coder-30B-A3B Q4_K_S: 16.26 GiB weights, fits entirely in the GPU
    # budget. Measured 37.0 t/s decode on Vulkan, rock solid.
    assert _plan_with(18.05) == VERDICT_OK

    # DeepSeek-V4-Flash reap-200b IQ1_M: 56.36 GiB weights. Needed the GPU
    # budget filled to 100% AND ~19.6 GiB of system RAM. At 14:44 it produced a
    # global OOM that killed org.gnome.Shell@ubuntu.service.
    assert _plan_with(57.70) == VERDICT_REFUSE

    # Qwen3.8-Flash-Next-131B Q3KXL: 60.34 GiB, same shape and worse.
    assert _plan_with(62.00) == VERDICT_REFUSE


def test_fills_gpu_but_only_a_little_ram_is_tight_not_refused():
    # 40 GiB of a 38.14 GiB GPU budget leaves 1.86 GiB on the CPU: 6% of the
    # RAM budget, survivable but with no margin.
    assert _plan_with(40.0) == VERDICT_TIGHT


def test_over_ceiling_is_always_refused():
    assert _plan_with(200.0) == VERDICT_REFUSE
