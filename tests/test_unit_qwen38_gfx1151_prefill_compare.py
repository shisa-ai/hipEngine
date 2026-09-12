"""Candidate policy scopes must not leak into later benchmark arms."""

import pytest

from hipengine.kernels import hip_gfx1151
from hipengine.runtime import gguf_linear
from hipengine.kernels.registry import KernelKey, register, resolve
from scripts.qwen38_gfx1151_prefill_compare import candidate_scope, dispatch_counts


def test_unequal_policy_is_restored_after_failure():
    name = "GGUF_Q4_T16_UNEQUAL_PAIR_PREFILL_POLICIES"
    present = hasattr(hip_gfx1151, name)
    previous = getattr(hip_gfx1151, name, None)
    with pytest.raises(RuntimeError):
        with candidate_scope("unequal_pair"):
            assert any(getattr(hip_gfx1151, name).values())
            raise RuntimeError("capture failed")
    assert hasattr(hip_gfx1151, name) is present
    assert getattr(hip_gfx1151, name, None) is previous


def test_retile_scope_restores_cached_selector():
    previous = gguf_linear._Q4_T16_DUAL_SILU_RETILE_RESOLVED
    with candidate_scope("no_retiles"):
        assert gguf_linear._q4_t16_dual_silu_retile_enabled() is False
    assert gguf_linear._Q4_T16_DUAL_SILU_RETILE_RESOLVED is previous


def test_unknown_candidate_fails_closed():
    with pytest.raises(ValueError):
        with candidate_scope("typo"):
            pass


def test_dispatch_probe_resolves_real_backend_keys_without_launching():
    with dispatch_counts() as counts:
        assert len(counts) == 4
        assert set(counts.values()) == {0}


def test_unequal_probe_counts_the_runtime_pair_dispatch_key():
    key = KernelKey("hip_gfx1151", "linear_pair", "gguf_q4_k_t16_v1",
                    "dense_unequal_dual_wmma_prefill_bf16_bf16_out")
    original = resolve(backend=key.backend, layer=key.layer, quant=key.quant,
                       variant=key.variant)
    register(key, lambda: None, replace=True)
    try:
        with dispatch_counts() as counts:
            resolve(backend=key.backend, layer=key.layer, quant=key.quant,
                    variant=key.variant)()
            assert counts[key.variant] == 1
    finally:
        register(key, original, replace=True)


def test_row48_scope_restores_backend_capability():
    name = "GGUF_Q4_DUAL_SILU_PREFILL_ROW48_MAX_ROWS"
    present = hasattr(hip_gfx1151, name)
    previous = getattr(hip_gfx1151, name, None)
    with candidate_scope("row48"):
        assert getattr(hip_gfx1151, name) == 48
    assert hasattr(hip_gfx1151, name) is present
    assert getattr(hip_gfx1151, name, None) is previous
