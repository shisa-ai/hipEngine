"""Candidate policy scopes must not leak into later benchmark arms."""

import pytest

from hipengine.kernels import hip_gfx1151
from hipengine.runtime import gguf_linear
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
        assert len(counts) == 3
        assert set(counts.values()) == {0}
