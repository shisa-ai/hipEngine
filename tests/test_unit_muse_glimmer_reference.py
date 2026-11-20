"""Unit tests for the Muse Glimmer CPU reference primitives (synthetic)."""

from __future__ import annotations

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.muse_glimmer import (
    MuseGlimmerConfig,
    apply_rope_interleaved,
    causal_swa_mask,
    gqa_attention,
    head_rmsnorm,
    rmsnorm,
    rope_tables,
)


def _config(**overrides) -> MuseGlimmerConfig:
    base = dict(
        n_layers=4,
        n_embd=8,
        n_ff=16,
        n_heads=2,
        n_kv_heads=1,
        head_dim=4,
        vocab_size=16,
        rope_freq_base=10000.0,
        rms_eps=1e-5,
        post_norm_eps=1e-8,
        sliding_window=3,
        swa_pattern=(True, True, True, False),
        logit_scale=0.5,
        final_logit_softcapping=20.0,
    )
    base.update(overrides)
    return MuseGlimmerConfig(**base)


def test_config_rejects_pattern_length_mismatch() -> None:
    with pytest.raises(ValueError, match="swa_pattern"):
        _config(swa_pattern=(True, False))


def test_config_rejects_non_divisible_heads() -> None:
    with pytest.raises(ValueError, match="multiple"):
        _config(n_heads=3, n_kv_heads=2)


def test_rmsnorm_weightless_matches_manual() -> None:
    x = np.array([[3.0, 4.0]], dtype=np.float32)
    out = rmsnorm(x, None, 0.0)
    expected = x / np.sqrt(np.mean(x * x))
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_head_rmsnorm_normalizes_each_head_independently() -> None:
    x = np.array([[[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 5.0, 12.0]]], dtype=np.float32)
    weight = np.full(4, 2.0, dtype=np.float32)
    out = head_rmsnorm(x, weight, 0.0)
    np.testing.assert_allclose(out[0, 0], [2.4, 3.2, 0.0, 0.0], rtol=1e-5)
    np.testing.assert_allclose(out[0, 1], [0.0, 0.0, 20.0 / 13, 48.0 / 13], rtol=1e-5)


def test_rope_interleaves_adjacent_pairs_not_halves() -> None:
    # Position 1, head_dim 4, base 10000: angles = [1, 0.01]
    cos, sin = rope_tables(np.array([1]), 4, 10000.0)
    x = np.array([[[1.0, 0.0, 0.0, 1.0]]], dtype=np.float32)
    out = apply_rope_interleaved(x, cos, sin)
    # pair0: (1,0) rotated by 1 rad; pair1: (0,1) rotated by 0.01 rad
    np.testing.assert_allclose(
        out[0, 0],
        [np.cos(1.0), np.sin(1.0), -np.sin(0.01), np.cos(0.01)],
        rtol=1e-5,
        atol=1e-7,
    )


def test_rope_position_zero_is_identity() -> None:
    cos, sin = rope_tables(np.array([0]), 8, 500000.0)
    x = np.random.default_rng(7).normal(size=(1, 2, 8)).astype(np.float32)
    np.testing.assert_allclose(apply_rope_interleaved(x, cos, sin), x, atol=1e-7)


def test_swa_mask_window_and_causality() -> None:
    positions = np.array([4, 5])
    kv_positions = np.arange(6)
    mask = causal_swa_mask(positions, 6, kv_positions, sliding_window=3)
    # query 4 attends keys 2,3,4 (k > 4-3); query 5 attends 3,4,5
    assert mask[0].tolist() == [False, False, True, True, True, False]
    assert mask[1].tolist() == [False, False, False, True, True, True]
    full = causal_swa_mask(positions, 6, kv_positions, None)
    assert full[0].tolist() == [True, True, True, True, True, False]


def test_swa_mask_boundary_values_on_both_sides() -> None:
    # Exercise window edges and an unrelated position, not just the threshold.
    kv_positions = np.arange(10)
    for window in (2, 5, 7):
        mask = causal_swa_mask(np.array([9]), 10, kv_positions, window)
        allowed = np.flatnonzero(mask[0])
        assert allowed.tolist() == list(range(10 - window, 10))


def test_gqa_attention_broadcasts_kv_heads() -> None:
    q = np.ones((1, 4, 2), dtype=np.float32)
    k = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
    v = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32)
    mask = np.ones((1, 2), dtype=bool)
    out = gqa_attention(q, k, v, mask, n_heads=4, n_kv_heads=1, scale=1.0)
    # Identical queries over both keys -> mean of values for every head.
    np.testing.assert_allclose(out, np.full((1, 4, 2), [2.0, 3.0]), rtol=1e-6)
