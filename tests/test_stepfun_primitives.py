from __future__ import annotations

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.ops import (
    step_apply_rope,
    step_headwise_attention_gate,
    step_rmsnorm,
    step_rope_tables,
)


def test_step_rmsnorm_uses_weight_offset_semantics() -> None:
    x = np.asarray([[1.0, -2.0, 3.0, -4.0]], dtype=np.float32)
    weight = np.asarray([0.0, 0.5, -0.25, 1.0], dtype=np.float32)

    out = step_rmsnorm(x, weight, eps=1.0e-5)

    variance = np.mean(x * x, axis=-1, keepdims=True)
    expected = (x / np.sqrt(variance + 1.0e-5)) * (weight + 1.0)
    np.testing.assert_allclose(out, expected.astype(np.float32), rtol=1e-6, atol=1e-6)


def test_step_rope_tables_cover_full_and_sliding_modes() -> None:
    full_cos, full_sin = step_rope_tables(
        max_positions=8192,
        head_dim=128,
        partial_factor=0.5,
        theta=5_000_000.0,
        llama3_scaling=True,
    )
    sliding_cos, sliding_sin = step_rope_tables(
        max_positions=8,
        head_dim=128,
        partial_factor=1.0,
        theta=10_000.0,
        llama3_scaling=False,
    )
    unscaled_full_cos, _ = step_rope_tables(
        max_positions=8192,
        head_dim=128,
        partial_factor=0.5,
        theta=5_000_000.0,
        llama3_scaling=False,
    )

    assert full_cos.shape == full_sin.shape == (8192, 32)
    assert sliding_cos.shape == sliding_sin.shape == (8, 64)
    np.testing.assert_allclose(full_cos[0], 1.0)
    np.testing.assert_allclose(full_sin[0], 0.0)
    np.testing.assert_allclose(sliding_cos[0], 1.0)
    np.testing.assert_allclose(sliding_sin[0], 0.0)
    assert not np.allclose(full_cos[-1], unscaled_full_cos[-1])


def test_step_apply_rope_rotates_only_partial_full_attention_dimension() -> None:
    x = np.zeros((2, 3, 128), dtype=np.float32)
    x[..., :64] = np.arange(64, dtype=np.float32)
    x[..., 64:] = 100.0

    out = step_apply_rope(
        x,
        np.asarray([0, 1], dtype=np.int64),
        head_dim=128,
        partial_factor=0.5,
        theta=5_000_000.0,
        llama3_scaling=True,
    )

    np.testing.assert_allclose(out[0, :, :64], x[0, :, :64], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out[..., 64:], 100.0)
    assert not np.allclose(out[1, :, :64], x[1, :, :64])


def test_step_headwise_attention_gate_applies_sigmoid_per_head() -> None:
    attn = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    logits = np.asarray([[0.0, 2.0, -2.0], [1.0, -1.0, 0.5]], dtype=np.float32)

    out = step_headwise_attention_gate(attn, logits)

    expected = attn * (1.0 / (1.0 + np.exp(-logits)))[..., None]
    np.testing.assert_allclose(out, expected.astype(np.float32), rtol=1e-6, atol=1e-6)

    with pytest.raises(ValueError, match="gate_logits"):
        step_headwise_attention_gate(attn, logits[:, :2])
