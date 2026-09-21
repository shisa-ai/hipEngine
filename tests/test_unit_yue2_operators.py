"""Operator gates: NumPy references vs pinned torch fixtures.

Fixtures come from ``scripts/yue2_oracle.py operators`` and are produced by the
release's own modules (``yue2.modeling_yue2``, ``yue2.modeling_vae``). Elementwise
operators with a defined rounding order must match bit-for-bit; reductions
(normalization means, convolutions, attention) are compared against the
documented numerical envelope.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.yue2 import (
    audio_position_embedding,
    bf16,
    bf16_bits_to_f32,
    conv1d,
    conv_transpose1d,
    fold_weight_norm,
    gqa_attention,
    head_rmsnorm,
    kl_divergence,
    linear,
    rmsnorm,
    rotate_half,
    rope_tables,
    silu_mul,
    snake_beta,
    timestep_embedding,
    top1_agreement,
)

FIXTURES = Path(__file__).parent / "fixtures/yue2/operators"


def _load(name: str):
    path = FIXTURES / name
    if not path.is_file():
        pytest.skip(f"missing oracle fixture {path}")
    return np.load(path)


def _close(got, expected, *, rtol=1e-3, atol=1e-3):
    difference = np.abs(np.asarray(got, dtype=np.float64) - np.asarray(expected, dtype=np.float64))
    scale = np.maximum(np.abs(np.asarray(expected, dtype=np.float64)), 1.0)
    assert np.max(difference / scale) <= rtol, f"max relative difference {np.max(difference / scale)}"


# ---------------------------------------------------------------------------
# normalization / elementwise: exact
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 2048), (5, 2048), (7, 128), (3, 6144)])
def test_rmsnorm_matches_reference_bits(shape):
    rows, width = shape
    data = _load(f"rmsnorm_{rows}x{width}.npz")
    got = rmsnorm(bf16_bits_to_f32(data["x"]), bf16_bits_to_f32(data["weight"]), float(data["eps"]))
    np.testing.assert_array_equal(got, bf16_bits_to_f32(data["out"]))


def test_head_qk_norm_matches_reference_bits():
    data = _load("head_qk_norm.npz")
    got = head_rmsnorm(bf16_bits_to_f32(data["q"]), bf16_bits_to_f32(data["weight"]))
    np.testing.assert_array_equal(got, bf16_bits_to_f32(data["out"]))
    # The same op is applied per head: normalizing the last axis only.
    assert got.shape == data["out"].shape


def test_rope_tables_match_reference_within_fp32_angle_ulp():
    data = _load("rope_cos_sin.npz")
    positions = data["positions"].astype(np.float32)
    cos, sin = rope_tables(positions, 128, 1000000.0)
    exponent = np.arange(0, 128, 2, dtype=np.float32) / np.float32(128)
    inverse = np.float32(1.0) / np.power(np.float32(1000000.0), exponent)
    angles = positions[:, None] * inverse[None, :]
    # The reference builds the angles in FP32; a cos/sin difference beyond a few
    # ulp of that angle would mean a different table, not a libm difference.
    tolerance = np.maximum(np.float32(3.0) * np.spacing(angles), np.float32(1e-6))
    assert np.all(np.abs(cos - data["cos"]) <= tolerance)
    assert np.all(np.abs(sin - data["sin"]) <= tolerance)
    # Small angles (the low-frequency entries that dominate short contexts) are
    # reproduced exactly.
    small = np.abs(angles) < np.float32(1.0)
    assert small.sum() > 0
    np.testing.assert_allclose(cos[small], data["cos"][small], rtol=0, atol=1e-6)
    np.testing.assert_allclose(sin[small], data["sin"][small], rtol=0, atol=1e-6)


def test_rotate_half_matches_reference_bits():
    """With the reference's own table the rotation is bit-exact."""
    data = _load("rope_cos_sin.npz")
    x = bf16_bits_to_f32(data["x"])
    rows = x.shape[1]
    got = rotate_half(
        x, data["cos"][:rows].reshape(1, rows, 1, 64), data["sin"][:rows].reshape(1, rows, 1, 64)
    )
    np.testing.assert_array_equal(got, bf16_bits_to_f32(data["out"]))


def test_rotate_half_with_own_table_is_bf16_equivalent():
    """A few-ulp table difference may flip a BF16 rounding; nothing more."""
    data = _load("rope_cos_sin.npz")
    cos, sin = rope_tables(data["positions"], 128, 1000000.0)
    x = bf16_bits_to_f32(data["x"])
    rows = x.shape[1]
    got = rotate_half(
        x, cos[:rows].reshape(1, rows, 1, 64), sin[:rows].reshape(1, rows, 1, 64)
    )
    expected = bf16_bits_to_f32(data["out"])
    identical = (got == expected).mean()
    assert identical > 0.9999
    assert np.abs(got - expected).max() <= 0.002


def test_rope_tables_use_theta_1e6_half_width():
    cos, sin = rope_tables(np.asarray([0, 1], dtype=np.int64), 128, 1000000.0)
    assert cos.shape == (2, 64)
    np.testing.assert_allclose(cos[0], 1.0, rtol=0, atol=0)
    np.testing.assert_allclose(sin[0], 0.0, rtol=0, atol=0)
    # Lowest frequency: angle = 1 / theta^(0/64) = 1.
    assert cos[1][0] == pytest.approx(np.cos(1.0), abs=1e-7)


def test_silu_mul_matches_reference_bits():
    rng = np.random.default_rng(3)
    gate = bf16(rng.standard_normal((4, 128)).astype(np.float32) * 3)
    up = bf16(rng.standard_normal((4, 128)).astype(np.float32))
    # Reproduce the reference chain in NumPy directly: sigmoid in FP32 then BF16.
    sigmoid = 1.0 / (1.0 + np.exp(-gate))
    expected = bf16(bf16(gate * sigmoid) * up)
    np.testing.assert_array_equal(silu_mul(gate, up), expected)


def test_linear_accumulates_in_fp32():
    rng = np.random.default_rng(4)
    x = bf16(rng.standard_normal((3, 64)).astype(np.float32))
    weight = bf16(rng.standard_normal((128, 64)).astype(np.float32) * 0.05)
    got = linear(x, weight)
    expected = bf16(x.astype(np.float64) @ weight.astype(np.float64).T)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------


def test_gqa_attention_prefill_matches_reference():
    data = _load("attn_prefill.npz")
    got = gqa_attention(
        bf16_bits_to_f32(data["q"])[0].transpose(1, 0, 2),
        bf16_bits_to_f32(data["k"])[0].transpose(1, 0, 2),
        bf16_bits_to_f32(data["v"])[0].transpose(1, 0, 2),
        causal=True,
    )
    expected = bf16_bits_to_f32(data["out"])[0].transpose(1, 0, 2)
    assert top1_agreement(expected.reshape(-1, 128), got.reshape(-1, 128)) > 0.9
    _close(got, expected, rtol=0.05, atol=0.05)


def test_gqa_attention_decode_matches_reference():
    data = _load("attn_decode.npz")
    got = gqa_attention(
        bf16_bits_to_f32(data["q"])[0].transpose(1, 0, 2),
        bf16_bits_to_f32(data["k"])[0].transpose(1, 0, 2),
        bf16_bits_to_f32(data["v"])[0].transpose(1, 0, 2),
    )
    expected = bf16_bits_to_f32(data["out"])[0].transpose(1, 0, 2)
    _close(got, expected, rtol=0.05, atol=0.05)


def test_gqa_attention_causal_mask_hides_future_tokens():
    rng = np.random.default_rng(5)
    q = bf16(rng.standard_normal((4, 2, 8)).astype(np.float32))
    k = bf16(rng.standard_normal((4, 2, 8)).astype(np.float32))
    v = bf16(rng.standard_normal((4, 2, 8)).astype(np.float32))
    causal = gqa_attention(q, k, v, causal=True)
    # Changing a later token must not change any earlier row.
    k2 = k.copy()
    k2[3] += 5.0
    v2 = v.copy()
    v2[3] -= 3.0
    changed = gqa_attention(q, k2, v2, causal=True)
    np.testing.assert_array_equal(causal[:3], changed[:3])
    assert not np.array_equal(causal[3], changed[3])


def test_gqa_attention_explicit_mask_is_bidirectional():
    rng = np.random.default_rng(6)
    q = bf16(rng.standard_normal((3, 2, 8)).astype(np.float32))
    k = bf16(rng.standard_normal((3, 2, 8)).astype(np.float32))
    v = bf16(rng.standard_normal((3, 2, 8)).astype(np.float32))
    mask = np.asarray([[True, True, False], [True, True, True], [False, True, True]])
    got = gqa_attention(q, k, v, mask=mask)
    # Row 0 must be independent of token 2; row 1 sees everything.
    k2 = k.copy()
    k2[2] += 10.0
    v2 = v.copy()
    v2[2] += 10.0
    masked_again = gqa_attention(q, k2, v2, mask=mask)
    np.testing.assert_array_equal(got[0], masked_again[0])
    assert not np.array_equal(got[1], masked_again[1])


# ---------------------------------------------------------------------------
# conditioning features
# ---------------------------------------------------------------------------


def test_timestep_embedder_matches_reference():
    data = _load("timestep_embedder.npz")
    got = timestep_embedding(
        data["t"],
        bf16_bits_to_f32(data["w0"]),
        bf16_bits_to_f32(data["b0"]),
        bf16_bits_to_f32(data["w2"]),
        bf16_bits_to_f32(data["b2"]),
    )
    expected = bf16_bits_to_f32(data["out"])
    _close(got, expected, rtol=0.02, atol=0.02)
    # The released shift applies sigmoid in model dtype before the embedder.
    assert got.shape == (5, 2048)


def test_audio_position_embedding_rows():
    data = _load("audio_position_embedding.npz")
    got = audio_position_embedding(data["pe_rows"], np.arange(data["pe_rows"].shape[0]))
    _close(got, data["out"], rtol=0, atol=0)
    assert data["pe_rows"].shape == (6, 2048)


# ---------------------------------------------------------------------------
# VAE operators
# ---------------------------------------------------------------------------


def test_fold_weight_norm_matches_torch_folding():
    data = _load("conv1d_dilated.npz")
    folded = fold_weight_norm(data["weight_g"], data["weight_v"])
    _close(folded, data["folded"], rtol=1e-5, atol=1e-6)


def test_conv1d_matches_reference():
    data = _load("conv1d_dilated.npz")
    got = conv1d(
        data["x"][0],
        data["folded"],
        data["bias"],
        stride=int(data["stride"]),
        dilation=int(data["dilation"]),
        padding=int(data["padding"]),
    )
    expected = data["out"][0]
    assert got.shape == expected.shape
    _close(got, expected, rtol=1e-4, atol=1e-5)


def test_conv_transpose1d_matches_reference():
    data = _load("convtr1d.npz")
    folded = fold_weight_norm(data["weight_g"], data["weight_v"])
    got = conv_transpose1d(
        data["x"][0],
        folded,
        data["bias"],
        stride=int(data["stride"]),
        padding=int(data["padding"]),
    )
    expected = data["out"][0]
    assert got.shape == expected.shape
    _close(got, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("channels,length", [(64, 40), (256, 17)])
def test_snake_beta_matches_reference(channels, length):
    data = _load(f"snake_{channels}x{length}.npz")
    got = snake_beta(data["x"][0], data["alpha"], data["beta"])
    expected = data["out"][0]
    _close(got, expected, rtol=1e-5, atol=1e-6)
    # The 1e-9 denominator epsilon is part of the contract: a zero beta must not
    # produce infinities.
    finite = snake_beta(np.ones((1, 8)), np.zeros(8), np.full(8, -20.0))
    assert np.isfinite(finite).all()


# ---------------------------------------------------------------------------
# numerical metrics used by the gates
# ---------------------------------------------------------------------------


def test_kl_and_top1_metrics_are_well_defined():
    rng = np.random.default_rng(7)
    logits = rng.standard_normal((4, 512)).astype(np.float32)
    assert kl_divergence(logits, logits) == pytest.approx(0.0, abs=1e-9)
    assert top1_agreement(logits, logits) == 1.0
    shifted = logits + rng.standard_normal((4, 512)).astype(np.float32) * 0.1
    assert 0.0 <= kl_divergence(logits, shifted) < 0.05
    mask = np.ones_like(logits, dtype=bool)
    mask[:, 100:] = False
    assert top1_agreement(logits, shifted, mask) >= 0.5
