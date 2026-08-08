"""Hand-checkable NumPy oracle fixtures for Maple ternary inference."""

from __future__ import annotations

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.maple import (
    affine4_gemv_f32,
    attention_decode,
    bf16_round,
    clamped_swiglu,
    dequantize_affine4,
    dequantize_ternary,
    partial_rope,
    router_topk,
    ternary_gemv,
    unpack_affine4_codes,
    unpack_ternary_codes,
    weighted_residual,
)
from hipengine.kernels.registry import resolve


def pack2(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.uint32)
    return np.sum(codes.reshape(*codes.shape[:-1], -1, 16) << np.arange(0, 32, 2), axis=-1).astype(
        np.uint32
    )


def pack4(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.uint32)
    return np.sum(codes.reshape(*codes.shape[:-1], -1, 8) << np.arange(0, 32, 4), axis=-1).astype(
        np.uint32
    )


def test_ternary_unpack_dequant_and_gemv_are_lsb_first() -> None:
    ternary = np.asarray(
        [
            [-1, 0, 1, -1, 1, 0, 0, 1, -1, 1, 0, -1, 1, 1, 0, -1],
            [1, 1, 0, -1, 0, 1, -1, 0, 1, -1, 1, 0, -1, 0, 1, 0],
        ],
        dtype=np.int8,
    )
    packed = pack2(ternary.astype(np.int32) + 1)
    alpha = np.asarray([0.5, 2.0], dtype=np.float32)
    x = np.arange(1, 17, dtype=np.float32)

    assert np.array_equal(unpack_ternary_codes(packed), ternary)
    assert np.array_equal(dequantize_ternary(packed, alpha), ternary * alpha[:, None])
    assert np.allclose(ternary_gemv(x, packed, alpha), (ternary * alpha[:, None]) @ x)


def test_affine4_unpack_dequant_and_gemv_use_group_scales_and_biases() -> None:
    codes = np.asarray(
        [
            list(range(16)),
            list(reversed(range(16))),
        ],
        dtype=np.uint8,
    )
    packed = pack4(codes)
    scales = np.asarray([[0.5, 2.0], [1.0, -0.25]], dtype=np.float32)
    biases = np.asarray([[-1.0, 3.0], [0.5, 4.0]], dtype=np.float32)
    x = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    expected = (
        codes.reshape(2, 2, 8).astype(np.float32) * scales[..., None]
        + biases[..., None]
    ).reshape(2, 16)

    assert np.array_equal(unpack_affine4_codes(packed), codes)
    assert np.allclose(
        dequantize_affine4(packed, scales, biases, group_size=8), expected
    )
    assert np.allclose(
        affine4_gemv_f32(x, packed, scales, biases, group_size=8), expected @ x
    )


def test_partial_rope_uses_unmodified_rotate_half_pairs_and_passes_tail() -> None:
    x = np.asarray([[1.0, 2.0, 3.0, 4.0, 9.0, 10.0]], dtype=np.float32)
    got = partial_rope(x, pos=1, rope_theta=1.0, rope_dim=4)
    c = np.cos(np.float32(1.0))
    s = np.sin(np.float32(1.0))
    expected = np.asarray(
        [[1 * c - 3 * s, 2 * c - 4 * s, 3 * c + 1 * s, 4 * c + 2 * s, 9, 10]],
        dtype=np.float32,
    )
    assert np.allclose(got, expected, atol=1e-6, rtol=1e-6)


def test_gqa_attention_maps_contiguous_q_groups_to_their_kv_head() -> None:
    q = np.ones((4, 2), dtype=np.float32)
    k = np.asarray([[[1, 0], [0, 1]]], dtype=np.float32)
    v = np.asarray([[[2, 3], [7, 11]]], dtype=np.float32)
    got = attention_decode(q, k, v, scale=1.0)
    assert np.array_equal(got, np.asarray([[2, 3], [2, 3], [7, 11], [7, 11]]))


def test_router_softmax_topk_renormalizes_selected_scores() -> None:
    x = np.asarray([1.0, 0.0], dtype=np.float32)
    gate = np.asarray([[4, 0], [3, 0], [2, 0], [1, 0]], dtype=np.float32)
    ids, scores = router_topk(x, gate, top_k=2)
    expected = np.exp(np.asarray([4.0, 3.0], dtype=np.float32))
    expected /= expected.sum()
    assert ids.tolist() == [0, 1]
    assert np.allclose(scores, expected, atol=1e-7)
    assert float(scores.sum()) == pytest.approx(1.0)


def test_clamped_swiglu_and_weighted_residual_preserve_trained_boundaries() -> None:
    gate = np.asarray([-10.0, 0.0, 10.0], dtype=np.float32)
    up = np.asarray([-10.0, 2.0, 10.0], dtype=np.float32)
    clipped_gate = np.minimum(gate, 7.0)
    expected = clipped_gate / (1.0 + np.exp(-clipped_gate)) * np.clip(up, -7.0, 7.0)
    assert np.allclose(clamped_swiglu(gate, up), expected)

    residual = bf16_round(np.asarray([1.1, -2.2, 3.3], dtype=np.float32))
    experts = bf16_round(
        np.asarray([[0.25, 1.5, -4.0], [1.0, -0.5, 0.75]], dtype=np.float32)
    )
    weights = np.asarray([0.25, 0.75], dtype=np.float32)
    combined = bf16_round(np.sum(experts * weights[:, None], axis=0, dtype=np.float32))
    expected_residual = bf16_round(residual + combined)
    assert np.array_equal(weighted_residual(residual, experts, weights), expected_residual)
    with pytest.raises(ValueError, match="one routing weight"):
        weighted_residual(residual, experts, weights[:1])


def test_maple_cpu_reference_primitives_resolve_through_four_axis_registry() -> None:
    assert resolve(
        backend="cpu_reference",
        layer="maple_ternary_gemv",
        quant="maple_ternary2",
        variant="row_alpha",
    ) is ternary_gemv
