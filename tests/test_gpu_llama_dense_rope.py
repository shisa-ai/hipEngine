"""Standard rotate-half RoPE over prefill rows, checked bit-for-bit.

The kernel is the dense Llama rotary owner: per-row cos/sin tables, BF16 Q/K
rotated in place, and one thread per dim-pair so an in-place rotation has no
read/write race. The reference is the HF ``rotate_half`` formulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._llama_tiny import bf16_bits, bf16_round
from tests._rocm_guard import hip_runtime_available

pytestmark = pytest.mark.skipif(not hip_runtime_available(), reason="HIP runtime unavailable")


def _rope_table(positions: np.ndarray, rotary_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(positions, dtype=np.float64)
    inv_freq = 1.0 / np.power(theta, np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim)
    angles = pos[:, None] * inv_freq[None, :]
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    return np.concatenate([cos, cos], axis=1), np.concatenate([sin, sin], axis=1)


def _rotate_half_reference(
    values: np.ndarray,
    positions: np.ndarray,
    rotary_dim: int,
    theta: float,
) -> np.ndarray:
    cos, sin = _rope_table(positions, rotary_dim, theta)
    x = np.asarray(values, dtype=np.float32).copy()
    half = rotary_dim // 2
    first = x[..., :half].copy()
    second = x[..., half:rotary_dim].copy()
    out = x.copy()
    out[..., :half] = first * cos[:, None, :half] - second * sin[:, None, :half]
    out[..., half:rotary_dim] = second * cos[:, None, half:] + first * sin[:, None, half:]
    return bf16_round(out)


def _run_rope(query, key, positions, *, head_dim, rotary_dim, theta=10000.0, q_heads=None, kv_heads=None):
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.rotary.dense_rope import (
        dense_rope_positions_bf16_inplace,
    )

    rows = query.shape[0]
    q_heads = query.shape[1] if q_heads is None else q_heads
    kv_heads = key.shape[1] if kv_heads is None else kv_heads
    cos, sin = _rope_table(np.asarray(positions, dtype=np.int64), rotary_dim, theta)
    buffers = []

    def upload(array):
        array = np.ascontiguousarray(array)
        buffer = malloc(array.nbytes)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
        return buffer

    try:
        q_buf = upload(query)
        k_buf = upload(key)
        cos_buf = upload(cos)
        sin_buf = upload(sin)
        dense_rope_positions_bf16_inplace(
            q_buf.ptr,
            k_buf.ptr,
            cos_buf.ptr,
            sin_buf.ptr,
            rows,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
        )
        out_q = np.empty(query.shape, dtype=np.uint16)
        out_k = np.empty(key.shape, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(out_q), q_buf, out_q.nbytes)
        copy_device_to_host(host_array_ptr(out_k), k_buf, out_k.nbytes)
    finally:
        for buffer in buffers:
            free(buffer)
    to_f32 = lambda bits: (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return to_f32(out_q), to_f32(out_k)


@pytest.mark.parametrize(
    ("rows", "q_heads", "kv_heads", "head_dim", "rotary_dim"),
    [
        (3, 4, 2, 8, 8),
        (9, 8, 4, 32, 32),
        (5, 7, 1, 128, 128),
        (4, 6, 2, 64, 16),
        (1, 2, 2, 16, 16),
    ],
)
def test_rope_matches_rotate_half_bit_for_bit(rows, q_heads, kv_heads, head_dim, rotary_dim):
    rng = np.random.default_rng(rows * 31 + head_dim)
    query = bf16_bits(rng.standard_normal((rows, q_heads, head_dim)))
    key = bf16_bits(rng.standard_normal((rows, kv_heads, head_dim)))
    positions = np.arange(rows, dtype=np.int64) * 3 + 5

    got_q, got_k = _run_rope(
        query, key, positions, head_dim=head_dim, rotary_dim=rotary_dim
    )
    to_f32 = lambda bits: (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    expected_q = _rotate_half_reference(to_f32(query), positions, rotary_dim, 10000.0)
    expected_k = _rotate_half_reference(to_f32(key), positions, rotary_dim, 10000.0)
    assert np.array_equal(got_q, expected_q)
    assert np.array_equal(got_k, expected_k)


def test_partial_rotary_leaves_the_tail_untouched():
    rng = np.random.default_rng(11)
    query = bf16_bits(rng.standard_normal((3, 4, 32)))
    key = bf16_bits(rng.standard_normal((3, 2, 32)))
    positions = np.array([0, 7, 64], dtype=np.int64)
    got_q, got_k = _run_rope(query, key, positions, head_dim=32, rotary_dim=8)
    to_f32 = lambda bits: (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    assert np.array_equal(got_q[..., 8:], to_f32(query)[..., 8:])
    assert np.array_equal(got_k[..., 8:], to_f32(key)[..., 8:])
    assert not np.array_equal(got_q[..., :8], to_f32(query)[..., :8])


def test_rope_validates_its_geometry():
    from hipengine.kernels.hip_gfx1100.rotary.dense_rope import (
        dense_rope_positions_bf16_inplace,
    )

    pointer = 1 << 20
    with pytest.raises(ValueError, match="rotary_dim"):
        dense_rope_positions_bf16_inplace(pointer, pointer, pointer, pointer, 2, 4, 2, 8, 10)
    with pytest.raises(ValueError, match="even"):
        dense_rope_positions_bf16_inplace(pointer, pointer, pointer, pointer, 2, 4, 2, 8, 3)
    with pytest.raises(ValueError, match="threads"):
        dense_rope_positions_bf16_inplace(
            pointer, pointer, pointer, pointer, 2, 4, 2, 8, 8, threads=32
        )
    with pytest.raises(ValueError, match="rows"):
        dense_rope_positions_bf16_inplace(pointer, pointer, pointer, pointer, 0, 4, 2, 8, 8)
