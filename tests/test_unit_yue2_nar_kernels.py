"""M4 kernel gate: YuE2 NAR elementwise and attention kernels vs NumPy.

The elementwise kernels must be bit-exact: each mirrors one torch op on BF16
tensors, so the only arithmetic is a single FP32 add/multiply followed by one
BF16 rounding. The attention kernel accumulates FP32 dot products with FMA and a
tiled online softmax, so it is gated against an FP64 NumPy reference with a
relative tolerance.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_array_to_device,
    host_array_ptr,
    malloc,
)
from hipengine.runtime.yue2_ar import bf16_bits_to_f32
from hipengine.runtime.yue2_nar import to_bf16_bits

FIXTURES = Path(__file__).resolve().parent / "fixtures/yue2"


def _has_hip() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


requires_hip = pytest.mark.skipif(not _has_hip(), reason="ROCm/HIP runtime is not available")


def _upload(array: np.ndarray):
    host = np.ascontiguousarray(array)
    buffer = malloc(max(host.nbytes, 8))
    copy_host_array_to_device(buffer, host)
    return buffer


def _download(buffer, shape, dtype):
    out = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(out), buffer)
    return out


def _bf16(values) -> np.ndarray:
    return to_bf16_bits(np.asarray(values, dtype=np.float32))


def widen(bits) -> np.ndarray:
    return bf16_bits_to_f32(bits)


@requires_hip
def test_gather_add_is_bit_exact():
    from hipengine.kernels.hip_gfx1100.yue2 import nar

    rows, width = 5, 7
    rng = np.random.default_rng(0)
    x = _bf16(rng.standard_normal((rows, width)) * 3)
    table = _bf16(rng.standard_normal((9, width)))
    positions = np.asarray([0, 3, 8, 8, 1], dtype=np.int64)
    x_buf = _upload(x)
    table_buf = _upload(table)
    pos_buf = _upload(positions)
    out_buf = _upload(np.zeros((rows, width), dtype=np.uint16))
    nar.nar_gather_add_bf16(
        x_buf.ptr, table_buf.ptr, pos_buf.ptr, out_buf.ptr, rows, width
    )
    got = _download(out_buf, (rows, width), np.uint16)
    expected = _bf16(widen(x) + widen(table)[positions])
    assert np.array_equal(got, expected)


@requires_hip
def test_add_broadcast_is_bit_exact():
    from hipengine.kernels.hip_gfx1100.yue2 import nar

    rows, width = 6, 5
    rng = np.random.default_rng(1)
    x = _bf16(rng.standard_normal((rows, width)))
    vector = _bf16(rng.standard_normal(width))
    x_buf = _upload(x)
    vec_buf = _upload(vector)
    out_buf = _upload(np.zeros((rows, width), dtype=np.uint16))
    nar.nar_add_broadcast_bf16(x_buf.ptr, vec_buf.ptr, out_buf.ptr, rows, width)
    got = _download(out_buf, (rows, width), np.uint16)
    assert np.array_equal(got, _bf16(widen(x) + widen(vector)[None, :]))


@requires_hip
def test_state_update_rounds_twice_like_torch():
    from hipengine.kernels.hip_gfx1100.yue2 import nar

    rows, width = 4, 6
    rng = np.random.default_rng(2)
    state = _bf16(rng.standard_normal((rows, width)) * 2)
    velocity = _bf16(rng.standard_normal((rows, width)) * 3)
    scale = -0.015625
    state_buf = _upload(state)
    vel_buf = _upload(velocity)
    out_buf = _upload(np.zeros((rows, width), dtype=np.uint16))
    nar.nar_state_update_bf16(state_buf.ptr, vel_buf.ptr, scale, out_buf.ptr, rows, width)
    got = _download(out_buf, (rows, width), np.uint16)
    product = widen(_bf16(widen(velocity) * np.float32(scale)))
    assert np.array_equal(got, _bf16(widen(state) - product))


def _attention_reference(q, nar_k, nar_v, ar_k, ar_v, num_q_heads, num_kv_heads, scale):
    """FP64 NumPy reference for bidirectional GQA over [AR | NAR] keys."""
    keys = widen(np.concatenate([ar_k, nar_k], axis=0)).astype(np.float64)
    values = widen(np.concatenate([ar_v, nar_v], axis=0)).astype(np.float64)
    group = num_q_heads // num_kv_heads
    qf = np.asarray(q, dtype=np.float64)
    out = np.zeros((q.shape[0], num_q_heads, q.shape[2]), dtype=np.float64)
    for head in range(num_q_heads):
        kv_head = head // group
        scores = qf[:, head, :] @ keys[:, kv_head, :].T * scale
        scores -= scores.max(axis=1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=1, keepdims=True)
        out[:, head, :] = weights @ values[:, kv_head, :]
    return out


@requires_hip
@pytest.mark.parametrize("ar_rows,nar_rows", [(512, 34), (0, 40), (300, 7), (1, 1), (0, 1)])
@pytest.mark.parametrize("head_dim", [128, 100])
def test_attention_matches_the_numpy_reference(ar_rows, nar_rows, head_dim):
    """Both the packed and the scalar K path agree with the reference.

    ``head_dim`` 128 takes the eight-wide loads; 100 is not a multiple of eight,
    so the rows are not 16-byte aligned and the kernel must fall back to the
    scalar walk. Both have to produce the same attention.
    """
    from hipengine.kernels.hip_gfx1100.yue2 import nar

    num_q_heads, num_kv_heads = 16, 8
    rng = np.random.default_rng(ar_rows * 131 + nar_rows + head_dim)
    q = (rng.standard_normal((nar_rows, num_q_heads, head_dim)) * 0.5).astype(np.float32)
    nar_k = _bf16(rng.standard_normal((nar_rows, num_kv_heads, head_dim)))
    nar_v = _bf16(rng.standard_normal((nar_rows, num_kv_heads, head_dim)))
    ar_k = _bf16(rng.standard_normal((ar_rows, num_kv_heads, head_dim))) if ar_rows else np.zeros((0, num_kv_heads, head_dim), dtype=np.uint16)
    ar_v = _bf16(rng.standard_normal((ar_rows, num_kv_heads, head_dim))) if ar_rows else np.zeros((0, num_kv_heads, head_dim), dtype=np.uint16)
    scale = 1.0 / float(np.sqrt(head_dim))
    q_buf = _upload(q)
    nk_buf = _upload(nar_k)
    nv_buf = _upload(nar_v)
    ak_buf = _upload(ar_k)
    av_buf = _upload(ar_v)
    out_buf = _upload(np.zeros((nar_rows, num_q_heads, head_dim), dtype=np.float32))
    nar.nar_attention_f32(
        q_buf.ptr, nk_buf.ptr, nv_buf.ptr, ak_buf.ptr, av_buf.ptr, out_buf.ptr,
        nar_rows, ar_rows, num_q_heads, num_kv_heads, head_dim, scale,
    )
    got = _download(out_buf, (nar_rows, num_q_heads, head_dim), np.float32)
    expected = _attention_reference(q, nar_k, nar_v, ar_k, ar_v, num_q_heads, num_kv_heads, scale)
    assert np.allclose(got, expected, rtol=2e-3, atol=2e-3), (
        f"max abs diff {np.abs(got - expected).max()}"
    )


@requires_hip
@pytest.mark.parametrize(
    "ar_rows,nar_rows",
    [(512, 34), (0, 40), (300, 7), (1, 1), (0, 1), (256, 96), (128, 256)],
)
def test_wmma_attention_matches_the_numpy_reference(ar_rows, nar_rows):
    """The tensor-core attention is a production-profile variant, not an exact one.

    It runs the query/key and probability/value products on f16 WMMA with an f16
    output accumulator, which is the arithmetic class the pinned upstream's own
    attention kernel uses, so it cannot be bit-identical to the scalar kernel.
    What it must do is stay inside the M4 solver gate's envelope against the
    independent FP64 reference: the gate allows rel L2 0.05 and cosine 0.999, and
    this asserts a tighter 0.02 / 0.9995 so a regression shows up before the gate.
    """
    from hipengine.kernels.hip_gfx1100.yue2 import nar

    num_q_heads, num_kv_heads, head_dim = 16, 8, 128
    rng = np.random.default_rng(ar_rows * 977 + nar_rows)
    q = (rng.standard_normal((nar_rows, num_q_heads, head_dim)) * 0.5).astype(np.float32)
    nar_k = _bf16(rng.standard_normal((nar_rows, num_kv_heads, head_dim)))
    nar_v = _bf16(rng.standard_normal((nar_rows, num_kv_heads, head_dim)))
    ar_k = _bf16(rng.standard_normal((ar_rows, num_kv_heads, head_dim))) if ar_rows else np.zeros((0, num_kv_heads, head_dim), dtype=np.uint16)
    ar_v = _bf16(rng.standard_normal((ar_rows, num_kv_heads, head_dim))) if ar_rows else np.zeros((0, num_kv_heads, head_dim), dtype=np.uint16)
    scale = 1.0 / float(np.sqrt(head_dim))
    q_buf = _upload(q)
    nk_buf = _upload(nar_k)
    nv_buf = _upload(nar_v)
    ak_buf = _upload(ar_k)
    av_buf = _upload(ar_v)
    out_buf = _upload(np.zeros((nar_rows, num_q_heads, head_dim), dtype=np.float32))
    nar.nar_attention_wmma(
        q_buf.ptr, nk_buf.ptr, nv_buf.ptr, ak_buf.ptr, av_buf.ptr, out_buf.ptr,
        nar_rows, ar_rows, num_q_heads, num_kv_heads, head_dim, scale,
    )
    got = _download(out_buf, (nar_rows, num_q_heads, head_dim), np.float32)
    expected = _attention_reference(q, nar_k, nar_v, ar_k, ar_v, num_q_heads, num_kv_heads, scale)
    relative = float(np.linalg.norm(got - expected) / np.linalg.norm(expected))
    cosine = float(
        np.dot(got.ravel(), expected.ravel())
        / (np.linalg.norm(got) * np.linalg.norm(expected))
    )
    assert relative < 0.02, f"relative L2 {relative} exceeds the variant's envelope"
    assert cosine > 0.9995, f"cosine {cosine} falls outside the variant's envelope"


@requires_hip
def test_wmma_attention_rejects_a_geometry_it_does_not_implement():
    """The tensor-core path guards its own geometry so callers can fall back.

    It implements the production head geometry only (16 query heads over 8
    key/value heads at head_dim 128). Anything else must return a HIP error
    rather than compute something wrong, because the runtime dispatches on this
    kernel's availability.
    """
    from hipengine.core.hip import HipError
    from hipengine.kernels.hip_gfx1100.yue2 import nar

    nar_rows, ar_rows = 4, 4
    q = np.zeros((nar_rows, 16, 128), dtype=np.float32)
    k = np.zeros((nar_rows, 8, 128), dtype=np.uint16)
    v = np.zeros((nar_rows, 8, 128), dtype=np.uint16)
    out = np.zeros((nar_rows, 16, 128), dtype=np.float32)
    q_buf, nk_buf, nv_buf, out_buf = _upload(q), _upload(k), _upload(v), _upload(out)
    with pytest.raises(HipError):
        nar.nar_attention_wmma(
            q_buf.ptr, nk_buf.ptr, nv_buf.ptr, nk_buf.ptr, nv_buf.ptr, out_buf.ptr,
            nar_rows, ar_rows, 16, 4, 128, 0.1,
        )


@requires_hip
def test_attention_matches_the_parent_kernel_bit_for_bit():
    """Exact parent parity for the attention kernel's tiled reduction.

    ``tests/fixtures/yue2/operators/nar_attention_parent.npz`` holds the parent
    implementation's exact output on a multi-tile GQA shape (198 keys over tiles
    of 128, so the second tile is partial). The reduction was rewritten to compute
    each key's exponential once instead of once per lane; that rewrite must not
    move a single bit, and this is the contract that says so.
    """

    from hipengine.kernels.hip_gfx1100.yue2 import nar

    golden = np.load(FIXTURES / "operators/nar_attention_parent.npz")
    q = golden["q"]
    nar_k = golden["nar_k"]
    nar_v = golden["nar_v"]
    ar_k = golden["ar_k"]
    ar_v = golden["ar_v"]
    nar_rows = int(golden["nar_rows"])
    ar_rows = int(golden["ar_rows"])
    num_q_heads = int(golden["num_q_heads"])
    num_kv_heads = int(golden["num_kv_heads"])
    head_dim = int(golden["head_dim"])
    scale = float(golden["scale"])
    assert ar_rows + nar_rows > head_dim, "fixture must span more than one tile"

    q_buf = _upload(q)
    nk_buf = _upload(nar_k)
    nv_buf = _upload(nar_v)
    ak_buf = _upload(ar_k)
    av_buf = _upload(ar_v)
    out_buf = _upload(np.zeros((nar_rows, num_q_heads, head_dim), dtype=np.float32))
    nar.nar_attention_f32(
        q_buf.ptr, nk_buf.ptr, nv_buf.ptr, ak_buf.ptr, av_buf.ptr, out_buf.ptr,
        nar_rows, ar_rows, num_q_heads, num_kv_heads, head_dim, scale,
    )
    got = _download(out_buf, (nar_rows, num_q_heads, head_dim), np.float32)
    expected = golden["out"]
    assert np.array_equal(got.view(np.uint32), expected.view(np.uint32)), (
        "attention output is not bit-identical to the parent kernel: max abs diff "
        f"{np.abs(got - expected).max()}"
    )


@requires_hip
def test_attention_is_not_causal():
    """A later query row must see earlier NAR keys: NAR attention is bidirectional."""
    from hipengine.kernels.hip_gfx1100.yue2 import nar

    num_q_heads, num_kv_heads, head_dim = 2, 1, 8
    rng = np.random.default_rng(5)
    nar_rows = 4
    q = np.zeros((nar_rows, num_q_heads, head_dim), dtype=np.float32)
    nar_k = _bf16(np.zeros((nar_rows, num_kv_heads, head_dim)))
    nar_v = _bf16(rng.standard_normal((nar_rows, num_kv_heads, head_dim)))
    empty = np.zeros((0, num_kv_heads, head_dim), dtype=np.uint16)
    q_buf = _upload(q)
    nk_buf = _upload(nar_k)
    nv_buf = _upload(nar_v)
    ak_buf = _upload(empty)
    av_buf = _upload(empty)
    out_buf = _upload(np.zeros((nar_rows, num_q_heads, head_dim), dtype=np.float32))
    nar.nar_attention_f32(
        q_buf.ptr, nk_buf.ptr, nv_buf.ptr, ak_buf.ptr, av_buf.ptr, out_buf.ptr,
        nar_rows, 0, num_q_heads, num_kv_heads, head_dim, 1.0,
    )
    got = _download(out_buf, (nar_rows, num_q_heads, head_dim), np.float32)
    mean_value = widen(nar_v).mean(axis=0)[0]
    # Equal scores over all keys, so every row returns the same mean of all values.
    for row in range(nar_rows):
        assert np.allclose(got[row, 0], mean_value, rtol=1e-3, atol=1e-3)
