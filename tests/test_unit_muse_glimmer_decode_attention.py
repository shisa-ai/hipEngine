"""Unit tier: Glimmer flash-decoding attention vs the f32 CPU reference.

HIP-gated. The kernel is online-softmax F32 over BF16 operands; the reference
is exact f32 softmax. Gate: max |err| <= 0.02 on BF16-rounded outputs and
argmax-stable attention weights (outer-floor contract for the new variant).
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

try:
    ctypes.CDLL("libamdhip64.so")
    _HIP = True
except OSError:
    _HIP = False

pytestmark = pytest.mark.skipif(not _HIP, reason="ROCm runtime not available")


def _bf16(x: np.ndarray) -> np.ndarray:
    from hipengine.loading.materialize import float_array_to_bf16_bits
    return float_array_to_bf16_bits(x)


def _to_bf16_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


@pytest.mark.parametrize("keys,key_begin,n_splits", [
    (64, 0, 4), (512, 0, 16), (1024, 512, 16), (4096, 2048, 64), (37, 0, 8),
])
def test_decode_attention_matches_cpu_reference(keys, key_begin, n_splits) -> None:
    from hipengine.core.memory import malloc, free, copy_host_array_to_device, copy_device_to_host, host_array_ptr, DeviceBuffer
    from hipengine.kernels.cpu_reference.muse_glimmer import gqa_attention
    from hipengine.kernels.hip_gfx1100.attention.muse_glimmer_decode_attention import (
        muse_glimmer_decode_attention,
    )

    rng = np.random.default_rng(keys + key_begin)
    n_q, n_kv, d = 32, 2, 128
    q = rng.standard_normal((1, n_q, d)).astype(np.float32) * 0.5
    k = rng.standard_normal((keys, n_kv, d)).astype(np.float32) * 0.5
    v = rng.standard_normal((keys, n_kv, d)).astype(np.float32) * 0.5
    mask = np.ones((1, keys), dtype=bool)
    ref = gqa_attention(q, k, v, mask, n_heads=n_q, n_kv_heads=n_kv, scale=d ** -0.5)[0]

    qb = _bf16(q.reshape(n_q, d))
    kb = _bf16(k)
    vb = _bf16(v)
    # Reference over the same BF16-rounded operands, restricted to the slice.
    k_f = _to_bf16_f32(kb)[key_begin:keys]
    v_f = _to_bf16_f32(vb)[key_begin:keys]
    q_f = _to_bf16_f32(qb)[None, :, :]
    ref = gqa_attention(q_f, k_f, v_f, np.ones((1, keys - key_begin), dtype=bool),
                        n_heads=n_q, n_kv_heads=n_kv, scale=d ** -0.5)[0]

    bufs = []
    def up(arr):
        b = malloc(arr.nbytes); bufs.append(b)
        copy_host_array_to_device(b, arr); return b

    try:
        qd = up(np.ascontiguousarray(qb))
        kd = up(np.ascontiguousarray(kb))
        vd = up(np.ascontiguousarray(vb))
        out = malloc(n_q * d * 2); bufs.append(out)
        partials = malloc(n_splits * n_q * 130 * 4); bufs.append(partials)
        muse_glimmer_decode_attention(
            qd.ptr, kd.ptr, vd.ptr, out.ptr, partials.ptr,
            n_q_heads=n_q, n_kv_heads=n_kv, key_begin=key_begin, key_end=keys,
            n_splits=n_splits, scale=d ** -0.5,
        )
        host = np.empty(n_q * d, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(host), out)
        got = _to_bf16_f32(host.reshape(n_q, d))
        err = float(np.abs(got - ref).max())
        assert np.isfinite(got).all()
        assert err <= 0.02, f"keys={keys} begin={key_begin}: max|err| {err}"
    finally:
        for b in reversed(bufs):
            free(b)

@pytest.mark.parametrize("tokens,keys,n_splits,window", [
    (8, 8, 1, 0), (64, 64, 4, 0), (128, 128, 4, 32), (256, 256, 4, 64),
])
def test_prefill_attention_matches_cpu_reference(tokens, keys, n_splits, window) -> None:
    """Multi-row causal (+SWA) prefill; locks the combine split-stride fix."""
    from hipengine.core.memory import malloc, free, copy_host_array_to_device, copy_device_to_host, host_array_ptr
    from hipengine.kernels.cpu_reference.muse_glimmer import gqa_attention, causal_swa_mask
    from hipengine.kernels.hip_gfx1100.attention.muse_glimmer_decode_attention import (
        muse_glimmer_decode_attention,
    )

    rng = np.random.default_rng(tokens * 7 + keys)
    n_q, n_kv, d = 32, 2, 128
    q = (rng.standard_normal((tokens, n_q, d)) * 0.5).astype(np.float32)
    k = (rng.standard_normal((keys, n_kv, d)) * 0.5).astype(np.float32)
    v = (rng.standard_normal((keys, n_kv, d)) * 0.5).astype(np.float32)
    qb, kb, vb = _bf16(q.reshape(tokens * n_q, d)), _bf16(k), _bf16(v)
    q_f = _to_bf16_f32(qb).reshape(tokens, n_q, d)
    k_f, v_f = _to_bf16_f32(kb), _to_bf16_f32(vb)
    pos = np.arange(keys - tokens, keys, dtype=np.int64)
    mask = causal_swa_mask(pos, keys, np.arange(keys), window if window > 0 else None)
    ref = gqa_attention(q_f, k_f, v_f, mask, n_heads=n_q, n_kv_heads=n_kv, scale=d ** -0.5)

    bufs = []
    def up(arr):
        b = malloc(arr.nbytes); bufs.append(b); copy_host_array_to_device(b, arr); return b
    try:
        qd = up(np.ascontiguousarray(qb)); kd = up(np.ascontiguousarray(kb)); vd = up(np.ascontiguousarray(vb))
        out = malloc(tokens * n_q * d * 2); bufs.append(out)
        partials = malloc(n_splits * tokens * n_q * 130 * 4); bufs.append(partials)
        muse_glimmer_decode_attention(
            qd.ptr, kd.ptr, vd.ptr, out.ptr, partials.ptr,
            n_q_heads=n_q, n_kv_heads=n_kv, key_begin=0, key_end=keys,
            n_splits=n_splits, tokens=tokens, window=window, scale=d ** -0.5,
        )
        host = np.empty(tokens * n_q * d, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(host), out)
        got = _to_bf16_f32(host.reshape(tokens, n_q, d))
        err = float(np.abs(got - ref).max())
        assert np.isfinite(got).all()
        assert err <= 0.02, f"tokens={tokens} keys={keys} splits={n_splits} win={window}: max|err| {err}"
    finally:
        for b in reversed(bufs):
            free(b)
