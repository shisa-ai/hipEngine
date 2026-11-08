"""GPU tier: the DFlash non-causal WMMA draft window attention matches the
non-causal scalar reference (cosine + argmax), across window boundaries.

The drafter attends its block rows over a sliding window of context plus the
whole block bidirectionally (mask key > position - window, no causal upper
bound). The WMMA block-flash is a production route (online softmax), so it is
compared to the scalar Gemma4 block kernel run with window=0 and an explicit
non-causal keep mask (the true reference), not bit-exact. Requires ROCm.
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
    return float_array_to_bf16_bits(np.ascontiguousarray(x, dtype=np.float32))


def _to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


@pytest.mark.parametrize("tokens,context,window", [
    (5, 4096, 2048),     # window filled (long-context regime)
    (5, 100, 2048),      # context shorter than window
    (8, 3000, 2048),     # larger block, window filled
    (2, 2048, 2048),     # minimal block at the window edge
    (5, 4096, 0),        # no window (attend whole prefix & block)
])
def test_draft_flash_matches_noncausal_scalar(tokens, context, window):
    from hipengine.core.memory import malloc, free, copy_host_array_to_device, copy_device_to_host
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import gemma4_attention_prefill_bf16
    from hipengine.kernels.hip_gfx1100.attention.muse_glimmer_draft_attention import (
        muse_glimmer_draft_flash_attention,
    )

    rt = get_hip_runtime()
    n_q, n_kv, d = 32, 8, 128
    position_offset = context
    keys = position_offset + tokens
    key_start = max(0, position_offset - window + 1) if window > 0 else 0
    span = keys - key_start
    row_offset = position_offset - key_start
    rng = np.random.default_rng(0)
    q = _bf16(rng.standard_normal(tokens * n_q * d) * 0.3)
    k = _bf16(rng.standard_normal(span * n_kv * d) * 0.3)
    v = _bf16(rng.standard_normal(span * n_kv * d) * 0.3)
    qpos = np.arange(position_offset, keys)[:, None]
    kpos = np.arange(key_start, keys)[None, :]
    mask = np.ascontiguousarray((kpos > qpos - window).astype(np.uint8) if window > 0
                                else np.ones((tokens, span), dtype=np.uint8))
    scale = d ** -0.5

    qb = malloc(q.nbytes, runtime=rt); kb = malloc(k.nbytes, runtime=rt)
    vb = malloc(v.nbytes, runtime=rt); mb = malloc(mask.nbytes, runtime=rt)
    ref_b = malloc(tokens * n_q * d * 2, runtime=rt); got_b = malloc(tokens * n_q * d * 2, runtime=rt)
    try:
        copy_host_array_to_device(qb, q, runtime=rt)
        copy_host_array_to_device(kb, k, runtime=rt)
        copy_host_array_to_device(vb, v, runtime=rt)
        copy_host_array_to_device(mb, mask, runtime=rt)
        # Reference: scalar block kernel, window=0 (no causal trim) + explicit mask.
        gemma4_attention_prefill_bf16(
            qb.ptr, kb.ptr, vb.ptr, mb.ptr, ref_b.ptr, tokens=tokens,
            num_heads=n_q, num_kv_heads=n_kv, head_dim=d, scale=scale, keys=span,
            window=0, row_offset=row_offset, runtime=rt)
        muse_glimmer_draft_flash_attention(
            qb.ptr, kb.ptr, vb.ptr, got_b.ptr, n_q_heads=n_q, n_kv_heads=n_kv,
            keys=span, tokens=tokens, window=window, scale=scale, runtime=rt)
        rt.device_synchronize()
        ref = np.empty(tokens * n_q * d, dtype=np.uint16)
        got = np.empty_like(ref)
        copy_device_to_host(ref.ctypes.data, ref_b, runtime=rt)
        copy_device_to_host(got.ctypes.data, got_b, runtime=rt)
    finally:
        for b in (qb, kb, vb, mb, ref_b, got_b):
            free(b, runtime=rt)

    rf = _to_f32(ref).reshape(tokens, n_q, d)
    gf = _to_f32(got).reshape(tokens, n_q, d)
    assert np.isfinite(gf).all()
    cos = float(np.sum(rf * gf) / (np.linalg.norm(rf) * np.linalg.norm(gf)))
    assert cos >= 0.999, f"cosine {cos} < 0.999"
    # Argmax over the head dim must agree (the draft proposes by argmax).
    assert np.array_equal(np.argmax(rf, axis=-1), np.argmax(gf, axis=-1))