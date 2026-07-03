"""Correctness gate for mtp_dense_attn_f32 — multi-position causal GQA context
attention used by the NextN draft chain/catch-up.

Previously UNGATED: ``test_m4_gguf_mtp_draft.py`` validates the NextN draft only
at ``rows == 1`` (context-free). This gate exercises the multi-position context
attention path (the chain/catch-up scenario) against a NumPy causal-GQA
reference. Skips without HIP.

Kernel math (hipengine_mtp_dense_attn_f32_kernel): for query token t, head qh:
  kh = qh // (heads // kv_heads); position = positions[t]; context = context_counts[t]
  vcount = min(position, context - 1) + 1
  scores[v] = (key[v, kh] . query[t, qh]) * scale  for v in 0..vcount-1
  out[t, qh] = softmax(scores) @ value[0..vcount-1, kh]
"""
from __future__ import annotations

import ctypes
import os

import numpy as np
import pytest

os.environ.setdefault("HIPENGINE_GGUF_DECODE_REPACK", "1")
os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")


def _hip() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _cpu_ref(query, key, value, positions, context_counts, heads, kv_heads, scale):
    tokens = query.shape[0]
    vhd = value.shape[2]
    kv_group = heads // kv_heads
    out = np.zeros((tokens, heads, vhd), dtype=np.float32)
    for t in range(tokens):
        pos = int(positions[t])
        ctx = int(context_counts[t])
        vcount = min(pos, ctx - 1) + 1
        if vcount <= 0:
            continue
        for qh in range(heads):
            kh = qh // kv_group
            qr = query[t, qh]
            sc = np.array([key[v, kh] @ qr * scale for v in range(vcount)], dtype=np.float32)
            sc = sc - sc.max()
            e = np.exp(sc)
            p = e / e.sum()
            out[t, qh] = sum(p[v] * value[v, kh] for v in range(vcount))
    return out


@pytest.mark.skipif(not _hip(), reason="HIP runtime is not available")
def test_mtp_dense_attn_f32_multiposition_matches_cpu_reference() -> None:
    from hipengine.core.hip import get_hip_runtime, HipMemcpyKind
    from hipengine.core.memory import malloc, free, host_array_ptr
    from hipengine.kernels.hip_gfx1100.speculative.mtp_nextn import (
        mtp_dense_attn_f32,
        build_mtp_nextn,
    )

    rt = get_hip_runtime()
    lib = build_mtp_nextn(load=True)
    rng = np.random.default_rng(0)
    heads, kv_heads, qkd, vhd = 16, 2, 128, 128
    cache_tokens = 8
    scale = 1.0 / float(np.sqrt(qkd))
    tokens = 4
    query = (rng.standard_normal((tokens, heads, qkd)) * 0.5).astype(np.float32)
    key = (rng.standard_normal((cache_tokens, kv_heads, qkd)) * 0.5).astype(np.float32)
    value = (rng.standard_normal((cache_tokens, kv_heads, vhd)) * 0.5).astype(np.float32)
    positions = np.array([4, 5, 6, 7], dtype=np.int64)
    context_counts = np.array([5, 6, 7, 8], dtype=np.int64)

    ref = _cpu_ref(query, key, value, positions, context_counts, heads, kv_heads, scale)

    bufs = []

    def up(a):
        a = np.ascontiguousarray(a)
        b = malloc(a.nbytes, runtime=rt)
        rt.memcpy(b.ptr, host_array_ptr(a), a.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
        bufs.append(b)
        return b

    q = up(query)
    k = up(key)
    v = up(value)
    posb = up(positions)
    ctxb = up(context_counts)
    outb = malloc(tokens * heads * vhd * 4, runtime=rt)
    bufs.append(outb)
    try:
        mtp_dense_attn_f32(
            q.ptr, k.ptr, v.ptr, posb.ptr, ctxb.ptr, outb.ptr,
            tokens, heads, kv_heads, qkd, vhd, cache_tokens, scale,
            library=lib, runtime=rt,
        )
        rt.device_synchronize()
        gpu = np.empty((tokens, heads, vhd), dtype=np.float32)
        rt.memcpy(host_array_ptr(gpu), outb.ptr, gpu.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
    finally:
        for b in bufs:
            free(b, runtime=rt)

    max_abs = float(np.max(np.abs(ref - gpu)))
    assert max_abs < 1e-3, f"mtp_dense_attn_f32 multi-position mismatch: max_abs={max_abs}"
