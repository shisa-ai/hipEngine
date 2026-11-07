"""GPU tier: multi-token Glimmer flash prefill matches the one-token kernel.

Production contract: the multi-token kernel shares staged K/V tiles across
eight adjacent query tokens and aligns tiles to absolute 16-key boundaries, so
SWA rows may associate the online softmax differently. Outputs must stay
within BF16 rounding of the one-token kernel (max abs error <= 1/64 of the
output scale, cosine >= 0.99999) for full and SWA layers, chunk offsets, and
token counts that are not multiples of the block width.
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

N_Q, N_KV, D = 32, 2, 128


def _bf16(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


@pytest.mark.parametrize(
    "tokens,key_end,window",
    [(37, 37, 0), (64, 64, 2048), (1024, 3000, 0), (1024, 3000, 2048), (13, 2061, 2048), (300, 300, 0)],
)
def test_multi_token_flash_prefill_matches_single_token(tokens, key_end, window) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, copy_host_array_to_device, free, malloc
    from hipengine.kernels.hip_gfx1100.attention.muse_glimmer_flash_prefill import (
        muse_glimmer_flash_prefill,
    )

    rt = get_hip_runtime()
    rng = np.random.default_rng(tokens * 7 + key_end + window)
    q = _bf16(rng.standard_normal(tokens * N_Q * D))
    k = _bf16(rng.standard_normal(key_end * N_KV * D))
    v = _bf16(rng.standard_normal(key_end * N_KV * D))
    bufs = []

    def up(a):
        b = malloc(a.nbytes, runtime=rt)
        copy_host_array_to_device(b, np.ascontiguousarray(a), runtime=rt)
        bufs.append(b)
        return b

    def read(b, n):
        out = np.empty(n, dtype=np.uint16)
        rt.device_synchronize()
        copy_device_to_host(out.ctypes.data, b, runtime=rt)
        return out

    try:
        dq, dk, dv = up(q), up(k), up(v)
        n_out = tokens * N_Q * D
        ref, got = up(np.zeros(n_out, np.uint16)), up(np.zeros(n_out, np.uint16))
        common = dict(n_q_heads=N_Q, n_kv_heads=N_KV, key_begin=0, key_end=key_end,
                      tokens=tokens, window=window, scale=D ** -0.5, runtime=rt)
        muse_glimmer_flash_prefill(dq.ptr, dk.ptr, dv.ptr, ref.ptr, multi_token=False, **common)
        muse_glimmer_flash_prefill(dq.ptr, dk.ptr, dv.ptr, got.ptr, multi_token=True, **common)
        a, b = _f32(read(ref, n_out)), _f32(read(got, n_out))
        assert np.all(np.isfinite(b))
        scale = np.abs(a).max()
        assert np.max(np.abs(a - b)) <= scale / 64
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert cos >= 0.99999
    finally:
        for buf in bufs:
            free(buf, runtime=rt)
