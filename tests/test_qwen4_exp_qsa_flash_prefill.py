"""Correctness test for the QSA flash prefill kernel vs a NumPy reference."""

from __future__ import annotations

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa_flash import (
    build_qwen4_exp_qsa_flash,
    qwen4_exp_qsa_flash_prefill,
)
from tests.test_qwen4_exp_gdn_hip import _hip_available

_TOLERANCE = 5e-3


def _to_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,context_len,block_table",
    [
        (16, 32, np.array([0], dtype=np.int32)),
        (48, 700, np.array([2, 0, 1], dtype=np.int32)),
        (130, 300, np.array([0, 1], dtype=np.int32)),
    ],
)
def test_qsa_flash_prefill_matches_numpy_reference(
    rows: int, context_len: int, block_table: np.ndarray
) -> None:
    from hipengine.core.hip import get_hip_runtime

    rng = np.random.default_rng(rows * 31 + context_len)
    q_heads, kv_heads, head_dim, block_size = 24, 2, 256, 256
    kv_group = q_heads // kv_heads
    scale = head_dim ** -0.5
    start = context_len - rows

    kv_elems = int(block_table.max() + 1) * block_size * kv_heads * head_dim
    k_cache = _to_bf16(rng.standard_normal(kv_elems) * 0.3)
    v_cache = _to_bf16(rng.standard_normal(kv_elems) * 0.3)
    query = (rng.standard_normal((rows, q_heads, head_dim)) * 0.4).astype(np.float32)
    positions = np.arange(start, start + rows, dtype=np.int64)

    def cache_view(cache: np.ndarray) -> np.ndarray:
        pages = cache.reshape(-1, block_size, kv_heads, head_dim)
        gathered = np.zeros((context_len, kv_heads, head_dim), dtype=np.float32)
        for token in range(context_len):
            gathered[token] = _bf16_to_f32(
                pages[block_table[token // block_size], token % block_size]
            ).reshape(kv_heads, head_dim)
        return gathered

    k_full = cache_view(k_cache)
    v_full = cache_view(v_cache)
    ref = np.zeros((rows, q_heads, head_dim), dtype=np.float64)
    for row in range(rows):
        pos = start + row
        for h in range(q_heads):
            kv = h // kv_group
            q = query[row, h].astype(np.float64)
            s = (k_full[: pos + 1, kv, :] @ q) * scale
            w = np.exp(s - s.max())
            w /= w.sum()
            ref[row, h] = w @ v_full[: pos + 1, kv, :]

    runtime = get_hip_runtime()
    build_qwen4_exp_qsa_flash(load=True)
    allocs = []

    def dev(arr):
        d = malloc(arr.nbytes, runtime=runtime)
        allocs.append(d)
        copy_host_to_device(d, host_array_ptr(np.ascontiguousarray(arr)), runtime=runtime)
        return d

    try:
        q_d = dev(query)
        bt_d = dev(block_table)
        kc = dev(k_cache)
        vc = dev(v_cache)
        pos_d = dev(positions)
        ks = malloc(context_len * kv_heads * head_dim * 2, runtime=runtime)
        allocs.append(ks)
        vs = malloc(context_len * kv_heads * head_dim * 2, runtime=runtime)
        allocs.append(vs)
        out_d = malloc(rows * q_heads * head_dim * 2, runtime=runtime)
        allocs.append(out_d)
        qwen4_exp_qsa_flash_prefill(
            q_d.ptr, kc.ptr, vc.ptr, bt_d.ptr, pos_d.ptr, ks.ptr, vs.ptr, out_d.ptr,
            rows, q_heads, kv_heads, head_dim, block_size, int(block_table.size),
            context_len, scale, runtime=runtime,
        )
        runtime.device_synchronize()
        out_h = np.empty(rows * q_heads * head_dim, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(out_h), out_d, out_h.nbytes, runtime=runtime)
    finally:
        for allocation in reversed(allocs):
            free(allocation, runtime=runtime)

    got = _bf16_to_f32(out_h).reshape(rows, q_heads, head_dim).astype(np.float64)
    err = np.abs(got - ref)
    tolerance = _TOLERANCE * np.maximum(np.abs(ref).max(), 1.0)
    assert err.max() < tolerance, f"max err {err.max()} vs tol {tolerance}"
