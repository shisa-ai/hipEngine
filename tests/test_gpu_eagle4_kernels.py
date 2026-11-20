"""GPU RED: EAGLE-4 RoPE-only + top-K retrieve == CPU oracle."""
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


def test_gpu_rope_matches_cpu():
    from hipengine.core.memory import malloc, free, copy_host_array_to_device, copy_device_to_host, host_array_ptr
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.cpu_reference.eagle4 import _rope_rotate
    from hipengine.kernels.hip_gfx1100.speculative.eagle4 import eagle4_rope

    rt = get_hip_runtime()
    rows, n_heads, head_dim, theta = 5, 4, 16, 1e7
    rng = np.random.default_rng(0)
    x = rng.standard_normal((rows, n_heads, head_dim)).astype(np.float32)
    positions = np.array([3, 7, 11, 20, 50], dtype=np.int32)
    expected = _rope_rotate(x, positions.astype(np.int64), theta, head_dim)
    xb = malloc(x.nbytes, runtime=rt); copy_host_array_to_device(xb, np.ascontiguousarray(x), runtime=rt)
    pb = malloc(positions.nbytes, runtime=rt); copy_host_array_to_device(pb, positions, runtime=rt)
    try:
        eagle4_rope(xb.ptr, pb.ptr, rows, n_heads, head_dim, theta, runtime=rt)
        rt.device_synchronize()
        got = np.empty(x.size, dtype=np.float32)
        copy_device_to_host(host_array_ptr(got), xb, runtime=rt)
        got = got.reshape(rows, n_heads, head_dim)
    finally:
        free(xb, runtime=rt); free(pb, runtime=rt)
    assert np.allclose(got, expected, atol=1e-4), f"max diff {np.max(np.abs(got-expected))}"


def test_gpu_topk_matches_argsort_set():
    from hipengine.core.memory import malloc, free, copy_host_array_to_device, copy_device_to_host, host_array_ptr
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.speculative.eagle4 import eagle4_topk

    rt = get_hip_runtime()
    rows, vocab, K = 3, 5000, 128
    rng = np.random.default_rng(1)
    scores = rng.standard_normal((rows, vocab)).astype(np.float32)
    expected = np.argsort(-scores, axis=-1)[:, :K]
    sb = malloc(scores.nbytes, runtime=rt); copy_host_array_to_device(sb, np.ascontiguousarray(scores), runtime=rt)
    ob = malloc(rows * K * 4, runtime=rt)
    try:
        eagle4_topk(sb.ptr, ob.ptr, rows, vocab, K, runtime=rt)
        rt.device_synchronize()
        got = np.empty(rows * K, dtype=np.int32)
        copy_device_to_host(host_array_ptr(got), ob, runtime=rt)
        got = got.reshape(rows, K)
    finally:
        free(sb, runtime=rt); free(ob, runtime=rt)
    for r in range(rows):
        assert set(got[r].tolist()) == set(expected[r].tolist()), f"row {r} top-K set mismatch"


@pytest.mark.parametrize("rows,vocab,K", [(1, 248320, 128), (2, 248320, 128), (1, 5000, 128), (1, 2049, 128), (1, 300, 128)])
def test_gpu_topk_two_stage_large_vocab(rows, vocab, K):
    """Two-stage parallel top-K must return the exact argsort top-K SET at the
    real draft-head vocab (248320, many chunks) and at chunk-boundary/small
    geometries. Guards the stage1/chunk + stage2/candidate selection semantics."""
    from hipengine.core.memory import malloc, free, copy_host_array_to_device, copy_device_to_host, host_array_ptr
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.speculative.eagle4 import eagle4_topk

    rt = get_hip_runtime()
    rng = np.random.default_rng(vocab + rows)
    scores = rng.standard_normal((rows, vocab)).astype(np.float32)
    expected = np.argsort(-scores, axis=-1)[:, :K]
    sb = malloc(scores.nbytes, runtime=rt); copy_host_array_to_device(sb, np.ascontiguousarray(scores), runtime=rt)
    ob = malloc(rows * K * 4, runtime=rt)
    try:
        eagle4_topk(sb.ptr, ob.ptr, rows, vocab, K, runtime=rt)
        rt.device_synchronize()
        got = np.empty(rows * K, dtype=np.int32)
        copy_device_to_host(host_array_ptr(got), ob, runtime=rt)
        got = got.reshape(rows, K)
    finally:
        free(sb, runtime=rt); free(ob, runtime=rt)
    for r in range(rows):
        assert set(got[r].tolist()) == set(expected[r].tolist()), \
            f"rows={rows} vocab={vocab} K={K} row {r} top-K set mismatch"