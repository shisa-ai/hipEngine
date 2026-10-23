"""K-only prompt preparation preserves the split query/key kernel's bytes."""
import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.hip_gfx1100.attention import qwen4_exp_qsa as kernels


def _hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime unavailable")


@pytest.mark.parametrize("rows,head_dim", [(1, 128), (17, 256), (64, 256)])
@pytest.mark.parametrize("mrope", [False, True])
def test_key_only_norm_rope_matches_split_key_bytes(rows, head_dim, mrope):
    runtime = get_hip_runtime()
    rng = np.random.default_rng(381 + rows)
    query_heads, kv_heads, rotary = 4, 2, 64
    buffers = []

    def upload(array):
        array = np.ascontiguousarray(array)
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), array.nbytes, runtime=runtime)
        return buffer

    try:
        query = upload(rng.normal(size=(rows, query_heads, 2 * head_dim)).astype(np.float32))
        key = upload(rng.normal(size=(rows, kv_heads, head_dim)).astype(np.float32))
        weight = upload(rng.uniform(.5, 1.5, head_dim).astype(np.float32))
        positions = np.arange(997, 997 + rows, dtype=np.int64)
        if mrope:
            positions = np.stack((positions, positions + 11, positions + 39))
        positions = upload(positions)
        query_out = upload(np.zeros((rows, query_heads, head_dim), dtype=np.float32))
        expected = upload(np.zeros((rows, kv_heads, head_dim), dtype=np.float32))
        actual = upload(np.full((rows, kv_heads, head_dim), np.nan, dtype=np.float32))
        split = (kernels.qwen4_exp_qsa_split_norm_mrope_rows_f32 if mrope
                 else kernels.qwen4_exp_qsa_split_norm_rope_rows_f32)
        norm = (kernels.qwen4_exp_qsa_norm_mrope_rows_f32 if mrope
                else kernels.qwen4_exp_qsa_norm_rope_rows_f32)
        split(query.ptr, key.ptr, weight.ptr, weight.ptr, positions.ptr,
              query_out.ptr, expected.ptr, 0, rows, query_heads, kv_heads,
              head_dim, rotary, 10000., 1e-6, runtime=runtime)
        norm(key.ptr, weight.ptr, positions.ptr, actual.ptr, rows, kv_heads,
             head_dim, rotary, 10000., 1e-6, runtime=runtime)
        left = np.empty(expected.nbytes, dtype=np.uint8)
        right = np.empty(actual.nbytes, dtype=np.uint8)
        copy_device_to_host(host_array_ptr(left), expected, left.nbytes, runtime=runtime)
        copy_device_to_host(host_array_ptr(right), actual, right.nbytes, runtime=runtime)
        np.testing.assert_array_equal(left, right)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
