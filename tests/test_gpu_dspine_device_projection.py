"""GPU RED: DSpine device-resident tap projection (task #31 P1 port).

dspine_concat_taps fuses the f32->bf16 cast + feature-dim concat of the target
tap hidden states (separate FP32 device buffers) into the BF16 fc input. Gate:
the kernel output matches the numpy cast+concat oracle bit-for-bit across tap
counts 1..5 and odd row/hidden geometries (catches layout/offset/rounding bugs).
The full fc+norm projection and cycle integration are gated by the real-weights
AR-exact mtpbench run (acceptance preservation confirms projection correctness).
HIP-guarded for no-ROCm CI.
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


def _f32_to_bf16(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _bf16_to_f32(b):
    return (np.asarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


@pytest.mark.parametrize("n_taps", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("rows,H", [(1, 64), (3, 64), (6, 128), (5, 96)])
def test_concat_taps_matches_numpy(n_taps, rows, H):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (malloc, free, copy_host_array_to_device,
                                       copy_device_to_host, host_array_ptr)
    from hipengine.kernels.hip_gfx1100.speculative.dspine import dspine_concat_taps

    rt = get_hip_runtime()
    rng = np.random.default_rng(n_taps * 100 + rows)
    chunks = [rng.standard_normal((rows, H)).astype(np.float32) for _ in range(n_taps)]
    owners = []
    ptrs = []
    out_b = malloc(rows * n_taps * H * 2, runtime=rt)
    owners.append(out_b)
    try:
        for c in chunks:
            b = malloc(c.nbytes, runtime=rt)
            copy_host_array_to_device(b, np.ascontiguousarray(c), runtime=rt)
            ptrs.append(b.ptr)
            owners.append(b)
        dspine_concat_taps(ptrs, out_b.ptr, rows, H, runtime=rt)
        rt.device_synchronize()
        got = np.empty(rows * n_taps * H, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(got), out_b, runtime=rt)
        got = _bf16_to_f32(got).reshape(rows, n_taps * H)
        expected = _bf16_to_f32(_f32_to_bf16(np.concatenate(chunks, axis=1)))
        assert np.array_equal(got, expected), \
            f"n_taps={n_taps} rows={rows} H={H} max_diff={np.max(np.abs(got - expected))}"
    finally:
        for b in owners:
            free(b, runtime=rt)
