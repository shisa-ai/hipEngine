from __future__ import annotations

import ctypes

import numpy as np
import pytest


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("tokens", [1, 2, 4])
def test_paro_rotate1_f32_to_fp16_matches_cast_then_rotate(tokens: int) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.convert.cast import build_cast, f32_to_fp16
    from hipengine.kernels.hip_gfx1100.rotary.paro_rotate import (
        build_paro_rotate,
        paro_rotate1_f32_to_fp16,
        paro_rotate1_fp16,
    )

    runtime = get_hip_runtime()
    build_cast(load=True)
    build_paro_rotate(load=True)

    rng = np.random.default_rng(17 + tokens)
    hidden = 256
    group_size = 128
    krot = 2
    half = group_size // 2
    x = np.ascontiguousarray((rng.standard_normal((tokens, hidden)) * 0.25).astype(np.float32))
    pairs = np.zeros((krot, hidden), dtype=np.int16)
    for r in range(krot):
        for group in range(hidden // group_size):
            base = group * group_size
            for lane in range(half):
                pairs[r, base + 2 * lane] = 2 * lane
                pairs[r, base + 2 * lane + 1] = 2 * lane + 1
    theta = np.ascontiguousarray(rng.uniform(-0.75, 0.75, (krot, hidden // 2)).astype(np.float16))
    scales = np.ascontiguousarray(rng.uniform(0.5, 1.5, hidden).astype(np.float16))

    buffers = []

    def upload(array: np.ndarray):
        array = np.ascontiguousarray(array)
        buf = malloc(array.nbytes)
        copy_host_to_device(buf, host_array_ptr(array), array.nbytes)
        buffers.append(buf)
        return buf

    try:
        x_b = upload(x)
        pairs_b = upload(pairs)
        theta_b = upload(theta)
        scales_b = upload(scales)
        cast_b = malloc(tokens * hidden * np.dtype(np.float16).itemsize)
        ref_b = malloc(tokens * hidden * np.dtype(np.float16).itemsize)
        fused_b = malloc(tokens * hidden * np.dtype(np.float16).itemsize)
        buffers.extend([cast_b, ref_b, fused_b])

        f32_to_fp16(x_b.ptr, cast_b.ptr, tokens * hidden)
        paro_rotate1_fp16(
            cast_b.ptr,
            ref_b.ptr,
            pairs_b.ptr,
            theta_b.ptr,
            scales_b.ptr,
            tokens,
            hidden,
            group_size,
            krot,
        )
        paro_rotate1_f32_to_fp16(
            x_b.ptr,
            fused_b.ptr,
            pairs_b.ptr,
            theta_b.ptr,
            scales_b.ptr,
            tokens,
            hidden,
            group_size,
            krot,
        )
        runtime.device_synchronize()

        ref = np.empty((tokens, hidden), dtype=np.float16)
        fused = np.empty((tokens, hidden), dtype=np.float16)
        copy_device_to_host(host_array_ptr(ref), ref_b, ref.nbytes)
        copy_device_to_host(host_array_ptr(fused), fused_b, fused.nbytes)
    finally:
        for buf in buffers:
            free(buf)

    np.testing.assert_array_equal(fused.view(np.uint16), ref.view(np.uint16))
