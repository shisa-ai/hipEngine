import ctypes
from contextlib import closing
from functools import partial

import numpy as np
import pytest

from hipengine.runtime.inkling2_mini_reference import _mix_residual_rows, _rms


def _device():
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.backends import hip_target_arch_for_backend, resolve_backend

    backend = resolve_backend("auto", execution_device=True)
    if backend not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip("test requires gfx1100 or gfx1151")
    return get_hip_runtime(), hip_target_arch_for_backend(backend)


@pytest.mark.parametrize("rows,width", [(1, 2304), (3, 2304), (64, 2304), (7, 384)])
def test_row_kernels_match_the_host_normalization(rows, width):
    runtime, arch = _device()
    from hipengine.core.hip import HipMemcpyKind
    from hipengine.core.memory import free, malloc
    from hipengine.kernels.hip_gfx1100.norm import inkling2_residual as res

    rng = np.random.default_rng(31)
    eps = 1e-6
    x = (rng.standard_normal((rows, width)) * rng.uniform(0.3, 2.0, (rows, 1))).astype(np.float32)
    other = rng.standard_normal((rows, width)).astype(np.float32)
    weight = rng.uniform(0.5, 1.5, width).astype(np.float32)
    buffers = {}

    def upload(name, array):
        buffer = malloc(array.nbytes, runtime=runtime)
        runtime.memcpy(buffer.ptr, array.ctypes.data, array.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
        buffers[name] = buffer
        return buffer.ptr

    x_ptr, other_ptr, weight_ptr = upload("x", x), upload("other", other), upload("weight", weight)
    copy_ptr = upload("copy", x)
    out = malloc(x.nbytes, runtime=runtime)
    try:
        def run(launch):
            launch()
            runtime.stream_synchronize(0)
            actual = np.empty_like(x)
            runtime.memcpy(actual.ctypes.data, out.ptr, actual.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
            return actual

        expected = x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
        np.testing.assert_allclose(run(lambda: res.launch_row_rms(x_ptr, out.ptr, rows, width, eps, target_arch=arch)),
                                   expected, rtol=3e-6, atol=2e-6)
        # In-place normalization must produce the same values, on its own copy.
        np.testing.assert_allclose(
            run(lambda: res.launch_row_rms(copy_ptr, copy_ptr, rows, width, eps, target_arch=arch)), expected,
            rtol=3e-6, atol=2e-6)
        np.testing.assert_allclose(
            run(lambda: res.launch_row_norm_mul(x_ptr, weight_ptr, out.ptr, rows, width, eps, target_arch=arch)),
            expected * weight, rtol=3e-6, atol=2e-6)
        np.testing.assert_allclose(
            run(lambda: res.launch_row_add(x_ptr, other_ptr, out.ptr, rows, width, target_arch=arch)), x + other,
            rtol=1e-7, atol=1e-7)
    finally:
        for buffer in (*buffers.values(), out):
            free(buffer, runtime=runtime)


@pytest.mark.parametrize("rows,blocks,queries,width", [(1, 18, 1, 2304), (64, 18, 3, 2304), (64, 18, 4, 2304),
                                                        (3, 2, 3, 2304), (2, 9, 1, 384)])
def test_residual_read_matches_the_host_mix(rows, blocks, queries, width):
    runtime, arch = _device()
    from hipengine.core.hip import HipMemcpyKind
    from hipengine.core.memory import free, malloc
    from hipengine.kernels.hip_gfx1100.norm import inkling2_residual as res

    rng = np.random.default_rng(77)
    query = rng.standard_normal((queries, width)).astype(np.float32)
    values = rng.standard_normal((blocks, rows, width)).astype(np.float32)
    source = malloc(query.nbytes, runtime=runtime)
    stack = malloc(values.nbytes, runtime=runtime)
    out = malloc(rows * queries * width * 4, runtime=runtime)
    runtime.memcpy(source.ptr, query.ctypes.data, query.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
    runtime.memcpy(stack.ptr, values.ctypes.data, values.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
    try:
        res.launch_residual_read(source.ptr, stack.ptr, out.ptr, rows, blocks, queries, width, target_arch=arch)
        runtime.stream_synchronize(0)
        actual = np.empty((queries, rows, width), np.float32)
        runtime.memcpy(actual.ctypes.data, out.ptr, actual.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
        expected = np.moveaxis(_mix_residual_rows(query, values.transpose(1, 0, 2)), -2, 0)
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-6)
        # The same stack and query must reproduce bit for bit.
        res.launch_residual_read(source.ptr, stack.ptr, out.ptr, rows, blocks, queries, width, target_arch=arch)
        runtime.stream_synchronize(0)
        repeat = np.empty_like(actual)
        runtime.memcpy(repeat.ctypes.data, out.ptr, repeat.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
        np.testing.assert_array_equal(repeat.view(np.uint32), actual.view(np.uint32))
    finally:
        for buffer in (source, stack, out):
            free(buffer, runtime=runtime)
