"""The direct F16 hyper-connection launches are bit-identical to the routes they replace."""
import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as dw
from tests.test_gpu_gguf_k_gemv import make_q8_0_weight


def _hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime unavailable")


def _run(fn, x, w, rows, k, n, runtime, library):
    out = np.full(rows * n, np.float32(-7.25), np.float32)
    xd, wd, od = (malloc(a.nbytes, runtime=runtime) for a in (x, w, out))
    try:
        copy_host_to_device(xd, host_array_ptr(x), runtime=runtime)
        copy_host_to_device(wd, host_array_ptr(w), runtime=runtime)
        copy_host_to_device(od, host_array_ptr(out), runtime=runtime)
        fn(xd.ptr, wd.ptr, od.ptr, rows, k, n, library=library, runtime=runtime)
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), od, runtime=runtime)
        return out
    finally:
        for b in (xd, wd, od):
            free(b, runtime=runtime)


@pytest.mark.parametrize("rows,k,n", [(512, 1280, 320), (300, 640, 320), (1024, 2560, 320)])
def test_down_tile_128x128_matches_the_padded_256_row_tile(rows, k, n):
    runtime = get_hip_runtime()
    library = dw.build_gguf_q8_0_dense_wide(load=True)
    rng = np.random.default_rng(4411)
    x = rng.normal(0, 0.1, (rows, k)).astype(np.float16)
    w = make_q8_0_weight(n, k)
    padded = _run(dw._WRAPPERS["dense_wide256_padded_f16in_f32_f32_out"],
                  x, w, rows, k, n, runtime, library)
    direct = _run(dw._WRAPPERS["dense_wide128x128_f16in_f32_f32_out"],
                  x, w, rows, k, n, runtime, library)
    np.testing.assert_array_equal(direct.view(np.uint32), padded.view(np.uint32))


@pytest.mark.parametrize("rows,k,n", [(512, 320, 2560), (300, 320, 1280)])
def test_up_with_a_converted_activation_matches_the_f32_input_tile(rows, k, n):
    runtime = get_hip_runtime()
    library = dw.build_gguf_q8_0_dense_wide(load=True)
    rng = np.random.default_rng(4412)
    x = rng.normal(0, 0.1, (rows, k)).astype(np.float32)
    w = make_q8_0_weight(n, k)
    parent = _run(dw._WRAPPERS["dense_wide256_f32_f32_out"],
                  x, w, rows, k, n, runtime, library)
    x16 = np.zeros(rows * k, np.float16)
    xd, hd = malloc(x.nbytes, runtime=runtime), malloc(x16.nbytes, runtime=runtime)
    try:
        copy_host_to_device(xd, host_array_ptr(x), runtime=runtime)
        dw.f32_to_f16(xd.ptr, hd.ptr, rows * k, library=library, runtime=runtime)
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(x16), hd, runtime=runtime)
    finally:
        free(xd, runtime=runtime)
        free(hd, runtime=runtime)
    direct = _run(dw._WRAPPERS["dense_wide256_f16in_f32_f32_out"],
                  x16.reshape(rows, k), w, rows, k, n, runtime, library)
    np.testing.assert_array_equal(direct.view(np.uint32), parent.view(np.uint32))


@pytest.mark.parametrize("rows,k,n", [(1024, 1280, 640), (300, 640, 200)])
def test_the_build_without_cu_mode_gives_the_same_bits(rows, k, n):
    runtime = get_hip_runtime()
    rng = np.random.default_rng(4413)
    x = rng.normal(0, 0.1, (rows, k)).astype(np.float16)
    w = make_q8_0_weight(n, k)
    cu = _run(dw._WRAPPERS["dense_wide128x128_f16in_f32_f32_out"],
              x, w, rows, k, n, runtime, dw.build_gguf_q8_0_dense_wide(load=True))
    wgp = _run(dw._WRAPPERS[dw.LONG_K_F16_VARIANT], x, w, rows, k, n, runtime, None)
    parent = _run(dw._WRAPPERS["dense_wide256_f16in_f32_f32_out"],
                  x, w, rows, k, n, runtime, dw.build_gguf_q8_0_dense_wide(load=True))
    np.testing.assert_array_equal(wgp.view(np.uint32), cu.view(np.uint32))
    np.testing.assert_array_equal(wgp.view(np.uint32), parent.view(np.uint32))
