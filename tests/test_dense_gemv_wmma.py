"""GPU correctness tests for verifier-side dense GEMV WMMA variants."""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest


def _has_gfx1100() -> bool:
    try:
        from hipengine.core.hip import get_hip_runtime
    except Exception:
        return False
    try:
        get_hip_runtime()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gfx1100(), reason="gfx1100 HIP runtime not available")


@pytest.fixture(scope="module")
def _dense_lib():
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.linear import build_dense_gemv

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = pathlib.Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment("gfx1100"):
        return build_dense_gemv(load=True, compiler_version=compiler_version)


@pytest.fixture(scope="module")
def _runtime():
    from hipengine.core.hip import get_hip_runtime

    return get_hip_runtime()


def _upload(runtime, bufs, array):
    from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc

    arr = np.ascontiguousarray(array)
    buf = malloc(arr.nbytes, runtime=runtime)
    bufs.append(buf)
    copy_host_to_device(buf, host_array_ptr(arr), runtime=runtime)
    return buf


def _download(runtime, buf, shape, dtype):
    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    arr = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(arr), buf, runtime=runtime)
    return arr


def _free_all(runtime, bufs) -> None:
    from hipengine.core.memory import free

    for buf in reversed(bufs):
        free(buf, runtime=runtime)


def _to_bf16_bits(x_f32: np.ndarray) -> np.ndarray:
    bits = x_f32.view(np.uint32)
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _from_bf16_bits(x_u16: np.ndarray) -> np.ndarray:
    return (x_u16.astype(np.uint32) << 16).view(np.float32)


def test_dense_gemv_out_fp16_wmma_matches_naive(_dense_lib, _runtime):
    from hipengine.kernels.hip_gfx1100.linear import dense_gemv_out_fp16, dense_gemv_out_fp16_wmma

    rng = np.random.default_rng(0xD3A5E)
    rows, in_features, out_features = 5, 512, 48
    x = (rng.standard_normal((rows, in_features)).astype(np.float32) * 0.02).astype(np.float16)
    w = (rng.standard_normal((out_features, in_features)).astype(np.float32) * 0.02).astype(np.float16)
    bufs = []
    try:
        x_dev = _upload(_runtime, bufs, x)
        w_dev = _upload(_runtime, bufs, w)
        out_naive = _upload(_runtime, bufs, np.zeros((rows, out_features), np.float16))
        out_wmma = _upload(_runtime, bufs, np.zeros((rows, out_features), np.float16))
        dense_gemv_out_fp16(
            x_dev.ptr,
            w_dev.ptr,
            out_naive.ptr,
            rows,
            in_features,
            out_features,
            library=_dense_lib,
            runtime=_runtime,
        )
        dense_gemv_out_fp16_wmma(
            x_dev.ptr,
            w_dev.ptr,
            out_wmma.ptr,
            rows,
            in_features,
            out_features,
            library=_dense_lib,
            runtime=_runtime,
        )
        _runtime.device_synchronize()
        naive = _download(_runtime, out_naive, (rows, out_features), np.float16).astype(np.float32)
        wmma = _download(_runtime, out_wmma, (rows, out_features), np.float16).astype(np.float32)
        np.testing.assert_allclose(wmma, naive, atol=5e-4, rtol=5e-3)
    finally:
        _free_all(_runtime, bufs)


def test_dense_dual_gemv_out_fp16_wmma_matches_naive(_dense_lib, _runtime):
    from hipengine.kernels.hip_gfx1100.linear import dense_dual_gemv_out_fp16, dense_dual_gemv_out_fp16_wmma

    rng = np.random.default_rng(0xD3A5F)
    rows, in_features, out_a, out_b = 5, 512, 48, 32
    x = (rng.standard_normal((rows, in_features)).astype(np.float32) * 0.02).astype(np.float16)
    w_a = (rng.standard_normal((out_a, in_features)).astype(np.float32) * 0.02).astype(np.float16)
    w_b = (rng.standard_normal((out_b, in_features)).astype(np.float32) * 0.02).astype(np.float16)
    bufs = []
    try:
        x_dev = _upload(_runtime, bufs, x)
        wa_dev = _upload(_runtime, bufs, w_a)
        wb_dev = _upload(_runtime, bufs, w_b)
        out_naive = _upload(_runtime, bufs, np.zeros((rows, out_a + out_b), np.float16))
        out_wmma = _upload(_runtime, bufs, np.zeros((rows, out_a + out_b), np.float16))
        dense_dual_gemv_out_fp16(
            x_dev.ptr,
            wa_dev.ptr,
            wb_dev.ptr,
            out_naive.ptr,
            rows,
            in_features,
            out_a,
            out_b,
            library=_dense_lib,
            runtime=_runtime,
        )
        dense_dual_gemv_out_fp16_wmma(
            x_dev.ptr,
            wa_dev.ptr,
            wb_dev.ptr,
            out_wmma.ptr,
            rows,
            in_features,
            out_a,
            out_b,
            library=_dense_lib,
            runtime=_runtime,
        )
        _runtime.device_synchronize()
        naive = _download(_runtime, out_naive, (rows, out_a + out_b), np.float16).astype(np.float32)
        wmma = _download(_runtime, out_wmma, (rows, out_a + out_b), np.float16).astype(np.float32)
        np.testing.assert_allclose(wmma, naive, atol=5e-4, rtol=5e-3)
    finally:
        _free_all(_runtime, bufs)


def test_dense_dual_gemv_separate_out_fp16_matches_two_singles(_dense_lib, _runtime):
    from hipengine.kernels.hip_gfx1100.linear import dense_dual_gemv_separate_out_fp16, dense_gemv_out_fp16

    rng = np.random.default_rng(0xD3A61)
    rows, in_features, out_a, out_b = 5, 512, 48, 32
    x = (rng.standard_normal((rows, in_features)).astype(np.float32) * 0.02).astype(np.float16)
    w_a = (rng.standard_normal((out_a, in_features)).astype(np.float32) * 0.02).astype(np.float16)
    w_b = (rng.standard_normal((out_b, in_features)).astype(np.float32) * 0.02).astype(np.float16)
    bufs = []
    try:
        x_dev = _upload(_runtime, bufs, x)
        wa_dev = _upload(_runtime, bufs, w_a)
        wb_dev = _upload(_runtime, bufs, w_b)
        out_a_single = _upload(_runtime, bufs, np.zeros((rows, out_a), np.float16))
        out_b_single = _upload(_runtime, bufs, np.zeros((rows, out_b), np.float16))
        out_a_dual = _upload(_runtime, bufs, np.zeros((rows, out_a), np.float16))
        out_b_dual = _upload(_runtime, bufs, np.zeros((rows, out_b), np.float16))
        dense_gemv_out_fp16(
            x_dev.ptr,
            wa_dev.ptr,
            out_a_single.ptr,
            rows,
            in_features,
            out_a,
            library=_dense_lib,
            runtime=_runtime,
        )
        dense_gemv_out_fp16(
            x_dev.ptr,
            wb_dev.ptr,
            out_b_single.ptr,
            rows,
            in_features,
            out_b,
            library=_dense_lib,
            runtime=_runtime,
        )
        dense_dual_gemv_separate_out_fp16(
            x_dev.ptr,
            wa_dev.ptr,
            wb_dev.ptr,
            out_a_dual.ptr,
            out_b_dual.ptr,
            rows,
            in_features,
            out_a,
            out_b,
            library=_dense_lib,
            runtime=_runtime,
        )
        _runtime.device_synchronize()
        a_single = _download(_runtime, out_a_single, (rows, out_a), np.float16)
        b_single = _download(_runtime, out_b_single, (rows, out_b), np.float16)
        a_dual = _download(_runtime, out_a_dual, (rows, out_a), np.float16)
        b_dual = _download(_runtime, out_b_dual, (rows, out_b), np.float16)
        np.testing.assert_array_equal(a_dual, a_single)
        np.testing.assert_array_equal(b_dual, b_single)
    finally:
        _free_all(_runtime, bufs)


def test_dense_gemv_out_bf16_wmma_matches_naive(_dense_lib, _runtime):
    from hipengine.kernels.hip_gfx1100.linear import dense_gemv_out_bf16, dense_gemv_out_bf16_wmma

    rng = np.random.default_rng(0xD3A60)
    rows, in_features, out_features = 5, 512, 48
    x = _to_bf16_bits(rng.standard_normal((rows, in_features)).astype(np.float32) * 0.02)
    w = _to_bf16_bits(rng.standard_normal((out_features, in_features)).astype(np.float32) * 0.02)
    bufs = []
    try:
        x_dev = _upload(_runtime, bufs, x)
        w_dev = _upload(_runtime, bufs, w)
        out_naive = _upload(_runtime, bufs, np.zeros((rows, out_features), np.uint16))
        out_wmma = _upload(_runtime, bufs, np.zeros((rows, out_features), np.uint16))
        dense_gemv_out_bf16(
            x_dev.ptr,
            w_dev.ptr,
            out_naive.ptr,
            rows,
            in_features,
            out_features,
            library=_dense_lib,
            runtime=_runtime,
        )
        dense_gemv_out_bf16_wmma(
            x_dev.ptr,
            w_dev.ptr,
            out_wmma.ptr,
            rows,
            in_features,
            out_features,
            library=_dense_lib,
            runtime=_runtime,
        )
        _runtime.device_synchronize()
        naive = _from_bf16_bits(_download(_runtime, out_naive, (rows, out_features), np.uint16))
        wmma = _from_bf16_bits(_download(_runtime, out_wmma, (rows, out_features), np.uint16))
        np.testing.assert_allclose(wmma, naive, atol=5e-4, rtol=5e-3)
    finally:
        _free_all(_runtime, bufs)
