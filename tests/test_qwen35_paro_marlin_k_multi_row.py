"""GPU bit-exactness test for the M15.2 multi-row Marlin-K GEMV.

The multi-row Marlin-K kernel weight-amortizes a verifier projection across the
``B+1`` rows while preserving the single-row kernel's per-row k-order and
reduction.  Each row must therefore be byte-identical to calling the single-row
``gemv_paro_marlin_k_fma_fp16`` once per row, which is what makes it exact vs
AR's rows==1 Marlin-K output for every prompt.
"""

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
def _marlin_lib():
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.quant.paro_marlin_k import build_paro_marlin_k

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = pathlib.Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment("gfx1100"):
        return build_paro_marlin_k(load=True, compiler_version=compiler_version)


@pytest.fixture(scope="module")
def _runtime():
    from hipengine.core.hip import get_hip_runtime

    return get_hip_runtime()


def _upload(runtime, bufs, array):
    from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc

    arr = np.ascontiguousarray(array)
    buf = malloc(max(arr.nbytes, 4), runtime=runtime)
    bufs.append(buf)
    if arr.nbytes:
        copy_host_to_device(buf, host_array_ptr(arr), runtime=runtime)
    return buf


def _alloc(runtime, bufs, nbytes):
    from hipengine.core.memory import malloc

    buf = malloc(max(nbytes, 4), runtime=runtime)
    bufs.append(buf)
    return buf


def _download(runtime, buf, shape, dtype):
    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    arr = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(arr), buf, runtime=runtime)
    return arr


def _free(runtime, bufs) -> None:
    from hipengine.core.memory import free

    for buf in reversed(bufs):
        free(buf, runtime=runtime)


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("threads", [64, 128])
def test_multi_row_marlin_k_matches_per_row_single(_marlin_lib, _runtime, rows: int, threads: int) -> None:
    from hipengine.kernels.hip_gfx1100.quant.paro_marlin_k import (
        gemv_paro_marlin_k_fma_fp16,
        gemv_paro_marlin_k_fma_multi_row_fp16,
    )

    rng = np.random.default_rng(0x317E1A + rows * 17 + threads)
    in_features = 512
    group_size = 128
    out_packed = 12
    out_features = out_packed * 8
    groups = in_features // group_size

    x = (rng.standard_normal((rows, in_features)).astype(np.float32) * 0.05).astype(np.float16)
    # Marlin-K layouts: qweight_mk[out_packed, groups, 128], qzeros_mk[out_packed, groups],
    # scales_mk[out_packed, groups, 8].  qweight is packed as in_features-major within a row.
    qweight_mk = rng.integers(0, 2**32, size=(out_packed, in_features), dtype=np.uint32).view(np.int32)
    qzeros_mk = rng.integers(0, 2**32, size=(out_packed, groups), dtype=np.uint32).view(np.int32)
    scales_mk = rng.uniform(0.001, 0.04, size=(out_packed, groups, 8)).astype(np.float16)

    bufs = []
    try:
        x_dev = _upload(_runtime, bufs, x)
        qw_dev = _upload(_runtime, bufs, qweight_mk)
        qz_dev = _upload(_runtime, bufs, qzeros_mk)
        sc_dev = _upload(_runtime, bufs, scales_mk)
        ref_dev = _alloc(_runtime, bufs, rows * out_features * np.dtype(np.float16).itemsize)
        fast_dev = _alloc(_runtime, bufs, rows * out_features * np.dtype(np.float16).itemsize)

        # Reference: single-row kernel once per row (each row is grid.y=0).
        for r in range(rows):
            gemv_paro_marlin_k_fma_fp16(
                x_dev.ptr + r * in_features * np.dtype(np.float16).itemsize,
                qw_dev.ptr,
                qz_dev.ptr,
                sc_dev.ptr,
                ref_dev.ptr + r * out_features * np.dtype(np.float16).itemsize,
                1,
                in_features,
                out_packed,
                group_size,
                threads=threads,
                library=_marlin_lib,
                runtime=_runtime,
            )
        gemv_paro_marlin_k_fma_multi_row_fp16(
            x_dev.ptr,
            qw_dev.ptr,
            qz_dev.ptr,
            sc_dev.ptr,
            fast_dev.ptr,
            rows,
            in_features,
            out_packed,
            group_size,
            threads=threads,
            library=_marlin_lib,
            runtime=_runtime,
        )
        _runtime.stream_synchronize(0)

        ref = _download(_runtime, ref_dev, (rows, out_features), np.float16)
        fast = _download(_runtime, fast_dev, (rows, out_features), np.float16)
        assert np.array_equal(ref.view(np.uint16), fast.view(np.uint16))
    finally:
        _free(_runtime, bufs)
