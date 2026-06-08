"""GPU correctness tests for exact-dequant multi-row AWQ pack8 GEMV.

The DFlash verifier-sized down-projection experiment needs a weight-sharing
multi-row launcher that is bit-identical to the stock row-wise pack8 GEMV.  The
older M12.6 multi-row launcher intentionally follows FP16 prefill-WMMA
dequantization, so this test compares the new decode-dequant variant directly
against ``gemv_awq_pack8_transposed_fp16`` on small synthetic verifier shapes.
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
def _awq_lib():
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.quant.paro_awq_gemv import build_paro_awq_gemv

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = pathlib.Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment("gfx1100"):
        return build_paro_awq_gemv(load=True, compiler_version=compiler_version)


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


def _packed_u4_words(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.integers(0, 2**32, size=shape, dtype=np.uint32).view(np.int32)


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 8])
def test_multi_row_decode_transposed_fp16_matches_rowwise_pack8(_awq_lib, _runtime, rows: int) -> None:
    from hipengine.kernels.hip_gfx1100.quant.paro_awq_gemv import (
        gemv_awq_pack8_multi_row_decode_transposed_fp16,
        gemv_awq_pack8_transposed_fp16,
    )

    rng = np.random.default_rng(0xDFA5A11 + rows)
    in_features = 256
    group_size = 64
    out_packed = 16
    out_features = out_packed * 8
    groups = in_features // group_size

    x = (rng.standard_normal((rows, in_features)).astype(np.float32) * 0.05).astype(np.float16)
    qweight_t = _packed_u4_words(rng, (out_packed, in_features))
    qzeros = _packed_u4_words(rng, (groups, out_packed))
    scales = rng.uniform(0.001, 0.04, size=(groups, out_features)).astype(np.float16)

    bufs = []
    try:
        x_dev = _upload(_runtime, bufs, x)
        qw_dev = _upload(_runtime, bufs, qweight_t)
        qz_dev = _upload(_runtime, bufs, qzeros)
        scales_dev = _upload(_runtime, bufs, scales)
        ref_dev = _alloc(_runtime, bufs, rows * out_features * np.dtype(np.float16).itemsize)
        fast_dev = _alloc(_runtime, bufs, rows * out_features * np.dtype(np.float16).itemsize)

        gemv_awq_pack8_transposed_fp16(
            x_dev.ptr,
            qw_dev.ptr,
            qz_dev.ptr,
            scales_dev.ptr,
            ref_dev.ptr,
            rows,
            in_features,
            out_packed,
            group_size,
            threads=128,
            library=_awq_lib,
            runtime=_runtime,
        )
        gemv_awq_pack8_multi_row_decode_transposed_fp16(
            x_dev.ptr,
            qw_dev.ptr,
            qz_dev.ptr,
            scales_dev.ptr,
            fast_dev.ptr,
            rows,
            in_features,
            out_packed,
            group_size,
            threads=128,
            library=_awq_lib,
            runtime=_runtime,
        )
        _runtime.stream_synchronize(0)

        ref = _download(_runtime, ref_dev, (rows, out_features), np.float16)
        fast = _download(_runtime, fast_dev, (rows, out_features), np.float16)
        assert np.array_equal(ref.view(np.uint16), fast.view(np.uint16))
    finally:
        _free(_runtime, bufs)


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 8])
def test_dual_decode_split_matches_two_decode_singles(_awq_lib, _runtime, rows: int) -> None:
    """M15.3: the decode-dequant split dual must equal two decode singles bit-for-bit."""

    from hipengine.kernels.hip_gfx1100.quant.paro_awq_gemv import (
        gemv_awq_dual_pack8_multi_row_decode_split_transposed_fp16,
        gemv_awq_pack8_multi_row_decode_transposed_fp16,
    )

    rng = np.random.default_rng(0xD0A15 + rows)
    in_features = 256
    group_size = 64
    out_packed_a = 16
    out_packed_b = 10
    out_features_a = out_packed_a * 8
    out_features_b = out_packed_b * 8
    groups = in_features // group_size
    fp16 = np.dtype(np.float16).itemsize

    x_a = (rng.standard_normal((rows, in_features)).astype(np.float32) * 0.05).astype(np.float16)
    x_b = (rng.standard_normal((rows, in_features)).astype(np.float32) * 0.05).astype(np.float16)
    qw_a = _packed_u4_words(rng, (out_packed_a, in_features))
    qz_a = _packed_u4_words(rng, (groups, out_packed_a))
    sc_a = rng.uniform(0.001, 0.04, size=(groups, out_features_a)).astype(np.float16)
    qw_b = _packed_u4_words(rng, (out_packed_b, in_features))
    qz_b = _packed_u4_words(rng, (groups, out_packed_b))
    sc_b = rng.uniform(0.001, 0.04, size=(groups, out_features_b)).astype(np.float16)

    bufs = []
    try:
        xa = _upload(_runtime, bufs, x_a)
        xb = _upload(_runtime, bufs, x_b)
        wa = _upload(_runtime, bufs, qw_a)
        za = _upload(_runtime, bufs, qz_a)
        sa = _upload(_runtime, bufs, sc_a)
        wb = _upload(_runtime, bufs, qw_b)
        zb = _upload(_runtime, bufs, qz_b)
        sb = _upload(_runtime, bufs, sc_b)
        ref_a = _alloc(_runtime, bufs, rows * out_features_a * fp16)
        ref_b = _alloc(_runtime, bufs, rows * out_features_b * fp16)
        fast_a = _alloc(_runtime, bufs, rows * out_features_a * fp16)
        fast_b = _alloc(_runtime, bufs, rows * out_features_b * fp16)

        gemv_awq_pack8_multi_row_decode_transposed_fp16(
            xa.ptr, wa.ptr, za.ptr, sa.ptr, ref_a.ptr, rows, in_features, out_packed_a, group_size,
            threads=128, library=_awq_lib, runtime=_runtime,
        )
        gemv_awq_pack8_multi_row_decode_transposed_fp16(
            xb.ptr, wb.ptr, zb.ptr, sb.ptr, ref_b.ptr, rows, in_features, out_packed_b, group_size,
            threads=128, library=_awq_lib, runtime=_runtime,
        )
        gemv_awq_dual_pack8_multi_row_decode_split_transposed_fp16(
            xa.ptr, xb.ptr, wa.ptr, za.ptr, sa.ptr, wb.ptr, zb.ptr, sb.ptr,
            fast_a.ptr, fast_b.ptr, rows, in_features, out_packed_a, out_packed_b, group_size,
            threads=128, library=_awq_lib, runtime=_runtime,
        )
        _runtime.stream_synchronize(0)

        ra = _download(_runtime, ref_a, (rows, out_features_a), np.float16)
        rb = _download(_runtime, ref_b, (rows, out_features_b), np.float16)
        fa = _download(_runtime, fast_a, (rows, out_features_a), np.float16)
        fb = _download(_runtime, fast_b, (rows, out_features_b), np.float16)
        assert np.array_equal(ra.view(np.uint16), fa.view(np.uint16))
        assert np.array_equal(rb.view(np.uint16), fb.view(np.uint16))
    finally:
        _free(_runtime, bufs)
