"""Bit-exactness gate for the output-column-tiled c>1 decode AWQ GEMV (C3.0c).

The output-tiled kernel loads each weight pack once and accumulates all ``C``
active columns, with the *same* per-output accumulation order (over k and across
warps) as the per-row ``gemv_awq_pack8_strided`` kernel.  It must therefore
produce byte-identical results.  These tests feed both kernels identical random
AWQ pack8 inputs and assert exact equality, so no tolerance/CPU-reference is
needed.  HIP-guarded so no-ROCm runners skip.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.paro_awq_gemv import (
    build_paro_awq_gemv,
    gemv_awq_pack8_output_tiled_bf16,
    gemv_awq_pack8_output_tiled_fp16,
    gemv_awq_pack8_strided_bf16,
    gemv_awq_pack8_strided_fp16,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="requires HIP/ROCm")


def _bf16_bits(arr_f32: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(arr_f32, dtype=np.float32).view(np.uint32)
    return (u >> 16).astype(np.uint16)


def _fp16_bits(arr_f32: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr_f32, dtype=np.float16).view(np.uint16)


def _dev(arr: np.ndarray):
    buf = malloc(arr.nbytes)
    copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes)
    return buf


_SHAPES = [(256, 2), (512, 16), (2048, 4), (4096, 8)]


def _run_case(rows, in_features, out_packed, threads, *, dtype):
    rng = np.random.default_rng(1234 + rows * 17 + in_features + out_packed * 3 + threads + (1 if dtype == "fp16" else 0))
    group_size = 128
    groups = in_features // group_size
    out_features = out_packed * 8
    bits = _bf16_bits if dtype == "bf16" else _fp16_bits
    strided = gemv_awq_pack8_strided_bf16 if dtype == "bf16" else gemv_awq_pack8_strided_fp16
    tiled = gemv_awq_pack8_output_tiled_bf16 if dtype == "bf16" else gemv_awq_pack8_output_tiled_fp16

    library = build_paro_awq_gemv(load=True)
    x = bits(rng.standard_normal((rows, in_features)).astype(np.float32))
    qweight = rng.integers(0, 2**32, size=(in_features, out_packed), dtype=np.uint32).view(np.int32)
    qzeros = rng.integers(0, 2**32, size=(groups, out_packed), dtype=np.uint32).view(np.int32)
    scales = bits((0.01 * rng.standard_normal((groups, out_features))).astype(np.float32))
    out_ref = np.zeros((rows, out_features), dtype=np.uint16)
    out_test = np.full((rows, out_features), 0xDEAD, dtype=np.uint16)

    bufs = []
    try:
        x_d = _dev(np.ascontiguousarray(x)); bufs.append(x_d)
        qw_d = _dev(np.ascontiguousarray(qweight)); bufs.append(qw_d)
        qz_d = _dev(np.ascontiguousarray(qzeros)); bufs.append(qz_d)
        sc_d = _dev(np.ascontiguousarray(scales)); bufs.append(sc_d)
        ref_d = _dev(out_ref); bufs.append(ref_d)
        test_d = _dev(out_test); bufs.append(test_d)
        strided(x_d.ptr, qw_d.ptr, qz_d.ptr, sc_d.ptr, ref_d.ptr, rows, in_features, out_packed, group_size, threads=threads, library=library)
        tiled(x_d.ptr, qw_d.ptr, qz_d.ptr, sc_d.ptr, test_d.ptr, rows, in_features, out_packed, group_size, threads=threads, library=library)
        copy_device_to_host(host_array_ptr(out_ref), ref_d, out_ref.nbytes)
        copy_device_to_host(host_array_ptr(out_test), test_d, out_test.nbytes)
    finally:
        for b in bufs:
            free(b)
    np.testing.assert_array_equal(
        out_test, out_ref,
        err_msg=f"output-tiled != strided (bf16/fp16={dtype}, rows={rows}, in={in_features}, out_packed={out_packed}, threads={threads})",
    )


@pytest.mark.parametrize("rows", [2, 4, 8])
@pytest.mark.parametrize("in_features,out_packed", _SHAPES)
@pytest.mark.parametrize("threads", [64, 128])
def test_output_tiled_bitexact_bf16(rows, in_features, out_packed, threads):
    _run_case(rows, in_features, out_packed, threads, dtype="bf16")


@pytest.mark.parametrize("rows", [2, 4, 8])
@pytest.mark.parametrize("in_features,out_packed", _SHAPES)
@pytest.mark.parametrize("threads", [64, 128])
def test_output_tiled_bitexact_fp16(rows, in_features, out_packed, threads):
    _run_case(rows, in_features, out_packed, threads, dtype="fp16")
