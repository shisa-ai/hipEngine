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
    gemv_awq_dual_pack8_output_tiled_strided_bf16,
    gemv_awq_dual_pack8_output_tiled_strided_fp16,
    gemv_awq_dual_pack8_output_tiled_split_transposed_fp16,
    gemv_awq_dual_pack8_output_tiled_transposed_bf16,
    gemv_awq_dual_pack8_output_tiled_transposed_fp16,
    gemv_awq_dual_pack8_strided_bf16,
    gemv_awq_dual_pack8_strided_fp16,
    gemv_awq_dual_pack8_transposed_bf16,
    gemv_awq_dual_pack8_transposed_fp16,
    gemv_awq_pack8_output_tiled_bf16,
    gemv_awq_pack8_output_tiled_combine_residual_transposed_fp16,
    gemv_awq_pack8_output_tiled_fp16,
    gemv_awq_pack8_output_tiled_transposed_bf16,
    gemv_awq_pack8_output_tiled_transposed_fp16,
    gemv_awq_pack8_strided_bf16,
    gemv_awq_pack8_strided_fp16,
    gemv_awq_pack8_transposed_bf16,
    gemv_awq_pack8_transposed_fp16,
)
from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
    build_paro_combine,
    weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w,
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


def _run_case(rows, in_features, out_packed, threads, *, dtype, layout):
    rng = np.random.default_rng(
        1234 + rows * 17 + in_features + out_packed * 3 + threads
        + (1 if dtype == "fp16" else 0) + (7 if layout == "transposed" else 0)
    )
    group_size = 128
    groups = in_features // group_size
    out_features = out_packed * 8
    bits = _bf16_bits if dtype == "bf16" else _fp16_bits
    if layout == "strided":
        ref = gemv_awq_pack8_strided_bf16 if dtype == "bf16" else gemv_awq_pack8_strided_fp16
        tiled = gemv_awq_pack8_output_tiled_bf16 if dtype == "bf16" else gemv_awq_pack8_output_tiled_fp16
        qw_shape = (in_features, out_packed)
    else:
        ref = gemv_awq_pack8_transposed_bf16 if dtype == "bf16" else gemv_awq_pack8_transposed_fp16
        tiled = (
            gemv_awq_pack8_output_tiled_transposed_bf16
            if dtype == "bf16"
            else gemv_awq_pack8_output_tiled_transposed_fp16
        )
        qw_shape = (out_packed, in_features)

    library = build_paro_awq_gemv(load=True)
    x = bits(rng.standard_normal((rows, in_features)).astype(np.float32))
    qweight = rng.integers(0, 2**32, size=qw_shape, dtype=np.uint32).view(np.int32)
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
        ref(x_d.ptr, qw_d.ptr, qz_d.ptr, sc_d.ptr, ref_d.ptr, rows, in_features, out_packed, group_size, threads=threads, library=library)
        tiled(x_d.ptr, qw_d.ptr, qz_d.ptr, sc_d.ptr, test_d.ptr, rows, in_features, out_packed, group_size, threads=threads, library=library)
        copy_device_to_host(host_array_ptr(out_ref), ref_d, out_ref.nbytes)
        copy_device_to_host(host_array_ptr(out_test), test_d, out_test.nbytes)
    finally:
        for b in bufs:
            free(b)
    np.testing.assert_array_equal(
        out_test, out_ref,
        err_msg=f"output-tiled != per-row ({layout}, dtype={dtype}, rows={rows}, in={in_features}, out_packed={out_packed}, threads={threads})",
    )


@pytest.mark.parametrize("layout", ["strided", "transposed"])
@pytest.mark.parametrize("rows", [2, 4, 8])
@pytest.mark.parametrize("in_features,out_packed", _SHAPES)
@pytest.mark.parametrize("threads", [64, 128])
def test_output_tiled_bitexact_bf16(rows, in_features, out_packed, threads, layout):
    _run_case(rows, in_features, out_packed, threads, dtype="bf16", layout=layout)


@pytest.mark.parametrize("layout", ["strided", "transposed"])
@pytest.mark.parametrize("rows", [2, 4, 8])
@pytest.mark.parametrize("in_features,out_packed", _SHAPES)
@pytest.mark.parametrize("threads", [64, 128])
def test_output_tiled_bitexact_fp16(rows, in_features, out_packed, threads, layout):
    _run_case(rows, in_features, out_packed, threads, dtype="fp16", layout=layout)


def _run_output_tiled_combine_case(rows, threads):
    rng = np.random.default_rng(20260612 + rows * 17 + threads)
    group_size = 128
    in_features = 512
    out_packed = 4
    out_features = out_packed * 8
    groups = in_features // group_size
    top_k = 8
    gate_stride = 257

    x = _fp16_bits(rng.standard_normal((rows, in_features)).astype(np.float32))
    qweight = rng.integers(0, 2**32, size=(out_packed, in_features), dtype=np.uint32).view(np.int32)
    qzeros = rng.integers(0, 2**32, size=(groups, out_packed), dtype=np.uint32).view(np.int32)
    scales = _fp16_bits((0.01 * rng.standard_normal((groups, out_features))).astype(np.float32))
    selected = _fp16_bits(rng.standard_normal((rows * top_k, out_features)).astype(np.float32))
    routing = rng.standard_normal((rows * top_k,)).astype(np.float32)
    gate_logits = rng.standard_normal((rows, gate_stride)).astype(np.float32)
    residual = _fp16_bits(rng.standard_normal((rows, out_features)).astype(np.float32))
    shared = np.full((rows, out_features), 0xBEEF, dtype=np.uint16)
    out_ref = np.full((rows, out_features), 0xDEAD, dtype=np.uint16)
    out_test = np.full((rows, out_features), 0xCAFE, dtype=np.uint16)

    awq_lib = build_paro_awq_gemv(load=True)
    combine_lib = build_paro_combine(load=True)
    bufs = []
    try:
        x_d = _dev(np.ascontiguousarray(x)); bufs.append(x_d)
        qw_d = _dev(np.ascontiguousarray(qweight)); bufs.append(qw_d)
        qz_d = _dev(np.ascontiguousarray(qzeros)); bufs.append(qz_d)
        sc_d = _dev(np.ascontiguousarray(scales)); bufs.append(sc_d)
        selected_d = _dev(np.ascontiguousarray(selected)); bufs.append(selected_d)
        routing_d = _dev(np.ascontiguousarray(routing)); bufs.append(routing_d)
        gate_d = _dev(np.ascontiguousarray(gate_logits)); bufs.append(gate_d)
        residual_d = _dev(np.ascontiguousarray(residual)); bufs.append(residual_d)
        shared_d = _dev(shared); bufs.append(shared_d)
        ref_d = _dev(out_ref); bufs.append(ref_d)
        test_d = _dev(out_test); bufs.append(test_d)

        gemv_awq_pack8_output_tiled_transposed_fp16(
            x_d.ptr, qw_d.ptr, qz_d.ptr, sc_d.ptr, shared_d.ptr,
            rows, in_features, out_packed, group_size,
            threads=threads, library=awq_lib,
        )
        weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w(
            selected_d.ptr, routing_d.ptr, shared_d.ptr, gate_d.ptr,
            residual_d.ptr, ref_d.ptr,
            rows, top_k, out_features, gate_stride,
            library=combine_lib,
        )
        gemv_awq_pack8_output_tiled_combine_residual_transposed_fp16(
            x_d.ptr, qw_d.ptr, qz_d.ptr, sc_d.ptr,
            selected_d.ptr, routing_d.ptr, gate_d.ptr, residual_d.ptr,
            test_d.ptr,
            rows, in_features, out_packed, group_size, top_k, gate_stride,
            threads=threads, library=awq_lib,
        )
        copy_device_to_host(host_array_ptr(out_ref), ref_d, out_ref.nbytes)
        copy_device_to_host(host_array_ptr(out_test), test_d, out_test.nbytes)
    finally:
        for b in bufs:
            free(b)

    np.testing.assert_array_equal(
        out_test,
        out_ref,
        err_msg=f"output-tiled shared-down+combine != unfused chain ({rows=}, {threads=})",
    )


@pytest.mark.parametrize("rows", [2, 4, 8])
@pytest.mark.parametrize("threads", [64, 128])
def test_output_tiled_combine_residual_transposed_fp16_matches_unfused(rows, threads):
    _run_output_tiled_combine_case(rows, threads)


_DUAL_SHAPES = [(256, 2, 2), (512, 8, 4), (2048, 4, 2), (4096, 8, 8)]


def _run_dual_case(rows, in_features, out_packed_a, out_packed_b, threads, *, dtype, layout):
    rng = np.random.default_rng(
        99 + rows * 17 + in_features + out_packed_a * 3 + out_packed_b * 5 + threads
        + (1 if dtype == "fp16" else 0) + (7 if layout == "transposed" else 0)
    )
    group_size = 128
    groups = in_features // group_size
    out_features = (out_packed_a + out_packed_b) * 8
    bits = _bf16_bits if dtype == "bf16" else _fp16_bits
    if layout == "strided":
        ref = gemv_awq_dual_pack8_strided_bf16 if dtype == "bf16" else gemv_awq_dual_pack8_strided_fp16
        tiled = (
            gemv_awq_dual_pack8_output_tiled_strided_bf16
            if dtype == "bf16"
            else gemv_awq_dual_pack8_output_tiled_strided_fp16
        )
        qw_shape_a = (in_features, out_packed_a)
        qw_shape_b = (in_features, out_packed_b)
    else:
        ref = gemv_awq_dual_pack8_transposed_bf16 if dtype == "bf16" else gemv_awq_dual_pack8_transposed_fp16
        tiled = (
            gemv_awq_dual_pack8_output_tiled_transposed_bf16
            if dtype == "bf16"
            else gemv_awq_dual_pack8_output_tiled_transposed_fp16
        )
        qw_shape_a = (out_packed_a, in_features)
        qw_shape_b = (out_packed_b, in_features)

    library = build_paro_awq_gemv(load=True)
    x_a = bits(rng.standard_normal((rows, in_features)).astype(np.float32))
    x_b = bits(rng.standard_normal((rows, in_features)).astype(np.float32))
    qweight_a = rng.integers(0, 2**32, size=qw_shape_a, dtype=np.uint32).view(np.int32)
    qweight_b = rng.integers(0, 2**32, size=qw_shape_b, dtype=np.uint32).view(np.int32)
    qzeros_a = rng.integers(0, 2**32, size=(groups, out_packed_a), dtype=np.uint32).view(np.int32)
    qzeros_b = rng.integers(0, 2**32, size=(groups, out_packed_b), dtype=np.uint32).view(np.int32)
    scales_a = bits((0.01 * rng.standard_normal((groups, out_packed_a * 8))).astype(np.float32))
    scales_b = bits((0.01 * rng.standard_normal((groups, out_packed_b * 8))).astype(np.float32))
    out_ref = np.zeros((rows, out_features), dtype=np.uint16)
    out_test = np.full((rows, out_features), 0xDEAD, dtype=np.uint16)

    bufs = []
    try:
        xa_d = _dev(np.ascontiguousarray(x_a)); bufs.append(xa_d)
        xb_d = _dev(np.ascontiguousarray(x_b)); bufs.append(xb_d)
        qwa_d = _dev(np.ascontiguousarray(qweight_a)); bufs.append(qwa_d)
        qwb_d = _dev(np.ascontiguousarray(qweight_b)); bufs.append(qwb_d)
        qza_d = _dev(np.ascontiguousarray(qzeros_a)); bufs.append(qza_d)
        qzb_d = _dev(np.ascontiguousarray(qzeros_b)); bufs.append(qzb_d)
        sca_d = _dev(np.ascontiguousarray(scales_a)); bufs.append(sca_d)
        scb_d = _dev(np.ascontiguousarray(scales_b)); bufs.append(scb_d)
        ref_d = _dev(out_ref); bufs.append(ref_d)
        test_d = _dev(out_test); bufs.append(test_d)
        common = dict(threads=threads, library=library)
        args = (
            qwa_d.ptr, qza_d.ptr, sca_d.ptr, qwb_d.ptr, qzb_d.ptr, scb_d.ptr,
        )
        tail = (rows, in_features, out_packed_a, out_packed_b, group_size)
        if layout == "strided":
            # per-row strided dual takes a single x (separate_inputs=False uses x_a
            # for both groups); the output-tiled strided kernel ignores x_b, so feed
            # the same x_a buffer to both groups to keep the comparison identical.
            ref(xa_d.ptr, *args, ref_d.ptr, *tail, **common)
        else:
            ref(xa_d.ptr, xb_d.ptr, *args, ref_d.ptr, *tail, **common)
        tiled(xa_d.ptr, xb_d.ptr, *args, test_d.ptr, *tail, **common)
        copy_device_to_host(host_array_ptr(out_ref), ref_d, out_ref.nbytes)
        copy_device_to_host(host_array_ptr(out_test), test_d, out_test.nbytes)
    finally:
        for b in bufs:
            free(b)
    np.testing.assert_array_equal(
        out_test, out_ref,
        err_msg=f"dual output-tiled != per-row ({layout}, dtype={dtype}, rows={rows}, in={in_features}, a={out_packed_a}, b={out_packed_b}, threads={threads})",
    )


@pytest.mark.parametrize("layout", ["strided", "transposed"])
@pytest.mark.parametrize("rows", [2, 4, 8])
@pytest.mark.parametrize("in_features,out_packed_a,out_packed_b", _DUAL_SHAPES)
@pytest.mark.parametrize("threads", [64, 128])
def test_dual_output_tiled_bitexact_bf16(rows, in_features, out_packed_a, out_packed_b, threads, layout):
    _run_dual_case(rows, in_features, out_packed_a, out_packed_b, threads, dtype="bf16", layout=layout)


@pytest.mark.parametrize("layout", ["strided", "transposed"])
@pytest.mark.parametrize("rows", [2, 4, 8])
@pytest.mark.parametrize("in_features,out_packed_a,out_packed_b", _DUAL_SHAPES)
@pytest.mark.parametrize("threads", [64, 128])
def test_dual_output_tiled_bitexact_fp16(rows, in_features, out_packed_a, out_packed_b, threads, layout):
    _run_dual_case(rows, in_features, out_packed_a, out_packed_b, threads, dtype="fp16", layout=layout)


def _run_dual_split_case(rows, in_features, out_packed_a, out_packed_b, threads):
    rng = np.random.default_rng(707 + rows * 17 + in_features + out_packed_a * 3 + out_packed_b * 5 + threads)
    group_size = 128
    groups = in_features // group_size
    out_features_a = out_packed_a * 8
    out_features_b = out_packed_b * 8
    out_features = out_features_a + out_features_b
    x_a = _fp16_bits(rng.standard_normal((rows, in_features)).astype(np.float32))
    x_b = _fp16_bits(rng.standard_normal((rows, in_features)).astype(np.float32))
    qweight_a = rng.integers(0, 2**32, size=(out_packed_a, in_features), dtype=np.uint32).view(np.int32)
    qweight_b = rng.integers(0, 2**32, size=(out_packed_b, in_features), dtype=np.uint32).view(np.int32)
    qzeros_a = rng.integers(0, 2**32, size=(groups, out_packed_a), dtype=np.uint32).view(np.int32)
    qzeros_b = rng.integers(0, 2**32, size=(groups, out_packed_b), dtype=np.uint32).view(np.int32)
    scales_a = _fp16_bits((0.01 * rng.standard_normal((groups, out_features_a))).astype(np.float32))
    scales_b = _fp16_bits((0.01 * rng.standard_normal((groups, out_features_b))).astype(np.float32))
    out_packed_ref = np.zeros((rows, out_features), dtype=np.uint16)
    out_a = np.full((rows, out_features_a), 0xDEAD, dtype=np.uint16)
    out_b = np.full((rows, out_features_b), 0xBEEF, dtype=np.uint16)

    library = build_paro_awq_gemv(load=True)
    bufs = []
    try:
        xa_d = _dev(np.ascontiguousarray(x_a)); bufs.append(xa_d)
        xb_d = _dev(np.ascontiguousarray(x_b)); bufs.append(xb_d)
        qwa_d = _dev(np.ascontiguousarray(qweight_a)); bufs.append(qwa_d)
        qwb_d = _dev(np.ascontiguousarray(qweight_b)); bufs.append(qwb_d)
        qza_d = _dev(np.ascontiguousarray(qzeros_a)); bufs.append(qza_d)
        qzb_d = _dev(np.ascontiguousarray(qzeros_b)); bufs.append(qzb_d)
        sca_d = _dev(np.ascontiguousarray(scales_a)); bufs.append(sca_d)
        scb_d = _dev(np.ascontiguousarray(scales_b)); bufs.append(scb_d)
        packed_d = _dev(out_packed_ref); bufs.append(packed_d)
        out_a_d = _dev(out_a); bufs.append(out_a_d)
        out_b_d = _dev(out_b); bufs.append(out_b_d)
        args = (qwa_d.ptr, qza_d.ptr, sca_d.ptr, qwb_d.ptr, qzb_d.ptr, scb_d.ptr)
        tail = (rows, in_features, out_packed_a, out_packed_b, group_size)
        common = dict(threads=threads, library=library)
        gemv_awq_dual_pack8_output_tiled_transposed_fp16(xa_d.ptr, xb_d.ptr, *args, packed_d.ptr, *tail, **common)
        gemv_awq_dual_pack8_output_tiled_split_transposed_fp16(
            xa_d.ptr, xb_d.ptr, *args, out_a_d.ptr, out_b_d.ptr, *tail, **common
        )
        copy_device_to_host(host_array_ptr(out_packed_ref), packed_d, out_packed_ref.nbytes)
        copy_device_to_host(host_array_ptr(out_a), out_a_d, out_a.nbytes)
        copy_device_to_host(host_array_ptr(out_b), out_b_d, out_b.nbytes)
    finally:
        for b in bufs:
            free(b)

    np.testing.assert_array_equal(
        out_a,
        out_packed_ref[:, :out_features_a],
        err_msg=f"split output-tiled A != packed output ({rows=}, {in_features=}, {out_packed_a=}, {threads=})",
    )
    np.testing.assert_array_equal(
        out_b,
        out_packed_ref[:, out_features_a:],
        err_msg=f"split output-tiled B != packed output ({rows=}, {in_features=}, {out_packed_b=}, {threads=})",
    )


@pytest.mark.parametrize("rows", [2, 4, 8])
@pytest.mark.parametrize("in_features,out_packed_a,out_packed_b", _DUAL_SHAPES)
@pytest.mark.parametrize("threads", [64, 128])
def test_dual_output_tiled_split_transposed_bitexact_fp16(rows, in_features, out_packed_a, out_packed_b, threads):
    _run_dual_split_case(rows, in_features, out_packed_a, out_packed_b, threads)
