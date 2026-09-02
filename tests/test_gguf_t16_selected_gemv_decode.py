"""Correctness fixtures for selected GGUF K-family T16 GEMV decode (P9.H3)."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.specdec2_scope import (
    physical_exact_rowtiles_session,
    q5_t16_physical_rowtile_session,
)
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
    build_paro_combine,
    weighted_sum_out_bf16_f32w,
)
from hipengine.kernels.hip_gfx1100.quant import (
    gguf_t16_selected_gemv as selected_t16_mod,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_pack8_dual_rowtile_silu_bf16_bf16_out,
    gguf_q4_k_pack8_gemv_bf16_bf16_out,
    gguf_q4_k_pack8_rowtile_bf16_bf16_out,
    gguf_q4_k_quantize_bf16_q8_1,
    gguf_q4_k_quantize_bf16_q8_1x2,
    gguf_q4_k_selected_dual_gemv_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    build_gguf_t16_selected_gemv,
    gguf_q4_k_qmicro_t16_dense_dual_q8_1x2_split_weight_dp4a_silu_bf16_bf16_out,
    gguf_q4_k_qmicro_t16_dense_dual_rowtile_silu_bf16_bf16_out,
    gguf_q4_k_qmicro_t16_dense_rowtile_bf16_bf16_out,
    gguf_q4_k_qmicro_t16_dense_single_local32_bf16_bf16_out,
    gguf_q4_k_t16_dense_dual_interleaved_tile2_local32_silu_bf16_bf16_out,
    gguf_q4_k_t16_dense_dual_local32_silu_bf16_bf16_out,
    gguf_q4_k_t16_dense_dual_q8_1x2_dp4a_silu_bf16_bf16_out,
    gguf_q4_k_t16_dense_dual_q8_1x2_split_weight_dp4a_silu_bf16_bf16_out,
    gguf_q4_k_t16_dense_dual_rowtile_silu_bf16_bf16_out,
    gguf_q4_k_t16_dense_rowtile_bf16_bf16_out,
    gguf_q4_k_t16_dense_rowtile16_w2_bf16_bf16_out,
    gguf_q4_k_t16_dense_rowtile16_w2_grouped_rows6_bf16_bf16_out,
    gguf_q4_k_t16_dense_rowtile16_w2_grouped_rows8_bf16_bf16_out,
    gguf_q4_k_t16_dense_rowtile_col4_bf16_bf16_out,
    gguf_q4_k_t16_dense_single_col4_bf16_bf16_out,
    gguf_q4_k_t16_dense_single_local32_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_natural_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_natural_tile8_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_natural_tile8_parallel_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_halfdot_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_interleaved_natural_tile8_parallel_silu_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_interleaved_natural_tile8_parallel_silu_halfdot_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_paircoeff_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_pairq_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_pairreuse_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out,
    gguf_q4_k_qmicro_t16_selected_dual_silu_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_silu_q8_1_dp4a_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_silu_q8_1x2_dp4a_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_gemv_decode_compact_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_gemv_decode_compact_fp16_fp16_out,
    gguf_q4_k_t16_selected_dual_grouped_smallm_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_pairreuse_gemv_decode_compact_bf16_bf16_out,
    gguf_q4_k_t16_selected_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_natural_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_natural_parallel_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_natural_parallel_paircoeff_weighted_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_natural_parallel_weighted_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_gemv_decode_compact_bf16_bf16_out,
    gguf_q4_k_t16_selected_grouped_smallm_bf16_bf16_out,
    gguf_q4_k_t16_selected_pairreuse_gemv_decode_compact_bf16_bf16_out,
    gguf_q4_k_t16_selected_gemv_decode_compact_fp16_fp16_out,
    gguf_q5_k_t16_gemv_decode_bf16_bf16_out,
    gguf_q5_k_t16_gemv_decode_tile8_bf16_bf16_out,
    gguf_q5_k_t16_gemv_rowtile_bf16_bf16_out,
    gguf_q5_k_t16_gemv_rowtile_col8_bf16_bf16_out,
    gguf_q5_k_t16_gemv_rowtile_grouped_rows6_bf16_bf16_out,
    gguf_q5_k_qmicro_t16_selected_qwen_tile8_gemv_bf16_bf16_out,
    gguf_q5_k_t16_selected_gemv_bf16_bf16_out,
    gguf_q5_k_t16_selected_qwen_tile8_gemv_bf16_bf16_out,
    gguf_q5_k_t16_selected_pairreuse_gemv_bf16_bf16_out,
    gguf_q5_k_t16_selected_q8_1_dp4a_gemv_bf16_bf16_out,
    gguf_q5_k_t16_selected_gemv_decode_compact_bf16_bf16_out,
    gguf_q5_k_t16_selected_gemv_decode_compact_fp16_fp16_out,
    gguf_q5_k_t16_selected_pairreuse_gemv_decode_compact_bf16_bf16_out,
    gguf_q6_k_t16_selected_gemv_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_selected_natural_gemv_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_selected_natural_parallel_gemv_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_selected_natural_parallel_weighted_gemv_bf16_bf16_out,
    gguf_q6_k_t16_selected_pairreuse_gemv_bf16_bf16_out,
    gguf_q6_k_t16_selected_gemv_decode_compact_bf16_bf16_out,
    gguf_q6_k_t16_selected_gemv_decode_compact_fp16_fp16_out,
    gguf_q6_k_t16_selected_grouped_smallm_bf16_bf16_out,
    gguf_q6_k_t16_selected_pairreuse_gemv_decode_compact_bf16_bf16_out,
    plan_gguf_t16_selected_gemv_build,
    register_gguf_t16_selected_gemv_kernels,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_q4_k import (
    interleave_gguf_q4_k_tile16_dual,
    repack_gguf_q4_k_pack8,
    repack_gguf_q4_k_tile16,
    repack_gguf_q4_k_tile16_qmicro,
)
from hipengine.quant.gguf_t16 import (
    repack_gguf_q5_k_qmicro_tile16,
    repack_gguf_q5_k_tile16,
    repack_gguf_q6_k_tile16,
    repack_gguf_q6_k_tile16_qmicro,
    repack_gguf_q6_k_tile16_qmicro_planar,
)
from tests._gguf_synthetic_weights import make_q4_k_weight, make_q5_k_weight, make_q6_k_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def t16_selected_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_t16_selected_gemv(load=True)


@pytest.fixture(scope="module")
def combine_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_paro_combine(load=True)


@pytest.fixture(scope="module")
def q4_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_q4_k_gemv(load=True)


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    nan_mask = np.isnan(f32)
    lsb = (u32 >> 16) & 1
    rounded = ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16)
    rounded[nan_mask] = 0x7FC0
    return rounded.reshape(f32.shape)


def _bf16_u16_to_f32(arr: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(arr, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(u16.shape).copy()


def _stack_experts(builder, out_features: int, in_features: int, num_experts: int, seed: int) -> np.ndarray:
    base = builder(out_features, in_features)
    return np.stack([np.roll(base, shift=e + seed, axis=0) for e in range(num_experts)], axis=0)


def _expected_dual(
    x_ref: np.ndarray,
    expert_start: np.ndarray,
    qa: np.ndarray,
    qb: np.ndarray,
    out_features_a: int,
    out_features_b: int,
) -> np.ndarray:
    compact_rows = int(expert_start[-1])
    out = np.zeros((compact_rows, out_features_a + out_features_b), dtype=np.float32)
    for e in range(len(expert_start) - 1):
        s, sl = int(expert_start[e]), int(expert_start[e + 1])
        if sl == s:
            continue
        out[s:sl, :out_features_a] = gguf_quant_gemv(x_ref[s:sl], qa[e], GGMLQuantizationType.Q4_K)
        out[s:sl, out_features_a:] = gguf_quant_gemv(x_ref[s:sl], qb[e], GGMLQuantizationType.Q4_K)
    return out


def _expected_single(
    x_ref: np.ndarray,
    expert_start: np.ndarray,
    qw: np.ndarray,
    out_features: int,
    qtype_enum: GGMLQuantizationType,
) -> np.ndarray:
    compact_rows = int(expert_start[-1])
    out = np.zeros((compact_rows, out_features), dtype=np.float32)
    for e in range(len(expert_start) - 1):
        s, sl = int(expert_start[e]), int(expert_start[e + 1])
        if sl == s:
            continue
        out[s:sl] = gguf_quant_gemv(x_ref[s:sl], qw[e], qtype_enum)
    return out


def _x_by_selected_lane(x_ref: np.ndarray, rows: int) -> np.ndarray:
    lanes_per_x_row = rows // x_ref.shape[0]
    return x_ref[np.arange(rows) // lanes_per_x_row]


def _expected_direct_dual(
    x_ref: np.ndarray,
    selected: np.ndarray,
    qa: np.ndarray,
    qb: np.ndarray,
    out_features_a: int,
    out_features_b: int,
) -> np.ndarray:
    x_rows = _x_by_selected_lane(x_ref, int(selected.size))
    out = np.zeros((int(selected.size), out_features_a + out_features_b), dtype=np.float32)
    for row, expert in enumerate(selected.astype(np.int64).tolist()):
        out[row : row + 1, :out_features_a] = gguf_quant_gemv(x_rows[row : row + 1], qa[expert], GGMLQuantizationType.Q4_K)
        out[row : row + 1, out_features_a:] = gguf_quant_gemv(x_rows[row : row + 1], qb[expert], GGMLQuantizationType.Q4_K)
    return out


def _expected_direct_single(
    x_ref: np.ndarray,
    selected: np.ndarray,
    qw: np.ndarray,
    out_features: int,
    qtype_enum: GGMLQuantizationType,
) -> np.ndarray:
    x_rows = _x_by_selected_lane(x_ref, int(selected.size))
    out = np.zeros((int(selected.size), out_features), dtype=np.float32)
    for row, expert in enumerate(selected.astype(np.int64).tolist()):
        out[row] = gguf_quant_gemv(x_rows[row : row + 1], qw[expert], qtype_enum)[0]
    return out


def _softmax_kl(ref: np.ndarray, cand: np.ndarray) -> tuple[float, float]:
    ref64 = ref.astype(np.float64)
    cand64 = cand.astype(np.float64)

    def logsm(x: np.ndarray) -> np.ndarray:
        shifted = x - x.max(axis=-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

    log_ref = logsm(ref64)
    log_cand = logsm(cand64)
    row_kl = np.sum(np.exp(log_ref) * (log_ref - log_cand), axis=-1)
    return float(np.mean(row_kl)), float(np.max(row_kl))


def _top1(ref: np.ndarray, cand: np.ndarray) -> float:
    return float(np.mean(ref.argmax(axis=-1) == cand.argmax(axis=-1)))


def _run_dual(fn, x_dev, expert_start, ta, tb, out_features_a, out_features_b, out_dtype, library) -> np.ndarray:
    compact_rows = int(expert_start[-1])
    in_features = x_dev.shape[1]
    x_buf = malloc(x_dev.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
    es_buf = malloc(expert_start.nbytes)
    copy_host_to_device(es_buf, host_array_ptr(expert_start), expert_start.nbytes)
    ta_buf = malloc(ta.nbytes)
    copy_host_to_device(ta_buf, host_array_ptr(ta), ta.nbytes)
    tb_buf = malloc(tb.nbytes)
    copy_host_to_device(tb_buf, host_array_ptr(tb), tb.nbytes)
    out_arr = np.zeros((compact_rows, out_features_a + out_features_b), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    try:
        fn(
            x_buf.ptr,
            es_buf.ptr,
            ta_buf.ptr,
            tb_buf.ptr,
            out_buf.ptr,
            compact_rows,
            in_features,
            out_features_a,
            out_features_b,
            ta.shape[0],
            library=library,
        )
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for buf in (x_buf, es_buf, ta_buf, tb_buf, out_buf):
            free(buf)


def _run_grouped_dual(
    fn,
    x_dev,
    expert_start,
    ta,
    tb,
    out_features,
    out_dtype,
    library,
) -> tuple[np.ndarray, np.ndarray]:
    compact_rows = int(expert_start[-1])
    in_features = x_dev.shape[1]
    x_buf = malloc(x_dev.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
    es_buf = malloc(expert_start.nbytes)
    copy_host_to_device(es_buf, host_array_ptr(expert_start), expert_start.nbytes)
    active = np.flatnonzero(np.diff(expert_start)).astype(np.int64)
    active_count = np.asarray([active.size], dtype=np.int64)
    active_buf = malloc(active.nbytes)
    copy_host_to_device(active_buf, host_array_ptr(active), active.nbytes)
    active_count_buf = malloc(active_count.nbytes)
    copy_host_to_device(
        active_count_buf, host_array_ptr(active_count), active_count.nbytes
    )
    ta_buf = malloc(ta.nbytes)
    copy_host_to_device(ta_buf, host_array_ptr(ta), ta.nbytes)
    tb_buf = malloc(tb.nbytes)
    copy_host_to_device(tb_buf, host_array_ptr(tb), tb.nbytes)
    out_a = np.zeros((compact_rows, out_features), dtype=out_dtype)
    out_b = np.zeros((compact_rows, out_features), dtype=out_dtype)
    out_a_buf = malloc(out_a.nbytes)
    out_b_buf = malloc(out_b.nbytes)
    try:
        fn(
            x_buf.ptr,
            es_buf.ptr,
            active_buf.ptr,
            active_count_buf.ptr,
            ta_buf.ptr,
            tb_buf.ptr,
            out_a_buf.ptr,
            out_b_buf.ptr,
            compact_rows,
            in_features,
            out_features,
            len(expert_start) - 1,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out_a), out_a_buf, out_a.nbytes)
        copy_device_to_host(host_array_ptr(out_b), out_b_buf, out_b.nbytes)
        return out_a, out_b
    finally:
        for buf in (
            x_buf,
            es_buf,
            active_buf,
            active_count_buf,
            ta_buf,
            tb_buf,
            out_a_buf,
            out_b_buf,
        ):
            free(buf)


def _run_grouped_smallm_single(
    fn, x_dev, expert_start, tiles, out_features, out_dtype, library, **kwargs
) -> np.ndarray:
    compact_rows = int(expert_start[-1])
    in_features = x_dev.shape[1]
    x_buf = malloc(x_dev.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
    es_buf = malloc(expert_start.nbytes)
    copy_host_to_device(es_buf, host_array_ptr(expert_start), expert_start.nbytes)
    active = np.flatnonzero(np.diff(expert_start)).astype(np.int64)
    active_count = np.asarray([active.size], dtype=np.int64)
    active_buf = malloc(active.nbytes)
    copy_host_to_device(active_buf, host_array_ptr(active), active.nbytes)
    active_count_buf = malloc(active_count.nbytes)
    copy_host_to_device(
        active_count_buf, host_array_ptr(active_count), active_count.nbytes
    )
    w_buf = malloc(tiles.nbytes)
    copy_host_to_device(w_buf, host_array_ptr(tiles), tiles.nbytes)
    out_arr = np.zeros((compact_rows, out_features), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    try:
        fn(
            x_buf.ptr,
            es_buf.ptr,
            active_buf.ptr,
            active_count_buf.ptr,
            w_buf.ptr,
            out_buf.ptr,
            compact_rows,
            in_features,
            out_features,
            tiles.shape[0],
            library=library,
            **kwargs,
        )
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for buf in (
            x_buf,
            es_buf,
            active_buf,
            active_count_buf,
            w_buf,
            out_buf,
        ):
            free(buf)


def _run_single(fn, x_dev, expert_start, tiles, out_features, out_dtype, library) -> np.ndarray:
    compact_rows = int(expert_start[-1])
    in_features = x_dev.shape[1]
    x_buf = malloc(x_dev.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
    es_buf = malloc(expert_start.nbytes)
    copy_host_to_device(es_buf, host_array_ptr(expert_start), expert_start.nbytes)
    w_buf = malloc(tiles.nbytes)
    copy_host_to_device(w_buf, host_array_ptr(tiles), tiles.nbytes)
    out_arr = np.zeros((compact_rows, out_features), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    try:
        fn(
            x_buf.ptr,
            es_buf.ptr,
            w_buf.ptr,
            out_buf.ptr,
            compact_rows,
            in_features,
            out_features,
            tiles.shape[0],
            library=library,
        )
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for buf in (x_buf, es_buf, w_buf, out_buf):
            free(buf)


def _run_direct_dual(fn, x_dev, selected, ta, tb, out_features, out_dtype, library) -> tuple[np.ndarray, np.ndarray]:
    rows = int(selected.size)
    in_features = x_dev.shape[1]
    x_buf = malloc(x_dev.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
    sel_buf = malloc(selected.nbytes)
    copy_host_to_device(sel_buf, host_array_ptr(selected), selected.nbytes)
    ta_buf = malloc(ta.nbytes)
    copy_host_to_device(ta_buf, host_array_ptr(ta), ta.nbytes)
    tb_buf = malloc(tb.nbytes)
    copy_host_to_device(tb_buf, host_array_ptr(tb), tb.nbytes)
    out_a = np.zeros((rows, out_features), dtype=out_dtype)
    out_b = np.zeros((rows, out_features), dtype=out_dtype)
    out_a_buf = malloc(out_a.nbytes)
    out_b_buf = malloc(out_b.nbytes)
    try:
        fn(
            x_buf.ptr,
            sel_buf.ptr,
            ta_buf.ptr,
            tb_buf.ptr,
            out_a_buf.ptr,
            out_b_buf.ptr,
            x_dev.shape[0],
            rows,
            ta.shape[0],
            in_features,
            out_features,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out_a), out_a_buf, out_a.nbytes)
        copy_device_to_host(host_array_ptr(out_b), out_b_buf, out_b.nbytes)
        return out_a, out_b
    finally:
        for buf in (x_buf, sel_buf, ta_buf, tb_buf, out_a_buf, out_b_buf):
            free(buf)


def _run_direct_dual_q8_dp4a(
    x_dev,
    selected,
    ta,
    tb,
    out_features,
    out_dtype,
    t16_library,
    q4_library,
) -> tuple[np.ndarray, np.ndarray]:
    rows = int(selected.size)
    in_features = x_dev.shape[1]
    x_buf = malloc(x_dev.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
    sel_buf = malloc(selected.nbytes)
    copy_host_to_device(sel_buf, host_array_ptr(selected), selected.nbytes)
    ta_buf = malloc(ta.nbytes)
    copy_host_to_device(ta_buf, host_array_ptr(ta), ta.nbytes)
    tb_buf = malloc(tb.nbytes)
    copy_host_to_device(tb_buf, host_array_ptr(tb), tb.nbytes)
    xq_buf = malloc(x_dev.shape[0] * (in_features // 32) * 36)
    out_a = np.zeros((rows, out_features), dtype=out_dtype)
    out_b = np.zeros((rows, out_features), dtype=out_dtype)
    out_a_buf = malloc(out_a.nbytes)
    out_b_buf = malloc(out_b.nbytes)
    try:
        gguf_q4_k_quantize_bf16_q8_1(
            x_buf.ptr,
            xq_buf.ptr,
            x_dev.shape[0],
            in_features,
            library=q4_library,
        )
        gguf_q4_k_t16_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out(
            xq_buf.ptr,
            sel_buf.ptr,
            ta_buf.ptr,
            tb_buf.ptr,
            out_a_buf.ptr,
            out_b_buf.ptr,
            x_dev.shape[0],
            rows,
            ta.shape[0],
            in_features,
            out_features,
            library=t16_library,
        )
        copy_device_to_host(host_array_ptr(out_a), out_a_buf, out_a.nbytes)
        copy_device_to_host(host_array_ptr(out_b), out_b_buf, out_b.nbytes)
        return out_a, out_b
    finally:
        for buf in (x_buf, sel_buf, ta_buf, tb_buf, xq_buf, out_a_buf, out_b_buf):
            free(buf)


def _run_direct_dual_silu(fn, x_dev, selected, ta, tb, out_features, out_dtype, library) -> np.ndarray:
    rows = int(selected.size)
    in_features = x_dev.shape[1]
    x_buf = malloc(x_dev.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
    sel_buf = malloc(selected.nbytes)
    copy_host_to_device(sel_buf, host_array_ptr(selected), selected.nbytes)
    ta_buf = malloc(ta.nbytes)
    copy_host_to_device(ta_buf, host_array_ptr(ta), ta.nbytes)
    tb_buf = malloc(tb.nbytes)
    copy_host_to_device(tb_buf, host_array_ptr(tb), tb.nbytes)
    out_arr = np.zeros((rows, out_features), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    try:
        fn(
            x_buf.ptr,
            sel_buf.ptr,
            ta_buf.ptr,
            tb_buf.ptr,
            out_buf.ptr,
            x_dev.shape[0],
            rows,
            ta.shape[0],
            in_features,
            out_features,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for buf in (x_buf, sel_buf, ta_buf, tb_buf, out_buf):
            free(buf)


def _run_dense_dual_silu(
    x_dev,
    ta,
    tb,
    out_features,
    out_dtype,
    library,
    *,
    fn=gguf_q4_k_t16_dense_dual_local32_silu_bf16_bf16_out,
) -> np.ndarray:
    rows, in_features = x_dev.shape
    x_buf = malloc(x_dev.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
    ta_buf = malloc(ta.nbytes)
    copy_host_to_device(ta_buf, host_array_ptr(ta), ta.nbytes)
    tb_buf = malloc(tb.nbytes)
    copy_host_to_device(tb_buf, host_array_ptr(tb), tb.nbytes)
    out_arr = np.zeros((rows, out_features), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    try:
        fn(
            x_buf.ptr,
            ta_buf.ptr,
            tb_buf.ptr,
            out_buf.ptr,
            rows,
            in_features,
            out_features,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for buf in (x_buf, ta_buf, tb_buf, out_buf):
            free(buf)


def _run_dense_dual_interleaved_silu(
    fn,
    x_dev,
    tiles_dual,
    out_features,
    out_dtype,
    library,
) -> np.ndarray:
    in_features = x_dev.shape[1]
    x_buf = malloc(x_dev.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
    tiles_buf = malloc(tiles_dual.nbytes)
    copy_host_to_device(
        tiles_buf,
        host_array_ptr(tiles_dual),
        tiles_dual.nbytes,
    )
    out_arr = np.zeros((1, out_features), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    try:
        fn(
            x_buf.ptr,
            tiles_buf.ptr,
            out_buf.ptr,
            1,
            in_features,
            out_features,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for buf in (x_buf, tiles_buf, out_buf):
            free(buf)


def _run_dense_single(
    fn,
    x_dev,
    tiles,
    out_features,
    out_dtype,
    library,
) -> np.ndarray:
    rows, in_features = x_dev.shape
    x_buf = malloc(x_dev.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
    tiles_buf = malloc(tiles.nbytes)
    copy_host_to_device(tiles_buf, host_array_ptr(tiles), tiles.nbytes)
    out_arr = np.zeros((rows, out_features), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    try:
        fn(
            x_buf.ptr,
            tiles_buf.ptr,
            out_buf.ptr,
            rows,
            in_features,
            out_features,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for buf in (x_buf, tiles_buf, out_buf):
            free(buf)


def _run_dense_residual(
    fn,
    x_dev,
    tiles,
    residual,
    out_features,
    library,
) -> np.ndarray:
    rows, in_features = x_dev.shape
    buffers = []
    try:
        device = []
        for value in (x_dev, tiles, residual):
            buffer = malloc(value.nbytes)
            buffers.append(buffer)
            device.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(value), value.nbytes)
        out = np.zeros((rows, out_features), dtype=np.uint16)
        out_buffer = malloc(out.nbytes)
        buffers.append(out_buffer)
        fn(
            device[0].ptr,
            device[1].ptr,
            device[2].ptr,
            out_buffer.ptr,
            rows,
            in_features,
            out_features,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out), out_buffer, out.nbytes)
        return out
    finally:
        for buffer in reversed(buffers):
            free(buffer)


def _run_pack8_single(
    x_dev,
    packed,
    out_features,
    q4_library,
) -> np.ndarray:
    rows, in_features = x_dev.shape
    buffers = []
    try:
        x_buf = malloc(x_dev.nbytes)
        buffers.append(x_buf)
        copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
        packed_buffers = []
        for value in (packed.qweight, packed.scales, packed.mins):
            buffer = malloc(value.nbytes)
            buffers.append(buffer)
            packed_buffers.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(value), value.nbytes)
        out_arr = np.zeros((rows, out_features), dtype=np.uint16)
        out_buf = malloc(out_arr.nbytes)
        buffers.append(out_buf)
        fn = (
            gguf_q4_k_pack8_gemv_bf16_bf16_out
            if rows == 1
            else gguf_q4_k_pack8_rowtile_bf16_bf16_out
        )
        fn(
            x_buf.ptr,
            packed_buffers[0].ptr,
            packed_buffers[1].ptr,
            packed_buffers[2].ptr,
            out_buf.ptr,
            rows,
            in_features,
            out_features,
            threads=32,
            library=q4_library,
        )
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for buffer in reversed(buffers):
            free(buffer)


def test_q4_t16_dense_single_matches_pack8_production_bits(
    t16_selected_library,
    q4_library,
) -> None:
    rng = np.random.default_rng(20260731)
    in_features = 512
    out_features = 32
    raw = make_q4_k_weight(out_features, in_features)
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.4, size=(1, in_features)).astype(np.float32)
    )
    packed = repack_gguf_q4_k_pack8(raw)
    tiles = repack_gguf_q4_k_tile16(raw[None, ...]).tiles

    control = _run_pack8_single(
        x_bf16,
        packed,
        out_features,
        q4_library,
    )
    actual = _run_dense_single(
        gguf_q4_k_t16_dense_single_local32_bf16_bf16_out,
        x_bf16,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )

    np.testing.assert_array_equal(actual, control)
    expected = gguf_quant_gemv(
        _bf16_u16_to_f32(x_bf16),
        raw,
        GGMLQuantizationType.Q4_K,
    )
    np.testing.assert_allclose(
        _bf16_u16_to_f32(actual),
        expected,
        **_TOL,
    )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_q4_t16_dense_single_col4_is_bit_exact(
    t16_selected_library,
) -> None:
    rng = np.random.default_rng(20260819)
    in_features = 1024
    out_features = 512
    raw = make_q4_k_weight(out_features, in_features)
    tiles = repack_gguf_q4_k_tile16(raw[None, ...]).tiles
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.4, size=(1, in_features)).astype(np.float32)
    )
    control = _run_dense_single(
        gguf_q4_k_t16_dense_single_local32_bf16_bf16_out,
        x_bf16,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    candidate = _run_dense_single(
        gguf_q4_k_t16_dense_single_col4_bf16_bf16_out,
        x_bf16,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(candidate, control)


def _run_pack8_dual_rowtile_silu(
    x_dev,
    packed_a,
    packed_b,
    out_features,
    q4_library,
) -> np.ndarray:
    rows, in_features = x_dev.shape
    buffers = []
    try:
        x_buf = malloc(x_dev.nbytes)
        buffers.append(x_buf)
        copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
        packed_buffers = []
        for value in (
            packed_a.qweight,
            packed_a.scales,
            packed_a.mins,
            packed_b.qweight,
            packed_b.scales,
            packed_b.mins,
        ):
            buffer = malloc(value.nbytes)
            buffers.append(buffer)
            packed_buffers.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(value), value.nbytes)
        out_arr = np.zeros((rows, out_features), dtype=np.uint16)
        out_buf = malloc(out_arr.nbytes)
        buffers.append(out_buf)
        gguf_q4_k_pack8_dual_rowtile_silu_bf16_bf16_out(
            x_buf.ptr,
            packed_buffers[0].ptr,
            packed_buffers[1].ptr,
            packed_buffers[2].ptr,
            packed_buffers[3].ptr,
            packed_buffers[4].ptr,
            packed_buffers[5].ptr,
            out_buf.ptr,
            rows,
            in_features,
            out_features,
            threads=64,
            library=q4_library,
        )
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for buffer in reversed(buffers):
            free(buffer)


def _pack8_rowtile_chunks(rows: int) -> tuple[int, ...]:
    """Compose the retained 2-4-row pack8 oracle without a one-row tail."""

    if rows <= 4:
        return (rows,)
    first = 3 if rows == 5 else 4
    return (first, rows - first)


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 7, 8])
def test_q4_t16_dense_rowtiles_match_pack8_production_bits(
    rows: int,
    t16_selected_library,
    q4_library,
) -> None:
    rng = np.random.default_rng(20260804 + rows)
    in_features = 512
    out_features = 32
    raw_a = make_q4_k_weight(out_features, in_features)
    raw_b = np.roll(raw_a, shift=7, axis=0).copy()
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.4, size=(rows, in_features)).astype(np.float32)
    )
    packed_a = repack_gguf_q4_k_pack8(raw_a)
    packed_b = repack_gguf_q4_k_pack8(raw_b)
    tiles_a = repack_gguf_q4_k_tile16(raw_a[None, ...]).tiles
    tiles_b = repack_gguf_q4_k_tile16(raw_b[None, ...]).tiles

    chunks = _pack8_rowtile_chunks(rows)
    single_control = np.concatenate(
        [
            _run_pack8_single(
                chunk,
                packed_a,
                out_features,
                q4_library,
            )
            for chunk in np.split(x_bf16, np.cumsum(chunks)[:-1])
        ],
        axis=0,
    )
    single_actual = _run_dense_single(
        gguf_q4_k_t16_dense_rowtile_bf16_bf16_out,
        x_bf16,
        tiles_a,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    dual_control = np.concatenate(
        [
            _run_pack8_dual_rowtile_silu(
                chunk,
                packed_a,
                packed_b,
                out_features,
                q4_library,
            )
            for chunk in np.split(x_bf16, np.cumsum(chunks)[:-1])
        ],
        axis=0,
    )
    dual_actual = _run_dense_dual_silu(
        x_bf16,
        tiles_a,
        tiles_b,
        out_features,
        np.uint16,
        t16_selected_library,
        fn=gguf_q4_k_t16_dense_dual_rowtile_silu_bf16_bf16_out,
    )
    single_col4 = _run_dense_single(
        gguf_q4_k_t16_dense_rowtile_col4_bf16_bf16_out,
        x_bf16,
        tiles_a,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(single_actual, single_control)
    np.testing.assert_array_equal(dual_actual, dual_control)
    np.testing.assert_array_equal(single_col4, single_control)
    expected_single = gguf_quant_gemv(
        _bf16_u16_to_f32(x_bf16),
        raw_a,
        GGMLQuantizationType.Q4_K,
    )
    np.testing.assert_allclose(
        _bf16_u16_to_f32(single_actual),
        expected_single,
        **_TOL,
    )


@pytest.mark.parametrize("rows", [2, 3, 4, 6, 8])
def test_q4_t16_dense_rowtile16_w2_matches_rowtile8_bits(
    rows: int,
    t16_selected_library,
) -> None:
    rng = np.random.default_rng(20260823 + rows)
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.4, size=(rows, 512)).astype(np.float32)
    )
    raw = make_q4_k_weight(32, 512)
    tiles = repack_gguf_q4_k_tile16(raw[None, ...]).tiles
    control = _run_dense_single(
        gguf_q4_k_t16_dense_rowtile_bf16_bf16_out,
        x_bf16,
        tiles,
        32,
        np.uint16,
        t16_selected_library,
    )
    candidate = _run_dense_single(
        gguf_q4_k_t16_dense_rowtile16_w2_bf16_bf16_out,
        x_bf16,
        tiles,
        32,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(candidate, control)


@pytest.mark.parametrize("rows", [12, 24, 36])
def test_q4_t16_dense_rowtile16_w2_grouped_rows6_matches_repeated_bits(
    rows: int,
    t16_selected_library,
) -> None:
    rng = np.random.default_rng(20260901 + rows)
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.4, size=(rows, 512)).astype(np.float32)
    )
    raw = make_q4_k_weight(32, 512)
    tiles = repack_gguf_q4_k_tile16(raw[None, ...]).tiles
    control = np.concatenate(
        [
            _run_dense_single(
                gguf_q4_k_t16_dense_rowtile16_w2_bf16_bf16_out,
                chunk,
                tiles,
                32,
                np.uint16,
                t16_selected_library,
            )
            for chunk in np.split(x_bf16, rows // 6)
        ],
        axis=0,
    )
    candidate = _run_dense_single(
        gguf_q4_k_t16_dense_rowtile16_w2_grouped_rows6_bf16_bf16_out,
        x_bf16,
        tiles,
        32,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(candidate, control)


def test_q4_t16_dense_rowtile16_w2_grouped_rows8_matches_grouped_rows6_bits(
    t16_selected_library,
) -> None:
    rows = 24
    rng = np.random.default_rng(20260902)
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.4, size=(rows, 512)).astype(np.float32)
    )
    raw = make_q4_k_weight(32, 512)
    tiles = repack_gguf_q4_k_tile16(raw[None, ...]).tiles
    control = _run_dense_single(
        gguf_q4_k_t16_dense_rowtile16_w2_grouped_rows6_bf16_bf16_out,
        x_bf16,
        tiles,
        32,
        np.uint16,
        t16_selected_library,
    )
    candidate = _run_dense_single(
        gguf_q4_k_t16_dense_rowtile16_w2_grouped_rows8_bf16_bf16_out,
        x_bf16,
        tiles,
        32,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(candidate, control)



@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_qmicro_q4_dense_primitives_match_t16_bits(
    rows: int,
    t16_selected_library,
) -> None:
    rng = np.random.default_rng(20260819 + rows)
    in_features = 512
    out_features = 32
    raw = make_q4_k_weight(out_features, in_features)
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.4, size=(rows, in_features)).astype(np.float32)
    )
    control_tiles = repack_gguf_q4_k_tile16(raw[None, ...]).tiles
    candidate_tiles = repack_gguf_q4_k_tile16_qmicro(raw[None, ...]).tiles
    control_fn = (
        gguf_q4_k_t16_dense_single_local32_bf16_bf16_out
        if rows == 1
        else gguf_q4_k_t16_dense_rowtile_bf16_bf16_out
    )
    candidate_fn = (
        gguf_q4_k_qmicro_t16_dense_single_local32_bf16_bf16_out
        if rows == 1
        else gguf_q4_k_qmicro_t16_dense_rowtile_bf16_bf16_out
    )
    control = _run_dense_single(
        control_fn,
        x_bf16,
        control_tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    candidate = _run_dense_single(
        candidate_fn,
        x_bf16,
        candidate_tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(candidate, control)


@pytest.mark.parametrize("rows", [2, 3, 4])
def test_qmicro_q4_dense_dual_rowtile_matches_t16_bits(
    rows: int,
    t16_selected_library,
) -> None:
    rng = np.random.default_rng(20260818 + rows)
    in_features = 512
    out_features = 32
    raw_a = make_q4_k_weight(out_features, in_features)
    raw_b = np.roll(raw_a, shift=11, axis=0).copy()
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.4, size=(rows, in_features)).astype(np.float32)
    )
    control_tiles_a = repack_gguf_q4_k_tile16(raw_a[None, ...]).tiles
    control_tiles_b = repack_gguf_q4_k_tile16(raw_b[None, ...]).tiles
    candidate_tiles_a = repack_gguf_q4_k_tile16_qmicro(raw_a[None, ...]).tiles
    candidate_tiles_b = repack_gguf_q4_k_tile16_qmicro(raw_b[None, ...]).tiles

    control = _run_dense_dual_silu(
        x_bf16,
        control_tiles_a,
        control_tiles_b,
        out_features,
        np.uint16,
        t16_selected_library,
        fn=gguf_q4_k_t16_dense_dual_rowtile_silu_bf16_bf16_out,
    )
    candidate = _run_dense_dual_silu(
        x_bf16,
        candidate_tiles_a,
        candidate_tiles_b,
        out_features,
        np.uint16,
        t16_selected_library,
        fn=gguf_q4_k_qmicro_t16_dense_dual_rowtile_silu_bf16_bf16_out,
    )
    np.testing.assert_array_equal(candidate, control)


@pytest.mark.parametrize("rows", [2, 3, 4])
def test_q4_t16_dense_down_residual_is_bit_exact(
    rows: int,
    t16_selected_library,
) -> None:
    rng = np.random.default_rng(20260806 + rows)
    in_features = 512
    out_features = 32
    raw = make_q4_k_weight(out_features, in_features)
    tiles = repack_gguf_q4_k_tile16(raw[None, ...]).tiles
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.4, size=(rows, in_features)).astype(np.float32)
    )
    residual = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, out_features)).astype(np.float32)
    )
    projected = _run_dense_single(
        gguf_q4_k_t16_dense_rowtile_bf16_bf16_out,
        x_bf16,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    expected = _f32_to_bf16_u16(
        _bf16_u16_to_f32(residual) + _bf16_u16_to_f32(projected)
    )
    candidate = _run_dense_residual(
        selected_t16_mod.gguf_q4_k_t16_dense_rowtile_bf16_residual_bf16_out,
        x_bf16,
        tiles,
        residual,
        out_features,
        t16_selected_library,
    )

    np.testing.assert_array_equal(candidate, expected)


def test_q4_t16_dense_c1_down_residual_is_bit_exact(
    t16_selected_library,
) -> None:
    rng = np.random.default_rng(20260816)
    rows = 1
    in_features = 512
    out_features = 32
    raw = make_q4_k_weight(out_features, in_features)
    tiles = repack_gguf_q4_k_tile16(raw[None, ...]).tiles
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.4, size=(rows, in_features)).astype(np.float32)
    )
    residual = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, out_features)).astype(np.float32)
    )
    projected = _run_dense_single(
        gguf_q4_k_t16_dense_single_local32_bf16_bf16_out,
        x_bf16,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    expected = _f32_to_bf16_u16(
        _bf16_u16_to_f32(residual) + _bf16_u16_to_f32(projected)
    )
    candidate = _run_dense_residual(
        selected_t16_mod.gguf_q4_k_t16_dense_single_local32_bf16_residual_bf16_out,
        x_bf16,
        tiles,
        residual,
        out_features,
        t16_selected_library,
    )

    np.testing.assert_array_equal(candidate, expected)


def _run_dense_single_chunked(
    fn,
    x_dev,
    tiles,
    out_features,
    out_dtype,
    library,
    groups,
) -> np.ndarray:
    """Launch ``fn`` once per ``(chunk_rows, row_base)`` group into shared buffers.

    Mirrors the native c=N rowtile8 chunking in ``launch_gguf_linear``: each
    group writes its own row slice of a shared output buffer, so the composed
    result must match the serial c1 owner row-for-row bit-exactly.
    """

    rows, in_features = x_dev.shape
    x_buf = malloc(x_dev.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
    tiles_buf = malloc(tiles.nbytes)
    copy_host_to_device(tiles_buf, host_array_ptr(tiles), tiles.nbytes)
    out_arr = np.zeros((rows, out_features), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    element_nbytes = np.dtype(out_dtype).itemsize
    try:
        for chunk_rows, row_base in groups:
            fn(
                x_buf.ptr + row_base * in_features * np.dtype(
                    x_dev.dtype
                ).itemsize,
                tiles_buf.ptr,
                out_buf.ptr + row_base * out_features * element_nbytes,
                chunk_rows,
                in_features,
                out_features,
                library=library,
            )
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for buf in (x_buf, tiles_buf, out_buf):
            free(buf)


@pytest.mark.parametrize(
    "rows,groups",
    [
        (9, [(7, 0), (2, 7)]),
        (10, [(8, 0), (2, 8)]),
        (16, [(8, 0), (8, 8)]),
        (17, [(8, 0), (7, 8), (2, 15)]),
    ],
)
def test_q4_t16_dense_c_n_chunked_rowtile_composes_bit_exact_vs_c1(
    rows: int,
    groups: list[tuple[int, int]],
    t16_selected_library,
) -> None:
    """Chunked rowtile8 decomposition is row-for-row bit-exact to serial c1.

    The native c=N path decomposes rows 9..511 into ``_rowtile8_row_chunks``
    groups; each group is an independent rowtile8 launch with row offsets. This
    RED asserts the composed output matches the serial c1 owner exactly.
    """

    rng = np.random.default_rng(20260818 + rows)
    in_features = 512
    out_features = 32
    raw = make_q4_k_weight(out_features, in_features)
    tiles = repack_gguf_q4_k_tile16(raw[None, ...]).tiles
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.4, size=(rows, in_features)).astype(np.float32)
    )
    expected = np.concatenate(
        [
            _run_dense_single(
                gguf_q4_k_t16_dense_single_local32_bf16_bf16_out,
                x_bf16[row : row + 1],
                tiles,
                out_features,
                np.uint16,
                t16_selected_library,
            )
            for row in range(rows)
        ],
        axis=0,
    )
    candidate = _run_dense_single_chunked(
        gguf_q4_k_t16_dense_rowtile_bf16_bf16_out,
        x_bf16,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
        groups,
    )
    np.testing.assert_array_equal(candidate, expected)


@pytest.mark.parametrize("rows", [12, 24, 36])
def test_q5_t16_grouped_rows6_matches_repeated_rows6_bits(
    rows: int,
    t16_selected_library,
) -> None:
    rng = np.random.default_rng(20260901 + rows)
    in_features = 512
    out_features = 32
    raw = make_q5_k_weight(out_features, in_features)
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.4, size=(rows, in_features)).astype(np.float32)
    )
    tiles = repack_gguf_q5_k_tile16(raw[None, ...]).tiles
    repeated = _run_dense_single_chunked(
        gguf_q5_k_t16_gemv_rowtile_bf16_bf16_out,
        x_bf16,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
        [(6, row_base) for row_base in range(0, rows, 6)],
    )
    grouped = _run_dense_single(
        gguf_q5_k_t16_gemv_rowtile_grouped_rows6_bf16_bf16_out,
        x_bf16,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )

    np.testing.assert_array_equal(grouped, repeated)


def test_q5_t16_grouped_rows8_matches_repeated_rows8_bits(
    t16_selected_library,
) -> None:
    grouped_fn = getattr(
        selected_t16_mod,
        "gguf_q5_k_t16_gemv_rowtile_grouped_rows8_bf16_bf16_out",
        None,
    )
    assert callable(grouped_fn)
    rng = np.random.default_rng(20260902)
    rows = 16
    in_features = 512
    out_features = 32
    raw = make_q5_k_weight(out_features, in_features)
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.4, size=(rows, in_features)).astype(np.float32)
    )
    tiles = repack_gguf_q5_k_tile16(raw[None, ...]).tiles
    repeated = _run_dense_single_chunked(
        gguf_q5_k_t16_gemv_rowtile_bf16_bf16_out,
        x_bf16,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
        ((8, 0), (8, 8)),
    )
    grouped = _run_dense_single(
        grouped_fn,
        x_bf16,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(grouped, repeated)


def test_q5_t16_dense_decode_defaults_to_grouped_physical_rows6_launches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.delenv(
        "HIPENGINE_GGUF_Q5_T16_GROUPED_TARGET_ROWS6",
        raising=False,
    )
    monkeypatch.setattr(
        selected_t16_mod,
        "_launch_dense_q5_t16",
        lambda symbol, *args, **kwargs: calls.append((symbol, int(args[3]))),
    )

    with q5_t16_physical_rowtile_session(True):
        gguf_q5_k_t16_gemv_decode_bf16_bf16_out(
            1, 2, 3, 24, 6_144, 5_120
        )

    assert calls == [
        (
            "hipengine_gguf_q5_k_t16_gemv_rowtile_grouped_rows6_"
            "bf16_bf16_out",
            24,
        )
    ]


def test_q5_t16_exact_c7_rows_use_grouped_rows6_prefix_and_strict_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, int, int]] = []
    monkeypatch.delenv(
        "HIPENGINE_GGUF_Q5_T16_GROUPED_TARGET_ROWS6",
        raising=False,
    )
    monkeypatch.delenv(
        "HIPENGINE_GGUF_Q5_T16_GROUPED_ROWS8_C8",
        raising=False,
    )
    monkeypatch.setattr(
        selected_t16_mod,
        "_launch_dense_q5_t16",
        lambda symbol, x, _tiles, out, rows, *_args, **_kwargs: calls.append(
            (str(symbol), int(rows), int(x), int(out))
        ),
    )
    x_ptr = 0x100_000
    out_ptr = 0x200_000
    grouped = selected_t16_mod._Q5_DENSE_ROWTILE_GROUPED_ROWS6_BF16
    rowtile = selected_t16_mod._Q5_DENSE_ROWTILE_BF16

    with (
        q5_t16_physical_rowtile_session(True),
        physical_exact_rowtiles_session(True),
    ):
        gguf_q5_k_t16_gemv_decode_bf16_bf16_out(
            x_ptr, 2, out_ptr, 28, 6_144, 5_120
        )
        assert calls == [
            (grouped, 24, x_ptr, out_ptr),
            (
                rowtile,
                4,
                x_ptr + 24 * 6_144 * 2,
                out_ptr + 24 * 5_120 * 2,
            ),
        ]
        calls.clear()

        gguf_q5_k_t16_gemv_decode_bf16_bf16_out(
            x_ptr, 2, out_ptr, 32, 6_144, 5_120
        )
        assert calls == [
            (
                selected_t16_mod._Q5_DENSE_ROWTILE_GROUPED_ROWS8_BF16,
                32,
                x_ptr,
                out_ptr,
            )
        ]


def test_q5_t16_exact_c8_rows_select_grouped_rows8_with_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setenv("HIPENGINE_GGUF_Q5_T16_GROUPED_ROWS8_C8", "1")
    monkeypatch.setattr(
        selected_t16_mod,
        "_launch_dense_q5_t16",
        lambda symbol, *args, **kwargs: calls.append((str(symbol), int(args[3]))),
    )
    with (
        q5_t16_physical_rowtile_session(True),
        physical_exact_rowtiles_session(True),
    ):
        gguf_q5_k_t16_gemv_decode_bf16_bf16_out(
            1, 2, 3, 32, 6_144, 5_120
        )
        assert calls == [
            (selected_t16_mod._Q5_DENSE_ROWTILE_GROUPED_ROWS8_BF16, 32)
        ]
        calls.clear()
        monkeypatch.setenv("HIPENGINE_GGUF_Q5_T16_GROUPED_ROWS8_C8", "0")
        gguf_q5_k_t16_gemv_decode_bf16_bf16_out(
            1, 2, 3, 32, 6_144, 5_120
        )
    assert calls == [
        (selected_t16_mod._Q5_DENSE_ROWTILE_GROUPED_ROWS6_BF16, 30),
        (selected_t16_mod._Q5_DENSE_ROWTILE_BF16, 2),
    ]


def test_q5_t16_grouped_rows6_policy_has_explicit_repeated_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setenv(
        "HIPENGINE_GGUF_Q5_T16_GROUPED_TARGET_ROWS6",
        "0",
    )
    monkeypatch.setattr(
        selected_t16_mod,
        "_launch_dense_q5_t16",
        lambda symbol, *args, **kwargs: calls.append((symbol, int(args[3]))),
    )

    with q5_t16_physical_rowtile_session(True):
        gguf_q5_k_t16_gemv_decode_bf16_bf16_out(
            1, 2, 3, 12, 6_144, 5_120
        )

    assert calls == [
        (selected_t16_mod._Q5_DENSE_ROWTILE_BF16, 6),
        (selected_t16_mod._Q5_DENSE_ROWTILE_BF16, 6),
    ]


def test_q5_t16_dense_decode_uses_request_scoped_physical_rowtile(
    monkeypatch,
) -> None:
    symbols: list[str] = []
    monkeypatch.setattr(
        selected_t16_mod,
        "_launch_dense_q5_t16",
        lambda symbol, *args, **kwargs: symbols.append(symbol),
    )

    gguf_q5_k_t16_gemv_decode_bf16_bf16_out(1, 2, 3, 6, 6_144, 5_120)
    with q5_t16_physical_rowtile_session(True):
        gguf_q5_k_t16_gemv_decode_bf16_bf16_out(1, 2, 3, 6, 6_144, 5_120)
        gguf_q5_k_t16_gemv_decode_bf16_bf16_out(1, 2, 3, 8, 6_144, 5_120)
        with physical_exact_rowtiles_session(True):
            gguf_q5_k_t16_gemv_decode_bf16_bf16_out(
                1, 2, 3, 8, 6_144, 5_120
            )
        gguf_q5_k_t16_gemv_decode_bf16_bf16_out(1, 2, 3, 3, 6_144, 5_120)

    assert symbols == [
        selected_t16_mod._Q5_DENSE_DIRECT_BF16,
        selected_t16_mod._Q5_DENSE_ROWTILE_BF16,
        selected_t16_mod._Q5_DENSE_DIRECT_BF16,
        selected_t16_mod._Q5_DENSE_ROWTILE_BF16,
        selected_t16_mod._Q5_DENSE_DIRECT_BF16,
    ]


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
def test_q5_t16_dense_decode_and_rowtile_match_selected_production_bits(
    rows: int,
    t16_selected_library,
) -> None:
    rng = np.random.default_rng(20260805 + rows)
    in_features = 512
    out_features = 32
    raw = make_q5_k_weight(out_features, in_features)
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.4, size=(rows, in_features)).astype(np.float32)
    )
    tiles = repack_gguf_q5_k_tile16(raw[None, ...]).tiles
    selected = np.zeros(rows, dtype=np.int64)

    control = _run_direct_single(
        gguf_q5_k_t16_selected_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    candidate = _run_dense_single(
        (
            gguf_q5_k_t16_gemv_decode_bf16_bf16_out
            if rows == 1
            else gguf_q5_k_t16_gemv_rowtile_bf16_bf16_out
        ),
        x_bf16,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )

    np.testing.assert_array_equal(candidate, control)
    expected = gguf_quant_gemv(
        _bf16_u16_to_f32(x_bf16),
        raw,
        GGMLQuantizationType.Q5_K,
    )
    np.testing.assert_allclose(
        _bf16_u16_to_f32(candidate),
        expected,
        **_TOL,
    )


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 7, 8])
def test_q5_t16_dense_rowtile_col8_matches_col4_bits(
    rows: int,
    t16_selected_library,
) -> None:
    rng = np.random.default_rng(20260824 + rows)
    in_features = 512
    out_features = 32
    raw = make_q5_k_weight(out_features, in_features)
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.4, size=(rows, in_features)).astype(np.float32)
    )
    tiles = repack_gguf_q5_k_tile16(raw[None, ...]).tiles
    control = _run_dense_single(
        gguf_q5_k_t16_gemv_rowtile_bf16_bf16_out,
        x_bf16,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    candidate = _run_dense_single(
        gguf_q5_k_t16_gemv_rowtile_col8_bf16_bf16_out,
        x_bf16,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )

    np.testing.assert_array_equal(candidate, control)


def test_q5_t16_dense_tile8_matches_production_bits(
    t16_selected_library,
) -> None:
    rng = np.random.default_rng(0x38A58)
    rows, in_features, out_features = 1, 512, 32
    raw = make_q5_k_weight(out_features, in_features)
    tiles = repack_gguf_q5_k_tile16(raw[None, ...]).tiles
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.4, size=(rows, in_features)).astype(np.float32)
    )
    control = _run_dense_single(
        gguf_q5_k_t16_gemv_decode_bf16_bf16_out,
        x_bf16,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    candidate = _run_dense_single(
        gguf_q5_k_t16_gemv_decode_tile8_bf16_bf16_out,
        x_bf16,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )

    np.testing.assert_array_equal(candidate, control)
    expected = gguf_quant_gemv(
        _bf16_u16_to_f32(x_bf16),
        raw,
        GGMLQuantizationType.Q5_K,
    )
    np.testing.assert_allclose(
        _bf16_u16_to_f32(candidate),
        expected,
        **_TOL,
    )


def _run_direct_dual_silu_q8_dp4a(
    x_dev,
    selected,
    ta,
    tb,
    out_features,
    out_dtype,
    t16_library,
    q4_library,
) -> np.ndarray:
    rows = int(selected.size)
    in_features = x_dev.shape[1]
    x_buf = malloc(x_dev.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
    sel_buf = malloc(selected.nbytes)
    copy_host_to_device(sel_buf, host_array_ptr(selected), selected.nbytes)
    ta_buf = malloc(ta.nbytes)
    copy_host_to_device(ta_buf, host_array_ptr(ta), ta.nbytes)
    tb_buf = malloc(tb.nbytes)
    copy_host_to_device(tb_buf, host_array_ptr(tb), tb.nbytes)
    xq_buf = malloc(x_dev.shape[0] * (in_features // 32) * 36)
    out_arr = np.zeros((rows, out_features), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    try:
        gguf_q4_k_quantize_bf16_q8_1(
            x_buf.ptr,
            xq_buf.ptr,
            x_dev.shape[0],
            in_features,
            library=q4_library,
        )
        gguf_q4_k_t16_selected_dual_silu_q8_1_dp4a_gemv_bf16_bf16_out(
            xq_buf.ptr,
            sel_buf.ptr,
            ta_buf.ptr,
            tb_buf.ptr,
            out_buf.ptr,
            x_dev.shape[0],
            rows,
            ta.shape[0],
            in_features,
            out_features,
            library=t16_library,
        )
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for buf in (x_buf, sel_buf, ta_buf, tb_buf, xq_buf, out_buf):
            free(buf)


def _run_direct_dual_silu_q8x2_dp4a(
    x_dev,
    selected,
    ta,
    tb,
    out_features,
    out_dtype,
    t16_library,
    q4_library,
) -> np.ndarray:
    rows = int(selected.size)
    in_features = x_dev.shape[1]
    x_buf = malloc(x_dev.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
    sel_buf = malloc(selected.nbytes)
    copy_host_to_device(sel_buf, host_array_ptr(selected), selected.nbytes)
    ta_buf = malloc(ta.nbytes)
    copy_host_to_device(ta_buf, host_array_ptr(ta), ta.nbytes)
    tb_buf = malloc(tb.nbytes)
    copy_host_to_device(tb_buf, host_array_ptr(tb), tb.nbytes)
    xq_buf = malloc(2 * x_dev.shape[0] * (in_features // 32) * 36)
    out_arr = np.zeros((rows, out_features), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    try:
        gguf_q4_k_quantize_bf16_q8_1x2(
            x_buf.ptr,
            xq_buf.ptr,
            x_dev.shape[0],
            in_features,
            library=q4_library,
        )
        gguf_q4_k_t16_selected_dual_silu_q8_1x2_dp4a_gemv_bf16_bf16_out(
            xq_buf.ptr,
            sel_buf.ptr,
            ta_buf.ptr,
            tb_buf.ptr,
            out_buf.ptr,
            x_dev.shape[0],
            rows,
            ta.shape[0],
            in_features,
            out_features,
            library=t16_library,
        )
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for buf in (x_buf, sel_buf, ta_buf, tb_buf, xq_buf, out_buf):
            free(buf)


def _run_dense_dual_silu_q8x2_dp4a(
    fn,
    x_dev,
    ta,
    tb,
    out_features,
    out_dtype,
    t16_library,
    q4_library,
) -> np.ndarray:
    rows, in_features = x_dev.shape
    x_buf = malloc(x_dev.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
    ta_buf = malloc(ta.nbytes)
    copy_host_to_device(ta_buf, host_array_ptr(ta), ta.nbytes)
    tb_buf = malloc(tb.nbytes)
    copy_host_to_device(tb_buf, host_array_ptr(tb), tb.nbytes)
    xq_buf = malloc(2 * rows * (in_features // 32) * 36)
    out_arr = np.zeros((rows, out_features), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    try:
        gguf_q4_k_quantize_bf16_q8_1x2(
            x_buf.ptr,
            xq_buf.ptr,
            rows,
            in_features,
            library=q4_library,
        )
        fn(
            xq_buf.ptr,
            ta_buf.ptr,
            tb_buf.ptr,
            out_buf.ptr,
            rows,
            in_features,
            out_features,
            library=t16_library,
        )
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for buf in (x_buf, ta_buf, tb_buf, xq_buf, out_buf):
            free(buf)


def _run_direct_single(
    fn, x_dev, selected, tiles, out_features, out_dtype, library, **kwargs
) -> np.ndarray:
    rows = int(selected.size)
    in_features = x_dev.shape[1]
    x_buf = malloc(x_dev.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
    sel_buf = malloc(selected.nbytes)
    copy_host_to_device(sel_buf, host_array_ptr(selected), selected.nbytes)
    w_buf = malloc(tiles.nbytes)
    copy_host_to_device(w_buf, host_array_ptr(tiles), tiles.nbytes)
    out_arr = np.zeros((rows, out_features), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    try:
        fn(
            x_buf.ptr,
            sel_buf.ptr,
            w_buf.ptr,
            out_buf.ptr,
            x_dev.shape[0],
            rows,
            tiles.shape[0],
            in_features,
            out_features,
            library=library,
            **kwargs,
        )
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for buf in (x_buf, sel_buf, w_buf, out_buf):
            free(buf)


def _run_weighted_sum(
    values: np.ndarray,
    weights: np.ndarray,
    library,
) -> np.ndarray:
    values = np.ascontiguousarray(values, dtype=np.uint16)
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    out = np.empty(values.shape[1], dtype=np.uint16)
    values_buf = malloc(values.nbytes)
    weights_buf = malloc(weights.nbytes)
    out_buf = malloc(out.nbytes)
    copy_host_to_device(values_buf, host_array_ptr(values), values.nbytes)
    copy_host_to_device(weights_buf, host_array_ptr(weights), weights.nbytes)
    try:
        weighted_sum_out_bf16_f32w(
            values_buf.ptr,
            weights_buf.ptr,
            out_buf.ptr,
            values.shape[0],
            values.shape[1],
            library=library,
        )
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
        return out
    finally:
        for buf in (values_buf, weights_buf, out_buf):
            free(buf)


def _run_direct_parallel_weighted(
    fn,
    x_dev: np.ndarray,
    selected: np.ndarray,
    tiles: np.ndarray,
    weights: np.ndarray,
    out_features: int,
    library,
) -> tuple[np.ndarray, np.ndarray, int]:
    rows = int(selected.size)
    in_features = x_dev.shape[1]
    x_buf = malloc(x_dev.nbytes)
    selected_buf = malloc(selected.nbytes)
    tiles_buf = malloc(tiles.nbytes)
    out = np.empty((rows, out_features), dtype=np.uint16)
    out_buf = malloc(out.nbytes)
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    weights_buf = malloc(weights.nbytes)
    routed = np.empty(out_features, dtype=np.uint16)
    routed_buf = malloc(routed.nbytes)
    counter = np.zeros(out_features // 16, dtype=np.int32)
    counter_buf = malloc(counter.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
    copy_host_to_device(selected_buf, host_array_ptr(selected), selected.nbytes)
    copy_host_to_device(tiles_buf, host_array_ptr(tiles), tiles.nbytes)
    copy_host_to_device(weights_buf, host_array_ptr(weights), weights.nbytes)
    copy_host_to_device(counter_buf, host_array_ptr(counter), counter.nbytes)
    try:
        fn(
            x_buf.ptr,
            selected_buf.ptr,
            tiles_buf.ptr,
            out_buf.ptr,
            weights_buf.ptr,
            routed_buf.ptr,
            counter_buf.ptr,
            x_dev.shape[0],
            rows,
            tiles.shape[0],
            in_features,
            out_features,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
        copy_device_to_host(host_array_ptr(routed), routed_buf, routed.nbytes)
        copy_device_to_host(host_array_ptr(counter), counter_buf, counter.nbytes)
        return out, routed, int(np.count_nonzero(counter))
    finally:
        for buf in (
            x_buf,
            selected_buf,
            tiles_buf,
            out_buf,
            weights_buf,
            routed_buf,
            counter_buf,
        ):
            free(buf)


def _run_direct_single_q8_dp4a(
    fn,
    x_dev,
    selected,
    tiles,
    out_features,
    out_dtype,
    t16_library,
    q4_library,
) -> np.ndarray:
    rows = int(selected.size)
    in_features = x_dev.shape[1]
    x_buf = malloc(x_dev.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
    sel_buf = malloc(selected.nbytes)
    copy_host_to_device(sel_buf, host_array_ptr(selected), selected.nbytes)
    w_buf = malloc(tiles.nbytes)
    copy_host_to_device(w_buf, host_array_ptr(tiles), tiles.nbytes)
    xq_buf = malloc(x_dev.shape[0] * (in_features // 32) * 36)
    out_arr = np.zeros((rows, out_features), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    try:
        gguf_q4_k_quantize_bf16_q8_1(
            x_buf.ptr,
            xq_buf.ptr,
            x_dev.shape[0],
            in_features,
            library=q4_library,
        )
        fn(
            xq_buf.ptr,
            sel_buf.ptr,
            w_buf.ptr,
            out_buf.ptr,
            x_dev.shape[0],
            rows,
            tiles.shape[0],
            in_features,
            out_features,
            library=t16_library,
        )
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for buf in (x_buf, sel_buf, w_buf, xq_buf, out_buf):
            free(buf)


def _run_raw_direct_dual(x_dev, selected, qa, qb, out_features, library) -> tuple[np.ndarray, np.ndarray]:
    rows = int(selected.size)
    in_features = x_dev.shape[1]
    x_buf = malloc(x_dev.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_dev), x_dev.nbytes)
    sel_buf = malloc(selected.nbytes)
    copy_host_to_device(sel_buf, host_array_ptr(selected), selected.nbytes)
    qa_buf = malloc(qa.nbytes)
    copy_host_to_device(qa_buf, host_array_ptr(qa), qa.nbytes)
    qb_buf = malloc(qb.nbytes)
    copy_host_to_device(qb_buf, host_array_ptr(qb), qb.nbytes)
    out_a = np.zeros((rows, out_features), dtype=np.uint16)
    out_b = np.zeros((rows, out_features), dtype=np.uint16)
    out_a_buf = malloc(out_a.nbytes)
    out_b_buf = malloc(out_b.nbytes)
    try:
        gguf_q4_k_selected_dual_gemv_bf16_bf16_out(
            x_buf.ptr,
            sel_buf.ptr,
            qa_buf.ptr,
            qb_buf.ptr,
            out_a_buf.ptr,
            out_b_buf.ptr,
            x_dev.shape[0],
            rows,
            qa.shape[0],
            in_features,
            out_features,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out_a), out_a_buf, out_a.nbytes)
        copy_device_to_host(host_array_ptr(out_b), out_b_buf, out_b.nbytes)
        return out_a, out_b
    finally:
        for buf in (x_buf, sel_buf, qa_buf, qb_buf, out_a_buf, out_b_buf):
            free(buf)


_TOL = dict(atol=1.0e-3, rtol=1.0e-2)

_EXPERT_LAYOUTS = [
    pytest.param([8], id="single-expert-rows=8"),
    pytest.param([1], id="single-row"),
    pytest.param([3, 5], id="two-uneven"),
    pytest.param([0, 8], id="empty-start"),
    pytest.param([4, 0, 4], id="empty-middle"),
    pytest.param([1, 1, 1, 1, 1, 1, 1, 1], id="qwen35moe-top_k=8"),
]


def test_p9_h3d_registry_keys_resolve() -> None:
    register_gguf_t16_selected_gemv_kernels()
    for quant, variants in {
        "gguf_q4_k_t16_v1": (
            "selected_dual_t16_gemv_decode_compact_bf16_bf16_out",
            "selected_dual_t16_gemv_decode_compact_fp16_fp16_out",
            "selected_dual_t16_gemv_decode_bf16_bf16_out",
            "selected_dual_t16_gemv_decode_fp16_fp16_out",
            "selected_dual_t16_natural_tile8_parallel_gemv_decode_bf16_bf16_out",
            "selected_dual_t16_natural_tile8_parallel_silu_gemv_decode_bf16_bf16_out",
            "selected_dual_t16_natural_tile8_parallel_silu_paircoeff_gemv_decode_bf16_bf16_out",
            "selected_dual_t16_natural_tile8_parallel_silu_pairq_gemv_decode_bf16_bf16_out",
            "selected_dual_t16_grouped_smallm_bf16_bf16_out",
            "selected_dual_t16_silu_gemv_decode_bf16_bf16_out",
            "selected_dual_t16_q8_1_dp4a_gemv_decode_bf16_bf16_out",
            "selected_dual_t16_silu_q8_1_dp4a_gemv_decode_bf16_bf16_out",
            "selected_t16_gemv_decode_compact_bf16_bf16_out",
            "selected_t16_gemv_decode_compact_fp16_fp16_out",
            "selected_t16_gemv_decode_bf16_bf16_out",
            "selected_t16_gemv_decode_fp16_fp16_out",
            "selected_t16_grouped_smallm_bf16_bf16_out",
        ),
        "gguf_q5_k_t16_v1": (
            "selected_t16_gemv_decode_compact_bf16_bf16_out",
            "selected_t16_gemv_decode_compact_fp16_fp16_out",
            "selected_t16_gemv_decode_bf16_bf16_out",
            "selected_t16_gemv_decode_fp16_fp16_out",
            "selected_t16_q8_1_dp4a_gemv_decode_bf16_bf16_out",
        ),
        "gguf_q6_k_t16_v1": (
            "selected_t16_gemv_decode_compact_bf16_bf16_out",
            "selected_t16_gemv_decode_compact_fp16_fp16_out",
            "selected_t16_gemv_decode_bf16_bf16_out",
            "selected_t16_gemv_decode_fp16_fp16_out",
            "selected_t16_grouped_smallm_bf16_bf16_out",
        ),
    }.items():
        for variant in variants:
            assert resolve(backend="hip_gfx1100", layer="moe_linear", quant=quant, variant=variant) is not None
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q4_k_t16_v1",
        variant="dense_rowtile_bf16_bf16_out",
    ) is gguf_q4_k_t16_dense_rowtile_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q4_k_t16_v1",
        variant="dense_single_col4_bf16_bf16_out",
    ) is gguf_q4_k_t16_dense_single_col4_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear+residual",
        quant="gguf_q4_k_t16_v1",
        variant="dense_rowtile_bf16_residual_bf16_out",
    ) is selected_t16_mod.gguf_q4_k_t16_dense_rowtile_bf16_residual_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear_pair_silu",
        quant="gguf_q4_k_t16_v1",
        variant="dense_dual_rowtile_bf16_bf16_out",
    ) is gguf_q4_k_t16_dense_dual_rowtile_silu_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q5_k_t16_v1",
        variant="t16_gemv_decode_bf16_bf16_out",
    ) is gguf_q5_k_t16_gemv_decode_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q5_k_t16_v1",
        variant="t16_gemv_decode_tile8_bf16_bf16_out",
    ) is gguf_q5_k_t16_gemv_decode_tile8_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q5_k_t16_v1",
        variant="t16_gemv_rowtile_bf16_bf16_out",
    ) is gguf_q5_k_t16_gemv_rowtile_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q5_k_t16_v1",
        variant="t16_gemv_rowtile_grouped_rows6_bf16_bf16_out",
    ) is gguf_q5_k_t16_gemv_rowtile_grouped_rows6_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q4_k_t16_v1",
        variant="dense_rowtile_col4_bf16_bf16_out",
    ) is gguf_q4_k_t16_dense_rowtile_col4_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_q4_k_t16_dual_interleaved_v1",
        variant=(
            "selected_dual_t16_natural_tile8_parallel_silu_"
            "gemv_decode_bf16_bf16_out"
        ),
    ) is (
        gguf_q4_k_t16_selected_dual_interleaved_natural_tile8_parallel_silu_gemv_bf16_bf16_out
    )
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_q4_k_t16_dual_interleaved_v1",
        variant=(
            "selected_dual_t16_natural_tile8_parallel_silu_halfdot_"
            "gemv_decode_bf16_bf16_out"
        ),
    ) is (
        gguf_q4_k_t16_selected_dual_interleaved_natural_tile8_parallel_silu_halfdot_gemv_bf16_bf16_out
    )


def test_p9_h3d_build_plan_is_dry_run_safe() -> None:
    plan = plan_gguf_t16_selected_gemv_build()
    assert plan.output_path.name == "gguf_t16_selected_gemv.so"
    assert plan.sources[0].name == "gguf_t16_selected_gemv.hip"


def test_p9_h3d_wrappers_validate_args() -> None:
    with pytest.raises(ValueError, match="rows in 2..8"):
        gguf_q4_k_t16_dense_rowtile_bf16_bf16_out(0, 0, 0, 1, 256, 16)
    with pytest.raises(ValueError, match="rows in 2..8"):
        gguf_q4_k_t16_dense_dual_rowtile_silu_bf16_bf16_out(
            0, 0, 0, 0, 9, 256, 16
        )
    with pytest.raises(ValueError, match=r"rows in \{2,3,4,6,8\}"):
        gguf_q4_k_t16_dense_rowtile16_w2_bf16_bf16_out(
            0, 0, 0, 5, 256, 16
        )
    with pytest.raises(ValueError, match="rows in 2..4"):
        gguf_q4_k_qmicro_t16_dense_rowtile_bf16_bf16_out(
            0, 0, 0, 5, 256, 16
        )
    with pytest.raises(ValueError, match="rows in 2..8"):
        gguf_q5_k_t16_gemv_rowtile_bf16_bf16_out(0, 0, 0, 1, 256, 16)
    with pytest.raises(ValueError, match="compact_rows must be positive"):
        gguf_q4_k_t16_selected_dual_gemv_decode_compact_bf16_bf16_out(0, 0, 0, 0, 0, 0, 256, 16, 16, 1)
    with pytest.raises(ValueError, match="block size 256"):
        gguf_q5_k_t16_selected_gemv_decode_compact_bf16_bf16_out(0, 0, 0, 0, 1, 255, 16, 1)
    with pytest.raises(ValueError, match="multiple of 16"):
        gguf_q6_k_t16_selected_gemv_decode_compact_bf16_bf16_out(0, 0, 0, 0, 1, 256, 8, 1)
    with pytest.raises(ValueError, match="out_features_b must be a multiple of 16"):
        gguf_q4_k_t16_selected_dual_gemv_decode_compact_bf16_bf16_out(0, 0, 0, 0, 0, 1, 256, 16, 8, 1)
    with pytest.raises(ValueError, match="rows must be divisible by x_rows"):
        gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out(0, 0, 0, 0, 0, 3, 8, 1, 256, 16)
    with pytest.raises(ValueError, match="rows must be divisible by x_rows"):
        gguf_q4_k_t16_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out(0, 0, 0, 0, 0, 0, 3, 8, 1, 256, 16)
    with pytest.raises(ValueError, match="rows must be divisible by x_rows"):
        gguf_q4_k_t16_selected_dual_silu_q8_1_dp4a_gemv_bf16_bf16_out(0, 0, 0, 0, 0, 3, 8, 1, 256, 16)
    with pytest.raises(ValueError, match="rows must be divisible by x_rows"):
        gguf_q5_k_t16_selected_q8_1_dp4a_gemv_bf16_bf16_out(0, 0, 0, 0, 3, 8, 1, 256, 16)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("counts", _EXPERT_LAYOUTS)
@pytest.mark.parametrize(
    "in_features,out_features_a,out_features_b",
    [(256, 16, 16), (512, 256, 256), (2048, 512, 512)],
)
def test_p9_h3d_q4_t16_dual_bf16_matches_cpu_oracle(
    counts: list[int], in_features: int, out_features_a: int, out_features_b: int, t16_selected_library,
) -> None:
    num_experts = len(counts)
    expert_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    compact_rows = int(expert_start[-1])
    rng = np.random.default_rng(num_experts * 101 + in_features + out_features_a + out_features_b + compact_rows)
    qa = _stack_experts(make_q4_k_weight, out_features_a, in_features, num_experts, seed=1)
    qb = _stack_experts(make_q4_k_weight, out_features_b, in_features, num_experts, seed=2)
    ta = repack_gguf_q4_k_tile16(qa).tiles
    tb = repack_gguf_q4_k_tile16(qb).tiles
    x = rng.normal(0.0, 0.3, size=(compact_rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)

    actual = _run_dual(
        gguf_q4_k_t16_selected_dual_gemv_decode_compact_bf16_bf16_out,
        x_bf16,
        expert_start,
        ta,
        tb,
        out_features_a,
        out_features_b,
        np.uint16,
        t16_selected_library,
    )

    expected = _expected_dual(x_ref, expert_start, qa, qb, out_features_a, out_features_b)
    expected_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected))
    np.testing.assert_allclose(_bf16_u16_to_f32(actual), expected_bf16, **_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_p9_h3d_q4_t16_direct_dual_bf16_matches_cpu_oracle(t16_selected_library) -> None:
    x_rows = 2
    selected = np.array([2, 0, 1, 1, 2, 0], dtype=np.int64)
    in_features, out_features = 512, 256
    num_experts = 3
    rng = np.random.default_rng(451)
    qa = _stack_experts(make_q4_k_weight, out_features, in_features, num_experts, seed=13)
    qb = _stack_experts(make_q4_k_weight, out_features, in_features, num_experts, seed=17)
    ta = repack_gguf_q4_k_tile16(qa).tiles
    tb = repack_gguf_q4_k_tile16(qb).tiles
    x = rng.normal(0.0, 0.3, size=(x_rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)

    actual_a, actual_b = _run_direct_dual(
        gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        ta,
        tb,
        out_features,
        np.uint16,
        t16_selected_library,
    )

    expected = _expected_direct_dual(x_ref, selected, qa, qb, out_features, out_features)
    expected_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected))
    actual = np.concatenate([_bf16_u16_to_f32(actual_a), _bf16_u16_to_f32(actual_b)], axis=1)
    np.testing.assert_allclose(actual, expected_bf16, **_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_laguna_t16_natural_selected_decode_matches_production_bits(
    t16_selected_library,
    combine_library,
) -> None:
    """Natural Laguna gate/up and mixed down shapes preserve exact BF16 bits."""

    x_rows, rows, num_experts = 1, 10, 3
    selected = np.array([2, 0, 1, 1, 2, 0, 2, 1, 0, 2], dtype=np.int64)
    rng = np.random.default_rng(20260728)

    gate_in, gate_out = 3072, 1024
    gate_a = _stack_experts(
        make_q4_k_weight,
        gate_out,
        gate_in,
        num_experts,
        seed=71,
    )
    gate_b = _stack_experts(
        make_q4_k_weight,
        gate_out,
        gate_in,
        num_experts,
        seed=73,
    )
    gate_tiles_a = repack_gguf_q4_k_tile16(gate_a).tiles
    gate_tiles_b = repack_gguf_q4_k_tile16(gate_b).tiles
    gate_x = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(x_rows, gate_in)).astype(np.float32)
    )
    gate_ref = _run_direct_dual(
        gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
        gate_x,
        selected,
        gate_tiles_a,
        gate_tiles_b,
        gate_out,
        np.uint16,
        t16_selected_library,
    )
    gate_actual = _run_direct_dual(
        gguf_q4_k_t16_selected_dual_natural_gemv_bf16_bf16_out,
        gate_x,
        selected,
        gate_tiles_a,
        gate_tiles_b,
        gate_out,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(gate_actual[0], gate_ref[0])
    np.testing.assert_array_equal(gate_actual[1], gate_ref[1])
    gate_tile8 = _run_direct_dual(
        gguf_q4_k_t16_selected_dual_natural_tile8_gemv_bf16_bf16_out,
        gate_x,
        selected,
        gate_tiles_a,
        gate_tiles_b,
        gate_out,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(gate_tile8[0], gate_ref[0])
    np.testing.assert_array_equal(gate_tile8[1], gate_ref[1])
    gate_tile8_parallel = _run_direct_dual(
        gguf_q4_k_t16_selected_dual_natural_tile8_parallel_gemv_bf16_bf16_out,
        gate_x,
        selected,
        gate_tiles_a,
        gate_tiles_b,
        gate_out,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(gate_tile8_parallel[0], gate_tile8[0])
    np.testing.assert_array_equal(gate_tile8_parallel[1], gate_tile8[1])
    gate_tile8_parallel_silu = _run_direct_dual_silu(
        gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_gemv_bf16_bf16_out,
        gate_x,
        selected,
        gate_tiles_a,
        gate_tiles_b,
        gate_out,
        np.uint16,
        t16_selected_library,
    )
    gate_f32 = _bf16_u16_to_f32(gate_tile8_parallel[0])
    up_f32 = _bf16_u16_to_f32(gate_tile8_parallel[1])
    with np.errstate(over="ignore"):
        expected_silu = _f32_to_bf16_u16(
            gate_f32 * (1.0 / (1.0 + np.exp(-gate_f32))) * up_f32
        )
    np.testing.assert_array_equal(gate_tile8_parallel_silu, expected_silu)
    gate_tile8_parallel_silu_paircoeff = _run_direct_dual_silu(
        gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_paircoeff_gemv_bf16_bf16_out,
        gate_x,
        selected,
        gate_tiles_a,
        gate_tiles_b,
        gate_out,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(
        gate_tile8_parallel_silu_paircoeff,
        gate_tile8_parallel_silu,
    )
    gate_tile8_parallel_silu_halfdot = _run_direct_dual_silu(
        gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_halfdot_gemv_bf16_bf16_out,
        gate_x,
        selected,
        gate_tiles_a,
        gate_tiles_b,
        gate_out,
        np.uint16,
        t16_selected_library,
    )
    halfdot_f32 = _bf16_u16_to_f32(gate_tile8_parallel_silu_halfdot)
    exact_f32 = _bf16_u16_to_f32(gate_tile8_parallel_silu_paircoeff)
    assert np.isfinite(halfdot_f32).all()
    relative_error = np.abs(halfdot_f32 - exact_f32) / np.maximum(
        np.abs(exact_f32),
        np.float32(0.25),
    )
    assert np.quantile(relative_error, 0.99) <= 0.05
    gate_tile8_parallel_silu_pairq = _run_direct_dual_silu(
        gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_pairq_gemv_bf16_bf16_out,
        gate_x,
        selected,
        gate_tiles_a,
        gate_tiles_b,
        gate_out,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(
        gate_tile8_parallel_silu_pairq,
        gate_tile8_parallel_silu_paircoeff,
    )
    gate_tiles_interleaved = interleave_gguf_q4_k_tile16_dual(
        gate_tiles_a,
        gate_tiles_b,
    )
    gate_tile8_parallel_silu_interleaved = _run_direct_dual_silu(
        gguf_q4_k_t16_selected_dual_interleaved_natural_tile8_parallel_silu_gemv_bf16_bf16_out,
        gate_x,
        selected,
        gate_tiles_interleaved,
        gate_tiles_interleaved,
        gate_out,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(
        gate_tile8_parallel_silu_interleaved,
        gate_tile8_parallel_silu_paircoeff,
    )
    gate_tile8_parallel_silu_interleaved_halfdot = _run_direct_dual_silu(
        gguf_q4_k_t16_selected_dual_interleaved_natural_tile8_parallel_silu_halfdot_gemv_bf16_bf16_out,
        gate_x,
        selected,
        gate_tiles_interleaved,
        gate_tiles_interleaved,
        gate_out,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(
        gate_tile8_parallel_silu_interleaved_halfdot,
        gate_tile8_parallel_silu_halfdot,
    )
    small_gate_x = np.concatenate(
        (
            gate_x,
            _f32_to_bf16_u16(
                _bf16_u16_to_f32(gate_x) * np.float32(-0.625)
            ),
        ),
        axis=0,
    )
    small_selected = np.tile(selected, 2)
    small_gate_ref = _run_direct_dual(
        gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
        small_gate_x,
        small_selected,
        gate_tiles_a,
        gate_tiles_b,
        gate_out,
        np.uint16,
        t16_selected_library,
    )
    small_gate_f32 = _bf16_u16_to_f32(small_gate_ref[0])
    small_up_f32 = _bf16_u16_to_f32(small_gate_ref[1])
    with np.errstate(over="ignore"):
        small_expected_silu = _f32_to_bf16_u16(
            small_gate_f32
            * (1.0 / (1.0 + np.exp(-small_gate_f32)))
            * small_up_f32
        )
    small_interleaved = _run_direct_dual_silu(
        gguf_q4_k_t16_selected_dual_interleaved_natural_tile8_parallel_silu_gemv_bf16_bf16_out,
        small_gate_x,
        small_selected,
        gate_tiles_interleaved,
        gate_tiles_interleaved,
        gate_out,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(small_interleaved, small_expected_silu)
    dense_tiles_a = repack_gguf_q4_k_tile16(gate_a[0:1]).tiles
    dense_tiles_b = repack_gguf_q4_k_tile16(gate_b[0:1]).tiles
    dense_actual = _run_dense_dual_silu(
        gate_x,
        dense_tiles_a,
        dense_tiles_b,
        gate_out,
        np.uint16,
        t16_selected_library,
    )
    dense_interleaved = interleave_gguf_q4_k_tile16_dual(
        dense_tiles_a,
        dense_tiles_b,
    )
    dense_interleaved_actual = _run_dense_dual_interleaved_silu(
        gguf_q4_k_t16_dense_dual_interleaved_tile2_local32_silu_bf16_bf16_out,
        gate_x,
        dense_interleaved,
        gate_out,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(
        dense_interleaved_actual,
        dense_actual,
    )
    dense_expected_pair = _expected_direct_dual(
        _bf16_u16_to_f32(gate_x),
        np.asarray([0], dtype=np.int64),
        gate_a,
        gate_b,
        gate_out,
        gate_out,
    )
    dense_gate = _bf16_u16_to_f32(
        _f32_to_bf16_u16(dense_expected_pair[:, :gate_out])
    )
    dense_up = _bf16_u16_to_f32(
        _f32_to_bf16_u16(dense_expected_pair[:, gate_out:])
    )
    with np.errstate(over="ignore"):
        dense_expected = _bf16_u16_to_f32(
            _f32_to_bf16_u16(
                dense_gate *
                (1.0 / (1.0 + np.exp(-dense_gate))) *
                dense_up
            )
        )
    np.testing.assert_allclose(
        _bf16_u16_to_f32(dense_actual),
        dense_expected,
        **_TOL,
    )

    down_in, down_out = 1024, 3072
    down_x = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, down_in)).astype(np.float32)
    )
    q4_down = _stack_experts(
        make_q4_k_weight,
        down_out,
        down_in,
        num_experts,
        seed=79,
    )
    q4_tiles = repack_gguf_q4_k_tile16(q4_down).tiles
    q4_ref = _run_direct_single(
        gguf_q4_k_t16_selected_gemv_bf16_bf16_out,
        down_x,
        selected,
        q4_tiles,
        down_out,
        np.uint16,
        t16_selected_library,
    )
    q4_actual = _run_direct_single(
        gguf_q4_k_t16_selected_natural_gemv_bf16_bf16_out,
        down_x,
        selected,
        q4_tiles,
        down_out,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(q4_actual, q4_ref)
    q4_parallel = _run_direct_single(
        gguf_q4_k_t16_selected_natural_parallel_gemv_bf16_bf16_out,
        down_x,
        selected,
        q4_tiles,
        down_out,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(q4_parallel, q4_actual)
    route_weights = rng.normal(0.1, 0.3, size=rows).astype(np.float32)
    q4_weighted_ref = _run_weighted_sum(
        q4_parallel,
        route_weights,
        combine_library,
    )
    q4_fused_down, q4_weighted, q4_counter = _run_direct_parallel_weighted(
        gguf_q4_k_t16_selected_natural_parallel_weighted_gemv_bf16_bf16_out,
        down_x,
        selected,
        q4_tiles,
        route_weights,
        down_out,
        t16_selected_library,
    )
    np.testing.assert_array_equal(q4_fused_down, q4_parallel)
    np.testing.assert_array_equal(q4_weighted, q4_weighted_ref)
    assert q4_counter == 0
    q4_pair_down, q4_pair_weighted, q4_pair_counter = (
        _run_direct_parallel_weighted(
            gguf_q4_k_t16_selected_natural_parallel_paircoeff_weighted_gemv_bf16_bf16_out,
            down_x,
            selected,
            q4_tiles,
            route_weights,
            down_out,
            t16_selected_library,
        )
    )
    np.testing.assert_array_equal(q4_pair_down, q4_parallel)
    np.testing.assert_array_equal(q4_pair_weighted, q4_weighted_ref)
    assert q4_pair_counter == 0

    q6_down = _stack_experts(
        make_q6_k_weight,
        down_out,
        down_in,
        num_experts,
        seed=83,
    )
    q6_tiles = repack_gguf_q6_k_tile16_qmicro_planar(q6_down).tiles

    def q6_ref_fn(*args, **kwargs):
        return gguf_q6_k_t16_selected_gemv_bf16_bf16_out(
            *args,
            qmicro=True,
            qmicro_planar=True,
            **kwargs,
        )

    q6_ref = _run_direct_single(
        q6_ref_fn,
        down_x,
        selected,
        q6_tiles,
        down_out,
        np.uint16,
        t16_selected_library,
    )
    q6_actual = _run_direct_single(
        gguf_q6_k_t16_qmicro_planar_selected_natural_gemv_bf16_bf16_out,
        down_x,
        selected,
        q6_tiles,
        down_out,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(q6_actual, q6_ref)
    q6_parallel = _run_direct_single(
        gguf_q6_k_t16_qmicro_planar_selected_natural_parallel_gemv_bf16_bf16_out,
        down_x,
        selected,
        q6_tiles,
        down_out,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(q6_parallel, q6_actual)
    q6_weighted_ref = _run_weighted_sum(
        q6_parallel,
        route_weights,
        combine_library,
    )
    q6_fused_down, q6_weighted, q6_counter = _run_direct_parallel_weighted(
        gguf_q6_k_t16_qmicro_planar_selected_natural_parallel_weighted_gemv_bf16_bf16_out,
        down_x,
        selected,
        q6_tiles,
        route_weights,
        down_out,
        t16_selected_library,
    )
    np.testing.assert_array_equal(q6_fused_down, q6_parallel)
    np.testing.assert_array_equal(q6_weighted, q6_weighted_ref)
    assert q6_counter == 0


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_p9_d4_q4_t16_direct_dual_silu_matches_split_kernel_bits(t16_selected_library) -> None:
    x_rows = 2
    selected = np.array([2, 0, 1, 1, 2, 0], dtype=np.int64)
    in_features, out_features = 512, 256
    num_experts = 3
    rng = np.random.default_rng(20260520)
    qa = _stack_experts(make_q4_k_weight, out_features, in_features, num_experts, seed=31)
    qb = _stack_experts(make_q4_k_weight, out_features, in_features, num_experts, seed=37)
    ta = repack_gguf_q4_k_tile16(qa).tiles
    tb = repack_gguf_q4_k_tile16(qb).tiles
    x = rng.normal(0.0, 0.1, size=(x_rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)

    gate_bits, up_bits = _run_direct_dual(
        gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        ta,
        tb,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    fused_bits = _run_direct_dual_silu(
        gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        ta,
        tb,
        out_features,
        np.uint16,
        t16_selected_library,
    )

    gate = _bf16_u16_to_f32(gate_bits)
    up = _bf16_u16_to_f32(up_bits)
    with np.errstate(over="ignore"):
        expected_bits = _f32_to_bf16_u16((gate / (1.0 + np.exp(-gate))) * up)
    np.testing.assert_array_equal(fused_bits, expected_bits)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_sh2_m4_q4_qmicro_fused_silu_matches_t16_production_bits(t16_selected_library) -> None:
    x_rows = 1
    in_features, out_features, num_experts = 512, 32, 3
    rng = np.random.default_rng(202608061)
    x_bf16 = _f32_to_bf16_u16(rng.normal(0.0, 0.2, size=(x_rows, in_features)).astype(np.float32))
    selected = np.array([2, 0, 1, 1, 2, 0, 2, 1], dtype=np.int64)
    qa = _stack_experts(make_q4_k_weight, out_features, in_features, num_experts, seed=29)
    qb = _stack_experts(make_q4_k_weight, out_features, in_features, num_experts, seed=43)

    baseline = _run_direct_dual_silu(
        gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        repack_gguf_q4_k_tile16(qa).tiles,
        repack_gguf_q4_k_tile16(qb).tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    compact = _run_direct_dual_silu(
        gguf_q4_k_qmicro_t16_selected_dual_silu_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        repack_gguf_q4_k_tile16_qmicro(qa).tiles,
        repack_gguf_q4_k_tile16_qmicro(qb).tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )

    np.testing.assert_array_equal(compact, baseline)


def test_sh2_m4_q5_qmicro_qwen_tile8_matches_t16_production_bits(t16_selected_library) -> None:
    rows = 8
    in_features, out_features, num_experts = 512, 2048, 3
    rng = np.random.default_rng(202608062)
    x_bf16 = _f32_to_bf16_u16(rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32))
    selected = np.array([2, 0, 1, 1, 2, 0, 2, 1], dtype=np.int64)
    qweight = _stack_experts(make_q5_k_weight, out_features, in_features, num_experts, seed=59)

    baseline = _run_direct_single(
        gguf_q5_k_t16_selected_qwen_tile8_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        repack_gguf_q5_k_tile16(qweight).tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    compact = _run_direct_single(
        gguf_q5_k_qmicro_t16_selected_qwen_tile8_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        repack_gguf_q5_k_qmicro_tile16(qweight).tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )

    np.testing.assert_array_equal(compact, baseline)


def test_q4_t16_direct_dual_pairreuse_matches_production_bits(t16_selected_library) -> None:
    """Repeated expert IDs may share weights without changing row arithmetic."""

    x_rows = 8
    # Cover byte-identical repeated inputs, different-input repeated experts,
    # and unpaired IDs in the same physical-C8 launch.
    selected = np.array(
        [
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            32, 33, 34, 35, 36, 37, 38, 39,
        ],
        dtype=np.int64,
    )
    in_features, out_features = 2048, 512
    num_experts = 40
    rng = np.random.default_rng(20260720)
    qa = _stack_experts(make_q4_k_weight, out_features, in_features, num_experts, seed=61)
    qb = _stack_experts(make_q4_k_weight, out_features, in_features, num_experts, seed=67)
    ta = repack_gguf_q4_k_tile16(qa).tiles
    tb = repack_gguf_q4_k_tile16(qb).tiles
    x = rng.normal(0.0, 0.3, size=(x_rows, in_features)).astype(np.float32)
    x[4] = x[0]
    x[5] = x[1]
    x_bf16 = _f32_to_bf16_u16(x)

    ref_a, ref_b = _run_direct_dual(
        gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        ta,
        tb,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    actual_a, actual_b = _run_direct_dual(
        gguf_q4_k_t16_selected_dual_pairreuse_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        ta,
        tb,
        out_features,
        np.uint16,
        t16_selected_library,
    )

    np.testing.assert_array_equal(actual_a, ref_a)
    np.testing.assert_array_equal(actual_b, ref_b)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_q4_t16_compact_pairreuse_matches_direct_bits_beyond_64_lanes(
    t16_selected_library,
) -> None:
    """Compact expert groups may pair arbitrary prompt lanes without reassociation."""

    counts = [0, 3, 17, 1, 64, 2, 45]
    expert_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    selected = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    rows = int(selected.size)
    in_features, out_features = 512, 256
    rng = np.random.default_rng(20260723)
    qa = _stack_experts(make_q4_k_weight, out_features, in_features, len(counts), seed=79)
    qb = _stack_experts(make_q4_k_weight, out_features, in_features, len(counts), seed=83)
    ta = repack_gguf_q4_k_tile16(qa).tiles
    tb = repack_gguf_q4_k_tile16(qb).tiles
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    )

    ref_a, ref_b = _run_direct_dual(
        gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        ta,
        tb,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    candidate = _run_dual(
        gguf_q4_k_t16_selected_dual_pairreuse_gemv_decode_compact_bf16_bf16_out,
        x_bf16,
        expert_start,
        ta,
        tb,
        out_features,
        out_features,
        np.uint16,
        t16_selected_library,
    )

    np.testing.assert_array_equal(candidate[:, :out_features], ref_a)
    np.testing.assert_array_equal(candidate[:, out_features:], ref_b)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("builder", "repack", "direct_fn", "compact_pair_fn"),
    (
        (
            make_q4_k_weight,
            repack_gguf_q4_k_tile16,
            gguf_q4_k_t16_selected_gemv_bf16_bf16_out,
            gguf_q4_k_t16_selected_pairreuse_gemv_decode_compact_bf16_bf16_out,
        ),
        (
            make_q5_k_weight,
            repack_gguf_q5_k_tile16,
            gguf_q5_k_t16_selected_gemv_bf16_bf16_out,
            gguf_q5_k_t16_selected_pairreuse_gemv_decode_compact_bf16_bf16_out,
        ),
        (
            make_q6_k_weight,
            repack_gguf_q6_k_tile16,
            gguf_q6_k_t16_selected_gemv_bf16_bf16_out,
            gguf_q6_k_t16_selected_pairreuse_gemv_decode_compact_bf16_bf16_out,
        ),
    ),
)
def test_t16_compact_down_pairreuse_matches_direct_bits_beyond_64_lanes(
    builder,
    repack,
    direct_fn,
    compact_pair_fn,
    t16_selected_library,
) -> None:
    counts = [0, 3, 17, 1, 64, 2, 45]
    expert_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    selected = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    rows = int(selected.size)
    in_features, out_features = 512, 256
    rng = np.random.default_rng(20260724)
    qweight = _stack_experts(builder, out_features, in_features, len(counts), seed=89)
    tiles = repack(qweight).tiles
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    )

    reference = _run_direct_single(
        direct_fn,
        x_bf16,
        selected,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    candidate = _run_single(
        compact_pair_fn,
        x_bf16,
        expert_start,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )

    np.testing.assert_array_equal(candidate, reference)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_q4_q6_t16_grouped_smallm_matches_direct_bits_and_cpu_oracle(
    t16_selected_library,
) -> None:
    """Grouped 1/2/4/8-row buckets must retain the direct reduction exactly."""

    counts = [0, 1, 2, 4, 8, 3, 5]
    expert_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    selected = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    rows = int(selected.size)
    in_features, out_features = 512, 64
    rng = np.random.default_rng(20260725)
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    )
    x_ref = _bf16_u16_to_f32(x_bf16)

    qa = _stack_experts(make_q4_k_weight, out_features, in_features, len(counts), seed=97)
    qb = _stack_experts(make_q4_k_weight, out_features, in_features, len(counts), seed=101)
    ta = repack_gguf_q4_k_tile16(qa).tiles
    tb = repack_gguf_q4_k_tile16(qb).tiles
    direct_a, direct_b = _run_direct_dual(
        gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        ta,
        tb,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    grouped_a, grouped_b = _run_grouped_dual(
        gguf_q4_k_t16_selected_dual_grouped_smallm_bf16_bf16_out,
        x_bf16,
        expert_start,
        ta,
        tb,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(grouped_a, direct_a)
    np.testing.assert_array_equal(grouped_b, direct_b)
    expected_dual = _expected_dual(
        x_ref, expert_start, qa, qb, out_features, out_features
    )
    actual_dual = np.concatenate(
        [_bf16_u16_to_f32(grouped_a), _bf16_u16_to_f32(grouped_b)], axis=1
    )
    np.testing.assert_allclose(
        actual_dual,
        _bf16_u16_to_f32(_f32_to_bf16_u16(expected_dual)),
        **_TOL,
    )

    q6 = _stack_experts(make_q6_k_weight, out_features, in_features, len(counts), seed=103)
    t6 = repack_gguf_q6_k_tile16(q6).tiles
    direct_down = _run_direct_single(
        gguf_q6_k_t16_selected_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        t6,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    grouped_down = _run_grouped_smallm_single(
        gguf_q6_k_t16_selected_grouped_smallm_bf16_bf16_out,
        x_bf16,
        expert_start,
        t6,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(grouped_down, direct_down)
    expected_down = _expected_single(
        x_ref, expert_start, q6, out_features, GGMLQuantizationType.Q6_K
    )
    np.testing.assert_allclose(
        _bf16_u16_to_f32(grouped_down),
        _bf16_u16_to_f32(_f32_to_bf16_u16(expected_down)),
        **_TOL,
    )

    t6_qmicro = repack_gguf_q6_k_tile16_qmicro(q6).tiles
    direct_down_qmicro = _run_direct_single(
        gguf_q6_k_t16_selected_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        t6_qmicro,
        out_features,
        np.uint16,
        t16_selected_library,
        qmicro=True,
    )
    grouped_down_qmicro = _run_grouped_smallm_single(
        gguf_q6_k_t16_selected_grouped_smallm_bf16_bf16_out,
        x_bf16,
        expert_start,
        t6_qmicro,
        out_features,
        np.uint16,
        t16_selected_library,
        qmicro=True,
    )
    np.testing.assert_array_equal(direct_down_qmicro, direct_down)
    np.testing.assert_array_equal(grouped_down_qmicro, direct_down)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_q4_q6_t16_grouped_smallm_matches_laguna_production_shape_bits(
    t16_selected_library,
) -> None:
    counts = [1, 2, 4, 8]
    expert_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    selected = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    rows = int(selected.size)
    hidden_size, expert_ffn = 3_072, 1_024
    rng = np.random.default_rng(20260726)
    hidden_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.1, size=(rows, hidden_size)).astype(np.float32)
    )

    gate = _stack_experts(
        make_q4_k_weight, expert_ffn, hidden_size, len(counts), seed=107
    )
    up = _stack_experts(
        make_q4_k_weight, expert_ffn, hidden_size, len(counts), seed=109
    )
    gate_tiles = repack_gguf_q4_k_tile16(gate).tiles
    up_tiles = repack_gguf_q4_k_tile16(up).tiles
    direct_gate, direct_up = _run_direct_dual(
        gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
        hidden_bf16,
        selected,
        gate_tiles,
        up_tiles,
        expert_ffn,
        np.uint16,
        t16_selected_library,
    )
    grouped_gate, grouped_up = _run_grouped_dual(
        gguf_q4_k_t16_selected_dual_grouped_smallm_bf16_bf16_out,
        hidden_bf16,
        expert_start,
        gate_tiles,
        up_tiles,
        expert_ffn,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(grouped_gate, direct_gate)
    np.testing.assert_array_equal(grouped_up, direct_up)

    intermediate_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.1, size=(rows, expert_ffn)).astype(np.float32)
    )
    for builder, repack, direct_fn, grouped_fn in (
        (
            make_q4_k_weight,
            repack_gguf_q4_k_tile16,
            gguf_q4_k_t16_selected_gemv_bf16_bf16_out,
            gguf_q4_k_t16_selected_grouped_smallm_bf16_bf16_out,
        ),
        (
            make_q6_k_weight,
            repack_gguf_q6_k_tile16,
            gguf_q6_k_t16_selected_gemv_bf16_bf16_out,
            gguf_q6_k_t16_selected_grouped_smallm_bf16_bf16_out,
        ),
    ):
        weight = _stack_experts(
            builder, hidden_size, expert_ffn, len(counts), seed=113
        )
        tiles = repack(weight).tiles
        direct = _run_direct_single(
            direct_fn,
            intermediate_bf16,
            selected,
            tiles,
            hidden_size,
            np.uint16,
            t16_selected_library,
        )
        grouped = _run_grouped_smallm_single(
            grouped_fn,
            intermediate_bf16,
            expert_start,
            tiles,
            hidden_size,
            np.uint16,
            t16_selected_library,
        )
        np.testing.assert_array_equal(grouped, direct)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_p9_h3d_q4_t16_direct_dual_matches_legacy_raw_reduction_order(t16_selected_library) -> None:
    """Guard P9.E2 exactness for Qwen-shaped row-bulk selected MoE.

    The full-model fixture compares the T16 replacement path against the legacy
    raw-GGUF row-GEMV path.  Matching the CPU oracle is not sufficient here:
    the direct T16 Q4 gate/up kernel must also preserve the raw kernel's BF16
    reduction topology so tiny per-layer drift does not amplify across 40 MoE
    layers.
    """

    x_rows, top_k = 4, 8
    rows = x_rows * top_k
    selected = (np.arange(rows, dtype=np.int64) * 3) % 4
    in_features, out_features = 2048, 512
    num_experts = 4
    rng = np.random.default_rng(20260519)
    qa = _stack_experts(make_q4_k_weight, out_features, in_features, num_experts, seed=23)
    qb = _stack_experts(make_q4_k_weight, out_features, in_features, num_experts, seed=29)
    ta = repack_gguf_q4_k_tile16(qa).tiles
    tb = repack_gguf_q4_k_tile16(qb).tiles
    x = rng.normal(0.0, 0.3, size=(x_rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)

    raw_a, raw_b = _run_raw_direct_dual(
        x_bf16,
        selected,
        qa,
        qb,
        out_features,
        build_gguf_q4_k_gemv(load=True),
    )
    t16_a, t16_b = _run_direct_dual(
        gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        ta,
        tb,
        out_features,
        np.uint16,
        t16_selected_library,
    )

    np.testing.assert_array_equal(t16_a, raw_a)
    np.testing.assert_array_equal(t16_b, raw_b)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_q5_t16_direct_pairreuse_matches_production_bits(t16_selected_library) -> None:
    rows = 64
    selected = np.array(
        [
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            32, 33, 34, 35, 36, 37, 38, 39,
        ],
        dtype=np.int64,
    )
    in_features, out_features, num_experts = 512, 512, 40
    rng = np.random.default_rng(20260721)
    qweight = _stack_experts(make_q5_k_weight, out_features, in_features, num_experts, seed=71)
    tiles = repack_gguf_q5_k_tile16(qweight).tiles
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    )

    reference = _run_direct_single(
        gguf_q5_k_t16_selected_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    candidate = _run_direct_single(
        gguf_q5_k_t16_selected_pairreuse_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )

    np.testing.assert_array_equal(candidate, reference)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_q6_t16_direct_pairreuse_matches_production_bits(t16_selected_library) -> None:
    rows = 64
    selected = np.array(
        [
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            32, 33, 34, 35, 36, 37, 38, 39,
        ],
        dtype=np.int64,
    )
    in_features, out_features, num_experts = 512, 512, 40
    rng = np.random.default_rng(20260722)
    qweight = _stack_experts(make_q6_k_weight, out_features, in_features, num_experts, seed=73)
    tiles = repack_gguf_q6_k_tile16(qweight).tiles
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    )

    reference = _run_direct_single(
        gguf_q6_k_t16_selected_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    candidate = _run_direct_single(
        gguf_q6_k_t16_selected_pairreuse_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )

    np.testing.assert_array_equal(candidate, reference)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_t16_q4_direct_dual_q8_1_dp4a_matches_float_path_quality_gate(t16_selected_library) -> None:
    x_rows, top_k = 4, 8
    rows = x_rows * top_k
    selected = (np.arange(rows, dtype=np.int64) * 5) % 4
    in_features, out_features = 2048, 512
    num_experts = 4
    rng = np.random.default_rng(20260627)
    qa = _stack_experts(make_q4_k_weight, out_features, in_features, num_experts, seed=41)
    qb = _stack_experts(make_q4_k_weight, out_features, in_features, num_experts, seed=43)
    ta = repack_gguf_q4_k_tile16(qa).tiles
    tb = repack_gguf_q4_k_tile16(qb).tiles
    x = rng.normal(0.0, 0.3, size=(x_rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)

    ref_a_bits, ref_b_bits = _run_direct_dual(
        gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        ta,
        tb,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    dp4a_a_bits, dp4a_b_bits = _run_direct_dual_q8_dp4a(
        x_bf16,
        selected,
        ta,
        tb,
        out_features,
        np.uint16,
        t16_selected_library,
        build_gguf_q4_k_gemv(load=True),
    )

    for ref_bits, dp4a_bits in ((ref_a_bits, dp4a_a_bits), (ref_b_bits, dp4a_b_bits)):
        ref = _bf16_u16_to_f32(ref_bits)
        dp4a = _bf16_u16_to_f32(dp4a_bits)
        kl_mean, kl_max = _softmax_kl(ref, dp4a)
        assert kl_mean <= 0.05
        assert kl_max <= 0.10
        assert _top1(ref, dp4a) >= 0.90


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_t16_q4_direct_dual_silu_q8_1_dp4a_matches_split_dp4a_rounding(t16_selected_library) -> None:
    x_rows = 2
    selected = np.array([2, 0, 1, 1, 2, 0], dtype=np.int64)
    in_features, out_features = 512, 256
    num_experts = 3
    rng = np.random.default_rng(20260628)
    qa = _stack_experts(make_q4_k_weight, out_features, in_features, num_experts, seed=47)
    qb = _stack_experts(make_q4_k_weight, out_features, in_features, num_experts, seed=53)
    ta = repack_gguf_q4_k_tile16(qa).tiles
    tb = repack_gguf_q4_k_tile16(qb).tiles
    x = rng.normal(0.0, 0.3, size=(x_rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    q4_library = build_gguf_q4_k_gemv(load=True)

    gate_bits, up_bits = _run_direct_dual_q8_dp4a(
        x_bf16,
        selected,
        ta,
        tb,
        out_features,
        np.uint16,
        t16_selected_library,
        q4_library,
    )
    fused_bits = _run_direct_dual_silu_q8_dp4a(
        x_bf16,
        selected,
        ta,
        tb,
        out_features,
        np.uint16,
        t16_selected_library,
        q4_library,
    )

    gate = _bf16_u16_to_f32(gate_bits)
    up = _bf16_u16_to_f32(up_bits)
    with np.errstate(over="ignore"):
        expected_bits = _f32_to_bf16_u16((gate / (1.0 + np.exp(-gate))) * up)
    np.testing.assert_array_equal(fused_bits, expected_bits)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_t16_q4_direct_dual_silu_q8_1x2_dp4a_repairs_quality(
    t16_selected_library,
) -> None:
    x_rows, top_k = 3, 4
    rows = x_rows * top_k
    selected = (np.arange(rows, dtype=np.int64) * 5) % 3
    in_features, out_features = 1024, 512
    num_experts = 3
    rng = np.random.default_rng(20260815)
    qa = _stack_experts(
        make_q4_k_weight, out_features, in_features, num_experts, seed=71
    )
    qb = _stack_experts(
        make_q4_k_weight, out_features, in_features, num_experts, seed=73
    )
    ta = repack_gguf_q4_k_tile16(qa).tiles
    tb = repack_gguf_q4_k_tile16(qb).tiles
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(x_rows, in_features)).astype(np.float32)
    )
    q4_library = build_gguf_q4_k_gemv(load=True)

    reference_bits = _run_direct_dual_silu(
        gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        ta,
        tb,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    one_plane_bits = _run_direct_dual_silu_q8_dp4a(
        x_bf16,
        selected,
        ta,
        tb,
        out_features,
        np.uint16,
        t16_selected_library,
        q4_library,
    )
    two_plane_bits = _run_direct_dual_silu_q8x2_dp4a(
        x_bf16,
        selected,
        ta,
        tb,
        out_features,
        np.uint16,
        t16_selected_library,
        q4_library,
    )

    reference = _bf16_u16_to_f32(reference_bits)
    one_plane = _bf16_u16_to_f32(one_plane_bits)
    two_plane = _bf16_u16_to_f32(two_plane_bits)
    kl_mean, kl_max = _softmax_kl(reference, two_plane)
    assert kl_mean <= 0.05
    assert kl_max <= 0.10
    assert _top1(reference, two_plane) >= 0.90
    assert np.mean(np.abs(two_plane - reference)) < np.mean(
        np.abs(one_plane - reference)
    )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_t16_q4_dense_dual_q8_1x2_split_weight_is_bit_exact(
    t16_selected_library,
    q4_library,
) -> None:
    rng = np.random.default_rng(20260816)
    in_features, out_features = 1024, 512
    raw_a = make_q4_k_weight(out_features, in_features)
    raw_b = np.roll(raw_a, shift=11, axis=0).copy()
    tiles_a = repack_gguf_q4_k_tile16(raw_a[None, ...]).tiles
    tiles_b = repack_gguf_q4_k_tile16(raw_b[None, ...]).tiles
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(1, in_features)).astype(np.float32)
    )

    control = _run_dense_dual_silu_q8x2_dp4a(
        gguf_q4_k_t16_dense_dual_q8_1x2_dp4a_silu_bf16_bf16_out,
        x_bf16,
        tiles_a,
        tiles_b,
        out_features,
        np.uint16,
        t16_selected_library,
        q4_library,
    )
    candidate = _run_dense_dual_silu_q8x2_dp4a(
        gguf_q4_k_t16_dense_dual_q8_1x2_split_weight_dp4a_silu_bf16_bf16_out,
        x_bf16,
        tiles_a,
        tiles_b,
        out_features,
        np.uint16,
        t16_selected_library,
        q4_library,
    )
    np.testing.assert_array_equal(candidate, control)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_qmicro_q4_dense_dual_q8_1x2_split_weight_is_bit_exact(
    t16_selected_library,
    q4_library,
) -> None:
    rng = np.random.default_rng(20260817)
    in_features, out_features = 1024, 512
    raw_a = make_q4_k_weight(out_features, in_features)
    raw_b = np.roll(raw_a, shift=13, axis=0).copy()
    control_tiles_a = repack_gguf_q4_k_tile16(raw_a[None, ...]).tiles
    control_tiles_b = repack_gguf_q4_k_tile16(raw_b[None, ...]).tiles
    candidate_tiles_a = repack_gguf_q4_k_tile16_qmicro(raw_a[None, ...]).tiles
    candidate_tiles_b = repack_gguf_q4_k_tile16_qmicro(raw_b[None, ...]).tiles
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(1, in_features)).astype(np.float32)
    )

    control = _run_dense_dual_silu_q8x2_dp4a(
        gguf_q4_k_t16_dense_dual_q8_1x2_split_weight_dp4a_silu_bf16_bf16_out,
        x_bf16,
        control_tiles_a,
        control_tiles_b,
        out_features,
        np.uint16,
        t16_selected_library,
        q4_library,
    )
    candidate = _run_dense_dual_silu_q8x2_dp4a(
        gguf_q4_k_qmicro_t16_dense_dual_q8_1x2_split_weight_dp4a_silu_bf16_bf16_out,
        x_bf16,
        candidate_tiles_a,
        candidate_tiles_b,
        out_features,
        np.uint16,
        t16_selected_library,
        q4_library,
    )
    np.testing.assert_array_equal(candidate, control)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 7, 8])
def test_qmicro_q4_dense_dual_q8_1x2_shared_rowbatch_matches_serial_c1_bits(
    rows,
    t16_selected_library,
    q4_library,
) -> None:
    shared_rowbatch = getattr(
        selected_t16_mod,
        "gguf_q4_k_qmicro_t16_dense_dual_q8_1x2_rowtile8_dp4a_silu_bf16_bf16_out",
    )
    rng = np.random.default_rng(20260824 + rows)
    in_features, out_features = 1024, 512
    raw_a = make_q4_k_weight(out_features, in_features)
    raw_b = np.roll(raw_a, shift=13, axis=0).copy()
    tiles_a = repack_gguf_q4_k_tile16_qmicro(raw_a[None, ...]).tiles
    tiles_b = repack_gguf_q4_k_tile16_qmicro(raw_b[None, ...]).tiles
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    )

    serial = np.concatenate(
        [
            _run_dense_dual_silu_q8x2_dp4a(
                gguf_q4_k_qmicro_t16_dense_dual_q8_1x2_split_weight_dp4a_silu_bf16_bf16_out,
                x_bf16[row : row + 1],
                tiles_a,
                tiles_b,
                out_features,
                np.uint16,
                t16_selected_library,
                q4_library,
            )
            for row in range(rows)
        ],
        axis=0,
    )
    batched = _run_dense_dual_silu_q8x2_dp4a(
        shared_rowbatch,
        x_bf16,
        tiles_a,
        tiles_b,
        out_features,
        np.uint16,
        t16_selected_library,
        q4_library,
    )
    np.testing.assert_array_equal(batched, serial)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "name,builder,repack,fn_float,fn_dp4a",
    [
        pytest.param(
            "Q5_K",
            make_q5_k_weight,
            repack_gguf_q5_k_tile16,
            gguf_q5_k_t16_selected_gemv_bf16_bf16_out,
            gguf_q5_k_t16_selected_q8_1_dp4a_gemv_bf16_bf16_out,
            id="Q5_K",
        ),
    ],
)
def test_t16_qk_direct_selected_down_q8_1_dp4a_matches_float_path_quality_gate(
    name, builder, repack, fn_float, fn_dp4a, t16_selected_library, monkeypatch
) -> None:
    x_rows, top_k = 4, 8
    rows = x_rows * top_k
    selected = (np.arange(rows, dtype=np.int64) * 7) % 5
    in_features, out_features = 512, 2048
    num_experts = 5
    rng = np.random.default_rng(20260629 if name == "Q5_K" else 20260630)
    qw = _stack_experts(builder, out_features, in_features, num_experts, seed=61 if name == "Q5_K" else 67)
    tiles = repack(qw).tiles
    x = rng.normal(0.0, 0.1, size=(x_rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)

    ref_bits = _run_direct_single(fn_float, x_bf16, selected, tiles, out_features, np.uint16, t16_selected_library)
    monkeypatch.setenv("HIPENGINE_GGUF_T16_SELECTED_Q5_DP4A_THREADS", "32")
    dp4a_bits = _run_direct_single_q8_dp4a(
        fn_dp4a,
        x_bf16,
        selected,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
        build_gguf_q4_k_gemv(load=True),
    )

    ref = _bf16_u16_to_f32(ref_bits)
    dp4a = _bf16_u16_to_f32(dp4a_bits)
    kl_mean, kl_max = _softmax_kl(ref, dp4a)
    assert kl_mean <= 0.05
    assert kl_max <= 0.10
    assert _top1(ref, dp4a) >= 0.90


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_p9_h3d_q4_t16_dual_fp16_matches_cpu_oracle(t16_selected_library) -> None:
    counts = [3, 1, 4]
    in_features, out_features_a, out_features_b = 512, 256, 256
    num_experts = len(counts)
    expert_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    compact_rows = int(expert_start[-1])
    rng = np.random.default_rng(441)
    qa = _stack_experts(make_q4_k_weight, out_features_a, in_features, num_experts, seed=3)
    qb = _stack_experts(make_q4_k_weight, out_features_b, in_features, num_experts, seed=5)
    ta = repack_gguf_q4_k_tile16(qa).tiles
    tb = repack_gguf_q4_k_tile16(qb).tiles
    x_f16 = rng.normal(0.0, 0.3, size=(compact_rows, in_features)).astype(np.float16)

    actual = _run_dual(
        gguf_q4_k_t16_selected_dual_gemv_decode_compact_fp16_fp16_out,
        x_f16,
        expert_start,
        ta,
        tb,
        out_features_a,
        out_features_b,
        np.float16,
        t16_selected_library,
    )

    expected = _expected_dual(x_f16.astype(np.float32), expert_start, qa, qb, out_features_a, out_features_b)
    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float16).astype(np.float32), **_TOL)


_QUANT_CASES = [
    pytest.param(
        "Q4_K",
        make_q4_k_weight,
        repack_gguf_q4_k_tile16,
        gguf_q4_k_t16_selected_gemv_decode_compact_bf16_bf16_out,
        gguf_q4_k_t16_selected_gemv_decode_compact_fp16_fp16_out,
        GGMLQuantizationType.Q4_K,
        id="Q4_K",
    ),
    pytest.param(
        "Q5_K",
        make_q5_k_weight,
        repack_gguf_q5_k_tile16,
        gguf_q5_k_t16_selected_gemv_decode_compact_bf16_bf16_out,
        gguf_q5_k_t16_selected_gemv_decode_compact_fp16_fp16_out,
        GGMLQuantizationType.Q5_K,
        id="Q5_K",
    ),
    pytest.param(
        "Q6_K",
        make_q6_k_weight,
        repack_gguf_q6_k_tile16,
        gguf_q6_k_t16_selected_gemv_decode_compact_bf16_bf16_out,
        gguf_q6_k_t16_selected_gemv_decode_compact_fp16_fp16_out,
        GGMLQuantizationType.Q6_K,
        id="Q6_K",
    ),
]


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("counts", _EXPERT_LAYOUTS)
@pytest.mark.parametrize("in_features,out_features", [(256, 16), (512, 256), (2048, 2048)])
@pytest.mark.parametrize("_name,builder,repack,fn_bf16,_fn_fp16,qtype_enum", _QUANT_CASES)
def test_p9_h3d_qk_t16_bf16_matches_cpu_oracle(
    _name, builder, repack, fn_bf16, _fn_fp16, qtype_enum, counts, in_features, out_features, t16_selected_library,
) -> None:
    num_experts = len(counts)
    expert_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    compact_rows = int(expert_start[-1])
    rng = np.random.default_rng(num_experts * 211 + in_features + out_features)
    qw = _stack_experts(builder, out_features, in_features, num_experts, seed=7)
    tiles = repack(qw).tiles
    x = rng.normal(0.0, 0.3, size=(compact_rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)

    actual = _run_single(fn_bf16, x_bf16, expert_start, tiles, out_features, np.uint16, t16_selected_library)

    expected = _expected_single(x_ref, expert_start, qw, out_features, qtype_enum)
    expected_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected))
    np.testing.assert_allclose(_bf16_u16_to_f32(actual), expected_bf16, **_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "_name,builder,repack,fn_direct_bf16,qtype_enum",
    [
        pytest.param("Q4_K", make_q4_k_weight, repack_gguf_q4_k_tile16, gguf_q4_k_t16_selected_gemv_bf16_bf16_out, GGMLQuantizationType.Q4_K, id="Q4_K"),
        pytest.param("Q5_K", make_q5_k_weight, repack_gguf_q5_k_tile16, gguf_q5_k_t16_selected_gemv_bf16_bf16_out, GGMLQuantizationType.Q5_K, id="Q5_K"),
        pytest.param("Q6_K", make_q6_k_weight, repack_gguf_q6_k_tile16, gguf_q6_k_t16_selected_gemv_bf16_bf16_out, GGMLQuantizationType.Q6_K, id="Q6_K"),
    ],
)
def test_p9_h3d_qk_t16_direct_bf16_matches_cpu_oracle(
    _name, builder, repack, fn_direct_bf16, qtype_enum, t16_selected_library,
) -> None:
    x_rows = 2
    selected = np.array([2, 0, 1, 1, 2, 0], dtype=np.int64)
    in_features, out_features = 512, 256
    num_experts = 3
    rng = np.random.default_rng(557)
    qw = _stack_experts(builder, out_features, in_features, num_experts, seed=19)
    tiles = repack(qw).tiles
    x = rng.normal(0.0, 0.3, size=(x_rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)

    actual = _run_direct_single(fn_direct_bf16, x_bf16, selected, tiles, out_features, np.uint16, t16_selected_library)

    expected = _expected_direct_single(x_ref, selected, qw, out_features, qtype_enum)
    expected_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected))
    np.testing.assert_allclose(_bf16_u16_to_f32(actual), expected_bf16, **_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "builder,repack,production_fn,candidate_fn,qtype_enum",
    [
        pytest.param(
            make_q5_k_weight,
            repack_gguf_q5_k_tile16,
            gguf_q5_k_t16_selected_gemv_bf16_bf16_out,
            gguf_q5_k_t16_selected_qwen_tile8_gemv_bf16_bf16_out,
            GGMLQuantizationType.Q5_K,
            id="Q5_K",
        ),
    ],
)
def test_sh_d1_qwen_selected_down_tile8_matches_production_bits_and_cpu_oracle(
    builder,
    repack,
    production_fn,
    candidate_fn,
    qtype_enum,
    t16_selected_library,
) -> None:
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_q5_k_t16_v1",
        variant="selected_t16_qwen_tile8_gemv_decode_bf16_bf16_out",
    ) is candidate_fn
    rows = 8
    selected = np.array([2, 0, 1, 1, 2, 0, 2, 1], dtype=np.int64)
    in_features, out_features, num_experts = 512, 2048, 3
    rng = np.random.default_rng(20260806 + int(qtype_enum))
    qweight = _stack_experts(
        builder,
        out_features,
        in_features,
        num_experts,
        seed=97,
    )
    tiles = repack(qweight).tiles
    x_bf16 = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    )

    production = _run_direct_single(
        production_fn,
        x_bf16,
        selected,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    candidate = _run_direct_single(
        candidate_fn,
        x_bf16,
        selected,
        tiles,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(candidate, production)

    expected = _expected_direct_single(
        _bf16_u16_to_f32(x_bf16),
        selected,
        qweight,
        out_features,
        qtype_enum,
    )
    expected_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected))
    np.testing.assert_allclose(
        _bf16_u16_to_f32(candidate),
        expected_bf16,
        **_TOL,
    )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("x_rows", [1, 2, 4, 8])
def test_q6_t16_qmicro_direct_decode_matches_production_bits(
    x_rows: int,
    t16_selected_library,
) -> None:
    top_k = 3
    selected = np.resize(
        np.array([2, 0, 1, 1, 2, 0], dtype=np.int64),
        x_rows * top_k,
    )
    in_features, out_features, num_experts = 512, 256, 3
    rng = np.random.default_rng(20260726)
    qweight = _stack_experts(
        make_q6_k_weight,
        out_features,
        in_features,
        num_experts,
        seed=23,
    )
    legacy = repack_gguf_q6_k_tile16(qweight).tiles
    qmicro = repack_gguf_q6_k_tile16_qmicro_planar(qweight).tiles
    x = rng.normal(0.0, 0.3, size=(x_rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)

    production = _run_direct_single(
        gguf_q6_k_t16_selected_gemv_bf16_bf16_out,
        x_bf16,
        selected,
        legacy,
        out_features,
        np.uint16,
        t16_selected_library,
    )

    def qmicro_decode(*args, **kwargs):
        return gguf_q6_k_t16_selected_gemv_bf16_bf16_out(
            *args,
            qmicro=True,
            qmicro_planar=True,
            **kwargs,
        )

    candidate = _run_direct_single(
        qmicro_decode,
        x_bf16,
        selected,
        qmicro,
        out_features,
        np.uint16,
        t16_selected_library,
    )
    np.testing.assert_array_equal(candidate, production)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("_name,builder,repack,_fn_bf16,fn_fp16,qtype_enum", _QUANT_CASES)
def test_p9_h3d_qk_t16_fp16_matches_cpu_oracle(
    _name, builder, repack, _fn_bf16, fn_fp16, qtype_enum, t16_selected_library,
) -> None:
    counts = [2, 0, 3]
    in_features, out_features = 512, 256
    num_experts = len(counts)
    expert_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    compact_rows = int(expert_start[-1])
    rng = np.random.default_rng(983)
    qw = _stack_experts(builder, out_features, in_features, num_experts, seed=11)
    tiles = repack(qw).tiles
    x_f16 = rng.normal(0.0, 0.3, size=(compact_rows, in_features)).astype(np.float16)

    actual = _run_single(fn_fp16, x_f16, expert_start, tiles, out_features, np.float16, t16_selected_library)

    expected = _expected_single(x_f16.astype(np.float32), expert_start, qw, out_features, qtype_enum)
    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float16).astype(np.float32), **_TOL)
