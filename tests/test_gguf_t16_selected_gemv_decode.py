"""Correctness fixtures for selected GGUF K-family T16 GEMV decode (P9.H3)."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_selected_dual_gemv_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    build_gguf_t16_selected_gemv,
    gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_gemv_fp16_fp16_out,
    gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_gemv_decode_compact_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_gemv_decode_compact_fp16_fp16_out,
    gguf_q4_k_t16_selected_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_gemv_fp16_fp16_out,
    gguf_q4_k_t16_selected_gemv_decode_compact_bf16_bf16_out,
    gguf_q4_k_t16_selected_gemv_decode_compact_fp16_fp16_out,
    gguf_q5_k_t16_selected_gemv_bf16_bf16_out,
    gguf_q5_k_t16_selected_gemv_fp16_fp16_out,
    gguf_q5_k_t16_selected_gemv_decode_compact_bf16_bf16_out,
    gguf_q5_k_t16_selected_gemv_decode_compact_fp16_fp16_out,
    gguf_q6_k_t16_selected_gemv_bf16_bf16_out,
    gguf_q6_k_t16_selected_gemv_fp16_fp16_out,
    gguf_q6_k_t16_selected_gemv_decode_compact_bf16_bf16_out,
    gguf_q6_k_t16_selected_gemv_decode_compact_fp16_fp16_out,
    plan_gguf_t16_selected_gemv_build,
    register_gguf_t16_selected_gemv_kernels,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16
from hipengine.quant.gguf_t16 import repack_gguf_q5_k_tile16, repack_gguf_q6_k_tile16
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


def _run_direct_single(fn, x_dev, selected, tiles, out_features, out_dtype, library) -> np.ndarray:
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
        )
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for buf in (x_buf, sel_buf, w_buf, out_buf):
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
            "selected_dual_t16_silu_gemv_decode_bf16_bf16_out",
            "selected_t16_gemv_decode_compact_bf16_bf16_out",
            "selected_t16_gemv_decode_compact_fp16_fp16_out",
            "selected_t16_gemv_decode_bf16_bf16_out",
            "selected_t16_gemv_decode_fp16_fp16_out",
        ),
        "gguf_q5_k_t16_v1": (
            "selected_t16_gemv_decode_compact_bf16_bf16_out",
            "selected_t16_gemv_decode_compact_fp16_fp16_out",
            "selected_t16_gemv_decode_bf16_bf16_out",
            "selected_t16_gemv_decode_fp16_fp16_out",
        ),
        "gguf_q6_k_t16_v1": (
            "selected_t16_gemv_decode_compact_bf16_bf16_out",
            "selected_t16_gemv_decode_compact_fp16_fp16_out",
            "selected_t16_gemv_decode_bf16_bf16_out",
            "selected_t16_gemv_decode_fp16_fp16_out",
        ),
    }.items():
        for variant in variants:
            assert resolve(backend="hip_gfx1100", layer="moe_linear", quant=quant, variant=variant) is not None


def test_p9_h3d_build_plan_is_dry_run_safe() -> None:
    plan = plan_gguf_t16_selected_gemv_build()
    assert plan.output_path.name == "gguf_t16_selected_gemv.so"
    assert plan.sources[0].name == "gguf_t16_selected_gemv.hip"


def test_p9_h3d_wrappers_validate_args() -> None:
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
    x_rows, top_k = 2, 3
    rows = x_rows * top_k
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
def test_p9_d4_q4_t16_direct_dual_silu_matches_split_kernel_bits(t16_selected_library) -> None:
    x_rows, top_k = 2, 3
    rows = x_rows * top_k
    selected = np.array([2, 0, 1, 1, 2, 0], dtype=np.int64)
    in_features, out_features = 512, 256
    num_experts = 3
    rng = np.random.default_rng(20260520)
    qa = _stack_experts(make_q4_k_weight, out_features, in_features, num_experts, seed=31)
    qb = _stack_experts(make_q4_k_weight, out_features, in_features, num_experts, seed=37)
    ta = repack_gguf_q4_k_tile16(qa).tiles
    tb = repack_gguf_q4_k_tile16(qb).tiles
    x = rng.normal(0.0, 0.3, size=(x_rows, in_features)).astype(np.float32)
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
    x_rows, top_k = 2, 3
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
