"""Raw-pointer wrappers for PARO/AWQ compact WMMA grouped-MoE kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("paro_awq_wmma.hip")
_OUTPUT_NAME = "paro_awq_wmma.so"
_SYMBOL_DUAL_BF16 = "hipengine_gemm_awq_selected_dual_pack8_wmma_compact_bf16"
_SYMBOL_DUAL_FP16 = "hipengine_gemm_awq_selected_dual_pack8_wmma_compact_fp16"
_SYMBOL_SINGLE_BF16 = "hipengine_gemm_awq_selected_pack8_wmma_compact_bf16"
_SYMBOL_SINGLE_FP16 = "hipengine_gemm_awq_selected_pack8_wmma_compact_fp16"


def plan_paro_awq_wmma_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="paro_awq_wmma",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_paro_awq_wmma(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_hip(
        sources=[_SOURCE],
        family="paro_awq_wmma",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gemm_awq_selected_dual_pack8_wmma_compact_bf16(
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_t_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_t_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_dual(
        _SYMBOL_DUAL_BF16,
        x_ptr,
        expert_start_compact_ptr,
        expert_start_wmma_ptr,
        tile_expert_ptr,
        qweight_a_t_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_t_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_packed_a,
        out_packed_b,
        num_experts,
        group_size,
        wmma_total_rows,
        stream,
        library,
        runtime,
    )


def gemm_awq_selected_dual_pack8_wmma_compact_fp16(
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_t_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_t_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_dual(
        _SYMBOL_DUAL_FP16,
        x_ptr,
        expert_start_compact_ptr,
        expert_start_wmma_ptr,
        tile_expert_ptr,
        qweight_a_t_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_t_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_packed_a,
        out_packed_b,
        num_experts,
        group_size,
        wmma_total_rows,
        stream,
        library,
        runtime,
    )


def gemm_awq_selected_pack8_wmma_compact_bf16(
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_t_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_packed: int,
    num_experts: int,
    group_size: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_single(
        _SYMBOL_SINGLE_BF16,
        x_ptr,
        expert_start_compact_ptr,
        expert_start_wmma_ptr,
        tile_expert_ptr,
        qweight_t_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_packed,
        num_experts,
        group_size,
        wmma_total_rows,
        stream,
        library,
        runtime,
    )


def gemm_awq_selected_pack8_wmma_compact_fp16(
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_t_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_packed: int,
    num_experts: int,
    group_size: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_single(
        _SYMBOL_SINGLE_FP16,
        x_ptr,
        expert_start_compact_ptr,
        expert_start_wmma_ptr,
        tile_expert_ptr,
        qweight_t_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_packed,
        num_experts,
        group_size,
        wmma_total_rows,
        stream,
        library,
        runtime,
    )


def register_paro_awq_wmma_kernels(*, replace: bool = True) -> None:
    for quant in ("bf16", "w4_paro"):
        register(
            KernelKey("hip_gfx1100", "awq_wmma", quant, "selected_dual_pack8_compact"),
            gemm_awq_selected_dual_pack8_wmma_compact_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "awq_wmma", quant, "selected_pack8_compact"),
            gemm_awq_selected_pack8_wmma_compact_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "awq_wmma", quant, "selected_dual_pack8_compact_fp16"),
            gemm_awq_selected_dual_pack8_wmma_compact_fp16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "awq_wmma", quant, "selected_pack8_compact_fp16"),
            gemm_awq_selected_pack8_wmma_compact_fp16,
            replace=replace,
        )


def _launch_dual(
    symbol: str,
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_t_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_t_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    wmma_total_rows: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_common(compact_rows, in_features, num_experts, group_size, wmma_total_rows)
    _check_positive(out_packed_a, "out_packed_a")
    _check_positive(out_packed_b, "out_packed_b")
    library = library or build_paro_awq_wmma(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_a_t_ptr),
        ctypes.c_void_p(qzeros_a_ptr),
        ctypes.c_void_p(scales_a_ptr),
        ctypes.c_void_p(qweight_b_t_ptr),
        ctypes.c_void_p(qzeros_b_ptr),
        ctypes.c_void_p(scales_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_packed_a),
        ctypes.c_int64(out_packed_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(group_size),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_single(
    symbol: str,
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_t_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_packed: int,
    num_experts: int,
    group_size: int,
    wmma_total_rows: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_common(compact_rows, in_features, num_experts, group_size, wmma_total_rows)
    _check_positive(out_packed, "out_packed")
    library = library or build_paro_awq_wmma(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_t_ptr),
        ctypes.c_void_p(qzeros_ptr),
        ctypes.c_void_p(scales_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_packed),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(group_size),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _check_common(
    compact_rows: int,
    in_features: int,
    num_experts: int,
    group_size: int,
    wmma_total_rows: int,
) -> None:
    _check_positive(compact_rows, "compact_rows")
    _check_positive(in_features, "in_features")
    _check_positive(num_experts, "num_experts")
    _check_positive(group_size, "group_size")
    _check_positive(wmma_total_rows, "wmma_total_rows")
    if group_size % 16 != 0:
        raise ValueError("group_size must be a multiple of 16")
    if in_features % group_size != 0:
        raise ValueError("in_features must be divisible by group_size")
    if wmma_total_rows % 16 != 0:
        raise ValueError("wmma_total_rows must be a multiple of 16")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_paro_awq_wmma_kernels()
