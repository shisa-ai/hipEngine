"""Wrappers for selected GGUF K-family T16 GEMV decode kernels.

P9.H3 replacement-layout implementation for compact MoE decode.  The wrappers
consume the T16 tile layouts produced by the resident materializer:

* Q4_K gate/up: ``tiles[E, out_tiles16, blocks_per_row, 2368]`` dual output.
* Q4_K / Q5_K / Q6_K down: single-output selected GEMV for the corresponding
  T16 tile layout.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.core.dtype import DType
from hipengine.core.specdec2_scope import (
    physical_exact_rowtiles_enabled,
    q5_t16_physical_rowtile_enabled,
)
from hipengine.kernels.hip_gfx1100 import (
    GGUF_SPECDEC2_PRODUCTION_PHYSICAL_EXACT_ROWTILE_ROWS,
)
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_t16_selected_gemv.hip")
_OUTPUT_NAME = "gguf_t16_selected_gemv.so"

_Q4_DUAL_DIRECT_BF16 = "hipengine_gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out"
_Q4_DUAL_NATURAL_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_dual_natural_gemv_bf16_bf16_out"
)
_Q4_DUAL_NATURAL_TILE8_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_dual_natural_tile8_gemv_"
    "bf16_bf16_out"
)
_Q4_DUAL_NATURAL_TILE8_PARALLEL_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_dual_natural_tile8_parallel_gemv_"
    "bf16_bf16_out"
)
_Q4_DUAL_NATURAL_TILE8_PARALLEL_SILU_PAIRQ_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_pairq_"
    "gemv_bf16_bf16_out"
)
_Q4_DUAL_NATURAL_TILE8_PARALLEL_SILU_PAIRCOEFF_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_"
    "paircoeff_gemv_bf16_bf16_out"
)
_Q4_DUAL_NATURAL_TILE8_PARALLEL_SILU_HALFDOT_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_"
    "halfdot_gemv_bf16_bf16_out"
)
_Q4_DUAL_INTERLEAVED_NATURAL_TILE8_PARALLEL_SILU_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_dual_interleaved_natural_tile8_"
    "parallel_silu_gemv_bf16_bf16_out"
)
_Q4_DUAL_INTERLEAVED_NATURAL_TILE8_PARALLEL_SILU_HALFDOT_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_dual_interleaved_natural_tile8_"
    "parallel_silu_halfdot_gemv_bf16_bf16_out"
)
_Q4_DENSE_DUAL_LOCAL32_SILU_BF16 = (
    "hipengine_gguf_q4_k_t16_dense_dual_local32_silu_gemv_bf16_bf16_out"
)
_Q4_DENSE_SINGLE_LOCAL32_BF16 = (
    "hipengine_gguf_q4_k_t16_dense_single_local32_gemv_bf16_bf16_out"
)
_Q4_QMICRO_DENSE_SINGLE_LOCAL32_BF16 = (
    "hipengine_gguf_q4_k_qmicro_t16_dense_single_local32_gemv_"
    "bf16_bf16_out"
)
_Q4_DENSE_SINGLE_LOCAL32_RESIDUAL_BF16 = (
    "hipengine_gguf_q4_k_t16_dense_single_local32_gemv_"
    "bf16_residual_bf16_out"
)
_Q4_DENSE_DUAL_Q8X2_DP4A_SILU_BF16 = (
    "hipengine_gguf_q4_k_t16_dense_dual_q8_1x2_dp4a_silu_gemv_"
    "bf16_bf16_out"
)
_Q4_DENSE_DUAL_Q8X2_SPLIT_WEIGHT_DP4A_SILU_BF16 = (
    "hipengine_gguf_q4_k_t16_dense_dual_q8_1x2_split_weight_dp4a_silu_"
    "gemv_bf16_bf16_out"
)
_Q4_QMICRO_DENSE_DUAL_Q8X2_SPLIT_WEIGHT_DP4A_SILU_BF16 = (
    "hipengine_gguf_q4_k_qmicro_t16_dense_dual_q8_1x2_split_weight_dp4a_"
    "silu_gemv_bf16_bf16_out"
)
_Q4_QMICRO_DENSE_DUAL_Q8X2_ROWTILE8_DP4A_SILU_BF16 = (
    "hipengine_gguf_q4_k_qmicro_t16_dense_dual_q8_1x2_rowtile8_dp4a_"
    "silu_gemv_bf16_bf16_out"
)
_Q4_DENSE_ROWTILE_BF16 = (
    "hipengine_gguf_q4_k_t16_dense_rowtile_gemv_bf16_bf16_out"
)
_Q4_DENSE_ROWTILE16_W2_BF16 = (
    "hipengine_gguf_q4_k_t16_dense_rowtile16_w2_gemv_bf16_bf16_out"
)
_Q4_DENSE_ROWTILE16_W2_GROUPED_ROWS6_BF16 = (
    "hipengine_gguf_q4_k_t16_dense_rowtile16_w2_grouped_rows6_gemv_"
    "bf16_bf16_out"
)
_Q4_QMICRO_DENSE_ROWTILE_BF16 = (
    "hipengine_gguf_q4_k_qmicro_t16_dense_rowtile_gemv_bf16_bf16_out"
)
_Q4_DENSE_ROWTILE_RESIDUAL_BF16 = (
    "hipengine_gguf_q4_k_t16_dense_rowtile_gemv_bf16_residual_bf16_out"
)
_Q4_DENSE_DUAL_ROWTILE_SILU_BF16 = (
    "hipengine_gguf_q4_k_t16_dense_dual_rowtile_silu_gemv_bf16_bf16_out"
)
_Q4_QMICRO_DENSE_DUAL_ROWTILE_SILU_BF16 = (
    "hipengine_gguf_q4_k_qmicro_t16_dense_dual_rowtile_silu_gemv_"
    "bf16_bf16_out"
)
_Q4_DENSE_ROWTILE_COL4_BF16 = (
    "hipengine_gguf_q4_k_t16_dense_rowtile_col4_gemv_bf16_bf16_out"
)
_Q4_DENSE_DUAL_INTERLEAVED_TILE2_LOCAL32_SILU_BF16 = (
    "hipengine_gguf_q4_k_t16_dense_dual_interleaved_tile2_local32_silu_"
    "gemv_bf16_bf16_out"
)
_Q4_DUAL_DIRECT_FP16 = "hipengine_gguf_q4_k_t16_selected_dual_gemv_fp16_fp16_out"
_Q4_DUAL_PAIRREUSE_DIRECT_BF16 = "hipengine_gguf_q4_k_t16_selected_dual_pairreuse_gemv_bf16_bf16_out"
_Q4_DUAL_SILU_DIRECT_BF16 = "hipengine_gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out"
_Q4_QMICRO_DUAL_DIRECT_BF16 = (
    "hipengine_gguf_q4_k_qmicro_t16_selected_dual_gemv_bf16_bf16_out"
)
_Q4_QMICRO_DUAL_SILU_DIRECT_BF16 = (
    "hipengine_gguf_q4_k_qmicro_t16_selected_dual_silu_gemv_bf16_bf16_out"
)
_Q4_DUAL_DIRECT_Q8_DP4A_BF16 = "hipengine_gguf_q4_k_t16_selected_dual_gemv_q8_1_dp4a_bf16_bf16_out"
_Q4_DUAL_SILU_DIRECT_Q8_DP4A_BF16 = "hipengine_gguf_q4_k_t16_selected_dual_silu_gemv_q8_1_dp4a_bf16_bf16_out"
_Q4_DUAL_SILU_DIRECT_Q8X2_DP4A_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_dual_silu_gemv_q8_1x2_dp4a_"
    "bf16_bf16_out"
)
_Q4_SINGLE_DIRECT_BF16 = "hipengine_gguf_q4_k_t16_selected_gemv_bf16_bf16_out"
_Q4_SINGLE_NATURAL_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_natural_gemv_bf16_bf16_out"
)
_Q4_SINGLE_NATURAL_PARALLEL_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_natural_parallel_gemv_bf16_bf16_out"
)
_Q4_SINGLE_NATURAL_PARALLEL_WEIGHTED_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_natural_parallel_weighted_gemv_"
    "bf16_bf16_out"
)
_Q4_SINGLE_NATURAL_PARALLEL_PAIRCOEFF_WEIGHTED_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_natural_parallel_paircoeff_weighted_"
    "gemv_bf16_bf16_out"
)
_Q4_SINGLE_DIRECT_FP16 = "hipengine_gguf_q4_k_t16_selected_gemv_fp16_fp16_out"
_Q5_DENSE_DIRECT_BF16 = (
    "hipengine_gguf_q5_k_t16_gemv_decode_bf16_bf16_out"
)
_Q5_DENSE_TILE8_BF16 = (
    "hipengine_gguf_q5_k_t16_gemv_decode_tile8_bf16_bf16_out"
)
_Q5_DENSE_ROWTILE_BF16 = (
    "hipengine_gguf_q5_k_t16_gemv_rowtile_bf16_bf16_out"
)
_Q5_DENSE_ROWTILE_COL8_BF16 = (
    "hipengine_gguf_q5_k_t16_gemv_rowtile_col8_bf16_bf16_out"
)
_Q5_DENSE_ROWTILE_GROUPED_ROWS6_BF16 = (
    "hipengine_gguf_q5_k_t16_gemv_rowtile_grouped_rows6_bf16_bf16_out"
)
_Q5_DENSE_ROWTILE_GROUPED_ROWS6_ENV = (
    "HIPENGINE_GGUF_Q5_T16_GROUPED_TARGET_ROWS6"
)
_Q5_SINGLE_DIRECT_BF16 = "hipengine_gguf_q5_k_t16_selected_gemv_bf16_bf16_out"
_Q5_QMICRO_SINGLE_DIRECT_BF16 = (
    "hipengine_gguf_q5_k_qmicro_t16_selected_gemv_bf16_bf16_out"
)
_Q5_SINGLE_QWEN_TILE8_BF16 = (
    "hipengine_gguf_q5_k_t16_selected_qwen_tile8_gemv_bf16_bf16_out"
)
_Q5_SINGLE_PAIRREUSE_DIRECT_BF16 = "hipengine_gguf_q5_k_t16_selected_pairreuse_gemv_bf16_bf16_out"
_Q5_SINGLE_DIRECT_Q8_DP4A_BF16 = "hipengine_gguf_q5_k_t16_selected_gemv_q8_1_dp4a_bf16_bf16_out"
_Q5_SINGLE_DIRECT_FP16 = "hipengine_gguf_q5_k_t16_selected_gemv_fp16_fp16_out"
_Q6_SINGLE_DIRECT_BF16 = "hipengine_gguf_q6_k_t16_selected_gemv_bf16_bf16_out"
_Q6_QMICRO_SINGLE_DIRECT_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_selected_gemv_bf16_bf16_out"
)
_Q6_QMICRO_PLANAR_SINGLE_DIRECT_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_selected_gemv_bf16_bf16_out"
)
_Q6_QMICRO_PLANAR_SINGLE_NATURAL_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_selected_natural_gemv_"
    "bf16_bf16_out"
)
_Q6_QMICRO_PLANAR_SINGLE_NATURAL_PARALLEL_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_selected_natural_parallel_gemv_"
    "bf16_bf16_out"
)
_Q6_QMICRO_PLANAR_SINGLE_NATURAL_PARALLEL_WEIGHTED_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_selected_natural_parallel_weighted_"
    "gemv_bf16_bf16_out"
)
_Q6_SINGLE_PAIRREUSE_DIRECT_BF16 = "hipengine_gguf_q6_k_t16_selected_pairreuse_gemv_bf16_bf16_out"
_Q6_SINGLE_DIRECT_FP16 = "hipengine_gguf_q6_k_t16_selected_gemv_fp16_fp16_out"
_Q4_DUAL_BF16 = "hipengine_gguf_q4_k_t16_selected_dual_gemv_decode_compact_bf16_bf16_out"
_Q4_DUAL_FP16 = "hipengine_gguf_q4_k_t16_selected_dual_gemv_decode_compact_fp16_fp16_out"
_Q4_DUAL_PAIRREUSE_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_dual_pairreuse_gemv_decode_compact_bf16_bf16_out"
)
_Q4_DUAL_GROUPED_SMALLM_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_dual_grouped_smallm_bf16_bf16_out"
)
_Q4_SINGLE_GROUPED_SMALLM_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_grouped_smallm_bf16_bf16_out"
)
_Q6_SINGLE_GROUPED_SMALLM_BF16 = (
    "hipengine_gguf_q6_k_t16_selected_grouped_smallm_bf16_bf16_out"
)
_Q6_QMICRO_SINGLE_GROUPED_SMALLM_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_selected_grouped_smallm_bf16_bf16_out"
)
_Q6_QMICRO_PLANAR_SINGLE_GROUPED_SMALLM_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_selected_grouped_smallm_bf16_bf16_out"
)
_Q4_SINGLE_BF16 = "hipengine_gguf_q4_k_t16_selected_gemv_decode_compact_bf16_bf16_out"
_Q4_SINGLE_FP16 = "hipengine_gguf_q4_k_t16_selected_gemv_decode_compact_fp16_fp16_out"
_Q4_SINGLE_PAIRREUSE_BF16 = (
    "hipengine_gguf_q4_k_t16_selected_pairreuse_gemv_decode_compact_bf16_bf16_out"
)
_Q5_SINGLE_BF16 = "hipengine_gguf_q5_k_t16_selected_gemv_decode_compact_bf16_bf16_out"
_Q5_SINGLE_FP16 = "hipengine_gguf_q5_k_t16_selected_gemv_decode_compact_fp16_fp16_out"
_Q5_SINGLE_PAIRREUSE_BF16 = (
    "hipengine_gguf_q5_k_t16_selected_pairreuse_gemv_decode_compact_bf16_bf16_out"
)
_Q6_SINGLE_BF16 = "hipengine_gguf_q6_k_t16_selected_gemv_decode_compact_bf16_bf16_out"
_Q6_SINGLE_FP16 = "hipengine_gguf_q6_k_t16_selected_gemv_decode_compact_fp16_fp16_out"
_Q6_SINGLE_PAIRREUSE_BF16 = (
    "hipengine_gguf_q6_k_t16_selected_pairreuse_gemv_decode_compact_bf16_bf16_out"
)

_QK_K = 256
_T16_COLS = 16


def plan_gguf_t16_selected_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_t16_selected_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_t16_selected_gemv(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_hip(
        sources=[_SOURCE],
        family="gguf_t16_selected_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


_T16_SELECTED_GEMV_LIBRARY: ctypes.CDLL | None = None


def _t16_selected_gemv_library() -> ctypes.CDLL:
    """Memoized default build so per-call launches skip build_hip entirely.

    The per-call launch path calls ``build_gguf_t16_selected_gemv(load=True)``
    on every launch (~19 us/call of build_hip fast-path overhead even on a
    cache hit); on the Qwen3.6-27B dense decode path ~128 launches/step go
    through this family, so this is ~2.5 ms/step of host CPU on the critical
    path. Hoist it once, mirroring the PN5/PN6 router and GEMV patterns.
    """

    global _T16_SELECTED_GEMV_LIBRARY
    if _T16_SELECTED_GEMV_LIBRARY is None:
        _T16_SELECTED_GEMV_LIBRARY = build_gguf_t16_selected_gemv(load=True)
    return _T16_SELECTED_GEMV_LIBRARY


def gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 selected Q4T16 dual GEMV preserving selected-row order."""

    _launch_dual_direct(
        _Q4_DUAL_DIRECT_BF16,
        x_ptr,
        selected_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_a_ptr,
        out_b_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_natural_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact natural-shape Laguna Q4T16 gate/up GEMV."""

    _check_laguna_natural_selected_shape(
        x_rows,
        rows,
        in_features,
        out_features,
        expected_x_rows=1,
        expected_in=3072,
        expected_out=1024,
    )
    _launch_dual_direct(
        _Q4_DUAL_NATURAL_BF16,
        x_ptr,
        selected_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_a_ptr,
        out_b_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_natural_tile8_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact tile8 natural-shape Laguna Q4T16 gate/up GEMV."""

    _check_laguna_natural_selected_shape(
        x_rows,
        rows,
        in_features,
        out_features,
        expected_x_rows=1,
        expected_in=3072,
        expected_out=1024,
    )
    _launch_dual_direct(
        _Q4_DUAL_NATURAL_TILE8_BF16,
        x_ptr,
        selected_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_a_ptr,
        out_b_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_natural_tile8_parallel_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact tile8 Laguna gate/up with an eight-lane reduction tail."""

    _check_laguna_natural_selected_shape(
        x_rows,
        rows,
        in_features,
        out_features,
        expected_x_rows=1,
        expected_in=3072,
        expected_out=1024,
    )
    _launch_dual_direct(
        _Q4_DUAL_NATURAL_TILE8_PARALLEL_BF16,
        x_ptr,
        selected_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_a_ptr,
        out_b_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_pairreuse_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact BF16 Q4T16 dual GEMV with dynamic duplicate-expert reuse."""

    _launch_dual_direct(
        _Q4_DUAL_PAIRREUSE_DIRECT_BF16,
        x_ptr,
        selected_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_a_ptr,
        out_b_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_gemv_fp16_fp16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 selected Q4T16 dual GEMV preserving selected-row order."""

    _launch_dual_direct(
        _Q4_DUAL_DIRECT_FP16,
        x_ptr,
        selected_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_a_ptr,
        out_b_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact tile8 parallel-tail gate/up fused with BF16 SiLU."""

    _check_laguna_natural_selected_shape(
        x_rows,
        rows,
        in_features,
        out_features,
        expected_x_rows=1,
        expected_in=3072,
        expected_out=1024,
    )
    _launch_dual_silu_direct(
        _Q4_DUAL_NATURAL_TILE8_PARALLEL_SILU_PAIRCOEFF_BF16,
        x_ptr,
        selected_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_paircoeff_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch pair-Q with adjacent coefficient vector loads."""

    _check_laguna_natural_selected_shape(
        x_rows,
        rows,
        in_features,
        out_features,
        expected_x_rows=1,
        expected_in=3072,
        expected_out=1024,
    )
    _launch_dual_silu_direct(
        _Q4_DUAL_NATURAL_TILE8_PARALLEL_SILU_PAIRCOEFF_BF16,
        x_ptr,
        selected_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_halfdot_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the quality-gated adjacent-K FP16 dot2 c=1 screen."""

    _check_laguna_natural_selected_shape(
        x_rows,
        rows,
        in_features,
        out_features,
        expected_x_rows=1,
        expected_in=3072,
        expected_out=1024,
    )
    _launch_dual_silu_direct(
        _Q4_DUAL_NATURAL_TILE8_PARALLEL_SILU_HALFDOT_BF16,
        x_ptr,
        selected_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_interleaved_natural_tile8_parallel_silu_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_dual_ptr: int,
    tiles_unused_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact tile8 gate/up from a dual-interleaved T16 layout."""

    if (
        x_rows <= 0
        or rows != x_rows * 10
        or in_features != 3072
        or out_features != 1024
    ):
        raise ValueError(
            "dual-interleaved Laguna selected GEMV requires positive "
            "x_rows, rows=x_rows*10, in_features=3072, and "
            "out_features=1024"
        )
    _launch_dual_silu_direct(
        _Q4_DUAL_INTERLEAVED_NATURAL_TILE8_PARALLEL_SILU_BF16,
        x_ptr,
        selected_ptr,
        tiles_dual_ptr,
        tiles_unused_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_interleaved_natural_tile8_parallel_silu_halfdot_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_dual_ptr: int,
    tiles_unused_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the interleaved-resident adjacent-K FP16 dot2 screen."""

    if (
        x_rows <= 0
        or rows != x_rows * 10
        or in_features != 3072
        or out_features != 1024
    ):
        raise ValueError(
            "dual-interleaved Laguna selected halfdot GEMV requires positive "
            "x_rows, rows=x_rows*10, in_features=3072, and "
            "out_features=1024"
        )
    _launch_dual_silu_direct(
        _Q4_DUAL_INTERLEAVED_NATURAL_TILE8_PARALLEL_SILU_HALFDOT_BF16,
        x_ptr,
        selected_ptr,
        tiles_dual_ptr,
        tiles_unused_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_pairq_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the exact pre-pair-coefficient pair-Q rollback."""

    _check_laguna_natural_selected_shape(
        x_rows,
        rows,
        in_features,
        out_features,
        expected_x_rows=1,
        expected_in=3072,
        expected_out=1024,
    )
    _launch_dual_silu_direct(
        _Q4_DUAL_NATURAL_TILE8_PARALLEL_SILU_PAIRQ_BF16,
        x_ptr,
        selected_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_dense_dual_local32_silu_bf16_bf16_out(
    x_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the exact local32 dense/shared Q4T16 SiLU decode owner."""

    if rows != 1:
        raise ValueError("dense Q4T16 local32 decode requires rows == 1")
    if in_features <= 0 or in_features % _QK_K:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % _T16_COLS:
        raise ValueError("out_features must be a positive multiple of 16")
    lib = library or _t16_selected_gemv_library()
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_DENSE_DUAL_LOCAL32_SILU_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(tiles_a_ptr),
        ctypes.c_void_p(tiles_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if status != HIP_SUCCESS:
        raise RuntimeError(
            f"{_Q4_DENSE_DUAL_LOCAL32_SILU_BF16} failed with HIP status "
            f"{status}: {rt.error_string(status)}"
        )


def gguf_q4_k_t16_dense_single_local32_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the exact local32 dense/shared Q4T16 single-output owner."""

    if rows != 1:
        raise ValueError("dense Q4T16 local32 decode requires rows == 1")
    if in_features <= 0 or in_features % _QK_K:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % _T16_COLS:
        raise ValueError("out_features must be a positive multiple of 16")
    lib = library or _t16_selected_gemv_library()
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_DENSE_SINGLE_LOCAL32_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if status != HIP_SUCCESS:
        raise RuntimeError(
            f"{_Q4_DENSE_SINGLE_LOCAL32_BF16} failed with HIP status "
            f"{status}: {rt.error_string(status)}"
        )


def gguf_q4_k_qmicro_t16_dense_single_local32_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the exact sole-qmicro local32 primitive."""

    if rows != 1:
        raise ValueError("dense qmicro Q4T16 local32 decode requires rows == 1")
    if in_features <= 0 or in_features % _QK_K:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % _T16_COLS:
        raise ValueError("out_features must be a positive multiple of 16")
    lib = library or _t16_selected_gemv_library()
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_QMICRO_DENSE_SINGLE_LOCAL32_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if status != HIP_SUCCESS:
        raise RuntimeError(
            f"{_Q4_QMICRO_DENSE_SINGLE_LOCAL32_BF16} failed with HIP status "
            f"{status}: {rt.error_string(status)}"
        )


def gguf_q4_k_t16_dense_single_local32_bf16_residual_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact serial-c1 Q4 projection plus rounded residual."""

    if rows != 1:
        raise ValueError("dense Q4T16 local32 residual decode requires rows == 1")
    if in_features <= 0 or in_features % _QK_K:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % _T16_COLS:
        raise ValueError("out_features must be a positive multiple of 16")
    lib = library or _t16_selected_gemv_library()
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_DENSE_SINGLE_LOCAL32_RESIDUAL_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(residual_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if status != HIP_SUCCESS:
        raise RuntimeError(
            f"{_Q4_DENSE_SINGLE_LOCAL32_RESIDUAL_BF16} failed with HIP status "
            f"{status}: {rt.error_string(status)}"
        )


def gguf_q4_k_t16_dense_dual_q8_1x2_dp4a_silu_bf16_bf16_out(
    xq_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch dense residual-Q8_1x2 Q4T16 dual dp4a plus SiLU."""

    if rows != 1:
        raise ValueError("dense Q4T16 Q8_1x2 dp4a decode requires rows == 1")
    if in_features <= 0 or in_features % _QK_K:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % _T16_COLS:
        raise ValueError("out_features must be a positive multiple of 16")
    lib = library or _t16_selected_gemv_library()
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_DENSE_DUAL_Q8X2_DP4A_SILU_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(xq_ptr),
        ctypes.c_void_p(tiles_a_ptr),
        ctypes.c_void_p(tiles_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if status != HIP_SUCCESS:
        raise RuntimeError(
            f"{_Q4_DENSE_DUAL_Q8X2_DP4A_SILU_BF16} failed with HIP status "
            f"{status}: {rt.error_string(status)}"
        )


def gguf_q4_k_t16_dense_dual_q8_1x2_split_weight_dp4a_silu_bf16_bf16_out(
    xq_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the exact four-wave split-weight residual-Q8_1x2 owner."""

    if rows != 1:
        raise ValueError("dense split-weight Q4T16 Q8_1x2 decode requires rows == 1")
    if in_features <= 0 or in_features % _QK_K:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % _T16_COLS:
        raise ValueError("out_features must be a positive multiple of 16")
    lib = library or _t16_selected_gemv_library()
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_DENSE_DUAL_Q8X2_SPLIT_WEIGHT_DP4A_SILU_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(xq_ptr),
        ctypes.c_void_p(tiles_a_ptr),
        ctypes.c_void_p(tiles_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if status != HIP_SUCCESS:
        raise RuntimeError(
            f"{_Q4_DENSE_DUAL_Q8X2_SPLIT_WEIGHT_DP4A_SILU_BF16} failed with "
            f"HIP status {status}: {rt.error_string(status)}"
        )


def gguf_q4_k_qmicro_t16_dense_dual_q8_1x2_split_weight_dp4a_silu_bf16_bf16_out(
    xq_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the qmicro-payload split-weight residual-Q8_1x2 screen."""

    if rows != 1:
        raise ValueError("dense qmicro Q4T16 Q8_1x2 decode requires rows == 1")
    if in_features <= 0 or in_features % _QK_K:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % _T16_COLS:
        raise ValueError("out_features must be a positive multiple of 16")
    lib = library or _t16_selected_gemv_library()
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_QMICRO_DENSE_DUAL_Q8X2_SPLIT_WEIGHT_DP4A_SILU_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(xq_ptr),
        ctypes.c_void_p(tiles_a_ptr),
        ctypes.c_void_p(tiles_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if status != HIP_SUCCESS:
        raise RuntimeError(
            f"{_Q4_QMICRO_DENSE_DUAL_Q8X2_SPLIT_WEIGHT_DP4A_SILU_BF16} "
            f"failed with HIP status {status}: {rt.error_string(status)}"
        )


def gguf_q4_k_qmicro_t16_dense_dual_q8_1x2_rowtile8_dp4a_silu_bf16_bf16_out(
    xq_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact c1-association Q8_1x2 while sharing weights across rows."""

    if rows < 2 or rows > 8:
        raise ValueError("dense qmicro Q8_1x2 rowtile8 requires rows in [2, 8]")
    if in_features <= 0 or in_features % _QK_K:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % _T16_COLS:
        raise ValueError("out_features must be a positive multiple of 16")
    lib = library or _t16_selected_gemv_library()
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_QMICRO_DENSE_DUAL_Q8X2_ROWTILE8_DP4A_SILU_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(xq_ptr),
        ctypes.c_void_p(tiles_a_ptr),
        ctypes.c_void_p(tiles_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if status != HIP_SUCCESS:
        raise RuntimeError(
            f"{_Q4_QMICRO_DENSE_DUAL_Q8X2_ROWTILE8_DP4A_SILU_BF16} "
            f"failed with HIP status {status}: {rt.error_string(status)}"
        )


def gguf_q4_k_t16_dense_rowtile_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the exact compact-T16 row-reuse owner for rows 2-8."""

    _check_dense_q4_t16_rowtile_shape(rows, in_features, out_features, max_rows=8)
    lib = library or _t16_selected_gemv_library()
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_DENSE_ROWTILE_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if status != HIP_SUCCESS:
        raise RuntimeError(
            f"{_Q4_DENSE_ROWTILE_BF16} failed with HIP status "
            f"{status}: {rt.error_string(status)}"
        )

def gguf_q4_k_t16_dense_rowtile16_w2_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the exact two-wave 16-column geometry."""

    _check_dense_q4_t16_rowtile_shape(rows, in_features, out_features, max_rows=8)
    if rows not in {2, 3, 4, 6, 8}:
        raise ValueError("dense Q4 T16 rowtile16-w2 requires rows in {2,3,4,6,8}")
    lib = library or _t16_selected_gemv_library()
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_DENSE_ROWTILE16_W2_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if status != HIP_SUCCESS:
        raise RuntimeError(
            f"{_Q4_DENSE_ROWTILE16_W2_BF16} failed with HIP status "
            f"{status}: {rt.error_string(status)}"
        )


def gguf_q4_k_t16_dense_rowtile16_w2_grouped_rows6_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact row6 chunks through one two-dimensional grid."""

    _check_dense_q4_t16_rowtile_geometry(in_features, out_features)
    if rows < 12 or rows % 6:
        raise ValueError(
            "dense Q4 T16 grouped rowtile16-w2 requires rows >= 12 divisible by 6"
        )
    lib = library or _t16_selected_gemv_library()
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_DENSE_ROWTILE16_W2_GROUPED_ROWS6_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if status != HIP_SUCCESS:
        raise RuntimeError(
            f"{_Q4_DENSE_ROWTILE16_W2_GROUPED_ROWS6_BF16} failed with HIP "
            f"status {status}: {rt.error_string(status)}"
        )


def gguf_q4_k_qmicro_t16_dense_rowtile_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the exact sole-qmicro rows2-4 primitive."""

    _check_dense_q4_t16_rowtile4_shape(rows, in_features, out_features)
    lib = library or _t16_selected_gemv_library()
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_QMICRO_DENSE_ROWTILE_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if status != HIP_SUCCESS:
        raise RuntimeError(
            f"{_Q4_QMICRO_DENSE_ROWTILE_BF16} failed with HIP status "
            f"{status}: {rt.error_string(status)}"
        )


def gguf_q4_k_t16_dense_rowtile_bf16_residual_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact compact-Q4 FFN-down plus rounded-BF16 residual."""

    _check_dense_q4_t16_rowtile4_shape(rows, in_features, out_features)
    lib = library or _t16_selected_gemv_library()
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_DENSE_ROWTILE_RESIDUAL_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(residual_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if status != HIP_SUCCESS:
        raise RuntimeError(
            f"{_Q4_DENSE_ROWTILE_RESIDUAL_BF16} failed with HIP status "
            f"{status}: {rt.error_string(status)}"
        )


def gguf_q4_k_t16_dense_dual_rowtile_silu_bf16_bf16_out(
    x_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the exact two-wave compact-T16 FFN rowtile for rows 2-8."""

    _check_dense_q4_t16_rowtile_shape(rows, in_features, out_features, max_rows=8)
    lib = library or _t16_selected_gemv_library()
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_DENSE_DUAL_ROWTILE_SILU_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(tiles_a_ptr),
        ctypes.c_void_p(tiles_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if status != HIP_SUCCESS:
        raise RuntimeError(
            f"{_Q4_DENSE_DUAL_ROWTILE_SILU_BF16} failed with HIP status "
            f"{status}: {rt.error_string(status)}"
        )


def gguf_q4_k_t16_dense_single_col4_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the exact four-column compact-T16 rows1 projection."""

    if rows != 1:
        raise ValueError("dense Q4T16 col4 decode requires rows == 1")
    if in_features <= 0 or in_features % _QK_K:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % _T16_COLS:
        raise ValueError("out_features must be a positive multiple of 16")
    lib = library or _t16_selected_gemv_library()
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_DENSE_ROWTILE_COL4_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if status != HIP_SUCCESS:
        raise RuntimeError(
            f"{_Q4_DENSE_ROWTILE_COL4_BF16} failed with HIP status "
            f"{status}: {rt.error_string(status)}"
        )


def gguf_q4_k_qmicro_t16_dense_dual_rowtile_silu_bf16_bf16_out(
    x_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the exact qmicro FFN rowtile for rows 2-4."""

    _check_dense_q4_t16_rowtile4_shape(rows, in_features, out_features)
    lib = library or _t16_selected_gemv_library()
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_QMICRO_DENSE_DUAL_ROWTILE_SILU_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(tiles_a_ptr),
        ctypes.c_void_p(tiles_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if status != HIP_SUCCESS:
        raise RuntimeError(
            f"{_Q4_QMICRO_DENSE_DUAL_ROWTILE_SILU_BF16} failed with HIP "
            f"status {status}: {rt.error_string(status)}"
        )


def gguf_q4_k_t16_dense_rowtile_col4_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the exact four-column compact-T16 row-reuse control (rows 1-8)."""

    _check_dense_q4_t16_rowtile_shape(rows, in_features, out_features, max_rows=8)
    lib = library or _t16_selected_gemv_library()
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_DENSE_ROWTILE_COL4_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if status != HIP_SUCCESS:
        raise RuntimeError(
            f"{_Q4_DENSE_ROWTILE_COL4_BF16} failed with HIP status "
            f"{status}: {rt.error_string(status)}"
        )


def _check_dense_q4_t16_rowtile_shape(
    rows: int,
    in_features: int,
    out_features: int,
    *,
    max_rows: int = 4,
) -> None:
    if not 2 <= rows <= int(max_rows):
        raise ValueError(f"dense Q4T16 rowtile requires rows in 2..{int(max_rows)}")
    _check_dense_q4_t16_rowtile_geometry(in_features, out_features)


def _check_dense_q4_t16_rowtile4_shape(
    rows: int,
    in_features: int,
    out_features: int,
) -> None:
    if not 2 <= rows <= 4:
        raise ValueError("dense Q4T16 rowtile requires rows in 2..4")
    _check_dense_q4_t16_rowtile_geometry(in_features, out_features)


def _check_dense_q4_t16_rowtile_geometry(
    in_features: int,
    out_features: int,
) -> None:
    if in_features <= 0 or in_features % _QK_K:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % _T16_COLS:
        raise ValueError("out_features must be a positive multiple of 16")


def launch_physical_row_chunks(
    launch_one,
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    chunks: tuple[int, ...],
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> bool:
    """Run one physical projection as an exact contiguous row partition."""

    total = int(rows)
    counts = tuple(int(value) for value in chunks)
    if (
        not counts
        or sum(counts) != total
        or any(not 2 <= value <= 8 for value in counts)
    ):
        return False
    element = DType.BF16.itemsize
    row_base = 0
    for count in counts:
        launch_one(
            x_ptr + row_base * int(in_features) * element,
            tiles_ptr,
            out_ptr + row_base * int(out_features) * element,
            count,
            int(in_features),
            int(out_features),
            stream=stream,
            library=library,
            runtime=runtime,
        )
        row_base += count
    return True


def launch_physical_rows6_chunked(
    launch_one,
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> bool:
    """Run one physical-scope projection as admitted rows6 launches.

    A verify group padded to a rows6 multiple splits into consecutive rows6
    launches over the same tiles so every launch matches the qualified shape
    bit-for-bit. Returns False when ``rows`` is not a chunkable multiple.
    """

    total = int(rows)
    if total < 12 or total % 6:
        return False
    return launch_physical_row_chunks(
        launch_one,
        x_ptr,
        tiles_ptr,
        out_ptr,
        total,
        in_features,
        out_features,
        (6,) * (total // 6),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _q5_dense_rowtile_grouped_rows6_enabled() -> bool:
    raw = os.environ.get(_Q5_DENSE_ROWTILE_GROUPED_ROWS6_ENV, "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{_Q5_DENSE_ROWTILE_GROUPED_ROWS6_ENV} must be a boolean value"
    )


def gguf_q5_k_t16_gemv_decode_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the one-expert dense Q5T16 producer."""

    if (
        q5_t16_physical_rowtile_enabled()
        and int(rows) >= 12
        and int(rows) % 6 == 0
        and _q5_dense_rowtile_grouped_rows6_enabled()
    ):
        _check_dense_q5_t16_shape(6, in_features, out_features, rowtile=True)
        _launch_dense_q5_t16(
            _Q5_DENSE_ROWTILE_GROUPED_ROWS6_BF16,
            x_ptr,
            tiles_ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            stream=stream,
            library=library,
            runtime=runtime,
        )
        return
    if q5_t16_physical_rowtile_enabled() and launch_physical_rows6_chunked(
        lambda x, tiles, out, row_count, in_f, out_f, **kw: (
            _check_dense_q5_t16_shape(row_count, in_f, out_f, rowtile=True),
            _launch_dense_q5_t16(
                _Q5_DENSE_ROWTILE_BF16,
                x,
                tiles,
                out,
                row_count,
                in_f,
                out_f,
                **kw,
            ),
        ),
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    ):
        return
    physical_rowtile = bool(
        q5_t16_physical_rowtile_enabled()
        and (
            int(rows) == 6
            or (
                physical_exact_rowtiles_enabled()
                and int(rows)
                in GGUF_SPECDEC2_PRODUCTION_PHYSICAL_EXACT_ROWTILE_ROWS
            )
        )
    )
    _check_dense_q5_t16_shape(
        rows,
        in_features,
        out_features,
        rowtile=physical_rowtile,
    )
    _launch_dense_q5_t16(
        _Q5_DENSE_ROWTILE_BF16 if physical_rowtile else _Q5_DENSE_DIRECT_BF16,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q5_k_t16_gemv_decode_tile8_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact dense Q5T16 c1 with eight-column output ownership."""

    if rows != 1:
        raise ValueError("dense Q5T16 tile8 decode requires rows == 1")
    _check_dense_q5_t16_shape(rows, in_features, out_features, rowtile=False)
    _launch_dense_q5_t16(
        _Q5_DENSE_TILE8_BF16,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q5_k_t16_gemv_rowtile_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact four-column Q5T16 row reuse for rows 2-8."""

    _check_dense_q5_t16_shape(rows, in_features, out_features, rowtile=True)
    _launch_dense_q5_t16(
        _Q5_DENSE_ROWTILE_BF16,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q5_k_t16_gemv_rowtile_grouped_rows6_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact Q5 row6 chunks through one two-dimensional grid."""

    if int(rows) < 12 or int(rows) % 6:
        raise ValueError(
            "dense Q5T16 grouped rowtile requires rows >= 12 divisible by 6"
        )
    _check_dense_q5_t16_shape(6, in_features, out_features, rowtile=True)
    _launch_dense_q5_t16(
        _Q5_DENSE_ROWTILE_GROUPED_ROWS6_BF16,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q5_k_t16_gemv_rowtile_col8_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact eight-column Q5T16 row reuse for rows 2-8."""

    _check_dense_q5_t16_shape(rows, in_features, out_features, rowtile=True)
    _launch_dense_q5_t16(
        _Q5_DENSE_ROWTILE_COL8_BF16,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _check_dense_q5_t16_shape(
    rows: int,
    in_features: int,
    out_features: int,
    *,
    rowtile: bool,
) -> None:
    if rowtile:
        if rows not in (2, 3, 4, 5, 6, 7, 8):
            raise ValueError("dense Q5T16 rowtile requires rows in 2..8")
    elif rows <= 0:
        raise ValueError("dense Q5T16 decode requires rows to be positive")
    if in_features <= 0 or in_features % _QK_K:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % _T16_COLS:
        raise ValueError("out_features must be a positive multiple of 16")


def _launch_dense_q5_t16(
    symbol: str,
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    lib = library or _t16_selected_gemv_library()
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if status != HIP_SUCCESS:
        raise RuntimeError(
            f"{symbol} failed with HIP status {status}: {rt.error_string(status)}"
        )


def gguf_q4_k_t16_dense_dual_interleaved_tile2_local32_silu_bf16_bf16_out(
    x_ptr: int,
    tiles_dual_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the retained two-column dual-interleaved dense/shared owner."""

    _launch_dense_dual_interleaved_local32_silu(
        _Q4_DENSE_DUAL_INTERLEAVED_TILE2_LOCAL32_SILU_BF16,
        x_ptr,
        tiles_dual_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _launch_dense_dual_interleaved_local32_silu(
    symbol: str,
    x_ptr: int,
    tiles_dual_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    if rows != 1:
        raise ValueError("dense Q4T16 local32 decode requires rows == 1")
    if in_features <= 0 or in_features % _QK_K:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % _T16_COLS:
        raise ValueError("out_features must be a positive multiple of 16")
    lib = library or _t16_selected_gemv_library()
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(tiles_dual_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if status != HIP_SUCCESS:
        raise RuntimeError(
            f"{symbol} failed with HIP status {status}: "
            f"{rt.error_string(status)}"
        )


def gguf_q4_k_qmicro_t16_selected_dual_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact byte-neutral qmicro Q4 gate/up without fused SiLU."""

    _launch_dual_direct(
        _Q4_QMICRO_DUAL_DIRECT_BF16,
        x_ptr,
        selected_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_a_ptr,
        out_b_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_qmicro_t16_selected_dual_silu_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact byte-neutral qmicro Q4 gate/up fused with BF16 SiLU."""

    _launch_dual_silu_direct(
        _Q4_QMICRO_DUAL_SILU_DIRECT_BF16,
        x_ptr,
        selected_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 selected Q4T16 dual GEMV fused with split-kernel-equivalent SiLU."""

    _launch_dual_silu_direct(
        _Q4_DUAL_SILU_DIRECT_BF16,
        x_ptr,
        selected_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out(
    xq_ptr: int,
    selected_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 selected Q4T16 dual GEMV from prequantized q8_1 activations."""

    _launch_dual_direct(
        _Q4_DUAL_DIRECT_Q8_DP4A_BF16,
        xq_ptr,
        selected_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_a_ptr,
        out_b_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_silu_q8_1_dp4a_gemv_bf16_bf16_out(
    xq_ptr: int,
    selected_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 selected Q4T16 q8_1+dp4a dual GEMV fused with SiLU."""

    _launch_dual_silu_direct(
        _Q4_DUAL_SILU_DIRECT_Q8_DP4A_BF16,
        xq_ptr,
        selected_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_silu_q8_1x2_dp4a_gemv_bf16_bf16_out(
    xq_ptr: int,
    selected_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch residual-Q8_1x2 Q4T16 dual dp4a fused with SiLU."""

    _launch_dual_silu_direct(
        _Q4_DUAL_SILU_DIRECT_Q8X2_DP4A_BF16,
        xq_ptr,
        selected_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 selected Q4T16 single-output GEMV preserving selected-row order."""

    _launch_single_direct(
        _Q4_SINGLE_DIRECT_BF16,
        x_ptr,
        selected_ptr,
        tiles_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_natural_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact natural-shape Laguna Q4T16 down GEMV."""

    _check_laguna_natural_selected_shape(
        x_rows,
        rows,
        in_features,
        out_features,
        expected_x_rows=10,
        expected_in=1024,
        expected_out=3072,
    )
    _launch_single_direct(
        _Q4_SINGLE_NATURAL_BF16,
        x_ptr,
        selected_ptr,
        tiles_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_natural_parallel_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact natural Q4T16 down GEMV with a 16-lane tail."""

    _check_laguna_natural_selected_shape(
        x_rows,
        rows,
        in_features,
        out_features,
        expected_x_rows=10,
        expected_in=1024,
        expected_out=3072,
    )
    _launch_single_direct(
        _Q4_SINGLE_NATURAL_PARALLEL_BF16,
        x_ptr,
        selected_ptr,
        tiles_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_natural_parallel_weighted_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    routing_weights_ptr: int,
    routed_out_ptr: int,
    completion_counter_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch route-parallel natural Q4T16 down plus exact weighted tail."""

    _check_laguna_natural_selected_shape(
        x_rows,
        rows,
        in_features,
        out_features,
        expected_x_rows=10,
        expected_in=1024,
        expected_out=3072,
    )
    _launch_single_direct_weighted(
        _Q4_SINGLE_NATURAL_PARALLEL_WEIGHTED_BF16,
        x_ptr,
        selected_ptr,
        tiles_ptr,
        out_ptr,
        routing_weights_ptr,
        routed_out_ptr,
        completion_counter_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_gemv_fp16_fp16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 selected Q4T16 single-output GEMV preserving selected-row order."""

    _launch_single_direct(
        _Q4_SINGLE_DIRECT_FP16,
        x_ptr,
        selected_ptr,
        tiles_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_natural_parallel_paircoeff_weighted_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    routing_weights_ptr: int,
    routed_out_ptr: int,
    completion_counter_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch paired-payload natural Q4T16 down plus exact weighted tail."""

    _check_laguna_natural_selected_shape(
        x_rows,
        rows,
        in_features,
        out_features,
        expected_x_rows=10,
        expected_in=1024,
        expected_out=3072,
    )
    _launch_single_direct_weighted(
        _Q4_SINGLE_NATURAL_PARALLEL_PAIRCOEFF_WEIGHTED_BF16,
        x_ptr,
        selected_ptr,
        tiles_ptr,
        out_ptr,
        routing_weights_ptr,
        routed_out_ptr,
        completion_counter_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q5_k_qmicro_t16_selected_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact byte-neutral qmicro Q5 selected-down GEMV."""

    _launch_single_direct(
        _Q5_QMICRO_SINGLE_DIRECT_BF16,
        x_ptr,
        selected_ptr,
        tiles_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q5_k_qmicro_t16_selected_qwen_tile8_gemv_bf16_bf16_out(
    *args,
    **kwargs,
) -> None:
    """Qwen-shaped alias for the exact eight-column compact Q5 owner."""

    _check_qwen_selected_down_shape(args[4], args[5], args[7], args[8])
    gguf_q5_k_qmicro_t16_selected_gemv_bf16_bf16_out(*args, **kwargs)


def gguf_q5_k_t16_selected_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 selected Q5T16 GEMV preserving selected-row order."""

    _launch_single_direct(
        _Q5_SINGLE_DIRECT_BF16,
        x_ptr,
        selected_ptr,
        tiles_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q5_k_t16_selected_qwen_tile8_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the exact Qwen Q5T16 selected-down tile8 screen."""

    _check_qwen_selected_down_shape(x_rows, rows, in_features, out_features)
    _launch_single_direct(
        _Q5_SINGLE_QWEN_TILE8_BF16,
        x_ptr,
        selected_ptr,
        tiles_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q5_k_t16_selected_pairreuse_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact BF16 Q5T16 selected-down with dynamic expert pair reuse."""

    _launch_single_direct(
        _Q5_SINGLE_PAIRREUSE_DIRECT_BF16,
        x_ptr,
        selected_ptr,
        tiles_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q5_k_t16_selected_gemv_fp16_fp16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 selected Q5T16 GEMV preserving selected-row order."""

    _launch_single_direct(
        _Q5_SINGLE_DIRECT_FP16,
        x_ptr,
        selected_ptr,
        tiles_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q5_k_t16_selected_q8_1_dp4a_gemv_bf16_bf16_out(
    xq_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 selected Q5T16 GEMV from prequantized q8_1 activations."""

    _launch_single_direct(
        _Q5_SINGLE_DIRECT_Q8_DP4A_BF16,
        xq_ptr,
        selected_ptr,
        tiles_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_selected_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    qmicro: bool = False,
    qmicro_planar: bool = False,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 selected Q6T16 GEMV preserving selected-row order."""

    if qmicro_planar and not qmicro:
        raise ValueError("qmicro_planar requires qmicro=True")
    _launch_single_direct(
        (
            _Q6_QMICRO_PLANAR_SINGLE_DIRECT_BF16
            if qmicro_planar
            else _Q6_QMICRO_SINGLE_DIRECT_BF16
            if qmicro
            else _Q6_SINGLE_DIRECT_BF16
        ),
        x_ptr,
        selected_ptr,
        tiles_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_qmicro_planar_selected_natural_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact natural-shape Laguna planar-Q6T16 down GEMV."""

    _check_laguna_natural_selected_shape(
        x_rows,
        rows,
        in_features,
        out_features,
        expected_x_rows=10,
        expected_in=1024,
        expected_out=3072,
    )
    _launch_single_direct(
        _Q6_QMICRO_PLANAR_SINGLE_NATURAL_BF16,
        x_ptr,
        selected_ptr,
        tiles_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_qmicro_planar_selected_natural_parallel_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact natural planar-Q6T16 down GEMV with a 16-lane tail."""

    _check_laguna_natural_selected_shape(
        x_rows,
        rows,
        in_features,
        out_features,
        expected_x_rows=10,
        expected_in=1024,
        expected_out=3072,
    )
    _launch_single_direct(
        _Q6_QMICRO_PLANAR_SINGLE_NATURAL_PARALLEL_BF16,
        x_ptr,
        selected_ptr,
        tiles_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_qmicro_planar_selected_natural_parallel_weighted_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    routing_weights_ptr: int,
    routed_out_ptr: int,
    completion_counter_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch route-parallel planar-Q6T16 down plus exact weighted tail."""

    _check_laguna_natural_selected_shape(
        x_rows,
        rows,
        in_features,
        out_features,
        expected_x_rows=10,
        expected_in=1024,
        expected_out=3072,
    )
    _launch_single_direct_weighted(
        _Q6_QMICRO_PLANAR_SINGLE_NATURAL_PARALLEL_WEIGHTED_BF16,
        x_ptr,
        selected_ptr,
        tiles_ptr,
        out_ptr,
        routing_weights_ptr,
        routed_out_ptr,
        completion_counter_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_selected_pairreuse_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact BF16 Q6T16 selected-down with dynamic expert pair reuse."""

    _launch_single_direct(
        _Q6_SINGLE_PAIRREUSE_DIRECT_BF16,
        x_ptr,
        selected_ptr,
        tiles_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_selected_gemv_fp16_fp16_out(
    x_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 selected Q6T16 GEMV preserving selected-row order."""

    _launch_single_direct(
        _Q6_SINGLE_DIRECT_FP16,
        x_ptr,
        selected_ptr,
        tiles_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_grouped_smallm_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    active_experts_ptr: int,
    active_count_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact BF16 Q4T16 gate/up over grouped 1/2/4/8-row buckets."""

    _launch_grouped_dual(
        _Q4_DUAL_GROUPED_SMALLM_BF16,
        x_ptr,
        expert_start_compact_ptr,
        active_experts_ptr,
        active_count_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_a_ptr,
        out_b_ptr,
        compact_rows,
        in_features,
        out_features,
        num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_gemv_decode_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 selected compact Q4T16 dual gate/up GEMV decode."""

    _launch_dual(
        _Q4_DUAL_BF16,
        x_ptr,
        expert_start_compact_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_pairreuse_gemv_decode_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact BF16 compact Q4T16 dual GEMV with adjacent pair reuse."""

    _launch_dual(
        _Q4_DUAL_PAIRREUSE_BF16,
        x_ptr,
        expert_start_compact_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_gemv_decode_compact_fp16_fp16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 selected compact Q4T16 dual gate/up GEMV decode."""

    _launch_dual(
        _Q4_DUAL_FP16,
        x_ptr,
        expert_start_compact_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_gemv_decode_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 selected compact Q4T16 single-output GEMV decode."""

    _launch_single(
        _Q4_SINGLE_BF16,
        x_ptr,
        expert_start_compact_ptr,
        tiles_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features,
        num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_grouped_smallm_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    active_experts_ptr: int,
    active_count_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact BF16 Q4T16 down over grouped 1/2/4/8-row buckets."""

    _launch_grouped_single(
        _Q4_SINGLE_GROUPED_SMALLM_BF16,
        x_ptr,
        expert_start_compact_ptr,
        active_experts_ptr,
        active_count_ptr,
        tiles_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features,
        num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_pairreuse_gemv_decode_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact BF16 compact Q4T16 down GEMV with adjacent pair reuse."""

    _launch_single(
        _Q4_SINGLE_PAIRREUSE_BF16,
        x_ptr,
        expert_start_compact_ptr,
        tiles_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features,
        num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_gemv_decode_compact_fp16_fp16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 selected compact Q4T16 single-output GEMV decode."""

    _launch_single(
        _Q4_SINGLE_FP16,
        x_ptr,
        expert_start_compact_ptr,
        tiles_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features,
        num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q5_k_t16_selected_gemv_decode_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 selected compact Q5T16 down GEMV decode."""

    _launch_single(
        _Q5_SINGLE_BF16,
        x_ptr,
        expert_start_compact_ptr,
        tiles_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features,
        num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q5_k_t16_selected_pairreuse_gemv_decode_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact BF16 compact Q5T16 down GEMV with adjacent pair reuse."""

    _launch_single(
        _Q5_SINGLE_PAIRREUSE_BF16,
        x_ptr,
        expert_start_compact_ptr,
        tiles_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features,
        num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q5_k_t16_selected_gemv_decode_compact_fp16_fp16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 selected compact Q5T16 down GEMV decode."""

    _launch_single(
        _Q5_SINGLE_FP16,
        x_ptr,
        expert_start_compact_ptr,
        tiles_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features,
        num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_selected_gemv_decode_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 selected compact Q6T16 down GEMV decode."""

    _launch_single(
        _Q6_SINGLE_BF16,
        x_ptr,
        expert_start_compact_ptr,
        tiles_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features,
        num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_selected_grouped_smallm_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    active_experts_ptr: int,
    active_count_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    qmicro: bool = False,
    qmicro_planar: bool = False,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact BF16 Q6T16 down over grouped 1/2/4/8-row buckets."""

    if qmicro_planar and not qmicro:
        raise ValueError("qmicro_planar requires qmicro=True")
    _launch_grouped_single(
        (
            _Q6_QMICRO_PLANAR_SINGLE_GROUPED_SMALLM_BF16
            if qmicro_planar
            else _Q6_QMICRO_SINGLE_GROUPED_SMALLM_BF16
            if qmicro
            else _Q6_SINGLE_GROUPED_SMALLM_BF16
        ),
        x_ptr,
        expert_start_compact_ptr,
        active_experts_ptr,
        active_count_ptr,
        tiles_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features,
        num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_selected_pairreuse_gemv_decode_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact BF16 compact Q6T16 down GEMV with adjacent pair reuse."""

    _launch_single(
        _Q6_SINGLE_PAIRREUSE_BF16,
        x_ptr,
        expert_start_compact_ptr,
        tiles_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features,
        num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_selected_gemv_decode_compact_fp16_fp16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 selected compact Q6T16 down GEMV decode."""

    _launch_single(
        _Q6_SINGLE_FP16,
        x_ptr,
        expert_start_compact_ptr,
        tiles_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features,
        num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _launch_grouped_dual(
    symbol: str,
    x_ptr: int,
    expert_start_compact_ptr: int,
    active_experts_ptr: int,
    active_count_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_common(compact_rows, in_features, num_experts)
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if out_features % _T16_COLS != 0:
        raise ValueError("out_features must be a multiple of 16 (T16 tile)")
    library = library or _t16_selected_gemv_library()
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(active_experts_ptr),
        ctypes.c_void_p(active_count_ptr),
        ctypes.c_void_p(tiles_a_ptr),
        ctypes.c_void_p(tiles_b_ptr),
        ctypes.c_void_p(out_a_ptr),
        ctypes.c_void_p(out_b_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_dual(
    symbol: str,
    x_ptr: int,
    expert_start_compact_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_common(compact_rows, in_features, num_experts)
    if out_features_a <= 0:
        raise ValueError("out_features_a must be positive")
    if out_features_b <= 0:
        raise ValueError("out_features_b must be positive")
    if out_features_a % _T16_COLS != 0:
        raise ValueError("out_features_a must be a multiple of 16 (T16 tile)")
    if out_features_b % _T16_COLS != 0:
        raise ValueError("out_features_b must be a multiple of 16 (T16 tile)")
    library = library or _t16_selected_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(tiles_a_ptr),
        ctypes.c_void_p(tiles_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_grouped_single(
    symbol: str,
    x_ptr: int,
    expert_start_compact_ptr: int,
    active_experts_ptr: int,
    active_count_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_common(compact_rows, in_features, num_experts)
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if out_features % _T16_COLS != 0:
        raise ValueError("out_features must be a multiple of 16 (T16 tile)")
    library = library or _t16_selected_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(active_experts_ptr),
        ctypes.c_void_p(active_count_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_single(
    symbol: str,
    x_ptr: int,
    expert_start_compact_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_common(compact_rows, in_features, num_experts)
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if out_features % _T16_COLS != 0:
        raise ValueError("out_features must be a multiple of 16 (T16 tile)")
    library = library or _t16_selected_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_dual_direct(
    symbol: str,
    x_ptr: int,
    selected_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_direct_common(x_rows, rows, in_features, num_experts)
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if out_features % _T16_COLS != 0:
        raise ValueError("out_features must be a multiple of 16 (T16 tile)")
    library = library or _t16_selected_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(tiles_a_ptr),
        ctypes.c_void_p(tiles_b_ptr),
        ctypes.c_void_p(out_a_ptr),
        ctypes.c_void_p(out_b_ptr),
        ctypes.c_int64(x_rows),
        ctypes.c_int64(rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_dual_silu_direct(
    symbol: str,
    x_ptr: int,
    selected_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_direct_common(x_rows, rows, in_features, num_experts)
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if out_features % _T16_COLS != 0:
        raise ValueError("out_features must be a multiple of 16 (T16 tile)")
    library = library or _t16_selected_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(tiles_a_ptr),
        ctypes.c_void_p(tiles_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(x_rows),
        ctypes.c_int64(rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_single_direct(
    symbol: str,
    x_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_direct_common(x_rows, rows, in_features, num_experts)
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if out_features % _T16_COLS != 0:
        raise ValueError("out_features must be a multiple of 16 (T16 tile)")
    library = library or _t16_selected_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(x_rows),
        ctypes.c_int64(rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_single_direct_weighted(
    symbol: str,
    x_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    routing_weights_ptr: int,
    routed_out_ptr: int,
    completion_counter_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_direct_common(x_rows, rows, in_features, num_experts)
    if out_features <= 0 or out_features % _T16_COLS != 0:
        raise ValueError("out_features must be a positive multiple of 16")
    if min(
        int(routing_weights_ptr),
        int(routed_out_ptr),
        int(completion_counter_ptr),
    ) <= 0:
        raise ValueError(
            "routing weights, routed output, and completion counter "
            "pointers must be nonzero"
        )
    library = library or _t16_selected_gemv_library()
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
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(routing_weights_ptr),
        ctypes.c_void_p(routed_out_ptr),
        ctypes.c_void_p(completion_counter_ptr),
        ctypes.c_int64(x_rows),
        ctypes.c_int64(rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _check_qwen_selected_down_shape(
    x_rows: int,
    rows: int,
    in_features: int,
    out_features: int,
) -> None:
    if (
        x_rows != 8
        or rows != 8
        or in_features != 512
        or out_features != 2048
    ):
        raise ValueError(
            "Qwen selected-down tile8 requires x_rows=rows=8, "
            "in_features=512, and out_features=2048"
        )


def _check_laguna_natural_selected_shape(
    x_rows: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    expected_x_rows: int,
    expected_in: int,
    expected_out: int,
) -> None:
    if (
        x_rows != expected_x_rows
        or rows != 10
        or in_features != expected_in
        or out_features != expected_out
    ):
        raise ValueError(
            "natural Laguna selected GEMV requires "
            f"x_rows={expected_x_rows}, rows=10, "
            f"in_features={expected_in}, and out_features={expected_out}"
        )


def _check_direct_common(x_rows: int, rows: int, in_features: int, num_experts: int) -> None:
    if x_rows <= 0:
        raise ValueError("x_rows must be positive")
    if rows <= 0:
        raise ValueError("rows must be positive")
    if rows % x_rows != 0:
        raise ValueError("rows must be divisible by x_rows")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if in_features % _QK_K != 0:
        raise ValueError("in_features must be divisible by GGUF K block size 256")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")


def _check_common(compact_rows: int, in_features: int, num_experts: int) -> None:
    if compact_rows <= 0:
        raise ValueError("compact_rows must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if in_features % _QK_K != 0:
        raise ValueError("in_features must be divisible by GGUF K block size 256")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")


def register_gguf_t16_selected_gemv_kernels(*, replace: bool = True) -> None:
    """Register compact selected T16 GEMV decode kernels."""

    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q4_k_t16_v1",
            "dense_single_local32_bf16_bf16_out",
        ),
        gguf_q4_k_t16_dense_single_local32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q4_k_qmicro_t16_v1",
            "dense_single_local32_bf16_bf16_out",
        ),
        gguf_q4_k_qmicro_t16_dense_single_local32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear+residual",
            "gguf_q4_k_t16_v1",
            "dense_single_local32_bf16_residual_bf16_out",
        ),
        gguf_q4_k_t16_dense_single_local32_bf16_residual_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q4_k_t16_v1",
            "dense_single_col4_bf16_bf16_out",
        ),
        gguf_q4_k_t16_dense_single_col4_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_q4_k_t16_v1",
            "dense_dual_local32_bf16_bf16_out",
        ),
        gguf_q4_k_t16_dense_dual_local32_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_q4_k_t16_v1",
            "dense_dual_q8_1x2_dp4a_bf16_bf16_out",
        ),
        gguf_q4_k_t16_dense_dual_q8_1x2_dp4a_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_q4_k_t16_v1",
            "dense_dual_q8_1x2_split_weight_dp4a_bf16_bf16_out",
        ),
        gguf_q4_k_t16_dense_dual_q8_1x2_split_weight_dp4a_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_q4_k_qmicro_t16_v1",
            "dense_dual_q8_1x2_split_weight_dp4a_bf16_bf16_out",
        ),
        gguf_q4_k_qmicro_t16_dense_dual_q8_1x2_split_weight_dp4a_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_q4_k_qmicro_t16_v1",
            "dense_dual_q8_1x2_rowtile8_dp4a_bf16_bf16_out",
        ),
        gguf_q4_k_qmicro_t16_dense_dual_q8_1x2_rowtile8_dp4a_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q4_k_t16_v1",
            "dense_rowtile_bf16_bf16_out",
        ),
        gguf_q4_k_t16_dense_rowtile_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q4_k_t16_v1",
            "dense_rowtile16_w2_bf16_bf16_out",
        ),
        gguf_q4_k_t16_dense_rowtile16_w2_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q4_k_t16_v1",
            "dense_rowtile16_w2_grouped_rows6_bf16_bf16_out",
        ),
        gguf_q4_k_t16_dense_rowtile16_w2_grouped_rows6_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q4_k_qmicro_t16_v1",
            "dense_rowtile_bf16_bf16_out",
        ),
        gguf_q4_k_qmicro_t16_dense_rowtile_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear+residual",
            "gguf_q4_k_t16_v1",
            "dense_rowtile_bf16_residual_bf16_out",
        ),
        gguf_q4_k_t16_dense_rowtile_bf16_residual_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_q4_k_t16_v1",
            "dense_dual_rowtile_bf16_bf16_out",
        ),
        gguf_q4_k_t16_dense_dual_rowtile_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_q4_k_qmicro_t16_v1",
            "dense_dual_rowtile_bf16_bf16_out",
        ),
        gguf_q4_k_qmicro_t16_dense_dual_rowtile_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q4_k_t16_v1",
            "dense_rowtile_col4_bf16_bf16_out",
        ),
        gguf_q4_k_t16_dense_rowtile_col4_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q5_k_t16_v1",
            "t16_gemv_decode_bf16_bf16_out",
        ),
        gguf_q5_k_t16_gemv_decode_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q5_k_t16_v1",
            "t16_gemv_decode_tile8_bf16_bf16_out",
        ),
        gguf_q5_k_t16_gemv_decode_tile8_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q5_k_t16_v1",
            "t16_gemv_rowtile_bf16_bf16_out",
        ),
        gguf_q5_k_t16_gemv_rowtile_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q5_k_t16_v1",
            "t16_gemv_rowtile_col8_bf16_bf16_out",
        ),
        gguf_q5_k_t16_gemv_rowtile_col8_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q5_k_t16_v1",
            "t16_gemv_rowtile_grouped_rows6_bf16_bf16_out",
        ),
        gguf_q5_k_t16_gemv_rowtile_grouped_rows6_bf16_bf16_out,
        replace=replace,
    )
    for variant, fn in (
        (
            "selected_dual_t16_gemv_decode_bf16_bf16_out",
            gguf_q4_k_qmicro_t16_selected_dual_gemv_bf16_bf16_out,
        ),
        (
            "selected_dual_silu_gemv_decode_bf16_bf16_out",
            gguf_q4_k_qmicro_t16_selected_dual_silu_gemv_bf16_bf16_out,
        ),
    ):
        register(
            KernelKey(
                "hip_gfx1100",
                "moe_linear",
                "gguf_q4_k_qmicro_t16_v1",
                variant,
            ),
            fn,
            replace=replace,
        )

    for variant, fn in (
        (
            "selected_gemv_decode_bf16_bf16_out",
            gguf_q5_k_qmicro_t16_selected_gemv_bf16_bf16_out,
        ),
        (
            "selected_t16_gemv_decode_bf16_bf16_out",
            gguf_q5_k_qmicro_t16_selected_gemv_bf16_bf16_out,
        ),
        (
            "selected_t16_qwen_tile8_gemv_decode_bf16_bf16_out",
            gguf_q5_k_qmicro_t16_selected_qwen_tile8_gemv_bf16_bf16_out,
        ),
    ):
        register(
            KernelKey(
                "hip_gfx1100",
                "moe_linear",
                "gguf_q5_k_qmicro_t16_v1",
                variant,
            ),
            fn,
            replace=replace,
        )

    for variant, fn in (
        (
            "selected_dual_t16_gemv_decode_compact_bf16_bf16_out",
            gguf_q4_k_t16_selected_dual_gemv_decode_compact_bf16_bf16_out,
        ),
        (
            "selected_dual_t16_gemv_decode_compact_fp16_fp16_out",
            gguf_q4_k_t16_selected_dual_gemv_decode_compact_fp16_fp16_out,
        ),
        (
            "selected_dual_t16_grouped_smallm_bf16_bf16_out",
            gguf_q4_k_t16_selected_dual_grouped_smallm_bf16_bf16_out,
        ),
        (
            "selected_dual_t16_pairreuse_gemv_decode_compact_bf16_bf16_out",
            gguf_q4_k_t16_selected_dual_pairreuse_gemv_decode_compact_bf16_bf16_out,
        ),
        (
            "selected_dual_t16_gemv_decode_bf16_bf16_out",
            gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
        ),
        (
            "selected_dual_t16_natural_gemv_decode_bf16_bf16_out",
            gguf_q4_k_t16_selected_dual_natural_gemv_bf16_bf16_out,
        ),
        (
            "selected_dual_t16_natural_tile8_gemv_decode_bf16_bf16_out",
            gguf_q4_k_t16_selected_dual_natural_tile8_gemv_bf16_bf16_out,
        ),
        (
            "selected_dual_t16_natural_tile8_parallel_gemv_decode_bf16_bf16_out",
            gguf_q4_k_t16_selected_dual_natural_tile8_parallel_gemv_bf16_bf16_out,
        ),
        (
            "selected_dual_t16_natural_tile8_parallel_silu_gemv_decode_bf16_bf16_out",
            gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_gemv_bf16_bf16_out,
        ),
        (
            "selected_dual_t16_natural_tile8_parallel_silu_paircoeff_gemv_decode_bf16_bf16_out",
            gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_paircoeff_gemv_bf16_bf16_out,
        ),
        (
            "selected_dual_t16_natural_tile8_parallel_silu_pairq_gemv_decode_bf16_bf16_out",
            gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_pairq_gemv_bf16_bf16_out,
        ),
        (
            "selected_dual_t16_gemv_decode_fp16_fp16_out",
            gguf_q4_k_t16_selected_dual_gemv_fp16_fp16_out,
        ),
        (
            "selected_dual_t16_pairreuse_gemv_decode_bf16_bf16_out",
            gguf_q4_k_t16_selected_dual_pairreuse_gemv_bf16_bf16_out,
        ),
        (
            "selected_dual_t16_silu_gemv_decode_bf16_bf16_out",
            gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out,
        ),
        (
            "selected_dual_t16_q8_1_dp4a_gemv_decode_bf16_bf16_out",
            gguf_q4_k_t16_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out,
        ),
        (
            "selected_dual_t16_silu_q8_1_dp4a_gemv_decode_bf16_bf16_out",
            gguf_q4_k_t16_selected_dual_silu_q8_1_dp4a_gemv_bf16_bf16_out,
        ),
    ):
        register(
            KernelKey("hip_gfx1100", "moe_linear", "gguf_q4_k_t16_v1", variant),
            fn,
            replace=replace,
        )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q4_k_t16_dual_interleaved_v1",
            (
                "selected_dual_t16_natural_tile8_parallel_silu_"
                "gemv_decode_bf16_bf16_out"
            ),
        ),
        gguf_q4_k_t16_selected_dual_interleaved_natural_tile8_parallel_silu_gemv_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q4_k_t16_dual_interleaved_v1",
            (
                "selected_dual_t16_natural_tile8_parallel_silu_halfdot_"
                "gemv_decode_bf16_bf16_out"
            ),
        ),
        gguf_q4_k_t16_selected_dual_interleaved_natural_tile8_parallel_silu_halfdot_gemv_bf16_bf16_out,
        replace=replace,
    )

    for quant_key, fn_bf16, fn_fp16, direct_bf16, direct_fp16 in (
        (
            "gguf_q4_k_t16_v1",
            gguf_q4_k_t16_selected_gemv_decode_compact_bf16_bf16_out,
            gguf_q4_k_t16_selected_gemv_decode_compact_fp16_fp16_out,
            gguf_q4_k_t16_selected_gemv_bf16_bf16_out,
            gguf_q4_k_t16_selected_gemv_fp16_fp16_out,
        ),
        (
            "gguf_q5_k_t16_v1",
            gguf_q5_k_t16_selected_gemv_decode_compact_bf16_bf16_out,
            gguf_q5_k_t16_selected_gemv_decode_compact_fp16_fp16_out,
            gguf_q5_k_t16_selected_gemv_bf16_bf16_out,
            gguf_q5_k_t16_selected_gemv_fp16_fp16_out,
        ),
        (
            "gguf_q6_k_t16_v1",
            gguf_q6_k_t16_selected_gemv_decode_compact_bf16_bf16_out,
            gguf_q6_k_t16_selected_gemv_decode_compact_fp16_fp16_out,
            gguf_q6_k_t16_selected_gemv_bf16_bf16_out,
            gguf_q6_k_t16_selected_gemv_fp16_fp16_out,
        ),
    ):
        register(
            KernelKey("hip_gfx1100", "moe_linear", quant_key, "selected_t16_gemv_decode_compact_bf16_bf16_out"),
            fn_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "moe_linear", quant_key, "selected_t16_gemv_decode_compact_fp16_fp16_out"),
            fn_fp16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "moe_linear", quant_key, "selected_t16_gemv_decode_bf16_bf16_out"),
            direct_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "moe_linear", quant_key, "selected_t16_gemv_decode_fp16_fp16_out"),
            direct_fp16,
            replace=replace,
        )

    for quant_key, natural_fn in (
        (
            "gguf_q4_k_t16_v1",
            gguf_q4_k_t16_selected_natural_gemv_bf16_bf16_out,
        ),
        (
            "gguf_q6_k_t16_v1",
            gguf_q6_k_t16_qmicro_planar_selected_natural_gemv_bf16_bf16_out,
        ),
    ):
        register(
            KernelKey(
                "hip_gfx1100",
                "moe_linear",
                quant_key,
                "selected_t16_natural_gemv_decode_bf16_bf16_out",
            ),
            natural_fn,
            replace=replace,
        )

    for quant_key, natural_parallel_fn in (
        (
            "gguf_q4_k_t16_v1",
            gguf_q4_k_t16_selected_natural_parallel_gemv_bf16_bf16_out,
        ),
        (
            "gguf_q6_k_t16_v1",
            gguf_q6_k_t16_qmicro_planar_selected_natural_parallel_gemv_bf16_bf16_out,
        ),
    ):
        register(
            KernelKey(
                "hip_gfx1100",
                "moe_linear",
                quant_key,
                "selected_t16_natural_parallel_gemv_decode_bf16_bf16_out",
            ),
            natural_parallel_fn,
            replace=replace,
        )

    for quant_key, natural_parallel_weighted_fn in (
        (
            "gguf_q4_k_t16_v1",
            gguf_q4_k_t16_selected_natural_parallel_weighted_gemv_bf16_bf16_out,
        ),
        (
            "gguf_q6_k_t16_v1",
            gguf_q6_k_t16_qmicro_planar_selected_natural_parallel_weighted_gemv_bf16_bf16_out,
        ),
    ):
        register(
            KernelKey(
                "hip_gfx1100",
                "moe_linear+weighted_sum",
                quant_key,
                "selected_t16_natural_parallel_weighted_gemv_decode_"
                "bf16_bf16_out",
            ),
            natural_parallel_weighted_fn,
            replace=replace,
        )

    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear+weighted_sum",
            "gguf_q4_k_t16_v1",
            "selected_t16_natural_parallel_paircoeff_weighted_"
            "gemv_decode_bf16_bf16_out",
        ),
        gguf_q4_k_t16_selected_natural_parallel_paircoeff_weighted_gemv_bf16_bf16_out,
        replace=replace,
    )

    for quant_key, grouped_smallm_fn in (
        (
            "gguf_q4_k_t16_v1",
            gguf_q4_k_t16_selected_grouped_smallm_bf16_bf16_out,
        ),
        (
            "gguf_q6_k_t16_v1",
            gguf_q6_k_t16_selected_grouped_smallm_bf16_bf16_out,
        ),
    ):
        register(
            KernelKey(
                "hip_gfx1100",
                "moe_linear",
                quant_key,
                "selected_t16_grouped_smallm_bf16_bf16_out",
            ),
            grouped_smallm_fn,
            replace=replace,
        )

    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q5_k_t16_v1",
            "selected_t16_qwen_tile8_gemv_decode_bf16_bf16_out",
        ),
        gguf_q5_k_t16_selected_qwen_tile8_gemv_bf16_bf16_out,
        replace=replace,
    )

    for quant_key, pairreuse_fn in (
        ("gguf_q5_k_t16_v1", gguf_q5_k_t16_selected_pairreuse_gemv_bf16_bf16_out),
        ("gguf_q6_k_t16_v1", gguf_q6_k_t16_selected_pairreuse_gemv_bf16_bf16_out),
    ):
        register(
            KernelKey(
                "hip_gfx1100",
                "moe_linear",
                quant_key,
                "selected_t16_pairreuse_gemv_decode_bf16_bf16_out",
            ),
            pairreuse_fn,
            replace=replace,
        )

    for quant_key, compact_pairreuse_fn in (
        (
            "gguf_q4_k_t16_v1",
            gguf_q4_k_t16_selected_pairreuse_gemv_decode_compact_bf16_bf16_out,
        ),
        (
            "gguf_q5_k_t16_v1",
            gguf_q5_k_t16_selected_pairreuse_gemv_decode_compact_bf16_bf16_out,
        ),
        (
            "gguf_q6_k_t16_v1",
            gguf_q6_k_t16_selected_pairreuse_gemv_decode_compact_bf16_bf16_out,
        ),
    ):
        register(
            KernelKey(
                "hip_gfx1100",
                "moe_linear",
                quant_key,
                "selected_t16_pairreuse_gemv_decode_compact_bf16_bf16_out",
            ),
            compact_pairreuse_fn,
            replace=replace,
        )

    for quant_key, direct_q8_dp4a in (
        ("gguf_q5_k_t16_v1", gguf_q5_k_t16_selected_q8_1_dp4a_gemv_bf16_bf16_out),
    ):
        register(
            KernelKey("hip_gfx1100", "moe_linear", quant_key, "selected_t16_q8_1_dp4a_gemv_decode_bf16_bf16_out"),
            direct_q8_dp4a,
            replace=replace,
        )


register_gguf_t16_selected_gemv_kernels()


__all__ = [
    "build_gguf_t16_selected_gemv",
    "gguf_q4_k_t16_dense_dual_interleaved_tile2_local32_silu_bf16_bf16_out",
    "gguf_q4_k_t16_dense_dual_local32_silu_bf16_bf16_out",
    "gguf_q4_k_t16_dense_dual_rowtile_silu_bf16_bf16_out",
    "gguf_q4_k_t16_dense_rowtile_bf16_bf16_out",
    "gguf_q4_k_t16_dense_rowtile16_w2_bf16_bf16_out",
    "gguf_q4_k_t16_dense_rowtile16_w2_grouped_rows6_bf16_bf16_out",
    "gguf_q4_k_t16_dense_rowtile_bf16_residual_bf16_out",
    "gguf_q4_k_t16_dense_rowtile_col4_bf16_bf16_out",
    "gguf_q4_k_t16_dense_single_col4_bf16_bf16_out",
    "gguf_q4_k_qmicro_t16_dense_single_local32_bf16_bf16_out",
    "gguf_q4_k_qmicro_t16_dense_rowtile_bf16_bf16_out",
    "gguf_q4_k_t16_dense_single_local32_bf16_bf16_out",
    "gguf_q4_k_t16_dense_single_local32_bf16_residual_bf16_out",
    "gguf_q4_k_t16_dense_dual_q8_1x2_dp4a_silu_bf16_bf16_out",
    "gguf_q4_k_t16_dense_dual_q8_1x2_split_weight_dp4a_silu_bf16_bf16_out",
    "gguf_q4_k_qmicro_t16_dense_dual_q8_1x2_split_weight_dp4a_silu_bf16_bf16_out",
    "gguf_q4_k_qmicro_t16_dense_dual_q8_1x2_rowtile8_dp4a_silu_bf16_bf16_out",
    "gguf_q4_k_qmicro_t16_dense_dual_rowtile_silu_bf16_bf16_out",
    "gguf_q4_k_qmicro_t16_selected_dual_gemv_bf16_bf16_out",
    "gguf_q4_k_qmicro_t16_selected_dual_silu_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_dual_natural_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_dual_natural_tile8_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_dual_natural_tile8_parallel_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_halfdot_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_dual_interleaved_natural_tile8_parallel_silu_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_dual_interleaved_natural_tile8_parallel_silu_halfdot_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_paircoeff_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_pairq_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_dual_pairreuse_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_dual_gemv_fp16_fp16_out",
    "gguf_q4_k_t16_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_dual_silu_q8_1_dp4a_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_dual_silu_q8_1x2_dp4a_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_dual_gemv_decode_compact_bf16_bf16_out",
    "gguf_q4_k_t16_selected_dual_gemv_decode_compact_fp16_fp16_out",
    "gguf_q4_k_t16_selected_dual_grouped_smallm_bf16_bf16_out",
    "gguf_q4_k_t16_selected_dual_pairreuse_gemv_decode_compact_bf16_bf16_out",
    "gguf_q4_k_t16_selected_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_natural_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_natural_parallel_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_natural_parallel_paircoeff_weighted_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_natural_parallel_weighted_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_gemv_fp16_fp16_out",
    "gguf_q4_k_t16_selected_gemv_decode_compact_bf16_bf16_out",
    "gguf_q4_k_t16_selected_gemv_decode_compact_fp16_fp16_out",
    "gguf_q4_k_t16_selected_grouped_smallm_bf16_bf16_out",
    "gguf_q4_k_t16_selected_pairreuse_gemv_decode_compact_bf16_bf16_out",
    "gguf_q5_k_t16_gemv_decode_bf16_bf16_out",
    "gguf_q5_k_t16_gemv_decode_tile8_bf16_bf16_out",
    "gguf_q5_k_t16_gemv_rowtile_bf16_bf16_out",
    "gguf_q5_k_t16_gemv_rowtile_col8_bf16_bf16_out",
    "gguf_q5_k_t16_gemv_rowtile_grouped_rows6_bf16_bf16_out",
    "gguf_q5_k_qmicro_t16_selected_gemv_bf16_bf16_out",
    "gguf_q5_k_qmicro_t16_selected_qwen_tile8_gemv_bf16_bf16_out",
    "gguf_q5_k_t16_selected_gemv_bf16_bf16_out",
    "gguf_q5_k_t16_selected_qwen_tile8_gemv_bf16_bf16_out",
    "gguf_q5_k_t16_selected_pairreuse_gemv_bf16_bf16_out",
    "gguf_q5_k_t16_selected_q8_1_dp4a_gemv_bf16_bf16_out",
    "gguf_q5_k_t16_selected_gemv_fp16_fp16_out",
    "gguf_q5_k_t16_selected_gemv_decode_compact_bf16_bf16_out",
    "gguf_q5_k_t16_selected_gemv_decode_compact_fp16_fp16_out",
    "gguf_q5_k_t16_selected_pairreuse_gemv_decode_compact_bf16_bf16_out",
    "gguf_q6_k_t16_selected_gemv_bf16_bf16_out",
    "gguf_q6_k_t16_qmicro_planar_selected_natural_gemv_bf16_bf16_out",
    "gguf_q6_k_t16_qmicro_planar_selected_natural_parallel_gemv_bf16_bf16_out",
    "gguf_q6_k_t16_selected_pairreuse_gemv_bf16_bf16_out",
    "gguf_q6_k_t16_selected_gemv_fp16_fp16_out",
    "gguf_q6_k_t16_selected_gemv_decode_compact_bf16_bf16_out",
    "gguf_q6_k_t16_selected_gemv_decode_compact_fp16_fp16_out",
    "gguf_q6_k_t16_selected_grouped_smallm_bf16_bf16_out",
    "gguf_q6_k_t16_selected_pairreuse_gemv_decode_compact_bf16_bf16_out",
    "plan_gguf_t16_selected_gemv_build",
    "register_gguf_t16_selected_gemv_kernels",
]
