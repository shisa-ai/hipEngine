"""Wrappers for selected GGUF K-family T16 GEMV decode kernels.

P9.H3 replacement-layout implementation for compact MoE decode.  The wrappers
consume the T16 tile layouts produced by the resident materializer:

* Q4_K gate/up: ``tiles[E, out_tiles16, blocks_per_row, 2368]`` dual output.
* Q4_K / Q5_K / Q6_K down: single-output selected GEMV for the corresponding
  T16 tile layout.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_t16_selected_gemv.hip")
_OUTPUT_NAME = "gguf_t16_selected_gemv.so"

_Q4_DUAL_DIRECT_BF16 = "hipengine_gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out"
_Q4_DUAL_DIRECT_FP16 = "hipengine_gguf_q4_k_t16_selected_dual_gemv_fp16_fp16_out"
_Q4_DUAL_SILU_DIRECT_BF16 = "hipengine_gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out"
_Q4_SINGLE_DIRECT_BF16 = "hipengine_gguf_q4_k_t16_selected_gemv_bf16_bf16_out"
_Q4_SINGLE_DIRECT_FP16 = "hipengine_gguf_q4_k_t16_selected_gemv_fp16_fp16_out"
_Q5_SINGLE_DIRECT_BF16 = "hipengine_gguf_q5_k_t16_selected_gemv_bf16_bf16_out"
_Q5_SINGLE_DIRECT_FP16 = "hipengine_gguf_q5_k_t16_selected_gemv_fp16_fp16_out"
_Q6_SINGLE_DIRECT_BF16 = "hipengine_gguf_q6_k_t16_selected_gemv_bf16_bf16_out"
_Q6_SINGLE_DIRECT_FP16 = "hipengine_gguf_q6_k_t16_selected_gemv_fp16_fp16_out"
_Q4_DUAL_BF16 = "hipengine_gguf_q4_k_t16_selected_dual_gemv_decode_compact_bf16_bf16_out"
_Q4_DUAL_FP16 = "hipengine_gguf_q4_k_t16_selected_dual_gemv_decode_compact_fp16_fp16_out"
_Q4_SINGLE_BF16 = "hipengine_gguf_q4_k_t16_selected_gemv_decode_compact_bf16_bf16_out"
_Q4_SINGLE_FP16 = "hipengine_gguf_q4_k_t16_selected_gemv_decode_compact_fp16_fp16_out"
_Q5_SINGLE_BF16 = "hipengine_gguf_q5_k_t16_selected_gemv_decode_compact_bf16_bf16_out"
_Q5_SINGLE_FP16 = "hipengine_gguf_q5_k_t16_selected_gemv_decode_compact_fp16_fp16_out"
_Q6_SINGLE_BF16 = "hipengine_gguf_q6_k_t16_selected_gemv_decode_compact_bf16_bf16_out"
_Q6_SINGLE_FP16 = "hipengine_gguf_q6_k_t16_selected_gemv_decode_compact_fp16_fp16_out"

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
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 selected Q6T16 GEMV preserving selected-row order."""

    _launch_single_direct(
        _Q6_SINGLE_DIRECT_BF16,
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
    library = library or build_gguf_t16_selected_gemv(load=True)
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
    library = library or build_gguf_t16_selected_gemv(load=True)
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
    library = library or build_gguf_t16_selected_gemv(load=True)
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
    library = library or build_gguf_t16_selected_gemv(load=True)
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
    library = library or build_gguf_t16_selected_gemv(load=True)
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
            "selected_dual_t16_gemv_decode_bf16_bf16_out",
            gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
        ),
        (
            "selected_dual_t16_gemv_decode_fp16_fp16_out",
            gguf_q4_k_t16_selected_dual_gemv_fp16_fp16_out,
        ),
        (
            "selected_dual_t16_silu_gemv_decode_bf16_bf16_out",
            gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out,
        ),
    ):
        register(
            KernelKey("hip_gfx1100", "moe_linear", "gguf_q4_k_t16_v1", variant),
            fn,
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


register_gguf_t16_selected_gemv_kernels()


__all__ = [
    "build_gguf_t16_selected_gemv",
    "gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_dual_gemv_fp16_fp16_out",
    "gguf_q4_k_t16_selected_dual_gemv_decode_compact_bf16_bf16_out",
    "gguf_q4_k_t16_selected_dual_gemv_decode_compact_fp16_fp16_out",
    "gguf_q4_k_t16_selected_gemv_bf16_bf16_out",
    "gguf_q4_k_t16_selected_gemv_fp16_fp16_out",
    "gguf_q4_k_t16_selected_gemv_decode_compact_bf16_bf16_out",
    "gguf_q4_k_t16_selected_gemv_decode_compact_fp16_fp16_out",
    "gguf_q5_k_t16_selected_gemv_bf16_bf16_out",
    "gguf_q5_k_t16_selected_gemv_fp16_fp16_out",
    "gguf_q5_k_t16_selected_gemv_decode_compact_bf16_bf16_out",
    "gguf_q5_k_t16_selected_gemv_decode_compact_fp16_fp16_out",
    "gguf_q6_k_t16_selected_gemv_bf16_bf16_out",
    "gguf_q6_k_t16_selected_gemv_fp16_fp16_out",
    "gguf_q6_k_t16_selected_gemv_decode_compact_bf16_bf16_out",
    "gguf_q6_k_t16_selected_gemv_decode_compact_fp16_fp16_out",
    "plan_gguf_t16_selected_gemv_build",
    "register_gguf_t16_selected_gemv_kernels",
]
