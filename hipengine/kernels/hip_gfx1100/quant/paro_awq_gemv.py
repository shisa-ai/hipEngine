"""Raw-pointer wrappers for PARO AWQ pack8 GEMV kernels."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

# Cached argtypes tuples for the hot AWQ pack8 launchers used by the verifier.
# See hipengine/core/ctypes_cache.py for the caching rationale.
_ARGTYPES_PACK8_SINGLE = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # 5 ptrs
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,        # rows, in_features, out_packed, group_size, threads
    ctypes.c_void_p,                                                                       # stream
)
_ARGTYPES_PACK8_SINGLE_COMBINE = (
    ctypes.c_void_p,                                                                       # x
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                                     # qweight, qzeros, scales
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,   # selected, weights, gate, residual, out
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,                        # rows, in_features, out_packed, group_size
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,                                        # rows_per_token, gate_stride, threads
    ctypes.c_void_p,                                                                       # stream
)
_ARGTYPES_PACK8_DUAL_1 = (
    ctypes.c_void_p,                                                                       # input ptr
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                                     # qweight_a, qzeros_a, scales_a
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                                     # qweight_b, qzeros_b, scales_b
    ctypes.c_void_p,                                                                       # out
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # rows, in_features, out_packed_a, out_packed_b, group_size, threads
    ctypes.c_void_p,                                                                       # stream
)
_ARGTYPES_PACK8_DUAL_2 = (
    ctypes.c_void_p, ctypes.c_void_p,                                                      # input_a, input_b
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGTYPES_SELECTED_DUAL = (
    ctypes.c_void_p, ctypes.c_void_p,                                                      # x, selected
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                                     # qweight_a, qzeros_a, scales_a
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                                     # qweight_b, qzeros_b, scales_b
    ctypes.c_void_p,                                                                       # out
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    # x_rows, rows, in_features, out_packed_a, out_packed_b, num_experts, group_size, threads
    ctypes.c_void_p,                                                                       # stream
)
_ARGTYPES_SELECTED_DUAL_ROTATE_STAGED = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                                     # x, rotated, selected
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                                     # pairs, theta, channel_scales
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                                     # qweight_a, qzeros_a, scales_a
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                                     # qweight_b, qzeros_b, scales_b
    ctypes.c_void_p, ctypes.c_void_p,                                                      # out, barrier
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,        # x_rows, rows, in_features, out_packed_a, out_packed_b
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,                        # num_experts, group_size, krot, threads
    ctypes.c_void_p,                                                                       # stream
)
_ARGTYPES_SELECTED_DUAL_ROTATE_STAGED_KEYED = (
    *_ARGTYPES_SELECTED_DUAL_ROTATE_STAGED[:-1],
    ctypes.c_int64, ctypes.c_int64,                                                        # barrier_count_target, barrier_ready_value
    ctypes.c_void_p,
)
_ARGTYPES_SELECTED_SINGLE_SILU_ROTATE_STAGED = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                                     # gate_up, down_input, selected
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                                     # pairs, theta, channel_scales
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                                     # qweight, qzeros, scales
    ctypes.c_void_p, ctypes.c_void_p,                                                      # out, barrier
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,        # rows, in_features, out_packed, num_experts, group_size
    ctypes.c_int64, ctypes.c_int64,                                                        # krot, threads
    ctypes.c_void_p,                                                                       # stream
)
_ARGTYPES_SELECTED_SINGLE_SILU_ROTATE_STAGED_KEYED = (
    *_ARGTYPES_SELECTED_SINGLE_SILU_ROTATE_STAGED[:-1],
    ctypes.c_int64, ctypes.c_int64,                                                        # barrier_count_target, barrier_ready_value
    ctypes.c_void_p,
)
_ARGTYPES_SELECTED_SINGLE = (
    ctypes.c_void_p, ctypes.c_void_p,                                                      # x, selected
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                                     # qweight, qzeros, scales
    ctypes.c_void_p,                                                                       # out
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    # rows, in_features, out_packed, num_experts, group_size, threads
    ctypes.c_void_p,                                                                       # stream
)
_ARGTYPES_FUSEDW4_PREFILL_SINGLE = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    # rows, in_features, out_packed, group_size, tile_m, tile_n
    ctypes.c_void_p,
)
_ARGTYPES_FUSEDW4_PREFILL_DUAL = (
    ctypes.c_void_p, ctypes.c_void_p,                                                      # x_a, x_b
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,                                                      # out_a, out_b
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    # rows, in_features, out_packed_a, out_packed_b, group_size, tile_m, tile_n
    ctypes.c_void_p,
)
_ARGTYPES_PACK8_DUAL_ROTATE_STAGED = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                                     # x, rotated_a, rotated_b
    ctypes.c_void_p, ctypes.c_void_p,                                                      # pairs_a, pairs_b
    ctypes.c_void_p, ctypes.c_void_p,                                                      # theta_a, theta_b
    ctypes.c_void_p, ctypes.c_void_p,                                                      # channel_scales_a, channel_scales_b
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                                     # qweight_a, qzeros_a, scales_a
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                                     # qweight_b, qzeros_b, scales_b
    ctypes.c_void_p, ctypes.c_void_p,                                                      # out, barrier
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    # rows, in_features, out_packed_a, out_packed_b, group_size, krot, threads
    ctypes.c_void_p,                                                                       # stream
)
_ARGTYPES_PACK8_DUAL_ROTATE_STAGED_KEYED = (
    *_ARGTYPES_PACK8_DUAL_ROTATE_STAGED[:-1],
    ctypes.c_int64, ctypes.c_int64,                                                        # barrier_count_target, barrier_ready_value
    ctypes.c_void_p,
)

_SOURCE = Path(__file__).with_name("paro_awq_gemv.hip")
_OUTPUT_NAME = "paro_awq_gemv.so"
_SYMBOL_PACK8_STRIDED = "hipengine_gemv_awq_pack8_strided_bf16"
_SYMBOL_PACK8_TRANSPOSED = "hipengine_gemv_awq_pack8_transposed_bf16"
_SYMBOL_DUAL_PACK8_STRIDED = "hipengine_gemv_awq_dual_pack8_strided_bf16"
_SYMBOL_DUAL_PACK8_TRANSPOSED = "hipengine_gemv_awq_dual_pack8_transposed_bf16"
_SYMBOL_DUAL_PACK8_TRANSPOSED_ROTATE_STAGED = "hipengine_gemv_awq_dual_pack8_transposed_rotate_staged_bf16"
_SYMBOL_DUAL_PACK8_TRANSPOSED_ROTATE_STAGED_KEYED = "hipengine_gemv_awq_dual_pack8_transposed_rotate_staged_keyed_bf16"
_SYMBOL_PACK8_STRIDED_FP16 = "hipengine_gemv_awq_pack8_strided_fp16"
_SYMBOL_PACK8_TRANSPOSED_FP16 = "hipengine_gemv_awq_pack8_transposed_fp16"
_SYMBOL_PACK8_MULTI_ROW_STRIDED_FP16 = "hipengine_gemv_awq_pack8_multi_row_strided_fp16"
_SYMBOL_PACK8_MULTI_ROW_TRANSPOSED_FP16 = "hipengine_gemv_awq_pack8_multi_row_transposed_fp16"
_SYMBOL_PACK8_MULTI_ROW_DECODE_STRIDED_FP16 = "hipengine_gemv_awq_pack8_multi_row_decode_strided_fp16"
_SYMBOL_PACK8_MULTI_ROW_DECODE_TRANSPOSED_FP16 = "hipengine_gemv_awq_pack8_multi_row_decode_transposed_fp16"
_SYMBOL_PACK8_MULTI_ROW_STRIDED_BF16 = "hipengine_gemv_awq_pack8_multi_row_strided_bf16"
_SYMBOL_PACK8_MULTI_ROW_TRANSPOSED_BF16 = "hipengine_gemv_awq_pack8_multi_row_transposed_bf16"
_SYMBOL_PACK8_OUTPUT_TILED = "hipengine_gemv_awq_pack8_output_tiled_bf16"
_SYMBOL_PACK8_OUTPUT_TILED_FP16 = "hipengine_gemv_awq_pack8_output_tiled_fp16"
_SYMBOL_PACK8_OUTPUT_TILED_TRANSPOSED = "hipengine_gemv_awq_pack8_output_tiled_transposed_bf16"
_SYMBOL_PACK8_OUTPUT_TILED_TRANSPOSED_FP16 = "hipengine_gemv_awq_pack8_output_tiled_transposed_fp16"
_SYMBOL_PACK8_OUTPUT_TILED_COMBINE_TRANSPOSED_FP16 = (
    "hipengine_gemv_awq_pack8_output_tiled_combine_residual_transposed_fp16"
)
_SYMBOL_DUAL_PACK8_OUTPUT_TILED_TRANSPOSED = "hipengine_gemv_awq_dual_pack8_output_tiled_transposed_bf16"
_SYMBOL_DUAL_PACK8_OUTPUT_TILED_TRANSPOSED_FP16 = "hipengine_gemv_awq_dual_pack8_output_tiled_transposed_fp16"
_SYMBOL_DUAL_PACK8_OUTPUT_TILED_SPLIT_TRANSPOSED_FP16 = "hipengine_gemv_awq_dual_pack8_output_tiled_split_transposed_fp16"
_SYMBOL_DUAL_PACK8_OUTPUT_TILED_STRIDED = "hipengine_gemv_awq_dual_pack8_output_tiled_strided_bf16"
_SYMBOL_DUAL_PACK8_OUTPUT_TILED_STRIDED_FP16 = "hipengine_gemv_awq_dual_pack8_output_tiled_strided_fp16"
_SYMBOL_FUSEDW4_PREFILL_FP16 = "hipengine_awq_fusedw4_prefill_fp16"
_SYMBOL_FUSEDW4_PREFILL_DUAL_FP16 = "hipengine_awq_fusedw4_prefill_dual_fp16"
_SYMBOL_FUSEDW4_PREFILL_STRIDED_FP16 = "hipengine_awq_fusedw4_prefill_strided_fp16"
_SYMBOL_DUAL_PACK8_STRIDED_FP16 = "hipengine_gemv_awq_dual_pack8_strided_fp16"
_SYMBOL_DUAL_PACK8_TRANSPOSED_FP16 = "hipengine_gemv_awq_dual_pack8_transposed_fp16"
_SYMBOL_DUAL_PACK8_MULTI_ROW_STRIDED_FP16 = "hipengine_gemv_awq_dual_pack8_multi_row_strided_fp16"
_SYMBOL_DUAL_PACK8_MULTI_ROW_TRANSPOSED_FP16 = "hipengine_gemv_awq_dual_pack8_multi_row_transposed_fp16"
_SYMBOL_DUAL_PACK8_MULTI_ROW_SPLIT_TRANSPOSED_FP16 = "hipengine_gemv_awq_dual_pack8_multi_row_split_transposed_fp16"
_SYMBOL_DUAL_PACK8_MULTI_ROW_DECODE_SPLIT_TRANSPOSED_FP16 = "hipengine_gemv_awq_dual_pack8_multi_row_decode_split_transposed_fp16"
_SYMBOL_DUAL_PACK8_TRANSPOSED_ROTATE_STAGED_FP16 = "hipengine_gemv_awq_dual_pack8_transposed_rotate_staged_fp16"
_SYMBOL_DUAL_PACK8_TRANSPOSED_ROTATE_STAGED_KEYED_FP16 = "hipengine_gemv_awq_dual_pack8_transposed_rotate_staged_keyed_fp16"
_SYMBOL_SELECTED_DUAL_ROTATE_STRIDED = "hipengine_gemv_awq_selected_dual_pack8_strided_rotate_out_bf16"
_SYMBOL_SELECTED_DUAL_STRIDED = "hipengine_gemv_awq_selected_dual_pack8_strided_bf16"
_SYMBOL_SELECTED_DUAL_TRANSPOSED = "hipengine_gemv_awq_selected_dual_pack8_transposed_bf16"
_SYMBOL_SELECTED_STRIDED = "hipengine_gemv_awq_selected_pack8_strided_bf16"
_SYMBOL_SELECTED_TRANSPOSED = "hipengine_gemv_awq_selected_pack8_transposed_bf16"
_SYMBOL_SELECTED_DUAL_ROTATE_STRIDED_FP16 = "hipengine_gemv_awq_selected_dual_pack8_strided_rotate_out_fp16"
_SYMBOL_SELECTED_DUAL_ROTATE_TRANSPOSED_FP16 = "hipengine_gemv_awq_selected_dual_pack8_transposed_rotate_out_fp16"
_SYMBOL_SELECTED_DUAL_ROTATE_STAGED_TRANSPOSED_FP16 = "hipengine_gemv_awq_selected_dual_pack8_transposed_rotate_staged_fp16"
_SYMBOL_SELECTED_DUAL_ROTATE_STAGED_KEYED_TRANSPOSED_FP16 = "hipengine_gemv_awq_selected_dual_pack8_transposed_rotate_staged_keyed_fp16"
_SYMBOL_SELECTED_DUAL_STRIDED_FP16 = "hipengine_gemv_awq_selected_dual_pack8_strided_fp16"
_SYMBOL_SELECTED_DUAL_TRANSPOSED_FP16 = "hipengine_gemv_awq_selected_dual_pack8_transposed_fp16"
_SYMBOL_SELECTED_STRIDED_FP16 = "hipengine_gemv_awq_selected_pack8_strided_fp16"
_SYMBOL_SELECTED_TRANSPOSED_FP16 = "hipengine_gemv_awq_selected_pack8_transposed_fp16"
_SYMBOL_SELECTED_TRANSPOSED_SILU_ROTATE_STAGED_FP16 = "hipengine_gemv_awq_selected_pack8_transposed_silu_rotate_staged_fp16"
_SYMBOL_SELECTED_TRANSPOSED_SILU_ROTATE_STAGED_KEYED_FP16 = "hipengine_gemv_awq_selected_pack8_transposed_silu_rotate_staged_keyed_fp16"
_ALLOWED_THREADS = {64, 128}


def plan_paro_awq_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="paro_awq_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_paro_awq_gemv(
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
        family="paro_awq_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )



def gemv_awq_pack8_strided_bf16(
    x_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch generic single-projection pack8 GEMV with strided qweight layout."""

    _launch_pack8_single(
        _SYMBOL_PACK8_STRIDED,
        x_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_pack8_output_tiled_bf16(
    x_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Output-column-tiled c>1 decode GEMV (strided qweight); rows in {2,4,8}.

    One block per output pack loads each weight pack once and accumulates all
    ``rows`` columns; bit-exact vs the per-row ``gemv_awq_pack8_strided`` path.
    """

    _launch_pack8_single(
        _SYMBOL_PACK8_OUTPUT_TILED,
        x_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_pack8_output_tiled_fp16(
    x_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """FP16 output-column-tiled c>1 decode GEMV (strided qweight); rows in {2,4,8}."""

    _launch_pack8_single(
        _SYMBOL_PACK8_OUTPUT_TILED_FP16,
        x_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_pack8_output_tiled_transposed_bf16(
    x_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Output-column-tiled c>1 decode GEMV (transposed qweight); rows in {2,4,8}.

    Byte-exact vs the per-row ``gemv_awq_pack8_transposed`` path.
    """

    _launch_pack8_single(
        _SYMBOL_PACK8_OUTPUT_TILED_TRANSPOSED,
        x_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_pack8_output_tiled_transposed_fp16(
    x_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """FP16 output-column-tiled c>1 decode GEMV (transposed qweight); rows in {2,4,8}."""

    _launch_pack8_single(
        _SYMBOL_PACK8_OUTPUT_TILED_TRANSPOSED_FP16,
        x_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_pack8_output_tiled_combine_residual_transposed_fp16(
    x_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    selected_ptr: int,
    routing_weights_ptr: int,
    shared_gate_logits_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    rows_per_token: int,
    gate_stride: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """FP16 output-tiled shared-down GEMV fused with selected/shared residual combine.

    This preserves the old two-launch rounding points: the shared-down dot is
    rounded to FP16 first, the selected weighted sum is rounded to FP16 next,
    and only then are residual + selected + sigmoid(shared_gate) * shared added.
    ``rows`` is the verifier token count and must be in {2, 4, 8}.
    """

    _check_pack8_single_shape(rows, in_features, out_packed, group_size, threads)
    _check_positive(rows_per_token, "rows_per_token")
    _check_positive(gate_stride, "gate_stride")
    library = library or build_paro_awq_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        _SYMBOL_PACK8_OUTPUT_TILED_COMBINE_TRANSPOSED_FP16,
        _ARGTYPES_PACK8_SINGLE_COMBINE,
        ctypes.c_int,
    )
    err = fn(
        x_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        selected_ptr,
        routing_weights_ptr,
        shared_gate_logits_ptr,
        residual_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        group_size,
        rows_per_token,
        gate_stride,
        threads,
        stream,
    )
    _check_launch(runtime, err)


def _gemv_awq_dual_pack8_output_tiled(
    symbol: str,
    x_a_ptr: int,
    x_b_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _launch_pack8_dual(
        symbol,
        (x_a_ptr, x_b_ptr),
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_dual_pack8_output_tiled_transposed_fp16(
    x_a_ptr: int,
    x_b_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """FP16 output-column-tiled c>1 dual GEMV (transposed); bit-exact vs dual transposed."""

    _gemv_awq_dual_pack8_output_tiled(
        _SYMBOL_DUAL_PACK8_OUTPUT_TILED_TRANSPOSED_FP16,
        x_a_ptr, x_b_ptr, qweight_a_ptr, qzeros_a_ptr, scales_a_ptr,
        qweight_b_ptr, qzeros_b_ptr, scales_b_ptr, out_ptr, rows, in_features,
        out_packed_a, out_packed_b, group_size,
        threads=threads, stream=stream, library=library, runtime=runtime,
    )


def gemv_awq_dual_pack8_output_tiled_split_transposed_fp16(
    x_a_ptr: int,
    x_b_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """FP16 output-column-tiled c>1 dual GEMV with split outputs."""

    _launch_dual_pack8_multi_row_split_transposed_fp16(
        _SYMBOL_DUAL_PACK8_OUTPUT_TILED_SPLIT_TRANSPOSED_FP16,
        x_a_ptr, x_b_ptr, qweight_a_ptr, qzeros_a_ptr, scales_a_ptr,
        qweight_b_ptr, qzeros_b_ptr, scales_b_ptr, out_a_ptr, out_b_ptr,
        rows, in_features, out_packed_a, out_packed_b, group_size,
        threads=threads, stream=stream, library=library, runtime=runtime,
    )


def gemv_awq_dual_pack8_output_tiled_transposed_bf16(
    x_a_ptr: int,
    x_b_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """BF16 output-column-tiled c>1 dual GEMV (transposed); bit-exact vs dual transposed."""

    _gemv_awq_dual_pack8_output_tiled(
        _SYMBOL_DUAL_PACK8_OUTPUT_TILED_TRANSPOSED,
        x_a_ptr, x_b_ptr, qweight_a_ptr, qzeros_a_ptr, scales_a_ptr,
        qweight_b_ptr, qzeros_b_ptr, scales_b_ptr, out_ptr, rows, in_features,
        out_packed_a, out_packed_b, group_size,
        threads=threads, stream=stream, library=library, runtime=runtime,
    )


def gemv_awq_dual_pack8_output_tiled_strided_fp16(
    x_a_ptr: int,
    x_b_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """FP16 output-column-tiled c>1 dual GEMV (strided); bit-exact vs dual strided."""

    _gemv_awq_dual_pack8_output_tiled(
        _SYMBOL_DUAL_PACK8_OUTPUT_TILED_STRIDED_FP16,
        x_a_ptr, x_b_ptr, qweight_a_ptr, qzeros_a_ptr, scales_a_ptr,
        qweight_b_ptr, qzeros_b_ptr, scales_b_ptr, out_ptr, rows, in_features,
        out_packed_a, out_packed_b, group_size,
        threads=threads, stream=stream, library=library, runtime=runtime,
    )


def gemv_awq_dual_pack8_output_tiled_strided_bf16(
    x_a_ptr: int,
    x_b_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """BF16 output-column-tiled c>1 dual GEMV (strided); bit-exact vs dual strided."""

    _gemv_awq_dual_pack8_output_tiled(
        _SYMBOL_DUAL_PACK8_OUTPUT_TILED_STRIDED,
        x_a_ptr, x_b_ptr, qweight_a_ptr, qzeros_a_ptr, scales_a_ptr,
        qweight_b_ptr, qzeros_b_ptr, scales_b_ptr, out_ptr, rows, in_features,
        out_packed_a, out_packed_b, group_size,
        threads=threads, stream=stream, library=library, runtime=runtime,
    )


def gemv_awq_pack8_transposed_bf16(
    x_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch generic single-projection pack8 GEMV with transposed qweight layout."""

    _launch_pack8_single(
        _SYMBOL_PACK8_TRANSPOSED,
        x_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_dual_pack8_strided_bf16(
    x_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch generic dual pack8 GEMV with one shared input and strided qweights."""

    _launch_pack8_dual(
        _SYMBOL_DUAL_PACK8_STRIDED,
        (x_ptr,),
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_dual_pack8_transposed_bf16(
    x_a_ptr: int,
    x_b_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch generic dual pack8 GEMV with separate inputs and transposed qweights."""

    _launch_pack8_dual(
        _SYMBOL_DUAL_PACK8_TRANSPOSED,
        (x_a_ptr, x_b_ptr),
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_dual_pack8_transposed_rotate_staged_bf16(
    x_ptr: int,
    rotated_a_ptr: int,
    rotated_b_ptr: int,
    pairs_a_ptr: int,
    pairs_b_ptr: int,
    theta_a_ptr: int,
    theta_b_ptr: int,
    channel_scales_a_ptr: int,
    channel_scales_b_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    barrier_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    krot: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch decode fused rotate-stage + dual transposed pack8 GEMV for BF16 buffers."""

    _launch_pack8_dual_rotate_staged(
        _SYMBOL_DUAL_PACK8_TRANSPOSED_ROTATE_STAGED,
        x_ptr,
        rotated_a_ptr,
        rotated_b_ptr,
        pairs_a_ptr,
        pairs_b_ptr,
        theta_a_ptr,
        theta_b_ptr,
        channel_scales_a_ptr,
        channel_scales_b_ptr,
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        barrier_ptr,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        group_size,
        krot,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_dual_pack8_transposed_rotate_staged_keyed_bf16(
    x_ptr: int,
    rotated_a_ptr: int,
    rotated_b_ptr: int,
    pairs_a_ptr: int,
    pairs_b_ptr: int,
    theta_a_ptr: int,
    theta_b_ptr: int,
    channel_scales_a_ptr: int,
    channel_scales_b_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    barrier_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    krot: int,
    barrier_count_target: int,
    barrier_ready_value: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Keyed-barrier BF16 rotate-stage + dual transposed pack8 GEMV.

    The caller must initialize ``barrier`` once and pass monotonically
    increasing cumulative ``barrier_count_target`` / ``barrier_ready_value``
    values.  This avoids the non-keyed launcher's per-call hipMemsetAsync.
    """

    _launch_pack8_dual_rotate_staged(
        _SYMBOL_DUAL_PACK8_TRANSPOSED_ROTATE_STAGED_KEYED,
        x_ptr,
        rotated_a_ptr,
        rotated_b_ptr,
        pairs_a_ptr,
        pairs_b_ptr,
        theta_a_ptr,
        theta_b_ptr,
        channel_scales_a_ptr,
        channel_scales_b_ptr,
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        barrier_ptr,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        group_size,
        krot,
        barrier_count_target=barrier_count_target,
        barrier_ready_value=barrier_ready_value,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_pack8_strided_fp16(
    x_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch generic single-projection pack8 GEMV for FP16 buffers."""

    _launch_pack8_single(
        _SYMBOL_PACK8_STRIDED_FP16,
        x_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_pack8_transposed_fp16(
    x_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch generic single-projection transposed pack8 GEMV for FP16 buffers."""

    _launch_pack8_single(
        _SYMBOL_PACK8_TRANSPOSED_FP16,
        x_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_pack8_multi_row_transposed_fp16(
    x_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """M12.6: pack8 W4 GEMV that loops over rows internally so the weight tile
    streams from HBM once per block (no row-grid replication).  Supports
    ``rows <= 8``; callers above that bound use the stock prefill kernel.
    Same I/O semantics as ``gemv_awq_pack8_transposed_fp16``."""

    _launch_pack8_single(
        _SYMBOL_PACK8_MULTI_ROW_TRANSPOSED_FP16,
        x_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_pack8_multi_row_strided_fp16(
    x_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Strided/non-transposed variant of ``gemv_awq_pack8_multi_row_transposed_fp16``."""

    _launch_pack8_single(
        _SYMBOL_PACK8_MULTI_ROW_STRIDED_FP16,
        x_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )



def gemv_awq_pack8_multi_row_decode_transposed_fp16(
    x_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Multi-row FP16 pack8 GEMV using row-wise GEMV f32 dequant semantics.

    This shares the weight tile across ``rows <= 8`` like the M12.6 multi-row
    path, but matches ``gemv_awq_pack8_transposed_fp16`` arithmetic instead of
    the FP16 prefill-WMMA compatibility dequantization.
    """

    _launch_pack8_single(
        _SYMBOL_PACK8_MULTI_ROW_DECODE_TRANSPOSED_FP16,
        x_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )



def gemv_awq_pack8_multi_row_decode_strided_fp16(
    x_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Strided variant of ``gemv_awq_pack8_multi_row_decode_transposed_fp16``."""

    _launch_pack8_single(
        _SYMBOL_PACK8_MULTI_ROW_DECODE_STRIDED_FP16,
        x_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )



def awq_fusedw4_prefill_fp16(
    x_ptr: int,
    qweight_t_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    tile_m: int | None = None,
    tile_n: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 AWQ pack8 W4 -> WMMA prefill GEMM for transposed weights."""

    _launch_fusedw4_prefill_fp16(
        _SYMBOL_FUSEDW4_PREFILL_FP16,
        x_ptr,
        qweight_t_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        group_size,
        tile_m=tile_m,
        tile_n=tile_n,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def awq_fusedw4_prefill_dual_fp16(
    x_a_ptr: int,
    x_b_ptr: int,
    qweight_a_t_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_t_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    tile_m: int | None = None,
    tile_n: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch one FP16 AWQ W4 WMMA prefill kernel for two transposed projections."""

    _launch_fusedw4_prefill_dual_fp16(
        _SYMBOL_FUSEDW4_PREFILL_DUAL_FP16,
        x_a_ptr,
        x_b_ptr,
        qweight_a_t_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_t_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_a_ptr,
        out_b_ptr,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        group_size,
        tile_m=tile_m,
        tile_n=tile_n,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def awq_fusedw4_prefill_strided_fp16(
    x_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    tile_m: int | None = None,
    tile_n: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 AWQ pack8 W4 -> WMMA prefill GEMM for strided weights."""

    _launch_fusedw4_prefill_fp16(
        _SYMBOL_FUSEDW4_PREFILL_STRIDED_FP16,
        x_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        group_size,
        tile_m=tile_m,
        tile_n=tile_n,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_dual_pack8_strided_fp16(
    x_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch generic dual pack8 GEMV with one shared FP16 input."""

    _launch_pack8_dual(
        _SYMBOL_DUAL_PACK8_STRIDED_FP16,
        (x_ptr,),
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_dual_pack8_transposed_fp16(
    x_a_ptr: int,
    x_b_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch generic dual transposed pack8 GEMV for FP16 buffers."""

    _launch_pack8_dual(
        _SYMBOL_DUAL_PACK8_TRANSPOSED_FP16,
        (x_a_ptr, x_b_ptr),
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_dual_pack8_multi_row_strided_fp16(
    x_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """M12.6: weight-sharing multi-row dual pack8 W4 GEMV for ``rows <= 8``.
    Same I/O as ``gemv_awq_dual_pack8_strided_fp16``."""

    _launch_pack8_dual(
        _SYMBOL_DUAL_PACK8_MULTI_ROW_STRIDED_FP16,
        (x_ptr,),
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_dual_pack8_multi_row_transposed_fp16(
    x_a_ptr: int,
    x_b_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Transposed-weight variant of ``gemv_awq_dual_pack8_multi_row_strided_fp16``."""

    _launch_pack8_dual(
        _SYMBOL_DUAL_PACK8_MULTI_ROW_TRANSPOSED_FP16,
        (x_a_ptr, x_b_ptr),
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_dual_pack8_transposed_rotate_staged_fp16(
    x_ptr: int,
    rotated_a_ptr: int,
    rotated_b_ptr: int,
    pairs_a_ptr: int,
    pairs_b_ptr: int,
    theta_a_ptr: int,
    theta_b_ptr: int,
    channel_scales_a_ptr: int,
    channel_scales_b_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    barrier_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    krot: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch decode fused rotate-stage + dual transposed pack8 GEMV for FP16 buffers."""

    _launch_pack8_dual_rotate_staged(
        _SYMBOL_DUAL_PACK8_TRANSPOSED_ROTATE_STAGED_FP16,
        x_ptr,
        rotated_a_ptr,
        rotated_b_ptr,
        pairs_a_ptr,
        pairs_b_ptr,
        theta_a_ptr,
        theta_b_ptr,
        channel_scales_a_ptr,
        channel_scales_b_ptr,
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        barrier_ptr,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        group_size,
        krot,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_dual_pack8_transposed_rotate_staged_keyed_fp16(
    x_ptr: int,
    rotated_a_ptr: int,
    rotated_b_ptr: int,
    pairs_a_ptr: int,
    pairs_b_ptr: int,
    theta_a_ptr: int,
    theta_b_ptr: int,
    channel_scales_a_ptr: int,
    channel_scales_b_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    barrier_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    krot: int,
    barrier_count_target: int,
    barrier_ready_value: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Keyed-barrier FP16 rotate-stage + dual transposed pack8 GEMV."""

    _launch_pack8_dual_rotate_staged(
        _SYMBOL_DUAL_PACK8_TRANSPOSED_ROTATE_STAGED_KEYED_FP16,
        x_ptr,
        rotated_a_ptr,
        rotated_b_ptr,
        pairs_a_ptr,
        pairs_b_ptr,
        theta_a_ptr,
        theta_b_ptr,
        channel_scales_a_ptr,
        channel_scales_b_ptr,
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        barrier_ptr,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        group_size,
        krot,
        barrier_count_target=barrier_count_target,
        barrier_ready_value=barrier_ready_value,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_selected_dual_pack8_strided_rotate_out_bf16(
    x_ptr: int,
    selected_ptr: int,
    pairs_ptr: int,
    theta_ptr: int,
    channel_scales_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    krot: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch parent fused rotate + selected dual pack8 GEMV strided kernel."""

    _launch_selected_dual_rotate(
        _SYMBOL_SELECTED_DUAL_ROTATE_STRIDED,
        x_ptr,
        selected_ptr,
        pairs_ptr,
        theta_ptr,
        channel_scales_ptr,
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        x_rows,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        num_experts,
        group_size,
        krot,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )

def gemv_awq_selected_dual_pack8_strided_bf16(
    x_ptr: int,
    selected_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch selected-expert dual gate/up pack8 GEMV with strided qweight layout."""

    _launch_selected_dual(
        _SYMBOL_SELECTED_DUAL_STRIDED,
        x_ptr,
        selected_ptr,
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        x_rows,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        num_experts,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_selected_dual_pack8_transposed_bf16(
    x_ptr: int,
    selected_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch selected-expert dual gate/up pack8 GEMV with transposed qweight layout."""

    _launch_selected_dual(
        _SYMBOL_SELECTED_DUAL_TRANSPOSED,
        x_ptr,
        selected_ptr,
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        x_rows,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        num_experts,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_selected_pack8_strided_bf16(
    x_ptr: int,
    selected_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    num_experts: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch selected-expert single/down pack8 GEMV with strided qweight layout."""

    _launch_selected_single(
        _SYMBOL_SELECTED_STRIDED,
        x_ptr,
        selected_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        num_experts,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_selected_pack8_transposed_bf16(
    x_ptr: int,
    selected_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    num_experts: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch selected-expert single/down pack8 GEMV with transposed qweight layout."""

    _launch_selected_single(
        _SYMBOL_SELECTED_TRANSPOSED,
        x_ptr,
        selected_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        num_experts,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_selected_dual_pack8_strided_rotate_out_fp16(
    x_ptr: int,
    selected_ptr: int,
    pairs_ptr: int,
    theta_ptr: int,
    channel_scales_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    krot: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch parent fused rotate + selected dual pack8 GEMV for FP16 buffers.

    M13.B.1 (Option C): the kernel body now applies an LDS scalar_t round-trip
    after rotation, so this fused kernel is bit-exact with the unfused
    ``paro_rotate1_fp16`` + ``gemv_awq_selected_dual_pack8_strided_fp16``
    chain it replaces.
    """

    _launch_selected_dual_rotate(
        _SYMBOL_SELECTED_DUAL_ROTATE_STRIDED_FP16,
        x_ptr,
        selected_ptr,
        pairs_ptr,
        theta_ptr,
        channel_scales_ptr,
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        x_rows,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        num_experts,
        group_size,
        krot,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_selected_dual_pack8_transposed_rotate_out_fp16(
    x_ptr: int,
    selected_ptr: int,
    pairs_ptr: int,
    theta_ptr: int,
    channel_scales_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    krot: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch fused rotate + selected dual pack8 GEMV for FP16 buffers using
    the transposed `qweight_pack8_decode` layout the production MoE selected
    gate_up path uses.

    M13.B.1 (Option C): the kernel body applies an LDS scalar_t round-trip
    after rotation, so this fused kernel is bit-exact with the unfused
    ``paro_rotate1_fp16`` + ``gemv_awq_selected_dual_pack8_transposed_fp16``
    chain it replaces.
    """

    _launch_selected_dual_rotate(
        _SYMBOL_SELECTED_DUAL_ROTATE_TRANSPOSED_FP16,
        x_ptr,
        selected_ptr,
        pairs_ptr,
        theta_ptr,
        channel_scales_ptr,
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        x_rows,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        num_experts,
        group_size,
        krot,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_selected_dual_pack8_transposed_rotate_staged_fp16(
    x_ptr: int,
    rotated_ptr: int,
    selected_ptr: int,
    pairs_ptr: int,
    theta_ptr: int,
    channel_scales_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    barrier_ptr: int,
    x_rows: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    krot: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch HBM-staged selected gate/up rotate + transposed dual GEMV.

    The staged kernel rotates each verifier x-row once into ``rotated_ptr`` and
    then runs the selected ids-tensor GEMV after an in-kernel barrier.  This is
    bit-exact with ``paro_rotate1_fp16`` +
    ``gemv_awq_selected_dual_pack8_transposed_fp16`` because the GEMV consumes
    the same FP16 staged buffer the unfused chain would have written.
    """

    _launch_selected_dual_rotate_staged(
        _SYMBOL_SELECTED_DUAL_ROTATE_STAGED_TRANSPOSED_FP16,
        x_ptr,
        rotated_ptr,
        selected_ptr,
        pairs_ptr,
        theta_ptr,
        channel_scales_ptr,
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        barrier_ptr,
        x_rows,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        num_experts,
        group_size,
        krot,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_selected_dual_pack8_transposed_rotate_staged_keyed_fp16(
    x_ptr: int,
    rotated_ptr: int,
    selected_ptr: int,
    pairs_ptr: int,
    theta_ptr: int,
    channel_scales_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    barrier_ptr: int,
    x_rows: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    krot: int,
    barrier_count_target: int,
    barrier_ready_value: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Keyed-barrier HBM-staged selected rotate + transposed dual GEMV."""

    _launch_selected_dual_rotate_staged(
        _SYMBOL_SELECTED_DUAL_ROTATE_STAGED_KEYED_TRANSPOSED_FP16,
        x_ptr,
        rotated_ptr,
        selected_ptr,
        pairs_ptr,
        theta_ptr,
        channel_scales_ptr,
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        barrier_ptr,
        x_rows,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        num_experts,
        group_size,
        krot,
        barrier_count_target=barrier_count_target,
        barrier_ready_value=barrier_ready_value,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_selected_dual_pack8_strided_fp16(
    x_ptr: int,
    selected_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch selected-expert dual gate/up pack8 GEMV for FP16 buffers."""

    _launch_selected_dual(
        _SYMBOL_SELECTED_DUAL_STRIDED_FP16,
        x_ptr,
        selected_ptr,
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        x_rows,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        num_experts,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_selected_dual_pack8_transposed_fp16(
    x_ptr: int,
    selected_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch selected-expert dual transposed pack8 GEMV for FP16 buffers."""

    _launch_selected_dual(
        _SYMBOL_SELECTED_DUAL_TRANSPOSED_FP16,
        x_ptr,
        selected_ptr,
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        x_rows,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        num_experts,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_selected_pack8_strided_fp16(
    x_ptr: int,
    selected_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    num_experts: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch selected-expert single/down pack8 GEMV for FP16 buffers."""

    _launch_selected_single(
        _SYMBOL_SELECTED_STRIDED_FP16,
        x_ptr,
        selected_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        num_experts,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_selected_pack8_transposed_fp16(
    x_ptr: int,
    selected_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    num_experts: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch selected-expert single/down transposed pack8 GEMV for FP16 buffers."""

    _launch_selected_single(
        _SYMBOL_SELECTED_TRANSPOSED_FP16,
        x_ptr,
        selected_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        num_experts,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_selected_pack8_transposed_silu_rotate_staged_fp16(
    gate_up_ptr: int,
    down_input_ptr: int,
    selected_ptr: int,
    pairs_ptr: int,
    theta_ptr: int,
    channel_scales_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    barrier_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    num_experts: int,
    group_size: int,
    krot: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch staged selected SiLU+down-rotate + transposed single GEMV."""

    _launch_selected_single_silu_rotate_staged(
        _SYMBOL_SELECTED_TRANSPOSED_SILU_ROTATE_STAGED_FP16,
        gate_up_ptr,
        down_input_ptr,
        selected_ptr,
        pairs_ptr,
        theta_ptr,
        channel_scales_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        barrier_ptr,
        rows,
        in_features,
        out_packed,
        num_experts,
        group_size,
        krot,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_selected_pack8_transposed_silu_rotate_staged_keyed_fp16(
    gate_up_ptr: int,
    down_input_ptr: int,
    selected_ptr: int,
    pairs_ptr: int,
    theta_ptr: int,
    channel_scales_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    barrier_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    num_experts: int,
    group_size: int,
    krot: int,
    barrier_count_target: int,
    barrier_ready_value: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Keyed staged selected SiLU+down-rotate + transposed single GEMV."""

    _launch_selected_single_silu_rotate_staged(
        _SYMBOL_SELECTED_TRANSPOSED_SILU_ROTATE_STAGED_KEYED_FP16,
        gate_up_ptr,
        down_input_ptr,
        selected_ptr,
        pairs_ptr,
        theta_ptr,
        channel_scales_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        barrier_ptr,
        rows,
        in_features,
        out_packed,
        num_experts,
        group_size,
        krot,
        barrier_count_target=barrier_count_target,
        barrier_ready_value=barrier_ready_value,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def register_paro_awq_gemv_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "pack8_gemv", "w4_paro", "strided"),
        gemv_awq_pack8_strided_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "pack8_gemv", "w4_paro", "transposed"),
        gemv_awq_pack8_transposed_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dual_pack8_gemv", "w4_paro", "strided"),
        gemv_awq_dual_pack8_strided_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dual_pack8_gemv", "w4_paro", "transposed"),
        gemv_awq_dual_pack8_transposed_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "rotate+dual_pack8_gemv", "w4_paro", "transposed"),
        gemv_awq_dual_pack8_transposed_rotate_staged_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "rotate+dual_pack8_gemv", "w4_paro", "transposed_keyed"),
        gemv_awq_dual_pack8_transposed_rotate_staged_keyed_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "pack8_gemv", "w4_paro", "strided_fp16"),
        gemv_awq_pack8_strided_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "pack8_gemv", "w4_paro", "transposed_fp16"),
        gemv_awq_pack8_transposed_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "pack8_gemm", "w4_paro", "fusedw4_prefill_fp16"),
        awq_fusedw4_prefill_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "pack8_gemm", "w4_paro", "fusedw4_prefill_dual_fp16"),
        awq_fusedw4_prefill_dual_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "pack8_gemm", "w4_paro", "fusedw4_prefill_strided_fp16"),
        awq_fusedw4_prefill_strided_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dual_pack8_gemv", "w4_paro", "strided_fp16"),
        gemv_awq_dual_pack8_strided_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dual_pack8_gemv", "w4_paro", "transposed_fp16"),
        gemv_awq_dual_pack8_transposed_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "rotate+dual_pack8_gemv", "w4_paro", "transposed_fp16"),
        gemv_awq_dual_pack8_transposed_rotate_staged_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "rotate+dual_pack8_gemv", "w4_paro", "transposed_keyed_fp16"),
        gemv_awq_dual_pack8_transposed_rotate_staged_keyed_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "rotate+selected_dual_pack8_gemv", "w4_paro", "strided"),
        gemv_awq_selected_dual_pack8_strided_rotate_out_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "selected_dual_pack8_gemv", "w4_paro", "strided"),
        gemv_awq_selected_dual_pack8_strided_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "selected_dual_pack8_gemv", "w4_paro", "transposed"),
        gemv_awq_selected_dual_pack8_transposed_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "selected_pack8_gemv", "w4_paro", "strided"),
        gemv_awq_selected_pack8_strided_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "selected_pack8_gemv", "w4_paro", "transposed"),
        gemv_awq_selected_pack8_transposed_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "rotate+selected_dual_pack8_gemv", "w4_paro", "strided_fp16"),
        gemv_awq_selected_dual_pack8_strided_rotate_out_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "rotate+selected_dual_pack8_gemv", "w4_paro", "transposed_fp16"),
        gemv_awq_selected_dual_pack8_transposed_rotate_out_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "rotate+selected_dual_pack8_gemv", "w4_paro", "transposed_staged_fp16"),
        gemv_awq_selected_dual_pack8_transposed_rotate_staged_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "rotate+selected_dual_pack8_gemv", "w4_paro", "transposed_staged_keyed_fp16"),
        gemv_awq_selected_dual_pack8_transposed_rotate_staged_keyed_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "selected_dual_pack8_gemv", "w4_paro", "strided_fp16"),
        gemv_awq_selected_dual_pack8_strided_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "selected_dual_pack8_gemv", "w4_paro", "transposed_fp16"),
        gemv_awq_selected_dual_pack8_transposed_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "selected_pack8_gemv", "w4_paro", "strided_fp16"),
        gemv_awq_selected_pack8_strided_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "selected_pack8_gemv", "w4_paro", "transposed_fp16"),
        gemv_awq_selected_pack8_transposed_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "silu_rotate+selected_pack8_gemv", "w4_paro", "transposed_staged_fp16"),
        gemv_awq_selected_pack8_transposed_silu_rotate_staged_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "silu_rotate+selected_pack8_gemv", "w4_paro", "transposed_staged_keyed_fp16"),
        gemv_awq_selected_pack8_transposed_silu_rotate_staged_keyed_fp16,
        replace=replace,
    )



def _launch_pack8_single(
    symbol: str,
    x_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_pack8_single_shape(rows, in_features, out_packed, group_size, threads)
    library = library or build_paro_awq_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, symbol, _ARGTYPES_PACK8_SINGLE, ctypes.c_int)
    err = fn(x_ptr, qweight_ptr, qzeros_ptr, scales_ptr, out_ptr,
             rows, in_features, out_packed, group_size, threads, stream)
    _check_launch(runtime, err)


def gemv_awq_dual_pack8_multi_row_split_transposed_fp16(
    x_a_ptr: int,
    x_b_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """M12.6: split-output multi-row dual W4 pack8 GEMV.

    Matches the ``awq_fusedw4_prefill_dual_fp16`` ABI (two separate output
    buffers ``out_a`` and ``out_b``) but uses the per-block row-loop weight-
    sharing of the new multi-row kernel.  ``rows`` must be in [1, 8].
    """

    _launch_dual_pack8_multi_row_split_transposed_fp16(
        _SYMBOL_DUAL_PACK8_MULTI_ROW_SPLIT_TRANSPOSED_FP16,
        x_a_ptr, x_b_ptr, qweight_a_ptr, qzeros_a_ptr, scales_a_ptr,
        qweight_b_ptr, qzeros_b_ptr, scales_b_ptr, out_a_ptr, out_b_ptr,
        rows, in_features, out_packed_a, out_packed_b, group_size,
        threads=threads, stream=stream, library=library, runtime=runtime,
    )


def gemv_awq_dual_pack8_multi_row_decode_split_transposed_fp16(
    x_a_ptr: int,
    x_b_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """M15.3: decode-dequant split-output multi-row dual W4 pack8 GEMV.

    Bit-identical to two ``gemv_awq_pack8_multi_row_decode_transposed_fp16``
    single GEMVs (same f32 dequant, k-order, and PARO_PACK8 reduction per
    output) but fuses both projections into one launch.  ``rows`` in [1, 8].
    """

    _launch_dual_pack8_multi_row_split_transposed_fp16(
        _SYMBOL_DUAL_PACK8_MULTI_ROW_DECODE_SPLIT_TRANSPOSED_FP16,
        x_a_ptr, x_b_ptr, qweight_a_ptr, qzeros_a_ptr, scales_a_ptr,
        qweight_b_ptr, qzeros_b_ptr, scales_b_ptr, out_a_ptr, out_b_ptr,
        rows, in_features, out_packed_a, out_packed_b, group_size,
        threads=threads, stream=stream, library=library, runtime=runtime,
    )


def _launch_dual_pack8_multi_row_split_transposed_fp16(
    symbol: str,
    x_a_ptr: int,
    x_b_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_pack8_dual_shape(rows, in_features, out_packed_a, out_packed_b, group_size, threads)
    library = library or build_paro_awq_gemv(load=True)
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
        ctypes.c_void_p(x_a_ptr),
        ctypes.c_void_p(x_b_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qzeros_a_ptr),
        ctypes.c_void_p(scales_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(qzeros_b_ptr),
        ctypes.c_void_p(scales_b_ptr),
        ctypes.c_void_p(out_a_ptr),
        ctypes.c_void_p(out_b_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_packed_a),
        ctypes.c_int64(out_packed_b),
        ctypes.c_int64(group_size),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_pack8_dual(
    symbol: str,
    input_ptrs: tuple[int, ...],
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    if len(input_ptrs) not in (1, 2):
        raise ValueError("input_ptrs must contain one or two pointers")
    _check_pack8_dual_shape(rows, in_features, out_packed_a, out_packed_b, group_size, threads)
    library = library or build_paro_awq_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    argtypes = _ARGTYPES_PACK8_DUAL_1 if len(input_ptrs) == 1 else _ARGTYPES_PACK8_DUAL_2
    fn = signed_kernel_fn(library, symbol, argtypes, ctypes.c_int)
    err = fn(*input_ptrs,
             qweight_a_ptr, qzeros_a_ptr, scales_a_ptr,
             qweight_b_ptr, qzeros_b_ptr, scales_b_ptr,
             out_ptr,
             rows, in_features, out_packed_a, out_packed_b, group_size, threads,
             stream)
    _check_launch(runtime, err)


def _launch_pack8_dual_rotate_staged(
    symbol: str,
    x_ptr: int,
    rotated_a_ptr: int,
    rotated_b_ptr: int,
    pairs_a_ptr: int,
    pairs_b_ptr: int,
    theta_a_ptr: int,
    theta_b_ptr: int,
    channel_scales_a_ptr: int,
    channel_scales_b_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    barrier_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    krot: int,
    *,
    barrier_count_target: int | None = None,
    barrier_ready_value: int | None = None,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_pack8_dual_rotate_staged_shape(rows, in_features, out_packed_a, out_packed_b, group_size, krot, threads)
    keyed = barrier_count_target is not None or barrier_ready_value is not None
    if keyed:
        if barrier_count_target is None or barrier_ready_value is None:
            raise ValueError("keyed rotate-staged GEMV requires both barrier_count_target and barrier_ready_value")
        if barrier_count_target <= 0 or barrier_ready_value <= 0:
            raise ValueError("keyed rotate-staged barrier targets must be positive")
        if barrier_count_target > 0x7FFFFFFF or barrier_ready_value > 0x7FFFFFFF:
            raise ValueError("keyed rotate-staged barrier targets must fit int32")
    library = library or build_paro_awq_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    argtypes = _ARGTYPES_PACK8_DUAL_ROTATE_STAGED_KEYED if keyed else _ARGTYPES_PACK8_DUAL_ROTATE_STAGED
    fn = signed_kernel_fn(library, symbol, argtypes, ctypes.c_int)
    common_args = (
        x_ptr,
        rotated_a_ptr,
        rotated_b_ptr,
        pairs_a_ptr,
        pairs_b_ptr,
        theta_a_ptr,
        theta_b_ptr,
        channel_scales_a_ptr,
        channel_scales_b_ptr,
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        barrier_ptr,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        group_size,
        krot,
        threads,
    )
    if keyed:
        err = fn(*common_args, barrier_count_target, barrier_ready_value, stream)
    else:
        err = fn(*common_args, stream)
    _check_launch(runtime, err)



def _launch_selected_dual_rotate_staged(
    symbol: str,
    x_ptr: int,
    rotated_ptr: int,
    selected_ptr: int,
    pairs_ptr: int,
    theta_ptr: int,
    channel_scales_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    barrier_ptr: int,
    x_rows: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    krot: int,
    *,
    barrier_count_target: int | None = None,
    barrier_ready_value: int | None = None,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_selected_dual_shape(x_rows, rows, in_features, out_packed_a, out_packed_b, num_experts, group_size, threads)
    if krot < 0:
        raise ValueError("krot must be non-negative")
    if group_size % 8 != 0:
        raise ValueError("group_size must be divisible by 8 for staged rotation")
    keyed = barrier_count_target is not None or barrier_ready_value is not None
    if keyed:
        if barrier_count_target is None or barrier_ready_value is None:
            raise ValueError("keyed selected rotate-staged GEMV requires both barrier target values")
        if barrier_count_target <= 0 or barrier_ready_value <= 0:
            raise ValueError("keyed selected rotate-staged barrier targets must be positive")
        if barrier_count_target > 0x7FFFFFFF or barrier_ready_value > 0x7FFFFFFF:
            raise ValueError("keyed selected rotate-staged barrier targets must fit int32")
    library = library or build_paro_awq_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    argtypes = _ARGTYPES_SELECTED_DUAL_ROTATE_STAGED_KEYED if keyed else _ARGTYPES_SELECTED_DUAL_ROTATE_STAGED
    fn = signed_kernel_fn(library, symbol, argtypes, ctypes.c_int)
    common_args = (
        x_ptr,
        rotated_ptr,
        selected_ptr,
        pairs_ptr,
        theta_ptr,
        channel_scales_ptr,
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        barrier_ptr,
        x_rows,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        num_experts,
        group_size,
        krot,
        threads,
    )
    if keyed:
        err = fn(*common_args, barrier_count_target, barrier_ready_value, stream)
    else:
        err = fn(*common_args, stream)
    _check_launch(runtime, err)


def _launch_selected_dual_rotate(
    symbol: str,
    x_ptr: int,
    selected_ptr: int,
    pairs_ptr: int,
    theta_ptr: int,
    channel_scales_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    krot: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_selected_dual_rotate_shape(
        x_rows,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        num_experts,
        group_size,
        krot,
        threads,
    )
    library = library or build_paro_awq_gemv(load=True)
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
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
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
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(pairs_ptr),
        ctypes.c_void_p(theta_ptr),
        ctypes.c_void_p(channel_scales_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qzeros_a_ptr),
        ctypes.c_void_p(scales_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(qzeros_b_ptr),
        ctypes.c_void_p(scales_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(x_rows),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_packed_a),
        ctypes.c_int64(out_packed_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(group_size),
        ctypes.c_int64(krot),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)

def _launch_selected_dual(
    symbol: str,
    x_ptr: int,
    selected_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_selected_dual_shape(
        x_rows,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        num_experts,
        group_size,
        threads,
    )
    library = library or build_paro_awq_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, symbol, _ARGTYPES_SELECTED_DUAL, ctypes.c_int)
    err = fn(x_ptr, selected_ptr,
             qweight_a_ptr, qzeros_a_ptr, scales_a_ptr,
             qweight_b_ptr, qzeros_b_ptr, scales_b_ptr,
             out_ptr,
             x_rows, rows, in_features, out_packed_a, out_packed_b, num_experts, group_size, threads,
             stream)
    _check_launch(runtime, err)


def _launch_selected_single_silu_rotate_staged(
    symbol: str,
    gate_up_ptr: int,
    down_input_ptr: int,
    selected_ptr: int,
    pairs_ptr: int,
    theta_ptr: int,
    channel_scales_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    barrier_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    num_experts: int,
    group_size: int,
    krot: int,
    *,
    barrier_count_target: int | None = None,
    barrier_ready_value: int | None = None,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_selected_single_shape(rows, in_features, out_packed, num_experts, group_size, threads)
    if krot <= 0:
        raise ValueError("krot must be positive")
    if group_size % 2 != 0:
        raise ValueError("group_size must be even for staged SiLU rotation")
    keyed = barrier_count_target is not None or barrier_ready_value is not None
    if keyed:
        if barrier_count_target is None or barrier_ready_value is None:
            raise ValueError("keyed selected SiLU-rotate staged GEMV requires both barrier target values")
        if barrier_count_target <= 0 or barrier_ready_value <= 0:
            raise ValueError("keyed selected SiLU-rotate staged barrier targets must be positive")
        if barrier_count_target > 0x7FFFFFFF or barrier_ready_value > 0x7FFFFFFF:
            raise ValueError("keyed selected SiLU-rotate staged barrier targets must fit int32")
    library = library or build_paro_awq_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    argtypes = _ARGTYPES_SELECTED_SINGLE_SILU_ROTATE_STAGED_KEYED if keyed else _ARGTYPES_SELECTED_SINGLE_SILU_ROTATE_STAGED
    fn = signed_kernel_fn(library, symbol, argtypes, ctypes.c_int)
    common_args = (
        gate_up_ptr,
        down_input_ptr,
        selected_ptr,
        pairs_ptr,
        theta_ptr,
        channel_scales_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        barrier_ptr,
        rows,
        in_features,
        out_packed,
        num_experts,
        group_size,
        krot,
        threads,
    )
    if keyed:
        err = fn(*common_args, barrier_count_target, barrier_ready_value, stream)
    else:
        err = fn(*common_args, stream)
    _check_launch(runtime, err)


def _launch_selected_single(
    symbol: str,
    x_ptr: int,
    selected_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    num_experts: int,
    group_size: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_selected_single_shape(rows, in_features, out_packed, num_experts, group_size, threads)
    library = library or build_paro_awq_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, symbol, _ARGTYPES_SELECTED_SINGLE, ctypes.c_int)
    err = fn(x_ptr, selected_ptr, qweight_ptr, qzeros_ptr, scales_ptr, out_ptr,
             rows, in_features, out_packed, num_experts, group_size, threads,
             stream)
    _check_launch(runtime, err)




def _launch_fusedw4_prefill_dual_fp16(
    symbol: str,
    x_a_ptr: int,
    x_b_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    tile_m: int | None,
    tile_n: int | None,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_fusedw4_shape(rows, in_features, out_packed_a, group_size, tile_m, tile_n)
    _check_positive(out_packed_b, "out_packed_b")
    if tile_m is None:
        tile_m = _default_fusedw4_tile_m(rows, max(out_packed_a, out_packed_b))
    if tile_n is None:
        tile_n = 32 if rows >= 32 else 16
    library = library or build_paro_awq_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, symbol, _ARGTYPES_FUSEDW4_PREFILL_DUAL, ctypes.c_int)
    err = fn(x_a_ptr, x_b_ptr,
             qweight_a_ptr, qzeros_a_ptr, scales_a_ptr,
             qweight_b_ptr, qzeros_b_ptr, scales_b_ptr,
             out_a_ptr, out_b_ptr,
             rows, in_features, out_packed_a, out_packed_b, group_size, tile_m, tile_n,
             stream)
    _check_launch(runtime, err)


def _launch_fusedw4_prefill_fp16(
    symbol: str,
    x_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    tile_m: int | None,
    tile_n: int | None,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_fusedw4_shape(rows, in_features, out_packed, group_size, tile_m, tile_n)
    if tile_m is None:
        tile_m = _default_fusedw4_tile_m(rows, out_packed)
    if tile_n is None:
        tile_n = 32 if rows >= 32 else 16
    library = library or build_paro_awq_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, symbol, _ARGTYPES_FUSEDW4_PREFILL_SINGLE, ctypes.c_int)
    err = fn(x_ptr, qweight_ptr, qzeros_ptr, scales_ptr, out_ptr,
             rows, in_features, out_packed, group_size, tile_m, tile_n,
             stream)
    _check_launch(runtime, err)


def _default_fusedw4_tile_m(rows: int, out_packed: int) -> int:
    base = 32 if int(out_packed) * 8 >= 32 else 16
    if int(rows) > 8:
        return base
    # B+1 verifier prefill paths are launch-bound and waste the 16-row WMMA
    # token tile.  Keeping the output tile at 16 improves the small-B MTP prompt
    # suite while preserving the exact prefill-numerics kernel.
    base = 16
    value = os.environ.get("HIPENGINE_W4_PREFILL_SMALLBATCH_TILE_M")
    if value is None or value.strip() == "":
        return base
    parsed = int(value)
    if parsed not in {16, 32, 64}:
        raise ValueError("HIPENGINE_W4_PREFILL_SMALLBATCH_TILE_M must be 16, 32, or 64")
    return parsed


def _check_fusedw4_shape(
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    tile_m: int | None,
    tile_n: int | None,
) -> None:
    _check_positive(rows, "rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_packed, "out_packed")
    _check_positive(group_size, "group_size")
    if in_features % group_size != 0:
        raise ValueError("in_features must be divisible by group_size")
    if group_size % 16 != 0:
        raise ValueError("group_size must be a multiple of 16")
    if tile_m is not None and tile_m not in {16, 32, 64}:
        raise ValueError("tile_m must be one of 16, 32, or 64")
    if tile_n is not None and tile_n not in {16, 32}:
        raise ValueError("tile_n must be one of 16 or 32")


def _check_pack8_single_shape(
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    threads: int,
) -> None:
    _check_positive(rows, "rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_packed, "out_packed")
    _check_pack8_common(in_features, group_size, threads)


def _check_pack8_dual_shape(
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    threads: int,
) -> None:
    _check_positive(rows, "rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_packed_a, "out_packed_a")
    _check_positive(out_packed_b, "out_packed_b")
    _check_pack8_common(in_features, group_size, threads)


def _check_pack8_dual_rotate_staged_shape(
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    krot: int,
    threads: int,
) -> None:
    _check_pack8_dual_shape(rows, in_features, out_packed_a, out_packed_b, group_size, threads)
    # M13.B.2: rows > 1 is supported after the kernel barrier was patched to
    # count rotate_blocks * gridDim.y; the only remaining requirement is
    # rows >= 1 (covered by _check_pack8_dual_shape via _check_positive).
    if krot < 0:
        raise ValueError("krot must be non-negative")
    rotate_blocks = (in_features // group_size) * 2
    if out_packed_a + out_packed_b < rotate_blocks:
        raise ValueError("out_packed_a + out_packed_b must cover the rotate staging blocks")



def _check_pack8_common(in_features: int, group_size: int, threads: int) -> None:
    _check_positive(group_size, "group_size")
    if in_features % group_size != 0:
        raise ValueError("in_features must be divisible by group_size")
    if group_size % 8 != 0:
        raise ValueError("group_size must be a multiple of 8")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64 or 128")


def _check_selected_dual_rotate_shape(
    x_rows: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    krot: int,
    threads: int,
) -> None:
    _check_selected_dual_shape(
        x_rows,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        num_experts,
        group_size,
        threads,
    )
    if krot < 0:
        raise ValueError("krot must be non-negative")

def _check_selected_dual_shape(
    x_rows: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    threads: int,
) -> None:
    _check_positive(x_rows, "x_rows")
    _check_positive(rows, "rows")
    if rows % x_rows != 0:
        raise ValueError("rows must be divisible by x_rows")
    _check_common_quant_shape(in_features, num_experts, group_size, threads)
    _check_positive(out_packed_a, "out_packed_a")
    _check_positive(out_packed_b, "out_packed_b")


def _check_selected_single_shape(
    rows: int,
    in_features: int,
    out_packed: int,
    num_experts: int,
    group_size: int,
    threads: int,
) -> None:
    _check_positive(rows, "rows")
    _check_common_quant_shape(in_features, num_experts, group_size, threads)
    _check_positive(out_packed, "out_packed")


def _check_common_quant_shape(
    in_features: int,
    num_experts: int,
    group_size: int,
    threads: int,
) -> None:
    _check_positive(in_features, "in_features")
    _check_positive(num_experts, "num_experts")
    _check_positive(group_size, "group_size")
    if in_features % group_size != 0:
        raise ValueError("in_features must be divisible by group_size")
    if group_size % 8 != 0:
        raise ValueError("group_size must be a multiple of 8")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64 or 128")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_paro_awq_gemv_kernels()
