"""Raw-pointer wrappers for PARO weighted and shared-gate combine kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

# Cached argtypes tuples for the combine launchers used by the verifier.
_ARGTYPES_WEIGHTED_LANES = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGTYPES_WEIGHTED_LANES_SHARED_ADD = (
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
)
# shared / shared_batch input ptr counts: 3 (no-residual combines) or 4 (residual
# variants).  Verifier hot path is the 4-ptr residual variant.
_ARGTYPES_SHARED_3 = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,                                    # out
    ctypes.c_int64, ctypes.c_int64,                      # features, threads
    ctypes.c_void_p,                                    # stream
)
_ARGTYPES_SHARED_4 = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGTYPES_SHARED_BATCH_3 = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,                                    # out
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # tokens, features, gate_stride, threads
    ctypes.c_void_p,                                    # stream
)
_ARGTYPES_SHARED_BATCH_4 = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGTYPES_LAGUNA_WEIGHTED_TOP10_ROUTED_HIDDEN = (
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
)
_ARGTYPES_LAGUNA_AGGREGATE_TAIL_RMS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_void_p,
)
_ARGTYPES_TAIL_RMS_SHARED = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_void_p,
)
_ARGTYPES_TAIL_RMS_WEIGHTED = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_void_p,
)

_SOURCE = Path(__file__).with_name("paro_combine.hip")
_OUTPUT_NAME = "paro_combine.so"
_SYMBOL_WEIGHTED_LANES = "hipengine_weighted_lanes_sum_out_bf16_f32w"
_SYMBOL_WEIGHTED_LANES_FP16 = "hipengine_weighted_lanes_sum_out_fp16_f32w"
_SYMBOL_WEIGHTED_LANES_SHARED_ADD = (
    "hipengine_weighted_lanes_sum_shared_add_out_bf16_f32w"
)
_SYMBOL_WEIGHTED_LANES_SHARED_GATE_COMBINE_BATCH = (
    "hipengine_weighted_lanes_sum_shared_gate_combine_batch_out_bf16_f32w"
)
_ARGTYPES_WEIGHTED_LANES_SHARED_GATE_COMBINE_BATCH = (
    # values, weights, sorted_lanes, lane_to_row, shared, gate_logits, out
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
)
_SYMBOL_WEIGHTED_SUM = "hipengine_weighted_sum_out_bf16_f32w"
_SYMBOL_WEIGHTED_SUM_BATCH = "hipengine_weighted_sum_batch_out_bf16_f32w"
_SYMBOL_LAGUNA_WEIGHTED_TOP10_ROUTED_HIDDEN = (
    "hipengine_laguna_weighted_top10_routed_hidden_bf16_out"
)
_SYMBOL_WEIGHTED_SUM_FP16 = "hipengine_weighted_sum_out_fp16_f32w"
_SYMBOL_WEIGHTED_SHARED_RESIDUAL = "hipengine_weighted_sum_shared_gate_combine_residual_out_bf16_f32w"
_SYMBOL_WEIGHTED_SHARED_RESIDUAL_FP16 = "hipengine_weighted_sum_shared_gate_combine_residual_out_fp16_f32w"
_SYMBOL_WEIGHTED_SHARED_RESIDUAL_F32 = "hipengine_weighted_sum_shared_gate_combine_residual_out_f32_f32w"
_SYMBOL_WEIGHTED_SHARED_RESIDUAL_F32_ACCUM = "hipengine_weighted_sum_shared_gate_combine_residual_out_f32_accum_f32w"
_SYMBOL_WEIGHTED_F32_SHARED_RESIDUAL_F32_ACCUM = (
    "hipengine_weighted_sum_f32_shared_gate_combine_residual_out_f32_accum_f32w"
)
_SYMBOL_WEIGHTED_F32_SHARED_F32_RESIDUAL_F32_ACCUM = (
    "hipengine_weighted_sum_f32_shared_f32_gate_combine_residual_out_f32_accum_f32w"
)
_SYMBOL_WEIGHTED_SHARED_RESIDUAL_BATCH = "hipengine_weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w"
_SYMBOL_WEIGHTED_SHARED_RESIDUAL_BATCH_FP16 = "hipengine_weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w"
_SYMBOL_WEIGHTED_SHARED_RESIDUAL_BATCH_F32 = "hipengine_weighted_sum_shared_gate_combine_residual_batch_out_f32_f32w"
_SYMBOL_WEIGHTED_SHARED_RESIDUAL_BATCH_F32_ACCUM = "hipengine_weighted_sum_shared_gate_combine_residual_batch_out_f32_accum_f32w"
_SYMBOL_WEIGHTED_F32_SHARED_RESIDUAL_BATCH_F32_ACCUM = (
    "hipengine_weighted_sum_f32_shared_gate_combine_residual_batch_out_f32_accum_f32w"
)
_SYMBOL_WEIGHTED_F32_SHARED_F32_RESIDUAL_BATCH_F32_ACCUM = (
    "hipengine_weighted_sum_f32_shared_f32_gate_combine_residual_batch_out_f32_accum_f32w"
)
_SYMBOL_SHARED_COMBINE = "hipengine_shared_gate_combine_out_bf16"
_SYMBOL_SHARED_COMBINE_BATCH = "hipengine_shared_gate_combine_batch_out_bf16"
_SYMBOL_SHARED_COMBINE_FP16 = "hipengine_shared_gate_combine_out_fp16"
_SYMBOL_SHARED_RESIDUAL = "hipengine_shared_gate_combine_residual_out_bf16"
_SYMBOL_SHARED_RESIDUAL_FP16 = "hipengine_shared_gate_combine_residual_out_fp16"
_SYMBOL_SHARED_RESIDUAL_BATCH = "hipengine_shared_gate_combine_residual_batch_out_bf16"
_SYMBOL_SHARED_RESIDUAL_BATCH_FP16 = "hipengine_shared_gate_combine_residual_batch_out_fp16"
_SYMBOL_LAGUNA_AGGREGATE_TAIL_RMS = (
    "hipengine_laguna_aggregate_moe_tail_next_rmsnorm_gguf_bf16_out"
)
_SYMBOL_LAGUNA_AGGREGATE_TAIL_RMS_WAVE0_TREE = (
    "hipengine_laguna_aggregate_moe_tail_next_rmsnorm_wave0_tree_gguf_bf16_out"
)
_SYMBOL_TAIL_RMS_SHARED_GGUF_BF16 = "hipengine_shared_gate_combine_residual_rmsnorm_gguf_bf16_out"
_SYMBOL_TAIL_RMS_WEIGHTED_GGUF_BF16 = "hipengine_weighted_sum_shared_gate_combine_residual_rmsnorm_gguf_bf16_out"
_SYMBOL_TAIL_RMS_SHARED_PARO_BF16 = "hipengine_shared_gate_combine_residual_rmsnorm_paro_bf16_out"
_SYMBOL_TAIL_RMS_WEIGHTED_PARO_BF16 = "hipengine_weighted_sum_shared_gate_combine_residual_rmsnorm_paro_bf16_out"
_SYMBOL_TAIL_RMS_SHARED_PARO_FP16 = "hipengine_shared_gate_combine_residual_rmsnorm_paro_fp16_out"
_SYMBOL_TAIL_RMS_WEIGHTED_PARO_FP16 = "hipengine_weighted_sum_shared_gate_combine_residual_rmsnorm_paro_fp16_out"
_ALLOWED_THREADS = {64, 128, 256}


def plan_paro_combine_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="paro_combine",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_paro_combine(
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
        family="paro_combine",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def weighted_lanes_sum_out_bf16_f32w(
    values_ptr: int,
    weights_ptr: int,
    sorted_lanes_ptr: int,
    lane_to_row_ptr: int,
    out_ptr: int,
    tokens: int,
    top_k: int,
    features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch grouped-MoE sorted lane weighted sum into BF16 token-major rows."""

    _launch_weighted_lanes(
        _SYMBOL_WEIGHTED_LANES,
        values_ptr,
        weights_ptr,
        sorted_lanes_ptr,
        lane_to_row_ptr,
        out_ptr,
        tokens,
        top_k,
        features,
        threads,
        stream,
        library,
        runtime,
    )


def weighted_lanes_sum_shared_add_out_bf16_f32w(
    values_ptr: int,
    weights_ptr: int,
    sorted_lanes_ptr: int,
    lane_to_row_ptr: int,
    shared_ptr: int,
    out_ptr: int,
    tokens: int,
    top_k: int,
    features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact grouped weighted reduction plus BF16 shared-output add."""

    _check_positive(tokens, "tokens")
    _check_positive(top_k, "top_k")
    _check_vector_shape(features, threads)
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        _SYMBOL_WEIGHTED_LANES_SHARED_ADD,
        _ARGTYPES_WEIGHTED_LANES_SHARED_ADD,
        ctypes.c_int,
    )
    err = fn(
        values_ptr,
        weights_ptr,
        sorted_lanes_ptr,
        lane_to_row_ptr,
        shared_ptr,
        out_ptr,
        tokens,
        top_k,
        features,
        threads,
        stream,
    )
    _check_launch(runtime, err)


def weighted_lanes_sum_shared_gate_combine_batch_out_bf16_f32w(
    values_ptr: int,
    weights_ptr: int,
    sorted_lanes_ptr: int,
    lane_to_row_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    out_ptr: int,
    tokens: int,
    top_k: int,
    features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """PF-4 lever-2 T0 candidate: fused routed lanes-sum + gated shared combine.

    Bit-identical to the unfused production chain
    (``weighted_lanes_sum_out_bf16_f32w`` + ``shared_gate_combine_batch_out_bf16``)
    including the intermediate BF16 rounding of the routed sum. Strict
    fallback: that unfused chain (unchanged).
    """

    _check_positive(tokens, "tokens")
    _check_positive(top_k, "top_k")
    _check_vector_shape(features, threads)
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        _SYMBOL_WEIGHTED_LANES_SHARED_GATE_COMBINE_BATCH,
        _ARGTYPES_WEIGHTED_LANES_SHARED_GATE_COMBINE_BATCH,
        ctypes.c_int,
    )
    err = fn(
        values_ptr,
        weights_ptr,
        sorted_lanes_ptr,
        lane_to_row_ptr,
        shared_ptr,
        gate_logits_ptr,
        out_ptr,
        tokens,
        top_k,
        features,
        threads,
        stream,
    )
    _check_launch(runtime, err)


def weighted_lanes_sum_out_fp16_f32w(
    values_ptr: int,
    weights_ptr: int,
    sorted_lanes_ptr: int,
    lane_to_row_ptr: int,
    out_ptr: int,
    tokens: int,
    top_k: int,
    features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch grouped-MoE sorted lane weighted sum into FP16 token-major rows."""

    _launch_weighted_lanes(
        _SYMBOL_WEIGHTED_LANES_FP16,
        values_ptr,
        weights_ptr,
        sorted_lanes_ptr,
        lane_to_row_ptr,
        out_ptr,
        tokens,
        top_k,
        features,
        threads,
        stream,
        library,
        runtime,
    )


def weighted_sum_out_bf16_f32w(
    values_ptr: int,
    weights_ptr: int,
    out_ptr: int,
    rows: int,
    features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch selected-expert weighted sum into a caller-owned BF16 output row."""

    _check_matrix_shape(rows, features, threads)
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_WEIGHTED_SUM)
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
    err = fn(
        ctypes.c_void_p(values_ptr),
        ctypes.c_void_p(weights_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def weighted_sum_batch_out_bf16_f32w(
    values_ptr: int,
    weights_ptr: int,
    out_ptr: int,
    tokens: int,
    top_k: int,
    features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch token-local selected-expert weighted sums into BF16 rows."""

    _check_matrix_shape(tokens, features, threads)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_WEIGHTED_SUM_BATCH)
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 4 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(values_ptr),
        ctypes.c_void_p(weights_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(top_k),
        ctypes.c_int64(features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_weighted_top10_routed_hidden_bf16_out(
    expert_down_ptr: int,
    routing_weights_ptr: int,
    shared_ptr: int,
    post_attention_ptr: int,
    routed_out_ptr: int,
    hidden_out_ptr: int,
    rows: int,
    top_k: int,
    features: int,
    *,
    threads: int = 32,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Write exact Laguna top-10 routed and post-MoE BF16 rows."""

    if rows != 1:
        raise ValueError("rows must be exactly 1")
    if top_k != 10:
        raise ValueError("top_k must be exactly 10")
    if features != 3_072:
        raise ValueError("features must be exactly 3072")
    if threads != 32:
        raise ValueError("threads must be 32")
    _check_nonzero_pointers(
        expert_down_ptr=expert_down_ptr,
        routing_weights_ptr=routing_weights_ptr,
        shared_ptr=shared_ptr,
        post_attention_ptr=post_attention_ptr,
        routed_out_ptr=routed_out_ptr,
        hidden_out_ptr=hidden_out_ptr,
    )
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        _SYMBOL_LAGUNA_WEIGHTED_TOP10_ROUTED_HIDDEN,
        _ARGTYPES_LAGUNA_WEIGHTED_TOP10_ROUTED_HIDDEN,
        ctypes.c_int,
    )
    err = fn(
        expert_down_ptr,
        routing_weights_ptr,
        shared_ptr,
        post_attention_ptr,
        routed_out_ptr,
        hidden_out_ptr,
        rows,
        top_k,
        features,
        threads,
        stream,
    )
    _check_launch(runtime, err)


def weighted_sum_out_fp16_f32w(
    values_ptr: int,
    weights_ptr: int,
    out_ptr: int,
    rows: int,
    features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch selected-expert weighted sum into a caller-owned FP16 output row."""

    _check_matrix_shape(rows, features, threads)
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_WEIGHTED_SUM_FP16)
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
    err = fn(
        ctypes.c_void_p(values_ptr),
        ctypes.c_void_p(weights_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def weighted_sum_shared_gate_combine_residual_out_bf16_f32w(
    values_ptr: int,
    weights_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    rows: int,
    features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch c=1 selected weighted sum + shared-gate + residual fusion."""

    _check_matrix_shape(rows, features, threads)
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_WEIGHTED_SHARED_RESIDUAL)
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(values_ptr),
        ctypes.c_void_p(weights_ptr),
        ctypes.c_void_p(shared_ptr),
        ctypes.c_void_p(gate_logits_ptr),
        ctypes.c_void_p(residual_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def weighted_sum_shared_gate_combine_residual_out_fp16_f32w(
    values_ptr: int,
    weights_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    rows: int,
    features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch c=1 FP16 selected weighted sum + shared-gate + residual fusion."""

    _check_matrix_shape(rows, features, threads)
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_WEIGHTED_SHARED_RESIDUAL_FP16)
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(values_ptr),
        ctypes.c_void_p(weights_ptr),
        ctypes.c_void_p(shared_ptr),
        ctypes.c_void_p(gate_logits_ptr),
        ctypes.c_void_p(residual_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def weighted_sum_shared_gate_combine_residual_out_f32_f32w(
    values_ptr: int,
    weights_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    rows: int,
    features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 selected/shared MoE combine into an FP32 residual stream."""

    _check_matrix_shape(rows, features, threads)
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_WEIGHTED_SHARED_RESIDUAL_F32)
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(values_ptr),
        ctypes.c_void_p(weights_ptr),
        ctypes.c_void_p(shared_ptr),
        ctypes.c_void_p(gate_logits_ptr),
        ctypes.c_void_p(residual_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def weighted_sum_shared_gate_combine_residual_out_f32_accum_f32w(
    values_ptr: int,
    weights_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    rows: int,
    features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 selected/shared MoE combine into FP32 rows without selected-sum rounding."""

    _check_matrix_shape(rows, features, threads)
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_WEIGHTED_SHARED_RESIDUAL_F32_ACCUM)
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(values_ptr),
        ctypes.c_void_p(weights_ptr),
        ctypes.c_void_p(shared_ptr),
        ctypes.c_void_p(gate_logits_ptr),
        ctypes.c_void_p(residual_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def weighted_sum_f32_shared_gate_combine_residual_out_f32_accum_f32w(
    values_ptr: int,
    weights_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    rows: int,
    features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP32 selected/BF16 shared MoE combine into FP32 rows."""

    _check_matrix_shape(rows, features, threads)
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_WEIGHTED_F32_SHARED_RESIDUAL_F32_ACCUM)
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(values_ptr),
        ctypes.c_void_p(weights_ptr),
        ctypes.c_void_p(shared_ptr),
        ctypes.c_void_p(gate_logits_ptr),
        ctypes.c_void_p(residual_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def weighted_sum_f32_shared_f32_gate_combine_residual_out_f32_accum_f32w(
    values_ptr: int,
    weights_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    rows: int,
    features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP32 selected/FP32 shared MoE combine into FP32 rows."""

    _check_matrix_shape(rows, features, threads)
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_WEIGHTED_F32_SHARED_F32_RESIDUAL_F32_ACCUM)
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(values_ptr),
        ctypes.c_void_p(weights_ptr),
        ctypes.c_void_p(shared_ptr),
        ctypes.c_void_p(gate_logits_ptr),
        ctypes.c_void_p(residual_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w(
    values_ptr: int,
    weights_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    tokens: int,
    rows_per_token: int,
    features: int,
    gate_stride: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch batched selected weighted sum + shared-gate + residual fusion."""

    _check_positive(tokens, "tokens")
    _check_positive(rows_per_token, "rows_per_token")
    _check_vector_shape(features, threads)
    _check_positive(gate_stride, "gate_stride")
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_WEIGHTED_SHARED_RESIDUAL_BATCH)
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
        ctypes.c_void_p(values_ptr),
        ctypes.c_void_p(weights_ptr),
        ctypes.c_void_p(shared_ptr),
        ctypes.c_void_p(gate_logits_ptr),
        ctypes.c_void_p(residual_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(rows_per_token),
        ctypes.c_int64(features),
        ctypes.c_int64(gate_stride),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def weighted_sum_shared_gate_combine_residual_batch_out_f32_f32w(
    values_ptr: int,
    weights_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    tokens: int,
    rows_per_token: int,
    features: int,
    gate_stride: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch batched BF16 selected/shared MoE combine into FP32 rows."""

    _check_positive(tokens, "tokens")
    _check_positive(rows_per_token, "rows_per_token")
    _check_vector_shape(features, threads)
    _check_positive(gate_stride, "gate_stride")
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_WEIGHTED_SHARED_RESIDUAL_BATCH_F32)
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
        ctypes.c_void_p(values_ptr),
        ctypes.c_void_p(weights_ptr),
        ctypes.c_void_p(shared_ptr),
        ctypes.c_void_p(gate_logits_ptr),
        ctypes.c_void_p(residual_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(rows_per_token),
        ctypes.c_int64(features),
        ctypes.c_int64(gate_stride),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def weighted_sum_shared_gate_combine_residual_batch_out_f32_accum_f32w(
    values_ptr: int,
    weights_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    tokens: int,
    rows_per_token: int,
    features: int,
    gate_stride: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch batched BF16 selected/shared MoE combine into FP32 rows without selected-sum rounding."""

    _check_positive(tokens, "tokens")
    _check_positive(rows_per_token, "rows_per_token")
    _check_vector_shape(features, threads)
    _check_positive(gate_stride, "gate_stride")
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_WEIGHTED_SHARED_RESIDUAL_BATCH_F32_ACCUM)
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
        ctypes.c_void_p(values_ptr),
        ctypes.c_void_p(weights_ptr),
        ctypes.c_void_p(shared_ptr),
        ctypes.c_void_p(gate_logits_ptr),
        ctypes.c_void_p(residual_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(rows_per_token),
        ctypes.c_int64(features),
        ctypes.c_int64(gate_stride),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def weighted_sum_f32_shared_gate_combine_residual_batch_out_f32_accum_f32w(
    values_ptr: int,
    weights_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    tokens: int,
    rows_per_token: int,
    features: int,
    gate_stride: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch batched FP32 selected/BF16 shared MoE combine into FP32 rows."""

    _check_positive(tokens, "tokens")
    _check_positive(rows_per_token, "rows_per_token")
    _check_vector_shape(features, threads)
    _check_positive(gate_stride, "gate_stride")
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_WEIGHTED_F32_SHARED_RESIDUAL_BATCH_F32_ACCUM)
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
        ctypes.c_void_p(values_ptr),
        ctypes.c_void_p(weights_ptr),
        ctypes.c_void_p(shared_ptr),
        ctypes.c_void_p(gate_logits_ptr),
        ctypes.c_void_p(residual_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(rows_per_token),
        ctypes.c_int64(features),
        ctypes.c_int64(gate_stride),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def weighted_sum_f32_shared_f32_gate_combine_residual_batch_out_f32_accum_f32w(
    values_ptr: int,
    weights_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    tokens: int,
    rows_per_token: int,
    features: int,
    gate_stride: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch batched FP32 selected/FP32 shared MoE combine into FP32 rows."""

    _check_positive(tokens, "tokens")
    _check_positive(rows_per_token, "rows_per_token")
    _check_vector_shape(features, threads)
    _check_positive(gate_stride, "gate_stride")
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_WEIGHTED_F32_SHARED_F32_RESIDUAL_BATCH_F32_ACCUM)
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
        ctypes.c_void_p(values_ptr),
        ctypes.c_void_p(weights_ptr),
        ctypes.c_void_p(shared_ptr),
        ctypes.c_void_p(gate_logits_ptr),
        ctypes.c_void_p(residual_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(rows_per_token),
        ctypes.c_int64(features),
        ctypes.c_int64(gate_stride),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w(
    values_ptr: int,
    weights_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    tokens: int,
    rows_per_token: int,
    features: int,
    gate_stride: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch batched FP16 selected weighted sum + shared-gate + residual fusion."""

    _check_positive(tokens, "tokens")
    _check_positive(rows_per_token, "rows_per_token")
    _check_vector_shape(features, threads)
    _check_positive(gate_stride, "gate_stride")
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_WEIGHTED_SHARED_RESIDUAL_BATCH_FP16)
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
        ctypes.c_void_p(values_ptr),
        ctypes.c_void_p(weights_ptr),
        ctypes.c_void_p(shared_ptr),
        ctypes.c_void_p(gate_logits_ptr),
        ctypes.c_void_p(residual_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(rows_per_token),
        ctypes.c_int64(features),
        ctypes.c_int64(gate_stride),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_aggregate_moe_tail_next_rmsnorm_gguf_bf16_out(
    routed_ptr: int,
    shared_ptr: int,
    post_attention_ptr: int,
    norm_weight_ptr: int,
    norm_out_ptr: int,
    hidden_out_ptr: int,
    features: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Preserve both Laguna BF16 add boundaries and emit next RMSNorm."""

    _check_positive(features, "features")
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        _SYMBOL_LAGUNA_AGGREGATE_TAIL_RMS,
        _ARGTYPES_LAGUNA_AGGREGATE_TAIL_RMS,
        ctypes.c_int,
    )
    err = fn(
        routed_ptr,
        shared_ptr,
        post_attention_ptr,
        norm_weight_ptr,
        norm_out_ptr,
        hidden_out_ptr,
        features,
        float(eps),
        stream,
    )
    _check_launch(runtime, err)


def laguna_aggregate_moe_tail_next_rmsnorm_wave0_tree_gguf_bf16_out(
    routed_ptr: int,
    shared_ptr: int,
    post_attention_ptr: int,
    norm_weight_ptr: int,
    norm_out_ptr: int,
    hidden_out_ptr: int,
    features: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Emit exact Laguna D9 outputs with the wave-0 RMS reduction tree."""

    _check_positive(features, "features")
    _check_nonzero_pointers(
        routed_ptr=routed_ptr,
        shared_ptr=shared_ptr,
        post_attention_ptr=post_attention_ptr,
        norm_weight_ptr=norm_weight_ptr,
        norm_out_ptr=norm_out_ptr,
        hidden_out_ptr=hidden_out_ptr,
    )
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        _SYMBOL_LAGUNA_AGGREGATE_TAIL_RMS_WAVE0_TREE,
        _ARGTYPES_LAGUNA_AGGREGATE_TAIL_RMS,
        ctypes.c_int,
    )
    err = fn(
        routed_ptr,
        shared_ptr,
        post_attention_ptr,
        norm_weight_ptr,
        norm_out_ptr,
        hidden_out_ptr,
        features,
        float(eps),
        stream,
    )
    _check_launch(runtime, err)


def shared_gate_combine_residual_rmsnorm_gguf_bf16_out(
    selected_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    norm_weight_ptr: int,
    norm_out_ptr: int,
    residual_out_ptr: int,
    tokens: int,
    features: int,
    gate_stride: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Fuse an already-rounded BF16 MoE aggregate with GGUF next-input RMSNorm."""

    _launch_tail_rms_shared(
        _SYMBOL_TAIL_RMS_SHARED_GGUF_BF16,
        selected_ptr,
        shared_ptr,
        gate_logits_ptr,
        residual_ptr,
        norm_weight_ptr,
        norm_out_ptr,
        residual_out_ptr,
        tokens,
        features,
        gate_stride,
        eps,
        stream,
        library,
        runtime,
    )


def weighted_sum_shared_gate_combine_residual_rmsnorm_gguf_bf16_out(
    values_ptr: int,
    weights_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    norm_weight_ptr: int,
    norm_out_ptr: int,
    residual_out_ptr: int,
    tokens: int,
    rows_per_token: int,
    features: int,
    gate_stride: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Fuse slot-ordered BF16 selected sum with GGUF next-input RMSNorm."""

    _launch_tail_rms_weighted(
        _SYMBOL_TAIL_RMS_WEIGHTED_GGUF_BF16,
        values_ptr,
        weights_ptr,
        shared_ptr,
        gate_logits_ptr,
        residual_ptr,
        norm_weight_ptr,
        norm_out_ptr,
        residual_out_ptr,
        tokens,
        rows_per_token,
        features,
        gate_stride,
        eps,
        stream,
        library,
        runtime,
    )


def shared_gate_combine_residual_rmsnorm_paro_bf16_out(
    selected_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    norm_weight_ptr: int,
    norm_out_ptr: int,
    residual_out_ptr: int,
    tokens: int,
    features: int,
    gate_stride: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Fuse an already-rounded BF16 MoE aggregate with PARO next-input RMSNorm."""

    _launch_tail_rms_shared(
        _SYMBOL_TAIL_RMS_SHARED_PARO_BF16,
        selected_ptr,
        shared_ptr,
        gate_logits_ptr,
        residual_ptr,
        norm_weight_ptr,
        norm_out_ptr,
        residual_out_ptr,
        tokens,
        features,
        gate_stride,
        eps,
        stream,
        library,
        runtime,
    )


def weighted_sum_shared_gate_combine_residual_rmsnorm_paro_bf16_out(
    values_ptr: int,
    weights_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    norm_weight_ptr: int,
    norm_out_ptr: int,
    residual_out_ptr: int,
    tokens: int,
    rows_per_token: int,
    features: int,
    gate_stride: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Fuse slot-ordered BF16 selected sum with PARO next-input RMSNorm."""

    _launch_tail_rms_weighted(
        _SYMBOL_TAIL_RMS_WEIGHTED_PARO_BF16,
        values_ptr,
        weights_ptr,
        shared_ptr,
        gate_logits_ptr,
        residual_ptr,
        norm_weight_ptr,
        norm_out_ptr,
        residual_out_ptr,
        tokens,
        rows_per_token,
        features,
        gate_stride,
        eps,
        stream,
        library,
        runtime,
    )


def shared_gate_combine_residual_rmsnorm_paro_fp16_out(
    selected_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    norm_weight_ptr: int,
    norm_out_ptr: int,
    residual_out_ptr: int,
    tokens: int,
    features: int,
    gate_stride: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Fuse an already-rounded FP16 MoE aggregate with PARO next-input RMSNorm."""

    _launch_tail_rms_shared(
        _SYMBOL_TAIL_RMS_SHARED_PARO_FP16,
        selected_ptr,
        shared_ptr,
        gate_logits_ptr,
        residual_ptr,
        norm_weight_ptr,
        norm_out_ptr,
        residual_out_ptr,
        tokens,
        features,
        gate_stride,
        eps,
        stream,
        library,
        runtime,
    )


def weighted_sum_shared_gate_combine_residual_rmsnorm_paro_fp16_out(
    values_ptr: int,
    weights_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    norm_weight_ptr: int,
    norm_out_ptr: int,
    residual_out_ptr: int,
    tokens: int,
    rows_per_token: int,
    features: int,
    gate_stride: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Fuse slot-ordered FP16 selected sum with PARO next-input RMSNorm."""

    _launch_tail_rms_weighted(
        _SYMBOL_TAIL_RMS_WEIGHTED_PARO_FP16,
        values_ptr,
        weights_ptr,
        shared_ptr,
        gate_logits_ptr,
        residual_ptr,
        norm_weight_ptr,
        norm_out_ptr,
        residual_out_ptr,
        tokens,
        rows_per_token,
        features,
        gate_stride,
        eps,
        stream,
        library,
        runtime,
    )


def shared_gate_combine_out_bf16(
    expert_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    out_ptr: int,
    features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch expert + sigmoid(shared-gate) * shared combine."""

    _launch_shared(
        _SYMBOL_SHARED_COMBINE,
        (expert_ptr, shared_ptr, gate_logits_ptr),
        out_ptr,
        features,
        threads,
        stream,
        library,
        runtime,
    )


def shared_gate_combine_batch_out_bf16(
    expert_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    out_ptr: int,
    tokens: int,
    features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch token-local expert + sigmoid(shared gate) * shared BF16 rows."""

    _check_matrix_shape(tokens, features, threads)
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SHARED_COMBINE_BATCH)
    fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 3 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(expert_ptr),
        ctypes.c_void_p(shared_ptr),
        ctypes.c_void_p(gate_logits_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def shared_gate_combine_out_fp16(
    expert_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    out_ptr: int,
    features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 expert + sigmoid(shared-gate) * shared combine."""

    _launch_shared(
        _SYMBOL_SHARED_COMBINE_FP16,
        (expert_ptr, shared_ptr, gate_logits_ptr),
        out_ptr,
        features,
        threads,
        stream,
        library,
        runtime,
    )


def shared_gate_combine_residual_out_bf16(
    expert_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch residual + expert + sigmoid(shared-gate) * shared combine."""

    _launch_shared(
        _SYMBOL_SHARED_RESIDUAL,
        (expert_ptr, shared_ptr, gate_logits_ptr, residual_ptr),
        out_ptr,
        features,
        threads,
        stream,
        library,
        runtime,
    )


def shared_gate_combine_residual_batch_out_bf16(
    expert_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    tokens: int,
    features: int,
    gate_stride: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch batched residual + expert + sigmoid(shared-gate) * shared combine."""

    _launch_shared_batch(
        _SYMBOL_SHARED_RESIDUAL_BATCH,
        (expert_ptr, shared_ptr, gate_logits_ptr, residual_ptr),
        out_ptr,
        tokens,
        features,
        gate_stride,
        threads,
        stream,
        library,
        runtime,
    )


def shared_gate_combine_residual_batch_out_fp16(
    expert_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    tokens: int,
    features: int,
    gate_stride: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch batched FP16 residual + expert + sigmoid(shared-gate) * shared combine."""

    _launch_shared_batch(
        _SYMBOL_SHARED_RESIDUAL_BATCH_FP16,
        (expert_ptr, shared_ptr, gate_logits_ptr, residual_ptr),
        out_ptr,
        tokens,
        features,
        gate_stride,
        threads,
        stream,
        library,
        runtime,
    )


def shared_gate_combine_residual_out_fp16(
    expert_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 residual + expert + sigmoid(shared-gate) * shared combine."""

    _launch_shared(
        _SYMBOL_SHARED_RESIDUAL_FP16,
        (expert_ptr, shared_ptr, gate_logits_ptr, residual_ptr),
        out_ptr,
        features,
        threads,
        stream,
        library,
        runtime,
    )


def register_paro_combine_kernels(*, replace: bool = True) -> None:
    for quant in ("bf16", "w4_paro"):
        register(
            KernelKey("hip_gfx1100", "weighted_lanes_sum", quant, "out"),
            weighted_lanes_sum_out_bf16_f32w,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "weighted_lanes_sum", quant, "out_fp16"),
            weighted_lanes_sum_out_fp16_f32w,
            replace=replace,
        )
        register(
            KernelKey(
                "hip_gfx1100",
                "weighted_lanes_sum+shared_add",
                quant,
                "out",
            ),
            weighted_lanes_sum_shared_add_out_bf16_f32w,
            replace=replace,
        )
        register(
            KernelKey(
                "hip_gfx1100",
                "weighted_lanes_sum+shared_gate_combine",
                quant,
                "out",
            ),
            weighted_lanes_sum_shared_gate_combine_batch_out_bf16_f32w,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "weighted_sum", quant, "out"),
            weighted_sum_out_bf16_f32w,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "weighted_sum", quant, "out_fp16"),
            weighted_sum_out_fp16_f32w,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "weighted_sum+shared_gate+residual", quant, "out"),
            weighted_sum_shared_gate_combine_residual_out_bf16_f32w,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "weighted_sum+shared_gate+residual", quant, "out_fp16"),
            weighted_sum_shared_gate_combine_residual_out_fp16_f32w,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "weighted_sum+shared_gate+residual", quant, "out_f32"),
            weighted_sum_shared_gate_combine_residual_out_f32_f32w,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "weighted_sum+shared_gate+residual", quant, "out_f32_accum"),
            weighted_sum_shared_gate_combine_residual_out_f32_accum_f32w,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "weighted_sum+shared_gate+residual", quant, "batch_out"),
            weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "weighted_sum+shared_gate+residual", quant, "batch_out_fp16"),
            weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "weighted_sum+shared_gate+residual", quant, "batch_out_f32"),
            weighted_sum_shared_gate_combine_residual_batch_out_f32_f32w,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "weighted_sum+shared_gate+residual", quant, "batch_out_f32_accum"),
            weighted_sum_shared_gate_combine_residual_batch_out_f32_accum_f32w,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "shared_gate_combine", quant, "out"),
            shared_gate_combine_out_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "shared_gate_combine", quant, "out_fp16"),
            shared_gate_combine_out_fp16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "shared_gate_combine+residual", quant, "out"),
            shared_gate_combine_residual_out_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "shared_gate_combine+residual", quant, "out_fp16"),
            shared_gate_combine_residual_out_fp16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "shared_gate_combine+residual", quant, "batch_out"),
            shared_gate_combine_residual_batch_out_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "shared_gate_combine+residual", quant, "batch_out_fp16"),
            shared_gate_combine_residual_batch_out_fp16,
            replace=replace,
        )
    register(
        KernelKey(
            "hip_gfx1100",
            "weighted_sum+moe_tail",
            "bf16",
            "laguna_top10_routed_hidden_out",
        ),
        laguna_weighted_top10_routed_hidden_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_tail+next_rmsnorm",
            "bf16",
            "laguna_aggregate_gguf_f32_weight_out",
        ),
        laguna_aggregate_moe_tail_next_rmsnorm_gguf_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_tail+next_rmsnorm",
            "bf16",
            "laguna_aggregate_wave0_tree_gguf_f32_weight_out",
        ),
        laguna_aggregate_moe_tail_next_rmsnorm_wave0_tree_gguf_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "shared_gate_combine+residual+rmsnorm",
            "gguf_f32_weight",
            "bf16_out",
        ),
        shared_gate_combine_residual_rmsnorm_gguf_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "weighted_sum+shared_gate+residual+rmsnorm",
            "gguf_f32_weight",
            "bf16_out",
        ),
        weighted_sum_shared_gate_combine_residual_rmsnorm_gguf_bf16_out,
        replace=replace,
    )
    for quant in ("bf16", "w4_paro"):
        register(
            KernelKey(
                "hip_gfx1100",
                "shared_gate_combine+residual+rmsnorm",
                quant,
                "paro_out",
            ),
            shared_gate_combine_residual_rmsnorm_paro_bf16_out,
            replace=replace,
        )
        register(
            KernelKey(
                "hip_gfx1100",
                "weighted_sum+shared_gate+residual+rmsnorm",
                quant,
                "paro_out",
            ),
            weighted_sum_shared_gate_combine_residual_rmsnorm_paro_bf16_out,
            replace=replace,
        )
        register(
            KernelKey(
                "hip_gfx1100",
                "shared_gate_combine+residual+rmsnorm",
                quant,
                "paro_out_fp16",
            ),
            shared_gate_combine_residual_rmsnorm_paro_fp16_out,
            replace=replace,
        )
        register(
            KernelKey(
                "hip_gfx1100",
                "weighted_sum+shared_gate+residual+rmsnorm",
                quant,
                "paro_out_fp16",
            ),
            weighted_sum_shared_gate_combine_residual_rmsnorm_paro_fp16_out,
            replace=replace,
        )
    register(
        KernelKey("hip_gfx1100", "weighted_sum", "fp16", "out"),
        weighted_sum_out_fp16_f32w,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "weighted_sum+shared_gate+residual", "fp16", "out"),
        weighted_sum_shared_gate_combine_residual_out_fp16_f32w,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "weighted_sum+shared_gate+residual", "fp16", "batch_out"),
        weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "weighted_sum+shared_gate+residual", "f32", "out"),
        weighted_sum_shared_gate_combine_residual_out_f32_f32w,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "weighted_sum+shared_gate+residual", "f32", "out_accum"),
        weighted_sum_shared_gate_combine_residual_out_f32_accum_f32w,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "weighted_sum+shared_gate+residual", "f32_selected", "out_accum"),
        weighted_sum_f32_shared_gate_combine_residual_out_f32_accum_f32w,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "weighted_sum+shared_gate+residual", "f32_selected_shared", "out_accum"),
        weighted_sum_f32_shared_f32_gate_combine_residual_out_f32_accum_f32w,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "weighted_sum+shared_gate+residual", "f32", "batch_out"),
        weighted_sum_shared_gate_combine_residual_batch_out_f32_f32w,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "weighted_sum+shared_gate+residual", "f32", "batch_out_accum"),
        weighted_sum_shared_gate_combine_residual_batch_out_f32_accum_f32w,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "weighted_sum+shared_gate+residual", "f32_selected", "batch_out_accum"),
        weighted_sum_f32_shared_gate_combine_residual_batch_out_f32_accum_f32w,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "weighted_sum+shared_gate+residual", "f32_selected_shared", "batch_out_accum"),
        weighted_sum_f32_shared_f32_gate_combine_residual_batch_out_f32_accum_f32w,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "shared_gate_combine", "fp16", "out"),
        shared_gate_combine_out_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "shared_gate_combine+residual", "fp16", "out"),
        shared_gate_combine_residual_out_fp16,
        replace=replace,
    )


def _launch_tail_rms_shared(
    symbol: str,
    selected_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    norm_weight_ptr: int,
    norm_out_ptr: int,
    residual_out_ptr: int,
    tokens: int,
    features: int,
    gate_stride: int,
    eps: float,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_positive(tokens, "tokens")
    _check_positive(features, "features")
    _check_positive(gate_stride, "gate_stride")
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, symbol, _ARGTYPES_TAIL_RMS_SHARED, ctypes.c_int)
    err = fn(
        selected_ptr,
        shared_ptr,
        gate_logits_ptr,
        residual_ptr,
        norm_weight_ptr,
        norm_out_ptr,
        residual_out_ptr,
        tokens,
        features,
        gate_stride,
        float(eps),
        stream,
    )
    _check_launch(runtime, err)


def _launch_tail_rms_weighted(
    symbol: str,
    values_ptr: int,
    weights_ptr: int,
    shared_ptr: int,
    gate_logits_ptr: int,
    residual_ptr: int,
    norm_weight_ptr: int,
    norm_out_ptr: int,
    residual_out_ptr: int,
    tokens: int,
    rows_per_token: int,
    features: int,
    gate_stride: int,
    eps: float,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_positive(tokens, "tokens")
    _check_positive(rows_per_token, "rows_per_token")
    _check_positive(features, "features")
    _check_positive(gate_stride, "gate_stride")
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, symbol, _ARGTYPES_TAIL_RMS_WEIGHTED, ctypes.c_int)
    err = fn(
        values_ptr,
        weights_ptr,
        shared_ptr,
        gate_logits_ptr,
        residual_ptr,
        norm_weight_ptr,
        norm_out_ptr,
        residual_out_ptr,
        tokens,
        rows_per_token,
        features,
        gate_stride,
        float(eps),
        stream,
    )
    _check_launch(runtime, err)


def _launch_weighted_lanes(
    symbol: str,
    values_ptr: int,
    weights_ptr: int,
    sorted_lanes_ptr: int,
    lane_to_row_ptr: int,
    out_ptr: int,
    tokens: int,
    top_k: int,
    features: int,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_positive(tokens, "tokens")
    _check_positive(top_k, "top_k")
    _check_vector_shape(features, threads)
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, symbol, _ARGTYPES_WEIGHTED_LANES, ctypes.c_int)
    err = fn(values_ptr, weights_ptr, sorted_lanes_ptr, lane_to_row_ptr, out_ptr,
             tokens, top_k, features, threads, stream)
    _check_launch(runtime, err)


def _launch_shared(
    symbol: str,
    input_ptrs: tuple[int, ...],
    out_ptr: int,
    features: int,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_vector_shape(features, threads)
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    argtypes = _ARGTYPES_SHARED_3 if len(input_ptrs) == 3 else _ARGTYPES_SHARED_4
    fn = signed_kernel_fn(library, symbol, argtypes, ctypes.c_int)
    err = fn(*input_ptrs, out_ptr, features, threads, stream)
    _check_launch(runtime, err)


def _launch_shared_batch(
    symbol: str,
    input_ptrs: tuple[int, ...],
    out_ptr: int,
    tokens: int,
    features: int,
    gate_stride: int,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_positive(tokens, "tokens")
    _check_vector_shape(features, threads)
    _check_positive(gate_stride, "gate_stride")
    library = library or build_paro_combine(load=True)
    runtime = runtime or get_hip_runtime()
    argtypes = _ARGTYPES_SHARED_BATCH_3 if len(input_ptrs) == 3 else _ARGTYPES_SHARED_BATCH_4
    fn = signed_kernel_fn(library, symbol, argtypes, ctypes.c_int)
    err = fn(*input_ptrs, out_ptr, tokens, features, gate_stride, threads, stream)
    _check_launch(runtime, err)


def _check_matrix_shape(rows: int, features: int, threads: int) -> None:
    _check_positive(rows, "rows")
    _check_vector_shape(features, threads)


def _check_vector_shape(features: int, threads: int) -> None:
    _check_positive(features, "features")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, or 256")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_nonzero_pointers(**pointers: int) -> None:
    for name, pointer in pointers.items():
        if pointer == 0:
            raise ValueError(f"{name} must be non-zero")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_paro_combine_kernels()
