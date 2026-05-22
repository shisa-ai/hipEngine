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

_SOURCE = Path(__file__).with_name("paro_combine.hip")
_OUTPUT_NAME = "paro_combine.so"
_SYMBOL_WEIGHTED_LANES = "hipengine_weighted_lanes_sum_out_bf16_f32w"
_SYMBOL_WEIGHTED_LANES_FP16 = "hipengine_weighted_lanes_sum_out_fp16_f32w"
_SYMBOL_WEIGHTED_SUM = "hipengine_weighted_sum_out_bf16_f32w"
_SYMBOL_WEIGHTED_SUM_FP16 = "hipengine_weighted_sum_out_fp16_f32w"
_SYMBOL_WEIGHTED_SHARED_RESIDUAL = "hipengine_weighted_sum_shared_gate_combine_residual_out_bf16_f32w"
_SYMBOL_WEIGHTED_SHARED_RESIDUAL_FP16 = "hipengine_weighted_sum_shared_gate_combine_residual_out_fp16_f32w"
_SYMBOL_WEIGHTED_SHARED_RESIDUAL_BATCH = "hipengine_weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w"
_SYMBOL_WEIGHTED_SHARED_RESIDUAL_BATCH_FP16 = "hipengine_weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w"
_SYMBOL_SHARED_COMBINE = "hipengine_shared_gate_combine_out_bf16"
_SYMBOL_SHARED_COMBINE_FP16 = "hipengine_shared_gate_combine_out_fp16"
_SYMBOL_SHARED_RESIDUAL = "hipengine_shared_gate_combine_residual_out_bf16"
_SYMBOL_SHARED_RESIDUAL_FP16 = "hipengine_shared_gate_combine_residual_out_fp16"
_SYMBOL_SHARED_RESIDUAL_BATCH = "hipengine_shared_gate_combine_residual_batch_out_bf16"
_SYMBOL_SHARED_RESIDUAL_BATCH_FP16 = "hipengine_shared_gate_combine_residual_batch_out_fp16"
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
        KernelKey("hip_gfx1100", "shared_gate_combine", "fp16", "out"),
        shared_gate_combine_out_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "shared_gate_combine+residual", "fp16", "out"),
        shared_gate_combine_residual_out_fp16,
        replace=replace,
    )


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


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_paro_combine_kernels()
