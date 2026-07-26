"""Raw-pointer wrapper for Laguna sigmoid/correction top-k routing."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("laguna_router.hip")
_OUTPUT_NAME = "laguna_router.so"
_SYMBOL = "hipengine_laguna_sigmoid_correction_topk_f32"
_PERSISTENT_WAVE_TOP10_SYMBOL = (
    "hipengine_laguna_router_topk_bf16_f32w_correction_persistent_wave_top10"
)
_WEIGHTED_SUM_ROWS_SYMBOL = "hipengine_laguna_weighted_sum_rows_bf16_f32w"
_THREADS = 256
_MAX_HIDDEN_SIZE = 3_072
_MAX_EXPERTS = 256
_MAX_TOP_K = 16
_PERSISTENT_ARGTYPES = (
    ctypes.c_void_p,  # hidden [hidden_size] bf16 bits
    ctypes.c_void_p,  # router weight [experts, hidden_size] f32
    ctypes.c_void_p,  # correction bias [experts] f32
    ctypes.c_void_p,  # logits [experts] f32
    ctypes.c_void_p,  # unbiased sigmoid scores [experts] f32
    ctypes.c_void_p,  # corrected selection scores [experts] f32
    ctypes.c_void_p,  # selected ids [top_k] i64
    ctypes.c_void_p,  # normalized unbiased weights [top_k] f32
    ctypes.c_void_p,  # scaled normalized weights [top_k] f32
    ctypes.c_void_p,  # self-resetting completion counter [1] i32
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_WEIGHTED_SUM_ROWS_ARGTYPES = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGTYPES = (
    ctypes.c_void_p,  # logits [tokens, experts] f32
    ctypes.c_void_p,  # correction bias [experts] f32
    ctypes.c_void_p,  # unbiased sigmoid scores [tokens, experts] f32
    ctypes.c_void_p,  # corrected selection scores [tokens, experts] f32
    ctypes.c_void_p,  # selected ids [tokens, top_k] i64
    ctypes.c_void_p,  # normalized unbiased weights [tokens, top_k] f32
    ctypes.c_void_p,  # scaled normalized weights [tokens, top_k] f32
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_int64,
    ctypes.c_void_p,
)


def plan_laguna_router_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="laguna_router",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_laguna_router(
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
        family="laguna_router",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def laguna_sigmoid_correction_topk_f32(
    logits_ptr: int,
    correction_bias_ptr: int,
    routing_scores_ptr: int,
    selection_scores_ptr: int,
    selected_ptr: int,
    routing_ptr: int,
    scaled_routing_ptr: int,
    tokens: int,
    num_experts: int,
    top_k: int,
    routed_scaling_factor: float,
    *,
    threads: int = _THREADS,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply stable sigmoid, correction-only top-k, and unbiased normalization.

    The FP32 correction bias is used only for selection. ``routing_ptr`` is
    gathered from the uncorrected sigmoid scores and sum-normalized;
    ``scaled_routing_ptr`` additionally applies the model's routed scale.
    """

    _check_shape(tokens, num_experts, top_k, routed_scaling_factor, threads)
    library = library or build_laguna_router(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL, _ARGTYPES, ctypes.c_int)
    err = fn(
        logits_ptr,
        correction_bias_ptr,
        routing_scores_ptr,
        selection_scores_ptr,
        selected_ptr,
        routing_ptr,
        scaled_routing_ptr,
        tokens,
        num_experts,
        top_k,
        float(routed_scaling_factor),
        threads,
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def laguna_router_topk_bf16_hidden_correction_bias_persistent_wave_top10(
    hidden_ptr: int,
    weight_ptr: int,
    correction_bias_ptr: int,
    logits_ptr: int,
    routing_scores_ptr: int,
    selection_scores_ptr: int,
    selected_ptr: int,
    routing_ptr: int,
    scaled_routing_ptr: int,
    completion_counter_ptr: int,
    tokens: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    routed_scaling_factor: float,
    *,
    threads: int = _THREADS,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact c=1 routing with wave-local top-10 and a self-resetting counter."""

    if int(tokens) != 1:
        raise ValueError("persistent Laguna router is decode-only and requires tokens == 1")
    if int(hidden_size) <= 0:
        raise ValueError("hidden_size must be positive")
    if int(hidden_size) > _MAX_HIDDEN_SIZE:
        raise ValueError(f"hidden_size must be <= {_MAX_HIDDEN_SIZE}")
    if int(num_experts) != _MAX_EXPERTS:
        raise ValueError(f"persistent wave-top10 router requires num_experts == {_MAX_EXPERTS}")
    if int(top_k) != 10:
        raise ValueError("persistent wave-top10 router requires top_k == 10")
    _check_shape(tokens, num_experts, top_k, routed_scaling_factor, threads)
    if int(completion_counter_ptr) == 0:
        raise ValueError("completion_counter_ptr must be nonzero")
    library = library or build_laguna_router(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        _PERSISTENT_WAVE_TOP10_SYMBOL,
        _PERSISTENT_ARGTYPES,
        ctypes.c_int,
    )
    err = fn(
        hidden_ptr,
        weight_ptr,
        correction_bias_ptr,
        logits_ptr,
        routing_scores_ptr,
        selection_scores_ptr,
        selected_ptr,
        routing_ptr,
        scaled_routing_ptr,
        completion_counter_ptr,
        int(tokens),
        int(hidden_size),
        int(num_experts),
        int(top_k),
        float(routed_scaling_factor),
        int(threads),
        int(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def laguna_weighted_sum_rows_bf16_f32w(
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
    """Reduce contiguous selected-expert lanes independently for every token."""

    if int(tokens) <= 0 or int(top_k) <= 0 or int(top_k) > _MAX_TOP_K:
        raise ValueError("tokens must be positive and top_k must be within [1, 16]")
    if int(features) <= 0:
        raise ValueError("features must be positive")
    if int(threads) not in {64, 128, 256}:
        raise ValueError("threads must be one of 64, 128, 256")
    library = library or build_laguna_router(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        _WEIGHTED_SUM_ROWS_SYMBOL,
        _WEIGHTED_SUM_ROWS_ARGTYPES,
        ctypes.c_int,
    )
    err = fn(
        values_ptr,
        weights_ptr,
        out_ptr,
        int(tokens),
        int(top_k),
        int(features),
        int(threads),
        int(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_laguna_router_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey(
            "hip_gfx1100",
            "laguna_sigmoid_router_topk",
            "f32",
            "correction_bias",
        ),
        laguna_sigmoid_correction_topk_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "laguna_router_topk",
            "f32",
            "bf16_hidden_correction_bias_persistent_wave_top10",
        ),
        laguna_router_topk_bf16_hidden_correction_bias_persistent_wave_top10,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "weighted_sum", "bf16", "laguna_rows"),
        laguna_weighted_sum_rows_bf16_f32w,
        replace=replace,
    )


def _check_shape(
    tokens: int,
    num_experts: int,
    top_k: int,
    routed_scaling_factor: float,
    threads: int,
) -> None:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if num_experts > _MAX_EXPERTS:
        raise ValueError(f"num_experts must be <= {_MAX_EXPERTS}")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_k > _MAX_TOP_K:
        raise ValueError(f"top_k must be <= {_MAX_TOP_K}")
    if top_k > num_experts:
        raise ValueError("top_k must be <= num_experts")
    scale = float(routed_scaling_factor)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("routed_scaling_factor must be finite and positive")
    if threads != _THREADS:
        raise ValueError(f"Laguna router requires {_THREADS} threads")


register_laguna_router_kernels()


__all__ = [
    "build_laguna_router",
    "laguna_router_topk_bf16_hidden_correction_bias_persistent_wave_top10",
    "laguna_sigmoid_correction_topk_f32",
    "laguna_weighted_sum_rows_bf16_f32w",
    "plan_laguna_router_build",
    "register_laguna_router_kernels",
]
