"""Native host-call batching for unchanged Laguna decode kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("laguna_launch_batch.hip")
_OUTPUT_NAME = "laguna_launch_batch.so"
_Q4_SYMBOL = "hipengine_laguna_q4_shared_down_tail_batch"
_Q6_SYMBOL = "hipengine_laguna_q6_shared_down_tail_batch"
_Q4_PROJECTION_SYMBOL = "hipengine_gguf_q4_k_pack8_gemv_bf16_bf16_out"
_Q6_PROJECTION_SYMBOL = "hipengine_gguf_q6_k_pack8_gemv_decode_bf16_bf16_out"
_TAIL_SYMBOL = "hipengine_laguna_aggregate_moe_tail_next_rmsnorm_gguf_bf16_out"


def plan_laguna_launch_batch_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="laguna_launch_batch",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_laguna_launch_batch(
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
        family="laguna_launch_batch",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _function_pointer(function) -> ctypes.c_void_p:
    return ctypes.cast(function, ctypes.c_void_p)


def laguna_q4_shared_down_tail_batch(
    projection_function,
    tail_function,
    x_ptr: int,
    qweight_ptr: int,
    scales_ptr: int,
    mins_ptr: int,
    shared_out_ptr: int,
    routed_ptr: int,
    post_attention_ptr: int,
    norm_weight_ptr: int,
    norm_out_ptr: int,
    hidden_out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 32,
    eps: float = 1e-6,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Enqueue unchanged Q4 shared-down and D9 wrappers inside one host call."""

    if rows != 1:
        raise ValueError("Laguna shared-down launch batching requires rows == 1")
    if in_features <= 0 or out_features <= 0:
        raise ValueError("Laguna shared-down launch dimensions must be positive")
    if threads not in {32, 64, 128}:
        raise ValueError("Q4 shared-down threads must be one of 32, 64, 128")
    library = library or build_laguna_launch_batch(load=True)
    runtime = runtime or get_hip_runtime()
    function = getattr(library, _Q4_SYMBOL)
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        *([ctypes.c_void_p] * 10),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    error = function(
        _function_pointer(projection_function),
        _function_pointer(tail_function),
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(scales_ptr),
        ctypes.c_void_p(mins_ptr),
        ctypes.c_void_p(shared_out_ptr),
        ctypes.c_void_p(routed_ptr),
        ctypes.c_void_p(post_attention_ptr),
        ctypes.c_void_p(norm_weight_ptr),
        ctypes.c_void_p(norm_out_ptr),
        ctypes.c_void_p(hidden_out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(threads),
        ctypes.c_float(eps),
        ctypes.c_void_p(stream),
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def laguna_q6_shared_down_tail_batch(
    projection_function,
    tail_function,
    x_ptr: int,
    qweight_ptr: int,
    shared_out_ptr: int,
    routed_ptr: int,
    post_attention_ptr: int,
    norm_weight_ptr: int,
    norm_out_ptr: int,
    hidden_out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    eps: float = 1e-6,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Enqueue unchanged Q6 shared-down and D9 wrappers inside one host call."""

    if rows != 1:
        raise ValueError("Laguna shared-down launch batching requires rows == 1")
    if in_features <= 0 or out_features <= 0:
        raise ValueError("Laguna shared-down launch dimensions must be positive")
    library = library or build_laguna_launch_batch(load=True)
    runtime = runtime or get_hip_runtime()
    function = getattr(library, _Q6_SYMBOL)
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        *([ctypes.c_void_p] * 8),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    error = function(
        _function_pointer(projection_function),
        _function_pointer(tail_function),
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(shared_out_ptr),
        ctypes.c_void_p(routed_ptr),
        ctypes.c_void_p(post_attention_ptr),
        ctypes.c_void_p(norm_weight_ptr),
        ctypes.c_void_p(norm_out_ptr),
        ctypes.c_void_p(hidden_out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_float(eps),
        ctypes.c_void_p(stream),
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


laguna_q4_shared_down_tail_batch.projection_symbol = _Q4_PROJECTION_SYMBOL
laguna_q4_shared_down_tail_batch.tail_symbol = _TAIL_SYMBOL
laguna_q6_shared_down_tail_batch.projection_symbol = _Q6_PROJECTION_SYMBOL
laguna_q6_shared_down_tail_batch.tail_symbol = _TAIL_SYMBOL


def register_laguna_launch_batch_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey(
            "hip_gfx1100",
            "linear+moe_tail+next_rmsnorm_host_batch",
            "gguf_q4_k",
            "pack8_bf16_bf16_out",
        ),
        laguna_q4_shared_down_tail_batch,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear+moe_tail+next_rmsnorm_host_batch",
            "gguf_q6_k",
            "pack8_gemv_decode_bf16_bf16_out",
        ),
        laguna_q6_shared_down_tail_batch,
        replace=replace,
    )


register_laguna_launch_batch_kernels()


__all__ = [
    "build_laguna_launch_batch",
    "laguna_q4_shared_down_tail_batch",
    "laguna_q6_shared_down_tail_batch",
    "plan_laguna_launch_batch_build",
    "register_laguna_launch_batch_kernels",
]
