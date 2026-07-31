"""Raw-pointer Moonshine logical-dimension FP16 attention fallbacks."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("moonshine_attention.hip")
_OUTPUT_NAME = "moonshine_attention.so"
_HEADS = 8
_HEAD_DIM = 52
_THREADS = 32
_ATTENTION_ARGS = (
    *(ctypes.c_void_p for _ in range(5)),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_int64,
    ctypes.c_void_p,
)


def plan_moonshine_attention_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="moonshine_attention",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_moonshine_attention(
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
        family="moonshine_attention",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _validate_shape(heads: int, head_dim: int, threads: int) -> None:
    if heads != _HEADS:
        raise ValueError(f"heads must equal the Moonshine contract value {_HEADS}")
    if head_dim != _HEAD_DIM:
        raise ValueError(f"head_dim must equal the logical Moonshine dimension {_HEAD_DIM}")
    if threads != _THREADS:
        raise ValueError("threads must be 32 for the one-wave-per-head fallback")


def _scale(head_dim: int, scale: float | None) -> float:
    value = head_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("scale must be positive and finite")
    return value


def _launch(
    library: ctypes.CDLL,
    symbol: str,
    arguments: tuple[object, ...],
    runtime: HipRuntime,
) -> None:
    function = signed_kernel_fn(library, symbol, _ATTENTION_ARGS, ctypes.c_int)
    error = function(*arguments)
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def moonshine_self_attention_fp16(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    position_ptr: int,
    output_ptr: int,
    heads: int,
    head_dim: int,
    capacity: int,
    *,
    scale: float | None = None,
    threads: int = _THREADS,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Attend through the current device position in a fixed FP16 self cache."""

    _validate_shape(heads, head_dim, threads)
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    scale_value = _scale(head_dim, scale)
    library = library or build_moonshine_attention(load=True)
    runtime = runtime or get_hip_runtime()
    _launch(
        library,
        "hipengine_moonshine_self_attention_fp16",
        (
            query_ptr,
            key_cache_ptr,
            value_cache_ptr,
            position_ptr,
            output_ptr,
            heads,
            head_dim,
            capacity,
            scale_value,
            threads,
            stream,
        ),
        runtime,
    )


def moonshine_cross_attention_fp16(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    mask_ptr: int,
    output_ptr: int,
    heads: int,
    head_dim: int,
    encoder_length: int,
    *,
    scale: float | None = None,
    threads: int = _THREADS,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Attend over resident encoder K/V while applying its int32 visibility mask."""

    _validate_shape(heads, head_dim, threads)
    _launch_cross_attention(
        "hipengine_moonshine_cross_attention_fp16",
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        mask_ptr,
        output_ptr,
        heads,
        head_dim,
        encoder_length,
        scale=scale,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def moonshine_cross_attention_grouped_fp16(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    mask_ptr: int,
    output_ptr: int,
    heads: int,
    head_dim: int,
    encoder_length: int,
    *,
    scale: float | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run the exact fallback math with all eight head waves in one workgroup."""

    _validate_shape(heads, head_dim, _THREADS)
    _launch_cross_attention(
        "hipengine_moonshine_cross_attention_grouped_fp16",
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        mask_ptr,
        output_ptr,
        heads,
        head_dim,
        encoder_length,
        scale=scale,
        threads=heads * _THREADS,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def moonshine_cross_attention_parallel_fp16(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    mask_ptr: int,
    output_ptr: int,
    heads: int,
    head_dim: int,
    encoder_length: int,
    *,
    scale: float | None = None,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Partition masked tokens across 2/4/8 waves per head and merge in LDS."""

    _validate_shape(heads, head_dim, _THREADS)
    if threads not in (64, 128, 256):
        raise ValueError("threads must be one of 64, 128, or 256")
    _launch_cross_attention(
        "hipengine_moonshine_cross_attention_parallel_fp16",
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        mask_ptr,
        output_ptr,
        heads,
        head_dim,
        encoder_length,
        scale=scale,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _launch_cross_attention(
    symbol: str,
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    mask_ptr: int,
    output_ptr: int,
    heads: int,
    head_dim: int,
    encoder_length: int,
    *,
    scale: float | None,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    if encoder_length <= 0:
        raise ValueError("encoder_length must be positive")
    scale_value = _scale(head_dim, scale)
    library = library or build_moonshine_attention(load=True)
    runtime = runtime or get_hip_runtime()
    _launch(
        library,
        symbol,
        (
            query_ptr,
            key_cache_ptr,
            value_cache_ptr,
            mask_ptr,
            output_ptr,
            heads,
            head_dim,
            encoder_length,
            scale_value,
            threads,
            stream,
        ),
        runtime,
    )


def register_moonshine_attention_kernels(*, replace: bool = True) -> None:
    registrations = (
        (
            KernelKey(
                "hip_gfx1100",
                "moonshine_self_attention",
                "fp16",
                "fixed_cache_logical_dim",
            ),
            moonshine_self_attention_fp16,
        ),
        (
            KernelKey(
                "hip_gfx1100",
                "moonshine_cross_attention",
                "fp16",
                "resident_masked_logical_dim",
            ),
            moonshine_cross_attention_fp16,
        ),
        (
            KernelKey(
                "hip_gfx1100",
                "moonshine_cross_attention",
                "fp16",
                "resident_masked_grouped_heads",
            ),
            moonshine_cross_attention_grouped_fp16,
        ),
        (
            KernelKey(
                "hip_gfx1100",
                "moonshine_cross_attention",
                "fp16",
                "resident_masked_parallel_tokens",
            ),
            moonshine_cross_attention_parallel_fp16,
        ),
    )
    for key, kernel in registrations:
        register(key, kernel, replace=replace)


register_moonshine_attention_kernels()

__all__ = [
    "build_moonshine_attention",
    "moonshine_cross_attention_fp16",
    "moonshine_cross_attention_grouped_fp16",
    "moonshine_cross_attention_parallel_fp16",
    "moonshine_self_attention_fp16",
    "plan_moonshine_attention_build",
    "register_moonshine_attention_kernels",
]
