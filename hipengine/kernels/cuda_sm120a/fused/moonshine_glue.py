"""Raw-pointer Moonshine FP16 glue for CUDA ``sm_120a``."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_cuda, plan_cuda_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.cuda import CUDA_SUCCESS, CudaRuntime, get_cuda_runtime
from hipengine.kernels.backends import cuda_target_arch_for_backend
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("moonshine_glue.cu")
_OUTPUT_NAME = "moonshine_glue.so"
_BACKEND = "cuda_sm120a"
_TARGET_ARCH = cuda_target_arch_for_backend(_BACKEND)
_ALLOWED_THREADS = {32, 64, 128, 256}
_ARGMAX_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_EMBEDDING_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_RESIDUAL_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ROPE_ARGS = (
    *(ctypes.c_void_p for _ in range(7)),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_CACHE_ARGS = (
    *(ctypes.c_void_p for _ in range(5)),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_FUSED_ARGS = (
    *(ctypes.c_void_p for _ in range(10)),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)


def plan_moonshine_glue_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_cuda_build(
        sources=[_SOURCE],
        family="cuda_sm120a_moonshine_glue",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        output_name=_OUTPUT_NAME,
    )


def build_moonshine_glue(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_cuda(
        sources=[_SOURCE],
        family="cuda_sm120a_moonshine_glue",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _validate_threads(threads: int) -> None:
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 32, 64, 128, 256")


def _validate_rope(
    heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
    threads: int,
) -> None:
    if heads <= 0:
        raise ValueError("heads must be positive")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be positive, even, and <= head_dim")
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    _validate_threads(threads)


def _launch(
    library: ctypes.CDLL,
    symbol: str,
    argtypes,
    arguments: tuple[object, ...],
    runtime: CudaRuntime,
) -> None:
    fn = signed_kernel_fn(library, symbol, argtypes, ctypes.c_int)
    error = fn(*arguments)
    if int(error) != CUDA_SUCCESS:
        runtime.check(int(error))


def moonshine_argmax_fp16(
    logits_ptr: int,
    output_ptr: int,
    vocab_size: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    _validate_threads(threads)
    library = library or build_moonshine_glue(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_argmax_fp16",
        _ARGMAX_ARGS,
        (logits_ptr, output_ptr, vocab_size, threads, stream),
        runtime,
    )


def moonshine_embedding_lookup_fp16(
    embedding_ptr: int,
    token_ptr: int,
    output_ptr: int,
    hidden_size: int,
    vocab_size: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    _validate_threads(threads)
    library = library or build_moonshine_glue(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_embedding_lookup_fp16",
        _EMBEDDING_ARGS,
        (
            embedding_ptr,
            token_ptr,
            output_ptr,
            hidden_size,
            vocab_size,
            threads,
            stream,
        ),
        runtime,
    )


def moonshine_residual_fp16(
    hidden_ptr: int,
    residual_ptr: int,
    output_ptr: int,
    elements: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    if elements <= 0:
        raise ValueError("elements must be positive")
    _validate_threads(threads)
    library = library or build_moonshine_glue(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_residual_fp16",
        _RESIDUAL_ARGS,
        (hidden_ptr, residual_ptr, output_ptr, elements, threads, stream),
        runtime,
    )


def moonshine_partial_rope_fp16(
    query_ptr: int,
    key_ptr: int,
    cos_ptr: int,
    sin_ptr: int,
    position_ptr: int,
    query_output_ptr: int,
    key_output_ptr: int,
    heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    _validate_rope(heads, head_dim, rotary_dim, max_positions, threads)
    library = library or build_moonshine_glue(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_partial_rope_fp16",
        _ROPE_ARGS,
        (
            query_ptr,
            key_ptr,
            cos_ptr,
            sin_ptr,
            position_ptr,
            query_output_ptr,
            key_output_ptr,
            heads,
            head_dim,
            rotary_dim,
            max_positions,
            threads,
            stream,
        ),
        runtime,
    )


def moonshine_self_cache_append_fp16(
    key_ptr: int,
    value_ptr: int,
    position_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    heads: int,
    head_dim: int,
    capacity: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    if heads <= 0:
        raise ValueError("heads must be positive")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    _validate_threads(threads)
    library = library or build_moonshine_glue(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_self_cache_append_fp16",
        _CACHE_ARGS,
        (
            key_ptr,
            value_ptr,
            position_ptr,
            key_cache_ptr,
            value_cache_ptr,
            heads,
            head_dim,
            capacity,
            threads,
            stream,
        ),
        runtime,
    )


def moonshine_partial_rope_cache_append_fp16(
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    cos_ptr: int,
    sin_ptr: int,
    position_ptr: int,
    query_output_ptr: int,
    key_output_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    heads: int,
    head_dim: int,
    rotary_dim: int,
    capacity: int,
    max_positions: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    _validate_rope(heads, head_dim, rotary_dim, max_positions, threads)
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    library = library or build_moonshine_glue(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_partial_rope_cache_append_fp16",
        _FUSED_ARGS,
        (
            query_ptr,
            key_ptr,
            value_ptr,
            cos_ptr,
            sin_ptr,
            position_ptr,
            query_output_ptr,
            key_output_ptr,
            key_cache_ptr,
            value_cache_ptr,
            heads,
            head_dim,
            rotary_dim,
            capacity,
            max_positions,
            threads,
            stream,
        ),
        runtime,
    )


def register_moonshine_glue_kernels(*, replace: bool = True) -> None:
    registrations = (
        (
            KernelKey(_BACKEND, "moonshine_argmax", "fp16", "lowest_id"),
            moonshine_argmax_fp16,
        ),
        (
            KernelKey(_BACKEND, "moonshine_embedding", "fp16", "lookup_i64"),
            moonshine_embedding_lookup_fp16,
        ),
        (
            KernelKey(_BACKEND, "moonshine_residual", "fp16", "rounded"),
            moonshine_residual_fp16,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_partial_rope",
                "fp16",
                "interleaved",
            ),
            moonshine_partial_rope_fp16,
        ),
        (
            KernelKey(_BACKEND, "moonshine_self_cache", "fp16", "fixed"),
            moonshine_self_cache_append_fp16,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_partial_rope+moonshine_self_cache",
                "fp16",
                "interleaved_fixed_append",
            ),
            moonshine_partial_rope_cache_append_fp16,
        ),
    )
    for key, kernel in registrations:
        register(key, kernel, replace=replace)


register_moonshine_glue_kernels()

__all__ = [
    "build_moonshine_glue",
    "moonshine_argmax_fp16",
    "moonshine_embedding_lookup_fp16",
    "moonshine_partial_rope_cache_append_fp16",
    "moonshine_partial_rope_fp16",
    "moonshine_residual_fp16",
    "moonshine_self_cache_append_fp16",
    "plan_moonshine_glue_build",
    "register_moonshine_glue_kernels",
]
