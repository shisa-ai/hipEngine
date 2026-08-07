"""Raw-pointer Moonshine logical-dimension FP16 attention kernels for ``sm_120a``."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_cuda, plan_cuda_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.cuda import CUDA_SUCCESS, CudaRuntime, get_cuda_runtime
from hipengine.kernels.backends import cuda_target_arch_for_backend
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("moonshine_attention.cu")
_OUTPUT_NAME = "moonshine_attention.so"
_BACKEND = "cuda_sm120a"
_TARGET_ARCH = cuda_target_arch_for_backend(_BACKEND)
_HEADS = 8
_HEAD_DIM = 52
_WAVE_THREADS = 32
_PARALLEL_THREADS = (64, 128, 256)
# Measured batch-timed CUDA-event medians on exclusive GPU0 (RTX PRO 6000
# Blackwell). Cross attention: parallel t256 is best at every production
# encoder length (40/207/1248), so the cross default is flat t256. Self
# attention: one-wave t32 and parallel t256 are tied below ~8 visible tokens
# and parallel t256 wins 1.6x+ from ~16 upward, so the self default is t32 for
# visible < 8 and parallel t256 otherwise. These are general cache-position
# buckets; explicit ``threads=``/variant always overrides at composition.
_SELF_SHORT_THRESHOLD = 8
_SELF_SHORT_THREADS = _WAVE_THREADS
_SELF_LONG_THREADS = 256
_CROSS_THREADS = 256


def _default_self_threads(visible_length: int) -> int:
    """Measured thread/variant schedule for the self-attention visible prefix."""
    if visible_length < _SELF_SHORT_THRESHOLD:
        return _SELF_SHORT_THREADS
    return _SELF_LONG_THREADS


def _default_cross_threads(encoder_length: int) -> int:
    """Measured thread/variant schedule for masked cross attention."""
    return _CROSS_THREADS
_ATTENTION_ARGS = (
    *(ctypes.c_void_p for _ in range(5)),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_int64,
    ctypes.c_void_p,
)

# Batched (static B) variants add an int64 ``batch`` between threads and stream:
# (query, key, value, pos_or_mask, output, heads, head_dim, length, scale,
#  threads, batch, stream).
_BATCH_ATTENTION_ARGS = (
    *(ctypes.c_void_p for _ in range(5)),
    ctypes.c_int64,  # heads
    ctypes.c_int64,  # head_dim
    ctypes.c_int64,  # length (capacity or encoder_length)
    ctypes.c_float,  # scale
    ctypes.c_int64,  # threads
    ctypes.c_int64,  # batch
    ctypes.c_void_p,  # stream
)


def plan_moonshine_attention_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_cuda_build(
        sources=[_SOURCE],
        family="cuda_sm120a_moonshine_attention",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
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
    return build_cuda(
        sources=[_SOURCE],
        family="cuda_sm120a_moonshine_attention",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _validate_shape(heads: int, head_dim: int, length: int) -> None:
    if heads != _HEADS:
        raise ValueError(f"heads must equal the Moonshine contract value {_HEADS}")
    if head_dim != _HEAD_DIM:
        raise ValueError(
            f"head_dim must equal the logical Moonshine dimension {_HEAD_DIM}"
        )
    if length <= 0:
        raise ValueError("length must be positive")


def _scale(head_dim: int, scale: float | None) -> float:
    value = head_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("scale must be positive and finite")
    return value


def _launch(
    library: ctypes.CDLL,
    symbol: str,
    arguments: tuple[object, ...],
    runtime: CudaRuntime,
) -> None:
    function = signed_kernel_fn(library, symbol, _ATTENTION_ARGS, ctypes.c_int)
    error = function(*arguments)
    if int(error) != CUDA_SUCCESS:
        runtime.check(int(error))


def _launch_batch(
    library: ctypes.CDLL,
    symbol: str,
    arguments: tuple[object, ...],
    runtime: CudaRuntime,
) -> None:
    function = signed_kernel_fn(library, symbol, _BATCH_ATTENTION_ARGS, ctypes.c_int)
    error = function(*arguments)
    if int(error) != CUDA_SUCCESS:
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
    threads: int = _WAVE_THREADS,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Attend through the current device position in a fixed FP16 self cache.

    One wave per head; online FP32 softmax over the visible prefix. A non-32
    ``threads`` is rejected (use the parallel variant for multi-wave).
    """

    _validate_shape(heads, head_dim, capacity)
    if threads != _WAVE_THREADS:
        raise ValueError(f"threads must be {_WAVE_THREADS} for one-wave-per-head")
    scale_value = _scale(head_dim, scale)
    library = library or build_moonshine_attention(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_self_attention_fp16",
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


def moonshine_self_attention_parallel_fp16(
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
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Self attention partitioning the visible prefix across 2/4/8 waves/head."""

    _validate_shape(heads, head_dim, capacity)
    if threads not in _PARALLEL_THREADS:
        raise ValueError(f"threads must be one of {_PARALLEL_THREADS}")
    scale_value = _scale(head_dim, scale)
    library = library or build_moonshine_attention(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_self_attention_parallel_fp16",
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
    threads: int = _WAVE_THREADS,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Attend one query over resident encoder K/V with an FP16/INT32 mask.

    One wave per head; masked online FP32 softmax.
    """

    _validate_shape(heads, head_dim, encoder_length)
    if threads != _WAVE_THREADS:
        raise ValueError(f"threads must be {_WAVE_THREADS} for one-wave-per-head")
    scale_value = _scale(head_dim, scale)
    library = library or build_moonshine_attention(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_cross_attention_fp16",
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
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Cross attention with all heads in one 256-thread block (8 waves)."""

    _validate_shape(heads, head_dim, encoder_length)
    if threads != 256:
        raise ValueError("threads must be 256 for the grouped-heads variant")
    scale_value = _scale(head_dim, scale)
    library = library or build_moonshine_attention(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_cross_attention_grouped_fp16",
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
    runtime: CudaRuntime | None = None,
) -> None:
    """Cross attention partitioning the masked encoder frames across waves."""

    _validate_shape(heads, head_dim, encoder_length)
    if threads not in _PARALLEL_THREADS:
        raise ValueError(f"threads must be one of {_PARALLEL_THREADS}")
    scale_value = _scale(head_dim, scale)
    library = library or build_moonshine_attention(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_cross_attention_parallel_fp16",
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


def moonshine_cross_attention_parallel_batch_fp16(
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
    batch: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Batched masked cross attention: one launch for ``batch`` rows.

    Layouts: query/output ``[B, heads, head_dim]``, cache
    ``[B, heads, encoder_length, head_dim]``, mask ``[B, encoder_length]``.
    Each (row, head) block runs the identical FP32 arithmetic of the
    single-row parallel kernel, so outputs are bit-exact vs B sequential calls.
    """

    _validate_shape(heads, head_dim, encoder_length)
    if batch <= 0:
        raise ValueError("batch must be positive")
    if threads not in _PARALLEL_THREADS:
        raise ValueError(f"threads must be one of {_PARALLEL_THREADS}")
    scale_value = _scale(head_dim, scale)
    library = library or build_moonshine_attention(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch_batch(
        library,
        "hipengine_cuda_sm120a_moonshine_cross_attention_parallel_batch_fp16",
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
            batch,
            stream,
        ),
        runtime,
    )


def moonshine_cross_attention_batch_fp16(
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
    threads: int = 32,
    batch: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Batched one-wave-per-head masked cross attention (``threads`` must be 32)."""

    _validate_shape(heads, head_dim, encoder_length)
    if batch <= 0:
        raise ValueError("batch must be positive")
    if threads != _WAVE_THREADS:
        raise ValueError(f"threads must be {_WAVE_THREADS} for one-wave-per-head")
    scale_value = _scale(head_dim, scale)
    library = library or build_moonshine_attention(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch_batch(
        library,
        "hipengine_cuda_sm120a_moonshine_cross_attention_batch_fp16",
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
            batch,
            stream,
        ),
        runtime,
    )


def moonshine_self_attention_parallel_batch_fp16(
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
    threads: int = 256,
    batch: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Batched self attention over per-row positions (parallel variant).

    Layouts: query/output ``[B, heads, head_dim]``, cache
    ``[B, heads, capacity, head_dim]``, position ``[B]`` int64.  Each (row,
    head) block reproduces the single-row parallel kernel arithmetic, so the
    output is bit-exact vs B sequential single-row calls.
    """

    _validate_shape(heads, head_dim, capacity)
    if batch <= 0:
        raise ValueError("batch must be positive")
    if threads not in _PARALLEL_THREADS:
        raise ValueError(f"threads must be one of {_PARALLEL_THREADS}")
    scale_value = _scale(head_dim, scale)
    library = library or build_moonshine_attention(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch_batch(
        library,
        "hipengine_cuda_sm120a_moonshine_self_attention_parallel_batch_fp16",
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
            batch,
            stream,
        ),
        runtime,
    )


def moonshine_self_attention_batch_fp16(
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
    threads: int = 32,
    batch: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Batched one-wave-per-head self attention (``threads`` must be 32)."""

    _validate_shape(heads, head_dim, capacity)
    if batch <= 0:
        raise ValueError("batch must be positive")
    if threads != _WAVE_THREADS:
        raise ValueError(f"threads must be {_WAVE_THREADS} for one-wave-per-head")
    scale_value = _scale(head_dim, scale)
    library = library or build_moonshine_attention(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch_batch(
        library,
        "hipengine_cuda_sm120a_moonshine_self_attention_batch_fp16",
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
            batch,
            stream,
        ),
        runtime,
    )


def register_moonshine_attention_kernels(*, replace: bool = True) -> None:
    registrations = (
        (
            KernelKey(
                _BACKEND,
                "moonshine_self_attention",
                "fp16",
                "fixed_cache_logical_dim",
            ),
            moonshine_self_attention_fp16,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_self_attention",
                "fp16",
                "fixed_cache_parallel_tokens",
            ),
            moonshine_self_attention_parallel_fp16,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_cross_attention",
                "fp16",
                "resident_masked_logical_dim",
            ),
            moonshine_cross_attention_fp16,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_cross_attention",
                "fp16",
                "resident_masked_grouped_heads",
            ),
            moonshine_cross_attention_grouped_fp16,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_cross_attention",
                "fp16",
                "resident_masked_parallel_tokens",
            ),
            moonshine_cross_attention_parallel_fp16,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_self_attention",
                "fp16",
                "batch_fixed_cache_logical_dim",
            ),
            moonshine_self_attention_batch_fp16,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_self_attention",
                "fp16",
                "batch_parallel_tokens",
            ),
            moonshine_self_attention_parallel_batch_fp16,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_cross_attention",
                "fp16",
                "batch_masked_logical_dim",
            ),
            moonshine_cross_attention_batch_fp16,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_cross_attention",
                "fp16",
                "batch_masked_parallel_tokens",
            ),
            moonshine_cross_attention_parallel_batch_fp16,
        ),
    )
    for key, kernel in registrations:
        register(key, kernel, replace=replace)


register_moonshine_attention_kernels()

__all__ = [
    "build_moonshine_attention",
    "moonshine_cross_attention_batch_fp16",
    "moonshine_cross_attention_fp16",
    "moonshine_cross_attention_grouped_fp16",
    "moonshine_cross_attention_parallel_batch_fp16",
    "moonshine_cross_attention_parallel_fp16",
    "moonshine_self_attention_batch_fp16",
    "moonshine_self_attention_fp16",
    "moonshine_self_attention_parallel_batch_fp16",
    "moonshine_self_attention_parallel_fp16",
    "plan_moonshine_attention_build",
    "register_moonshine_attention_kernels",
]
