"""Raw-pointer wrappers for Qwen3.5 paged KV write kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.dtype import DType
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register
from hipengine.kvcache import KVLiveSpans

_SOURCE = Path(__file__).with_name("paged_kv_write.hip")
_OUTPUT_NAME = "qwen35_paged_kv_write.so"
_SYMBOL_MIXED_BF16 = "hipengine_qwen35_write_paged_kv_mixed_value_bf16_spans"
_SYMBOL_F32 = "hipengine_qwen35_write_paged_kv_f32_spans"


def plan_qwen35_paged_kv_write_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="qwen35_paged_kv_write",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_qwen35_paged_kv_write(
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
        family="qwen35_paged_kv_write",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def qwen35_write_paged_kv_mixed_value_bf16_spans(
    key_ptr: int,
    value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Append FP32 K + BF16 V to paged BF16 KV cache using ``KVLiveSpans``.

    For the fixed-page parent bridge, ``spans.base_offsets`` carries the int32
    physical block table and ``spans.live_counts`` carries the int64 decode
    position tensor. Callers never pass a naked block table to dispatch.
    """

    _launch_write(
        _SYMBOL_MIXED_BF16,
        key_ptr,
        value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        spans,
        block_size,
        num_kv_heads,
        head_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_write_paged_kv_f32_spans(
    key_ptr: int,
    value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Append FP32 K/V to paged BF16 KV cache using ``KVLiveSpans``."""

    _launch_write(
        _SYMBOL_F32,
        key_ptr,
        value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        spans,
        block_size,
        num_kv_heads,
        head_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def register_qwen35_paged_kv_write_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "paged_kv_write", "w4_paro", "mixed_bf16_spans"),
        qwen35_write_paged_kv_mixed_value_bf16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_kv_write", "w4_paro", "f32_spans"),
        qwen35_write_paged_kv_f32_spans,
        replace=replace,
    )


def _launch_write(
    symbol: str,
    key_ptr: int,
    value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_write_shape(spans, block_size, num_kv_heads, head_dim)
    library = library or build_qwen35_paged_kv_write(load=True)
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
        ctypes.c_void_p(key_ptr),
        ctypes.c_void_p(value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_int64(block_size),
        ctypes.c_int64(_block_table_len(spans)),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _check_write_shape(
    spans: KVLiveSpans,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    if spans.spans_mode != "uniform":
        raise ValueError("paged KV write currently requires uniform spans")
    if spans.storage_dtype != DType.BF16:
        raise ValueError("paged KV write currently requires bf16 storage spans")
    if spans.live_counts.dtype != DType.INT64:
        raise ValueError("paged KV write parent bridge requires int64 live_counts")
    _check_positive(_block_table_len(spans), "block_table_len")
    _check_positive(block_size, "block_size")
    _check_positive(num_kv_heads, "num_kv_heads")
    _check_positive(head_dim, "head_dim")
    if spans.max_live_count >= block_size * _block_table_len(spans):
        raise ValueError("max_live_count must fit within the paged span block table")


def _block_table_len(spans: KVLiveSpans) -> int:
    return spans.base_offsets.numel


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_qwen35_paged_kv_write_kernels()
