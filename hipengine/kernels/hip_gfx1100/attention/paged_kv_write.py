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
_SYMBOL_MIXED_BF16_BATCH = "hipengine_qwen35_write_paged_kv_mixed_value_bf16_batch_spans"
_SYMBOL_MIXED_FP16 = "hipengine_qwen35_write_paged_kv_mixed_value_fp16_spans"
_SYMBOL_MIXED_FP16_BATCH = "hipengine_qwen35_write_paged_kv_mixed_value_fp16_batch_spans"
_SYMBOL_MIXED_BF16_PROMPT = "hipengine_qwen35_write_paged_kv_mixed_value_bf16_prompt_spans"
_SYMBOL_MIXED_FP16_PROMPT = "hipengine_qwen35_write_paged_kv_mixed_value_fp16_prompt_spans"
_SYMBOL_F32 = "hipengine_qwen35_write_paged_kv_f32_spans"
_SYMBOL_INT8_SCALE_F32 = "hipengine_qwen35_write_paged_kv_int8_per_token_head_scale_f32_spans"
_SYMBOL_INT8_SCALE_F32_BATCH = "hipengine_qwen35_write_paged_kv_int8_per_token_head_scale_f32_batch_spans"
_SYMBOL_INT8_SCALE_F32_PROMPT = "hipengine_qwen35_write_paged_kv_int8_per_token_head_scale_f32_prompt_spans"
_SYMBOL_INT8_SCALE_FP16 = "hipengine_qwen35_write_paged_kv_int8_per_token_head_scale_fp16_spans"
_SYMBOL_INT8_SCALE_FP16_BATCH = "hipengine_qwen35_write_paged_kv_int8_per_token_head_scale_fp16_batch_spans"
_SYMBOL_INT8_SCALE_FP16_PROMPT = "hipengine_qwen35_write_paged_kv_int8_per_token_head_scale_fp16_prompt_spans"


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


def qwen35_write_paged_kv_mixed_value_bf16_batch_spans(
    key_ptr: int,
    value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Append batched FP32 K + BF16 V to row-major paged BF16 KV cache."""

    _launch_write_batch(
        _SYMBOL_MIXED_BF16_BATCH,
        key_ptr,
        value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        spans,
        rows,
        block_size,
        num_kv_heads,
        head_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_write_paged_kv_mixed_value_bf16_prompt_spans(
    key_ptr: int,
    value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Append BF16 prompt rows into one paged BF16 KV cache."""

    _launch_write_batch(
        _SYMBOL_MIXED_BF16_PROMPT,
        key_ptr,
        value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        spans,
        rows,
        block_size,
        num_kv_heads,
        head_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_write_paged_kv_mixed_value_fp16_spans(
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
    """Append FP32 K + FP16 V to paged BF16 KV cache using ``KVLiveSpans``."""

    _launch_write(
        _SYMBOL_MIXED_FP16,
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


def qwen35_write_paged_kv_mixed_value_fp16_batch_spans(
    key_ptr: int,
    value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Append batched FP32 K + FP16 V to row-major paged BF16 KV cache."""

    _launch_write_batch(
        _SYMBOL_MIXED_FP16_BATCH,
        key_ptr,
        value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        spans,
        rows,
        block_size,
        num_kv_heads,
        head_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_write_paged_kv_mixed_value_fp16_prompt_spans(
    key_ptr: int,
    value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Append prompt rows into one paged BF16 KV cache."""

    _launch_write_batch(
        _SYMBOL_MIXED_FP16_PROMPT,
        key_ptr,
        value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        spans,
        rows,
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


def qwen35_write_paged_kv_int8_per_token_head_spans(
    key_ptr: int,
    value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    k_scale_ptr: int,
    v_scale_ptr: int,
    spans: KVLiveSpans,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Append one FP32 K/V row to INT8 KV cache with per-token/head scales."""

    _launch_int8_write(
        _int8_symbol(spans, batch=False, prompt=False),
        key_ptr,
        value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        k_scale_ptr,
        v_scale_ptr,
        spans,
        block_size,
        num_kv_heads,
        head_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_write_paged_kv_int8_per_token_head_batch_spans(
    key_ptr: int,
    value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    k_scale_ptr: int,
    v_scale_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Append batched FP32 K/V rows into row-major INT8 KV cache arenas."""

    _launch_int8_write_batch(
        _int8_symbol(spans, batch=True, prompt=False),
        key_ptr,
        value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        k_scale_ptr,
        v_scale_ptr,
        spans,
        rows,
        block_size,
        num_kv_heads,
        head_dim,
        row_major_cache=True,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_write_paged_kv_int8_per_token_head_prompt_spans(
    key_ptr: int,
    value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    k_scale_ptr: int,
    v_scale_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Append prompt FP32 K/V rows into one shared INT8 KV cache arena."""

    _launch_int8_write_batch(
        _int8_symbol(spans, batch=True, prompt=True),
        key_ptr,
        value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        k_scale_ptr,
        v_scale_ptr,
        spans,
        rows,
        block_size,
        num_kv_heads,
        head_dim,
        row_major_cache=False,
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
        KernelKey("hip_gfx1100", "paged_kv_write", "w4_paro", "mixed_bf16_batch_spans"),
        qwen35_write_paged_kv_mixed_value_bf16_batch_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_kv_write", "w4_paro", "mixed_bf16_prompt_spans"),
        qwen35_write_paged_kv_mixed_value_bf16_prompt_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_kv_write", "gguf_qwen35", "mixed_bf16_prompt_spans"),
        qwen35_write_paged_kv_mixed_value_bf16_prompt_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_kv_write", "w4_paro", "mixed_fp16_spans"),
        qwen35_write_paged_kv_mixed_value_fp16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_kv_write", "w4_paro", "mixed_fp16_batch_spans"),
        qwen35_write_paged_kv_mixed_value_fp16_batch_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_kv_write", "w4_paro", "mixed_fp16_prompt_spans"),
        qwen35_write_paged_kv_mixed_value_fp16_prompt_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_kv_write", "w4_paro", "f32_spans"),
        qwen35_write_paged_kv_f32_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_kv_write", "int8_per_token_head", "per_token_head_spans"),
        qwen35_write_paged_kv_int8_per_token_head_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_kv_write", "int8_per_token_head", "per_token_head_prompt_spans"),
        qwen35_write_paged_kv_int8_per_token_head_prompt_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_kv_write", "int8_per_token_head", "per_token_head_batch_spans"),
        qwen35_write_paged_kv_int8_per_token_head_batch_spans,
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


def _launch_write_batch(
    symbol: str,
    key_ptr: int,
    value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    block_table_len = _check_write_batch_shape(spans, rows, block_size, num_kv_heads, head_dim)
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
        ctypes.c_int64(rows),
        ctypes.c_int64(block_size),
        ctypes.c_int64(block_table_len),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_int8_write(
    symbol: str,
    key_ptr: int,
    value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    k_scale_ptr: int,
    v_scale_ptr: int,
    spans: KVLiveSpans,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    block_table_len = _check_int8_write_shape(
        spans,
        block_size,
        num_kv_heads,
        head_dim,
        k_scale_ptr=k_scale_ptr,
        v_scale_ptr=v_scale_ptr,
    )
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
        ctypes.c_void_p(k_scale_ptr),
        ctypes.c_void_p(v_scale_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_int64(block_size),
        ctypes.c_int64(block_table_len),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_int8_write_batch(
    symbol: str,
    key_ptr: int,
    value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    k_scale_ptr: int,
    v_scale_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    row_major_cache: bool,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    block_table_len = _check_int8_write_batch_shape(
        spans,
        rows,
        block_size,
        num_kv_heads,
        head_dim,
        row_major_cache=row_major_cache,
        k_scale_ptr=k_scale_ptr,
        v_scale_ptr=v_scale_ptr,
    )
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
        ctypes.c_void_p(key_ptr),
        ctypes.c_void_p(value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(k_scale_ptr),
        ctypes.c_void_p(v_scale_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(block_size),
        ctypes.c_int64(block_table_len),
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


def _check_write_batch_shape(
    spans: KVLiveSpans,
    rows: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    _check_write_shape(spans, block_size, num_kv_heads, head_dim)
    _check_positive(rows, "rows")
    if spans.live_counts.numel < rows:
        raise ValueError("live_counts must have at least rows entries")
    if spans.base_offsets.numel % rows != 0:
        raise ValueError("base_offsets must contain an equal block table per row")
    block_table_len = spans.base_offsets.numel // rows
    _check_positive(block_table_len, "block_table_len_per_row")
    if spans.max_live_count >= block_size * block_table_len:
        raise ValueError("max_live_count must fit within each row block table")
    return block_table_len


def _check_int8_common_shape(
    spans: KVLiveSpans,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    if spans.spans_mode != "uniform":
        raise ValueError("INT8 paged KV write currently requires uniform spans")
    if spans.storage_dtype != DType.INT8_PER_TOKEN_HEAD:
        raise ValueError("INT8 paged KV write requires int8_per_token_head storage spans")
    if spans.live_counts.dtype != DType.INT64:
        raise ValueError("INT8 paged KV write requires int64 live_counts")
    block_table_len = _block_table_len(spans)
    _check_positive(block_table_len, "block_table_len")
    _check_positive(block_size, "block_size")
    _check_positive(num_kv_heads, "num_kv_heads")
    _check_positive(head_dim, "head_dim")
    if spans.max_live_count >= block_size * block_table_len:
        raise ValueError("max_live_count must fit within the INT8 paged span block table")
    return block_table_len


def _check_int8_write_shape(
    spans: KVLiveSpans,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    k_scale_ptr: int,
    v_scale_ptr: int,
) -> int:
    block_table_len = _check_int8_common_shape(spans, block_size, num_kv_heads, head_dim)
    _check_int8_scale_metadata(
        spans,
        block_size,
        num_kv_heads,
        required_blocks=block_table_len,
        k_scale_ptr=k_scale_ptr,
        v_scale_ptr=v_scale_ptr,
    )
    return block_table_len


def _check_int8_write_batch_shape(
    spans: KVLiveSpans,
    rows: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    row_major_cache: bool,
    k_scale_ptr: int,
    v_scale_ptr: int,
) -> int:
    block_table_len = _check_int8_common_shape(spans, block_size, num_kv_heads, head_dim)
    _check_positive(rows, "rows")
    if spans.live_counts.numel < rows:
        raise ValueError("live_counts must have at least rows entries")
    if spans.base_offsets.numel % rows != 0:
        raise ValueError("base_offsets must contain an equal block table per row")
    block_table_len = spans.base_offsets.numel // rows
    _check_positive(block_table_len, "block_table_len_per_row")
    if spans.max_live_count >= block_size * block_table_len:
        raise ValueError("max_live_count must fit within each INT8 row block table")
    required_scale_blocks = rows * block_table_len if row_major_cache else block_table_len
    _check_int8_scale_metadata(
        spans,
        block_size,
        num_kv_heads,
        required_blocks=required_scale_blocks,
        k_scale_ptr=k_scale_ptr,
        v_scale_ptr=v_scale_ptr,
    )
    return block_table_len


def _check_int8_scale_metadata(
    spans: KVLiveSpans,
    block_size: int,
    num_kv_heads: int,
    *,
    required_blocks: int,
    k_scale_ptr: int,
    v_scale_ptr: int,
) -> None:
    metadata = spans.scale_metadata
    if metadata is None:  # defensive; KVLiveSpans rejects this combination first.
        raise ValueError("int8_per_token_head spans require scale metadata")
    if int(k_scale_ptr) != int(metadata.k_scale.ptr):
        raise ValueError("k_scale_ptr must match spans.scale_metadata.k_scale")
    if int(v_scale_ptr) != int(metadata.v_scale.ptr):
        raise ValueError("v_scale_ptr must match spans.scale_metadata.v_scale")
    if metadata.scale_dtype not in {DType.FP16, DType.FP32}:
        raise ValueError("INT8 scale metadata must be fp16 or fp32")
    if len(metadata.k_scale.shape) != 3:
        raise ValueError("INT8 scale tensors must have shape [blocks, block_size, num_kv_heads]")
    scale_blocks, scale_block_size, scale_heads = (int(dim) for dim in metadata.k_scale.shape)
    if scale_block_size != block_size or scale_heads != num_kv_heads:
        raise ValueError("INT8 scale tensor shape must match block_size and num_kv_heads")
    if scale_blocks < required_blocks:
        raise ValueError("INT8 scale tensors must cover the paged block table")


def _int8_symbol(spans: KVLiveSpans, *, batch: bool, prompt: bool) -> str:
    metadata = spans.scale_metadata
    scale_dtype = metadata.scale_dtype if metadata is not None else None
    if scale_dtype == DType.FP32:
        if prompt:
            return _SYMBOL_INT8_SCALE_F32_PROMPT
        if batch:
            return _SYMBOL_INT8_SCALE_F32_BATCH
        return _SYMBOL_INT8_SCALE_F32
    if scale_dtype == DType.FP16:
        if prompt:
            return _SYMBOL_INT8_SCALE_FP16_PROMPT
        if batch:
            return _SYMBOL_INT8_SCALE_FP16_BATCH
        return _SYMBOL_INT8_SCALE_FP16
    if prompt:
        return _SYMBOL_INT8_SCALE_F32_PROMPT
    if batch:
        return _SYMBOL_INT8_SCALE_F32_BATCH
    return _SYMBOL_INT8_SCALE_F32


def _block_table_len(spans: KVLiveSpans) -> int:
    return spans.base_offsets.numel


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_qwen35_paged_kv_write_kernels()
