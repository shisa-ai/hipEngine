"""Laguna global and token-granular SWA ``KVLiveSpans`` kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.dtype import DType
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register
from hipengine.kvcache import KVLiveSpans

_SOURCE = Path(__file__).with_name("laguna_kv_attention.hip")
_OUTPUT_NAME = "laguna_kv_attention.so"
_SYMBOL_GLOBAL_WRITE = "hipengine_laguna_global_write_kv_f32_bf16_spans"
_SYMBOL_GLOBAL_WRITE_ROWS = "hipengine_laguna_global_write_kv_rows_f32_bf16_spans"
_SYMBOL_GLOBAL_ATTENTION = "hipengine_laguna_global_attention_decode_bf16_spans"
_SYMBOL_GLOBAL_PREFILL = "hipengine_laguna_global_attention_prefill_bf16_spans"
_SYMBOL_GLOBAL_PREFILL_QROW2_ONLINE = (
    "hipengine_laguna_global_attention_prefill_qrow2_online_bf16_spans"
)
_SYMBOL_GLOBAL_PREFILL_QROW4_ONLINE = (
    "hipengine_laguna_global_attention_prefill_qrow4_online_bf16_spans"
)
_SYMBOL_SWA_WRITE = "hipengine_laguna_swa_write_kv_f32_bf16_spans"
_SYMBOL_SWA_WRITE_ROWS = "hipengine_laguna_swa_write_kv_rows_f32_bf16_spans"
_SYMBOL_SWA_ATTENTION = "hipengine_laguna_swa_attention_decode_bf16_spans"
_SYMBOL_SWA_ATTENTION_TOKEN4_EXACT = (
    "hipengine_laguna_swa_attention_decode_token4_exact_bf16_spans"
)
_SYMBOL_SWA_PREFILL = "hipengine_laguna_swa_attention_prefill_bf16_spans"
_SYMBOL_SWA_PREFILL_WAVE32_EXACT = (
    "hipengine_laguna_swa_attention_prefill_wave32_exact_bf16_spans"
)
_SYMBOL_SWA_PREFILL_QROW2_EXACT = (
    "hipengine_laguna_swa_attention_prefill_qrow2_exact_bf16_spans"
)
_SYMBOL_SWA_PREFILL_QROW2_ONLINE = (
    "hipengine_laguna_swa_attention_prefill_qrow2_online_bf16_spans"
)
_SYMBOL_SWA_PREFILL_QROW4_ONLINE = (
    "hipengine_laguna_swa_attention_prefill_qrow4_online_bf16_spans"
)
_LAGUNA_KV_HEADS = 8
_LAGUNA_HEAD_DIM = 128
_GLOBAL_BLOCK_SIZE = 256
_MAX_EAGER_GLOBAL_CONTEXT = 4096


def plan_laguna_kv_attention_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="laguna_kv_attention",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_laguna_kv_attention(
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
        family="laguna_kv_attention",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def laguna_global_write_kv_f32_spans(
    key_ptr: int,
    value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Append one F32 K/V row to complete block-256 global spans."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_WRITE)
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
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_write_kv_rows_f32_spans(
    key_ptr: int,
    value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Append consecutive F32 K/V rows to complete block-256 global spans."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_WRITE_ROWS)
    fn.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int64] * 6 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(key_ptr),
        ctypes.c_void_p(value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run ungated block-256 global GQA through complete dense spans."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    if int(max_context_len) != capacity:
        raise ValueError("max_context_len must equal the global span capacity")
    if capacity > _MAX_EAGER_GLOBAL_CONTEXT:
        raise ValueError("eager Laguna global attention currently supports at most 4096 tokens")
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_ATTENTION)
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_prefill_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run causal block-256 global GQA over prior cache plus current rows."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    if int(max_context_len) != capacity:
        raise ValueError("max_context_len must equal the global span capacity")
    if capacity > _MAX_EAGER_GLOBAL_CONTEXT:
        raise ValueError("Laguna global prefill currently supports at most 4096 tokens")
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_PREFILL)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 7
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_prefill_qrow2_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run two-query-row online-softmax global GQA over prior and current rows."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    if int(max_context_len) != capacity:
        raise ValueError("max_context_len must equal the global span capacity")
    if capacity > _MAX_EAGER_GLOBAL_CONTEXT:
        raise ValueError("Laguna global prefill currently supports at most 4096 tokens")
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_PREFILL_QROW2_ONLINE)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 7
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_prefill_qrow4_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run four-query-row online-softmax global GQA over prior and current rows."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    if int(max_context_len) != capacity:
        raise ValueError("max_context_len must equal the global span capacity")
    if capacity > _MAX_EAGER_GLOBAL_CONTEXT:
        raise ValueError("Laguna global prefill currently supports at most 4096 tokens")
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_PREFILL_QROW4_ONLINE)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 7
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_prefill_qrow4_m128_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Use qrow4 for production M128 tiles and qrow2 for residual tiles."""

    kernel = (
        laguna_global_attention_prefill_qrow4_online_bf16_spans
        if int(rows) == 128
        else laguna_global_attention_prefill_qrow2_online_bf16_spans
    )
    kernel(
        query_ptr,
        current_key_ptr,
        current_value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        spans,
        rows,
        max_context_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_write_kv_f32_spans(
    key_ptr: int,
    value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Append one F32 K/V row into a token-granular BF16 SWA ring."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_WRITE)
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
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_write_kv_rows_f32_spans(
    key_ptr: int,
    value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Append consecutive F32 K/V rows into a token-granular BF16 ring."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_WRITE_ROWS)
    fn.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int64] * 4 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(key_ptr),
        ctypes.c_void_p(value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_decode_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Decode absolute-position GQA over live, non-evicted SWA ring slots."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if window <= 0 or window > capacity:
        raise ValueError("sliding_window must be in [1, ring capacity]")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_ATTENTION)
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(window),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_decode_token4_exact_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact token4 score-parallel decode over SWA ``KVLiveSpans``."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if window <= 0 or window > capacity:
        raise ValueError("sliding_window must be in [1, ring capacity]")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_ATTENTION_TOKEN4_EXACT)
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(window),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_prefill_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    start_position: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run causal GQA over prior ring state plus current consecutive rows."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if window <= 0 or window > capacity:
        raise ValueError("sliding_window must be in [1, ring capacity]")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_PREFILL)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 6
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(window),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_prefill_wave32_exact_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    start_position: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run reduction-order-exact causal SWA with one wave32 per query head."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if window <= 0 or window > capacity:
        raise ValueError("sliding_window must be in [1, ring capacity]")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_PREFILL_WAVE32_EXACT)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 6
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(window),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_prefill_qrow2_exact_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    start_position: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact SWA while reusing each K/V row across two query rows."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if window <= 0 or window > capacity:
        raise ValueError("sliding_window must be in [1, ring capacity]")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_PREFILL_QROW2_EXACT)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 6
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(window),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_prefill_qrow2_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    start_position: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run two-query-row online-softmax SWA over prior and current rows."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if window <= 0 or window > capacity:
        raise ValueError("sliding_window must be in [1, ring capacity]")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_PREFILL_QROW2_ONLINE)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 6
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(window),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_prefill_qrow4_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    start_position: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run four-query-row online-softmax SWA over prior and current rows."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if window <= 0 or window > capacity:
        raise ValueError("sliding_window must be in [1, ring capacity]")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_PREFILL_QROW4_ONLINE)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 6
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(window),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_prefill_qrow4_m128_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    start_position: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Use qrow4 for production M128 tiles and qrow2 for residual tiles."""

    kernel = (
        laguna_swa_attention_prefill_qrow4_online_bf16_spans
        if int(rows) == 128
        else laguna_swa_attention_prefill_qrow2_online_bf16_spans
    )
    kernel(
        query_ptr,
        current_key_ptr,
        current_value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        spans,
        rows,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=sliding_window,
        start_position=start_position,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_prefill_qrow2_m128_c128_exact_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    start_position: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Pair query rows only after both measured row/context crossovers."""

    kernel = (
        laguna_swa_attention_prefill_qrow2_exact_bf16_spans
        if int(rows) == 128
        and start_position is not None
        and int(start_position) >= 128
        else laguna_swa_attention_prefill_wave32_exact_bf16_spans
    )
    kernel(
        query_ptr,
        current_key_ptr,
        current_value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        spans,
        rows,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=sliding_window,
        start_position=start_position,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def register_laguna_kv_attention_kernels(*, replace: bool = True) -> None:
    registrations = (
        (
            "laguna_kv_write",
            "global_f32_spans",
            laguna_global_write_kv_f32_spans,
        ),
        (
            "laguna_kv_write",
            "swa_f32_spans",
            laguna_swa_write_kv_f32_spans,
        ),
        (
            "laguna_kv_write",
            "global_f32_rows_spans",
            laguna_global_write_kv_rows_f32_spans,
        ),
        (
            "laguna_kv_write",
            "swa_f32_rows_spans",
            laguna_swa_write_kv_rows_f32_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_spans",
            laguna_global_attention_decode_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_spans",
            laguna_swa_attention_decode_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_token4_exact_spans",
            laguna_swa_attention_decode_token4_exact_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "global_context_rows_spans",
            laguna_global_attention_prefill_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "global_context_rows_qrow2_online_spans",
            laguna_global_attention_prefill_qrow2_online_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "global_context_rows_qrow4_online_spans",
            laguna_global_attention_prefill_qrow4_online_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "global_context_rows_qrow4_m128_online_spans",
            laguna_global_attention_prefill_qrow4_m128_online_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "swa_context_rows_spans",
            laguna_swa_attention_prefill_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "swa_context_rows_wave32_exact_spans",
            laguna_swa_attention_prefill_wave32_exact_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "swa_context_rows_qrow2_exact_spans",
            laguna_swa_attention_prefill_qrow2_exact_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "swa_context_rows_qrow2_m128_c128_exact_spans",
            laguna_swa_attention_prefill_qrow2_m128_c128_exact_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "swa_context_rows_qrow2_online_spans",
            laguna_swa_attention_prefill_qrow2_online_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "swa_context_rows_qrow4_online_spans",
            laguna_swa_attention_prefill_qrow4_online_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "swa_context_rows_qrow4_m128_online_spans",
            laguna_swa_attention_prefill_qrow4_m128_online_bf16_spans,
        ),
    )
    for layer, variant, kernel in registrations:
        register(
            KernelKey("hip_gfx1100", layer, "bf16", variant),
            kernel,
            replace=replace,
        )


def _check_global_spans(
    spans: KVLiveSpans,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    if spans.spans_mode != "uniform":
        raise ValueError("Laguna global KV requires uniform spans")
    if spans.storage_dtype != DType.BF16:
        raise ValueError("Laguna global KV requires bf16 storage")
    if spans.token_positions is None or spans.evict_mask is None:
        raise ValueError("Laguna global KV requires token_positions and evict_mask")
    if spans.row_positions is None:
        raise ValueError("Laguna global KV requires absolute row_positions")
    capacity = spans.token_positions.numel
    if spans.evict_mask.numel != capacity or spans.max_live_count != capacity:
        raise ValueError("Laguna global span metadata must match its logical capacity")
    if (_GLOBAL_BLOCK_SIZE * spans.base_offsets.numel) < capacity:
        raise ValueError("Laguna global block table is too short for its logical capacity")
    _check_laguna_kv_shape(num_kv_heads, head_dim)
    return capacity


def _check_swa_spans(
    spans: KVLiveSpans,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    if spans.spans_mode != "sliding_ring":
        raise ValueError("Laguna SWA requires sliding_ring spans")
    if spans.storage_dtype != DType.BF16:
        raise ValueError("Laguna SWA requires bf16 storage")
    if spans.token_positions is None or spans.evict_mask is None:
        raise ValueError("Laguna SWA requires token_positions and evict_mask")
    if spans.row_positions is None:
        raise ValueError("Laguna SWA requires absolute row_positions")
    _check_laguna_kv_shape(num_kv_heads, head_dim)
    return spans.base_offsets.numel


def _check_prefill_rows(spans: KVLiveSpans, rows: int, capacity: int) -> None:
    parsed = int(rows)
    if parsed <= 0 or parsed > int(capacity):
        raise ValueError("rows must be within [1, span capacity]")
    assert spans.row_positions is not None
    if spans.row_positions.numel != 1:
        raise ValueError("row_positions must contain the consecutive chunk start scalar")


def _check_laguna_kv_shape(num_kv_heads: int, head_dim: int) -> None:
    if int(num_kv_heads) != _LAGUNA_KV_HEADS:
        raise ValueError(f"num_kv_heads must be {_LAGUNA_KV_HEADS} for Laguna")
    if int(head_dim) != _LAGUNA_HEAD_DIM:
        raise ValueError(f"head_dim must be {_LAGUNA_HEAD_DIM} for Laguna")


def _check_laguna_attention_shape(
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    _check_laguna_kv_shape(num_kv_heads, head_dim)
    if int(num_q_heads) % int(num_kv_heads):
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if int(num_q_heads) not in {48, 72}:
        raise ValueError("num_q_heads must be a Laguna production width (48 or 72)")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_laguna_kv_attention_kernels()

__all__ = [
    "build_laguna_kv_attention",
    "laguna_global_attention_decode_bf16_spans",
    "laguna_global_attention_prefill_bf16_spans",
    "laguna_global_attention_prefill_qrow2_online_bf16_spans",
    "laguna_global_attention_prefill_qrow4_online_bf16_spans",
    "laguna_global_attention_prefill_qrow4_m128_online_bf16_spans",
    "laguna_global_write_kv_f32_spans",
    "laguna_global_write_kv_rows_f32_spans",
    "laguna_swa_attention_decode_bf16_spans",
    "laguna_swa_attention_decode_token4_exact_bf16_spans",
    "laguna_swa_attention_prefill_bf16_spans",
    "laguna_swa_attention_prefill_qrow2_m128_c128_exact_bf16_spans",
    "laguna_swa_attention_prefill_qrow2_exact_bf16_spans",
    "laguna_swa_attention_prefill_qrow2_online_bf16_spans",
    "laguna_swa_attention_prefill_qrow4_online_bf16_spans",
    "laguna_swa_attention_prefill_qrow4_m128_online_bf16_spans",
    "laguna_swa_attention_prefill_wave32_exact_bf16_spans",
    "laguna_swa_write_kv_f32_spans",
    "laguna_swa_write_kv_rows_f32_spans",
    "plan_laguna_kv_attention_build",
    "register_laguna_kv_attention_kernels",
]
