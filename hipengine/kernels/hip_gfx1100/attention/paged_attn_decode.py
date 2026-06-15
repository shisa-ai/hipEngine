"""Raw-pointer wrappers for Qwen3.5 paged full-attention decode kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.dtype import DType
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register
from hipengine.kvcache import KVLiveSpans

_SOURCE = Path(__file__).with_name("paged_attn_decode.hip")
_OUTPUT_NAME = "qwen35_paged_attn_decode.so"
_SYMBOL_GATE_MUL_BF16 = "hipengine_qwen35_full_attn_gate_mul_bf16"
_SYMBOL_GATE_MUL_FP16 = "hipengine_qwen35_full_attn_gate_mul_fp16"
_SYMBOL_DENSE_CONTEXT = "hipengine_qwen35_full_attn_decode_context_bf16"
_SYMBOL_CONTEXT = "hipengine_qwen35_paged_full_attn_decode_context_bf16_spans"
_SYMBOL_CONTEXT_BATCH = "hipengine_qwen35_paged_full_attn_decode_context_bf16_batch_spans"
_SYMBOL_SPLIT_CONTEXT = "hipengine_qwen35_paged_full_attn_decode_split_k_context_bf16_spans"
_SYMBOL_SPLIT_WARP_CONTEXT = "hipengine_qwen35_paged_full_attn_decode_split_k_warp_context_bf16_spans"
_SYMBOL_SPLIT_GQA_CONTEXT = "hipengine_qwen35_paged_full_attn_decode_split_k_gqa_context_bf16_spans"
_SYMBOL_SPLIT_GQA_CONTEXT_BATCH = "hipengine_qwen35_paged_full_attn_decode_split_k_gqa_context_batch_bf16_spans"
_SYMBOL_SPLIT_REDUCE = "hipengine_qwen35_paged_full_attn_decode_split_k_reduce_f32"
_SYMBOL_SPLIT_REDUCE_GATE_F32 = "hipengine_qwen35_paged_full_attn_decode_split_k_reduce_gate_f32"
_SYMBOL_PREFILL_GQA_GATE_FP16 = "hipengine_qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans"
_SYMBOL_PREFILL_GQA_GATE_TREE_FP16 = "hipengine_qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans"
_SYMBOL_PREFILL_VARLEN_GQA_GATE_FP16 = "hipengine_qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_spans"
_SYMBOL_SPLIT_REDUCE_GATE_BF16 = "hipengine_qwen35_paged_full_attn_decode_split_k_reduce_gate_bf16"
_SYMBOL_SPLIT_REDUCE_GATE_FP16 = "hipengine_qwen35_paged_full_attn_decode_split_k_reduce_gate_fp16"
_SYMBOL_SPLIT_REDUCE_GATE_FP16_BATCH = "hipengine_qwen35_paged_full_attn_decode_split_k_reduce_gate_fp16_batch"
_SYMBOL_SPLIT_GQA_GATE_FP16_BATCH_DIRECT = (
    "hipengine_qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_direct_spans"
)
_SYMBOL_GATE_MUL_BF16_TO_BF16 = "hipengine_qwen35_full_attn_gate_mul_bf16_to_bf16"
_SYMBOL_SPLIT_GQA_INT8_CONTEXT_F32 = "hipengine_qwen35_paged_full_attn_decode_split_k_gqa_context_int8_scale_f32_spans"
_SYMBOL_SPLIT_GQA_INT8_CONTEXT_FP16 = "hipengine_qwen35_paged_full_attn_decode_split_k_gqa_context_int8_scale_fp16_spans"
_SYMBOL_PREFILL_GQA_GATE_BF16 = "hipengine_qwen35_paged_full_attn_prefill_gqa_gate_bf16_spans"
_SYMBOL_PREFILL_GQA_GATE_INT8_F32 = "hipengine_qwen35_paged_full_attn_prefill_gqa_gate_int8_scale_f32_spans"
_SYMBOL_PREFILL_GQA_GATE_INT8_FP16 = "hipengine_qwen35_paged_full_attn_prefill_gqa_gate_int8_scale_fp16_spans"


def plan_qwen35_paged_attn_decode_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="qwen35_paged_attn_decode",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_qwen35_paged_attn_decode(
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
        family="qwen35_paged_attn_decode",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def qwen35_full_attn_gate_mul_bf16(
    attn_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    total: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply BF16 sigmoid gate to a contiguous FP32 full-attention output."""

    _check_positive(total, "total")
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GATE_MUL_BF16)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(attn_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(total),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_full_attn_gate_mul_fp16(
    attn_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    total: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply FP16 sigmoid gate to a contiguous FP32 full-attention output."""

    _check_positive(total, "total")
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GATE_MUL_FP16)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(attn_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(total),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_full_attn_decode_context_bf16(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    context_len_ptr: int,
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
    """Decode dense BF16 full attention using a device context-length scalar."""

    _check_dense_decode_shape(max_context_len, num_q_heads, num_kv_heads, head_dim)
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DENSE_CONTEXT)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_void_p(context_len_ptr),
        ctypes.c_int64(max_context_len),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_paged_full_attn_decode_context_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    max_context_len: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Decode paged BF16 full attention using ``KVLiveSpans`` metadata.

    Fixed-page spans bridge to the preserved parent context-tensor kernel:
    ``spans.base_offsets`` is the int32 page table and ``spans.live_counts`` is
    the int64 context length tensor.
    """

    _check_decode_shape(spans, max_context_len, block_size, num_q_heads, num_kv_heads, head_dim)
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_CONTEXT)
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
        ctypes.c_int64(max_context_len),
        ctypes.c_int64(block_size),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_paged_full_attn_decode_context_bf16_batch_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run batched Qwen3.5 paged attention context decode via row spans."""

    block_table_len = _check_decode_batch_shape(spans, rows, max_context_len, block_size, num_q_heads, num_kv_heads, head_dim)
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_CONTEXT_BATCH)
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
        ctypes.c_int64(rows),
        ctypes.c_int64(max_context_len),
        ctypes.c_int64(block_size),
        ctypes.c_int64(block_table_len),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_paged_full_attn_decode_split_k_warp_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    spans: KVLiveSpans,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run parent Qwen3.5 warp-specialized split-K context and reduce."""

    _check_qwen35_gqa_shape(spans, chunk_size, num_splits, block_size, num_q_heads, num_kv_heads, head_dim)
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    _launch_split_context(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        spans,
        chunk_size,
        num_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        stream=stream,
        library=library,
        runtime=runtime,
        symbol=_SYMBOL_SPLIT_WARP_CONTEXT,
    )
    _launch_reduce(
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        out_ptr,
        num_q_heads,
        num_splits,
        head_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_paged_full_attn_decode_split_k_gqa_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    spans: KVLiveSpans,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run parent Qwen3.5 grouped-GQA split-K context and reduce."""

    _check_qwen35_gqa_shape(spans, chunk_size, num_splits, block_size, num_q_heads, num_kv_heads, head_dim)
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    _launch_split_context(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        spans,
        chunk_size,
        num_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        stream=stream,
        library=library,
        runtime=runtime,
        symbol=_SYMBOL_SPLIT_GQA_CONTEXT,
    )
    _launch_reduce(
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        out_ptr,
        num_q_heads,
        num_splits,
        head_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_paged_full_attn_decode_split_k_warp_gate_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    spans: KVLiveSpans,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gate_stride1: int,
    gate_stride2: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run parent warp-cooperative split-K context and BF16 gated reduce."""

    _check_qwen35_gqa_shape(spans, chunk_size, num_splits, block_size, num_q_heads, num_kv_heads, head_dim)
    _check_positive(gate_stride1, "gate_stride1")
    _check_positive(gate_stride2, "gate_stride2")
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    _launch_split_context(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        spans,
        chunk_size,
        num_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        stream=stream,
        library=library,
        runtime=runtime,
        symbol=_SYMBOL_SPLIT_WARP_CONTEXT,
    )
    _launch_gate_reduce(
        _SYMBOL_SPLIT_REDUCE_GATE_BF16,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        gate_ptr,
        out_ptr,
        num_q_heads,
        num_splits,
        head_dim,
        gate_stride1,
        gate_stride2,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_paged_full_attn_decode_split_k_warp_gate_fp16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    spans: KVLiveSpans,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gate_stride1: int,
    gate_stride2: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run parent warp-cooperative split-K context and FP16 gated reduce."""

    _check_qwen35_gqa_shape(spans, chunk_size, num_splits, block_size, num_q_heads, num_kv_heads, head_dim)
    _check_positive(gate_stride1, "gate_stride1")
    _check_positive(gate_stride2, "gate_stride2")
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    _launch_split_context(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        spans,
        chunk_size,
        num_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        stream=stream,
        library=library,
        runtime=runtime,
        symbol=_SYMBOL_SPLIT_WARP_CONTEXT,
    )
    _launch_gate_reduce(
        _SYMBOL_SPLIT_REDUCE_GATE_FP16,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        gate_ptr,
        out_ptr,
        num_q_heads,
        num_splits,
        head_dim,
        gate_stride1,
        gate_stride2,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    spans: KVLiveSpans,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gate_stride1: int,
    gate_stride2: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run parent grouped-GQA split-K context and BF16 gated reduce."""

    _check_qwen35_gqa_shape(spans, chunk_size, num_splits, block_size, num_q_heads, num_kv_heads, head_dim)
    _check_positive(gate_stride1, "gate_stride1")
    _check_positive(gate_stride2, "gate_stride2")
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    _launch_split_context(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        spans,
        chunk_size,
        num_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        stream=stream,
        library=library,
        runtime=runtime,
        symbol=_SYMBOL_SPLIT_GQA_CONTEXT,
    )
    _launch_gate_reduce(
        _SYMBOL_SPLIT_REDUCE_GATE_BF16,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        gate_ptr,
        out_ptr,
        num_q_heads,
        num_splits,
        head_dim,
        gate_stride1,
        gate_stride2,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    spans: KVLiveSpans,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gate_stride1: int,
    gate_stride2: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run parent grouped-GQA split-K context and FP16 gated reduce."""

    _check_qwen35_gqa_shape(spans, chunk_size, num_splits, block_size, num_q_heads, num_kv_heads, head_dim)
    _check_positive(gate_stride1, "gate_stride1")
    _check_positive(gate_stride2, "gate_stride2")
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    _launch_split_context(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        spans,
        chunk_size,
        num_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        stream=stream,
        library=library,
        runtime=runtime,
        symbol=_SYMBOL_SPLIT_GQA_CONTEXT,
    )
    _launch_gate_reduce(
        _SYMBOL_SPLIT_REDUCE_GATE_FP16,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        gate_ptr,
        out_ptr,
        num_q_heads,
        num_splits,
        head_dim,
        gate_stride1,
        gate_stride2,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gate_stride1: int,
    gate_stride2: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run small-B row-batched Qwen3.5 GQA decode and FP16 gated reduce."""

    block_table_len = _check_qwen35_gqa_batch_shape(
        spans,
        rows,
        chunk_size,
        num_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )
    _check_positive(gate_stride1, "gate_stride1")
    _check_positive(gate_stride2, "gate_stride2")
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    _launch_split_context_batch(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        spans,
        rows,
        chunk_size,
        num_splits,
        block_size,
        block_table_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        stream=stream,
        library=library,
        runtime=runtime,
        symbol=_SYMBOL_SPLIT_GQA_CONTEXT_BATCH,
    )
    _launch_gate_reduce_batch(
        _SYMBOL_SPLIT_REDUCE_GATE_FP16_BATCH,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        gate_ptr,
        out_ptr,
        rows,
        num_q_heads,
        num_splits,
        head_dim,
        gate_stride1,
        gate_stride2,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_direct_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gate_stride1: int,
    gate_stride2: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run the Qwen3.5 row-batched GQA decode directly gated for one split."""

    block_table_len = _check_qwen35_gqa_batch_shape(
        spans,
        rows,
        chunk_size,
        num_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )
    if num_splits != 1:
        raise ValueError("direct Qwen3.5 GQA batch gate requires num_splits=1")
    _check_positive(gate_stride1, "gate_stride1")
    _check_positive(gate_stride2, "gate_stride2")
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SPLIT_GQA_GATE_FP16_BATCH_DIRECT)
    fn.argtypes = [
        ctypes.c_void_p,  # query
        ctypes.c_void_p,  # key_cache
        ctypes.c_void_p,  # value_cache
        ctypes.c_void_p,  # gate
        ctypes.c_void_p,  # out
        ctypes.c_void_p,  # base_offsets
        ctypes.c_void_p,  # live_counts
        ctypes.c_int64,   # rows
        ctypes.c_int64,   # chunk_size
        ctypes.c_int64,   # num_splits
        ctypes.c_int64,   # block_size
        ctypes.c_int64,   # block_table_len
        ctypes.c_int64,   # num_q_heads
        ctypes.c_int64,   # num_kv_heads
        ctypes.c_int64,   # head_dim
        ctypes.c_int64,   # gate_stride1
        ctypes.c_int64,   # gate_stride2
        ctypes.c_float,   # scale
        ctypes.c_void_p,  # stream
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(chunk_size),
        ctypes.c_int64(num_splits),
        ctypes.c_int64(block_size),
        ctypes.c_int64(block_table_len),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_int64(gate_stride1),
        ctypes.c_int64(gate_stride2),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gate_stride1: int,
    gate_stride2: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run native append-then-attend causal GQA prefill with FP16 gate/output."""

    block_table_len = _check_prefill_gqa_shape(
        spans,
        rows,
        max_context_len,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )
    _check_positive(gate_stride1, "gate_stride1")
    _check_positive(gate_stride2, "gate_stride2")
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_GQA_GATE_FP16)
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    row_positions_ptr = 0 if spans.row_positions is None else spans.row_positions.ptr
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(row_positions_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(max_context_len),
        ctypes.c_int64(block_size),
        ctypes.c_int64(block_table_len),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_int64(gate_stride1),
        ctypes.c_int64(gate_stride2),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_paged_attn_prefill_int8_gqa_gate_fp16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    k_scale_ptr: int,
    v_scale_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gate_stride1: int,
    gate_stride2: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run streaming causal GQA prefill over INT8 per-token/head K/V.

    This is the INT8 retained-KV prefill path: it consumes the same paged
    cache and scale metadata used by decode and performs an online softmax
    reduction per row/head, without materializing a BF16 K/V oracle cache or
    row/head/split partial tensors proportional to context length.
    """

    block_table_len = _check_int8_prefill_gqa_shape(
        spans,
        rows,
        max_context_len,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        k_scale_ptr=k_scale_ptr,
        v_scale_ptr=v_scale_ptr,
    )
    _check_positive(gate_stride1, "gate_stride1")
    _check_positive(gate_stride2, "gate_stride2")
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _int8_prefill_gqa_symbol(spans))
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
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
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
    row_positions_ptr = 0 if spans.row_positions is None else spans.row_positions.ptr
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(k_scale_ptr),
        ctypes.c_void_p(v_scale_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(row_positions_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(max_context_len),
        ctypes.c_int64(block_size),
        ctypes.c_int64(block_table_len),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_int64(gate_stride1),
        ctypes.c_int64(gate_stride2),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    ancestor_mask_ptr: int,
    tree_committed_count_ptr: int,
    rows: int,
    max_context_len: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gate_stride1: int,
    gate_stride2: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Tree-aware prefill GQA gate attention.

    Identical kernel layout to ``qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans``
    but adds an ``ancestor_mask`` (``[rows, rows]`` uint8) that filters the
    verifier-row block ``[tree_committed_count, tree_committed_count + rows)``
    inside the K cache.  Committed-context positions below
    ``tree_committed_count`` are visible to every row; the mask only constrains
    which verifier rows each query row can attend to.  Used by the DDTree
    full-attention verifier.
    """

    block_table_len = _check_prefill_gqa_shape(
        spans,
        rows,
        max_context_len,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )
    _check_positive(gate_stride1, "gate_stride1")
    _check_positive(gate_stride2, "gate_stride2")
    if ancestor_mask_ptr <= 0:
        raise ValueError("ancestor_mask_ptr must reference a [rows, rows] uint8 device buffer")
    if tree_committed_count_ptr <= 0:
        raise ValueError("tree_committed_count_ptr must reference an int64 device scalar (graph-capture-safe)")
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_GQA_GATE_TREE_FP16)
    fn.argtypes = [
        ctypes.c_void_p,  # query
        ctypes.c_void_p,  # key_cache
        ctypes.c_void_p,  # value_cache
        ctypes.c_void_p,  # gate
        ctypes.c_void_p,  # out
        ctypes.c_void_p,  # block_tables
        ctypes.c_void_p,  # context_counts
        ctypes.c_void_p,  # row_positions
        ctypes.c_void_p,  # ancestor_mask
        ctypes.c_void_p,  # tree_committed_count_ptr
        ctypes.c_int64,   # rows
        ctypes.c_int64,   # max_context_len
        ctypes.c_int64,   # block_size
        ctypes.c_int64,   # block_table_len
        ctypes.c_int64,   # num_q_heads
        ctypes.c_int64,   # num_kv_heads
        ctypes.c_int64,   # head_dim
        ctypes.c_int64,   # gate_stride1
        ctypes.c_int64,   # gate_stride2
        ctypes.c_float,   # scale
        ctypes.c_void_p,  # stream
    ]
    fn.restype = ctypes.c_int
    row_positions_ptr = 0 if spans.row_positions is None else spans.row_positions.ptr
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(row_positions_ptr),
        ctypes.c_void_p(ancestor_mask_ptr),
        ctypes.c_void_p(tree_committed_count_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(max_context_len),
        ctypes.c_int64(block_size),
        ctypes.c_int64(block_table_len),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_int64(gate_stride1),
        ctypes.c_int64(gate_stride2),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    cu_seqlens_q_ptr: int,
    cu_seqlens_k_ptr: int,
    rows: int,
    segments: int,
    max_context_len: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gate_stride1: int,
    gate_stride2: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run varlen/block-diagonal append-then-attend prefill with FP16 gate/output.

    ``cu_seqlens_q`` and ``cu_seqlens_k`` are device int32 arrays with
    ``segments + 1`` entries. ``spans`` remains row-shaped so each packed query
    row can carry the block table and visible context for its owning request.
    """

    block_table_len = _check_prefill_gqa_shape(
        spans,
        rows,
        max_context_len,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )
    _check_positive(segments, "segments")
    _check_positive(gate_stride1, "gate_stride1")
    _check_positive(gate_stride2, "gate_stride2")
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_VARLEN_GQA_GATE_FP16)
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
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
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
    row_positions_ptr = 0 if spans.row_positions is None else spans.row_positions.ptr
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(row_positions_ptr),
        ctypes.c_void_p(cu_seqlens_q_ptr),
        ctypes.c_void_p(cu_seqlens_k_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(segments),
        ctypes.c_int64(max_context_len),
        ctypes.c_int64(block_size),
        ctypes.c_int64(block_table_len),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_int64(gate_stride1),
        ctypes.c_int64(gate_stride2),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_paged_full_attn_decode_split_k_gate_f32_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    spans: KVLiveSpans,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gate_stride1: int,
    gate_stride2: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run parent split-K paged BF16 attention and FP32 gated reduce via spans."""

    _check_split_shape(spans, chunk_size, num_splits, block_size, num_q_heads, num_kv_heads, head_dim)
    _check_positive(gate_stride1, "gate_stride1")
    _check_positive(gate_stride2, "gate_stride2")
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    _launch_split_context(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        spans,
        chunk_size,
        num_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        stream=stream,
        library=library,
        runtime=runtime,
    )

    _launch_gate_reduce(
        _SYMBOL_SPLIT_REDUCE_GATE_F32,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        gate_ptr,
        out_ptr,
        num_q_heads,
        num_splits,
        head_dim,
        gate_stride1,
        gate_stride2,
        stream=stream,
        library=library,
        runtime=runtime,
    )

def qwen35_paged_full_attn_decode_split_k_gate_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    spans: KVLiveSpans,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gate_stride1: int,
    gate_stride2: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run parent split-K paged BF16 attention and BF16 gated reduce via spans."""

    _check_split_shape(spans, chunk_size, num_splits, block_size, num_q_heads, num_kv_heads, head_dim)
    _check_positive(gate_stride1, "gate_stride1")
    _check_positive(gate_stride2, "gate_stride2")
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    _launch_split_context(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        spans,
        chunk_size,
        num_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        stream=stream,
        library=library,
        runtime=runtime,
    )

    _launch_gate_reduce(
        _SYMBOL_SPLIT_REDUCE_GATE_BF16,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        gate_ptr,
        out_ptr,
        num_q_heads,
        num_splits,
        head_dim,
        gate_stride1,
        gate_stride2,
        stream=stream,
        library=library,
        runtime=runtime,
    )

def qwen35_paged_full_attn_decode_split_k_gate_fp16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    spans: KVLiveSpans,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gate_stride1: int,
    gate_stride2: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run parent split-K paged BF16 attention and FP16 gated reduce via spans."""

    _check_split_shape(spans, chunk_size, num_splits, block_size, num_q_heads, num_kv_heads, head_dim)
    _check_positive(gate_stride1, "gate_stride1")
    _check_positive(gate_stride2, "gate_stride2")
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    _launch_split_context(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        spans,
        chunk_size,
        num_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        stream=stream,
        library=library,
        runtime=runtime,
    )

    _launch_gate_reduce(
        _SYMBOL_SPLIT_REDUCE_GATE_FP16,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        gate_ptr,
        out_ptr,
        num_q_heads,
        num_splits,
        head_dim,
        gate_stride1,
        gate_stride2,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_paged_full_attn_decode_split_k_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    spans: KVLiveSpans,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run parent split-K paged BF16 attention decode and reduce via spans."""

    _check_split_shape(spans, chunk_size, num_splits, block_size, num_q_heads, num_kv_heads, head_dim)
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    _launch_split_context(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        spans,
        chunk_size,
        num_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        stream=stream,
        library=library,
        runtime=runtime,
    )

    _launch_reduce(
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        out_ptr,
        num_q_heads,
        num_splits,
        head_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )

def qwen35_full_attn_gate_mul_bf16_to_bf16(
    attn_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    total: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply BF16 sigmoid gate to a contiguous BF16 full-attention output."""

    _check_positive(total, "total")
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GATE_MUL_BF16_TO_BF16)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(attn_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(total),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_paged_attn_decode_int8_gqa_splitk_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    k_scale_ptr: int,
    v_scale_ptr: int,
    out_ptr: int,
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    spans: KVLiveSpans,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Decode Qwen3.5 grouped-GQA split-K attention directly from INT8 K/V + scales."""

    _check_int8_qwen35_gqa_shape(
        spans,
        chunk_size,
        num_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        k_scale_ptr=k_scale_ptr,
        v_scale_ptr=v_scale_ptr,
    )
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    _launch_int8_gqa_split_context(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        k_scale_ptr,
        v_scale_ptr,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        spans,
        chunk_size,
        num_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        stream=stream,
        library=library,
        runtime=runtime,
    )
    _launch_reduce(
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        out_ptr,
        num_q_heads,
        num_splits,
        head_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    k_scale_ptr: int,
    v_scale_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    spans: KVLiveSpans,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gate_stride1: int,
    gate_stride2: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Decode INT8 grouped-GQA split-K attention and apply BF16 sigmoid gate."""

    _check_int8_qwen35_gqa_shape(
        spans,
        chunk_size,
        num_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        k_scale_ptr=k_scale_ptr,
        v_scale_ptr=v_scale_ptr,
    )
    _check_positive(gate_stride1, "gate_stride1")
    _check_positive(gate_stride2, "gate_stride2")
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    _launch_int8_gqa_split_context(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        k_scale_ptr,
        v_scale_ptr,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        spans,
        chunk_size,
        num_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        stream=stream,
        library=library,
        runtime=runtime,
    )
    _launch_gate_reduce(
        _SYMBOL_SPLIT_REDUCE_GATE_BF16,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        gate_ptr,
        out_ptr,
        num_q_heads,
        num_splits,
        head_dim,
        gate_stride1,
        gate_stride2,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_paged_attn_decode_int8_gqa_splitk_gate_fp16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    k_scale_ptr: int,
    v_scale_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    spans: KVLiveSpans,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gate_stride1: int,
    gate_stride2: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Decode INT8 grouped-GQA split-K attention and apply FP16 sigmoid gate."""

    _check_int8_qwen35_gqa_shape(
        spans,
        chunk_size,
        num_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        k_scale_ptr=k_scale_ptr,
        v_scale_ptr=v_scale_ptr,
    )
    _check_positive(gate_stride1, "gate_stride1")
    _check_positive(gate_stride2, "gate_stride2")
    library = library or build_qwen35_paged_attn_decode(load=True)
    runtime = runtime or get_hip_runtime()
    _launch_int8_gqa_split_context(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        k_scale_ptr,
        v_scale_ptr,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        spans,
        chunk_size,
        num_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        stream=stream,
        library=library,
        runtime=runtime,
    )
    _launch_gate_reduce(
        _SYMBOL_SPLIT_REDUCE_GATE_FP16,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        gate_ptr,
        out_ptr,
        num_q_heads,
        num_splits,
        head_dim,
        gate_stride1,
        gate_stride2,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_paged_full_attn_prefill_gqa_gate_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gate_stride1: int,
    gate_stride2: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run native append-then-attend causal GQA prefill with BF16 gate/output."""

    _launch_prefill_gqa_gate(
        _SYMBOL_PREFILL_GQA_GATE_BF16,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        gate_ptr,
        out_ptr,
        spans,
        rows,
        max_context_len,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        gate_stride1,
        gate_stride2,
        scale,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _launch_prefill_gqa_gate(
    symbol: str,
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gate_stride1: int,
    gate_stride2: int,
    scale: float,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    block_table_len = _check_prefill_gqa_shape(
        spans,
        rows,
        max_context_len,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )
    _check_positive(gate_stride1, "gate_stride1")
    _check_positive(gate_stride2, "gate_stride2")
    library = library or build_qwen35_paged_attn_decode(load=True)
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    row_positions_ptr = 0 if spans.row_positions is None else spans.row_positions.ptr
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(row_positions_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(max_context_len),
        ctypes.c_int64(block_size),
        ctypes.c_int64(block_table_len),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_int64(gate_stride1),
        ctypes.c_int64(gate_stride2),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_int8_gqa_split_context(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    k_scale_ptr: int,
    v_scale_ptr: int,
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    spans: KVLiveSpans,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int,
    library: ctypes.CDLL,
    runtime: HipRuntime,
) -> None:
    symbol = _int8_gqa_context_symbol(spans)
    split = getattr(library, symbol)
    split.argtypes = [
        ctypes.c_void_p,
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
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    split.restype = ctypes.c_int
    err = split(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(k_scale_ptr),
        ctypes.c_void_p(v_scale_ptr),
        ctypes.c_void_p(partial_out_ptr),
        ctypes.c_void_p(partial_m_ptr),
        ctypes.c_void_p(partial_l_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_int64(chunk_size),
        ctypes.c_int64(num_splits),
        ctypes.c_int64(block_size),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _check_int8_qwen35_gqa_shape(
    spans: KVLiveSpans,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    k_scale_ptr: int,
    v_scale_ptr: int,
) -> None:
    if spans.spans_mode != "uniform":
        raise ValueError("INT8 paged attention decode currently requires uniform spans")
    if spans.storage_dtype != DType.INT8_PER_TOKEN_HEAD:
        raise ValueError("INT8 paged attention decode requires int8_per_token_head storage spans")
    if spans.live_counts.dtype != DType.INT64:
        raise ValueError("INT8 paged attention decode requires int64 live_counts")
    _check_positive(chunk_size, "chunk_size")
    _check_positive(num_splits, "num_splits")
    _check_positive(block_size, "block_size")
    _check_positive(spans.base_offsets.numel, "block_table_len")
    _check_positive(num_q_heads, "num_q_heads")
    _check_positive(num_kv_heads, "num_kv_heads")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    _check_positive(head_dim, "head_dim")
    if block_size != 256 or num_q_heads != 16 or num_kv_heads != 2 or head_dim != 256:
        raise ValueError(
            "Qwen3.5 INT8 GQA split-K specialization requires block_size=256, "
            "num_q_heads=16, num_kv_heads=2, head_dim=256"
        )
    if ((chunk_size * num_splits + block_size - 1) // block_size) > spans.base_offsets.numel:
        raise ValueError("span base_offsets block table is too short for max split-K context")
    _check_int8_scale_metadata(spans, block_size, num_kv_heads, k_scale_ptr=k_scale_ptr, v_scale_ptr=v_scale_ptr)


def _check_int8_prefill_gqa_shape(
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    k_scale_ptr: int,
    v_scale_ptr: int,
) -> int:
    if spans.spans_mode != "uniform":
        raise ValueError("INT8 paged attention prefill currently requires uniform spans")
    if spans.storage_dtype != DType.INT8_PER_TOKEN_HEAD:
        raise ValueError("INT8 paged attention prefill requires int8_per_token_head storage spans")
    if spans.live_counts.dtype != DType.INT64:
        raise ValueError("INT8 paged attention prefill requires int64 live_counts")
    _check_positive(rows, "rows")
    _check_positive(max_context_len, "max_context_len")
    _check_positive(block_size, "block_size")
    _check_positive(num_q_heads, "num_q_heads")
    _check_positive(num_kv_heads, "num_kv_heads")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    _check_positive(head_dim, "head_dim")
    if head_dim > 256:
        raise ValueError("INT8 paged attention prefill currently requires head_dim <= 256")
    if spans.live_counts.numel < rows:
        raise ValueError("live_counts must have at least rows entries")
    if spans.row_positions is not None and spans.row_positions.dtype != DType.INT64:
        raise ValueError("INT8 paged attention prefill row_positions must be int64 when provided")
    if spans.row_positions is not None and spans.row_positions.numel < rows:
        raise ValueError("row_positions must have at least rows entries")
    if spans.base_offsets.numel % rows != 0:
        raise ValueError("prefill block table must be row-major [rows, blocks]")
    block_table_len = spans.base_offsets.numel // rows
    if block_table_len <= 0:
        raise ValueError("block_table_len must be positive")
    if ((max_context_len + block_size - 1) // block_size) > block_table_len:
        raise ValueError("span base_offsets block table is too short for max prefill context")
    _check_int8_prefill_scale_metadata(spans, block_size, num_kv_heads, k_scale_ptr=k_scale_ptr, v_scale_ptr=v_scale_ptr)
    return block_table_len


def _check_int8_prefill_scale_metadata(
    spans: KVLiveSpans,
    block_size: int,
    num_kv_heads: int,
    *,
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
        raise ValueError("INT8 attention scale metadata must be fp16 or fp32")
    if len(metadata.k_scale.shape) != 3 or len(metadata.v_scale.shape) != 3:
        raise ValueError("INT8 attention scale tensors must have shape [blocks, block_size, num_kv_heads]")
    scale_blocks, scale_block_size, scale_heads = (int(dim) for dim in metadata.k_scale.shape)
    v_scale_blocks, v_scale_block_size, v_scale_heads = (int(dim) for dim in metadata.v_scale.shape)
    if (
        scale_block_size != block_size
        or scale_heads != num_kv_heads
        or v_scale_block_size != block_size
        or v_scale_heads != num_kv_heads
    ):
        raise ValueError("INT8 attention scale tensor shape must match block_size and num_kv_heads")
    if v_scale_blocks != scale_blocks:
        raise ValueError("INT8 attention key/value scale tensors must have the same block count")
    if scale_blocks <= 0:
        raise ValueError("INT8 attention scale tensors must have at least one block")


def _check_int8_scale_metadata(
    spans: KVLiveSpans,
    block_size: int,
    num_kv_heads: int,
    *,
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
        raise ValueError("INT8 attention scale metadata must be fp16 or fp32")
    if len(metadata.k_scale.shape) != 3:
        raise ValueError("INT8 attention scale tensors must have shape [blocks, block_size, num_kv_heads]")
    scale_blocks, scale_block_size, scale_heads = (int(dim) for dim in metadata.k_scale.shape)
    if scale_block_size != block_size or scale_heads != num_kv_heads:
        raise ValueError("INT8 attention scale tensor shape must match block_size and num_kv_heads")
    if scale_blocks < spans.base_offsets.numel:
        raise ValueError("INT8 attention scale tensors must cover the paged block table")


def _int8_gqa_context_symbol(spans: KVLiveSpans) -> str:
    metadata = spans.scale_metadata
    scale_dtype = metadata.scale_dtype if metadata is not None else None
    if scale_dtype == DType.FP32:
        return _SYMBOL_SPLIT_GQA_INT8_CONTEXT_F32
    if scale_dtype == DType.FP16:
        return _SYMBOL_SPLIT_GQA_INT8_CONTEXT_FP16
    return _SYMBOL_SPLIT_GQA_INT8_CONTEXT_F32


def _int8_prefill_gqa_symbol(spans: KVLiveSpans) -> str:
    metadata = spans.scale_metadata
    scale_dtype = metadata.scale_dtype if metadata is not None else None
    if scale_dtype == DType.FP32:
        return _SYMBOL_PREFILL_GQA_GATE_INT8_F32
    if scale_dtype == DType.FP16:
        return _SYMBOL_PREFILL_GQA_GATE_INT8_FP16
    return _SYMBOL_PREFILL_GQA_GATE_INT8_F32


def register_qwen35_paged_attn_decode_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "full_attn_gate_mul", "w4_paro", "bf16"),
        qwen35_full_attn_gate_mul_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "full_attn_gate_mul", "w4_paro", "fp16"),
        qwen35_full_attn_gate_mul_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "full_attn_decode", "w4_paro", "bf16_context"),
        qwen35_full_attn_decode_context_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "w4_paro", "bf16_context_spans"),
        qwen35_paged_full_attn_decode_context_bf16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "w4_paro", "bf16_context_batch_spans"),
        qwen35_paged_full_attn_decode_context_bf16_batch_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "w4_paro", "bf16_split_k_spans"),
        qwen35_paged_full_attn_decode_split_k_bf16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "w4_paro", "bf16_split_k_warp_spans"),
        qwen35_paged_full_attn_decode_split_k_warp_bf16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "w4_paro", "bf16_split_k_warp_gate_bf16_spans"),
        qwen35_paged_full_attn_decode_split_k_warp_gate_bf16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "w4_paro", "bf16_split_k_warp_gate_fp16_spans"),
        qwen35_paged_full_attn_decode_split_k_warp_gate_fp16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "w4_paro", "bf16_split_k_gqa_spans"),
        qwen35_paged_full_attn_decode_split_k_gqa_bf16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "w4_paro", "bf16_split_k_gqa_gate_bf16_spans"),
        qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "w4_paro", "bf16_split_k_gqa_gate_fp16_spans"),
        qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "w4_paro", "bf16_split_k_gqa_gate_fp16_batch_spans"),
        qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "w4_paro", "bf16_split_k_gqa_gate_fp16_batch_direct_spans"),
        qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_direct_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "full_attn_prefill", "w4_paro", "qwen35_causal_gqa_gate_fp16"),
        qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_prefill", "w4_paro", "bf16_gqa_gate_fp16_spans"),
        qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_prefill", "w4_paro", "bf16_gqa_gate_bf16_spans"),
        qwen35_paged_full_attn_prefill_gqa_gate_bf16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_prefill", "int8_per_token_head", "per_token_head_gqa_gate_fp16_spans"),
        qwen35_paged_attn_prefill_int8_gqa_gate_fp16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "full_attn_prefill", "w4_paro", "qwen35_tree_gqa_gate_fp16"),
        qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "full_attn_prefill", "w4_paro", "qwen35_varlen_causal_gqa_gate_fp16"),
        qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "w4_paro", "bf16_split_k_gate_f32_spans"),
        qwen35_paged_full_attn_decode_split_k_gate_f32_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "w4_paro", "bf16_split_k_gate_bf16_spans"),
        qwen35_paged_full_attn_decode_split_k_gate_bf16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "w4_paro", "bf16_split_k_gate_fp16_spans"),
        qwen35_paged_full_attn_decode_split_k_gate_fp16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "int8_per_token_head", "gqa_splitk_spans"),
        qwen35_paged_attn_decode_int8_gqa_splitk_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "int8_per_token_head", "gqa_splitk_gate_bf16_spans"),
        qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "int8_per_token_head", "gqa_splitk_gate_fp16_spans"),
        qwen35_paged_attn_decode_int8_gqa_splitk_gate_fp16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "int8_per_token_head", "per_token_head_gqa_splitk_spans"),
        qwen35_paged_attn_decode_int8_gqa_splitk_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "int8_per_token_head", "per_token_head_gqa_splitk_gate_bf16_spans"),
        qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "int8_per_token_head", "per_token_head_gqa_splitk_gate_fp16_spans"),
        qwen35_paged_attn_decode_int8_gqa_splitk_gate_fp16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "full_attn_prefill", "gguf_qwen35", "causal_gqa_gate_bf16"),
        qwen35_paged_full_attn_prefill_gqa_gate_bf16_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_prefill", "gguf_qwen35", "bf16_gqa_gate_bf16_spans"),
        qwen35_paged_full_attn_prefill_gqa_gate_bf16_spans,
        replace=replace,
    )


def _check_dense_decode_shape(
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    _check_positive(max_context_len, "max_context_len")
    _check_positive(num_q_heads, "num_q_heads")
    _check_positive(num_kv_heads, "num_kv_heads")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    _check_positive(head_dim, "head_dim")
    if head_dim > 256:
        raise ValueError("head_dim must be <= 256")


def _check_decode_shape(
    spans: KVLiveSpans,
    max_context_len: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    if spans.spans_mode != "uniform":
        raise ValueError("paged attention decode currently requires uniform spans")
    if spans.storage_dtype != DType.BF16:
        raise ValueError("paged attention decode currently requires bf16 storage spans")
    if spans.live_counts.dtype != DType.INT64:
        raise ValueError("paged attention decode parent bridge requires int64 live_counts")
    _check_positive(max_context_len, "max_context_len")
    _check_positive(block_size, "block_size")
    if block_size != 256:
        raise ValueError("paged attention decode parent kernel requires block_size=256")
    _check_positive(spans.base_offsets.numel, "block_table_len")
    _check_positive(num_q_heads, "num_q_heads")
    _check_positive(num_kv_heads, "num_kv_heads")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    _check_positive(head_dim, "head_dim")
    if head_dim > 256:
        raise ValueError("head_dim must be <= 256")
    if ((max_context_len + block_size - 1) // block_size) > spans.base_offsets.numel:
        raise ValueError("span base_offsets block table is too short for max_context_len")


def _check_decode_batch_shape(
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    _check_decode_shape(spans, max_context_len, block_size, num_q_heads, num_kv_heads, head_dim)
    _check_positive(rows, "rows")
    if spans.live_counts.numel < rows:
        raise ValueError("live_counts must have at least rows entries")
    if spans.base_offsets.numel % rows != 0:
        raise ValueError("base_offsets must contain an equal block table per row")
    block_table_len = spans.base_offsets.numel // rows
    _check_positive(block_table_len, "block_table_len_per_row")
    if ((max_context_len + block_size - 1) // block_size) > block_table_len:
        raise ValueError("each row block table is too short for max_context_len")
    return block_table_len


def _check_prefill_gqa_shape(
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    block_table_len = _check_decode_batch_shape(
        spans,
        rows,
        max_context_len,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )
    if spans.row_positions is not None and spans.row_positions.dtype != DType.INT64:
        raise ValueError("prefill row_positions must be int64 when provided")
    return block_table_len


def _launch_reduce(
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    out_ptr: int,
    num_q_heads: int,
    num_splits: int,
    head_dim: int,
    *,
    stream: int,
    library: ctypes.CDLL,
    runtime: HipRuntime,
) -> None:
    reduce = getattr(library, _SYMBOL_SPLIT_REDUCE)
    reduce.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    reduce.restype = ctypes.c_int
    err = reduce(
        ctypes.c_void_p(partial_out_ptr),
        ctypes.c_void_p(partial_m_ptr),
        ctypes.c_void_p(partial_l_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_splits),
        ctypes.c_int64(head_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)

def _launch_gate_reduce(
    symbol: str,
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    num_q_heads: int,
    num_splits: int,
    head_dim: int,
    gate_stride1: int,
    gate_stride2: int,
    *,
    stream: int,
    library: ctypes.CDLL,
    runtime: HipRuntime,
) -> None:
    reduce_gate = getattr(library, symbol)
    reduce_gate.argtypes = [
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
    reduce_gate.restype = ctypes.c_int
    err = reduce_gate(
        ctypes.c_void_p(partial_out_ptr),
        ctypes.c_void_p(partial_m_ptr),
        ctypes.c_void_p(partial_l_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_splits),
        ctypes.c_int64(head_dim),
        ctypes.c_int64(gate_stride1),
        ctypes.c_int64(gate_stride2),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)

def _launch_gate_reduce_batch(
    symbol: str,
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    rows: int,
    num_q_heads: int,
    num_splits: int,
    head_dim: int,
    gate_stride1: int,
    gate_stride2: int,
    *,
    stream: int,
    library: ctypes.CDLL,
    runtime: HipRuntime,
) -> None:
    reduce_gate = getattr(library, symbol)
    reduce_gate.argtypes = [
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
        ctypes.c_void_p,
    ]
    reduce_gate.restype = ctypes.c_int
    err = reduce_gate(
        ctypes.c_void_p(partial_out_ptr),
        ctypes.c_void_p(partial_m_ptr),
        ctypes.c_void_p(partial_l_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_splits),
        ctypes.c_int64(head_dim),
        ctypes.c_int64(gate_stride1),
        ctypes.c_int64(gate_stride2),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_split_context(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    spans: KVLiveSpans,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int,
    library: ctypes.CDLL,
    runtime: HipRuntime,
    symbol: str = _SYMBOL_SPLIT_CONTEXT,
) -> None:
    split = getattr(library, symbol)
    split.argtypes = [
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
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    split.restype = ctypes.c_int
    err = split(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(partial_out_ptr),
        ctypes.c_void_p(partial_m_ptr),
        ctypes.c_void_p(partial_l_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_int64(chunk_size),
        ctypes.c_int64(num_splits),
        ctypes.c_int64(block_size),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)

def _launch_split_context_batch(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    block_table_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int,
    library: ctypes.CDLL,
    runtime: HipRuntime,
    symbol: str,
) -> None:
    split = getattr(library, symbol)
    split.argtypes = [
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    split.restype = ctypes.c_int
    err = split(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(partial_out_ptr),
        ctypes.c_void_p(partial_m_ptr),
        ctypes.c_void_p(partial_l_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(chunk_size),
        ctypes.c_int64(num_splits),
        ctypes.c_int64(block_size),
        ctypes.c_int64(block_table_len),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _check_split_shape(
    spans: KVLiveSpans,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    _check_decode_shape(
        spans,
        chunk_size * num_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )
    _check_positive(chunk_size, "chunk_size")
    _check_positive(num_splits, "num_splits")
    if head_dim % 8 != 0:
        raise ValueError("split-K paged attention requires head_dim divisible by 8")
    if head_dim > 1024:
        raise ValueError("head_dim must fit in one reduce block")


def _check_qwen35_gqa_shape(
    spans: KVLiveSpans,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    _check_split_shape(spans, chunk_size, num_splits, block_size, num_q_heads, num_kv_heads, head_dim)
    if block_size != 256 or num_q_heads != 16 or num_kv_heads != 2 or head_dim != 256:
        raise ValueError("Qwen3.5 GQA split-K specialization requires block_size=256, num_q_heads=16, num_kv_heads=2, head_dim=256")

def _check_qwen35_gqa_batch_shape(
    spans: KVLiveSpans,
    rows: int,
    chunk_size: int,
    num_splits: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    block_table_len = _check_decode_batch_shape(
        spans,
        rows,
        chunk_size * num_splits,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )
    _check_positive(chunk_size, "chunk_size")
    _check_positive(num_splits, "num_splits")
    if block_size != 256 or num_q_heads != 16 or num_kv_heads != 2 or head_dim != 256:
        raise ValueError("Qwen3.5 GQA split-K specialization requires block_size=256, num_q_heads=16, num_kv_heads=2, head_dim=256")
    return block_table_len


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_qwen35_paged_attn_decode_kernels()
