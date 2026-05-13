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
_SYMBOL_CONTEXT = "hipengine_qwen35_paged_full_attn_decode_context_bf16_spans"
_SYMBOL_SPLIT_CONTEXT = "hipengine_qwen35_paged_full_attn_decode_split_k_context_bf16_spans"
_SYMBOL_SPLIT_WARP_CONTEXT = "hipengine_qwen35_paged_full_attn_decode_split_k_warp_context_bf16_spans"
_SYMBOL_SPLIT_GQA_CONTEXT = "hipengine_qwen35_paged_full_attn_decode_split_k_gqa_context_bf16_spans"
_SYMBOL_SPLIT_REDUCE = "hipengine_qwen35_paged_full_attn_decode_split_k_reduce_f32"
_SYMBOL_SPLIT_REDUCE_GATE_F32 = "hipengine_qwen35_paged_full_attn_decode_split_k_reduce_gate_f32"
_SYMBOL_SPLIT_REDUCE_GATE_BF16 = "hipengine_qwen35_paged_full_attn_decode_split_k_reduce_gate_bf16"


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

def register_qwen35_paged_attn_decode_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "w4_paro", "bf16_context_spans"),
        qwen35_paged_full_attn_decode_context_bf16_spans,
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
        KernelKey("hip_gfx1100", "paged_attn_decode", "w4_paro", "bf16_split_k_gate_f32_spans"),
        qwen35_paged_full_attn_decode_split_k_gate_f32_spans,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paged_attn_decode", "w4_paro", "bf16_split_k_gate_bf16_spans"),
        qwen35_paged_full_attn_decode_split_k_gate_bf16_spans,
        replace=replace,
    )


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

def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_qwen35_paged_attn_decode_kernels()
