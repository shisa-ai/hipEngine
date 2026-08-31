"""Raw-pointer wrappers for strict Qwen4Exp QSA control primitives."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.dtype import DType
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register
from hipengine.kvcache import KVLiveSpans

_SOURCE = Path(__file__).with_name("qwen4_exp_qsa.hip")
_OUTPUT_NAME = "qwen4_exp_qsa.so"
_ARGS_SPLIT = (
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
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_void_p,
)
_ARGS_SPLIT_ROWS = (ctypes.c_void_p,) * 8 + (ctypes.c_int64,) * 5 + (
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_void_p,
)
_ARGS_NORM = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_void_p,
)
_ARGS_NORM_ROWS = (ctypes.c_void_p,) * 4 + (ctypes.c_int64,) * 4 + (
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_void_p,
)
_ARGS_SCATTER_INDEX = (ctypes.c_void_p,) * 3 + (ctypes.c_int64,) * 4 + (
    ctypes.c_void_p,
)
_ARGS_GATE = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGS_SPARSE_ATTN = (
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
)
_ARGS_SPARSE_ATTN_ROWS = (ctypes.c_void_p,) * 7 + (ctypes.c_int64,) * 7 + (
    ctypes.c_float,
    ctypes.c_void_p,
)
_ARGS_POOL = (
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
    ctypes.c_float,
    ctypes.c_void_p,
)
_ARGS_SCORE = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGS_TOPK_EXPAND_ROWS = (
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
)
_ARGS_TOPK_EXPAND = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGS_SELECT = (
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
)


def plan_qwen4_exp_qsa_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="qwen4_exp_qsa",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_qwen4_exp_qsa(
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
        family="qwen4_exp_qsa",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def qwen4_exp_qsa_split_norm_rope_f32(
    q_projected_ptr: int,
    key_ptr: int,
    q_weight_ptr: int,
    k_weight_ptr: int,
    position_ptr: int,
    query_out_ptr: int,
    key_out_ptr: int,
    gate_out_ptr: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    theta: float,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if num_q_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
        raise ValueError("head counts and head_dim must be positive")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be positive, even, and <= head_dim")
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_qsa_split_norm_rope_f32",
        _ARGS_SPLIT,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            q_projected_ptr, key_ptr, q_weight_ptr, k_weight_ptr, position_ptr,
            query_out_ptr, key_out_ptr, gate_out_ptr,
            num_q_heads, num_kv_heads, head_dim, rotary_dim,
            float(theta), float(eps), stream,
        ),
    )


def qwen4_exp_qsa_split_norm_rope_rows_f32(
    q_projected_ptr: int,
    key_ptr: int,
    q_weight_ptr: int,
    k_weight_ptr: int,
    positions_ptr: int,
    query_out_ptr: int,
    key_out_ptr: int,
    gate_out_ptr: int,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    theta: float,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply strict Q/K norm, split-half RoPE, and gate split to token rows."""

    if rows <= 0 or num_q_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
        raise ValueError("rows, head counts, and head_dim must be positive")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be positive, even, and <= head_dim")
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_qsa_split_norm_rope_rows_f32",
        _ARGS_SPLIT_ROWS,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            q_projected_ptr, key_ptr, q_weight_ptr, k_weight_ptr, positions_ptr,
            query_out_ptr, key_out_ptr, gate_out_ptr, rows, num_q_heads,
            num_kv_heads, head_dim, rotary_dim, float(theta), float(eps), stream,
        ),
    )


def qwen4_exp_qsa_norm_rope_f32(
    input_ptr: int,
    weight_ptr: int,
    position_ptr: int,
    output_ptr: int,
    heads: int,
    head_dim: int,
    rotary_dim: int,
    theta: float,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """RMS-normalize and split-half partial-RoPE raw QSA index queries."""

    if heads <= 0 or head_dim <= 0:
        raise ValueError("heads and head_dim must be positive")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be positive, even, and <= head_dim")
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_qsa_norm_rope_f32",
        _ARGS_NORM,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            input_ptr,
            weight_ptr,
            position_ptr,
            output_ptr,
            heads,
            head_dim,
            rotary_dim,
            float(theta),
            float(eps),
            stream,
        ),
    )


def qwen4_exp_qsa_norm_rope_rows_f32(
    input_ptr: int,
    weight_ptr: int,
    positions_ptr: int,
    output_ptr: int,
    rows: int,
    heads: int,
    head_dim: int,
    rotary_dim: int,
    theta: float,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """RMS-normalize and split-half partial-RoPE QSA index-query rows."""

    if rows <= 0 or heads <= 0 or head_dim <= 0:
        raise ValueError("rows, heads, and head_dim must be positive")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be positive, even, and <= head_dim")
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_qsa_norm_rope_rows_f32",
        _ARGS_NORM_ROWS,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            input_ptr, weight_ptr, positions_ptr, output_ptr, rows, heads,
            head_dim, rotary_dim, float(theta), float(eps), stream,
        ),
    )


def qwen4_exp_qsa_split_norm_mrope_f32(
    q_projected_ptr: int, key_ptr: int, q_weight_ptr: int, k_weight_ptr: int,
    positions_ptr: int, query_out_ptr: int, key_out_ptr: int, gate_out_ptr: int,
    num_q_heads: int, num_kv_heads: int, head_dim: int, rotary_dim: int,
    theta: float, eps: float = 1e-6, *, stream: int = 0,
    library: ctypes.CDLL | None = None, runtime: HipRuntime | None = None,
) -> None:
    """Apply strict interleaved T/H/W MRoPE to one QSA row."""
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_qwen4_exp_qsa_split_norm_mrope_f32",
        _ARGS_SPLIT, ctypes.c_int,
    )
    _check_launch(runtime, fn(
        q_projected_ptr, key_ptr, q_weight_ptr, k_weight_ptr, positions_ptr,
        query_out_ptr, key_out_ptr, gate_out_ptr, num_q_heads, num_kv_heads,
        head_dim, rotary_dim, float(theta), float(eps), stream,
    ))


def qwen4_exp_qsa_norm_mrope_f32(
    input_ptr: int, weight_ptr: int, positions_ptr: int, output_ptr: int,
    heads: int, head_dim: int, rotary_dim: int, theta: float,
    eps: float = 1e-6, *, stream: int = 0,
    library: ctypes.CDLL | None = None, runtime: HipRuntime | None = None,
) -> None:
    """Apply strict interleaved T/H/W MRoPE to index queries for one row."""
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_qwen4_exp_qsa_norm_mrope_f32",
        _ARGS_NORM, ctypes.c_int,
    )
    _check_launch(runtime, fn(
        input_ptr, weight_ptr, positions_ptr, output_ptr, heads, head_dim,
        rotary_dim, float(theta), float(eps), stream,
    ))


def qwen4_exp_qsa_split_norm_mrope_rows_f32(
    q_projected_ptr: int, key_ptr: int, q_weight_ptr: int, k_weight_ptr: int,
    positions_ptr: int, query_out_ptr: int, key_out_ptr: int, gate_out_ptr: int,
    rows: int, num_q_heads: int, num_kv_heads: int, head_dim: int,
    rotary_dim: int, theta: float, eps: float = 1e-6, *, stream: int = 0,
    library: ctypes.CDLL | None = None, runtime: HipRuntime | None = None,
) -> None:
    """Apply strict interleaved T/H/W MRoPE to QSA prompt rows."""
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_qwen4_exp_qsa_split_norm_mrope_rows_f32",
        _ARGS_SPLIT_ROWS, ctypes.c_int,
    )
    _check_launch(runtime, fn(
        q_projected_ptr, key_ptr, q_weight_ptr, k_weight_ptr, positions_ptr,
        query_out_ptr, key_out_ptr, gate_out_ptr, rows, num_q_heads,
        num_kv_heads, head_dim, rotary_dim, float(theta), float(eps), stream,
    ))


def qwen4_exp_qsa_norm_mrope_rows_f32(
    input_ptr: int, weight_ptr: int, positions_ptr: int, output_ptr: int,
    rows: int, heads: int, head_dim: int, rotary_dim: int, theta: float,
    eps: float = 1e-6, *, stream: int = 0,
    library: ctypes.CDLL | None = None, runtime: HipRuntime | None = None,
) -> None:
    """Apply strict interleaved T/H/W MRoPE to index-query prompt rows."""
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_qwen4_exp_qsa_norm_mrope_rows_f32",
        _ARGS_NORM_ROWS, ctypes.c_int,
    )
    _check_launch(runtime, fn(
        input_ptr, weight_ptr, positions_ptr, output_ptr, rows, heads,
        head_dim, rotary_dim, float(theta), float(eps), stream,
    ))


def qwen4_exp_qsa_gate_context_f32(
    context_ptr: int,
    gate_ptr: int,
    output_ptr: int,
    elements: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if elements <= 0:
        raise ValueError("elements must be positive")
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_qsa_gate_context_f32",
        _ARGS_GATE,
        ctypes.c_int,
    )
    _check_launch(runtime, fn(context_ptr, gate_ptr, output_ptr, elements, stream))


def qwen4_exp_qsa_sparse_attention_paged_bf16_f32(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    selected_positions_ptr: int,
    output_ptr: int,
    spans: KVLiveSpans,
    *,
    selected_count: int,
    block_size: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    scale: float | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if spans.spans_mode != "uniform" or spans.storage_dtype != DType.BF16:
        raise ValueError("sparse QSA attention requires uniform BF16 KVLiveSpans")
    if selected_count <= 0 or block_size <= 0 or query_heads <= 0 or kv_heads <= 0:
        raise ValueError("selected_count, block_size, and head counts must be positive")
    if query_heads % kv_heads or head_dim <= 0 or head_dim > 256:
        raise ValueError("invalid sparse QSA GQA geometry")
    attention_scale = head_dim ** -0.5 if scale is None else float(scale)
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_qsa_sparse_attention_paged_bf16_f32",
        _ARGS_SPARSE_ATTN,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            query_ptr, key_cache_ptr, value_cache_ptr, selected_positions_ptr,
            spans.base_offsets.ptr, output_ptr, selected_count, block_size,
            spans.base_offsets.numel, query_heads, kv_heads, head_dim,
            attention_scale, stream,
        ),
    )


def qwen4_exp_qsa_sparse_attention_paged_bf16_wave32_f32(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    selected_positions_ptr: int,
    output_ptr: int,
    spans: KVLiveSpans,
    *,
    selected_count: int,
    block_size: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    scale: float | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Production H128 sparse attention using one barrier-free wave32 per Q head."""

    if spans.spans_mode != "uniform" or spans.storage_dtype != DType.BF16:
        raise ValueError("sparse QSA attention requires uniform BF16 KVLiveSpans")
    if selected_count <= 0 or block_size <= 0 or query_heads <= 0 or kv_heads <= 0:
        raise ValueError("selected_count, block_size, and head counts must be positive")
    if query_heads % kv_heads or head_dim != 128:
        raise ValueError("wave32 sparse QSA requires divisible GQA with head_dim=128")
    attention_scale = head_dim ** -0.5 if scale is None else float(scale)
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_qsa_sparse_attention_paged_bf16_wave32_f32",
        _ARGS_SPARSE_ATTN,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            query_ptr, key_cache_ptr, value_cache_ptr, selected_positions_ptr,
            spans.base_offsets.ptr, output_ptr, selected_count, block_size,
            spans.base_offsets.numel, query_heads, kv_heads, head_dim,
            attention_scale, stream,
        ),
    )


def qwen4_exp_qsa_sparse_attention_paged_bf16_wave8_contiguous_h256_f32(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    selected_positions_ptr: int,
    output_ptr: int,
    spans: KVLiveSpans,
    *,
    selected_count: int,
    block_size: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    scale: float | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """T1 H256 sparse attention using eight contiguous token-chunk waves per head."""

    if spans.spans_mode != "uniform" or spans.storage_dtype != DType.BF16:
        raise ValueError("sparse QSA attention requires uniform BF16 KVLiveSpans")
    if selected_count <= 0 or block_size <= 0 or query_heads <= 0 or kv_heads <= 0:
        raise ValueError("selected_count, block_size, and head counts must be positive")
    if query_heads % kv_heads or head_dim != 256:
        raise ValueError("H256 wave8 QSA requires divisible GQA with head_dim=256")
    attention_scale = head_dim ** -0.5 if scale is None else float(scale)
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_qsa_sparse_attention_paged_bf16_wave8_contiguous_h256_f32",
        _ARGS_SPARSE_ATTN,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            query_ptr, key_cache_ptr, value_cache_ptr, selected_positions_ptr,
            spans.base_offsets.ptr, output_ptr, selected_count, block_size,
            spans.base_offsets.numel, query_heads, kv_heads, head_dim,
            attention_scale, stream,
        ),
    )


def qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    selected_positions_ptr: int,
    selected_counts_ptr: int,
    output_ptr: int,
    spans: KVLiveSpans,
    *,
    rows: int,
    selected_stride: int,
    block_size: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    scale: float | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run variable-selection QSA rows over one shared paged BF16 K/V owner."""

    if spans.spans_mode != "uniform" or spans.storage_dtype != DType.BF16:
        raise ValueError("sparse QSA attention requires uniform BF16 KVLiveSpans")
    if rows <= 0 or selected_stride <= 0 or block_size <= 0:
        raise ValueError("rows, selected_stride, and block_size must be positive")
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query heads must be divisible by positive KV heads")
    if head_dim <= 0 or head_dim > 256:
        raise ValueError("head_dim must be in 1..256")
    if spans.live_counts.numel != rows or spans.base_offsets.numel % rows:
        raise ValueError("row QSA spans must provide one table and live count per row")
    block_table_len = spans.base_offsets.numel // rows
    attention_scale = head_dim ** -0.5 if scale is None else float(scale)
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32",
        _ARGS_SPARSE_ATTN_ROWS,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            query_ptr, key_cache_ptr, value_cache_ptr, selected_positions_ptr,
            selected_counts_ptr, spans.base_offsets.ptr, output_ptr, rows,
            selected_stride, block_size, block_table_len, query_heads, kv_heads,
            head_dim, attention_scale, stream,
        ),
    )


def qwen4_exp_qsa_sparse_attention_paged_bf16_rows_wave32_f32(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    selected_positions_ptr: int,
    selected_counts_ptr: int,
    output_ptr: int,
    spans: KVLiveSpans,
    *,
    rows: int,
    selected_stride: int,
    block_size: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    scale: float | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run production H128 wave32 QSA over variable-selection prompt rows."""

    if spans.spans_mode != "uniform" or spans.storage_dtype != DType.BF16:
        raise ValueError("sparse QSA attention requires uniform BF16 KVLiveSpans")
    if rows <= 0 or selected_stride <= 0 or block_size <= 0:
        raise ValueError("rows, selected_stride, and block_size must be positive")
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query heads must be divisible by positive KV heads")
    if head_dim != 128:
        raise ValueError("wave32 sparse QSA requires head_dim=128")
    if spans.live_counts.numel != rows or spans.base_offsets.numel % rows:
        raise ValueError("row QSA spans must provide one table and live count per row")
    block_table_len = spans.base_offsets.numel // rows
    attention_scale = head_dim ** -0.5 if scale is None else float(scale)
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_qsa_sparse_attention_paged_bf16_rows_wave32_f32",
        _ARGS_SPARSE_ATTN_ROWS,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            query_ptr, key_cache_ptr, value_cache_ptr, selected_positions_ptr,
            selected_counts_ptr, spans.base_offsets.ptr, output_ptr, rows,
            selected_stride, block_size, block_table_len, query_heads, kv_heads,
            head_dim, attention_scale, stream,
        ),
    )


def qwen4_exp_qsa_scatter_index_keys_f32(
    source_ptr: int,
    destination_ptr: int,
    block_table_ptr: int,
    start_position: int,
    rows: int,
    block_size: int,
    index_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Scatter one contiguous index-key chunk through the paged block table."""

    if start_position < 0 or rows <= 0 or block_size <= 0 or index_dim <= 0:
        raise ValueError(
            "start_position must be nonnegative and rows/block_size/index_dim positive"
        )
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_qsa_scatter_index_keys_f32",
        _ARGS_SCATTER_INDEX,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            source_ptr,
            destination_ptr,
            block_table_ptr,
            start_position,
            rows,
            block_size,
            index_dim,
            stream,
        ),
    )


def qwen4_exp_qsa_pool_norm_rope_f32(
    raw_keys_ptr: int,
    member_indices_ptr: int,
    block_starts_ptr: int,
    weight_ptr: int,
    output_ptr: int,
    blocks: int,
    ratio: int,
    index_dim: int,
    rotary_dim: int,
    theta: float,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Pool complete raw index keys, normalize, and partial-RoPE at block starts."""

    if blocks <= 0 or ratio <= 0 or index_dim <= 0:
        raise ValueError("blocks, ratio, and index_dim must be positive")
    if rotary_dim <= 0 or rotary_dim > index_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be positive, even, and <= index_dim")
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_qsa_pool_norm_rope_f32",
        _ARGS_POOL,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            raw_keys_ptr,
            member_indices_ptr,
            block_starts_ptr,
            weight_ptr,
            output_ptr,
            blocks,
            ratio,
            index_dim,
            rotary_dim,
            float(theta),
            float(eps),
            stream,
        ),
    )


def qwen4_exp_qsa_score_f32(
    queries_ptr: int,
    pooled_keys_ptr: int,
    scores_ptr: int,
    query_count: int,
    blocks: int,
    heads: int,
    index_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Compute ReLU-per-index-head QSA block scores."""

    if query_count <= 0 or blocks <= 0 or heads <= 0 or index_dim <= 0:
        raise ValueError("query_count, blocks, heads, and index_dim must be positive")
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_qsa_score_f32",
        _ARGS_SCORE,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            queries_ptr,
            pooled_keys_ptr,
            scores_ptr,
            query_count,
            blocks,
            heads,
            index_dim,
            stream,
        ),
    )


def qwen4_exp_qsa_topk_expand_f32_i64(
    scores_ptr: int,
    selected_positions_ptr: int,
    selected_count_ptr: int,
    blocks: int,
    query_position: int,
    ratio: int,
    budget: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """GPU exact top-k with lower-index ties and sorted token expansion."""

    if blocks <= 0 or ratio <= 0 or budget <= 0 or budget > blocks:
        raise ValueError("blocks, ratio, and budget must satisfy blocks >= budget > 0")
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_qsa_topk_expand_f32_i64",
        _ARGS_TOPK_EXPAND,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            scores_ptr,
            selected_positions_ptr,
            selected_count_ptr,
            blocks,
            query_position,
            ratio,
            budget,
            stream,
        ),
    )


def qwen4_exp_qsa_topk_expand_rows_f32_i64(
    scores_ptr: int,
    query_positions_ptr: int,
    selected_positions_ptr: int,
    selected_counts_ptr: int,
    rows: int,
    score_stride: int,
    output_stride: int,
    ratio: int,
    budget: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Batched exact top-k over variable visible prefixes."""

    if rows <= 0 or score_stride < budget or ratio <= 0 or budget <= 0:
        raise ValueError("invalid batched top-k rows, score stride, ratio, or budget")
    if output_stride < budget * ratio + ratio - 1:
        raise ValueError("batched top-k output stride is too small")
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_qsa_topk_expand_rows_f32_i64",
        _ARGS_TOPK_EXPAND_ROWS,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            scores_ptr,
            query_positions_ptr,
            selected_positions_ptr,
            selected_counts_ptr,
            rows,
            score_stride,
            output_stride,
            ratio,
            budget,
            stream,
        ),
    )


def qwen4_exp_qsa_select_blocks_f32_i64(
    scores_ptr: int,
    block_starts_ptr: int,
    query_positions_ptr: int,
    selected_starts_ptr: int,
    selected_counts_ptr: int,
    query_count: int,
    blocks: int,
    ratio: int,
    budget: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Strict deterministic complete-block selection with lower-start tie break."""

    if query_count <= 0 or blocks <= 0 or ratio <= 0 or budget <= 0:
        raise ValueError("query_count, blocks, ratio, and budget must be positive")
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_qsa_select_blocks_f32_i64",
        _ARGS_SELECT,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            scores_ptr,
            block_starts_ptr,
            query_positions_ptr,
            selected_starts_ptr,
            selected_counts_ptr,
            query_count,
            blocks,
            ratio,
            budget,
            stream,
        ),
    )


def register_qwen4_exp_qsa_kernels(*, replace: bool = True) -> None:
    registrations = {
        KernelKey(
            "hip_gfx1100",
            "qsa_split_norm_rope",
            "f32",
            "strict",
        ): qwen4_exp_qsa_split_norm_rope_f32,
        KernelKey(
            "hip_gfx1100",
            "qsa_split_norm_rope",
            "f32",
            "strict_rows",
        ): qwen4_exp_qsa_split_norm_rope_rows_f32,
        KernelKey(
            "hip_gfx1100",
            "qsa_norm_rope",
            "f32",
            "strict",
        ): qwen4_exp_qsa_norm_rope_f32,
        KernelKey(
            "hip_gfx1100",
            "qsa_norm_rope",
            "f32",
            "strict_rows",
        ): qwen4_exp_qsa_norm_rope_rows_f32,
        KernelKey(
            "hip_gfx1100", "qsa_split_norm_rope", "f32", "strict_mrope"
        ): qwen4_exp_qsa_split_norm_mrope_f32,
        KernelKey(
            "hip_gfx1100", "qsa_split_norm_rope", "f32", "strict_mrope_rows"
        ): qwen4_exp_qsa_split_norm_mrope_rows_f32,
        KernelKey(
            "hip_gfx1100", "qsa_norm_rope", "f32", "strict_mrope"
        ): qwen4_exp_qsa_norm_mrope_f32,
        KernelKey(
            "hip_gfx1100", "qsa_norm_rope", "f32", "strict_mrope_rows"
        ): qwen4_exp_qsa_norm_mrope_rows_f32,
        KernelKey(
            "hip_gfx1100",
            "qsa_gate_context",
            "f32",
            "strict",
        ): qwen4_exp_qsa_gate_context_f32,
        KernelKey(
            "hip_gfx1100",
            "qsa_sparse_attention",
            "bf16_kv",
            "strict_spans",
        ): qwen4_exp_qsa_sparse_attention_paged_bf16_f32,
        KernelKey(
            "hip_gfx1100",
            "qsa_sparse_attention",
            "bf16_kv",
            "strict_rows_spans",
        ): qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32,
        KernelKey(
            "hip_gfx1100",
            "qsa_sparse_attention",
            "bf16_kv",
            "production_wave32_h128_spans",
        ): qwen4_exp_qsa_sparse_attention_paged_bf16_wave32_f32,
        KernelKey(
            "hip_gfx1100",
            "qsa_sparse_attention",
            "bf16_kv",
            "production_wave8_contiguous_h256_spans",
        ): qwen4_exp_qsa_sparse_attention_paged_bf16_wave8_contiguous_h256_f32,
        KernelKey(
            "hip_gfx1100",
            "qsa_sparse_attention",
            "bf16_kv",
            "production_rows_wave32_h128_spans",
        ): qwen4_exp_qsa_sparse_attention_paged_bf16_rows_wave32_f32,
        KernelKey(
            "hip_gfx1100",
            "qsa_index_append",
            "f32",
            "strict_rows_paged",
        ): qwen4_exp_qsa_scatter_index_keys_f32,
        KernelKey(
            "hip_gfx1100",
            "qsa_pool_norm_rope",
            "f32",
            "strict",
        ): qwen4_exp_qsa_pool_norm_rope_f32,
        KernelKey(
            "hip_gfx1100",
            "qsa_index_score",
            "f32",
            "strict",
        ): qwen4_exp_qsa_score_f32,
        KernelKey(
            "hip_gfx1100",
            "qsa_select_blocks",
            "f32_i64",
            "strict",
        ): qwen4_exp_qsa_select_blocks_f32_i64,
        KernelKey(
            "hip_gfx1100",
            "qsa_select_blocks",
            "f32_i64",
            "strict_device_expand",
        ): qwen4_exp_qsa_topk_expand_f32_i64,
        KernelKey(
            "hip_gfx1100",
            "qsa_select_blocks",
            "f32_i64",
            "strict_device_expand_rows",
        ): qwen4_exp_qsa_topk_expand_rows_f32_i64,
    }
    for key, function in registrations.items():
        register(key, function, replace=replace)


def _check_launch(runtime: HipRuntime, error: int) -> None:
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


register_qwen4_exp_qsa_kernels()


__all__ = [
    "build_qwen4_exp_qsa",
    "plan_qwen4_exp_qsa_build",
    "qwen4_exp_qsa_gate_context_f32",
    "qwen4_exp_qsa_norm_rope_f32",
    "qwen4_exp_qsa_norm_rope_rows_f32",
    "qwen4_exp_qsa_norm_mrope_f32",
    "qwen4_exp_qsa_norm_mrope_rows_f32",
    "qwen4_exp_qsa_pool_norm_rope_f32",
    "qwen4_exp_qsa_score_f32",
    "qwen4_exp_qsa_scatter_index_keys_f32",
    "qwen4_exp_qsa_split_norm_rope_f32",
    "qwen4_exp_qsa_split_norm_rope_rows_f32",
    "qwen4_exp_qsa_split_norm_mrope_f32",
    "qwen4_exp_qsa_split_norm_mrope_rows_f32",
    "qwen4_exp_qsa_select_blocks_f32_i64",
    "qwen4_exp_qsa_topk_expand_f32_i64",
    "qwen4_exp_qsa_topk_expand_rows_f32_i64",
    "qwen4_exp_qsa_sparse_attention_paged_bf16_f32",
    "qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32",
    "qwen4_exp_qsa_sparse_attention_paged_bf16_wave32_f32",
    "qwen4_exp_qsa_sparse_attention_paged_bf16_wave8_contiguous_h256_f32",
    "qwen4_exp_qsa_sparse_attention_paged_bf16_rows_wave32_f32",
    "register_qwen4_exp_qsa_kernels",
]
