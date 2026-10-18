"""Surya fp32 helper-kernel build + launch wrappers (see surya_ops.hip)."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hipengine.core.build import build_hip, plan_hip_build
from hipengine.core.build import BuildArtifact
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.kvcache.spans import KVLiveSpans

_SOURCE = Path(__file__).with_name("surya_ops.hip")
_OUTPUT_NAME = "surya_ops"

_P = ctypes.c_void_p
_F = ctypes.c_float
_I = ctypes.c_int64
_S = ctypes.c_void_p

# Surya text-decoder attention geometry. The fused decode kernel is compiled for
# exactly this GQA repeat and head width; other shapes fall back to the rocBLAS
# parent path in ``hipengine.runtime.surya``.
SURYA_DECODE_HEAD_DIM = 256
SURYA_DECODE_Q_PER_KV = 4
SURYA_SPAN_BLOCK_SIZE = 256
# The decode producer runs one block per ``(kv_head, context chunk)``, so its
# parallelism is ``num_kv_heads * num_splits`` and the chunk decides how much of
# the machine the launch fills.  Measured on gfx1151 over context lengths
# 170..16384, the optimum tracks a roughly constant split count rather than a
# constant chunk: chunk 1024 (grid 4) ran a 2048-token context at 0.96x the
# rocBLAS parent while chunk 32 (grid 128) ran it at 6.2x, and at 16384 tokens
# chunk 256 (grid 128) beat chunk 32 (grid 1024) because the split-K reduce then
# has 512 partials per head to combine.  Split alignment is one warp's worth of
# tokens so the lane-strided dim loop stays coherent; LDS is
# ``20 * chunk + 4224`` bytes, so every default stays far under the 64 KiB
# workgroup limit (the 16384-token default uses 9344 B).
SURYA_SPLIT_ALIGNMENT = 32
SURYA_TARGET_SPLITS = 64


def plan_surya_ops_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: str = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="surya_ops",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_surya_ops(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: str = "decode",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | None:
    return build_hip(
        sources=[_SOURCE],
        family="surya_ops",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


@dataclass(frozen=True)
class SuryaDenseSpanPlan:
    """Host-side description of Surya's dense uniform ``KVLiveSpans`` fill."""

    max_seq: int
    block_size: int
    block_table_len: int
    page_table: np.ndarray
    token_positions: np.ndarray
    evict_mask: np.ndarray
    chunk_size: int
    num_splits: int


def plan_surya_decode_splits(
    max_seq: int,
    *,
    chunk_size: int | None = None,
) -> tuple[int, int]:
    """Return the ``(chunk_size, num_splits)`` split-K plan for a context."""

    max_seq = int(max_seq)
    if max_seq <= 0:
        raise ValueError("max_seq must be positive")
    if chunk_size is None:
        ideal = (max_seq + SURYA_TARGET_SPLITS - 1) // SURYA_TARGET_SPLITS
        chunk = max(SURYA_SPLIT_ALIGNMENT, _round_up(ideal, SURYA_SPLIT_ALIGNMENT))
    else:
        chunk = int(chunk_size)
        if chunk <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk % SURYA_SPLIT_ALIGNMENT:
            raise ValueError(
                f"chunk_size must be a multiple of {SURYA_SPLIT_ALIGNMENT}"
            )
    num_splits = (max_seq + chunk - 1) // chunk
    return chunk, max(1, num_splits)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def plan_surya_dense_spans(
    max_seq: int,
    *,
    block_size: int = SURYA_SPAN_BLOCK_SIZE,
    chunk_size: int | None = None,
) -> SuryaDenseSpanPlan:
    """Build the identity page table, positions, mask, and split plan.

    The dense policy fills every span field uniformly rather than leaving
    ``token_positions``/``evict_mask`` null: the kernels honour them, so the
    default path exercises the ABI it claims to implement.
    """

    max_seq = int(max_seq)
    block_size = int(block_size)
    if max_seq <= 0:
        raise ValueError("max_seq must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    block_table_len = (max_seq + block_size - 1) // block_size
    resolved_chunk, num_splits = plan_surya_decode_splits(
        max_seq, chunk_size=chunk_size
    )
    return SuryaDenseSpanPlan(
        max_seq=max_seq,
        block_size=block_size,
        block_table_len=block_table_len,
        page_table=np.arange(block_table_len, dtype=np.int32),
        token_positions=np.arange(max_seq, dtype=np.int64),
        evict_mask=np.zeros(max_seq, dtype=np.bool_),
        chunk_size=resolved_chunk,
        num_splits=num_splits,
    )


def _require_nonzero(pointers: tuple[int, ...], names: tuple[str, ...]) -> None:
    for ptr, name in zip(pointers, names):
        if not int(ptr):
            raise ValueError(f"{name} pointer must be non-zero")


def _require_fp32_spans(spans: KVLiveSpans) -> None:
    if spans.storage_dtype != DType.FP32:
        raise ValueError(
            f"Surya KV spans require fp32 storage, got {spans.storage_dtype}"
        )


def surya_scatter_kv_f32_spans(
    src_ptr: int,
    dst_ptr: int,
    spans: KVLiveSpans,
    tokens: int,
    token_offset: int,
    block_size: int,
    kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Append fp32 K/V rows into the dense planes through ``KVLiveSpans``.

    ``src`` is ``(tokens, kv_heads * head_dim)``; the destination is the
    persistent ``(kv_heads, max_seq, head_dim)`` plane.  Logical token
    ``token_offset + i`` lands at the page-table slot for that index and is
    skipped when it is outside ``live_counts``, has a negative
    ``token_positions`` entry, or is marked in ``evict_mask``.
    """

    _require_nonzero((src_ptr, dst_ptr), ("src", "dst"))
    _require_fp32_spans(spans)
    tokens = int(tokens)
    token_offset = int(token_offset)
    block_size = int(block_size)
    kv_heads = int(kv_heads)
    head_dim = int(head_dim)
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if token_offset < 0:
        raise ValueError("token_offset must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if kv_heads <= 0:
        raise ValueError("kv_heads must be positive")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    max_seq = int(spans.max_live_count)
    block_table_len = int(spans.base_offsets.numel)
    if token_offset + tokens > max_seq:
        raise ValueError("token_offset + tokens exceeds the span capacity")

    runtime = runtime or get_hip_runtime()
    library = library or build_surya_ops(load=True)
    fn = _fn(library, "hipengine_surya_scatter_kv_f32_spans",
             [_P, _P, _P, _P, _P, _P, _I, _I, _I, _I, _I, _I, _I, _S])
    err = fn(
        _P(src_ptr), _P(dst_ptr),
        _P(spans.base_offsets.ptr), _P(spans.live_counts.ptr),
        _P(spans.token_positions.ptr) if spans.token_positions is not None else _P(0),
        _P(spans.evict_mask.ptr) if spans.evict_mask is not None else _P(0),
        _I(tokens), _I(token_offset), _I(kv_heads), _I(head_dim), _I(max_seq),
        _I(block_size), _I(block_table_len), _S(stream),
    )
    _check(err, runtime, "surya scatter_kv spans")


def surya_full_attn_decode_f32_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    spans: KVLiveSpans,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    chunk_size: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> tuple[int, int]:
    """Fused GQA split-K decode attention over ``KVLiveSpans``.

    Replaces the batched-SGEMM scores/softmax/AV chain with one producer launch
    plus one reduce launch.  Returns the resolved ``(chunk_size, num_splits)``
    so callers can record the plan they actually ran.
    """

    _require_nonzero(
        (query_ptr, key_cache_ptr, value_cache_ptr, out_ptr,
         partial_out_ptr, partial_m_ptr, partial_l_ptr),
        ("query", "key_cache", "value_cache", "out",
         "partial_out", "partial_m", "partial_l"),
    )
    _require_fp32_spans(spans)
    block_size = int(block_size)
    num_q_heads = int(num_q_heads)
    num_kv_heads = int(num_kv_heads)
    head_dim = int(head_dim)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    if num_q_heads != SURYA_DECODE_Q_PER_KV * num_kv_heads:
        raise ValueError(
            f"fused Surya decode requires GQA repeat {SURYA_DECODE_Q_PER_KV}, "
            f"got {num_q_heads} query heads over {num_kv_heads} KV heads"
        )
    if head_dim != SURYA_DECODE_HEAD_DIM:
        raise ValueError(
            f"fused Surya decode requires head_dim {SURYA_DECODE_HEAD_DIM}, got {head_dim}"
        )
    capacity = int(spans.max_live_count)
    block_table_len = int(spans.base_offsets.numel)
    resolved_chunk, num_splits = plan_surya_decode_splits(
        capacity, chunk_size=chunk_size
    )

    runtime = runtime or get_hip_runtime()
    library = library or build_surya_ops(load=True)
    producer = _fn(
        library, "hipengine_surya_full_attn_decode_split_k_f32_spans",
        [_P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P,
         _I, _I, _I, _I, _I, _I, _I, _I, _F, _S],
    )
    err = producer(
        _P(query_ptr), _P(key_cache_ptr), _P(value_cache_ptr),
        _P(partial_out_ptr), _P(partial_m_ptr), _P(partial_l_ptr),
        _P(spans.base_offsets.ptr), _P(spans.live_counts.ptr),
        _P(spans.token_positions.ptr) if spans.token_positions is not None else _P(0),
        _P(spans.evict_mask.ptr) if spans.evict_mask is not None else _P(0),
        _P(spans.row_positions.ptr) if spans.row_positions is not None else _P(0),
        _I(capacity), _I(resolved_chunk), _I(num_splits), _I(block_size),
        _I(block_table_len), _I(num_q_heads), _I(num_kv_heads), _I(head_dim),
        _F(float(scale)), _S(stream),
    )
    _check(err, runtime, "surya decode split_k spans")

    reduce = _fn(
        library, "hipengine_surya_full_attn_decode_split_k_reduce_f32",
        [_P, _P, _P, _P, _I, _I, _I, _S],
    )
    err = reduce(
        _P(partial_out_ptr), _P(partial_m_ptr), _P(partial_l_ptr), _P(out_ptr),
        _I(num_q_heads), _I(num_splits), _I(head_dim), _S(stream),
    )
    _check(err, runtime, "surya decode split_k reduce")
    return resolved_chunk, num_splits


def surya_split_qgate_f32(
    src_ptr: int,
    q_ptr: int,
    gate_ptr: int,
    tokens: int,
    heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    runtime = runtime or get_hip_runtime()
    library = library or build_surya_ops(load=True)
    fn = _fn(library, "hipengine_surya_split_qgate_f32", [_P, _P, _P, _I, _I, _I, _S])
    err = fn(_P(src_ptr), _P(q_ptr), _P(gate_ptr), _I(tokens), _I(heads), _I(head_dim), _S(stream))
    _check(err, runtime, "surya split_qgate")


def surya_gdn_l2norm_f32(
    src_ptr: int,
    q_ptr: int,
    k_ptr: int,
    q_scale: float,
    tokens: int,
    heads: int,
    head_dim: int,
    src_stride: int,
    k_offset: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    runtime = runtime or get_hip_runtime()
    library = library or build_surya_ops(load=True)
    fn = _fn(library, "hipengine_surya_gdn_l2norm_f32",
             [_P, _P, _P, _F, _I, _I, _I, _I, _I, _S])
    err = fn(_P(src_ptr), _P(q_ptr), _P(k_ptr), _F(q_scale), _I(tokens),
             _I(heads), _I(head_dim), _I(src_stride), _I(k_offset), _S(stream))
    _check(err, runtime, "surya gdn_l2norm")


def surya_rmsnorm_f32(
    x_ptr: int,
    w_ptr: int,
    out_ptr: int,
    rows: int,
    hidden: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    runtime = runtime or get_hip_runtime()
    library = library or build_surya_ops(load=True)
    fn = _fn(library, "hipengine_surya_rmsnorm_f32", [_P, _P, _P, _I, _I, _F, _S])
    err = fn(_P(x_ptr), _P(w_ptr), _P(out_ptr), _I(rows), _I(hidden), _F(eps), _S(stream))
    _check(err, runtime, "surya rmsnorm")


def surya_scatter_kv_f32(
    src_ptr: int,
    dst_ptr: int,
    tokens: int,
    token_offset: int,
    kv_heads: int,
    head_dim: int,
    max_seq: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    runtime = runtime or get_hip_runtime()
    library = library or build_surya_ops(load=True)
    fn = _fn(library, "hipengine_surya_scatter_kv_f32",
             [_P, _P, _I, _I, _I, _I, _I, _S])
    err = fn(_P(src_ptr), _P(dst_ptr), _I(tokens), _I(token_offset),
             _I(kv_heads), _I(head_dim), _I(max_seq), _S(stream))
    _check(err, runtime, "surya scatter_kv")


def surya_causal_mask_scale_f32(
    scores_ptr: int,
    scale: float,
    heads: int,
    queries: int,
    head_stride: int,
    query_offset: int = 0,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Mask and scale a causal score block.

    ``queries`` is the number of query rows in ``scores`` and ``head_stride``
    is both the key count and the row stride; entry ``(h, i, j)`` is dropped to
    ``-inf`` when ``j > query_offset + i``. A query-row tile passes the index
    of its first query as ``query_offset`` so the mask stays absolute.
    """

    runtime = runtime or get_hip_runtime()
    library = library or build_surya_ops(load=True)
    fn = _fn(library, "hipengine_surya_causal_mask_scale_f32",
             [_P, _F, _I, _I, _I, _I, _S])
    err = fn(_P(scores_ptr), _F(scale), _I(heads), _I(queries), _I(head_stride),
             _I(query_offset), _S(stream))
    _check(err, runtime, "surya causal_mask_scale")


def _fn(library: ctypes.CDLL, symbol: str, argtypes: list) -> ctypes._FuncPtr:
    fn = getattr(library, symbol, None)
    if fn is None:
        raise RuntimeError(f"missing symbol {symbol}")
    fn.argtypes = argtypes
    fn.restype = ctypes.c_int
    return fn


def _check(err: int, runtime: HipRuntime, what: str) -> None:
    if err != 0:
        raise RuntimeError(f"{what} failed: {err} ({runtime.last_error_message() if hasattr(runtime, 'last_error_message') else 'see hipGetLastError'})")
