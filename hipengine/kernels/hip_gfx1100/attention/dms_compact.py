"""Raw-pointer wrappers for the compact DMS kernel family (C2-7 device port).

Units of the streaming no-shadow DMS port: ``dms_extract_decision_bf16``
reads the borrowed decision neuron (last channel of the first query head of
each GQA group) from pre-RoPE Q and publishes per-KV-head eviction bits;
``dms_streaming_pack_bf16`` scatters the surviving prompt tokens into the
reserved compact extents; ``dms_append_decode_bf16`` advances one decode
step (strict parent keep-recompute + append, fail closed on overflow);
``dms_compact_attn_decode_bf16`` runs GQA decode attention over the dense
extents. The ``cpu_reference`` siblings are the registered strict fallbacks.
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("dms_compact.hip")
_OUTPUT_NAME = "dms_compact.so"
_SYMBOL_EXTRACT_DECISION = "hipengine_dms_extract_decision_bf16"
_SYMBOL_STREAMING_PACK = "hipengine_dms_streaming_pack_bf16"
_SYMBOL_APPEND_DECODE = "hipengine_dms_append_decode_bf16"
_SYMBOL_COMPACT_ATTN_DECODE = "hipengine_dms_compact_attn_decode_bf16"


def plan_dms_compact_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="dms_compact",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_dms_compact(
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
        family="dms_compact",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def dms_extract_decision_bf16(
    q_ptr: int,
    evict_ptr: int,
    alpha_scale: float,
    alpha_offset: float,
    tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Extract per-KV-head DMS eviction decisions from pre-RoPE Q (BF16).

    ``q`` is ``[tokens, q_heads, head_dim]`` BF16; ``evict`` is
    ``[tokens, kv_heads]`` uint8. The borrowed channel (first query head of
    each GQA group, last channel) is zeroed in place. The threshold
    arithmetic mirrors the CPU reference's float64 scalar comparisons.
    """
    if int(tokens) <= 0:
        raise ValueError("tokens must be positive")
    if int(kv_heads) <= 0:
        raise ValueError("kv_heads must be positive")
    if int(q_heads) <= 0 or int(q_heads) % int(kv_heads) != 0:
        raise ValueError("q_heads must be positive and divisible by kv_heads")
    if int(head_dim) <= 0:
        raise ValueError("head_dim must be positive")
    if not math.isfinite(float(alpha_scale)) or float(alpha_scale) == 0.0:
        raise ValueError("alpha_scale must be finite and non-zero")
    if not math.isfinite(float(alpha_offset)):
        raise ValueError("alpha_offset must be finite")

    library = library or build_dms_compact(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_EXTRACT_DECISION)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(q_ptr),
        ctypes.c_void_p(evict_ptr),
        ctypes.c_double(float(alpha_scale)),
        ctypes.c_double(float(alpha_offset)),
        ctypes.c_int64(tokens),
        ctypes.c_int64(q_heads),
        ctypes.c_int64(kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_void_p(stream),
    )
    if err != HIP_SUCCESS:
        raise RuntimeError(f"dms_extract_decision_bf16 failed with HIP error {err}")


def dms_streaming_pack_bf16(
    k_ptr: int,
    v_ptr: int,
    evict_ptr: int,
    base_offsets_ptr: int,
    range_capacity_ptr: int,
    live_counts_ptr: int,
    row_starts_ptr: int,
    row_tokens_ptr: int,
    k_slot_ptr: int,
    v_slot_ptr: int,
    token_positions_ptr: int,
    slot_evict_ptr: int,
    rows: int,
    heads: int,
    dim: int,
    window_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Scatter surviving prompt K/V rows straight into compact extents (BF16).

    Per-layer launch: ``k``/``v`` are ``[total_tokens, heads, dim]`` BF16
    (rows concatenated in order), ``evict`` is ``[total_tokens, heads]``
    uint8, ``row_starts``/``row_tokens`` are ``[rows]`` int32 (absolute
    position base and token count per row), and the per-(row, head) extent
    arrays are ``[rows, heads]`` int32. Surviving rows
    (``~evict | current - t <= window_size``) are scattered in token order
    into the extent at ``base_offsets``; ``live_counts``, per-slot
    ``token_positions`` (absolute) and ``evict_mask`` are published. Inputs
    are never retained (no dense shadow).

    Precondition: each extent's ``range_capacity`` covers its row's token
    count (admission reserves the worst case).
    """
    if int(rows) <= 0:
        raise ValueError("rows must be positive")
    if int(heads) <= 0:
        raise ValueError("heads must be positive")
    if int(dim) <= 0:
        raise ValueError("dim must be positive")
    if int(window_size) < 0:
        raise ValueError("window_size must be non-negative")

    library = library or build_dms_compact(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_STREAMING_PACK)
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
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(k_ptr),
        ctypes.c_void_p(v_ptr),
        ctypes.c_void_p(evict_ptr),
        ctypes.c_void_p(base_offsets_ptr),
        ctypes.c_void_p(range_capacity_ptr),
        ctypes.c_void_p(live_counts_ptr),
        ctypes.c_void_p(row_starts_ptr),
        ctypes.c_void_p(row_tokens_ptr),
        ctypes.c_void_p(k_slot_ptr),
        ctypes.c_void_p(v_slot_ptr),
        ctypes.c_void_p(token_positions_ptr),
        ctypes.c_void_p(slot_evict_ptr),
        ctypes.c_int(rows),
        ctypes.c_int(heads),
        ctypes.c_int(dim),
        ctypes.c_int(window_size),
        ctypes.c_void_p(stream),
    )
    if err != HIP_SUCCESS:
        raise RuntimeError(f"dms_streaming_pack_bf16 failed with HIP error {err}")


def dms_append_decode_bf16(
    k_new_ptr: int,
    v_new_ptr: int,
    evict_new_ptr: int,
    row_positions_ptr: int,
    base_offsets_ptr: int,
    range_capacity_ptr: int,
    live_counts_ptr: int,
    k_slot_ptr: int,
    v_slot_ptr: int,
    token_positions_ptr: int,
    slot_evict_ptr: int,
    status_ptr: int,
    rows: int,
    heads: int,
    dim: int,
    window_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Append one decode step to compact DMS extents (BF16).

    ``k_new``/``v_new`` are ``[heads, dim]`` BF16 for the new token, ``evict_new``
    is ``[heads]`` uint8, ``row_positions`` is ``[rows]`` int32 (the new
    token's absolute position per row), and the per-(row, head) extent arrays
    are ``[rows, heads]`` int32. Strict parent walk: keep = ~evict |
    p - pos <= window recomputed over the extent, new row appended.
    ``status`` (``[rows, heads]`` int32) is set to 1 when appending would
    overflow the extent (the host parent raises MemoryError without
    mutating state); the extent is left untouched in that case.
    """
    if int(rows) <= 0:
        raise ValueError("rows must be positive")
    if int(heads) <= 0:
        raise ValueError("heads must be positive")
    if int(dim) <= 0:
        raise ValueError("dim must be positive")
    if int(window_size) < 0:
        raise ValueError("window_size must be non-negative")

    library = library or build_dms_compact(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_APPEND_DECODE)
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
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(k_new_ptr),
        ctypes.c_void_p(v_new_ptr),
        ctypes.c_void_p(evict_new_ptr),
        ctypes.c_void_p(row_positions_ptr),
        ctypes.c_void_p(base_offsets_ptr),
        ctypes.c_void_p(range_capacity_ptr),
        ctypes.c_void_p(live_counts_ptr),
        ctypes.c_void_p(k_slot_ptr),
        ctypes.c_void_p(v_slot_ptr),
        ctypes.c_void_p(token_positions_ptr),
        ctypes.c_void_p(slot_evict_ptr),
        ctypes.c_void_p(status_ptr),
        ctypes.c_int(rows),
        ctypes.c_int(heads),
        ctypes.c_int(dim),
        ctypes.c_int(window_size),
        ctypes.c_void_p(stream),
    )
    if err != HIP_SUCCESS:
        raise RuntimeError(f"dms_append_decode_bf16 failed with HIP error {err}")


def dms_compact_attn_decode_bf16(
    q_ptr: int,
    k_slot_ptr: int,
    v_slot_ptr: int,
    base_offsets_ptr: int,
    live_counts_ptr: int,
    out_ptr: int,
    rows: int,
    q_heads: int,
    kv_heads: int,
    dim: int,
    scale: float,
    score_capacity: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """GQA decode attention over compact DMS extents (BF16 K/V, FP32 Q/out).

    ``q``/``out`` are ``[rows, q_heads, dim]`` FP32; ``k_slot``/``v_slot`` are
    slot-major ``[total_slots, dim]`` BF16 bits with each (row, kv_head) row
    dense in its extent at ``base_offsets`` for ``live_counts`` rows; the
    extent arrays are ``[rows, kv_heads]`` int32. ``score_capacity`` must
    cover the maximum live count (shared-memory sizing). live == 0 writes
    zeros; live == 1 is bit-exact (single-row softmax).
    """
    if int(rows) <= 0:
        raise ValueError("rows must be positive")
    if int(q_heads) <= 0:
        raise ValueError("q_heads must be positive")
    if int(kv_heads) <= 0:
        raise ValueError("kv_heads must be positive")
    if int(q_heads) % int(kv_heads) != 0:
        raise ValueError("q_heads must be a multiple of kv_heads for GQA")
    if int(dim) <= 0:
        raise ValueError("dim must be positive")
    if not (float(scale) > 0.0):
        raise ValueError("scale must be positive")
    if int(score_capacity) <= 0:
        raise ValueError("score_capacity must be positive")

    library = library or build_dms_compact(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_COMPACT_ATTN_DECODE)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(q_ptr),
        ctypes.c_void_p(k_slot_ptr),
        ctypes.c_void_p(v_slot_ptr),
        ctypes.c_void_p(base_offsets_ptr),
        ctypes.c_void_p(live_counts_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int(rows),
        ctypes.c_int(q_heads),
        ctypes.c_int(kv_heads),
        ctypes.c_int(dim),
        ctypes.c_float(scale),
        ctypes.c_int(score_capacity),
        ctypes.c_void_p(stream),
    )
    if err != HIP_SUCCESS:
        raise RuntimeError(f"dms_compact_attn_decode_bf16 failed with HIP error {err}")


def register_dms_compact_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "dms_extract_decision", "bf16", "corrected_mask"),
        dms_extract_decision_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dms_streaming_pack", "bf16", "count_rank_scatter"),
        dms_streaming_pack_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dms_append_decode", "bf16", "compact_append_evict"),
        dms_append_decode_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dms_compact_attn_decode", "bf16", "grouped_gqa"),
        dms_compact_attn_decode_bf16,
        replace=replace,
    )


register_dms_compact_kernels()
