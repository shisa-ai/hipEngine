"""Compact INT8 DMS device writers and bounded split-K attention."""
from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.kernels.registry import KernelKey, register


def build_dms_compact_int8(**kwargs):
    directory = Path(__file__).parent
    return build_hip(sources=[directory / "dms_compact.hip", directory / "dms_compact_int8.hip"],
                     family="dms_compact_int8", profile="decode",
                     output_name="dms_compact_int8.so", **kwargs)


def _call(symbol, pointers, scalars, *, stream, library):
    library = library if library is not None else build_dms_compact_int8(load=True)
    if any(int(p) <= 0 for p in pointers):
        raise ValueError("DMS INT8 requires non-null device pointers")
    fn = getattr(library, symbol)
    args = [ctypes.c_void_p(int(p)) for p in pointers] + scalars + [ctypes.c_void_p(stream)]
    fn.argtypes = [type(x) for x in args]
    fn.restype = ctypes.c_int
    error = fn(*args)
    if error:
        raise RuntimeError(f"{symbol} failed with HIP error {error}")


def dms_streaming_pack_int8(k, v, evict, base, capacity, live, starts, tokens,
                            ko, vo, positions, flags, rows, heads, dim, window,
                            *, k_scale_ptr, v_scale_ptr, stream=0, library=None, runtime=None):
    if min(rows, heads, dim) <= 0 or window < 0:
        raise ValueError("DMS INT8 pack dimensions must be positive and window nonnegative")
    _call("hipengine_dms_streaming_pack_int8",
          [k, v, evict, base, capacity, live, starts, tokens, ko, vo, positions, flags,
           k_scale_ptr, v_scale_ptr],
          [ctypes.c_int(x) for x in (rows, heads, dim, window)], stream=stream, library=library)


def dms_append_decode_int8(k, v, evict, row_positions, base, capacity, live,
                          ko, vo, positions, flags, status, rows, heads, dim, window,
                          *, k_scale_ptr, v_scale_ptr, stream=0, library=None, runtime=None):
    if min(rows, heads, dim) <= 0 or window < 0:
        raise ValueError("DMS INT8 append dimensions must be positive and window nonnegative")
    _call("hipengine_dms_append_decode_int8",
          [k, v, evict, row_positions, base, capacity, live, ko, vo, positions, flags, status,
           k_scale_ptr, v_scale_ptr],
          [ctypes.c_int(x) for x in (rows, heads, dim, window)], stream=stream, library=library)


def dms_compact_attn_decode_splitk_int8(q, k, v, base, live, po, pm, pl, out,
                                       rows, q_heads, kv_heads, dim, scale, chunk, splits,
                                       *, k_scale_ptr, v_scale_ptr, stream=0,
                                       library=None, runtime=None):
    if min(rows, q_heads, kv_heads, dim, chunk, splits) <= 0 or chunk > 256 or q_heads % kv_heads:
        raise ValueError("DMS INT8 attention requires valid GQA geometry and chunk <= 256")
    _call("hipengine_dms_compact_attn_decode_splitk_int8",
          [q, k, v, base, live, po, pm, pl, out, k_scale_ptr, v_scale_ptr],
          [ctypes.c_int(x) for x in (rows, q_heads, kv_heads, dim)] +
          [ctypes.c_float(scale), ctypes.c_int(chunk), ctypes.c_int(splits)],
          stream=stream, library=library)


def dms_compact_int8_verify_chain_gate_bf16_spans(
    query_ptr: int,
    key_slot_ptr: int,
    value_slot_ptr: int,
    k_scale_ptr: int,
    v_scale_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    result_ptr: int,
    partial_out_ptr: int,
    partial_m_ptr: int,
    partial_l_ptr: int,
    base_ptr: int,
    live_ptr: int,
    spans,
    rows: int,
    chunk_size: int,
    num_splits: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    query_row_stride: int,
    gate_row_stride: int,
    gate_head_stride: int,
    gate_dim_stride: int,
    out_row_stride: int,
    out_head_stride: int,
    out_dim_stride: int,
    scale: float,
    *,
    stream: int = 0,
    library=None,
    runtime=None,
) -> None:
    """INT8 verifier rows over a compact-DMS per-head-variable span set.

    This is the compact store's counterpart to the paged verify-chain leaf. The
    two differ in what ``spans`` means, which is why they are separate leaves
    under separate layers rather than one leaf with two readings: a paged span
    set carries one page table plus one live count per row, while a DMS span set
    declares a dense extent per ``(row, kv head)`` and indexes its int8 scale
    planes per slot.

    ``base_ptr``/``live_ptr`` are the ``[rows, kv_heads]`` int32 planes the
    compact store publishes for the layer being attended, the same way the AR
    route passes that store's ``[kv_heads]`` planes. They are explicit rather
    than read out of ``spans`` because a span set declares ``[rows, layers,
    heads]`` while the kernel indexes one layer's pair; ``spans`` is the
    contract guard on the layout being read.

    The attention family writes FP32, so the leaf takes its own result plane and
    finishes through the same FP32-in/BF16-out gate multiply the other verify
    paths use. ``result_ptr`` must hold ``[rows, q_heads, head_dim]`` FP32 and
    ``query``, ``gate`` and ``out`` must be contiguous at the strides given, since
    neither the DMS split-K producer nor the gate multiply reads a stride.
    """

    _validate_verify_chain_spans(
        spans,
        rows=rows,
        num_kv_heads=num_kv_heads,
    )
    rows = int(rows)
    num_q_heads = int(num_q_heads)
    num_kv_heads = int(num_kv_heads)
    head_dim = int(head_dim)
    if min(rows, num_q_heads, num_kv_heads, head_dim, num_splits) <= 0:
        raise ValueError("DMS verify-chain attention requires positive dimensions")
    if rows <= 1:
        raise ValueError("DMS verify-chain attention requires more than one row")
    if num_q_heads % num_kv_heads:
        raise ValueError("DMS verify-chain attention requires valid GQA geometry")
    if not 0 < int(chunk_size) <= 256:
        raise ValueError("DMS verify-chain attention requires chunk_size <= 256")
    if not float(scale) > 0.0:
        raise ValueError("DMS verify-chain attention requires a positive scale")
    contiguous = num_q_heads * head_dim
    for value, name, expected in (
        (query_row_stride, "query_row_stride", contiguous),
        (gate_row_stride, "gate_row_stride", contiguous),
        (gate_head_stride, "gate_head_stride", head_dim),
        (gate_dim_stride, "gate_dim_stride", 1),
        (out_row_stride, "out_row_stride", contiguous),
        (out_head_stride, "out_head_stride", head_dim),
        (out_dim_stride, "out_dim_stride", 1),
    ):
        if int(value) != expected:
            raise ValueError(
                f"DMS verify-chain attention requires contiguous {name} "
                f"({int(value)} != {expected})"
            )
    if any(
        int(pointer) <= 0
        for pointer in (
            query_ptr,
            key_slot_ptr,
            value_slot_ptr,
            k_scale_ptr,
            v_scale_ptr,
            gate_ptr,
            out_ptr,
            result_ptr,
            partial_out_ptr,
            partial_m_ptr,
            partial_l_ptr,
            base_ptr,
            live_ptr,
        )
    ):
        raise ValueError("DMS verify-chain attention requires non-null device pointers")

    dms_compact_attn_decode_splitk_int8(
        int(query_ptr),
        int(key_slot_ptr),
        int(value_slot_ptr),
        int(base_ptr),
        int(live_ptr),
        int(partial_out_ptr),
        int(partial_m_ptr),
        int(partial_l_ptr),
        int(result_ptr),
        rows,
        num_q_heads,
        num_kv_heads,
        head_dim,
        float(scale),
        int(chunk_size),
        int(num_splits),
        k_scale_ptr=int(k_scale_ptr),
        v_scale_ptr=int(v_scale_ptr),
        stream=int(stream),
        library=library,
        runtime=runtime,
    )
    from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
        qwen35_full_attn_gate_mul_bf16,
    )

    qwen35_full_attn_gate_mul_bf16(
        int(result_ptr),
        int(gate_ptr),
        int(out_ptr),
        rows * contiguous,
        stream=int(stream),
        runtime=runtime,
    )


def _validate_verify_chain_spans(spans, *, rows: int, num_kv_heads: int) -> None:
    """Reject a span set the compact verifier cannot read.

    The check is on the span set's declared layout, not on any artifact identity:
    a per-head-variable INT8 span set carrying one extent per ``(row, kv head)``
    of the layer being attended is exactly what this leaf reads.
    """

    mode = getattr(spans, "spans_mode", None)
    if mode != "per_head_variable":
        raise ValueError(
            "DMS verify-chain attention requires a per_head_variable span set "
            f"(got {mode!r})"
        )
    storage = getattr(spans, "storage_dtype", None)
    if getattr(storage, "value", storage) != "int8_per_token_head":
        raise ValueError(
            "DMS verify-chain attention requires int8_per_token_head storage "
            f"(got {storage!r})"
        )
    for name in ("base_offsets", "live_counts"):
        tensor = getattr(spans, name, None)
        if tensor is None:
            raise ValueError(f"DMS verify-chain attention requires {name}")
        shape = tuple(int(value) for value in getattr(tensor, "shape", ()))
        if len(shape) != 3:
            raise ValueError(
                f"DMS verify-chain attention requires {name} to declare "
                f"[rows, layers, kv_heads] (got {shape})"
            )
        if shape[0] != int(rows) or shape[2] != int(num_kv_heads):
            raise ValueError(
                f"DMS verify-chain attention requires {name} to cover every "
                f"(row, kv head) of {int(rows)} rows and {int(num_kv_heads)} "
                f"heads (got {shape})"
            )
    if tuple(spans.live_counts.shape) != tuple(spans.base_offsets.shape):
        raise ValueError(
            "DMS verify-chain attention requires base_offsets and live_counts "
            "to declare the same extent"
        )


def register_dms_compact_int8_kernels(*, replace=True):
    for layer, variant, function in (
        ("dms_streaming_pack", "count_rank_scatter", dms_streaming_pack_int8),
        ("dms_append_decode", "compact_append_evict", dms_append_decode_int8),
        ("dms_compact_attn_decode", "grouped_gqa_splitk", dms_compact_attn_decode_splitk_int8),
        (
            "dms_compact_attn_decode",
            "verify_chain_gate_bf16_spans",
            dms_compact_int8_verify_chain_gate_bf16_spans,
        ),
    ):
        register(KernelKey("hip_gfx1100", layer, "int8_per_token_head", variant), function, replace=replace)


register_dms_compact_int8_kernels()
