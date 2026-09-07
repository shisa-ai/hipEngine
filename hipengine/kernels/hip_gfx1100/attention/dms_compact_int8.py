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


def register_dms_compact_int8_kernels(*, replace=True):
    for layer, variant, function in (
        ("dms_streaming_pack", "count_rank_scatter", dms_streaming_pack_int8),
        ("dms_append_decode", "compact_append_evict", dms_append_decode_int8),
        ("dms_compact_attn_decode", "grouped_gqa_splitk", dms_compact_attn_decode_splitk_int8),
    ):
        register(KernelKey("hip_gfx1100", layer, "int8_per_token_head", variant), function, replace=replace)


register_dms_compact_int8_kernels()
