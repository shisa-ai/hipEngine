"""Launcher for the IQ4_XS T16-layout local32 decode GEMV.

The T16 tile is the same 2176 bytes per 16 columns as 16 raw 136-byte IQ4_XS
blocks, so adopting it costs no resident footprint. What it changes is the qs
plane: raw stores ``[group][byte]`` per column, so a lane's 8-byte payload
window is read twice, once per nibble half. T16 stores
``[group][element][column pair]``, so one u32 load at a fixed element yields all
four column pairs -- eight output columns -- for that element and the block
moves half the weight bytes.

The kernel keeps the raw owner's lane ownership, FMA order, shuffle tree and
cross-wave sum, so its output is bit-identical to ``launch_local32`` on the same
weights. This module only wires the launcher; the parity contract is asserted by
``tests/test_gguf_iq4_xs_t16_local32_parity.py``.
"""

from __future__ import annotations

import ctypes

from .gguf_iq_dense import _default_library, _local32_waves

_ROWS_HANDLES: dict[tuple[int, int], object] = {}


def _rows_handle(library, rows: int):
    key = (id(library), rows)
    fn = _ROWS_HANDLES.get(key)
    if fn is None:
        fn = library.hipengine_gguf_iq4_xs_t16_local32_rows_gemv
        fn.argtypes = ([ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3
                       + [ctypes.c_int32] + [ctypes.c_void_p])
        fn.restype = ctypes.c_int
        _ROWS_HANDLES[key] = fn
    return fn


def launch_iq4_xs_t16_local32(x_ptr, qweight_ptr, out_ptr, rows, in_features,
                              out_features, *, stream=0, library=None,
                              runtime=None):
    """Launch the T16-layout local32 IQ4_XS decode GEMV (bf16 out).

    ``qweight_ptr`` must point at an IQ4_XS T16 tile array as produced by
    :func:`hipengine.quant.gguf_t16.repack_gguf_iq4_xs_tile16`. Rows 1-4 are
    supported: rows > 1 is the verifier sibling, which owns ``rows`` prompt rows
    per block and keeps each row's accumulation order, so each row's output is
    bit-identical to the rows == 1 owner's output for that row.

    ``out_features`` must be divisible by 16, because one T16 tile spans 16
    columns and a block reads a whole tile's qs plane. Callers with a narrower
    tail column count must keep the raw-layout owner registered as the fallback.
    """
    if rows < 1 or rows > 4:
        raise ValueError('T16 local32 decode supports rows 1-4')
    if (in_features <= 0 or in_features % 256 or out_features <= 0
            or out_features % 16):
        raise ValueError('T16 local32 decode requires K divisible by 256 and '
                         'N divisible by 16')
    if not all((x_ptr, qweight_ptr, out_ptr)):
        raise ValueError('T16 local32 decode pointers must be nonzero')
    library = library or _default_library()
    waves = _local32_waves(in_features, out_features)
    args = (ctypes.c_void_p(x_ptr), ctypes.c_void_p(qweight_ptr),
            ctypes.c_void_p(out_ptr), ctypes.c_int64(rows),
            ctypes.c_int64(in_features), ctypes.c_int64(out_features),
            ctypes.c_int32(waves), ctypes.c_void_p(stream))
    if rows == 1:
        fn = library.hipengine_gguf_iq4_xs_t16_local32_gemv
        fn.argtypes = ([ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3
                       + [ctypes.c_int32] + [ctypes.c_void_p])
        fn.restype = ctypes.c_int
    else:
        fn = _rows_handle(library, rows)
    err = fn(*args)
    if err:
        from hipengine.core.hip_runtime import get_hip_runtime

        rt = runtime or get_hip_runtime()
        raise RuntimeError(f'T16 local32 decode failed: {rt.error_string(err)}')
