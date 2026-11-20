"""Exact rows-1 T16 decode GEMVs: grouped Q4T16 and hoisted planar Q6.

``gguf_q4_k_t16_tile16_grouped_gemv`` computes up to four Q4T16 projections
that share one BF16 input in a single launch; every output is bit-identical to
the ``dense_single_local32`` owner. ``gguf_q6_k_t16_qmicro_planar_hoisted``
is bit-identical to the ``q6_k_t16_qmicro_planar`` GEMV owner. Both keep the
parent K ownership and reduction order (see the .hip header), so the exact
row-independent verify owners that match those parents still match these.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from hipengine.core.build import build_hip
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_suffix(".hip")
_OUTPUT_NAME = "gguf_t16_grouped_gemv.so"
_P, _I64, _I32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
_QK_K = 256
_T16_COLS = 16
MAX_SEGMENTS = 4
GROUPED_WAVES = (1, 2, 4, 8)

Q4_GROUPED_VARIANT = "tile16_grouped_gemv_bf16_bf16_out"
Q6_HOISTED_VARIANT = "planar_hoisted_gemv_bf16_bf16_out"


def build_gguf_t16_grouped_gemv(*, load=True, require_cached=False, compiler_version=None):
    return build_hip(
        sources=[_SOURCE], family="gguf_t16_grouped_gemv", profile="decode",
        output_name=_OUTPUT_NAME, load=load, require_cached=require_cached,
        compiler_version=compiler_version,
    )


@lru_cache(maxsize=1)
def _library():
    return build_gguf_t16_grouped_gemv()


def gguf_q4_k_t16_tile16_grouped_gemv(
    x_ptr: int,
    segments: Sequence[tuple[int, int, int]],
    in_features: int,
    *,
    waves: int = 1,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch one grouped Q4T16 GEMV over ``(tiles_ptr, out_ptr, out_features)``."""

    n = len(segments)
    if not 1 <= n <= MAX_SEGMENTS:
        raise ValueError(f"grouped Q4T16 GEMV takes 1..{MAX_SEGMENTS} segments, got {n}")
    if in_features <= 0 or in_features % _QK_K:
        raise ValueError("in_features must be a positive multiple of 256")
    if waves not in GROUPED_WAVES:
        raise ValueError(f"waves must be one of {GROUPED_WAVES}")
    for _, _, out_features in segments:
        if out_features <= 0 or out_features % _T16_COLS:
            raise ValueError("every segment's out_features must be a positive multiple of 16")
    lib = library or _library()
    rt = runtime or get_hip_runtime()
    tiles = (_P * n)(*(int(s[0]) for s in segments))
    outs = (_P * n)(*(int(s[1]) for s in segments))
    outf = (_I64 * n)(*(int(s[2]) for s in segments))
    fn = signed_kernel_fn(
        lib, "hipengine_gguf_q4_k_t16_tile16_grouped_gemv_bf16_out",
        (_P, _P, _P, _P, _I32, _I64, _I32, _P), _I32,
    )
    rt.check(int(fn(x_ptr, tiles, outs, outf, n, in_features, waves, stream)))


def gguf_q6_k_t16_qmicro_planar_hoisted_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the exact hoisted planar-qmicro Q6T16 GEMV (row-independent)."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0 or in_features % _QK_K:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % _T16_COLS:
        raise ValueError("out_features must be a positive multiple of 16")
    lib = library or _library()
    rt = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        lib, "hipengine_gguf_q6_k_t16_qmicro_planar_hoisted_gemv_bf16_out",
        (_P, _P, _P, _I64, _I64, _I64, _P), _I32,
    )
    rt.check(int(fn(x_ptr, tiles_ptr, out_ptr, rows, in_features, out_features, stream)))


def register_gguf_t16_grouped_gemv_kernels(*, replace: bool = True) -> None:
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        register(
            KernelKey(backend, "linear_group", "gguf_q4_k_t16_v1", Q4_GROUPED_VARIANT),
            gguf_q4_k_t16_tile16_grouped_gemv,
            replace=replace,
        )
        register(
            KernelKey(backend, "linear", "gguf_q6_k_t16_qmicro_planar_v1", Q6_HOISTED_VARIANT),
            gguf_q6_k_t16_qmicro_planar_hoisted_bf16_bf16_out,
            replace=replace,
        )


register_gguf_t16_grouped_gemv_kernels()


__all__ = [
    "GROUPED_WAVES",
    "MAX_SEGMENTS",
    "Q4_GROUPED_VARIANT",
    "Q6_HOISTED_VARIANT",
    "build_gguf_t16_grouped_gemv",
    "gguf_q4_k_t16_tile16_grouped_gemv",
    "gguf_q6_k_t16_qmicro_planar_hoisted_bf16_bf16_out",
    "register_gguf_t16_grouped_gemv_kernels",
]
