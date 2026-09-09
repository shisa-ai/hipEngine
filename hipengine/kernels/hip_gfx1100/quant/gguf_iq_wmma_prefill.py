"""W4A16 prefill for raw dense GGUF IQ weights.

Expands raw IQ blocks into fp16 WMMA operands and reads activations straight
from the caller's bf16 buffer, so there is no quantized activation plane and no
workspace. See the kernel header for why fp16 rather than bf16.
"""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_iq_wmma_prefill.hip")
_OUTPUT_NAME = "gguf_iq_wmma_prefill.so"
_SYMBOL = "hipengine_gguf_iq_wmma_prefill_bf16_bf16_out"
_VARIANT = "dense_wmma_w4a16_prefill_bf16_bf16_out"
_TABLES = ("gguf_iq_dense_tables.h", "gguf_iq2_xs_dense_table.h", "gguf_iq3_s_grid.h")
QUANTS = {"gguf_iq4_xs": 0, "gguf_iq4_nl": 1, "gguf_iq3_s": 2, "gguf_q3_k": 3,
          "gguf_iq3_xxs": 4, "gguf_iq2_s": 5, "gguf_iq2_xs": 6}
# IQ4_NL blocks hold 32 elements; every other supported quant holds 256.
_BLOCK_ELEMENTS = {"gguf_iq4_nl": 32}
_HANDLES: dict[int, object] = {}
_LIBRARY = None


def _table_hash() -> str:
    return hashlib.sha256(b"".join(
        _SOURCE.with_name(name).read_bytes() for name in _TABLES)).hexdigest()


# Retained tile configs per target arch, as (small-rows tile, large-rows tile).
# gfx1151 keeps the 2026-09-09 zbook sweep's retained 16x128 at every row
# count. The gfx1100 re-sweep (2026-09-09, 7900 XTX and W7900, 512-row real
# IQ4_XS shapes from the UD-Q4_K_M artifact) retained 32x64 for large rows -
# 18-20% under 16x128 on BOTH cards and on every shape, unlike gfx1151 where
# 32x64 measured 50% worse - and 16x64 through 64 rows (2.4x 16x128 at 8
# rows: the TILE_N activation padding dominates small rows). The tile
# crossover sits between 64 and 128 rows (16x64 wins at 64, 32x64 at 128).
_ARCH_TILES = {
    "gfx1100": ((16, 64), (32, 64)),
    "gfx1151": ((16, 128), (16, 128)),
}
_SMALL_ROWS_MAX = 64
_detected_arch: str | None = None
_ROW_LIBRARIES: dict[tuple[int, int], object] = {}


def _target_arch() -> str:
    """The build's target arch: HIPENGINE_HIP_ARCH when a backend scoped it,
    otherwise the (single) visible HIP device's arch."""

    import os
    arch = (os.environ.get("HIPENGINE_HIP_ARCH") or "").strip()
    if arch:
        return arch
    global _detected_arch
    if _detected_arch is None:
        from hipengine.kernels.backends import detect_hip_target_arches
        arches = detect_hip_target_arches()
        _detected_arch = arches[0] if len(arches) == 1 else ""
    return _detected_arch


def _retained_tiles() -> tuple[tuple[int, int], tuple[int, int]]:
    """(small-rows, large-rows) retained tiles for the target arch."""

    return _ARCH_TILES.get(_target_arch(), ((16, 128), (16, 128)))


def _flags(tiles: tuple[int, int] | None = None) -> tuple[str, ...]:
    import os
    if tiles is None:
        tiles = _retained_tiles()[1]
    flags = [f"-DHIPENGINE_IQ_WMMA_TABLE_HASH={_table_hash()}",
             f"-DIQ_WMMA_TILE_M={tiles[0]}",
             f"-DIQ_WMMA_TILE_N={tiles[1]}"]
    # Tile sweep hook: explicit env overrides always win, and the values are
    # part of the build cache key, so variants do not collide.
    for name in ("IQ_WMMA_TILE_M", "IQ_WMMA_TILE_N"):
        value = os.environ.get(f"HIPENGINE_{name}")
        if value:
            flags[1 if name == "IQ_WMMA_TILE_M" else 2] = f"-D{name}={int(value)}"
    return tuple(flags)


def plan_gguf_iq_wmma_prefill_build(**kwargs) -> BuildArtifact:
    return plan_hip_build(sources=[_SOURCE], family="gguf_iq_wmma_prefill",
                          profile=kwargs.pop("profile", "prefill"),
                          extra_flags=_flags(kwargs.pop("tiles", None)),
                          output_name=_OUTPUT_NAME, **kwargs)


def build_gguf_iq_wmma_prefill(*, profile: ProfileName = "prefill",
                               tiles: tuple[int, int] | None = None, **kwargs):
    return build_hip(sources=[_SOURCE], family="gguf_iq_wmma_prefill",
                     profile=profile, extra_flags=_flags(tiles),
                     output_name=_OUTPUT_NAME, **kwargs)


def _library_for_rows(rows: int):
    """The retained library for this row count (small-rows vs large-rows tile)."""

    small, large = _retained_tiles()
    tiles = small if int(rows) <= _SMALL_ROWS_MAX else large
    lib = _ROW_LIBRARIES.get(tiles)
    if lib is None:
        lib = build_gguf_iq_wmma_prefill(load=True, tiles=tiles)
        _ROW_LIBRARIES[tiles] = lib
    return lib


def _default_library():
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = build_gguf_iq_wmma_prefill(load=True)
    return _LIBRARY


def launch(x_ptr: int, qweight_ptr: int, out_ptr: int, rows: int,
           in_features: int, out_features: int, *, quant: str,
           output: str = "bf16", stream: int = 0,
           library: ctypes.CDLL | None = None,
           runtime: HipRuntime | None = None, **_ignored) -> None:
    """Signature matches the raw GGUF linear launch ABI.

    ``output`` is validated rather than ignored: this kernel only writes bf16,
    so a caller asking for f32 must fail loudly instead of silently receiving
    the wrong dtype.
    """

    if output != "bf16":
        raise ValueError("W4A16 prefill writes bf16 only")
    if quant not in QUANTS:
        raise ValueError(f"unsupported dense IQ quant for W4A16 prefill: {quant!r}")
    block = _BLOCK_ELEMENTS.get(quant, 256)
    if rows <= 0 or in_features <= 0 or in_features % block or out_features <= 0:
        raise ValueError(
            f"W4A16 prefill requires positive dimensions and K divisible by {block}")
    if not all((x_ptr, qweight_ptr, out_ptr)):
        raise ValueError("W4A16 prefill pointers must be nonzero")
    library = library if isinstance(library, ctypes.CDLL) else _library_for_rows(rows)
    fn = _HANDLES.get(id(library))
    if fn is None:
        fn = getattr(library, _SYMBOL)
        fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3 + \
                      [ctypes.c_int, ctypes.c_void_p]
        fn.restype = ctypes.c_int
        _HANDLES[id(library)] = fn
    error = fn(ctypes.c_void_p(x_ptr), ctypes.c_void_p(qweight_ptr),
               ctypes.c_void_p(out_ptr), ctypes.c_int64(rows),
               ctypes.c_int64(in_features), ctypes.c_int64(out_features),
               ctypes.c_int(QUANTS[quant]), ctypes.c_void_p(stream))
    if int(error) != HIP_SUCCESS:
        (runtime or get_hip_runtime()).check(int(error))


def register_gguf_iq_wmma_prefill_kernels(*, replace: bool = True) -> None:
    from functools import partial
    for quant in QUANTS:
        register(KernelKey("hip_gfx1100", "linear", quant, _VARIANT),
                 partial(launch, quant=quant), replace=replace)


register_gguf_iq_wmma_prefill_kernels()

__all__ = ["QUANTS", "build_gguf_iq_wmma_prefill", "launch",
           "plan_gguf_iq_wmma_prefill_build",
           "register_gguf_iq_wmma_prefill_kernels"]
