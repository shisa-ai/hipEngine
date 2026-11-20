"""Raw-pointer device-side row gather for small runtime glue buffers.

``gather_f32_rows_by_i32id`` copies whole rows from a row-major FP32 ``table``
([vocab, hidden]) into a packed ``out`` ([rows, hidden]) using device-resident
int32 row indices. ``gather_bf16_rows_by_i32id`` is the bit-exact BF16 variant
(shortlist lm_head rows / embedding rows), and ``gather_i32_by_i32id`` picks
int32 scalars (shortlist argmax index -> top-K token id). The row id lives on
the device, so a resident chain (e.g. the MTP NextN or EAGLE-4 draft) can feed
one kernel's argmax straight into the next depth's lookup without a host
round-trip.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gather.hip")
_OUTPUT_NAME = "gather.so"
_SYMBOL_GATHER_F32_ROWS = "hipengine_gather_f32_rows_by_i32id"
_SYMBOL_GATHER_BF16_ROWS = "hipengine_gather_bf16_rows_by_i32id"
_SYMBOL_GATHER_I32 = "hipengine_gather_i32_by_i32id"


def plan_gather_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gather",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gather(
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
        family="gather",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gather_f32_rows_by_i32id(
    table_ptr: int,
    ids_ptr: int,
    out_ptr: int,
    rows: int,
    hidden: int,
    vocab: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Gather ``rows`` FP32 rows from ``table`` ([vocab, hidden]) by device id.

    ``ids_ptr`` points at ``rows`` device-resident int32 row indices; out-of-range
    ids ([0, vocab)) produce a zeroed output row.
    """

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0:
        raise ValueError("hidden must be positive")
    if vocab <= 0:
        raise ValueError("vocab must be positive")
    library = library or build_gather(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GATHER_F32_ROWS)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(table_ptr),
        ctypes.c_void_p(ids_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_int64(vocab),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gather_bf16_rows_by_i32id(
    table_ptr: int,
    ids_ptr: int,
    out_ptr: int,
    rows: int,
    hidden: int,
    vocab: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Bit-exact gather of ``rows`` BF16 rows from ``table`` ([vocab, hidden]).

    ``ids_ptr`` points at ``rows`` device-resident int32 row indices; out-of-range
    ids ([0, vocab)) produce a zeroed output row. Used by the device-resident
    EAGLE-4 head (shortlist ``lm_head`` rows) and embedding lookup.
    """

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0:
        raise ValueError("hidden must be positive")
    if vocab <= 0:
        raise ValueError("vocab must be positive")
    library = library or build_gather(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GATHER_BF16_ROWS)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(table_ptr),
        ctypes.c_void_p(ids_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_int64(vocab),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gather_i32_by_i32id(
    table_ptr: int,
    ids_ptr: int,
    out_ptr: int,
    rows: int,
    vocab: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Pick ``rows`` int32 scalars: ``out[r] = table[ids[r]]``.

    Both ``ids`` and ``table`` are device-resident; out-of-range ids produce 0.
    Maps a device shortlist argmax index back to its top-K token id.
    """

    if rows <= 0:
        raise ValueError("rows must be positive")
    if vocab <= 0:
        raise ValueError("vocab must be positive")
    library = library or build_gather(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GATHER_I32)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(table_ptr),
        ctypes.c_void_p(ids_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(vocab),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_gather_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "gather_f32_rows_by_i32id", "fp32"),
        gather_f32_rows_by_i32id,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gather_bf16_rows_by_i32id", "bf16"),
        gather_bf16_rows_by_i32id,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gather_i32_by_i32id", "i32"),
        gather_i32_by_i32id,
        replace=replace,
    )


register_gather_kernels()


__all__ = [
    "build_gather",
    "gather_bf16_rows_by_i32id",
    "gather_f32_rows_by_i32id",
    "gather_i32_by_i32id",
    "plan_gather_build",
    "register_gather_kernels",
]
