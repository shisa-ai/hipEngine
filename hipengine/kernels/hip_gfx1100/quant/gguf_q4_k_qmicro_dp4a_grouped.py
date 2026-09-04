"""Wrappers for the grouped q8_1-DP4A Q4_K qmicro-T16 decode candidate.

C8-P2 Q4 pool candidate: integer-dp4a sibling for the Q4_K qmicro T16
targets (rows 8-64). The tile layout is produced by
``repack_gguf_q4_k_tile16_qmicro`` and consumed DIRECTLY (no planar
re-conversion: the [subblock][k][col-pair] q section yields two
contiguous u32 loads per (subblock, quartet, 4-k pack)); activations
come from the d4s4-f32 q8_1 producer shared with the retained raw Q5
MMQ owners. This is a leaf-screen/experiment entry point — the
registered production owners (grouped rowtile16-w2 family) are
untouched until a production-profile numerics campaign promotes the
variant.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q4_k_qmicro_dp4a_grouped.hip")
_OUTPUT_NAME = "gguf_q4_k_qmicro_dp4a_grouped.so"
_Q4_QMICRO_DP4A_GROUPED_BF16_BF16 = (
    "hipengine_gguf_q4_k_qmicro_dp4a_grouped_bf16_bf16_out"
)
_Q4_QMICRO_DP4A_GROUPED_BF16_F32 = (
    "hipengine_gguf_q4_k_qmicro_dp4a_grouped_bf16_f32_out"
)


def plan_gguf_q4_k_qmicro_dp4a_grouped_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q4_k_qmicro_dp4a_grouped",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q4_k_qmicro_dp4a_grouped(
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
        family="gguf_q4_k_qmicro_dp4a_grouped",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


_LIBRARY: ctypes.CDLL | None = None


def _library() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = build_gguf_q4_k_qmicro_dp4a_grouped(load=True)
    return _LIBRARY


def _launch(
    symbol: str,
    xq_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    del runtime  # default stream/device context is used by the launcher
    resolved = library or _library()
    fn = getattr(resolved, symbol)
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
        ctypes.c_void_p(int(xq_ptr)),
        ctypes.c_void_p(int(tiles_ptr)),
        ctypes.c_void_p(int(out_ptr)),
        ctypes.c_int64(int(rows)),
        ctypes.c_int64(int(in_features)),
        ctypes.c_int64(int(out_features)),
        ctypes.c_void_p(int(stream)),
    )
    if err != 0:
        raise RuntimeError(f"q4 qmicro dp4a grouped launch failed: {err}")


def _validate_shape(rows: int, in_features: int, out_features: int) -> None:
    if rows < 8 or rows > 64 or rows % 8 != 0:
        raise ValueError("grouped qmicro Q8_1 rows must be a multiple of 8 in [8, 64]")
    if in_features <= 0 or in_features % 256 != 0:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % 16 != 0:
        raise ValueError("out_features must be a positive multiple of 16")


def gguf_q4_k_qmicro_q8_1_dp4a_grouped_bf16_bf16_out(
    xq_ptr: int,
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
    """Launch the grouped q8_1 DP4A qmicro-Q4 sibling for rows 8-64.

    Integer-decode candidate for the Q4_K qmicro targets (changed
    arithmetic vs the grouped rowtile16-w2 owner; the owner remains the
    registered strict fallback until a production-profile L4 campaign
    promotes this variant).
    """

    _validate_shape(rows, in_features, out_features)
    _launch(
        _Q4_QMICRO_DP4A_GROUPED_BF16_BF16,
        xq_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_qmicro_q8_1_dp4a_grouped_bf16_f32_out(
    xq_ptr: int,
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
    """F32-output variant for oracle comparisons."""
    _validate_shape(rows, in_features, out_features)
    _launch(
        _Q4_QMICRO_DP4A_GROUPED_BF16_F32,
        xq_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


_Q4_QMICRO_DP4A_VARIANT_BF16 = "q8_1_dp4a_grouped_bf16_bf16_out"
_Q4_QMICRO_DP4A_VARIANT_F32 = "q8_1_dp4a_grouped_bf16_f32_out"


def register_gguf_q4_k_qmicro_dp4a_grouped_kernels(*, replace: bool = True) -> None:
    """Register the grouped dp4a qmicro-Q4 variants on the linear axis."""

    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", _Q4_QMICRO_DP4A_VARIANT_BF16),
        gguf_q4_k_qmicro_q8_1_dp4a_grouped_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", _Q4_QMICRO_DP4A_VARIANT_F32),
        gguf_q4_k_qmicro_q8_1_dp4a_grouped_bf16_f32_out,
        replace=replace,
    )


# Module-import registration, mirroring the raw mmq family: the dispatch
# branch requires the key to be registered before the first launch, and
# `_ensure_linear_kernel_registered` only fires when a RESOLVED key is
# unregistered, which never happens for a fallback-gated variant
# (iter42 engagement fix on the Q5 axis).
register_gguf_q4_k_qmicro_dp4a_grouped_kernels()


__all__ = [
    "build_gguf_q4_k_qmicro_dp4a_grouped",
    "gguf_q4_k_qmicro_q8_1_dp4a_grouped_bf16_bf16_out",
    "gguf_q4_k_qmicro_q8_1_dp4a_grouped_bf16_f32_out",
    "plan_gguf_q4_k_qmicro_dp4a_grouped_build",
    "register_gguf_q4_k_qmicro_dp4a_grouped_kernels",
]
