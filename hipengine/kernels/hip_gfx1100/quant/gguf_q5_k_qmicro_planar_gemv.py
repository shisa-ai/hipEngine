"""Wrappers for the grouped q8_1-DP4A Q5_K qmicro-planar decode candidate.

C8-P2 residual candidate: integer-dp4a sibling for the Q5_K T16 qmicro
planar targets (rows 8-64). The tile layout is produced by
``convert_gguf_q5_k_qmicro_tile16_to_planar``; activations come from the
d4s4-f32 q8_1 producer already used by the retained raw Q5 MMQ owners.
This is a leaf-screen/experiment entry point — the registered production
owners (BF16 grouped / raw mmq) are untouched until a production-profile
numerics campaign promotes the variant.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HipRuntime, get_hip_runtime

_SOURCE = Path(__file__).with_name("gguf_q5_k_qmicro_planar_gemv.hip")
_OUTPUT_NAME = "gguf_q5_k_qmicro_planar_gemv.so"
_Q5_QMICRO_PLANAR_DP4A_GROUPED_BF16_BF16 = (
    "hipengine_gguf_q5_k_qmicro_planar_dp4a_grouped_bf16_bf16_out"
)
_Q5_QMICRO_PLANAR_DP4A_GROUPED_BF16_F32 = (
    "hipengine_gguf_q5_k_qmicro_planar_dp4a_grouped_bf16_f32_out"
)


def plan_gguf_q5_k_qmicro_planar_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q5_k_qmicro_planar_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q5_k_qmicro_planar_gemv(
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
        family="gguf_q5_k_qmicro_planar_gemv",
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
        _LIBRARY = build_gguf_q5_k_qmicro_planar_gemv(load=True)
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
        raise RuntimeError(f"q5 planar dp4a grouped launch failed: {err}")


def _validate_shape(rows: int, in_features: int, out_features: int) -> None:
    if rows < 8 or rows > 64 or rows % 8 != 0:
        raise ValueError("grouped planar Q8_1 rows must be a multiple of 8 in [8, 64]")
    if in_features <= 0 or in_features % 256 != 0:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % 16 != 0:
        raise ValueError("out_features must be a positive multiple of 16")


def gguf_q5_k_qmicro_planar_q8_1_dp4a_grouped_bf16_bf16_out(
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
    """Launch the grouped q8_1 DP4A planar-Q5 sibling for rows 8-64.

    Integer-decode candidate for the Q5_K qmicro-planar targets (changed
    arithmetic vs the BF16 grouped owner; the owner remains the registered
    strict fallback until a production-profile L4 campaign promotes this
    variant).
    """

    _validate_shape(rows, in_features, out_features)
    _launch(
        _Q5_QMICRO_PLANAR_DP4A_GROUPED_BF16_BF16,
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


def gguf_q5_k_qmicro_planar_q8_1_dp4a_grouped_bf16_f32_out(
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
        _Q5_QMICRO_PLANAR_DP4A_GROUPED_BF16_F32,
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
