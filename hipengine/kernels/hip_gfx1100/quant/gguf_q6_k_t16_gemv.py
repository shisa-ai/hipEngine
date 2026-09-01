"""Wrappers for dense GGUF Q6_K T16 GEMV decode kernels.

P9.H3 extension for the qwen35moe Q6_K lm-head fallback.  The kernel consumes
``repack_gguf_q6_k_tile16(raw[None, ...])`` output and exposes the regular
``linear`` registry ABI used by ``launch_gguf_linear``.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Mapping

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_pack8_gemv import (
    gguf_q6_k_pack8_top1_stage2_gather_f32,
)
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q6_k_t16_gemv.hip")
_OUTPUT_NAME = "gguf_q6_k_t16_gemv.so"
_Q6_T16_BF16_F32 = "hipengine_gguf_q6_k_t16_gemv_decode_bf16_f32_out"
_Q6_T16_BF16_BF16 = "hipengine_gguf_q6_k_t16_gemv_decode_bf16_bf16_out"
_Q6_T16_BF16_F32_TOP1_STAGE1 = (
    "hipengine_gguf_q6_k_t16_gemv_decode_bf16_f32_top1_stage1"
)
_Q6_T16_ROWTILE_BF16_F32 = "hipengine_gguf_q6_k_t16_gemv_rowtile_bf16_f32_out"
_Q6_T16_ROWTILE_BF16_BF16 = "hipengine_gguf_q6_k_t16_gemv_rowtile_bf16_bf16_out"
_Q6_T16_ROWTILE_COL8_BF16_F32 = (
    "hipengine_gguf_q6_k_t16_gemv_rowtile_col8_bf16_f32_out"
)
_Q6_T16_ROWTILE_COL8_BF16_BF16 = (
    "hipengine_gguf_q6_k_t16_gemv_rowtile_col8_bf16_bf16_out"
)
_Q6_T16_ROWTILE12_COL8_BF16_BF16 = (
    "hipengine_gguf_q6_k_t16_gemv_rowtile12_col8_bf16_bf16_out"
)
_Q6_T16_QMICRO_PLANAR_BF16_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out"
)
_Q6_T16_QMICRO_PLANAR_BF16_RESIDUAL_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_gemv_decode_"
    "bf16_residual_bf16_out"
)
_Q6_T16_QMICRO_PLANAR_BF16_F32 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_out"
)
_Q6_T16_QMICRO_PLANAR_BF16_F32_TOP1_STAGE1 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_top1_stage1"
)
_Q6_T16_QMICRO_PLANAR_ROWTILE_COL8_BF16_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_bf16_out"
)
_Q6_T16_QMICRO_PLANAR_ROWTILE_COL8_BF16_RESIDUAL_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_"
    "bf16_residual_bf16_out"
)
_Q6_T16_QMICRO_PLANAR_ROWTILE_BF16_F32 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_gemv_rowtile_bf16_f32_out"
)
_Q6_T16_QMICRO_PLANAR_ROWTILE_COL8_BF16_F32 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_f32_out"
)
_Q6_T16_QMICRO_PLANAR_Q8_1_DP4A_BF16_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_bf16_bf16_out"
)
_Q6_T16_QMICRO_PLANAR_Q8_1_DP4A_BF16_RESIDUAL_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_"
    "gemv_bf16_residual_bf16_out"
)
_Q6_T16_QMICRO_PLANAR_WMMA_PREFILL_BF16_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out"
)
_Q6_T16_QMICRO_PLANAR_WMMA_PREFILL_LOWVGPR_BF16_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr_bf16_bf16_out"
)
_Q6_T16_QMICRO_PLANAR_WMMA_PREFILL_LOWVGPR48_BF16_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr48_bf16_bf16_out"
)
_Q6_T16_QMICRO_PLANAR_WMMA_PREFILL_SHARED4R4_BF16_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r4_bf16_bf16_out"
)
_Q6_T16_QMICRO_PLANAR_WMMA_PREFILL_SHARED4R3_BF16_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r3_bf16_bf16_out"
)
_Q6_T16_QMICRO_PLANAR_WMMA_PREFILL_SHARED3R1_BF16_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared3r1_bf16_bf16_out"
)
_Q6_T16_QMICRO_PLANAR_WMMA_PREFILL_SHARED4_BF16_BF16 = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_bf16_out"
)
_Q6_T16_WMMA_PREFILL_BF16_BF16 = (
    "hipengine_gguf_q6_k_t16_wmma_prefill_bf16_bf16_out"
)
_Q6_T16_WMMA_PREFILL_SHARED3R1_BF16_BF16 = (
    "hipengine_gguf_q6_k_t16_wmma_prefill_shared3r1_bf16_bf16_out"
)
_Q6_T16_WMMA_PREFILL_SHARED4_BF16_BF16 = (
    "hipengine_gguf_q6_k_t16_wmma_prefill_shared4_bf16_bf16_out"
)
_Q6_T16_WMMA_PREFILL_SHARED6R1_BF16_BF16 = (
    "hipengine_gguf_q6_k_t16_wmma_prefill_shared6r1_bf16_bf16_out"
)
_Q6_T16_WMMA_PREFILL_SHARED8R3_BF16_BF16 = (
    "hipengine_gguf_q6_k_t16_wmma_prefill_shared8r3_bf16_bf16_out"
)
_QK_K = 256
_T16_COLS = 16


def plan_gguf_q6_k_t16_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q6_k_t16_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q6_k_t16_gemv(
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
        family="gguf_q6_k_t16_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


_Q6_K_T16_GEMV_LIBRARY: ctypes.CDLL | None = None


def _q6_k_t16_gemv_library() -> ctypes.CDLL:
    """Memoized default build so per-call launches skip build_hip entirely.

    The per-call launch path calls ``build_gguf_q6_k_t16_gemv(load=True)`` on
    every launch (~19 us/call of build_hip fast-path overhead even on a cache
    hit); at 141 launches/step that is ~2.6 ms/step of host CPU that lands on
    the critical path. Hoist it once, mirroring the router library pattern.
    """

    global _Q6_K_T16_GEMV_LIBRARY
    if _Q6_K_T16_GEMV_LIBRARY is None:
        _Q6_K_T16_GEMV_LIBRARY = build_gguf_q6_k_t16_gemv(load=True)
    return _Q6_K_T16_GEMV_LIBRARY


def gguf_q6_k_t16_gemv_decode_bf16_f32_out(
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
    """Launch dense Q6T16 GEMV with BF16 activations and FP32 output."""

    _launch(
        _Q6_T16_BF16_F32,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_gemv_decode_bf16_bf16_out(
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
    """Launch dense Q6T16 GEMV with BF16 activations and BF16 output."""

    _launch(
        _Q6_T16_BF16_BF16,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out(
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
    """Launch exact planar-qmicro Q6T16 GEMV with BF16 input/output."""

    _launch(
        _Q6_T16_QMICRO_PLANAR_BF16_BF16,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_residual_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact serial-c1 planar-Q6 projection plus rounded residual."""

    if rows != 1:
        raise ValueError("planar Q6 c1 down-residual requires rows == 1")
    _launch_residual(
        _Q6_T16_QMICRO_PLANAR_BF16_RESIDUAL_BF16,
        x_ptr,
        tiles_ptr,
        residual_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_out(
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
    """Launch exact planar-qmicro Q6T16 GEMV with FP32 output."""

    _launch(
        _Q6_T16_QMICRO_PLANAR_BF16_F32,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_wmma_prefill_bf16_bf16_out(
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
    """Launch direct dense Q6T16 WMMA prefill with BF16 input/output."""

    _launch(
        _Q6_T16_WMMA_PREFILL_BF16_BF16,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_wmma_prefill_shared3r1_bf16_bf16_out(
    x_ptr: int, tiles_ptr: int, out_ptr: int, rows: int,
    in_features: int, out_features: int, *, stream: int = 0,
    library: ctypes.CDLL | None = None, runtime: HipRuntime | None = None,
) -> None:
    """Launch 32-column/48-row standard-Q6 shared-weight prefill."""
    _launch(
        _Q6_T16_WMMA_PREFILL_SHARED3R1_BF16_BF16,
        x_ptr, tiles_ptr, out_ptr, rows, in_features, out_features,
        stream=stream, library=library, runtime=runtime,
    )


def gguf_q6_k_t16_wmma_prefill_shared6r1_bf16_bf16_out(
    x_ptr: int, tiles_ptr: int, out_ptr: int, rows: int,
    in_features: int, out_features: int, *, stream: int = 0,
    library: ctypes.CDLL | None = None, runtime: HipRuntime | None = None,
) -> None:
    """Launch 32-column/96-row standard-Q6 shared-weight prefill."""
    _launch(
        _Q6_T16_WMMA_PREFILL_SHARED6R1_BF16_BF16,
        x_ptr, tiles_ptr, out_ptr, rows, in_features, out_features,
        stream=stream, library=library, runtime=runtime,
    )


def gguf_q6_k_t16_wmma_prefill_shared8r3_bf16_bf16_out(
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
    """Launch 32-column/384-row standard-Q6 shared-weight prefill."""
    _launch(
        _Q6_T16_WMMA_PREFILL_SHARED8R3_BF16_BF16,
        x_ptr, tiles_ptr, out_ptr, rows, in_features, out_features,
        stream=stream, library=library, runtime=runtime,
    )


def gguf_q6_k_t16_wmma_prefill_shared4_bf16_bf16_out(
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
    """Launch four-wave shared-weight standard Q6 WMMA prefill."""

    _launch(
        _Q6_T16_WMMA_PREFILL_SHARED4_BF16_BF16,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r4_bf16_bf16_out(
    x_ptr: int, tiles_ptr: int, out_ptr: int, rows: int,
    in_features: int, out_features: int, *, stream: int = 0,
    library: ctypes.CDLL | None = None, runtime: HipRuntime | None = None,
) -> None:
    """Launch 32-column/256-row planar-Q6 shared-weight prefill."""
    _launch(
        _Q6_T16_QMICRO_PLANAR_WMMA_PREFILL_SHARED4R4_BF16_BF16,
        x_ptr, tiles_ptr, out_ptr, rows, in_features, out_features,
        stream=stream, library=library, runtime=runtime,
    )


def gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r3_bf16_bf16_out(
    x_ptr: int, tiles_ptr: int, out_ptr: int, rows: int,
    in_features: int, out_features: int, *, stream: int = 0,
    library: ctypes.CDLL | None = None, runtime: HipRuntime | None = None,
) -> None:
    """Launch 32-column/192-row planar-Q6 shared-weight prefill."""
    _launch(
        _Q6_T16_QMICRO_PLANAR_WMMA_PREFILL_SHARED4R3_BF16_BF16,
        x_ptr, tiles_ptr, out_ptr, rows, in_features, out_features,
        stream=stream, library=library, runtime=runtime,
    )


def gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared3r1_bf16_bf16_out(
    x_ptr: int, tiles_ptr: int, out_ptr: int, rows: int,
    in_features: int, out_features: int, *, stream: int = 0,
    library: ctypes.CDLL | None = None, runtime: HipRuntime | None = None,
) -> None:
    """Launch 32-column/48-row planar-Q6 shared-weight prefill."""
    _launch(
        _Q6_T16_QMICRO_PLANAR_WMMA_PREFILL_SHARED3R1_BF16_BF16,
        x_ptr, tiles_ptr, out_ptr, rows, in_features, out_features,
        stream=stream, library=library, runtime=runtime,
    )


def gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out(
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
    """Launch planar-qmicro Q6T16 WMMA prefill with BF16 input/output."""

    _launch(
        _Q6_T16_QMICRO_PLANAR_WMMA_PREFILL_BF16_BF16,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr_bf16_bf16_out(
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
    """Launch the low-VGPR 16-column planar-qmicro owner (32-row blocks)."""

    _launch(
        _Q6_T16_QMICRO_PLANAR_WMMA_PREFILL_LOWVGPR_BF16_BF16,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr48_bf16_bf16_out(
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
    """Launch the low-VGPR 16-column planar-qmicro owner (48-row blocks)."""

    _launch(
        _Q6_T16_QMICRO_PLANAR_WMMA_PREFILL_LOWVGPR48_BF16_BF16,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_bf16_out(
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
    """Launch four-wave shared-weight planar Q6 WMMA prefill."""

    _launch(
        _Q6_T16_QMICRO_PLANAR_WMMA_PREFILL_SHARED4_BF16_BF16,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_gemv_decode_bf16_f32_top1_stage1(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    tile_values_ptr: int,
    tile_indices_ptr: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact Q6T16 logits plus one top-1 pair per 16-logit tile."""

    if in_features <= 0 or in_features % _QK_K != 0:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % _T16_COLS != 0:
        raise ValueError("out_features must be a positive multiple of 16")
    library = library or _q6_k_t16_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _Q6_T16_BF16_F32_TOP1_STAGE1)
    fn.argtypes = [
        *([ctypes.c_void_p] * 5),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(tile_values_ptr),
        ctypes.c_void_p(tile_indices_ptr),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_top1_stage1(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    tile_values_ptr: int,
    tile_indices_ptr: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch planar-qmicro logits plus one exact top-1 pair per tile."""

    if in_features <= 0 or in_features % _QK_K != 0:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % _T16_COLS != 0:
        raise ValueError("out_features must be a positive multiple of 16")
    library = library or _q6_k_t16_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _Q6_T16_QMICRO_PLANAR_BF16_F32_TOP1_STAGE1)
    fn.argtypes = [
        *([ctypes.c_void_p] * 5),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(tile_values_ptr),
        ctypes.c_void_p(tile_indices_ptr),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q6_k_t16_proposal_top1_exact_bf16(
    weight: object,
    x_ptr: int,
    logits_f32_ptr: int,
    tile_values_f32_ptr: int,
    tile_indices_i32_ptr: int,
    out_indices_i32_ptr: int,
    out_values_f32_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    libraries: Mapping[str, ctypes.CDLL] | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact resident-T16 proposal logits and reduce them to one winner."""

    if rows != 1:
        raise ValueError("Q6T16 proposal top-1 currently requires rows=1")
    allocation = getattr(weight, "allocation")("tiles")
    t16_library = None if libraries is None else libraries.get("q6_t16")
    pack8_library = None if libraries is None else libraries.get("q6_pack8")
    gguf_q6_k_t16_gemv_decode_bf16_f32_top1_stage1(
        x_ptr,
        int(allocation.tensor.ptr),
        logits_f32_ptr,
        tile_values_f32_ptr,
        tile_indices_i32_ptr,
        in_features,
        out_features,
        stream=stream,
        library=t16_library,
        runtime=runtime,
    )
    gguf_q6_k_pack8_top1_stage2_gather_f32(
        tile_values_f32_ptr,
        tile_indices_i32_ptr,
        out_indices_i32_ptr,
        out_values_f32_ptr,
        None,
        None,
        rows,
        out_features // _T16_COLS,
        0,
        out_features,
        stream=stream,
        library=pack8_library,
        runtime=runtime,
    )


def gguf_q6_k_t16_qmicro_planar_proposal_top1_exact_bf16(
    weight: object,
    x_ptr: int,
    logits_f32_ptr: int,
    tile_values_f32_ptr: int,
    tile_indices_i32_ptr: int,
    out_indices_i32_ptr: int,
    out_values_f32_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    libraries: Mapping[str, ctypes.CDLL] | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact planar-qmicro proposal logits and reduce to one winner."""

    if rows != 1:
        raise ValueError("Q6T16 proposal top-1 currently requires rows=1")
    allocation = getattr(weight, "allocation")("tiles")
    t16_library = None if libraries is None else libraries.get("q6_t16")
    pack8_library = None if libraries is None else libraries.get("q6_pack8")
    gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_top1_stage1(
        x_ptr,
        int(allocation.tensor.ptr),
        logits_f32_ptr,
        tile_values_f32_ptr,
        tile_indices_i32_ptr,
        in_features,
        out_features,
        stream=stream,
        library=t16_library,
        runtime=runtime,
    )
    gguf_q6_k_pack8_top1_stage2_gather_f32(
        tile_values_f32_ptr,
        tile_indices_i32_ptr,
        out_indices_i32_ptr,
        out_values_f32_ptr,
        None,
        None,
        rows,
        out_features // _T16_COLS,
        0,
        out_features,
        stream=stream,
        library=pack8_library,
        runtime=runtime,
    )


def gguf_q6_k_t16_gemv_rowtile_bf16_f32_out(
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
    """Small-B (rows 2-8) weight-amortized Q6T16 GEMV, BF16 in / FP32 out.

    Reads each weight tile once and reuses it across all rows; bit-identical to
    the per-row decode kernel. For the small-B verifier lm-head path.
    """

    _launch(
        _Q6_T16_ROWTILE_BF16_F32,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


setattr(gguf_q6_k_t16_gemv_rowtile_bf16_f32_out, "_hipengine_max_rows", 6)


def gguf_q6_k_t16_gemv_rowtile_bf16_bf16_out(
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
    """Small-B (rows 2-6) weight-amortized Q6T16 GEMV, BF16 in / BF16 out."""

    _launch(
        _Q6_T16_ROWTILE_BF16_BF16,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_gemv_rowtile_col8_bf16_f32_out(
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
    """Exact local128 small-B Q6T16 rowtile split into eight-column blocks."""

    _launch(
        _Q6_T16_ROWTILE_COL8_BF16_F32,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_gemv_rowtile_col8_bf16_bf16_out(
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
    """Exact local128 small-B Q6T16 col8 rowtile with BF16 output."""

    _launch(
        _Q6_T16_ROWTILE_COL8_BF16_BF16,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_gemv_rowtile12_col8_bf16_bf16_out(
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
    """Exact standard-Q6 col8 true-R12 candidate."""

    if rows != 12:
        raise ValueError("standard Q6 true-R12 owner requires rows == 12")
    _launch(
        _Q6_T16_ROWTILE12_COL8_BF16_BF16,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_bf16_out(
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
    """Exact planar-qmicro true col8 rowtile for rows 2-8."""

    if rows < 2 or rows > 8:
        raise ValueError("qmicro planar rowtile requires rows in [2, 8]")
    _launch(
        _Q6_T16_QMICRO_PLANAR_ROWTILE_COL8_BF16_BF16,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_qmicro_planar_q8_1_threads(
    rows: int,
    in_features: int,
    out_features: int,
) -> int:
    """Return the transaction-consistent thread policy for qualified shapes."""

    if rows < 1 or rows > 4:
        return 0
    shape = (int(in_features), int(out_features))
    if shape == (17_408, 5_120):
        return 256
    if shape == (5_120, 10_240):
        return 64
    return 0


def _launch_planar_q8_1(
    symbol: str,
    xq_ptr: int,
    tiles_ptr: int,
    residual_ptr: int | None,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    threads: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    if threads not in (64, 128, 256):
        raise ValueError("threads must be 64, 128, or 256")
    if in_features <= 0 or in_features % _QK_K != 0:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % _T16_COLS != 0:
        raise ValueError("out_features must be a positive multiple of 16")
    library = library or _q6_k_t16_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    if residual_ptr is None:
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        args = (
            ctypes.c_void_p(xq_ptr),
            ctypes.c_void_p(tiles_ptr),
            ctypes.c_void_p(out_ptr),
            ctypes.c_int64(rows),
            ctypes.c_int64(in_features),
            ctypes.c_int64(out_features),
            ctypes.c_int64(threads),
            ctypes.c_void_p(stream),
        )
    else:
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        args = (
            ctypes.c_void_p(xq_ptr),
            ctypes.c_void_p(tiles_ptr),
            ctypes.c_void_p(residual_ptr),
            ctypes.c_void_p(out_ptr),
            ctypes.c_int64(rows),
            ctypes.c_int64(in_features),
            ctypes.c_int64(out_features),
            ctypes.c_int64(threads),
            ctypes.c_void_p(stream),
        )
    fn.restype = ctypes.c_int
    err = fn(*args)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_bf16_bf16_out(
    xq_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch qualified planar-Q6/Q8_1 scalar-DP4A c1 or row reuse."""

    if rows < 1 or rows > 4:
        raise ValueError("planar Q8_1 projection rows must be in [1, 4]")
    resolved_threads = (
        gguf_q6_k_t16_qmicro_planar_q8_1_threads(
            rows,
            in_features,
            out_features,
        )
        if threads is None
        else int(threads)
    )
    if resolved_threads == 0:
        raise ValueError("planar Q8_1 projection shape has no qualified thread policy")
    _launch_planar_q8_1(
        _Q6_T16_QMICRO_PLANAR_Q8_1_DP4A_BF16_BF16,
        xq_ptr,
        tiles_ptr,
        None,
        out_ptr,
        rows,
        in_features,
        out_features,
        resolved_threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_bf16_residual_bf16_out(
    xq_ptr: int,
    tiles_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch approximate planar-Q6 projection plus rounded-BF16 residual."""

    if rows < 2 or rows > 4:
        raise ValueError("planar Q8_1 residual rows must be in [2, 4]")
    resolved_threads = (
        gguf_q6_k_t16_qmicro_planar_q8_1_threads(
            rows,
            in_features,
            out_features,
        )
        if threads is None
        else int(threads)
    )
    if resolved_threads == 0:
        raise ValueError("planar Q8_1 residual shape has no qualified thread policy")
    _launch_planar_q8_1(
        _Q6_T16_QMICRO_PLANAR_Q8_1_DP4A_BF16_RESIDUAL_BF16,
        xq_ptr,
        tiles_ptr,
        residual_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        resolved_threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _planar_q8_1_supports(rows: int, in_features: int, out_features: int) -> bool:
    return bool(
        gguf_q6_k_t16_qmicro_planar_q8_1_threads(
            rows,
            in_features,
            out_features,
        )
    )


setattr(
    gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_bf16_bf16_out,
    "_hipengine_supports",
    _planar_q8_1_supports,
)
setattr(
    gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_bf16_residual_bf16_out,
    "_hipengine_supports",
    _planar_q8_1_supports,
)


def gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_residual_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Exact planar-Q6 FFN-down plus rounded-BF16 residual for rows 2-4."""

    if rows < 2 or rows > 4:
        raise ValueError("qmicro planar down-residual requires rows in [2, 4]")
    _launch_residual(
        _Q6_T16_QMICRO_PLANAR_ROWTILE_COL8_BF16_RESIDUAL_BF16,
        x_ptr,
        tiles_ptr,
        residual_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_qmicro_planar_gemv_rowtile_bf16_f32_out(
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
    """Exact planar-qmicro FP32 rowtile with a 16-column rows=2 owner."""

    if rows < 2 or rows > 8:
        raise ValueError("qmicro planar rowtile requires rows in [2, 8]")
    _launch(
        _Q6_T16_QMICRO_PLANAR_ROWTILE_BF16_F32,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


setattr(
    gguf_q6_k_t16_qmicro_planar_gemv_rowtile_bf16_f32_out,
    "_hipengine_max_rows",
    4,
)


def gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_f32_out(
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
    """Exact planar-qmicro FP32 rowtile for rows 2-4, scalar to 6."""

    if rows < 2 or rows > 6:
        raise ValueError("qmicro planar rowtile requires rows in [2, 6]")
    _launch(
        _Q6_T16_QMICRO_PLANAR_ROWTILE_COL8_BF16_F32,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _launch_residual(
    symbol: str,
    x_ptr: int,
    tiles_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    if in_features <= 0 or in_features % _QK_K != 0:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % _T16_COLS != 0:
        raise ValueError("out_features must be a positive multiple of 16")
    library = library or _q6_k_t16_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
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
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(residual_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch(
    symbol: str,
    x_ptr: int,
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
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0 or in_features % _QK_K != 0:
        raise ValueError("in_features must be a positive multiple of 256")
    if out_features <= 0 or out_features % _T16_COLS != 0:
        raise ValueError("out_features must be a positive multiple of 16")
    library = library or _q6_k_t16_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
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
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_gguf_q6_k_t16_gemv_kernels(*, replace: bool = True) -> None:
    """Register dense Q6T16 GEMV decode kernels."""

    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q6_k_t16_v1", "t16_gemv_decode_bf16_f32_out"),
        gguf_q6_k_t16_gemv_decode_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q6_k_t16_v1", "t16_gemv_decode_bf16_bf16_out"),
        gguf_q6_k_t16_gemv_decode_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_v1",
            "t16_gemv_rowtile_bf16_f32_out",
        ),
        gguf_q6_k_t16_gemv_rowtile_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_v1",
            "t16_gemv_rowtile_bf16_bf16_out",
        ),
        gguf_q6_k_t16_gemv_rowtile_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_v1",
            "t16_gemv_rowtile_col8_bf16_f32_out",
        ),
        gguf_q6_k_t16_gemv_rowtile_col8_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_v1",
            "t16_gemv_rowtile_col8_bf16_bf16_out",
        ),
        gguf_q6_k_t16_gemv_rowtile_col8_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_v1",
            "t16_gemv_rowtile12_col8_bf16_bf16_out",
        ),
        gguf_q6_k_t16_gemv_rowtile12_col8_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_gemv_decode_bf16_bf16_out",
        ),
        gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear+residual",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_gemv_decode_bf16_residual_bf16_out",
        ),
        gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_residual_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_gemv_decode_bf16_f32_out",
        ),
        gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_gemv_rowtile_bf16_bf16_out",
        ),
        gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_gemv_rowtile_col8_bf16_bf16_out",
        ),
        gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_q8_1",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_q8_1_dp4a_gemv_bf16_bf16_out",
        ),
        gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_q8_1+residual",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_q8_1_dp4a_gemv_bf16_residual_bf16_out",
        ),
        gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_bf16_residual_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear+residual",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_gemv_rowtile_bf16_residual_bf16_out",
        ),
        gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_residual_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_gemv_rowtile_bf16_f32_out",
        ),
        gguf_q6_k_t16_qmicro_planar_gemv_rowtile_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_wmma_prefill_bf16_bf16_out",
        ),
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_wmma_prefill_shared4_bf16_bf16_out",
        ),
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear+argmax",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_gemv_decode_bf16_f32_top1_stage1",
        ),
        gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_top1_stage1,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear+argmax",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "proposal_top1_exact_bf16",
        ),
        gguf_q6_k_t16_qmicro_planar_proposal_top1_exact_bf16,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_v1",
            "t16_wmma_prefill_bf16_bf16_out",
        ),
        gguf_q6_k_t16_wmma_prefill_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_v1",
            "t16_wmma_prefill_shared4_bf16_bf16_out",
        ),
        gguf_q6_k_t16_wmma_prefill_shared4_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear+argmax",
            "gguf_q6_k_t16_v1",
            "t16_gemv_decode_bf16_f32_top1_stage1",
        ),
        gguf_q6_k_t16_gemv_decode_bf16_f32_top1_stage1,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear+argmax",
            "gguf_q6_k_t16_v1",
            "proposal_top1_exact_bf16",
        ),
        gguf_q6_k_t16_proposal_top1_exact_bf16,
        replace=replace,
    )


register_gguf_q6_k_t16_gemv_kernels()


__all__ = [
    "build_gguf_q6_k_t16_gemv",
    "gguf_q6_k_t16_gemv_decode_bf16_bf16_out",
    "gguf_q6_k_t16_gemv_decode_bf16_f32_out",
    "gguf_q6_k_t16_gemv_decode_bf16_f32_top1_stage1",
    "gguf_q6_k_t16_gemv_rowtile_bf16_bf16_out",
    "gguf_q6_k_t16_gemv_rowtile_bf16_f32_out",
    "gguf_q6_k_t16_gemv_rowtile_col8_bf16_bf16_out",
    "gguf_q6_k_t16_gemv_rowtile12_col8_bf16_bf16_out",
    "gguf_q6_k_t16_gemv_rowtile_col8_bf16_f32_out",
    "gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out",
    "gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_residual_bf16_out",
    "gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_out",
    "gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_top1_stage1",
    "gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_bf16_out",
    "gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_residual_bf16_out",
    "gguf_q6_k_t16_qmicro_planar_gemv_rowtile_bf16_f32_out",
    "gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_f32_out",
    "gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_bf16_bf16_out",
    "gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_bf16_residual_bf16_out",
    "gguf_q6_k_t16_qmicro_planar_q8_1_threads",
    "gguf_q6_k_t16_qmicro_planar_proposal_top1_exact_bf16",
    "gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out",
    "gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr_bf16_bf16_out",
    "gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr48_bf16_bf16_out",
    "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r4_bf16_bf16_out",
    "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r3_bf16_bf16_out",
    "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared3r1_bf16_bf16_out",
    "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_bf16_out",
    "gguf_q6_k_t16_proposal_top1_exact_bf16",
    "gguf_q6_k_t16_wmma_prefill_bf16_bf16_out",
    "gguf_q6_k_t16_wmma_prefill_shared3r1_bf16_bf16_out",
    "gguf_q6_k_t16_wmma_prefill_shared4_bf16_bf16_out",
    "gguf_q6_k_t16_wmma_prefill_shared6r1_bf16_bf16_out",
    "gguf_q6_k_t16_wmma_prefill_shared8r3_bf16_bf16_out",
    "plan_gguf_q6_k_t16_gemv_build",
    "register_gguf_q6_k_t16_gemv_kernels",
]
