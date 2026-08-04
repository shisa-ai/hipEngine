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
_Q6_T16_WMMA_PREFILL_BF16_BF16 = (
    "hipengine_gguf_q6_k_t16_wmma_prefill_bf16_bf16_out"
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
    library = library or build_gguf_q6_k_t16_gemv(load=True)
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
    """Small-B (rows 2-6) weight-amortized Q6T16 GEMV, BF16 in / FP32 out.

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
    library = library or build_gguf_q6_k_t16_gemv(load=True)
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
            "t16_wmma_prefill_bf16_bf16_out",
        ),
        gguf_q6_k_t16_wmma_prefill_bf16_bf16_out,
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
    "gguf_q6_k_t16_gemv_rowtile_col8_bf16_f32_out",
    "gguf_q6_k_t16_proposal_top1_exact_bf16",
    "gguf_q6_k_t16_wmma_prefill_bf16_bf16_out",
    "plan_gguf_q6_k_t16_gemv_build",
    "register_gguf_q6_k_t16_gemv_kernels",
]
