"""Wrappers for compact selected-MoE WMMA prefill on GGUF Q4_K / Q5_K / Q6_K T16 tiles.

P10.B2 / P10.B3 / Q4_K_S follow-up: ports selected single-output WMMA prefill
kernels (``gguf_k_selected_prefill.hip``) to consume the T16 replacement
layout used by the decode-repack path. The exported callables share the
``selected_wmma_prefill_compact_bf16_bf16_out`` ABI used by the raw
versions so dispatch can swap quant keys without runtime / backend
branches.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.hip_gfx1100 import (
    GGUF_Q4_T16_PHYSICAL_C1_ROWTILE_ROWS,
    GGUF_Q4_T16_PHYSICAL_C1_ROWTILE_SHAPES,
    GGUF_Q4_T16_PHYSICAL_SINGLE_WAVE_SHAPES,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    gguf_q4_k_t16_dense_rowtile_bf16_bf16_out,
)
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_k_t16_selected_prefill.hip")
_OUTPUT_NAME = "gguf_k_t16_selected_prefill.so"
_ENV_LAUNCH_BOUNDS = "HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS"
_QK_K = 256

_SYMBOLS = {
    ("gguf_q4_k_t16", "bf16"): "hipengine_gguf_q4_k_t16_selected_wmma_prefill_compact_bf16_bf16_out",
    ("gguf_q4_k_t16", "fp16"): "hipengine_gguf_q4_k_t16_selected_wmma_prefill_compact_fp16_fp16_out",
    ("gguf_q5_k_t16", "bf16"): "hipengine_gguf_q5_k_t16_selected_wmma_prefill_compact_bf16_bf16_out",
    ("gguf_q5_k_t16", "fp16"): "hipengine_gguf_q5_k_t16_selected_wmma_prefill_compact_fp16_fp16_out",
    ("gguf_q5_k_qmicro_t16", "bf16"): "hipengine_gguf_q5_k_qmicro_t16_selected_wmma_prefill_compact_bf16_bf16_out",
    ("gguf_q5_k_qmicro_t16", "fp16"): "hipengine_gguf_q5_k_qmicro_t16_selected_wmma_prefill_compact_fp16_fp16_out",
    ("gguf_q6_k_t16", "bf16"): "hipengine_gguf_q6_k_t16_selected_wmma_prefill_compact_bf16_bf16_out",
    ("gguf_q6_k_t16", "fp16"): "hipengine_gguf_q6_k_t16_selected_wmma_prefill_compact_fp16_fp16_out",
}
_Q4_DENSE_WMMA_BF16 = (
    "hipengine_gguf_q4_k_t16_wmma_prefill_bf16_bf16_out"
)
_Q4_QMICRO_DENSE_WMMA_BF16 = (
    "hipengine_gguf_q4_k_qmicro_t16_wmma_prefill_bf16_bf16_out"
)
_Q4_DENSE_WMMA_SMALLM_BF16 = (
    "hipengine_gguf_q4_k_t16_wmma_prefill_smallm_bf16_bf16_out"
)
_Q4_DENSE_WMMA_LOWVGPR_BF16 = (
    "hipengine_gguf_q4_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out"
)
_Q4_DENSE_WMMA_LOWVGPR48_BF16 = (
    "hipengine_gguf_q4_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out"
)
_Q4_DENSE_WMMA_SHARED_B_BF16 = (
    "hipengine_gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out"
)
_Q4_DENSE_WMMA_SHARED_B2W2_BF16 = (
    "hipengine_gguf_q4_k_t16_wmma_prefill_shared_b2w2_bf16_bf16_out"
)
_Q4_DENSE_WMMA_SHARED_B3W8R3_BF16 = (
    "hipengine_gguf_q4_k_t16_wmma_prefill_shared_b3w8r3_bf16_bf16_out"
)
_Q4_DENSE_WMMA_SHARED_B2R1_BF16 = (
    "hipengine_gguf_q4_k_t16_wmma_prefill_shared_b2r1_bf16_bf16_out"
)
_Q4_DENSE_WMMA_SHARED_B2W4_BF16 = (
    "hipengine_gguf_q4_k_t16_wmma_prefill_shared_b2w4_bf16_bf16_out"
)
_Q4_DENSE_DUAL_WMMA_SILU_BF16 = (
    "hipengine_gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out"
)
_Q4_DENSE_DUAL_WMMA_SMALLM_SILU_BF16 = (
    "hipengine_gguf_q4_k_t16_dense_dual_wmma_smallm_silu_bf16_bf16_out"
)
_Q4_QMICRO_DENSE_DUAL_WMMA_SILU_BF16 = (
    "hipengine_gguf_q4_k_qmicro_t16_dense_dual_wmma_prefill_silu_"
    "bf16_bf16_out"
)
_Q4_QMICRO_DENSE_DUAL_WMMA_EXPANDED_META_SILU_BF16 = (
    "hipengine_gguf_q4_k_qmicro_t16_dense_dual_wmma_prefill_"
    "expanded_meta_silu_bf16_bf16_out"
)
_Q4_DENSE_UNEQUAL_DUAL_WMMA_BF16 = (
    "hipengine_gguf_q4_k_t16_dense_unequal_dual_wmma_prefill_bf16_bf16_out"
)
_Q5_DENSE_WMMA_BF16 = (
    "hipengine_gguf_q5_k_t16_wmma_prefill_bf16_bf16_out"
)
_Q5_DENSE_WMMA_SHARED8R3_BF16 = (
    "hipengine_gguf_q5_k_t16_wmma_prefill_shared8r3_bf16_bf16_out"
)
_Q5_DENSE_WMMA_LOWVGPR_BF16 = (
    "hipengine_gguf_q5_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out"
)
_Q5_DENSE_WMMA_LOWVGPR48_BF16 = (
    "hipengine_gguf_q5_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out"
)
_EXPERT_MAJOR_COMP_SYMBOLS = {
    "gguf_q4_k_t16": "hipengine_gguf_q4_k_t16_selected_expert_major_wmma_comp_bf16_bf16_out",
    "gguf_q6_k_t16": "hipengine_gguf_q6_k_t16_selected_expert_major_wmma_comp_bf16_bf16_out",
}


def _extra_flags() -> tuple[str, ...]:
    value = os.environ.get(_ENV_LAUNCH_BOUNDS)
    if not value:
        return ("-mcumode",)
    min_blocks = int(value)
    if min_blocks not in {1, 2, 4, 8}:
        raise ValueError(f"{_ENV_LAUNCH_BOUNDS} must be one of 1, 2, 4, 8")
    return ("-mcumode", f"-DHIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS={min_blocks}")


def plan_gguf_k_t16_selected_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_k_t16_selected_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_extra_flags(),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_k_t16_selected_prefill(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_hip(
        sources=[_SOURCE],
        family="gguf_k_t16_selected_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_extra_flags(),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _make_wrapper(quant: str, dtype: str):
    symbol = _SYMBOLS[(quant, dtype)]

    def wrapper(
        x_ptr: int,
        expert_start_compact_ptr: int,
        expert_start_wmma_ptr: int,
        tile_expert_ptr: int,
        tiles_ptr: int,
        out_ptr: int,
        compact_rows: int,
        in_features: int,
        out_features: int,
        num_experts: int,
        wmma_total_rows: int,
        *,
        stream: int = 0,
        library: ctypes.CDLL | None = None,
        runtime: HipRuntime | None = None,
        # Accepted-but-ignored kwargs so dispatch can call this wrapper with
        # the same (tile_m, tile_n) keyword arguments accepted by the raw
        # ``gguf_k_selected_prefill`` wrappers. The T16 kernel ships a single
        # 16x16 tile shape; the per-quant tile sweep belongs to P10.C2.
        tile_m: int | None = None,
        tile_n: int | None = None,
    ) -> None:
        del tile_m, tile_n  # tile sweep is P10.C2, not P10.B2/B3
        _launch_k_t16(
            symbol,
            x_ptr,
            expert_start_compact_ptr,
            expert_start_wmma_ptr,
            tile_expert_ptr,
            tiles_ptr,
            out_ptr,
            compact_rows,
            in_features,
            out_features,
            num_experts,
            wmma_total_rows,
            stream=stream,
            library=library,
            runtime=runtime,
        )

    wrapper.__name__ = f"{quant}_selected_wmma_prefill_compact_{dtype}_{dtype}_out"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__doc__ = (
        f"Launch {quant} T16 selected compact single-output WMMA prefill "
        f"({dtype}->{dtype})."
    )
    return wrapper


# Public wrapper functions.
gguf_q4_k_t16_selected_wmma_prefill_compact_bf16_bf16_out = _make_wrapper("gguf_q4_k_t16", "bf16")
gguf_q4_k_t16_selected_wmma_prefill_compact_fp16_fp16_out = _make_wrapper("gguf_q4_k_t16", "fp16")
gguf_q5_k_t16_selected_wmma_prefill_compact_bf16_bf16_out = _make_wrapper("gguf_q5_k_t16", "bf16")
gguf_q5_k_t16_selected_wmma_prefill_compact_fp16_fp16_out = _make_wrapper("gguf_q5_k_t16", "fp16")
gguf_q5_k_qmicro_t16_selected_wmma_prefill_compact_bf16_bf16_out = _make_wrapper(
    "gguf_q5_k_qmicro_t16", "bf16"
)
gguf_q5_k_qmicro_t16_selected_wmma_prefill_compact_fp16_fp16_out = _make_wrapper(
    "gguf_q5_k_qmicro_t16", "fp16"
)
gguf_q6_k_t16_selected_wmma_prefill_compact_bf16_bf16_out = _make_wrapper("gguf_q6_k_t16", "bf16")
gguf_q6_k_t16_selected_wmma_prefill_compact_fp16_fp16_out = _make_wrapper("gguf_q6_k_t16", "fp16")


def gguf_q4_k_t16_wmma_prefill_bf16_bf16_out(
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
    tile_m: int | None = None,
    tile_n: int | None = None,
) -> None:
    """Launch dense one-expert Q4T16 WMMA prefill (BF16->BF16)."""

    del tile_m, tile_n
    _launch_dense_t16(
        _Q4_DENSE_WMMA_BF16,
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


def gguf_q4_k_t16_wmma_prefill_smallm_bf16_bf16_out(
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
    tile_m: int | None = None,
    tile_n: int | None = None,
) -> None:
    """Launch the strict one-WMMA-row-tile Q4T16 owner for rows 1-16."""

    del tile_m, tile_n
    if int(rows) > 16:
        raise ValueError("small-M Q4T16 WMMA requires rows <= 16")
    _launch_dense_t16(
        _Q4_DENSE_WMMA_SMALLM_BF16,
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


def gguf_q4_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out(
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
    tile_m: int | None = None,
    tile_n: int | None = None,
) -> None:
    """Launch the low-VGPR 16-column Q4T16 owner for latency-bound low-M slabs.

    One 16-column output tile and two 16-row tiles per 32-thread block: the
    accumulator footprint drops to 16 floats so more waves fit per SIMD.
    Identical per-tile K16 WMMA association to the 48-column single-wave
    owner (bit-exact).
    """

    del tile_m, tile_n
    _launch_dense_t16(
        _Q4_DENSE_WMMA_LOWVGPR_BF16,
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


def gguf_q4_k_t16_physical_c1_rowtile_gfx1100_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    **kwargs,
) -> None:
    """Select the C1-equivalent rowtile for the admitted physical R6 shapes."""

    shape = (int(in_features), int(out_features))
    if int(rows) not in GGUF_Q4_T16_PHYSICAL_C1_ROWTILE_ROWS:
        fn = gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out
    elif shape in GGUF_Q4_T16_PHYSICAL_C1_ROWTILE_SHAPES:
        fn = gguf_q4_k_t16_dense_rowtile_bf16_bf16_out
    elif shape in GGUF_Q4_T16_PHYSICAL_SINGLE_WAVE_SHAPES:
        fn = gguf_q4_k_t16_wmma_prefill_bf16_bf16_out
    else:
        fn = gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out
    return fn(
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    )


def gguf_q4_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out(
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
    tile_m: int | None = None,
    tile_n: int | None = None,
) -> None:
    """Launch the low-VGPR 48-row-block 16-column Q4T16 owner.

    One 16-column output tile and three 16-row tiles per 32-thread block
    (24-float accumulator): one row-block for slabs up to 48 rows, avoiding
    the 32-row block-boundary padding. Identical per-tile K16 WMMA
    association to the 48-column single-wave owner (bit-exact).
    """

    del tile_m, tile_n
    _launch_dense_t16(
        _Q4_DENSE_WMMA_LOWVGPR48_BF16,
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


def gguf_q4_k_qmicro_t16_wmma_prefill_bf16_bf16_out(
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
    tile_m: int | None = None,
    tile_n: int | None = None,
) -> None:
    """Launch the exact sole-qmicro WMMA prefill primitive."""

    del tile_m, tile_n
    _launch_dense_t16(
        _Q4_QMICRO_DENSE_WMMA_BF16,
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


def gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(
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
    tile_m: int | None = None,
    tile_n: int | None = None,
) -> None:
    """Launch cooperative dense Q4T16 WMMA prefill (BF16->BF16)."""

    del tile_m, tile_n
    _launch_dense_t16(
        _Q4_DENSE_WMMA_SHARED_B_BF16,
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


def gguf_q4_k_t16_wmma_prefill_shared_b2r1_bf16_bf16_out(
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
    tile_m: int | None = None,
    tile_n: int | None = None,
) -> None:
    """Launch 32-column shared-B with one row tile per wave."""

    del tile_m, tile_n
    _launch_dense_t16(
        _Q4_DENSE_WMMA_SHARED_B2R1_BF16,
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


def gguf_q4_k_t16_wmma_prefill_shared_b3w8r3_bf16_bf16_out(
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
    tile_m: int | None = None,
    tile_n: int | None = None,
) -> None:
    """Launch 48-column/384-row eight-wave shared-B prefill."""

    del tile_m, tile_n
    _launch_dense_t16(
        _Q4_DENSE_WMMA_SHARED_B3W8R3_BF16,
        x_ptr, tiles_ptr, out_ptr, rows, in_features, out_features,
        stream=stream, library=library, runtime=runtime,
    )


def gguf_q4_k_t16_wmma_prefill_shared_b2w2_bf16_bf16_out(
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
    tile_m: int | None = None,
    tile_n: int | None = None,
) -> None:
    """Launch 32-column/128-row cooperative Q4T16 prefill."""

    del tile_m, tile_n
    _launch_dense_t16(
        _Q4_DENSE_WMMA_SHARED_B2W2_BF16,
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


def gguf_q4_k_t16_wmma_prefill_shared_b2w4_bf16_bf16_out(
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
    tile_m: int | None = None,
    tile_n: int | None = None,
) -> None:
    """Launch 32-column/256-row cooperative Q4T16 prefill."""

    del tile_m, tile_n
    _launch_dense_t16(
        _Q4_DENSE_WMMA_SHARED_B2W4_BF16,
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


def gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out(
    x_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch operation-complete dense Q4T16 gate/up WMMA prefill + SiLU."""

    _check_positive(rows, "rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_features, "out_features")
    if in_features % _QK_K != 0:
        raise ValueError(
            f"in_features must be divisible by GGUF K-family block size {_QK_K}"
        )
    if out_features % 32 != 0:
        raise ValueError("out_features must be a multiple of 32")
    lib = library or build_gguf_k_t16_selected_prefill(load=True)
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_DENSE_DUAL_WMMA_SILU_BF16)
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
        ctypes.c_void_p(tiles_a_ptr),
        ctypes.c_void_p(tiles_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        rt.check(int(err))


def gguf_q4_k_t16_dense_dual_wmma_smallm_silu_bf16_bf16_out(
    x_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the exact two-wave Q4T16 gate/up owner for rows 2-16."""

    if not 2 <= int(rows) <= 16:
        raise ValueError("small-M dual Q4T16 WMMA requires rows in 2..16")
    _check_positive(in_features, "in_features")
    _check_positive(out_features, "out_features")
    if in_features % _QK_K != 0:
        raise ValueError(
            f"in_features must be divisible by GGUF K-family block size {_QK_K}"
        )
    if out_features % 16 != 0:
        raise ValueError("out_features must be a multiple of 16")
    lib = library or build_gguf_k_t16_selected_prefill(load=True)
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_DENSE_DUAL_WMMA_SMALLM_SILU_BF16)
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
        ctypes.c_void_p(tiles_a_ptr),
        ctypes.c_void_p(tiles_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        rt.check(int(err))


def gguf_q4_k_qmicro_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out(
    x_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch sole-qmicro dense Q4 gate/up WMMA prefill + SiLU."""

    _check_positive(rows, "rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_features, "out_features")
    if in_features % _QK_K != 0:
        raise ValueError(
            f"in_features must be divisible by GGUF K-family block size {_QK_K}"
        )
    if out_features % 32 != 0:
        raise ValueError("out_features must be a multiple of 32")
    lib = library or build_gguf_k_t16_selected_prefill(load=True)
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_QMICRO_DENSE_DUAL_WMMA_SILU_BF16)
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
        ctypes.c_void_p(tiles_a_ptr),
        ctypes.c_void_p(tiles_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        rt.check(int(err))


def gguf_q4_k_qmicro_t16_dense_dual_wmma_prefill_expanded_meta_silu_bf16_bf16_out(
    x_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    metadata_a_ptr: int,
    metadata_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Expand compact metadata, then run exact dense Q4 gate/up WMMA+SiLU."""

    _check_positive(rows, "rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_features, "out_features")
    if in_features % _QK_K != 0:
        raise ValueError(
            f"in_features must be divisible by GGUF K-family block size {_QK_K}"
        )
    if out_features % 32 != 0:
        raise ValueError("out_features must be a multiple of 32")
    if int(metadata_a_ptr) <= 0 or int(metadata_b_ptr) <= 0:
        raise ValueError("expanded metadata workspaces must be non-null")
    lib = library or build_gguf_k_t16_selected_prefill(load=True)
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_QMICRO_DENSE_DUAL_WMMA_EXPANDED_META_SILU_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_void_p(tiles_a_ptr),
        ctypes.c_void_p(tiles_b_ptr),
        ctypes.c_void_p(metadata_a_ptr),
        ctypes.c_void_p(metadata_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        rt.check(int(err))


def gguf_q4_k_t16_dense_unequal_dual_wmma_prefill_bf16_bf16_out(
    x_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact unequal-width Q4T16 dual WMMA prefill (BF16->BF16)."""

    _check_positive(rows, "rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_features_a, "out_features_a")
    _check_positive(out_features_b, "out_features_b")
    if in_features % _QK_K != 0:
        raise ValueError(
            f"in_features must be divisible by GGUF K-family block size {_QK_K}"
        )
    if out_features_a < out_features_b:
        raise ValueError("out_features_a must be at least out_features_b")
    if out_features_a % 32 != 0 or out_features_b % 32 != 0:
        raise ValueError("out_features_a and out_features_b must be multiples of 32")
    lib = library or build_gguf_k_t16_selected_prefill(load=True)
    rt = runtime or get_hip_runtime()
    fn = getattr(lib, _Q4_DENSE_UNEQUAL_DUAL_WMMA_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
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
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(tiles_a_ptr),
        ctypes.c_void_p(tiles_b_ptr),
        ctypes.c_void_p(out_a_ptr),
        ctypes.c_void_p(out_b_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        rt.check(int(err))


def gguf_q5_k_t16_wmma_prefill_bf16_bf16_out(
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
    tile_m: int | None = None,
    tile_n: int | None = None,
) -> None:
    """Launch dense one-expert Q5T16 WMMA prefill (BF16->BF16)."""

    del tile_m, tile_n
    _launch_dense_t16(
        _Q5_DENSE_WMMA_BF16,
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


def gguf_q5_k_t16_wmma_prefill_shared8r3_bf16_bf16_out(
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
    tile_m: int | None = None,
    tile_n: int | None = None,
) -> None:
    """Launch the eight-wave, three-row-tile shared Q5T16 owner."""

    del tile_m, tile_n
    _launch_dense_t16(
        _Q5_DENSE_WMMA_SHARED8R3_BF16,
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


def gguf_q5_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out(
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
    tile_m: int | None = None,
    tile_n: int | None = None,
) -> None:
    """Launch the low-VGPR 16-column Q5T16 owner (32-row blocks)."""

    del tile_m, tile_n
    _launch_dense_t16(
        _Q5_DENSE_WMMA_LOWVGPR_BF16,
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


def gguf_q5_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out(
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
    tile_m: int | None = None,
    tile_n: int | None = None,
) -> None:
    """Launch the low-VGPR 16-column Q5T16 owner (48-row blocks)."""

    del tile_m, tile_n
    _launch_dense_t16(
        _Q5_DENSE_WMMA_LOWVGPR48_BF16,
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


def _make_expert_major_comp_wrapper(quant: str):
    symbol = _EXPERT_MAJOR_COMP_SYMBOLS[quant]

    def wrapper(
        x_ptr: int,
        expert_start_compact_ptr: int,
        active_experts_ptr: int,
        active_count_ptr: int,
        tiles_ptr: int,
        out_ptr: int,
        compact_rows: int,
        in_features: int,
        out_features: int,
        num_experts: int,
        *,
        stream: int = 0,
        library: ctypes.CDLL | None = None,
        runtime: HipRuntime | None = None,
    ) -> None:
        _launch_expert_major_comp(
            symbol,
            x_ptr,
            expert_start_compact_ptr,
            active_experts_ptr,
            active_count_ptr,
            tiles_ptr,
            out_ptr,
            compact_rows,
            in_features,
            out_features,
            num_experts,
            stream=stream,
            library=library,
            runtime=runtime,
        )

    wrapper.__name__ = (
        f"{quant}_selected_expert_major_wmma_comp_bf16_bf16_out"
    )
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__doc__ = (
        f"Launch {quant} expert-major compensated WMMA prefill (BF16->BF16)."
    )
    return wrapper


gguf_q4_k_t16_selected_expert_major_wmma_comp_bf16_bf16_out = (
    _make_expert_major_comp_wrapper("gguf_q4_k_t16")
)
gguf_q6_k_t16_selected_expert_major_wmma_comp_bf16_bf16_out = (
    _make_expert_major_comp_wrapper("gguf_q6_k_t16")
)


def _launch_k_t16(
    symbol: str,
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_common(compact_rows, in_features, out_features, num_experts, wmma_total_rows)
    library = library or build_gguf_k_t16_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_dense_t16(
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
    _check_positive(rows, "rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_features, "out_features")
    if in_features % _QK_K != 0:
        raise ValueError(
            f"in_features must be divisible by GGUF K-family block size {_QK_K}"
        )
    if out_features % 16 != 0:
        raise ValueError("out_features must be a multiple of 16")

    library = library or build_gguf_k_t16_selected_prefill(load=True)
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


def _launch_expert_major_comp(
    symbol: str,
    x_ptr: int,
    expert_start_compact_ptr: int,
    active_experts_ptr: int,
    active_count_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_expert_major_common(
        compact_rows, in_features, out_features, num_experts
    )
    library = library or build_gguf_k_t16_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
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
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(active_experts_ptr),
        ctypes.c_void_p(active_count_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _check_expert_major_common(
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
) -> None:
    _check_positive(compact_rows, "compact_rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_features, "out_features")
    _check_positive(num_experts, "num_experts")
    if in_features % _QK_K != 0:
        raise ValueError(
            f"in_features must be divisible by GGUF K-family block size {_QK_K}"
        )
    if out_features % 16 != 0:
        raise ValueError("out_features must be a multiple of 16")


def _check_common(
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    wmma_total_rows: int,
) -> None:
    _check_positive(compact_rows, "compact_rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_features, "out_features")
    _check_positive(num_experts, "num_experts")
    _check_positive(wmma_total_rows, "wmma_total_rows")
    if in_features % _QK_K != 0:
        raise ValueError(f"in_features must be divisible by GGUF K-family block size {_QK_K}")
    if out_features % 16 != 0:
        raise ValueError("out_features must be a multiple of 16")
    if wmma_total_rows % 16 != 0:
        raise ValueError("wmma_total_rows must be a multiple of 16")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def register_gguf_k_t16_selected_prefill_kernels(*, replace: bool = True) -> None:
    """Register Q4T16/Q5T16/Q6T16 selected WMMA prefill kernels.

    Each kernel is registered under its native ``gguf_q*_k_t16_v1`` quant key
    using the shared ``selected_wmma_prefill_compact_*`` alias spelling so
    ``_COMPACT_MOE_DOWN_KEYS`` in the runner can route on quant key alone.
    """

    for quant_key, fn in (
        (
            "gguf_q4_k_t16_v1",
            gguf_q4_k_t16_physical_c1_rowtile_gfx1100_bf16_bf16_out,
        ),
        (
            "gguf_q4_k_qmicro_t16_v1",
            gguf_q4_k_qmicro_t16_wmma_prefill_bf16_bf16_out,
        ),
        ("gguf_q5_k_t16_v1", gguf_q5_k_t16_wmma_prefill_bf16_bf16_out),
    ):
        register(
            KernelKey(
                "hip_gfx1100",
                "linear",
                quant_key,
                "t16_wmma_prefill_bf16_bf16_out",
            ),
            fn,
            replace=replace,
        )

    for variant, fn in (
        (
            "t16_physical_c1_rowtile_bf16_bf16_out",
            gguf_q4_k_t16_dense_rowtile_bf16_bf16_out,
        ),
        (
            "t16_wmma_prefill_single_wave_bf16_bf16_out",
            gguf_q4_k_t16_wmma_prefill_bf16_bf16_out,
        ),
        (
            "t16_wmma_prefill_smallm_bf16_bf16_out",
            gguf_q4_k_t16_wmma_prefill_smallm_bf16_bf16_out,
        ),
        (
            "t16_wmma_prefill_shared_b_bf16_bf16_out",
            gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out,
        ),
        (
            "t16_wmma_prefill_shared_b2r1_bf16_bf16_out",
            gguf_q4_k_t16_wmma_prefill_shared_b2r1_bf16_bf16_out,
        ),
    ):
        register(
            KernelKey(
                "hip_gfx1100",
                "linear",
                "gguf_q4_k_t16_v1",
                variant,
            ),
            fn,
            replace=replace,
        )

    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_q4_k_t16_v1",
            "dense_dual_wmma_prefill_bf16_bf16_out",
        ),
        gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_q4_k_t16_v1",
            "dense_dual_wmma_smallm_bf16_bf16_out",
        ),
        gguf_q4_k_t16_dense_dual_wmma_smallm_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_q4_k_qmicro_t16_v1",
            "dense_dual_wmma_prefill_bf16_bf16_out",
        ),
        gguf_q4_k_qmicro_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_q4_k_qmicro_t16_v1",
            "dense_dual_wmma_prefill_expanded_meta_bf16_bf16_out",
        ),
        gguf_q4_k_qmicro_t16_dense_dual_wmma_prefill_expanded_meta_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair",
            "gguf_q4_k_t16_v1",
            "dense_unequal_dual_wmma_prefill_bf16_bf16_out",
        ),
        gguf_q4_k_t16_dense_unequal_dual_wmma_prefill_bf16_bf16_out,
        replace=replace,
    )

    for quant_key, fn_bf16, fn_fp16 in (
        (
            "gguf_q4_k_t16_v1",
            gguf_q4_k_t16_selected_wmma_prefill_compact_bf16_bf16_out,
            gguf_q4_k_t16_selected_wmma_prefill_compact_fp16_fp16_out,
        ),
        (
            "gguf_q5_k_t16_v1",
            gguf_q5_k_t16_selected_wmma_prefill_compact_bf16_bf16_out,
            gguf_q5_k_t16_selected_wmma_prefill_compact_fp16_fp16_out,
        ),
        (
            "gguf_q5_k_qmicro_t16_v1",
            gguf_q5_k_qmicro_t16_selected_wmma_prefill_compact_bf16_bf16_out,
            gguf_q5_k_qmicro_t16_selected_wmma_prefill_compact_fp16_fp16_out,
        ),
        (
            "gguf_q6_k_t16_v1",
            gguf_q6_k_t16_selected_wmma_prefill_compact_bf16_bf16_out,
            gguf_q6_k_t16_selected_wmma_prefill_compact_fp16_fp16_out,
        ),
    ):
        register(
            KernelKey(
                "hip_gfx1100",
                "moe_linear",
                quant_key,
                "selected_wmma_prefill_compact_bf16_bf16_out",
            ),
            fn_bf16,
            replace=replace,
        )
        register(
            KernelKey(
                "hip_gfx1100",
                "moe_linear",
                quant_key,
                "selected_wmma_prefill_compact_fp16_fp16_out",
            ),
            fn_fp16,
            replace=replace,
        )

    for quant_key, fn in (
        (
            "gguf_q4_k_t16_v1",
            gguf_q4_k_t16_selected_expert_major_wmma_comp_bf16_bf16_out,
        ),
        (
            "gguf_q6_k_t16_v1",
            gguf_q6_k_t16_selected_expert_major_wmma_comp_bf16_bf16_out,
        ),
    ):
        register(
            KernelKey(
                "hip_gfx1100",
                "moe_linear",
                quant_key,
                "selected_t16_expert_major_wmma_comp_bf16_bf16_out",
            ),
            fn,
            replace=replace,
        )


register_gguf_k_t16_selected_prefill_kernels()


__all__ = [
    "build_gguf_k_t16_selected_prefill",
    "gguf_q4_k_t16_selected_expert_major_wmma_comp_bf16_bf16_out",
    "gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out",
    "gguf_q4_k_t16_dense_dual_wmma_smallm_silu_bf16_bf16_out",
    "gguf_q4_k_qmicro_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out",
    "gguf_q4_k_qmicro_t16_dense_dual_wmma_prefill_expanded_meta_silu_bf16_bf16_out",
    "gguf_q4_k_qmicro_t16_wmma_prefill_bf16_bf16_out",
    "gguf_q4_k_t16_dense_unequal_dual_wmma_prefill_bf16_bf16_out",
    "gguf_q4_k_t16_physical_c1_rowtile_gfx1100_bf16_bf16_out",
    "gguf_q4_k_t16_wmma_prefill_bf16_bf16_out",
    "gguf_q4_k_t16_wmma_prefill_smallm_bf16_bf16_out",
    "gguf_q4_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out",
    "gguf_q4_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out",
    "gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out",
    "gguf_q4_k_t16_wmma_prefill_shared_b3w8r3_bf16_bf16_out",
    "gguf_q4_k_t16_wmma_prefill_shared_b2r1_bf16_bf16_out",
    "gguf_q4_k_t16_wmma_prefill_shared_b2w2_bf16_bf16_out",
    "gguf_q4_k_t16_wmma_prefill_shared_b2w4_bf16_bf16_out",
    "gguf_q4_k_t16_selected_wmma_prefill_compact_bf16_bf16_out",
    "gguf_q4_k_t16_selected_wmma_prefill_compact_fp16_fp16_out",
    "gguf_q5_k_qmicro_t16_selected_wmma_prefill_compact_bf16_bf16_out",
    "gguf_q5_k_qmicro_t16_selected_wmma_prefill_compact_fp16_fp16_out",
    "gguf_q5_k_t16_selected_wmma_prefill_compact_bf16_bf16_out",
    "gguf_q5_k_t16_selected_wmma_prefill_compact_fp16_fp16_out",
    "gguf_q5_k_t16_wmma_prefill_bf16_bf16_out",
    "gguf_q5_k_t16_wmma_prefill_shared8r3_bf16_bf16_out",
    "gguf_q5_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out",
    "gguf_q5_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out",
    "gguf_q6_k_t16_selected_expert_major_wmma_comp_bf16_bf16_out",
    "gguf_q6_k_t16_selected_wmma_prefill_compact_bf16_bf16_out",
    "gguf_q6_k_t16_selected_wmma_prefill_compact_fp16_fp16_out",
    "plan_gguf_k_t16_selected_prefill_build",
    "register_gguf_k_t16_selected_prefill_kernels",
]
