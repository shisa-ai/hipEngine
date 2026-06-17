"""Wrappers for experimental GGUF Q4_K T16 selected WMMA prefill.

P9.C14 prototype: the HIP kernel consumes the P9.C13/P9.H2 Q4T16 tile
layout (``tiles[E, out_tiles16, blocks_per_row, 2368]``) while preserving the
compact selected-MoE scheduler/output ABI used by the raw-Q4_K selected WMMA
prefill path.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q4_k_t16_selected_prefill.hip")
_OUTPUT_NAME = "gguf_q4_k_t16_selected_prefill.so"
_SYMBOL_BF16 = "hipengine_gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out"
_SYMBOL_FP16 = "hipengine_gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_fp16_fp16_out"
_SYMBOL_Q8_1_DS4_WMMA32_BF16 = "hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out"
_ENV_LAUNCH_BOUNDS = "HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS"
_Q4_K_BLOCK = 256


def _extra_flags() -> tuple[str, ...]:
    value = os.environ.get(_ENV_LAUNCH_BOUNDS)
    if not value:
        return ("-mcumode",)
    min_blocks = int(value)
    if min_blocks not in {1, 2, 4, 8}:
        raise ValueError(f"{_ENV_LAUNCH_BOUNDS} must be one of 1, 2, 4, 8")
    return ("-mcumode", f"-DHIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS={min_blocks}")


def plan_gguf_q4_k_t16_selected_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q4_k_t16_selected_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_extra_flags(),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q4_k_t16_selected_prefill(
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
        family="gguf_q4_k_t16_selected_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_extra_flags(),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 selected compact Q4T16 dual gate+up WMMA prefill."""

    _launch(
        _SYMBOL_BF16,
        x_ptr,
        expert_start_compact_ptr,
        expert_start_wmma_ptr,
        tile_expert_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out(
    x_q8_ds4_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch diagnostic DS4 Q8_1 x resident-Q4T16 integer-WMMA32 prefill."""

    _launch(
        _SYMBOL_Q8_1_DS4_WMMA32_BF16,
        x_q8_ds4_ptr,
        expert_start_compact_ptr,
        expert_start_wmma_ptr,
        tile_expert_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_fp16_fp16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 selected compact Q4T16 dual gate+up WMMA prefill."""

    _launch(
        _SYMBOL_FP16,
        x_ptr,
        expert_start_compact_ptr,
        expert_start_wmma_ptr,
        tile_expert_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _launch(
    symbol: str,
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
    )
    library = library or build_gguf_q4_k_t16_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(tiles_a_ptr),
        ctypes.c_void_p(tiles_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _check_common(
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
) -> None:
    _check_positive(compact_rows, "compact_rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_features_a, "out_features_a")
    _check_positive(out_features_b, "out_features_b")
    _check_positive(num_experts, "num_experts")
    _check_positive(wmma_total_rows, "wmma_total_rows")
    if in_features % _Q4_K_BLOCK != 0:
        raise ValueError("in_features must be divisible by GGUF Q4_K block size 256")
    if out_features_a % 16 != 0:
        raise ValueError("out_features_a must be a multiple of 16")
    if out_features_b % 16 != 0:
        raise ValueError("out_features_b must be a multiple of 16")
    if wmma_total_rows % 16 != 0:
        raise ValueError("wmma_total_rows must be a multiple of 16")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def register_gguf_q4_k_t16_selected_prefill_kernels(*, replace: bool = True) -> None:
    """Register P9.C14 compact selected Q4T16 WMMA prototype kernels."""

    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q4_k_t16_v1",
            "selected_dual_wmma_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q4_k_t16_v1",
            "selected_dual_wmma_prefill_compact32_fp16_fp16_out",
        ),
        gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_fp16_fp16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q4_k_t16_v1",
            "selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_t16_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    # Alias the raw-Q4 variant spelling so dispatch can swap quant keys without
    # adding a runtime/backend branch when this prototype graduates.
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q4_k_t16_v1",
            "selected_dual_wmma_prefill_compact_bf16_bf16_out",
        ),
        gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q4_k_t16_v1",
            "selected_dual_wmma_prefill_compact_fp16_fp16_out",
        ),
        gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_fp16_fp16_out,
        replace=replace,
    )


register_gguf_q4_k_t16_selected_prefill_kernels()


__all__ = [
    "build_gguf_q4_k_t16_selected_prefill",
    "gguf_q4_k_t16_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out",
    "gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out",
    "gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_fp16_fp16_out",
    "plan_gguf_q4_k_t16_selected_prefill_build",
    "register_gguf_q4_k_t16_selected_prefill_kernels",
]
