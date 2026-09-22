"""Gemma 4 routed-expert FFN primitives and the orchestrated expert forward.

The per-expert GEMV reuses ``dense_gemv_out_bf16`` and the lane compaction reuses
``qwen35_moe_group_compact_active``; this module owns only the GeGLU activation
and the routing-weighted accumulate, plus the glue that runs one GEMV per active
expert over the compacted rows.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_ARGTYPES_GELU = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGTYPES_ZERO = (ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p)
_ARGTYPES_LANE_TO_ROW = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGTYPES_ACCUMULATE = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)

_SOURCE = Path(__file__).with_name("gemma4_moe.hip")
_OUTPUT_NAME = "gemma4_moe.so"

_SYMBOL_GELU = "hipengine_gemma4_gelu_tanh_mul_bf16"
_SYMBOL_ZERO = "hipengine_gemma4_moe_zero_bf16"
_SYMBOL_LANE_TO_ROW = "hipengine_gemma4_moe_lane_to_row_i32"
_SYMBOL_ACCUMULATE = "hipengine_gemma4_moe_weighted_accumulate_bf16"


def plan_gemma4_moe_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gemma4_moe",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gemma4_moe(
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
        family="gemma4_moe",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gemma4_gelu_tanh_mul_bf16(
    gate_up_ptr: int,
    out_ptr: int,
    rows: int,
    intermediate: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """``gelu_tanh(gate) * up`` over a fused ``(rows, 2 * intermediate)`` buffer."""

    if rows <= 0 or intermediate <= 0:
        raise ValueError("rows and intermediate must be positive")
    library = library or build_gemma4_moe(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_GELU, _ARGTYPES_GELU, ctypes.c_int)
    _check_launch(runtime, fn(gate_up_ptr, out_ptr, rows, intermediate, stream))


def gemma4_moe_zero_bf16(
    out_ptr: int,
    total: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if total <= 0:
        raise ValueError("total must be positive")
    library = library or build_gemma4_moe(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_ZERO, _ARGTYPES_ZERO, ctypes.c_int)
    _check_launch(runtime, fn(out_ptr, total, stream))


def gemma4_moe_lane_to_row_i32(
    sorted_lanes_ptr: int,
    lane_to_row_ptr: int,
    total_lanes: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Invert the compact permutation written by the group-compact kernel."""

    if total_lanes <= 0:
        raise ValueError("total_lanes must be positive")
    library = library or build_gemma4_moe(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_LANE_TO_ROW, _ARGTYPES_LANE_TO_ROW, ctypes.c_int)
    _check_launch(runtime, fn(sorted_lanes_ptr, lane_to_row_ptr, total_lanes, stream))


def gemma4_moe_weighted_accumulate_bf16(
    expert_out_ptr: int,
    lane_to_row_ptr: int,
    sorted_weights_ptr: int,
    out_ptr: int,
    tokens: int,
    hidden: int,
    top_k: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Accumulate compacted expert rows into their tokens, scaled by route weight."""

    if tokens <= 0 or hidden <= 0 or top_k <= 0:
        raise ValueError("tokens, hidden, and top_k must be positive")
    library = library or build_gemma4_moe(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_ACCUMULATE, _ARGTYPES_ACCUMULATE, ctypes.c_int)
    _check_launch(
        runtime,
        fn(
            expert_out_ptr,
            lane_to_row_ptr,
            sorted_weights_ptr,
            out_ptr,
            tokens,
            hidden,
            top_k,
            stream,
        ),
    )


def register_gemma4_moe_kernels(*, replace: bool = False) -> None:
    for quant in ("gguf_q4_k_m", "gguf_q4_k_xl", "gguf_q8_0", "gguf"):
        register(
            KernelKey("hip_gfx1100", "expert_geglu", quant, "gemma4_plain"),
            gemma4_gelu_tanh_mul_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "moe_weighted_accumulate", quant, "gemma4_plain"),
            gemma4_moe_weighted_accumulate_bf16,
            replace=replace,
        )


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_gemma4_moe_kernels()
