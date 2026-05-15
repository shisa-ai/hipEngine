"""Raw-pointer wrappers for Qwen3.5 grouped-MoE prefill metadata kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("group_scatter.hip")
_OUTPUT_NAME = "qwen35_moe_group_scatter.so"
_SYMBOL_COUNT = "hipengine_qwen35_moe_group_count"
_SYMBOL_PREFIX = "hipengine_qwen35_moe_group_prefix"
_SYMBOL_TILE_MAP = "hipengine_qwen35_moe_wmma_tile_map"
_SYMBOL_SCATTER = "hipengine_qwen35_moe_group_scatter"
_SYMBOL_SCATTER_GATHER = "hipengine_qwen35_moe_group_scatter_gather_lowp"
_SYMBOL_GATHER = "hipengine_qwen35_moe_gather_packed_hidden_lowp"


def plan_qwen35_moe_group_scatter_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="qwen35_moe_group_scatter",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_qwen35_moe_group_scatter(
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
        family="qwen35_moe_group_scatter",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def qwen35_moe_group_count(
    selected_experts_ptr: int,
    counts_ptr: int,
    total_lanes: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Count routed lanes per expert. Caller owns zeroing ``counts``."""

    _check_positive(total_lanes, "total_lanes")
    _check_positive(num_experts, "num_experts")
    library = library or build_qwen35_moe_group_scatter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_COUNT)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(selected_experts_ptr),
        ctypes.c_void_p(counts_ptr),
        ctypes.c_int64(total_lanes),
        ctypes.c_int64(num_experts),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_moe_group_prefix(
    counts_ptr: int,
    padded_counts_ptr: int,
    expert_start_ptr: int,
    total_padded_ptr: int,
    num_experts: int,
    pad_multiple: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Prefix-scan per-expert lane counts into compact/padded group starts."""

    _check_num_experts(num_experts)
    _check_positive(pad_multiple, "pad_multiple")
    library = library or build_qwen35_moe_group_scatter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFIX)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(counts_ptr),
        ctypes.c_void_p(padded_counts_ptr),
        ctypes.c_void_p(expert_start_ptr),
        ctypes.c_void_p(total_padded_ptr),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(pad_multiple),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_moe_wmma_tile_map(
    expert_start_compact_ptr: int,
    wmma_expert_start_ptr: int,
    tile_expert_ptr: int,
    wmma_total_ptr: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Map compact expert starts to 16-row WMMA tile starts and tile expert ids."""

    _check_num_experts(num_experts)
    library = library or build_qwen35_moe_group_scatter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_TILE_MAP)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(wmma_expert_start_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(wmma_total_ptr),
        ctypes.c_int64(num_experts),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_moe_group_scatter(
    selected_experts_ptr: int,
    routing_weights_ptr: int,
    expert_start_ptr: int,
    scatter_offsets_ptr: int,
    sorted_lanes_ptr: int,
    sorted_experts_ptr: int,
    sorted_weights_ptr: int,
    total_lanes: int,
    num_experts: int,
    *,
    include_routing_weights: bool = True,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Scatter routed lanes into per-expert compact order. Caller zeroes offsets."""

    _check_positive(total_lanes, "total_lanes")
    _check_positive(num_experts, "num_experts")
    library = library or build_qwen35_moe_group_scatter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SCATTER)
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
        ctypes.c_bool,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(selected_experts_ptr),
        ctypes.c_void_p(routing_weights_ptr),
        ctypes.c_void_p(expert_start_ptr),
        ctypes.c_void_p(scatter_offsets_ptr),
        ctypes.c_void_p(sorted_lanes_ptr),
        ctypes.c_void_p(sorted_experts_ptr),
        ctypes.c_void_p(sorted_weights_ptr),
        ctypes.c_int64(total_lanes),
        ctypes.c_int64(num_experts),
        ctypes.c_bool(bool(include_routing_weights)),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_moe_group_scatter_gather_lowp(
    hidden_states_ptr: int,
    selected_experts_ptr: int,
    routing_weights_ptr: int,
    expert_start_ptr: int,
    scatter_offsets_ptr: int,
    sorted_lanes_ptr: int,
    sorted_experts_ptr: int,
    sorted_weights_ptr: int,
    packed_hidden_ptr: int,
    total_lanes: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Scatter lanes and gather lowp hidden rows into packed expert order."""

    _check_positive(total_lanes, "total_lanes")
    _check_positive(num_experts, "num_experts")
    _check_positive(top_k, "top_k")
    _check_positive(hidden_size, "hidden_size")
    library = library or build_qwen35_moe_group_scatter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SCATTER_GATHER)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(hidden_states_ptr),
        ctypes.c_void_p(selected_experts_ptr),
        ctypes.c_void_p(routing_weights_ptr),
        ctypes.c_void_p(expert_start_ptr),
        ctypes.c_void_p(scatter_offsets_ptr),
        ctypes.c_void_p(sorted_lanes_ptr),
        ctypes.c_void_p(sorted_experts_ptr),
        ctypes.c_void_p(sorted_weights_ptr),
        ctypes.c_void_p(packed_hidden_ptr),
        ctypes.c_int64(total_lanes),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(top_k),
        ctypes.c_int64(hidden_size),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_moe_gather_packed_hidden_lowp(
    hidden_states_ptr: int,
    sorted_lanes_ptr: int,
    packed_hidden_ptr: int,
    total_elements: int,
    tokens: int,
    top_k: int,
    hidden_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Gather hidden rows by sorted routed lane ids into packed hidden rows."""

    _check_positive(total_elements, "total_elements")
    _check_positive(tokens, "tokens")
    _check_positive(top_k, "top_k")
    _check_positive(hidden_size, "hidden_size")
    library = library or build_qwen35_moe_group_scatter(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GATHER)
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
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(hidden_states_ptr),
        ctypes.c_void_p(sorted_lanes_ptr),
        ctypes.c_void_p(packed_hidden_ptr),
        ctypes.c_int64(total_elements),
        ctypes.c_int64(tokens),
        ctypes.c_int64(top_k),
        ctypes.c_int64(hidden_size),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def register_qwen35_moe_group_scatter_kernels(*, replace: bool = True) -> None:
    register(KernelKey("hip_gfx1100", "moe_group_count", "w4_paro", "qwen35"), qwen35_moe_group_count, replace=replace)
    register(KernelKey("hip_gfx1100", "moe_group_prefix", "w4_paro", "qwen35"), qwen35_moe_group_prefix, replace=replace)
    register(KernelKey("hip_gfx1100", "moe_wmma_tile_map", "w4_paro", "qwen35"), qwen35_moe_wmma_tile_map, replace=replace)
    register(KernelKey("hip_gfx1100", "moe_group_scatter", "w4_paro", "qwen35"), qwen35_moe_group_scatter, replace=replace)
    register(
        KernelKey("hip_gfx1100", "moe_group_scatter_gather", "w4_paro", "qwen35_lowp"),
        qwen35_moe_group_scatter_gather_lowp,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "moe_gather_packed_hidden", "w4_paro", "qwen35_lowp"),
        qwen35_moe_gather_packed_hidden_lowp,
        replace=replace,
    )


def _check_num_experts(num_experts: int) -> None:
    _check_positive(num_experts, "num_experts")
    if num_experts > 1024:
        raise ValueError("num_experts must be <= 1024")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_qwen35_moe_group_scatter_kernels()
