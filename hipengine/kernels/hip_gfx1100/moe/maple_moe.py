"""Raw-pointer wrappers for Maple's router and MoE elementwise kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("maple_moe.hip")
_OUTPUT_NAME = "maple_moe.so"
_QUANT = "maple_ternary2"
_PTR = ctypes.c_void_p
_I64 = ctypes.c_int64


def plan_maple_moe_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="maple_moe",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_maple_moe(
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
        family="maple_moe",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def maple_router_topk_bf16(
    x_ptr: int,
    weight_ptr: int,
    selected_experts_ptr: int,
    selected_weights_ptr: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch(
        "hipengine_maple_router_topk_bf16",
        (_PTR, _PTR, _PTR, _PTR, _I64, _I64, _I64, _PTR),
        (
            x_ptr,
            weight_ptr,
            selected_experts_ptr,
            selected_weights_ptr,
            hidden_size,
            num_experts,
            top_k,
        ),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_clamped_swiglu_bf16(
    gate_ptr: int,
    up_ptr: int,
    out_ptr: int,
    rows: int,
    features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch(
        "hipengine_maple_clamped_swiglu_bf16",
        (_PTR, _PTR, _PTR, _I64, _I64, _PTR),
        (gate_ptr, up_ptr, out_ptr, rows, features),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_weighted_residual_bf16(
    residual_ptr: int,
    expert_outputs_ptr: int,
    routing_weights_ptr: int,
    out_ptr: int,
    top_k: int,
    hidden_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch(
        "hipengine_maple_weighted_residual_bf16",
        (_PTR, _PTR, _PTR, _PTR, _I64, _I64, _PTR),
        (
            residual_ptr,
            expert_outputs_ptr,
            routing_weights_ptr,
            out_ptr,
            top_k,
            hidden_size,
        ),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_router_topk_parallel_bf16(
    x_ptr: int,
    weight_ptr: int,
    selected_experts_ptr: int,
    selected_weights_ptr: int,
    logits_scratch_ptr: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Parallel grid-over-experts router (coalesced dot + parallel softmax/top-k)."""

    _launch(
        "hipengine_maple_router_topk_parallel_bf16",
        (_PTR, _PTR, _PTR, _PTR, _PTR, _I64, _I64, _I64, _PTR),
        (
            x_ptr,
            weight_ptr,
            selected_experts_ptr,
            selected_weights_ptr,
            logits_scratch_ptr,
            hidden_size,
            num_experts,
            top_k,
        ),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_router_topk_parallel_batched_bf16(
    x_ptr: int,
    weight_ptr: int,
    selected_experts_ptr: int,
    selected_weights_ptr: int,
    logits_scratch_ptr: int,
    rows: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Batched router over T rows (P3 prefill): logits + parallel softmax/top-k."""

    _launch(
        "hipengine_maple_router_topk_parallel_batched_bf16",
        (_PTR, _PTR, _PTR, _PTR, _PTR, _I64, _I64, _I64, _I64, _PTR),
        (
            x_ptr,
            weight_ptr,
            selected_experts_ptr,
            selected_weights_ptr,
            logits_scratch_ptr,
            rows,
            hidden_size,
            num_experts,
            top_k,
        ),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def register_maple_moe_kernels(
    *,
    backend: str = "hip_gfx1100",
    replace: bool = True,
) -> None:
    kernels = {
        (
            "maple_router_topk",
            "bf16_fp32_softmax_renorm",
        ): maple_router_topk_bf16,
        (
            "maple_router_topk",
            "bf16_fp32_parallel_grid",
        ): maple_router_topk_parallel_bf16,
        (
            "maple_router_topk",
            "bf16_fp32_parallel_grid_batched",
        ): maple_router_topk_parallel_batched_bf16,
        ("maple_clamped_swiglu", "clamp7_bf16"): maple_clamped_swiglu_bf16,
        (
            "maple_weighted_residual",
            "two_bf16_boundaries",
        ): maple_weighted_residual_bf16,
    }
    for (layer, variant), kernel in kernels.items():
        register(
            KernelKey(backend, layer, _QUANT, variant),
            kernel,
            replace=replace,
        )


def _launch(
    symbol: str,
    argtypes: tuple,
    args: tuple,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    library = library or build_maple_moe(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, symbol, argtypes, ctypes.c_int)
    err = fn(*args, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_maple_moe_kernels()


__all__ = [
    "build_maple_moe",
    "maple_clamped_swiglu_bf16",
    "maple_router_topk_bf16",
    "maple_router_topk_parallel_batched_bf16",
    "maple_router_topk_parallel_bf16",
    "maple_weighted_residual_bf16",
    "plan_maple_moe_build",
    "register_maple_moe_kernels",
]
