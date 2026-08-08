"""CUDA wrappers for stable expert-major Maple prefill metadata."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_cuda, plan_cuda_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.cuda import CUDA_SUCCESS, CudaRuntime, get_cuda_runtime
from hipengine.kernels.backends import cuda_target_arch_for_backend
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("group_scatter.cu")
_OUTPUT_NAME = "qwen35_moe_group_scatter.so"
_BACKEND = "cuda_sm120a"
_TARGET_ARCH = cuda_target_arch_for_backend(_BACKEND)
_SYMBOL_COMPACT_ACTIVE_I32_PARALLEL = (
    "hipengine_qwen35_moe_group_compact_active_i32_parallel"
)
_PTR = ctypes.c_void_p
_I64 = ctypes.c_int64


def plan_qwen35_moe_group_scatter_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_cuda_build(
        sources=[_SOURCE],
        family="cuda_sm120a_qwen35_moe_group_scatter",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
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
    return build_cuda(
        sources=[_SOURCE],
        family="cuda_sm120a_qwen35_moe_group_scatter",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def qwen35_moe_group_compact_active_i32_parallel(
    selected_experts_ptr: int,
    routing_weights_ptr: int,
    expert_start_ptr: int,
    active_experts_ptr: int,
    active_count_ptr: int,
    sorted_lanes_ptr: int,
    sorted_experts_ptr: int,
    sorted_weights_ptr: int,
    total_lanes: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Build exact stable int32 route metadata in expert-major order."""

    if int(total_lanes) <= 0:
        raise ValueError("total_lanes must be positive")
    if int(num_experts) <= 0 or int(num_experts) > 256:
        raise ValueError("num_experts must be in [1, 256]")
    library = library or build_qwen35_moe_group_scatter(load=True)
    runtime = runtime or get_cuda_runtime()
    fn = signed_kernel_fn(
        library,
        _SYMBOL_COMPACT_ACTIVE_I32_PARALLEL,
        (_PTR,) * 8 + (_I64, _I64, _PTR),
        ctypes.c_int,
    )
    error = fn(
        selected_experts_ptr,
        routing_weights_ptr,
        expert_start_ptr,
        active_experts_ptr,
        active_count_ptr,
        sorted_lanes_ptr,
        sorted_experts_ptr,
        sorted_weights_ptr,
        total_lanes,
        num_experts,
        stream,
    )
    if int(error) != CUDA_SUCCESS:
        runtime.check(int(error))


def register_qwen35_moe_group_scatter_kernels(
    *,
    backend: str = _BACKEND,
    replace: bool = True,
) -> None:
    register(
        KernelKey(
            backend,
            "moe_group_compact",
            "generic",
            "active_experts_i32_parallel",
        ),
        qwen35_moe_group_compact_active_i32_parallel,
        replace=replace,
    )


register_qwen35_moe_group_scatter_kernels()


__all__ = [
    "build_qwen35_moe_group_scatter",
    "plan_qwen35_moe_group_scatter_build",
    "qwen35_moe_group_compact_active_i32_parallel",
    "register_qwen35_moe_group_scatter_kernels",
]
