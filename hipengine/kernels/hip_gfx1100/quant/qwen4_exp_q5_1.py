"""Raw Q5_1 selected-expert wrapper for the Unsloth Qwen4Exp comparator quant."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("qwen4_exp_q5_1.hip")
_OUTPUT_NAME = "qwen4_exp_q5_1.so"
_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)


def plan_qwen4_exp_q5_1_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="qwen4_exp_q5_1",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_qwen4_exp_q5_1(
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
        family="qwen4_exp_q5_1",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def qwen4_exp_q5_1_selected_gemv_bf16_bf16_out(
    input_ptr: int,
    selected_ptr: int,
    weights_ptr: int,
    output_ptr: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run one raw Q5_1 expert projection for each compact BF16 input row."""

    if rows <= 0 or num_experts <= 0 or in_features <= 0 or out_features <= 0:
        raise ValueError("rows, num_experts, in_features, and out_features must be positive")
    if in_features % 32:
        raise ValueError("Q5_1 in_features must be divisible by 32")
    library = library or build_qwen4_exp_q5_1(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_q5_1_selected_gemv_bf16_bf16_out",
        _ARGS,
        ctypes.c_int,
    )
    error = fn(
        input_ptr,
        selected_ptr,
        weights_ptr,
        output_ptr,
        rows,
        num_experts,
        in_features,
        out_features,
        stream,
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def register_qwen4_exp_q5_1_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q5_1",
            "selected_gemv_bf16_bf16_out",
        ),
        qwen4_exp_q5_1_selected_gemv_bf16_bf16_out,
        replace=replace,
    )


register_qwen4_exp_q5_1_kernels()


__all__ = [
    "build_qwen4_exp_q5_1",
    "plan_qwen4_exp_q5_1_build",
    "qwen4_exp_q5_1_selected_gemv_bf16_bf16_out",
    "register_qwen4_exp_q5_1_kernels",
]
