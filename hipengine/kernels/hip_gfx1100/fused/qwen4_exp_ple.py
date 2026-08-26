"""Raw-pointer wrappers for strict Qwen4Exp PLE HIP primitives."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("qwen4_exp_ple.hip")
_OUTPUT_NAME = "qwen4_exp_ple.so"
_ARGS_3PTR_3DIM = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGS_ADD = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGS_CONV = (
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


def plan_qwen4_exp_ple_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="qwen4_exp_ple",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_qwen4_exp_ple(
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
        family="qwen4_exp_ple",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def qwen4_exp_ple_signed_sqrt_gate_f32(
    key_ptr: int,
    query_ptr: int,
    gate_ptr: int,
    rows: int,
    branches: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Compute the per-branch signed-square-root PLE gate."""

    _check_shape(rows, branches, hidden)
    library = library or build_qwen4_exp_ple(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_ple_signed_sqrt_gate_f32",
        _ARGS_3PTR_3DIM,
        ctypes.c_int,
    )
    _check_launch(runtime, fn(key_ptr, query_ptr, gate_ptr, rows, branches, hidden, stream))


def qwen4_exp_ple_repeat_gated_value_f32(
    value_ptr: int,
    gate_ptr: int,
    output_ptr: int,
    rows: int,
    branches: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Broadcast one value row across branches and apply each branch gate."""

    _check_shape(rows, branches, hidden)
    library = library or build_qwen4_exp_ple(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_ple_repeat_gated_value_f32",
        _ARGS_3PTR_3DIM,
        ctypes.c_int,
    )
    _check_launch(runtime, fn(value_ptr, gate_ptr, output_ptr, rows, branches, hidden, stream))


def qwen4_exp_ple_add_delta_bf16_f32(
    residual_ptr: int,
    gated_value_ptr: int,
    conv_output_ptr: int,
    output_ptr: int,
    elements: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if elements <= 0:
        raise ValueError("elements must be positive")
    library = library or build_qwen4_exp_ple(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_ple_add_delta_bf16_f32",
        _ARGS_ADD,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            residual_ptr,
            gated_value_ptr,
            conv_output_ptr,
            output_ptr,
            elements,
            stream,
        ),
    )


def qwen4_exp_ple_dilated_depthwise_conv_f32(
    input_ptr: int,
    weights_ptr: int,
    history_ptr: int,
    output_ptr: int,
    rows: int,
    channels: int,
    kernel_size: int,
    dilation: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Compute causal PLE Conv and update caller-owned history in place."""

    if rows <= 0 or channels <= 0 or kernel_size <= 0 or dilation <= 0:
        raise ValueError("rows, channels, kernel_size, and dilation must be positive")
    library = library or build_qwen4_exp_ple(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_ple_dilated_depthwise_conv_f32",
        _ARGS_CONV,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            input_ptr,
            weights_ptr,
            history_ptr,
            output_ptr,
            rows,
            channels,
            kernel_size,
            dilation,
            stream,
        ),
    )


def register_qwen4_exp_ple_kernels(*, replace: bool = True) -> None:
    registrations = {
        KernelKey(
            "hip_gfx1100",
            "ple_signed_sqrt_gate",
            "f32",
            "strict",
        ): qwen4_exp_ple_signed_sqrt_gate_f32,
        KernelKey(
            "hip_gfx1100",
            "ple_repeat_gated_value",
            "f32",
            "strict",
        ): qwen4_exp_ple_repeat_gated_value_f32,
        KernelKey(
            "hip_gfx1100",
            "ple_add_delta",
            "bf16_f32",
            "strict",
        ): qwen4_exp_ple_add_delta_bf16_f32,
        KernelKey(
            "hip_gfx1100",
            "ple_dilated_depthwise_conv",
            "f32",
            "strict",
        ): qwen4_exp_ple_dilated_depthwise_conv_f32,
    }
    for key, function in registrations.items():
        register(key, function, replace=replace)


def _check_shape(rows: int, branches: int, hidden: int) -> None:
    if rows <= 0 or branches <= 0 or hidden <= 0:
        raise ValueError("rows, branches, and hidden must be positive")


def _check_launch(runtime: HipRuntime, error: int) -> None:
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


register_qwen4_exp_ple_kernels()


__all__ = [
    "build_qwen4_exp_ple",
    "plan_qwen4_exp_ple_build",
    "qwen4_exp_ple_add_delta_bf16_f32",
    "qwen4_exp_ple_dilated_depthwise_conv_f32",
    "qwen4_exp_ple_repeat_gated_value_f32",
    "qwen4_exp_ple_signed_sqrt_gate_f32",
    "register_qwen4_exp_ple_kernels",
]
