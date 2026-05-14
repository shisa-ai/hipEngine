"""Raw-pointer wrappers for Qwen3.5 linear-attention convolution kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("conv.hip")
_OUTPUT_NAME = "qwen35_linear_attn_conv.so"
_SYMBOL_F32 = "hipengine_qwen35_linear_attn_conv_decode_f32"
_SYMBOL_BF16 = "hipengine_qwen35_linear_attn_conv_decode_bf16"
_SYMBOL_FP16 = "hipengine_qwen35_linear_attn_conv_decode_fp16"
_SYMBOL_PREFILL_F32 = "hipengine_qwen35_linear_attn_conv_prefill_f32"


def plan_qwen35_linear_attn_conv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="qwen35_linear_attn_conv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_qwen35_linear_attn_conv(
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
        family="qwen35_linear_attn_conv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def qwen35_linear_attn_conv_decode_f32(
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP32 linear-attention decode convolution."""

    _launch_conv(
        _SYMBOL_F32,
        hidden_states_ptr,
        conv_state_ptr,
        conv_weight_ptr,
        out_ptr,
        channels,
        kernel_size,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_linear_attn_conv_decode_bf16(
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16-input lowp linear-attention decode convolution."""

    _launch_conv(
        _SYMBOL_BF16,
        hidden_states_ptr,
        conv_state_ptr,
        conv_weight_ptr,
        out_ptr,
        channels,
        kernel_size,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_linear_attn_conv_decode_fp16(
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16-input lowp linear-attention decode convolution."""

    _launch_conv(
        _SYMBOL_FP16,
        hidden_states_ptr,
        conv_state_ptr,
        conv_weight_ptr,
        out_ptr,
        channels,
        kernel_size,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_linear_attn_conv_prefill_f32(
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    tokens: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP32 native prefill convolution and update conv_state."""

    _launch_conv_prefill(
        _SYMBOL_PREFILL_F32,
        hidden_states_ptr,
        conv_state_ptr,
        conv_weight_ptr,
        out_ptr,
        tokens,
        channels,
        kernel_size,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def register_qwen35_linear_attn_conv_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "linear_attn_conv_decode", "w4_paro", "f32"),
        qwen35_linear_attn_conv_decode_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear_attn_conv_decode", "w4_paro", "bf16"),
        qwen35_linear_attn_conv_decode_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear_attn_conv_decode", "w4_paro", "fp16"),
        qwen35_linear_attn_conv_decode_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear_attn_conv_prefill", "w4_paro", "f32"),
        qwen35_linear_attn_conv_prefill_f32,
        replace=replace,
    )


def _launch_conv(
    symbol: str,
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_conv_shape(channels, kernel_size)
    library = library or build_qwen35_linear_attn_conv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
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
        ctypes.c_void_p(hidden_states_ptr),
        ctypes.c_void_p(conv_state_ptr),
        ctypes.c_void_p(conv_weight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(channels),
        ctypes.c_int64(kernel_size),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_conv_prefill(
    symbol: str,
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    tokens: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_conv_shape(channels, kernel_size)
    _check_positive(tokens, "tokens")
    if tokens < kernel_size:
        raise ValueError("native prefill conv currently requires tokens >= kernel_size")
    library = library or build_qwen35_linear_attn_conv(load=True)
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
        ctypes.c_void_p(hidden_states_ptr),
        ctypes.c_void_p(conv_state_ptr),
        ctypes.c_void_p(conv_weight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(channels),
        ctypes.c_int64(kernel_size),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _check_conv_shape(channels: int, kernel_size: int) -> None:
    _check_positive(channels, "channels")
    _check_positive(kernel_size, "kernel_size")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_qwen35_linear_attn_conv_kernels()
