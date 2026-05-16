"""Raw-pointer wrappers for Qwen3.5 W8A16 linear kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("w8a16_linear.hip")
_OUTPUT_NAME = "w8a16_linear.so"
_SYMBOL_BF16_F32 = "hipengine_w8a16_linear_bf16_f32_out"
_SYMBOL_BF16_LOWP = "hipengine_w8a16_linear_bf16_lowp_out"
_SYMBOL_FP16_LOWP = "hipengine_w8a16_linear_fp16_lowp_out"
_SYMBOL_SHARED_GATE_UP_SILU_FP16 = "hipengine_w8a16_shared_gate_up_silu_fp16"
_SYMBOL_F32_F32 = "hipengine_w8a16_linear_f32_f32_out"
_ALLOWED_THREADS = {64, 128, 256, 512}


def plan_w8a16_linear_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="w8a16_linear",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_w8a16_linear(
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
        family="w8a16_linear",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def w8a16_linear_bf16_f32_out(
    hidden_ptr: int,
    weight_ptr: int,
    weight_scale_ptr: int,
    out_ptr: int,
    tokens: int,
    hidden_size: int,
    out_features: int,
    *,
    threads: int = 64,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16-input, INT8-weight linear with FP32 output."""

    _launch(
        _SYMBOL_BF16_F32,
        hidden_ptr,
        weight_ptr,
        weight_scale_ptr,
        out_ptr,
        tokens,
        hidden_size,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def w8a16_linear_bf16_lowp_out(
    hidden_ptr: int,
    weight_ptr: int,
    weight_scale_ptr: int,
    out_ptr: int,
    tokens: int,
    hidden_size: int,
    out_features: int,
    *,
    threads: int = 64,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16-input, INT8-weight linear with BF16 output."""

    _launch(
        _SYMBOL_BF16_LOWP,
        hidden_ptr,
        weight_ptr,
        weight_scale_ptr,
        out_ptr,
        tokens,
        hidden_size,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def w8a16_linear_fp16_lowp_out(
    hidden_ptr: int,
    weight_ptr: int,
    weight_scale_ptr: int,
    out_ptr: int,
    tokens: int,
    hidden_size: int,
    out_features: int,
    *,
    threads: int = 64,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16-input, INT8-weight linear with FP16 output."""

    _launch(
        _SYMBOL_FP16_LOWP,
        hidden_ptr,
        weight_ptr,
        weight_scale_ptr,
        out_ptr,
        tokens,
        hidden_size,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def w8a16_shared_gate_up_silu_fp16(
    hidden_ptr: int,
    weight_ptr: int,
    weight_scale_ptr: int,
    out_ptr: int,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    threads: int = 64,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch fused FP16 W8A16 shared-expert gate/up + SiLU projection."""

    _launch(
        _SYMBOL_SHARED_GATE_UP_SILU_FP16,
        hidden_ptr,
        weight_ptr,
        weight_scale_ptr,
        out_ptr,
        tokens,
        hidden_size,
        intermediate_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def w8a16_linear_f32_f32_out(
    hidden_ptr: int,
    weight_ptr: int,
    weight_scale_ptr: int,
    out_ptr: int,
    tokens: int,
    hidden_size: int,
    out_features: int,
    *,
    threads: int = 64,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP32-input, INT8-weight linear with FP32 output."""

    _launch(
        _SYMBOL_F32_F32,
        hidden_ptr,
        weight_ptr,
        weight_scale_ptr,
        out_ptr,
        tokens,
        hidden_size,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def register_w8a16_linear_kernels(*, replace: bool = True) -> None:
    for quant in ("w8a16", "w4_paro"):
        register(
            KernelKey("hip_gfx1100", "w8a16_linear", quant, "bf16_f32_out"),
            w8a16_linear_bf16_f32_out,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "w8a16_linear", quant, "bf16_lowp_out"),
            w8a16_linear_bf16_lowp_out,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "w8a16_linear", quant, "fp16_lowp_out"),
            w8a16_linear_fp16_lowp_out,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "w8a16_linear", quant, "shared_gate_up_silu_fp16"),
            w8a16_shared_gate_up_silu_fp16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "w8a16_linear", quant, "f32_f32_out"),
            w8a16_linear_f32_f32_out,
            replace=replace,
        )


def _launch(
    symbol: str,
    hidden_ptr: int,
    weight_ptr: int,
    weight_scale_ptr: int,
    out_ptr: int,
    tokens: int,
    hidden_size: int,
    out_features: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_shape(tokens, hidden_size, out_features, threads)
    library = library or build_w8a16_linear(load=True)
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(hidden_ptr),
        ctypes.c_void_p(weight_ptr),
        ctypes.c_void_p(weight_scale_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(out_features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _check_shape(tokens: int, hidden_size: int, out_features: int, threads: int) -> None:
    _check_positive(tokens, "tokens")
    _check_positive(hidden_size, "hidden_size")
    _check_positive(out_features, "out_features")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, 256, or 512")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_w8a16_linear_kernels()
