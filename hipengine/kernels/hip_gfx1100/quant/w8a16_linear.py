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
_SYMBOL_BF16_F32_MULTI_ROW = "hipengine_w8a16_linear_bf16_f32_multi_row"
_SYMBOL_BF16_LOWP = "hipengine_w8a16_linear_bf16_lowp_out"
_SYMBOL_FP16_LOWP = "hipengine_w8a16_linear_fp16_lowp_out"
_SYMBOL_SHARED_GATE_UP_SILU_FP16 = "hipengine_w8a16_shared_gate_up_silu_fp16"
_SYMBOL_SHARED_GATE_UP_SILU_FP16_TOKEN_TILE2 = "hipengine_w8a16_shared_gate_up_silu_fp16_token_tile2"
_SYMBOL_SHARED_GATE_UP_SILU_FP16_TOKEN_TILE4 = "hipengine_w8a16_shared_gate_up_silu_fp16_token_tile4"
_SYMBOL_SHARED_GATE_SIGMOID_FP32 = "hipengine_w8a16_shared_gate_sigmoid_fp32"
_SYMBOL_SHARED_DOWN_COMBINE_RESIDUAL_FP16 = "hipengine_w8a16_shared_down_combine_residual_fp16"
_SYMBOL_SHARED_DOWN_COMBINE_RESIDUAL_FP16_TOKEN_TILE2 = "hipengine_w8a16_shared_down_combine_residual_fp16_token_tile2"
_SYMBOL_SHARED_DOWN_COMBINE_RESIDUAL_FP16_TOKEN_TILE4 = "hipengine_w8a16_shared_down_combine_residual_fp16_token_tile4"
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


def w8a16_linear_bf16_f32_multi_row(
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
    """M12.2: weight-sharing W8A16 GEMV.  Same output as ``bf16_f32_out`` but
    each block processes one vocab row across all ``tokens`` so the weight
    matrix streams from HBM once instead of ``tokens`` times.
    """

    _launch(
        _SYMBOL_BF16_F32_MULTI_ROW,
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


def w8a16_shared_gate_up_silu_fp16_token_tiled(
    hidden_ptr: int,
    weight_ptr: int,
    weight_scale_ptr: int,
    out_ptr: int,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    token_tile: int,
    threads: int = 64,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch token-tiled FP16 W8A16 shared-expert gate/up + SiLU projection."""

    symbols = {
        2: _SYMBOL_SHARED_GATE_UP_SILU_FP16_TOKEN_TILE2,
        4: _SYMBOL_SHARED_GATE_UP_SILU_FP16_TOKEN_TILE4,
    }
    try:
        symbol = symbols[int(token_tile)]
    except KeyError as exc:
        raise ValueError("token_tile must be 2 or 4") from exc
    _launch(
        symbol,
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


def w8a16_shared_gate_sigmoid_fp32(
    gate_logits_ptr: int,
    gate_values_ptr: int,
    tokens: int,
    gate_stride: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Precompute the shared-expert sigmoid gate once per token."""

    _launch_shared_gate_sigmoid(
        _SYMBOL_SHARED_GATE_SIGMOID_FP32,
        gate_logits_ptr,
        gate_values_ptr,
        tokens,
        gate_stride,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def w8a16_shared_down_combine_residual_fp16(
    shared_intermediate_ptr: int,
    down_weight_ptr: int,
    down_scale_ptr: int,
    selected_ptr: int,
    shared_gate_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    gate_stride: int,
    *,
    threads: int = 64,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch fused FP16 W8A16 shared down + precomputed shared-gate/residual combine."""

    _launch_shared_down_combine_residual(
        _SYMBOL_SHARED_DOWN_COMBINE_RESIDUAL_FP16,
        shared_intermediate_ptr,
        down_weight_ptr,
        down_scale_ptr,
        selected_ptr,
        shared_gate_ptr,
        residual_ptr,
        out_ptr,
        tokens,
        hidden_size,
        intermediate_size,
        gate_stride,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def w8a16_shared_down_combine_residual_fp16_token_tiled(
    shared_intermediate_ptr: int,
    down_weight_ptr: int,
    down_scale_ptr: int,
    selected_ptr: int,
    shared_gate_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    gate_stride: int,
    *,
    token_tile: int,
    threads: int = 64,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch token-tiled FP16 W8A16 shared down + shared-gate/residual combine."""

    symbols = {
        2: _SYMBOL_SHARED_DOWN_COMBINE_RESIDUAL_FP16_TOKEN_TILE2,
        4: _SYMBOL_SHARED_DOWN_COMBINE_RESIDUAL_FP16_TOKEN_TILE4,
    }
    try:
        symbol = symbols[int(token_tile)]
    except KeyError as exc:
        raise ValueError("token_tile must be 2 or 4") from exc
    _launch_shared_down_combine_residual(
        symbol,
        shared_intermediate_ptr,
        down_weight_ptr,
        down_scale_ptr,
        selected_ptr,
        shared_gate_ptr,
        residual_ptr,
        out_ptr,
        tokens,
        hidden_size,
        intermediate_size,
        gate_stride,
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
            KernelKey("hip_gfx1100", "w8a16_linear", quant, "bf16_f32_multi_row"),
            w8a16_linear_bf16_f32_multi_row,
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
            KernelKey("hip_gfx1100", "w8a16_linear", quant, "shared_gate_up_silu_fp16_token_tiled"),
            w8a16_shared_gate_up_silu_fp16_token_tiled,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "w8a16_linear", quant, "shared_gate_sigmoid_fp32"),
            w8a16_shared_gate_sigmoid_fp32,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "w8a16_linear", quant, "shared_down_combine_residual_fp16"),
            w8a16_shared_down_combine_residual_fp16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "w8a16_linear", quant, "shared_down_combine_residual_fp16_token_tiled"),
            w8a16_shared_down_combine_residual_fp16_token_tiled,
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


def _launch_shared_gate_sigmoid(
    symbol: str,
    gate_logits_ptr: int,
    gate_values_ptr: int,
    tokens: int,
    gate_stride: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_positive(tokens, "tokens")
    _check_positive(gate_stride, "gate_stride")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, 256, or 512")
    library = library or build_w8a16_linear(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(gate_logits_ptr),
        ctypes.c_void_p(gate_values_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(gate_stride),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_shared_down_combine_residual(
    symbol: str,
    shared_intermediate_ptr: int,
    down_weight_ptr: int,
    down_scale_ptr: int,
    selected_ptr: int,
    shared_gate_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    gate_stride: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_shape(tokens, hidden_size, intermediate_size, threads)
    _check_positive(gate_stride, "gate_stride")
    library = library or build_w8a16_linear(load=True)
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(shared_intermediate_ptr),
        ctypes.c_void_p(down_weight_ptr),
        ctypes.c_void_p(down_scale_ptr),
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(shared_gate_ptr),
        ctypes.c_void_p(residual_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(intermediate_size),
        ctypes.c_int64(gate_stride),
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
