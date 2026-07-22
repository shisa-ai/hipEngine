"""Raw-pointer wrappers for selected GGUF IQ2_XS/IQ3_XXS/IQ4_XS MoE GEMVs."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_iq_gemv.hip")
_OUTPUT_NAME = "gguf_iq_gemv.so"
_QK_K = 256
_ALLOWED_THREADS = {64, 128, 256}
_SYMBOL_IQ2_SELECTED = "hipengine_gguf_iq2_xs_selected_gemv_bf16_bf16_out"
_SYMBOL_IQ2_DUAL_SILU = (
    "hipengine_gguf_iq2_xs_selected_dual_silu_gemv_bf16_bf16_out"
)
_SYMBOL_IQ3_SELECTED = "hipengine_gguf_iq3_xxs_selected_gemv_bf16_bf16_out"
_SYMBOL_IQ3_DUAL_SILU = (
    "hipengine_gguf_iq3_xxs_selected_dual_silu_gemv_bf16_bf16_out"
)
_SYMBOL_IQ4_SELECTED = "hipengine_gguf_iq4_xs_selected_gemv_bf16_bf16_out"
_SYMBOL_IQ4_WEIGHTED_DOWN = (
    "hipengine_gguf_iq4_xs_weighted_selected_down_bf16_bf16_out"
)


def plan_gguf_iq_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_iq_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gguf_iq_gemv(
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
        family="gguf_iq_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gguf_iq2_xs_selected_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    threads: int = 64,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_selected(
        _SYMBOL_IQ2_SELECTED,
        x_ptr,
        selected_ptr,
        qweight_ptr,
        out_ptr,
        x_rows=x_rows,
        rows=rows,
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq3_xxs_selected_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_selected(
        _SYMBOL_IQ3_SELECTED,
        x_ptr,
        selected_ptr,
        qweight_ptr,
        out_ptr,
        x_rows=x_rows,
        rows=rows,
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq4_xs_selected_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_selected(
        _SYMBOL_IQ4_SELECTED,
        x_ptr,
        selected_ptr,
        qweight_ptr,
        out_ptr,
        x_rows=x_rows,
        rows=rows,
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq2_xs_selected_dual_silu_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    *,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    threads: int = 64,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_dual_silu(
        _SYMBOL_IQ2_DUAL_SILU,
        x_ptr,
        selected_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        out_ptr,
        x_rows=x_rows,
        rows=rows,
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq3_xxs_selected_dual_silu_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    *,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_dual_silu(
        _SYMBOL_IQ3_DUAL_SILU,
        x_ptr,
        selected_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        out_ptr,
        x_rows=x_rows,
        rows=rows,
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _launch_dual_silu(
    symbol: str,
    x_ptr: int,
    selected_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    *,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _validate_selected(
        x_rows=x_rows,
        rows=rows,
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        threads=threads,
    )
    library = library or build_gguf_iq_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
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
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(gate_weight_ptr),
        ctypes.c_void_p(up_weight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(x_rows),
        ctypes.c_int64(rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def iq_weighted_down_default_threads(*, top_k: int, in_features: int) -> int:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if in_features <= 0 or in_features % _QK_K != 0:
        raise ValueError("in_features must be positive and divisible by 256")
    tasks = top_k * (in_features // 32)
    return 128 if tasks <= 128 else 256


def gguf_iq4_xs_weighted_selected_down_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    routing_weights_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    tokens: int,
    top_k: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    threads: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    launch_threads = (
        iq_weighted_down_default_threads(top_k=top_k, in_features=in_features)
        if threads == 0
        else threads
    )
    _validate_shape(
        in_features=in_features,
        out_features=out_features,
        threads=launch_threads,
    )
    library = library or build_gguf_iq_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_IQ4_WEIGHTED_DOWN)
    fn.argtypes = [
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
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(routing_weights_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(top_k),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(launch_threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_selected(
    symbol: str,
    x_ptr: int,
    selected_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _validate_selected(
        x_rows=x_rows,
        rows=rows,
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        threads=threads,
    )
    library = library or build_gguf_iq_gemv(load=True)
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(x_rows),
        ctypes.c_int64(rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _validate_selected(
    *,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    threads: int,
) -> None:
    if x_rows <= 0:
        raise ValueError("x_rows must be positive")
    if rows <= 0 or rows % x_rows != 0:
        raise ValueError("rows must be positive and divisible by x_rows")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    _validate_shape(
        in_features=in_features,
        out_features=out_features,
        threads=threads,
    )


def _validate_shape(*, in_features: int, out_features: int, threads: int) -> None:
    if in_features <= 0 or in_features % _QK_K != 0:
        raise ValueError("in_features must be positive and divisible by 256")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if threads not in _ALLOWED_THREADS:
        allowed = ", ".join(str(value) for value in sorted(_ALLOWED_THREADS))
        raise ValueError(f"threads must be one of {allowed}")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_gguf_iq_gemv_kernels(*, replace: bool = True) -> None:
    for quant, variant, fn in (
        (
            "gguf_iq2_xs",
            "selected_gemv_decode_bf16_bf16_out",
            gguf_iq2_xs_selected_gemv_bf16_bf16_out,
        ),
        (
            "gguf_iq2_xs",
            "selected_dual_silu_gemv_decode_bf16_bf16_out",
            gguf_iq2_xs_selected_dual_silu_gemv_bf16_bf16_out,
        ),
        (
            "gguf_iq3_xxs",
            "selected_gemv_decode_bf16_bf16_out",
            gguf_iq3_xxs_selected_gemv_bf16_bf16_out,
        ),
        (
            "gguf_iq3_xxs",
            "selected_dual_silu_gemv_decode_bf16_bf16_out",
            gguf_iq3_xxs_selected_dual_silu_gemv_bf16_bf16_out,
        ),
        (
            "gguf_iq4_xs",
            "selected_gemv_decode_bf16_bf16_out",
            gguf_iq4_xs_selected_gemv_bf16_bf16_out,
        ),
        (
            "gguf_iq4_xs",
            "selected_weighted_down_gemv_decode_bf16_bf16_out",
            gguf_iq4_xs_weighted_selected_down_bf16_bf16_out,
        ),
    ):
        register(
            KernelKey("hip_gfx1100", "moe_linear", quant, variant),
            fn,
            replace=replace,
        )


register_gguf_iq_gemv_kernels()


__all__ = [
    "build_gguf_iq_gemv",
    "gguf_iq2_xs_selected_dual_silu_gemv_bf16_bf16_out",
    "gguf_iq2_xs_selected_gemv_bf16_bf16_out",
    "gguf_iq3_xxs_selected_dual_silu_gemv_bf16_bf16_out",
    "gguf_iq3_xxs_selected_gemv_bf16_bf16_out",
    "gguf_iq4_xs_selected_gemv_bf16_bf16_out",
    "gguf_iq4_xs_weighted_selected_down_bf16_bf16_out",
    "iq_weighted_down_default_threads",
    "plan_gguf_iq_gemv_build",
    "register_gguf_iq_gemv_kernels",
]
