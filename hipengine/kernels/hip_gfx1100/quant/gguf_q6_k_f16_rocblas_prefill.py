"""Bounded source-shaped Q6_K F16 dequantize/cast/rocBLAS prefill."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.core.rocblas import Rocblas, get_rocblas
from hipengine.kernels.hip_gfx1100.convert.cast import (
    bf16_to_fp16,
    fp16_to_bf16,
    fp16_to_f32,
)
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q6_k_f16_rocblas_prefill.hip")
_OUTPUT_NAME = "gguf_q6_k_f16_rocblas_prefill.so"
_DEQUANT_SYMBOL = "hipengine_gguf_q6_k_dequantize_f16_source"
_FUSED_PRODUCER_SYMBOL = (
    "hipengine_gguf_q6_k_dequantize_bf16_to_f16_source_fused"
)
_DEQUANT_VARIANT = "raw_f16_source_local64"
_FUSED_PRODUCER_VARIANT = "raw_f16_bf16_input_source_local64"
_LINEAR_VARIANT = "f16_rocblas_source_bf16_{output_dtype}_out"
_QK_K = 256
_F16_NBYTES = 2
_SESSION_MAX_IN_FEATURES = 12_288
_SESSION_MAX_OUT_FEATURES = 9_216
_SESSION_MAX_WEIGHT_ELEMENTS = 12_288 * 3_072


def plan_gguf_q6_k_f16_rocblas_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q6_k_f16_rocblas_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q6_k_f16_rocblas_prefill(
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
        family="gguf_q6_k_f16_rocblas_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _check_rows(rows: int) -> int:
    parsed = int(rows)
    if parsed <= 0:
        raise ValueError("rows must be positive")
    return parsed


def _check_in_features(in_features: int) -> int:
    parsed = int(in_features)
    if parsed <= 0 or parsed % _QK_K != 0:
        raise ValueError("in_features must be a positive multiple of 256")
    return parsed


def _check_out_features(out_features: int) -> int:
    parsed = int(out_features)
    if parsed <= 0:
        raise ValueError("out_features must be positive")
    return parsed


def q6_k_f16_weight_nbytes(in_features: int, out_features: int) -> int:
    hidden = _check_in_features(in_features)
    outputs = _check_out_features(out_features)
    return hidden * outputs * _F16_NBYTES


def q6_k_f16_input_nbytes(rows: int, in_features: int) -> int:
    parsed_rows = _check_rows(rows)
    hidden = _check_in_features(in_features)
    return parsed_rows * hidden * _F16_NBYTES


def q6_k_f16_output_nbytes(rows: int, out_features: int) -> int:
    parsed_rows = _check_rows(rows)
    outputs = _check_out_features(out_features)
    return parsed_rows * outputs * _F16_NBYTES


def q6_k_f16_rocblas_workspace_nbytes(
    rows: int,
    in_features: int,
    out_features: int,
) -> int:
    return (
        q6_k_f16_input_nbytes(rows, in_features)
        + q6_k_f16_weight_nbytes(in_features, out_features)
        + q6_k_f16_output_nbytes(rows, out_features)
    )


def q6_k_f16_rocblas_session_nbytes(max_rows: int) -> int:
    rows = _check_rows(max_rows)
    return _F16_NBYTES * (
        _SESSION_MAX_WEIGHT_ELEMENTS
        + rows * _SESSION_MAX_IN_FEATURES
        + rows * _SESSION_MAX_OUT_FEATURES
    )


def gguf_q6_k_dequantize_f16_source(
    qweight_ptr: int,
    out_ptr: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    hidden = _check_in_features(in_features)
    outputs = _check_out_features(out_features)
    library = library or build_gguf_q6_k_f16_rocblas_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    function = getattr(library, _DEQUANT_SYMBOL)
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    error = function(
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(hidden),
        ctypes.c_int64(outputs),
        ctypes.c_void_p(stream),
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def gguf_q6_k_dequantize_bf16_to_f16_source_fused(
    qweight_ptr: int,
    weight_f16_ptr: int,
    x_ptr: int,
    x_f16_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    parsed_rows = _check_rows(rows)
    hidden = _check_in_features(in_features)
    outputs = _check_out_features(out_features)
    library = library or build_gguf_q6_k_f16_rocblas_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    function = getattr(library, _FUSED_PRODUCER_SYMBOL)
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    error = function(
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(weight_f16_ptr),
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(x_f16_ptr),
        ctypes.c_int64(parsed_rows),
        ctypes.c_int64(hidden),
        ctypes.c_int64(outputs),
        ctypes.c_void_p(stream),
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def _launch_q6_f16_rocblas(
    output_dtype: str,
    x_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    x_f16_ptr: int,
    weight_f16_ptr: int,
    out_f16_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    dequant_library: ctypes.CDLL | None = None,
    cast_library: ctypes.CDLL | None = None,
    rocblas: Rocblas | None = None,
    runtime: HipRuntime | None = None,
    fused_producer: bool = True,
) -> None:
    parsed_rows = _check_rows(rows)
    hidden = _check_in_features(in_features)
    outputs = _check_out_features(out_features)
    if output_dtype not in {"bf16", "f32"}:
        raise ValueError("output_dtype must be bf16 or f32")

    runtime = runtime or get_hip_runtime()
    if fused_producer:
        gguf_q6_k_dequantize_bf16_to_f16_source_fused(
            qweight_ptr,
            weight_f16_ptr,
            x_ptr,
            x_f16_ptr,
            parsed_rows,
            hidden,
            outputs,
            stream=stream,
            library=dequant_library,
            runtime=runtime,
        )
    else:
        gguf_q6_k_dequantize_f16_source(
            qweight_ptr,
            weight_f16_ptr,
            hidden,
            outputs,
            stream=stream,
            library=dequant_library,
            runtime=runtime,
        )
        bf16_to_fp16(
            x_ptr,
            x_f16_ptr,
            parsed_rows * hidden,
            stream=stream,
            library=cast_library,
            runtime=runtime,
        )
    (rocblas or get_rocblas()).gemm_ex_rowmajor_nt_fp16_compute_f16(
        x_f16_ptr,
        weight_f16_ptr,
        out_f16_ptr,
        rows=parsed_rows,
        in_features=hidden,
        out_features=outputs,
        stream=stream,
    )
    output_cast = fp16_to_bf16 if output_dtype == "bf16" else fp16_to_f32
    output_cast(
        out_f16_ptr,
        out_ptr,
        parsed_rows * outputs,
        stream=stream,
        library=cast_library,
        runtime=runtime,
    )


def gguf_q6_k_f16_rocblas_bf16_bf16_out(*args, **kwargs) -> None:
    _launch_q6_f16_rocblas("bf16", *args, **kwargs)


def gguf_q6_k_f16_rocblas_bf16_f32_out(*args, **kwargs) -> None:
    _launch_q6_f16_rocblas("f32", *args, **kwargs)


def register_gguf_q6_k_f16_rocblas_prefill_kernels(
    *, replace: bool = True
) -> None:
    register(
        KernelKey("hip_gfx1100", "dequant", "gguf_q6_k", _DEQUANT_VARIANT),
        gguf_q6_k_dequantize_f16_source,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "dequant_cast",
            "gguf_q6_k",
            _FUSED_PRODUCER_VARIANT,
        ),
        gguf_q6_k_dequantize_bf16_to_f16_source_fused,
        replace=replace,
    )
    for output_dtype, function in (
        ("bf16", gguf_q6_k_f16_rocblas_bf16_bf16_out),
        ("f32", gguf_q6_k_f16_rocblas_bf16_f32_out),
    ):
        register(
            KernelKey(
                "hip_gfx1100",
                "linear",
                "gguf_q6_k",
                _LINEAR_VARIANT.format(output_dtype=output_dtype),
            ),
            function,
            replace=replace,
        )


register_gguf_q6_k_f16_rocblas_prefill_kernels()


__all__ = [
    "build_gguf_q6_k_f16_rocblas_prefill",
    "gguf_q6_k_dequantize_bf16_to_f16_source_fused",
    "gguf_q6_k_dequantize_f16_source",
    "gguf_q6_k_f16_rocblas_bf16_bf16_out",
    "gguf_q6_k_f16_rocblas_bf16_f32_out",
    "plan_gguf_q6_k_f16_rocblas_prefill_build",
    "q6_k_f16_input_nbytes",
    "q6_k_f16_output_nbytes",
    "q6_k_f16_rocblas_session_nbytes",
    "q6_k_f16_rocblas_workspace_nbytes",
    "q6_k_f16_weight_nbytes",
    "register_gguf_q6_k_f16_rocblas_prefill_kernels",
]
