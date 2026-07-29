"""Transient exact-value Q5_K F32 expansion plus rocBLAS SGEMM prefill."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.core.rocblas import Rocblas, get_rocblas
from hipengine.kernels.hip_gfx1100.convert.cast import (
    bf16_to_f32,
    f32_to_bf16,
)
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q5_k_f32_rocblas_prefill.hip")
_OUTPUT_NAME = "gguf_q5_k_f32_rocblas_prefill.so"
_DEQUANT_SYMBOL = "hipengine_gguf_q5_k_dequantize_f32_exact"
_FUSED_PRODUCER_SYMBOL = (
    "hipengine_gguf_q5_k_dequantize_bf16_to_f32_exact_fused"
)
_DEQUANT_VARIANT = "raw_f32_exact_local64"
_FUSED_PRODUCER_VARIANT = "raw_f32_bf16_input_exact_local64"
_LINEAR_VARIANT = "f32_rocblas_exact_values_bf16_{output_dtype}_out"
_QK_K = 256
_F32_NBYTES = 4
_SESSION_MAX_IN_FEATURES = 9_216
_SESSION_MAX_OUT_FEATURES = 12_288
_SESSION_MAX_WEIGHT_ELEMENTS = 3_072 * 12_288


def plan_gguf_q5_k_f32_rocblas_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q5_k_f32_rocblas_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q5_k_f32_rocblas_prefill(
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
        family="gguf_q5_k_f32_rocblas_prefill",
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


def q5_k_f32_weight_nbytes(in_features: int, out_features: int) -> int:
    hidden = _check_in_features(in_features)
    outputs = _check_out_features(out_features)
    return hidden * outputs * _F32_NBYTES


def q5_k_f32_input_nbytes(rows: int, in_features: int) -> int:
    parsed_rows = _check_rows(rows)
    hidden = _check_in_features(in_features)
    return parsed_rows * hidden * _F32_NBYTES


def q5_k_f32_output_nbytes(rows: int, out_features: int) -> int:
    parsed_rows = _check_rows(rows)
    outputs = _check_out_features(out_features)
    return parsed_rows * outputs * _F32_NBYTES


def q5_k_f32_rocblas_workspace_nbytes(
    rows: int,
    in_features: int,
    out_features: int,
) -> int:
    return (
        q5_k_f32_input_nbytes(rows, in_features)
        + q5_k_f32_weight_nbytes(in_features, out_features)
        + q5_k_f32_output_nbytes(rows, out_features)
    )


def q5_k_f32_rocblas_session_nbytes(max_rows: int) -> int:
    rows = _check_rows(max_rows)
    return _F32_NBYTES * (
        _SESSION_MAX_WEIGHT_ELEMENTS
        + rows * _SESSION_MAX_IN_FEATURES
        + rows * _SESSION_MAX_OUT_FEATURES
    )


def gguf_q5_k_dequantize_f32_exact(
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
    library = library or build_gguf_q5_k_f32_rocblas_prefill(load=True)
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


def gguf_q5_k_dequantize_bf16_to_f32_exact_fused(
    qweight_ptr: int,
    weight_f32_ptr: int,
    x_ptr: int,
    x_f32_ptr: int,
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
    library = library or build_gguf_q5_k_f32_rocblas_prefill(load=True)
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
        ctypes.c_void_p(weight_f32_ptr),
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(x_f32_ptr),
        ctypes.c_int64(parsed_rows),
        ctypes.c_int64(hidden),
        ctypes.c_int64(outputs),
        ctypes.c_void_p(stream),
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def _launch_q5_f32_rocblas(
    output_dtype: str,
    x_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    x_f32_ptr: int,
    weight_f32_ptr: int,
    out_f32_ptr: int,
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
        gguf_q5_k_dequantize_bf16_to_f32_exact_fused(
            qweight_ptr,
            weight_f32_ptr,
            x_ptr,
            x_f32_ptr,
            parsed_rows,
            hidden,
            outputs,
            stream=stream,
            library=dequant_library,
            runtime=runtime,
        )
    else:
        gguf_q5_k_dequantize_f32_exact(
            qweight_ptr,
            weight_f32_ptr,
            hidden,
            outputs,
            stream=stream,
            library=dequant_library,
            runtime=runtime,
        )
        bf16_to_f32(
            x_ptr,
            x_f32_ptr,
            parsed_rows * hidden,
            stream=stream,
            library=cast_library,
            runtime=runtime,
        )

    gemm_out_ptr = out_ptr if output_dtype == "f32" else out_f32_ptr
    (rocblas or get_rocblas()).sgemm_rowmajor_nt(
        x_f32_ptr,
        weight_f32_ptr,
        gemm_out_ptr,
        rows=parsed_rows,
        in_features=hidden,
        out_features=outputs,
        stream=stream,
    )
    if output_dtype == "bf16":
        f32_to_bf16(
            out_f32_ptr,
            out_ptr,
            parsed_rows * outputs,
            stream=stream,
            library=cast_library,
            runtime=runtime,
        )


def gguf_q5_k_f32_rocblas_bf16_bf16_out(*args, **kwargs) -> None:
    _launch_q5_f32_rocblas("bf16", *args, **kwargs)


def gguf_q5_k_f32_rocblas_bf16_f32_out(*args, **kwargs) -> None:
    _launch_q5_f32_rocblas("f32", *args, **kwargs)


def register_gguf_q5_k_f32_rocblas_prefill_kernels(
    *, replace: bool = True
) -> None:
    register(
        KernelKey("hip_gfx1100", "dequant", "gguf_q5_k", _DEQUANT_VARIANT),
        gguf_q5_k_dequantize_f32_exact,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "dequant_cast",
            "gguf_q5_k",
            _FUSED_PRODUCER_VARIANT,
        ),
        gguf_q5_k_dequantize_bf16_to_f32_exact_fused,
        replace=replace,
    )
    for output_dtype, function in (
        ("bf16", gguf_q5_k_f32_rocblas_bf16_bf16_out),
        ("f32", gguf_q5_k_f32_rocblas_bf16_f32_out),
    ):
        register(
            KernelKey(
                "hip_gfx1100",
                "linear",
                "gguf_q5_k",
                _LINEAR_VARIANT.format(output_dtype=output_dtype),
            ),
            function,
            replace=replace,
        )


register_gguf_q5_k_f32_rocblas_prefill_kernels()


__all__ = [
    "build_gguf_q5_k_f32_rocblas_prefill",
    "gguf_q5_k_dequantize_bf16_to_f32_exact_fused",
    "gguf_q5_k_dequantize_f32_exact",
    "gguf_q5_k_f32_rocblas_bf16_bf16_out",
    "gguf_q5_k_f32_rocblas_bf16_f32_out",
    "plan_gguf_q5_k_f32_rocblas_prefill_build",
    "q5_k_f32_input_nbytes",
    "q5_k_f32_output_nbytes",
    "q5_k_f32_rocblas_session_nbytes",
    "q5_k_f32_rocblas_workspace_nbytes",
    "q5_k_f32_weight_nbytes",
    "register_gguf_q5_k_f32_rocblas_prefill_kernels",
]
