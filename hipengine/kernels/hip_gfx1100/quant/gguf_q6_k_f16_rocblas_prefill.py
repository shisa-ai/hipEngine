"""Bounded Q4_K/Q5_K/Q6_K F16 dequantize/cast/rocBLAS prefill."""

from __future__ import annotations

import ctypes
from pathlib import Path
from threading import RLock

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.core.rocblas import Rocblas, get_rocblas
from hipengine.kernels.hip_gfx1100.convert.cast import (
    bf16_to_fp16,
    f32_to_bf16,
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
_Q4_T16_TILE_DEQUANT_SYMBOL = (
    "hipengine_gguf_q4_k_t16_dequantize_f16_tile"
)
_Q4_T16_TILE_DEQUANT_PAIR_SYMBOL = (
    "hipengine_gguf_q4_k_t16_dequantize_f16_tile_pair"
)
_Q5_T16_TILE_DEQUANT_SYMBOL = (
    "hipengine_gguf_q5_k_t16_dequantize_f16_tile"
)
_Q5_T16_TILE_DEQUANT_PAIR_SYMBOL = (
    "hipengine_gguf_q5_k_t16_dequantize_f16_tile_pair"
)
_Q5_T16_TILE_DEQUANT_OCTET_SYMBOL = (
    "hipengine_gguf_q5_k_t16_dequantize_f16_tile_octet"
)
_T16_TILE_DEQUANT_SYMBOL = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_dequantize_f16_tile"
)
_T16_TILE_DEQUANT_DIRECT_SYMBOL = (
    "hipengine_gguf_q6_k_t16_qmicro_planar_dequantize_f16_tile_direct"
)
_DEQUANT_VARIANT = "raw_f16_source_local64"
_FUSED_PRODUCER_VARIANT = "raw_f16_bf16_input_source_local64"
_Q4_T16_TILE_DEQUANT_VARIANT = "t16_f16_tile_local64"
_Q4_T16_TILE_DEQUANT_PAIR_VARIANT = "t16_f16_tile_pair_local64"
_Q5_T16_TILE_DEQUANT_VARIANT = "t16_f16_tile_local64"
_Q5_T16_TILE_DEQUANT_PAIR_VARIANT = "t16_f16_tile_pair_local64"
_Q5_T16_TILE_DEQUANT_OCTET_VARIANT = "t16_f16_tile_octet_local256"
_T16_TILE_DEQUANT_VARIANT = "t16_qmicro_planar_f16_tile_local64"
_T16_TILE_DEQUANT_DIRECT_VARIANT = "t16_qmicro_planar_f16_tile_record256"
_LINEAR_VARIANT = "f16_rocblas_source_bf16_{output_dtype}_out"
_Q4_T16_LINEAR_VARIANT = "f16_rocblas_t16_bf16_bf16_out"
_Q4_T16_PAIR_LINEAR_VARIANT = "f16_rocblas_t16_pair_bf16_bf16_out"
_Q5_T16_LINEAR_VARIANT = "f16_rocblas_t16_bf16_bf16_out"
_Q5_T16_PAIR_LINEAR_VARIANT = "f16_rocblas_t16_pair_bf16_bf16_out"
_Q5_T16_OCTET_LINEAR_VARIANT = "f16_rocblas_t16_octet_bf16_bf16_out"
_T16_LINEAR_VARIANT = "f16_rocblas_t16_qmicro_planar_bf16_{output_dtype}_out"
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


def q6_k_t16_f16_rocblas_workspace_nbytes(
    rows: int,
    in_features: int,
    out_features: int,
    *,
    tile_out_features: int,
) -> int:
    """Return bounded T16->F16 tile workspace (activation + weight tile)."""

    parsed_rows = _check_rows(rows)
    hidden = _check_in_features(in_features)
    outputs = _check_out_features(out_features)
    tile_outputs = _check_out_features(tile_out_features)
    if tile_outputs > outputs or outputs % tile_outputs:
        raise ValueError("tile_out_features must positively divide out_features")
    return _F16_NBYTES * (
        parsed_rows * hidden
        + tile_outputs * hidden
        + parsed_rows * tile_outputs
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


def _launch_t16_dequantize_f16_tile(
    symbol: str,
    tiles_ptr: int,
    out_ptr: int,
    in_features: int,
    out_features: int,
    *,
    col_start: int,
    col_count: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    hidden = _check_in_features(in_features)
    outputs = _check_out_features(out_features)
    start = int(col_start)
    count = _check_out_features(col_count)
    if outputs % 16 or start < 0 or start % 16 or start + count > outputs:
        raise ValueError(
            "out_features/col_start must be tile16 aligned and the tile in bounds"
        )
    library = library or build_gguf_q6_k_f16_rocblas_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    function = getattr(library, symbol)
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    error = function(
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(hidden),
        ctypes.c_int64(outputs),
        ctypes.c_int64(start),
        ctypes.c_int64(count),
        ctypes.c_void_p(stream),
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def gguf_q4_k_t16_dequantize_f16_tile(
    tiles_ptr: int,
    out_ptr: int,
    in_features: int,
    out_features: int,
    *,
    col_start: int,
    col_count: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_t16_dequantize_f16_tile(
        _Q4_T16_TILE_DEQUANT_SYMBOL,
        tiles_ptr,
        out_ptr,
        in_features,
        out_features,
        col_start=col_start,
        col_count=col_count,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_t16_dequantize_f16_tile_pair(
    tiles_ptr: int,
    out_ptr: int,
    in_features: int,
    out_features: int,
    *,
    col_start: int,
    col_count: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Expand adjacent Q4 columns from each shared packed payload byte."""

    _launch_t16_dequantize_f16_tile(
        _Q4_T16_TILE_DEQUANT_PAIR_SYMBOL,
        tiles_ptr,
        out_ptr,
        in_features,
        out_features,
        col_start=col_start,
        col_count=col_count,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q5_k_t16_dequantize_f16_tile(
    tiles_ptr: int,
    out_ptr: int,
    in_features: int,
    out_features: int,
    *,
    col_start: int,
    col_count: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_t16_dequantize_f16_tile(
        _Q5_T16_TILE_DEQUANT_SYMBOL,
        tiles_ptr,
        out_ptr,
        in_features,
        out_features,
        col_start=col_start,
        col_count=col_count,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q5_k_t16_dequantize_f16_tile_pair(
    tiles_ptr: int,
    out_ptr: int,
    in_features: int,
    out_features: int,
    *,
    col_start: int,
    col_count: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Expand each adjacent Q5 column pair from shared packed payload bytes."""

    _launch_t16_dequantize_f16_tile(
        _Q5_T16_TILE_DEQUANT_PAIR_SYMBOL,
        tiles_ptr,
        out_ptr,
        in_features,
        out_features,
        col_start=col_start,
        col_count=col_count,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q5_k_t16_dequantize_f16_tile_octet(
    tiles_ptr: int,
    out_ptr: int,
    in_features: int,
    out_features: int,
    *,
    col_start: int,
    col_count: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Expand each natural eight-column Q5 packed unit once."""

    _launch_t16_dequantize_f16_tile(
        _Q5_T16_TILE_DEQUANT_OCTET_SYMBOL,
        tiles_ptr,
        out_ptr,
        in_features,
        out_features,
        col_start=col_start,
        col_count=col_count,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_qmicro_planar_dequantize_f16_tile(
    tiles_ptr: int,
    out_ptr: int,
    in_features: int,
    out_features: int,
    *,
    col_start: int,
    col_count: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_t16_dequantize_f16_tile(
        _T16_TILE_DEQUANT_SYMBOL,
        tiles_ptr,
        out_ptr,
        in_features,
        out_features,
        col_start=col_start,
        col_count=col_count,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_t16_qmicro_planar_dequantize_f16_tile_direct(
    tiles_ptr: int,
    out_ptr: int,
    in_features: int,
    out_features: int,
    *,
    col_start: int,
    col_count: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Expand complete K256xN16 qmicro records to source-exact F16."""

    _launch_t16_dequantize_f16_tile(
        _T16_TILE_DEQUANT_DIRECT_SYMBOL,
        tiles_ptr,
        out_ptr,
        in_features,
        out_features,
        col_start=col_start,
        col_count=col_count,
        stream=stream,
        library=library,
        runtime=runtime,
    )


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


def _launch_source_f16_k16(
    x_ptr: int, weight_ptr: int, out_ptr: int,
    rows: int, in_features: int, out_features: int,
    dst_row_stride: int, dst_col_offset: int,
    *, stream: int = 0, library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Consume one bounded F16 tile in parent K16 order, publishing BF16."""
    parsed_rows = _check_rows(rows)
    hidden = _check_in_features(in_features)
    cols = _check_out_features(out_features)
    stride, offset = int(dst_row_stride), int(dst_col_offset)
    if cols % 16 or offset < 0 or stride < cols or offset > stride - cols:
        raise ValueError("invalid tile16 output or destination span")
    library = library or build_gguf_q6_k_f16_rocblas_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    function = library.hipengine_source_f16_k16_bf16
    function.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 5 + [ctypes.c_void_p]
    function.restype = ctypes.c_int
    error = function(
        ctypes.c_void_p(x_ptr), ctypes.c_void_p(weight_ptr), ctypes.c_void_p(out_ptr),
        ctypes.c_int64(parsed_rows), ctypes.c_int64(hidden), ctypes.c_int64(cols),
        ctypes.c_int64(stride), ctypes.c_int64(offset), ctypes.c_void_p(stream),
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def _launch_t16_f16_rocblas(
    output_dtype: str,
    dequantize_tile,
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    x_f16_ptr: int,
    weight_tile_f16_ptr: int,
    out_tile_f16_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    tile_out_features: int = 512,
    stream: int = 0,
    dequant_library: ctypes.CDLL | None = None,
    cast_library: ctypes.CDLL | None = None,
    rocblas: Rocblas | None = None,
    solution_index: int | None = None,
    cast_activation: bool = True,
    runtime: HipRuntime | None = None,
) -> None:
    """Run source-F16 arithmetic from a sole-resident T16 weight layout.

    The activation is cast once. Weight columns are dequantized one bounded
    tile at a time and consumed immediately in parent K16 accumulation order;
    no raw or F16 weight sidecar is retained. Q4T16, Q5T16, and planar Q6T16
    share this bounded ABI. The rocBLAS and output-tile arguments remain for
    existing callers; the K16 consumer publishes BF16 directly without using
    that scratch or acquiring a library handle.
    """

    parsed_rows = _check_rows(rows)
    hidden = _check_in_features(in_features)
    outputs = _check_out_features(out_features)
    tile_outputs = _check_out_features(tile_out_features)
    if output_dtype not in {"bf16", "f32"}:
        raise ValueError("output_dtype must be bf16 or f32")
    if outputs % 16 or tile_outputs % 16 or tile_outputs > outputs:
        raise ValueError(
            "out_features and tile_out_features must be tile16 aligned"
        )
    runtime = runtime or get_hip_runtime()
    if cast_activation:
        bf16_to_fp16(
            x_ptr,
            x_f16_ptr,
            parsed_rows * hidden,
            stream=stream,
            library=cast_library,
            runtime=runtime,
        )
    if output_dtype == "f32":
        raise NotImplementedError("T16 F16/rocBLAS F32 output is not yet supported")
    for col_start in range(0, outputs, tile_outputs):
        col_count = min(tile_outputs, outputs - col_start)
        dequantize_tile(
            tiles_ptr,
            weight_tile_f16_ptr,
            hidden,
            outputs,
            col_start=col_start,
            col_count=col_count,
            stream=stream,
            library=dequant_library,
            runtime=runtime,
        )
        _launch_source_f16_k16(
            x_f16_ptr, weight_tile_f16_ptr, out_ptr,
            parsed_rows, hidden, col_count, outputs, col_start,
            stream=stream,
            library=dequant_library,
            runtime=runtime,
        )


def gguf_q4_k_t16_f16_rocblas_bf16_bf16_out(*args, **kwargs) -> None:
    _launch_t16_f16_rocblas(
        "bf16", gguf_q4_k_t16_dequantize_f16_tile, *args, **kwargs
    )


def gguf_q4_k_t16_f16_rocblas_pair_bf16_bf16_out(*args, **kwargs) -> None:
    """Run the bounded Q4 chain with its exact adjacent-pair producer."""

    _launch_t16_f16_rocblas(
        "bf16", gguf_q4_k_t16_dequantize_f16_tile_pair, *args, **kwargs
    )


def gguf_q5_k_t16_f16_rocblas_bf16_bf16_out(*args, **kwargs) -> None:
    _launch_t16_f16_rocblas(
        "bf16", gguf_q5_k_t16_dequantize_f16_tile, *args, **kwargs
    )


def gguf_q5_k_t16_f16_rocblas_pair_bf16_bf16_out(*args, **kwargs) -> None:
    """Run the bounded Q5 chain with its exact adjacent-pair producer."""

    _launch_t16_f16_rocblas(
        "bf16", gguf_q5_k_t16_dequantize_f16_tile_pair, *args, **kwargs
    )


def gguf_q5_k_t16_f16_rocblas_octet_bf16_bf16_out(*args, **kwargs) -> None:
    """Run the bounded Q5 chain with its exact eight-column producer."""

    _launch_t16_f16_rocblas(
        "bf16", gguf_q5_k_t16_dequantize_f16_tile_octet, *args, **kwargs
    )


def gguf_q6_k_t16_qmicro_planar_f16_rocblas_bf16_bf16_out(
    *args, **kwargs
) -> None:
    _launch_t16_f16_rocblas(
        "bf16",
        gguf_q6_k_t16_qmicro_planar_dequantize_f16_tile_direct,
        *args,
        **kwargs,
    )


# Full-matrix F16 library-GEMM route for planar Q6T16 prefill. The single-wave
# Q6 WMMA owner is structurally limited (~11 TF); dequantizing the whole weight
# to F16 once per launch and running a hipBLASLt GEMM is materially faster on
# large projections. The activation arrives pre-cast to IEEE half (F16 staging
# sibling ABI); the weight F16 plane is shared across row counts and the
# per-shape best algorithm is selected once and cached.
class Q6FullGemmWorkspace:
    """Session-owned scratch, isolated by runtime, device and submission stream."""

    def __init__(self) -> None:
        self.weights: dict[tuple, object] = {}
        self.plans: dict[tuple, tuple] = {}
        self.bindings: dict[tuple, tuple[HipRuntime, int, int]] = {}
        self.resources: list[tuple[tuple, object, str]] = []
        self.lock = RLock()
        self.closed = False

    def binding(self, runtime: HipRuntime, stream: int) -> tuple:
        if self.closed:
            raise RuntimeError("Q6 full-GEMM workspace is closed")
        device = runtime.current_device()
        if any(rt is runtime and bound_device != device
               for rt, bound_device, _stream in self.bindings.values()):
            raise RuntimeError("Q6 workspace cannot change its allocation device")
        key = (id(runtime), device, int(stream))
        self.bindings[key] = (runtime, device, int(stream))
        return key

    def release_since(self, start: int) -> None:
        from hipengine.core.memory import free

        errors = []
        while len(self.resources) > start:
            binding, resource, kind = self.resources.pop()
            try:
                if kind == "buffer":
                    free(resource, runtime=self.bindings[binding][0])
                else:
                    resource.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            for error in errors[1:]:
                errors[0].add_note(f"additional Q6 workspace release failure: {error!r}")
            raise errors[0]

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            # Check every binding before touching resources. A failed retirement
            # leaves the owner available for retry rather than freeing live work.
            for runtime, device, stream in self.bindings.values():
                if runtime.current_device() != device:
                    raise RuntimeError("Q6 workspace close requires its allocation device")
                runtime.stream_synchronize(stream)
            self.plans.clear()
            self.weights.clear()
            try:
                self.release_since(0)
            finally:
                self.bindings.clear()
                self.closed = True


def gguf_q6_k_t16_qmicro_planar_fullgemm_fp16_in_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    workspace_owner: Q6FullGemmWorkspace | None = None,
    **kwargs,
) -> None:
    """Dequantize planar Q6T16 to F16 and run a hipBLASLt GEMM (BF16 out)."""
    del kwargs
    runtime = runtime or get_hip_runtime()
    owner = workspace_owner if workspace_owner is not None else Q6FullGemmWorkspace()
    try:
        with owner.lock:
            binding = owner.binding(runtime, stream)
            _launch_q6_fullgemm(
                owner, binding, x_ptr, tiles_ptr, out_ptr, rows, in_features,
                out_features, stream=stream, library=library, runtime=runtime,
            )
    finally:
        if workspace_owner is None:
            owner.close()


def _launch_q6_fullgemm(
    owner, binding, x_ptr, tiles_ptr, out_ptr, rows, in_features, out_features,
    *, stream, library, runtime,
) -> None:
    from hipengine.core.hipblaslt import (
        HipblasLt, HipblasLtProblem, HIP_R_16F, HIP_R_32F,
    )
    from hipengine.core.memory import malloc

    library = library or build_gguf_q6_k_f16_rocblas_prefill(load=True)
    rows = int(rows)
    if rows <= 0:
        raise ValueError("Q6 full-GEMM rows must be positive")
    in_features = _check_in_features(in_features)
    out_features = _check_out_features(out_features)

    resource_start = len(owner.resources)
    wkey = (*binding, in_features, out_features)
    weight_f16 = owner.weights.get(wkey)
    new_weight = weight_f16 is None
    if new_weight:
        weight_f16 = malloc(out_features * in_features * 2, runtime=runtime)
        owner.resources.append((binding, weight_f16, "buffer"))

    def dequant_weight() -> None:
        _launch_t16_dequantize_f16_tile(
            _T16_TILE_DEQUANT_DIRECT_SYMBOL,
            tiles_ptr,
            int(weight_f16.ptr),
            in_features,
            out_features,
            col_start=0,
            col_count=out_features,
            stream=stream,
            library=library,
            runtime=runtime,
        )

    pkey = (*binding, rows, in_features, out_features)
    plan = owner.plans.get(pkey)
    if plan is None:
        workspace_nbytes = 64 * 1024 * 1024
        out_f32 = workspace = lt = prob = None
        try:
            out_f32 = malloc(rows * out_features * 4, runtime=runtime)
            owner.resources.append((binding, out_f32, "buffer"))
            workspace = malloc(workspace_nbytes, runtime=runtime)
            owner.resources.append((binding, workspace, "buffer"))
            lt = HipblasLt()
            owner.resources.append((binding, lt, "handle"))
            # Register before initialization so partially created descriptors
            # remain owned if hipBLASLt rejects a layout or preference.
            prob = HipblasLtProblem.__new__(HipblasLtProblem)
            owner.resources.append((binding, prob, "problem"))
            prob.__init__(
                lt, rows, in_features, out_features, workspace_nbytes,
                ab_type=HIP_R_16F, cd_type=HIP_R_32F,
            )
            dequant_weight()
            runtime.stream_synchronize(stream)
            # A timing winner can change reduction order across fresh owners.
            # Keep the library's heuristic order and select the first executable
            # candidate, rather than making arithmetic depend on wall time.
            best = None
            last_error = None
            for algo in prob.algorithms(16):
                try:
                    prob.launch(algo, x_ptr, int(weight_f16.ptr), int(out_f32.ptr),
                                int(workspace.ptr), stream=stream)
                    runtime.stream_synchronize(stream)
                except Exception as error:
                    last_error = error
                    continue
                best = algo
                break
            if best is None:
                raise RuntimeError("no executable Q6 full-GEMM algorithm") from last_error
        except BaseException as error:
            try:
                runtime.stream_synchronize(stream)
            except BaseException as sync_error:
                # Keep unpublished resources owned until a later successful
                # close can establish that queued work has retired.
                error.add_note(f"Q6 workspace cleanup deferred: {sync_error!r}")
            else:
                try:
                    owner.release_since(resource_start)
                except BaseException as cleanup_error:
                    error.add_note(f"Q6 workspace cleanup failed: {cleanup_error!r}")
            raise
        plan = (prob, best, out_f32, workspace)
        owner.plans[pkey] = plan
    owner.weights[wkey] = weight_f16
    prob, algo, out_f32, workspace = plan
    dequant_weight()
    prob.launch(algo, x_ptr, int(weight_f16.ptr), int(out_f32.ptr),
                int(workspace.ptr), stream=stream)
    f32_to_bf16(int(out_f32.ptr), out_ptr, rows * out_features,
                stream=stream, runtime=runtime)


# Dispatch passes owned scratch only to registered callables declaring it.
gguf_q6_k_t16_qmicro_planar_fullgemm_fp16_in_bf16_out.workspace_owner_type = Q6FullGemmWorkspace


def register_gguf_q6_k_f16_rocblas_prefill_kernels(
    *, replace: bool = True
) -> None:
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_wmma_prefill_fp16_in_bf16_out",
        ),
        gguf_q6_k_t16_qmicro_planar_fullgemm_fp16_in_bf16_out,
        replace=replace,
    )
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
    register(
        KernelKey(
            "hip_gfx1100",
            "dequant",
            "gguf_q4_k_t16_v1",
            _Q4_T16_TILE_DEQUANT_VARIANT,
        ),
        gguf_q4_k_t16_dequantize_f16_tile,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "dequant",
            "gguf_q4_k_t16_v1",
            _Q4_T16_TILE_DEQUANT_PAIR_VARIANT,
        ),
        gguf_q4_k_t16_dequantize_f16_tile_pair,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "dequant",
            "gguf_q5_k_t16_v1",
            _Q5_T16_TILE_DEQUANT_VARIANT,
        ),
        gguf_q5_k_t16_dequantize_f16_tile,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "dequant",
            "gguf_q5_k_t16_v1",
            _Q5_T16_TILE_DEQUANT_PAIR_VARIANT,
        ),
        gguf_q5_k_t16_dequantize_f16_tile_pair,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "dequant",
            "gguf_q5_k_t16_v1",
            _Q5_T16_TILE_DEQUANT_OCTET_VARIANT,
        ),
        gguf_q5_k_t16_dequantize_f16_tile_octet,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "dequant",
            "gguf_q6_k_t16_qmicro_planar_v1",
            _T16_TILE_DEQUANT_VARIANT,
        ),
        gguf_q6_k_t16_qmicro_planar_dequantize_f16_tile,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "dequant",
            "gguf_q6_k_t16_qmicro_planar_v1",
            _T16_TILE_DEQUANT_DIRECT_VARIANT,
        ),
        gguf_q6_k_t16_qmicro_planar_dequantize_f16_tile_direct,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q4_k_t16_v1",
            _Q4_T16_LINEAR_VARIANT,
        ),
        gguf_q4_k_t16_f16_rocblas_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q4_k_t16_v1",
            _Q4_T16_PAIR_LINEAR_VARIANT,
        ),
        gguf_q4_k_t16_f16_rocblas_pair_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q5_k_t16_v1",
            _Q5_T16_LINEAR_VARIANT,
        ),
        gguf_q5_k_t16_f16_rocblas_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q5_k_t16_v1",
            _Q5_T16_PAIR_LINEAR_VARIANT,
        ),
        gguf_q5_k_t16_f16_rocblas_pair_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q5_k_t16_v1",
            _Q5_T16_OCTET_LINEAR_VARIANT,
        ),
        gguf_q5_k_t16_f16_rocblas_octet_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            _T16_LINEAR_VARIANT.format(output_dtype="bf16"),
        ),
        gguf_q6_k_t16_qmicro_planar_f16_rocblas_bf16_bf16_out,
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
    "gguf_q4_k_t16_dequantize_f16_tile",
    "gguf_q4_k_t16_dequantize_f16_tile_pair",
    "gguf_q4_k_t16_f16_rocblas_bf16_bf16_out",
    "gguf_q4_k_t16_f16_rocblas_pair_bf16_bf16_out",
    "gguf_q5_k_t16_dequantize_f16_tile",
    "gguf_q5_k_t16_dequantize_f16_tile_octet",
    "gguf_q5_k_t16_dequantize_f16_tile_pair",
    "gguf_q5_k_t16_f16_rocblas_bf16_bf16_out",
    "gguf_q5_k_t16_f16_rocblas_octet_bf16_bf16_out",
    "gguf_q5_k_t16_f16_rocblas_pair_bf16_bf16_out",
    "gguf_q6_k_dequantize_bf16_to_f16_source_fused",
    "gguf_q6_k_dequantize_f16_source",
    "gguf_q6_k_t16_qmicro_planar_dequantize_f16_tile",
    "gguf_q6_k_t16_qmicro_planar_dequantize_f16_tile_direct",
    "gguf_q6_k_t16_qmicro_planar_f16_rocblas_bf16_bf16_out",
    "gguf_q6_k_f16_rocblas_bf16_bf16_out",
    "gguf_q6_k_f16_rocblas_bf16_f32_out",
    "plan_gguf_q6_k_f16_rocblas_prefill_build",
    "q6_k_f16_input_nbytes",
    "q6_k_f16_output_nbytes",
    "q6_k_f16_rocblas_session_nbytes",
    "q6_k_f16_rocblas_workspace_nbytes",
    "q6_k_t16_f16_rocblas_workspace_nbytes",
    "q6_k_f16_weight_nbytes",
    "register_gguf_q6_k_f16_rocblas_prefill_kernels",
]
