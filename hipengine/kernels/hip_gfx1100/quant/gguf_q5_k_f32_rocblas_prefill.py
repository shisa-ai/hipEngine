"""Transient exact-value Q5_K/Q6_K F32 expansion and ordered prefill."""

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
_Q6_DEQUANT_SYMBOL = "hipengine_gguf_q6_k_dequantize_f32_exact"
_FUSED_PRODUCER_SYMBOL = (
    "hipengine_gguf_q5_k_dequantize_bf16_to_f32_exact_fused"
)
_DEQUANT_VARIANT = "raw_f32_exact_local64"
_FUSED_PRODUCER_VARIANT = "raw_f32_bf16_input_exact_local64"
_LINEAR_VARIANT = "f32_rocblas_exact_values_bf16_{output_dtype}_out"
_ORDERED_GEOMETRIES = (
    (4, 8),
    (8, 4),
    (4, 16),
    (8, 8),
    (16, 4),
    (12, 4),
    (8, 10),
    (16, 5),
    (8, 12),
    (12, 8),
)
_Q6_ORDERED_GEOMETRIES = frozenset(((8, 4), (16, 4), (16, 5)))
_Q6_ORDERED_WEIGHT_MAJOR_GEOMETRIES = (
    (16, 5, "bf16"),
    (16, 4, "bf16"),
    (16, 5, "f32"),
)
_ORDERED_WEIGHT_MAJOR_GEOMETRIES = (
    (8, 4, "bf16"),
    (8, 12, "bf16"),
    (16, 5, "bf16"),
    (12, 8, "bf16"),
    (16, 4, "bf16"),
    (16, 5, "f32"),
    (8, 10, "f32"),
)
_ORDERED_SYMBOL = (
    "hipengine_gguf_q5_k_f32_weight_ordered_coltile{col_tile}_"
    "rowbatch{row_batch}_bf16_{output_dtype}_out"
)
_ORDERED_PRIMITIVE_VARIANT = (
    "ordered_coltile{col_tile}_rowbatch{row_batch}_bf16_{output_dtype}_out"
)
_ORDERED_COMPOSITE_VARIANT = "f32_{variant}"
_ORDERED_WEIGHT_MAJOR_SYMBOL = (
    "hipengine_gguf_q5_k_f32_weight_ordered_weight_major_"
    "coltile{col_tile}_rowbatch{row_batch}_bf16_{output_dtype}_out"
)
_ORDERED_WEIGHT_MAJOR_PRIMITIVE_VARIANT = (
    "ordered_weight_major_coltile{col_tile}_rowbatch{row_batch}_"
    "bf16_{output_dtype}_out"
)
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


def q5_k_f32_ordered_workspace_nbytes(
    in_features: int,
    out_features: int,
) -> int:
    return q5_k_f32_weight_nbytes(in_features, out_features)


def q6_k_f32_ordered_workspace_nbytes(
    in_features: int,
    out_features: int,
) -> int:
    return q5_k_f32_weight_nbytes(in_features, out_features)


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


def gguf_q6_k_dequantize_f32_exact(
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
    function = getattr(library, _Q6_DEQUANT_SYMBOL)
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


def _launch_q5_f32_weight_ordered(
    *,
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    x_ptr: int,
    weight_f32_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    parsed_rows = _check_rows(rows)
    hidden = _check_in_features(in_features)
    outputs = _check_out_features(out_features)
    if (
        col_tile * row_batch not in {32, 48, 64, 80, 96}
        or (col_tile, row_batch) not in _ORDERED_GEOMETRIES
    ):
        raise ValueError(
            "ordered Q5 geometry must keep 32, 48, 64, 80, or 96 accumulators"
        )
    if outputs % col_tile != 0:
        raise ValueError(f"out_features must be divisible by {col_tile}")
    if output_dtype not in {"bf16", "f32"}:
        raise ValueError("output_dtype must be bf16 or f32")

    library = library or build_gguf_q5_k_f32_rocblas_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    symbol = _ORDERED_SYMBOL.format(
        col_tile=col_tile,
        row_batch=row_batch,
        output_dtype=output_dtype,
    )
    function = getattr(library, symbol)
    function.argtypes = [
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
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(weight_f32_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(parsed_rows),
        ctypes.c_int64(hidden),
        ctypes.c_int64(outputs),
        ctypes.c_void_p(stream),
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def _make_q5_f32_weight_ordered(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
):
    def launch(
        x_ptr: int,
        weight_f32_ptr: int,
        out_ptr: int,
        rows: int,
        in_features: int,
        out_features: int,
        **kwargs,
    ) -> None:
        _launch_q5_f32_weight_ordered(
            col_tile=col_tile,
            row_batch=row_batch,
            output_dtype=output_dtype,
            x_ptr=x_ptr,
            weight_f32_ptr=weight_f32_ptr,
            out_ptr=out_ptr,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            **kwargs,
        )

    return launch


def _launch_q5_f32_weight_ordered_weight_major(
    *,
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    x_ptr: int,
    weight_f32_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    parsed_rows = _check_rows(rows)
    hidden = _check_in_features(in_features)
    outputs = _check_out_features(out_features)
    if (
        col_tile * row_batch not in {32, 48, 64, 80, 96}
        or (col_tile, row_batch) not in _ORDERED_GEOMETRIES
    ):
        raise ValueError(
            "weight-major ordered Q5 geometry must keep a retained accumulator tile"
        )
    if outputs % col_tile != 0:
        raise ValueError(f"out_features must be divisible by {col_tile}")
    if output_dtype not in {"bf16", "f32"}:
        raise ValueError("output_dtype must be bf16 or f32")

    library = library or build_gguf_q5_k_f32_rocblas_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    symbol = _ORDERED_WEIGHT_MAJOR_SYMBOL.format(
        col_tile=col_tile,
        row_batch=row_batch,
        output_dtype=output_dtype,
    )
    function = getattr(library, symbol)
    function.argtypes = [
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
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(weight_f32_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(parsed_rows),
        ctypes.c_int64(hidden),
        ctypes.c_int64(outputs),
        ctypes.c_void_p(stream),
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def _make_q5_f32_weight_ordered_weight_major(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
):
    def launch(
        x_ptr: int,
        weight_f32_ptr: int,
        out_ptr: int,
        rows: int,
        in_features: int,
        out_features: int,
        **kwargs,
    ) -> None:
        _launch_q5_f32_weight_ordered_weight_major(
            col_tile=col_tile,
            row_batch=row_batch,
            output_dtype=output_dtype,
            x_ptr=x_ptr,
            weight_f32_ptr=weight_f32_ptr,
            out_ptr=out_ptr,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            **kwargs,
        )

    return launch


def _make_q_f32_ordered_composite(
    primitive,
    dequantize,
    *,
    col_tile: int,
):
    def launch(
        x_ptr: int,
        qweight_ptr: int,
        out_ptr: int,
        weight_f32_ptr: int,
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
        if outputs % col_tile != 0:
            raise ValueError(f"out_features must be divisible by {col_tile}")
        runtime = runtime or get_hip_runtime()
        dequantize(
            qweight_ptr,
            weight_f32_ptr,
            hidden,
            outputs,
            stream=stream,
            library=library,
            runtime=runtime,
        )
        primitive(
            x_ptr,
            weight_f32_ptr,
            out_ptr,
            parsed_rows,
            hidden,
            outputs,
            stream=stream,
            library=library,
            runtime=runtime,
        )

    return launch


_ORDERED_PRIMITIVES = {}
_ORDERED_COMPOSITES = {}
_Q6_ORDERED_COMPOSITES = {}
_ORDERED_WEIGHT_MAJOR_PRIMITIVES = {}
_ORDERED_WEIGHT_MAJOR_COMPOSITES = {}
_Q6_ORDERED_WEIGHT_MAJOR_COMPOSITES = {}
_ORDERED_EXPORT_NAMES = []
for _col_tile, _row_batch in _ORDERED_GEOMETRIES:
    for _output_dtype in ("bf16", "f32"):
        _variant = _ORDERED_PRIMITIVE_VARIANT.format(
            col_tile=_col_tile,
            row_batch=_row_batch,
            output_dtype=_output_dtype,
        )
        _primitive_name = f"gguf_q5_k_f32_weight_{_variant}"
        _composite_name = (
            f"gguf_q5_k_{_ORDERED_COMPOSITE_VARIANT.format(variant=_variant)}"
        )
        _primitive = _make_q5_f32_weight_ordered(
            _col_tile,
            _row_batch,
            _output_dtype,
        )
        _primitive.__name__ = _primitive_name
        _composite = _make_q_f32_ordered_composite(
            _primitive,
            gguf_q5_k_dequantize_f32_exact,
            col_tile=_col_tile,
        )
        _composite.__name__ = _composite_name
        globals()[_primitive_name] = _primitive
        globals()[_composite_name] = _composite
        _key = (_col_tile, _row_batch, _output_dtype)
        _ORDERED_PRIMITIVES[_key] = _primitive
        _ORDERED_COMPOSITES[_key] = _composite
        _ORDERED_EXPORT_NAMES.extend((_primitive_name, _composite_name))
        if (_col_tile, _row_batch) in _Q6_ORDERED_GEOMETRIES:
            _q6_composite_name = (
                "gguf_q6_k_"
                f"{_ORDERED_COMPOSITE_VARIANT.format(variant=_variant)}"
            )
            _q6_composite = _make_q_f32_ordered_composite(
                _primitive,
                gguf_q6_k_dequantize_f32_exact,
                col_tile=_col_tile,
            )
            _q6_composite.__name__ = _q6_composite_name
            globals()[_q6_composite_name] = _q6_composite
            _Q6_ORDERED_COMPOSITES[_key] = _q6_composite
            _ORDERED_EXPORT_NAMES.append(_q6_composite_name)
for _col_tile, _row_batch, _output_dtype in _ORDERED_WEIGHT_MAJOR_GEOMETRIES:
    _variant = _ORDERED_WEIGHT_MAJOR_PRIMITIVE_VARIANT.format(
        col_tile=_col_tile,
        row_batch=_row_batch,
        output_dtype=_output_dtype,
    )
    _primitive_name = f"gguf_q5_k_f32_weight_{_variant}"
    _composite_name = f"gguf_q5_k_f32_{_variant}"
    _primitive = _make_q5_f32_weight_ordered_weight_major(
        _col_tile,
        _row_batch,
        _output_dtype,
    )
    _primitive.__name__ = _primitive_name
    _composite = _make_q_f32_ordered_composite(
        _primitive,
        gguf_q5_k_dequantize_f32_exact,
        col_tile=_col_tile,
    )
    _composite.__name__ = _composite_name
    globals()[_primitive_name] = _primitive
    globals()[_composite_name] = _composite
    _key = (_col_tile, _row_batch, _output_dtype)
    _ORDERED_WEIGHT_MAJOR_PRIMITIVES[_key] = _primitive
    _ORDERED_WEIGHT_MAJOR_COMPOSITES[_key] = _composite
    _ORDERED_EXPORT_NAMES.extend((_primitive_name, _composite_name))
    if _key in _Q6_ORDERED_WEIGHT_MAJOR_GEOMETRIES:
        _q6_composite_name = f"gguf_q6_k_f32_{_variant}"
        _q6_composite = _make_q_f32_ordered_composite(
            _primitive,
            gguf_q6_k_dequantize_f32_exact,
            col_tile=_col_tile,
        )
        _q6_composite.__name__ = _q6_composite_name
        globals()[_q6_composite_name] = _q6_composite
        _Q6_ORDERED_WEIGHT_MAJOR_COMPOSITES[_key] = _q6_composite
        _ORDERED_EXPORT_NAMES.append(_q6_composite_name)
del _col_tile, _row_batch, _output_dtype, _variant
del _primitive_name, _composite_name, _q6_composite_name
del _primitive, _composite, _q6_composite, _key


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
        KernelKey("hip_gfx1100", "dequant", "gguf_q6_k", _DEQUANT_VARIANT),
        gguf_q6_k_dequantize_f32_exact,
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
    for (col_tile, row_batch, output_dtype), primitive in (
        _ORDERED_PRIMITIVES.items()
    ):
        composite = _ORDERED_COMPOSITES[(col_tile, row_batch, output_dtype)]
        q6_composite = _Q6_ORDERED_COMPOSITES.get(
            (col_tile, row_batch, output_dtype)
        )
        variant = _ORDERED_PRIMITIVE_VARIANT.format(
            col_tile=col_tile,
            row_batch=row_batch,
            output_dtype=output_dtype,
        )
        register(
            KernelKey("hip_gfx1100", "linear", "f32_weight", variant),
            primitive,
            replace=replace,
        )
        register(
            KernelKey(
                "hip_gfx1100",
                "linear",
                "gguf_q5_k",
                _ORDERED_COMPOSITE_VARIANT.format(variant=variant),
            ),
            composite,
            replace=replace,
        )
        if q6_composite is not None:
            register(
                KernelKey(
                    "hip_gfx1100",
                    "linear",
                    "gguf_q6_k",
                    _ORDERED_COMPOSITE_VARIANT.format(variant=variant),
                ),
                q6_composite,
                replace=replace,
            )
    for (col_tile, row_batch, output_dtype), primitive in (
        _ORDERED_WEIGHT_MAJOR_PRIMITIVES.items()
    ):
        composite = _ORDERED_WEIGHT_MAJOR_COMPOSITES[
            (col_tile, row_batch, output_dtype)
        ]
        q6_composite = _Q6_ORDERED_WEIGHT_MAJOR_COMPOSITES.get(
            (col_tile, row_batch, output_dtype)
        )
        variant = _ORDERED_WEIGHT_MAJOR_PRIMITIVE_VARIANT.format(
            col_tile=col_tile,
            row_batch=row_batch,
            output_dtype=output_dtype,
        )
        register(
            KernelKey("hip_gfx1100", "linear", "f32_weight", variant),
            primitive,
            replace=replace,
        )
        register(
            KernelKey(
                "hip_gfx1100",
                "linear",
                "gguf_q5_k",
                _ORDERED_COMPOSITE_VARIANT.format(variant=variant),
            ),
            composite,
            replace=replace,
        )
        if q6_composite is not None:
            register(
                KernelKey(
                    "hip_gfx1100",
                    "linear",
                    "gguf_q6_k",
                    _ORDERED_COMPOSITE_VARIANT.format(variant=variant),
                ),
                q6_composite,
                replace=replace,
            )


register_gguf_q5_k_f32_rocblas_prefill_kernels()


__all__ = [
    "build_gguf_q5_k_f32_rocblas_prefill",
    "gguf_q5_k_dequantize_bf16_to_f32_exact_fused",
    "gguf_q5_k_dequantize_f32_exact",
    "gguf_q6_k_dequantize_f32_exact",
    "gguf_q5_k_f32_rocblas_bf16_bf16_out",
    "gguf_q5_k_f32_rocblas_bf16_f32_out",
    "plan_gguf_q5_k_f32_rocblas_prefill_build",
    "q5_k_f32_ordered_workspace_nbytes",
    "q5_k_f32_input_nbytes",
    "q5_k_f32_output_nbytes",
    "q5_k_f32_rocblas_session_nbytes",
    "q5_k_f32_rocblas_workspace_nbytes",
    "q5_k_f32_weight_nbytes",
    "q6_k_f32_ordered_workspace_nbytes",
    "register_gguf_q5_k_f32_rocblas_prefill_kernels",
    *_ORDERED_EXPORT_NAMES,
]
