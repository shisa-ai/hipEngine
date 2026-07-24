"""Raw-pointer dtype cast helpers for small runtime glue buffers."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("cast.hip")
_OUTPUT_NAME = "cast.so"
_SYMBOL_F32_TO_BF16 = "hipengine_f32_to_bf16"
_SYMBOL_BF16_TO_F32 = "hipengine_bf16_to_f32"
_SYMBOL_F32_TO_FP16 = "hipengine_f32_to_fp16"
_SYMBOL_FP16_TO_F32 = "hipengine_fp16_to_f32"
_SYMBOL_FP16_TO_BF16 = "hipengine_fp16_to_bf16"
_SYMBOL_BF16_TO_FP16_SCALED_ROWS = "hipengine_bf16_to_fp16_scaled_rows"
_SYMBOL_F32_SCALE_ROWS_TO_BF16 = "hipengine_f32_scale_rows_to_bf16"
_SYMBOL_F32_SCALE_ROWS = "hipengine_f32_scale_rows"
_SYMBOL_FP16_TO_BF16_STRIDED_ROWS = "hipengine_fp16_to_bf16_strided_rows"


def plan_cast_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="cast",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_cast(
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
        family="cast",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def f32_to_bf16(
    x_ptr: int,
    out_ptr: int,
    count: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Convert a contiguous FP32 buffer to BF16 bits."""

    _launch_cast(_SYMBOL_F32_TO_BF16, x_ptr, out_ptr, count, stream=stream, library=library, runtime=runtime)


def bf16_to_f32(
    x_ptr: int,
    out_ptr: int,
    count: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Convert a contiguous BF16-bit buffer to FP32."""

    _launch_cast(_SYMBOL_BF16_TO_F32, x_ptr, out_ptr, count, stream=stream, library=library, runtime=runtime)


def f32_to_fp16(
    x_ptr: int,
    out_ptr: int,
    count: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Convert a contiguous FP32 buffer to FP16."""

    _launch_cast(_SYMBOL_F32_TO_FP16, x_ptr, out_ptr, count, stream=stream, library=library, runtime=runtime)


def fp16_to_f32(
    x_ptr: int,
    out_ptr: int,
    count: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Convert a contiguous FP16 buffer to FP32."""

    _launch_cast(_SYMBOL_FP16_TO_F32, x_ptr, out_ptr, count, stream=stream, library=library, runtime=runtime)


def fp16_to_bf16(
    x_ptr: int,
    out_ptr: int,
    count: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Convert a contiguous FP16 buffer to BF16 bits."""

    _launch_cast(_SYMBOL_FP16_TO_BF16, x_ptr, out_ptr, count, stream=stream, library=library, runtime=runtime)


def bf16_to_fp16_scaled_rows(
    x_ptr: int,
    out_ptr: int,
    row_scales_ptr: int,
    rows: int,
    cols: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Convert BF16 rows to finite FP16 using an exact power-of-two row scale."""

    _launch_scaled_rows(
        _SYMBOL_BF16_TO_FP16_SCALED_ROWS,
        x_ptr,
        out_ptr,
        row_scales_ptr,
        rows,
        cols,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def f32_scale_rows_to_bf16(
    x_ptr: int,
    row_scales_ptr: int,
    out_ptr: int,
    rows: int,
    cols: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Restore per-row power-of-two scale while converting FP32 to BF16."""

    _launch_scaled_rows(
        _SYMBOL_F32_SCALE_ROWS_TO_BF16,
        x_ptr,
        row_scales_ptr,
        out_ptr,
        rows,
        cols,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def f32_scale_rows(
    x_ptr: int,
    row_scales_ptr: int,
    rows: int,
    cols: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Restore a per-row power-of-two scale on an FP32 matrix in place."""

    _launch_scaled_rows(
        _SYMBOL_F32_SCALE_ROWS,
        x_ptr,
        row_scales_ptr,
        0,
        rows,
        cols,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def fp16_to_bf16_strided_rows(
    x_ptr: int,
    out_ptr: int,
    rows: int,
    cols: int,
    dst_row_stride: int,
    dst_col_offset: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Convert contiguous FP16 rows into a strided BF16 row-major destination."""

    _launch_cast_strided_rows(
        _SYMBOL_FP16_TO_BF16_STRIDED_ROWS,
        x_ptr,
        out_ptr,
        rows,
        cols,
        dst_row_stride,
        dst_col_offset,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def register_cast_kernels(*, replace: bool = True) -> None:
    register(KernelKey("hip_gfx1100", "cast_f32_to_bf16", "bf16"), f32_to_bf16, replace=replace)
    register(KernelKey("hip_gfx1100", "cast_bf16_to_f32", "fp32"), bf16_to_f32, replace=replace)
    register(KernelKey("hip_gfx1100", "cast_f32_to_fp16", "fp16"), f32_to_fp16, replace=replace)
    register(KernelKey("hip_gfx1100", "cast_fp16_to_f32", "fp32"), fp16_to_f32, replace=replace)
    register(KernelKey("hip_gfx1100", "cast_fp16_to_bf16", "bf16"), fp16_to_bf16, replace=replace)
    register(
        KernelKey("hip_gfx1100", "cast_bf16_to_fp16", "scaled_rows"),
        bf16_to_fp16_scaled_rows,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "cast_f32_to_bf16", "scaled_rows"),
        f32_scale_rows_to_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "cast_f32_scale_rows", "fp32"),
        f32_scale_rows,
        replace=replace,
    )
    register(KernelKey("hip_gfx1100", "cast_fp16_to_bf16_strided_rows", "bf16"), fp16_to_bf16_strided_rows, replace=replace)


def _launch_cast(
    symbol: str,
    x_ptr: int,
    out_ptr: int,
    count: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    if count <= 0:
        raise ValueError("count must be positive")
    library = library or build_cast(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(ctypes.c_void_p(x_ptr), ctypes.c_void_p(out_ptr), ctypes.c_int64(count), ctypes.c_void_p(stream))
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_scaled_rows(
    symbol: str,
    x_ptr: int,
    second_ptr: int,
    third_ptr: int,
    rows: int,
    cols: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if cols <= 0:
        raise ValueError("cols must be positive")
    library = library or build_cast(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    if symbol == _SYMBOL_F32_SCALE_ROWS:
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
    else:
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
    fn.restype = ctypes.c_int
    if symbol == _SYMBOL_F32_SCALE_ROWS:
        err = fn(
            ctypes.c_void_p(x_ptr),
            ctypes.c_void_p(second_ptr),
            ctypes.c_int64(rows),
            ctypes.c_int64(cols),
            ctypes.c_void_p(stream),
        )
    else:
        err = fn(
            ctypes.c_void_p(x_ptr),
            ctypes.c_void_p(second_ptr),
            ctypes.c_void_p(third_ptr),
            ctypes.c_int64(rows),
            ctypes.c_int64(cols),
            ctypes.c_void_p(stream),
        )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_cast_strided_rows(
    symbol: str,
    x_ptr: int,
    out_ptr: int,
    rows: int,
    cols: int,
    dst_row_stride: int,
    dst_col_offset: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if cols <= 0:
        raise ValueError("cols must be positive")
    if dst_row_stride < cols:
        raise ValueError("dst_row_stride must be at least cols")
    if dst_col_offset < 0 or dst_col_offset + cols > dst_row_stride:
        raise ValueError("dst_col_offset range must fit in dst_row_stride")
    library = library or build_cast(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(cols),
        ctypes.c_int64(dst_row_stride),
        ctypes.c_int64(dst_col_offset),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_cast_kernels()
