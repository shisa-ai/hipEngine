"""Raw-pointer wrappers for PARO dense BF16 GEMV."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_ARGTYPES_DENSE_GEMV_SINGLE = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,        # x, weight, out
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # rows, in_features, out_features, threads
    ctypes.c_void_p,                                          # stream
)
_ARGTYPES_DENSE_GEMV_RESIDUAL = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # x, weight, residual, out
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # rows, in_features, out_features, threads
    ctypes.c_void_p,
)
_ARGTYPES_DENSE_GEMV_F32W_PAIR = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # x, w_a, w_b, out_a, out_b
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # rows, in_features, out_features, threads
    ctypes.c_void_p,
)
_ARGTYPES_DENSE_GEMV_DUAL = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # x, w_a, w_b, out
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # rows, in_features, out_a, out_b, threads
    ctypes.c_void_p,
)
_ARGTYPES_DENSE_GEMV_DUAL_SEPARATE = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # x, w_a, w_b, out_a, out_b
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # rows, in_features, out_a, out_b, threads
    ctypes.c_void_p,
)
_ARGTYPES_DENSE_GEMV_SINGLE_WMMA = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,        # x, weight, out
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,            # rows, in_features, out_features
    ctypes.c_void_p,                                          # stream
)
_ARGTYPES_DENSE_GEMV_DUAL_WMMA = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # x, w_a, w_b, out
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,       # rows, in_features, out_a, out_b
    ctypes.c_void_p,
)

_SOURCE = Path(__file__).with_name("dense_gemv.hip")
_OUTPUT_NAME = "dense_gemv.so"
_SYMBOL_BF16_OUT = "hipengine_dense_gemv_out_bf16"
_SYMBOL_BF16_RESIDUAL_OUT = "hipengine_dense_gemv_out_bf16_residual_bf16_out"
_SYMBOL_BF16_VIRTUAL256_OUT = "hipengine_dense_gemv_virtual256_out_bf16"
_SYMBOL_BF16_VIRTUAL256_ROWTILE_OUT = "hipengine_dense_gemv_virtual256_rowtile_out_bf16"
_SYMBOL_BF16_ROWTILE_OUT = "hipengine_dense_gemv_rowtile_out_bf16"
_SYMBOL_BF16_F32W_BF16_OUT = "hipengine_dense_gemv_bf16_f32w_bf16_out"
_SYMBOL_BF16_F32W_PAIR_BF16_OUT = "hipengine_dense_pair_gemv_bf16_f32w_bf16_out"
_SYMBOL_BF16_F32W_PAIR_ROWTILE2_BF16_OUT = (
    "hipengine_dense_pair_gemv_bf16_f32w_bf16_out_rowtile2"
)
_SYMBOL_FP16_OUT = "hipengine_dense_gemv_out_fp16"
_SYMBOL_F32_OUT = "hipengine_dense_gemv_out_f32"
_SYMBOL_DENSE_PREFILL_BF16_OUT = "hipengine_dense_prefill_gemm_out_bf16"
_SYMBOL_DENSE_PREFILL_WMMA_OUT_BF16 = "hipengine_dense_prefill_wmma_out_bf16"
_SYMBOL_DENSE_PREFILL_WMMA_RESIDUAL_OUT_BF16 = (
    "hipengine_dense_prefill_wmma_out_bf16_residual_bf16_out"
)
_SYMBOL_DUAL_BF16_OUT = "hipengine_dense_dual_gemv_out_bf16"
_SYMBOL_DUAL_FP16_OUT = "hipengine_dense_dual_gemv_out_fp16"
_SYMBOL_DUAL_SEPARATE_BF16_OUT = "hipengine_dense_dual_gemv_separate_out_bf16"
_SYMBOL_DUAL_SEPARATE_FP16_OUT = "hipengine_dense_dual_gemv_separate_out_fp16"
_SYMBOL_BF16_OUT_WMMA = "hipengine_dense_gemv_out_bf16_wmma"
_SYMBOL_FP16_OUT_WMMA = "hipengine_dense_gemv_out_fp16_wmma"
_SYMBOL_DUAL_BF16_OUT_WMMA = "hipengine_dense_dual_gemv_out_bf16_wmma"
_SYMBOL_DUAL_FP16_OUT_WMMA = "hipengine_dense_dual_gemv_out_fp16_wmma"
_ALLOWED_THREADS = {64, 128, 256}
_WMMA_TILE_K = 16


def plan_dense_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="dense_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_dense_gemv(
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
        family="dense_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


_DENSE_GEMV_LIBRARY: ctypes.CDLL | None = None


def _dense_gemv_library() -> ctypes.CDLL:
    """Memoized default build so per-call launches skip build_hip entirely.

    The per-call launch path calls ``build_dense_gemv(load=True)`` on every
    launch (~19 us/call of build_hip fast-path overhead even on a cache hit);
    at 141 launches/step that is ~2.6 ms/step of host CPU that lands on the
    critical path. Hoist it once, mirroring the router library pattern.
    """

    global _DENSE_GEMV_LIBRARY
    if _DENSE_GEMV_LIBRARY is None:
        _DENSE_GEMV_LIBRARY = build_dense_gemv(load=True)
    return _DENSE_GEMV_LIBRARY


def dense_gemv_out_bf16(
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_shape(rows, in_features, out_features, threads)
    library = library or _dense_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_BF16_OUT, _ARGTYPES_DENSE_GEMV_SINGLE, ctypes.c_int)
    err = fn(x_ptr, weight_ptr, out_ptr, rows, in_features, out_features, threads, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dense_gemv_out_bf16_residual_bf16_out(
    x_ptr: int,
    weight_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run dense BF16 GEMV plus an exact rounded-BF16 residual boundary."""

    _check_shape(rows, in_features, out_features, threads)
    library = library or _dense_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        _SYMBOL_BF16_RESIDUAL_OUT,
        _ARGTYPES_DENSE_GEMV_RESIDUAL,
        ctypes.c_int,
    )
    err = fn(
        x_ptr,
        weight_ptr,
        residual_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads,
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dense_gemv_virtual256_out_bf16(
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run the exact local256 arithmetic partition on four physical waves."""

    if rows != 1:
        raise ValueError("rows must equal 1 for dense BF16 virtual256 GEMV")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if threads != 128:
        raise ValueError("threads must equal 128 for dense BF16 virtual256 GEMV")
    library = library or _dense_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        _SYMBOL_BF16_VIRTUAL256_OUT,
        _ARGTYPES_DENSE_GEMV_SINGLE,
        ctypes.c_int,
    )
    err = fn(
        x_ptr,
        weight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads,
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dense_gemv_virtual256_rowtile_out_bf16(
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact local256 rowtile partitions on four physical waves."""

    if rows < 2 or rows > 4:
        raise ValueError("rows must be between 2 and 4 for dense BF16 virtual256 rowtile")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if threads != 128:
        raise ValueError("threads must equal 128 for dense BF16 virtual256 rowtile")
    library = library or _dense_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        _SYMBOL_BF16_VIRTUAL256_ROWTILE_OUT,
        _ARGTYPES_DENSE_GEMV_SINGLE,
        ctypes.c_int,
    )
    err = fn(
        x_ptr,
        weight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads,
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dense_gemv_rowtile_out_bf16(
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if rows < 2 or rows > 4:
        raise ValueError("rows must be between 2 and 4 for dense BF16 rowtile")
    _check_shape(rows, in_features, out_features, threads)
    library = library or _dense_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_BF16_ROWTILE_OUT, _ARGTYPES_DENSE_GEMV_SINGLE, ctypes.c_int)
    err = fn(x_ptr, weight_ptr, out_ptr, rows, in_features, out_features, threads, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dense_gemv_bf16_f32w_bf16_out(
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_shape(rows, in_features, out_features, threads)
    library = library or _dense_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_BF16_F32W_BF16_OUT, _ARGTYPES_DENSE_GEMV_SINGLE, ctypes.c_int)
    err = fn(x_ptr, weight_ptr, out_ptr, rows, in_features, out_features, threads, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dense_pair_gemv_bf16_f32w_bf16_out(
    x_ptr: int,
    weight_a_ptr: int,
    weight_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the exact small-row dense-F32 pair selected on gfx1100."""

    if rows < 1 or rows > 4:
        raise ValueError("rows must be between 1 and 4 for dense F32 pair")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if threads != 256:
        raise ValueError("threads must equal 256 for dense F32 pair")
    library = library or _dense_gemv_library()
    runtime = runtime or get_hip_runtime()
    symbol = (
        _SYMBOL_BF16_F32W_PAIR_ROWTILE2_BF16_OUT
        if rows == 4
        else _SYMBOL_BF16_F32W_PAIR_BF16_OUT
    )
    fn = signed_kernel_fn(
        library,
        symbol,
        _ARGTYPES_DENSE_GEMV_F32W_PAIR,
        ctypes.c_int,
    )
    err = fn(
        x_ptr,
        weight_a_ptr,
        weight_b_ptr,
        out_a_ptr,
        out_b_ptr,
        rows,
        in_features,
        out_features,
        threads,
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dense_prefill_gemm_out_bf16(
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_shape(rows, in_features, out_features, threads)
    library = library or _dense_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DENSE_PREFILL_BF16_OUT)
    fn.argtypes = [
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
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(weight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dense_gemv_out_fp16(
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_shape(rows, in_features, out_features, threads)
    library = library or _dense_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_FP16_OUT, _ARGTYPES_DENSE_GEMV_SINGLE, ctypes.c_int)
    err = fn(x_ptr, weight_ptr, out_ptr, rows, in_features, out_features, threads, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dense_gemv_out_f32(
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_shape(rows, in_features, out_features, threads)
    library = library or _dense_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_F32_OUT, _ARGTYPES_DENSE_GEMV_SINGLE, ctypes.c_int)
    err = fn(x_ptr, weight_ptr, out_ptr, rows, in_features, out_features, threads, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dense_gemv_out_bf16_wmma(
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_wmma_shape(rows, in_features, out_features)
    library = library or _dense_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_BF16_OUT_WMMA, _ARGTYPES_DENSE_GEMV_SINGLE_WMMA, ctypes.c_int)
    err = fn(x_ptr, weight_ptr, out_ptr, rows, in_features, out_features, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dense_gemv_out_fp16_wmma(
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_wmma_shape(rows, in_features, out_features)
    library = library or _dense_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_FP16_OUT_WMMA, _ARGTYPES_DENSE_GEMV_SINGLE_WMMA, ctypes.c_int)
    err = fn(x_ptr, weight_ptr, out_ptr, rows, in_features, out_features, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dense_dual_gemv_out_bf16(
    x_ptr: int,
    weight_a_ptr: int,
    weight_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_shape(rows, in_features, out_features_a, threads)
    _check_shape(rows, in_features, out_features_b, threads)
    library = library or _dense_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_DUAL_BF16_OUT, _ARGTYPES_DENSE_GEMV_DUAL, ctypes.c_int)
    err = fn(x_ptr, weight_a_ptr, weight_b_ptr, out_ptr,
             rows, in_features, out_features_a, out_features_b, threads, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dense_dual_gemv_out_fp16(
    x_ptr: int,
    weight_a_ptr: int,
    weight_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_shape(rows, in_features, out_features_a, threads)
    _check_shape(rows, in_features, out_features_b, threads)
    library = library or _dense_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_DUAL_FP16_OUT, _ARGTYPES_DENSE_GEMV_DUAL, ctypes.c_int)
    err = fn(x_ptr, weight_a_ptr, weight_b_ptr, out_ptr,
             rows, in_features, out_features_a, out_features_b, threads, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dense_dual_gemv_separate_out_bf16(
    x_ptr: int,
    weight_a_ptr: int,
    weight_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_shape(rows, in_features, out_features_a, threads)
    _check_shape(rows, in_features, out_features_b, threads)
    library = library or _dense_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_DUAL_SEPARATE_BF16_OUT, _ARGTYPES_DENSE_GEMV_DUAL_SEPARATE, ctypes.c_int)
    err = fn(x_ptr, weight_a_ptr, weight_b_ptr, out_a_ptr, out_b_ptr,
             rows, in_features, out_features_a, out_features_b, threads, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dense_dual_gemv_separate_out_fp16(
    x_ptr: int,
    weight_a_ptr: int,
    weight_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_shape(rows, in_features, out_features_a, threads)
    _check_shape(rows, in_features, out_features_b, threads)
    library = library or _dense_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_DUAL_SEPARATE_FP16_OUT, _ARGTYPES_DENSE_GEMV_DUAL_SEPARATE, ctypes.c_int)
    err = fn(x_ptr, weight_a_ptr, weight_b_ptr, out_a_ptr, out_b_ptr,
             rows, in_features, out_features_a, out_features_b, threads, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dense_dual_gemv_out_bf16_wmma(
    x_ptr: int,
    weight_a_ptr: int,
    weight_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_wmma_shape(rows, in_features, out_features_a)
    if out_features_b <= 0:
        raise ValueError("out_features_b must be positive")
    library = library or _dense_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_DUAL_BF16_OUT_WMMA, _ARGTYPES_DENSE_GEMV_DUAL_WMMA, ctypes.c_int)
    err = fn(x_ptr, weight_a_ptr, weight_b_ptr, out_ptr, rows, in_features, out_features_a, out_features_b, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dense_dual_gemv_out_fp16_wmma(
    x_ptr: int,
    weight_a_ptr: int,
    weight_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_wmma_shape(rows, in_features, out_features_a)
    if out_features_b <= 0:
        raise ValueError("out_features_b must be positive")
    library = library or _dense_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_DUAL_FP16_OUT_WMMA, _ARGTYPES_DENSE_GEMV_DUAL_WMMA, ctypes.c_int)
    err = fn(x_ptr, weight_a_ptr, weight_b_ptr, out_ptr, rows, in_features, out_features_a, out_features_b, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dense_prefill_wmma_out_bf16(
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the D08-X2-K5 LDS-staged 128x64 WMMA dense BF16 bulk GEMM."""

    if rows <= 0 or in_features <= 0 or out_features <= 0:
        raise ValueError("shape must be positive")
    if in_features % 32:
        raise ValueError("dense WMMA prefill requires in_features % 32 == 0")
    if out_features % 128:
        raise ValueError("dense WMMA prefill requires out_features % 128 == 0")
    library = library or _dense_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DENSE_PREFILL_WMMA_OUT_BF16)
    fn.argtypes = [
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
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(weight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dense_prefill_wmma_out_bf16_residual_bf16_out(
    x_ptr: int,
    weight_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch WMMA dense projection plus an exact rounded-BF16 residual."""

    if rows <= 0 or in_features <= 0 or out_features <= 0:
        raise ValueError("shape must be positive")
    if in_features % 32:
        raise ValueError("dense WMMA prefill requires in_features % 32 == 0")
    if out_features % 128:
        raise ValueError("dense WMMA prefill requires out_features % 128 == 0")
    library = library or _dense_gemv_library()
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DENSE_PREFILL_WMMA_RESIDUAL_OUT_BF16)
    fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 3 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(weight_ptr),
        ctypes.c_void_p(residual_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_dense_gemv_kernels(*, replace: bool = True) -> None:
    for quant in ("bf16", "w4_paro"):
        register(
            KernelKey("hip_gfx1100", "dense_gemv", quant, "out"),
            dense_gemv_out_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "dense_gemv", quant, "prefill_out"),
            dense_prefill_gemm_out_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "dense_gemv", quant, "prefill_wmma_out"),
            dense_prefill_wmma_out_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "dense_gemv", quant, "rowtile_out"),
            dense_gemv_rowtile_out_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "dense_gemv", quant, "virtual256_out"),
            dense_gemv_virtual256_out_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "dense_gemv", quant, "virtual256_rowtile_out"),
            dense_gemv_virtual256_rowtile_out_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "dense_dual_gemv", quant, "out"),
            dense_dual_gemv_out_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "dense_gemv", quant, "out_fp16"),
            dense_gemv_out_fp16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "dense_dual_gemv", quant, "out_fp16"),
            dense_dual_gemv_out_fp16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "dense_dual_gemv", quant, "separate_out"),
            dense_dual_gemv_separate_out_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "dense_dual_gemv", quant, "separate_out_fp16"),
            dense_dual_gemv_separate_out_fp16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "dense_gemv", quant, "out_wmma"),
            dense_gemv_out_bf16_wmma,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "dense_dual_gemv", quant, "out_wmma"),
            dense_dual_gemv_out_bf16_wmma,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "dense_gemv", quant, "out_fp16_wmma"),
            dense_gemv_out_fp16_wmma,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "dense_dual_gemv", quant, "out_fp16_wmma"),
            dense_dual_gemv_out_fp16_wmma,
            replace=replace,
        )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear+residual",
            "bf16",
            "out_bf16_residual_bf16_out",
        ),
        dense_gemv_out_bf16_residual_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear+residual",
            "bf16",
            "prefill_wmma_out_bf16_residual_bf16_out",
        ),
        dense_prefill_wmma_out_bf16_residual_bf16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dense_gemv", "f32", "bf16_hidden_bf16_out"),
        dense_gemv_bf16_f32w_bf16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dense_gemv", "f32", "f32_hidden_f32_out"),
        dense_gemv_out_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear_pair", "f32", "bf16_hidden_bf16_out"),
        dense_pair_gemv_bf16_f32w_bf16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dense_gemv", "fp16", "out"),
        dense_gemv_out_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dense_dual_gemv", "fp16", "out"),
        dense_dual_gemv_out_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dense_dual_gemv", "fp16", "separate_out"),
        dense_dual_gemv_separate_out_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dense_gemv", "fp16", "out_wmma"),
        dense_gemv_out_fp16_wmma,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dense_dual_gemv", "fp16", "out_wmma"),
        dense_dual_gemv_out_fp16_wmma,
        replace=replace,
    )


def _check_shape(rows: int, in_features: int, out_features: int, threads: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, or 256")


def _check_wmma_shape(rows: int, in_features: int, out_features: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if (in_features % _WMMA_TILE_K) != 0:
        raise ValueError("in_features must be a multiple of 16 for WMMA dense GEMV")


register_dense_gemv_kernels()
