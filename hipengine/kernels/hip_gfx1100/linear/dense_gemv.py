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
_SYMBOL_FP16_OUT = "hipengine_dense_gemv_out_fp16"
_SYMBOL_DENSE_PREFILL_BF16_OUT = "hipengine_dense_prefill_gemm_out_bf16"
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
    library = library or build_dense_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_BF16_OUT, _ARGTYPES_DENSE_GEMV_SINGLE, ctypes.c_int)
    err = fn(x_ptr, weight_ptr, out_ptr, rows, in_features, out_features, threads, stream)
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
    library = library or build_dense_gemv(load=True)
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
    library = library or build_dense_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_FP16_OUT, _ARGTYPES_DENSE_GEMV_SINGLE, ctypes.c_int)
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
    library = library or build_dense_gemv(load=True)
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
    library = library or build_dense_gemv(load=True)
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
    library = library or build_dense_gemv(load=True)
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
    library = library or build_dense_gemv(load=True)
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
    library = library or build_dense_gemv(load=True)
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
    library = library or build_dense_gemv(load=True)
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
    library = library or build_dense_gemv(load=True)
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
    library = library or build_dense_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_DUAL_FP16_OUT_WMMA, _ARGTYPES_DENSE_GEMV_DUAL_WMMA, ctypes.c_int)
    err = fn(x_ptr, weight_a_ptr, weight_b_ptr, out_ptr, rows, in_features, out_features_a, out_features_b, stream)
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
