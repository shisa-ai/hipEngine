"""Raw-pointer wrappers for the PARO Marlin-K v0 FP16 GEMV decode kernel."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("paro_marlin_k.hip")
_OUTPUT_NAME = "paro_marlin_k.so"
_SYMBOL_FMA_FP16 = "hipengine_gemv_paro_marlin_k_fma_fp16"
_SYMBOL_FMA_MULTI_ROW_FP16 = "hipengine_gemv_paro_marlin_k_fma_multi_row_fp16"
_ALLOWED_THREADS = {32, 64, 128}


def plan_paro_marlin_k_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="paro_marlin_k",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_paro_marlin_k(
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
        family="paro_marlin_k",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def marlin_k_default_threads(in_features: int, out_features: int) -> int:
    """Match the retained parent wrapper's static shape thread choice."""

    if in_features <= 0 or out_features <= 0:
        raise ValueError("in_features and out_features must be positive")
    if in_features >= 4096 or (in_features == 2048 and out_features <= 2048) or out_features <= 512:
        return 128
    return 64


def gemv_paro_marlin_k_fma_fp16(
    x_ptr: int,
    qweight_mk_ptr: int,
    qzeros_mk_ptr: int,
    scales_mk_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int = 128,
    *,
    threads: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the Marlin-K v0 FP16 GEMV kernel.

    Layout contract:
    - ``qweight_mk``: int32 ``[out_packed, in_features/128, 128]``
    - ``qzeros_mk``: int32 ``[out_packed, in_features/128]``
    - ``scales_mk``: fp16 ``[out_packed, in_features/128, 8]``
    """

    if threads is None:
        threads = marlin_k_default_threads(int(in_features), int(out_packed) * 8)
    _validate_marlin_k_args(rows, in_features, out_packed, group_size, threads)
    library = library or build_paro_marlin_k(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_FMA_FP16)
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(qweight_mk_ptr),
        ctypes.c_void_p(qzeros_mk_ptr),
        ctypes.c_void_p(scales_mk_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_packed),
        ctypes.c_int64(group_size),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if err != HIP_SUCCESS:
        raise RuntimeError(f"{_SYMBOL_FMA_FP16} failed with HIP error {err}: {runtime.get_error_string(err)}")


def gemv_paro_marlin_k_fma_multi_row_fp16(
    x_ptr: int,
    qweight_mk_ptr: int,
    qzeros_mk_ptr: int,
    scales_mk_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int = 128,
    *,
    threads: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """M15.2: weight-amortized multi-row Marlin-K GEMV for ``1 <= rows <= 8``.

    Reads each weight element once and FMAs into all ``rows`` row accumulators in
    the same k-order and per-row reduction as ``gemv_paro_marlin_k_fma_fp16``, so
    each row is bit-identical to the single-row kernel.  Same layout contract.
    """

    if int(rows) < 1 or int(rows) > 8:
        raise ValueError("multi-row Marlin-K requires 1 <= rows <= 8")
    if threads is None:
        threads = marlin_k_default_threads(int(in_features), int(out_packed) * 8)
    _validate_marlin_k_args(rows, in_features, out_packed, group_size, threads)
    library = library or build_paro_marlin_k(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_FMA_MULTI_ROW_FP16)
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(qweight_mk_ptr),
        ctypes.c_void_p(qzeros_mk_ptr),
        ctypes.c_void_p(scales_mk_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_packed),
        ctypes.c_int64(group_size),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if err != HIP_SUCCESS:
        raise RuntimeError(
            f"{_SYMBOL_FMA_MULTI_ROW_FP16} failed with HIP error {err}: {runtime.get_error_string(err)}"
        )


def register_paro_marlin_k_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="marlin_k_gemv",
            quant="w4_paro",
            variant="fma_fp16",
        ),
        gemv_paro_marlin_k_fma_fp16,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="marlin_k_gemv",
            quant="w4_paro",
            variant="fma_multi_row_fp16",
        ),
        gemv_paro_marlin_k_fma_multi_row_fp16,
        replace=replace,
    )


def _validate_marlin_k_args(rows: int, in_features: int, out_packed: int, group_size: int, threads: int) -> None:
    if int(rows) <= 0:
        raise ValueError("rows must be positive")
    if int(in_features) <= 0:
        raise ValueError("in_features must be positive")
    if int(out_packed) <= 0:
        raise ValueError("out_packed must be positive")
    if int(group_size) != 128:
        raise ValueError("Marlin-K v0 requires group_size=128")
    if int(in_features) % int(group_size) != 0:
        raise ValueError("in_features must be divisible by group_size")
    if int(threads) not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 32, 64, or 128")


register_paro_marlin_k_kernels()


__all__ = [
    "build_paro_marlin_k",
    "gemv_paro_marlin_k_fma_fp16",
    "gemv_paro_marlin_k_fma_multi_row_fp16",
    "marlin_k_default_threads",
    "plan_paro_marlin_k_build",
    "register_paro_marlin_k_kernels",
]
