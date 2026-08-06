"""Raw-pointer Moonshine FP16 LayerNorm for CUDA ``sm_120a``.

LayerNorm is bias-free: the FP16 ``weight`` is a per-dimension scale applied
after FP32 mean/variance centering.  The fused residual+LayerNorm kernel writes
the rounded FP16 residual boundary first and reads that same buffer back so the
norm statistics are computed over the exact rounded boundary, matching the HIP
reference and the independent NumPy oracle.
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_cuda, plan_cuda_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.cuda import CUDA_SUCCESS, CudaRuntime, get_cuda_runtime
from hipengine.kernels.backends import cuda_target_arch_for_backend
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("moonshine_layernorm.cu")
_OUTPUT_NAME = "moonshine_layernorm.so"
_BACKEND = "cuda_sm120a"
_TARGET_ARCH = cuda_target_arch_for_backend(_BACKEND)
_ALLOWED_THREADS = {32, 64, 128, 256}
_LARGE_ROW_THREADS = 128
_SMALL_ROW_THREADS = 256
_LARGE_ROW_THRESHOLD = 768
_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_RESIDUAL_LAYERNORM_ARGS = (
    *(ctypes.c_void_p for _ in range(5)),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_int64,
    ctypes.c_void_p,
)


def plan_moonshine_layernorm_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_cuda_build(
        sources=[_SOURCE],
        family="cuda_sm120a_moonshine_layernorm",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        output_name=_OUTPUT_NAME,
    )


def build_moonshine_layernorm(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_cuda(
        sources=[_SOURCE],
        family="cuda_sm120a_moonshine_layernorm",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _default_threads(rows: int) -> int:
    """Measured per-bucket launch geometry on ``sm_120a`` (GPU0, Blackwell).

    One block per row with ``threads`` threads.  A batch-timed CUDA-event
    screen (medians over 2000-launch batches) shows 256 threads is best for
    row counts below 768 and 128 threads is best from 768 upward, for both the
    plain LayerNorm and the fused residual+LayerNorm kernels; 256 is roughly
    40-52% slower at 1,248 rows.  Explicit ``threads=`` always overrides.
    """
    return _LARGE_ROW_THREADS if rows >= _LARGE_ROW_THRESHOLD else _SMALL_ROW_THREADS


def _validate_contract(rows: int, hidden_size: int, eps: float, threads: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 32, 64, 128, 256")


def moonshine_layernorm_fp16(
    input_ptr: int,
    weight_ptr: int,
    output_ptr: int,
    rows: int,
    hidden_size: int,
    *,
    eps: float = 1.0e-5,
    threads: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    threads = _default_threads(rows) if threads is None else threads
    _validate_contract(rows, hidden_size, eps, threads)
    library = library or build_moonshine_layernorm(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_layernorm_fp16",
        _ARGS,
        (input_ptr, weight_ptr, output_ptr, rows, hidden_size, eps, threads, stream),
        runtime,
    )


def moonshine_residual_layernorm_fp16(
    residual_ptr: int,
    update_ptr: int,
    weight_ptr: int,
    residual_output_ptr: int,
    norm_output_ptr: int,
    rows: int,
    hidden_size: int,
    *,
    eps: float = 1.0e-5,
    threads: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    threads = _default_threads(rows) if threads is None else threads
    _validate_contract(rows, hidden_size, eps, threads)
    library = library or build_moonshine_layernorm(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_residual_layernorm_fp16",
        _RESIDUAL_LAYERNORM_ARGS,
        (
            residual_ptr,
            update_ptr,
            weight_ptr,
            residual_output_ptr,
            norm_output_ptr,
            rows,
            hidden_size,
            eps,
            threads,
            stream,
        ),
        runtime,
    )


def _launch(
    library: ctypes.CDLL,
    symbol: str,
    argtypes,
    arguments: tuple[object, ...],
    runtime: CudaRuntime,
) -> None:
    fn = signed_kernel_fn(library, symbol, argtypes, ctypes.c_int)
    error = fn(*arguments)
    if int(error) != CUDA_SUCCESS:
        runtime.check(int(error))


def register_moonshine_layernorm_kernels(*, replace: bool = True) -> None:
    registrations = (
        (
            KernelKey(
                _BACKEND,
                "moonshine_layernorm",
                "fp16",
                "fp32_stats",
            ),
            moonshine_layernorm_fp16,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_residual+moonshine_layernorm",
                "fp16",
                "rounded_fp32_stats",
            ),
            moonshine_residual_layernorm_fp16,
        ),
    )
    for key, kernel in registrations:
        register(key, kernel, replace=replace)


register_moonshine_layernorm_kernels()

__all__ = [
    "build_moonshine_layernorm",
    "moonshine_layernorm_fp16",
    "moonshine_residual_layernorm_fp16",
    "plan_moonshine_layernorm_build",
    "register_moonshine_layernorm_kernels",
]
