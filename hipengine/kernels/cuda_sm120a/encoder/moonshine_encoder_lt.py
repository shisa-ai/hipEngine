"""cuBLASLt long-bucket route epilogue kernels for the CUDA ``sm_120a`` batch encoder.

Two element-wise FP16-rounding epilogues for the cuBLASLt FP32 GEMM boundary
of the encoder's fc1/fc2 projections (see ``moonshine_encoder_lt.cu``):

- ``moonshine_f16_bias_round_fp32``: ``out = fp16(gemm_f32 + bias)`` (fc1);
- ``moonshine_f16_bias_residual_round_fp32``:
  ``out = fp16(gemm_f32 + bias + residual)`` (fc2, residual in-place).

These preserve the retained FP16-rounding contract (bias/residual added in
FP32, rounded once) so the only divergence from the exact custom route is the
cuBLASLt tensor-core reduction order -- the accepted C8 re-derived numerical
gate, not a boundary-rounding change.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_cuda, plan_cuda_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.cuda import CUDA_SUCCESS, CudaRuntime, get_cuda_runtime
from hipengine.kernels.backends import cuda_target_arch_for_backend
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("moonshine_encoder_lt.cu")
_OUTPUT_NAME = "moonshine_encoder_lt.so"
_BACKEND = "cuda_sm120a"
_TARGET_ARCH = cuda_target_arch_for_backend(_BACKEND)
_BIAS_ARGS = (
    ctypes.c_void_p,  # gemm (fp32)
    ctypes.c_void_p,  # bias (fp16)
    ctypes.c_void_p,  # output (fp16)
    ctypes.c_int64,   # rows
    ctypes.c_int64,   # out_features
    ctypes.c_int64,   # threads
    ctypes.c_void_p,  # stream
)
_RESIDUAL_ARGS = (
    ctypes.c_void_p,  # gemm (fp32)
    ctypes.c_void_p,  # bias (fp16)
    ctypes.c_void_p,  # residual (fp16)
    ctypes.c_void_p,  # output (fp16)
    ctypes.c_int64,   # rows
    ctypes.c_int64,   # out_features
    ctypes.c_int64,   # threads
    ctypes.c_void_p,  # stream
)
_ALLOWED_THREADS = (64, 128, 256)


def plan_moonshine_encoder_lt_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_cuda_build(
        sources=[_SOURCE],
        family="cuda_sm120a_moonshine_encoder_lt",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        output_name=_OUTPUT_NAME,
    )


def build_moonshine_encoder_lt(
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
        family="cuda_sm120a_moonshine_encoder_lt",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _launch(
    library: ctypes.CDLL,
    symbol: str,
    arg_types: tuple[object, ...],
    args: tuple[object, ...],
    runtime: CudaRuntime,
) -> None:
    function = signed_kernel_fn(library, symbol, arg_types, ctypes.c_int)
    error = function(*args)
    if int(error) != CUDA_SUCCESS:
        runtime.check(int(error))


def moonshine_f16_bias_round_fp32(
    gemm_ptr: int,
    bias_ptr: int,
    output_ptr: int,
    rows: int,
    out_features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    if rows <= 0 or out_features <= 0:
        raise ValueError("rows and out_features must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError(f"threads must be one of {_ALLOWED_THREADS}")
    library = library or build_moonshine_encoder_lt(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_f16_bias_round_fp32",
        _BIAS_ARGS,
        (
            gemm_ptr,
            bias_ptr,
            output_ptr,
            rows,
            out_features,
            threads,
            stream,
        ),
        runtime,
    )


def moonshine_f16_bias_residual_round_fp32(
    gemm_ptr: int,
    bias_ptr: int,
    residual_ptr: int,
    output_ptr: int,
    rows: int,
    out_features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    if rows <= 0 or out_features <= 0:
        raise ValueError("rows and out_features must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError(f"threads must be one of {_ALLOWED_THREADS}")
    library = library or build_moonshine_encoder_lt(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_f16_bias_residual_round_fp32",
        _RESIDUAL_ARGS,
        (
            gemm_ptr,
            bias_ptr,
            residual_ptr,
            output_ptr,
            rows,
            out_features,
            threads,
            stream,
        ),
        runtime,
    )


def register_moonshine_encoder_lt_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey(_BACKEND, "moonshine_bias_round", "fp32_gemm", "fp16_out"),
        moonshine_f16_bias_round_fp32,
        replace=replace,
    )
    register(
        KernelKey(
            _BACKEND,
            "moonshine_bias_residual_round",
            "fp32_gemm",
            "fp16_out",
        ),
        moonshine_f16_bias_residual_round_fp32,
        replace=replace,
    )


register_moonshine_encoder_lt_kernels()

__all__ = [
    "build_moonshine_encoder_lt",
    "moonshine_f16_bias_round_fp32",
    "moonshine_f16_bias_residual_round_fp32",
    "plan_moonshine_encoder_lt_build",
    "register_moonshine_encoder_lt_kernels",
]
