"""Raw-pointer wrappers for GGUF Q4_K GEMV kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q4_k_gemv.hip")
_OUTPUT_NAME = "gguf_q4_k_gemv.so"
_SYMBOL_F32_F32_OUT = "hipengine_gguf_q4_k_gemv_f32_f32_out"
_SYMBOL_FP16_F32_OUT = "hipengine_gguf_q4_k_gemv_fp16_f32_out"
_SYMBOL_BF16_F32_OUT = "hipengine_gguf_q4_k_gemv_bf16_f32_out"
_ALLOWED_THREADS = {64, 128, 256}
_Q4_K_BLOCK = 256


def plan_gguf_q4_k_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q4_k_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q4_k_gemv(
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
        family="gguf_q4_k_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gguf_q4_k_gemv_f32_f32_out(
    x_ptr: int,
    qweight_ptr: int,
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
    """Launch GGUF Q4_K GEMV with FP32 activation and FP32 output."""

    _launch(
        _SYMBOL_F32_F32_OUT,
        x_ptr,
        qweight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_gemv_fp16_f32_out(
    x_ptr: int,
    qweight_ptr: int,
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
    """Launch GGUF Q4_K GEMV with FP16 activation and FP32 output."""

    _launch(
        _SYMBOL_FP16_F32_OUT,
        x_ptr,
        qweight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_gemv_bf16_f32_out(
    x_ptr: int,
    qweight_ptr: int,
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
    """Launch GGUF Q4_K GEMV with BF16-bit activation and FP32 output."""

    _launch(
        _SYMBOL_BF16_F32_OUT,
        x_ptr,
        qweight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def register_gguf_q4_k_gemv_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "gemv_f32_f32_out"),
        gguf_q4_k_gemv_f32_f32_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "gemv_fp16_f32_out"),
        gguf_q4_k_gemv_fp16_f32_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "gemv_bf16_f32_out"),
        gguf_q4_k_gemv_bf16_f32_out,
        replace=replace,
    )


def _launch(
    symbol: str,
    x_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _validate(rows, in_features, out_features, threads)
    library = library or build_gguf_q4_k_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
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
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _validate(rows: int, in_features: int, out_features: int, threads: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if in_features % _Q4_K_BLOCK != 0:
        raise ValueError("in_features must be divisible by GGUF Q4_K block size 256")
    if threads not in _ALLOWED_THREADS:
        allowed = ", ".join(str(value) for value in sorted(_ALLOWED_THREADS))
        raise ValueError(f"threads must be one of {allowed}")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_gguf_q4_k_gemv_kernels()


__all__ = [
    "build_gguf_q4_k_gemv",
    "gguf_q4_k_gemv_bf16_f32_out",
    "gguf_q4_k_gemv_f32_f32_out",
    "gguf_q4_k_gemv_fp16_f32_out",
    "plan_gguf_q4_k_gemv_build",
    "register_gguf_q4_k_gemv_kernels",
]
