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
_SYMBOL_F32_FP16_OUT = "hipengine_gguf_q4_k_gemv_f32_fp16_out"
_SYMBOL_FP16_F32_OUT = "hipengine_gguf_q4_k_gemv_fp16_f32_out"
_SYMBOL_FP16_FP16_OUT = "hipengine_gguf_q4_k_gemv_fp16_fp16_out"
_SYMBOL_BF16_F32_OUT = "hipengine_gguf_q4_k_gemv_bf16_f32_out"
_SYMBOL_BF16_FP16_OUT = "hipengine_gguf_q4_k_gemv_bf16_fp16_out"
_SYMBOL_BF16_BF16_OUT = "hipengine_gguf_q4_k_gemv_bf16_bf16_out"
_SYMBOL_SELECTED_BF16_BF16_OUT = "hipengine_gguf_q4_k_selected_gemv_bf16_bf16_out"
_SYMBOL_SELECTED_DUAL_BF16_BF16_OUT = "hipengine_gguf_q4_k_selected_dual_gemv_bf16_bf16_out"
_SYMBOL_SELECTED_PACK8_BF16_BF16_OUT = "hipengine_gguf_q4_k_selected_pack8_gemv_bf16_bf16_out"
_SYMBOL_PACK8_F32_F32_OUT = "hipengine_gguf_q4_k_pack8_gemv_f32_f32_out"
_SYMBOL_PACK8_F32_FP16_OUT = "hipengine_gguf_q4_k_pack8_gemv_f32_fp16_out"
_SYMBOL_PACK8_FP16_F32_OUT = "hipengine_gguf_q4_k_pack8_gemv_fp16_f32_out"
_SYMBOL_PACK8_FP16_FP16_OUT = "hipengine_gguf_q4_k_pack8_gemv_fp16_fp16_out"
_SYMBOL_PACK8_BF16_F32_OUT = "hipengine_gguf_q4_k_pack8_gemv_bf16_f32_out"
_SYMBOL_PACK8_BF16_FP16_OUT = "hipengine_gguf_q4_k_pack8_gemv_bf16_fp16_out"
_SYMBOL_PACK8_BF16_BF16_OUT = "hipengine_gguf_q4_k_pack8_gemv_bf16_bf16_out"
_SYMBOL_PACK8_DUAL_BF16_BF16_OUT = "hipengine_gguf_q4_k_pack8_dual_gemv_bf16_bf16_out"
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


def gguf_q4_k_gemv_bf16_bf16_out(
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
    """Launch GGUF Q4_K GEMV with BF16-bit activation and BF16 output."""

    _launch(
        _SYMBOL_BF16_BF16_OUT,
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


def gguf_q4_k_pack8_gemv_f32_f32_out(
    x_ptr: int,
    qweight_ptr: int,
    scales_ptr: int,
    mins_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch repacked GGUF Q4_K GEMV with FP32 activation and FP32 output."""

    _launch_pack8(
        _SYMBOL_PACK8_F32_F32_OUT,
        x_ptr,
        qweight_ptr,
        scales_ptr,
        mins_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_pack8_gemv_fp16_f32_out(
    x_ptr: int,
    qweight_ptr: int,
    scales_ptr: int,
    mins_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch repacked GGUF Q4_K GEMV with FP16 activation and FP32 output."""

    _launch_pack8(
        _SYMBOL_PACK8_FP16_F32_OUT,
        x_ptr,
        qweight_ptr,
        scales_ptr,
        mins_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_pack8_gemv_bf16_f32_out(
    x_ptr: int,
    qweight_ptr: int,
    scales_ptr: int,
    mins_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch repacked GGUF Q4_K GEMV with BF16-bit activation and FP32 output."""

    _launch_pack8(
        _SYMBOL_PACK8_BF16_F32_OUT,
        x_ptr,
        qweight_ptr,
        scales_ptr,
        mins_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_pack8_gemv_bf16_bf16_out(
    x_ptr: int,
    qweight_ptr: int,
    scales_ptr: int,
    mins_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch repacked GGUF Q4_K GEMV with BF16-bit activation and BF16 output."""

    _launch_pack8(
        _SYMBOL_PACK8_BF16_BF16_OUT,
        x_ptr,
        qweight_ptr,
        scales_ptr,
        mins_ptr,
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
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "gemv_bf16_bf16_out"),
        gguf_q4_k_gemv_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "pack8_f32_f32_out"),
        gguf_q4_k_pack8_gemv_f32_f32_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "pack8_fp16_f32_out"),
        gguf_q4_k_pack8_gemv_fp16_f32_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "pack8_bf16_f32_out"),
        gguf_q4_k_pack8_gemv_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "pack8_bf16_bf16_out"),
        gguf_q4_k_pack8_gemv_bf16_bf16_out,
        replace=replace,
    )
    for variant, fn in _EXTRA_Q4_K_WRAPPERS.items():
        register(KernelKey("hip_gfx1100", "linear", "gguf_q4_k", variant), fn, replace=replace)


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


def _launch_selected(
    symbol: str,
    x_ptr: int,
    selected_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _validate_selected(x_rows, rows, num_experts, in_features, out_features, threads)
    library = library or build_gguf_q4_k_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
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
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(x_rows),
        ctypes.c_int64(rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_pack8(
    symbol: str,
    x_ptr: int,
    qweight_ptr: int,
    scales_ptr: int,
    mins_ptr: int,
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
    launch_threads = _pack8_threads(in_features, threads)
    _validate(rows, in_features, out_features, launch_threads, require_pack8=True)
    library = library or build_gguf_q4_k_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(scales_ptr),
        ctypes.c_void_p(mins_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(launch_threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def gguf_q4_k_pack8_dual_prefill_bf16_bf16_out(
    x_ptr: int,
    qweight_a_ptr: int,
    scales_a_ptr: int,
    mins_a_ptr: int,
    qweight_b_ptr: int,
    scales_b_ptr: int,
    mins_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    launch_threads = _pack8_threads(in_features, threads)
    _validate(rows, in_features, out_features, launch_threads, require_pack8=True)
    library = library or build_gguf_q4_k_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PACK8_DUAL_BF16_BF16_OUT)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(scales_a_ptr),
        ctypes.c_void_p(mins_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(scales_b_ptr),
        ctypes.c_void_p(mins_b_ptr),
        ctypes.c_void_p(out_a_ptr),
        ctypes.c_void_p(out_b_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(launch_threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _pack8_threads(in_features: int, threads: int) -> int:
    if threads == 0:
        return 32
    return threads


def _validate_selected(
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    threads: int,
) -> None:
    if x_rows <= 0:
        raise ValueError("x_rows must be positive")
    if rows <= 0 or rows % x_rows != 0:
        raise ValueError("rows must be positive and divisible by x_rows")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    _validate(rows, in_features, out_features, threads)


def _validate(
    rows: int,
    in_features: int,
    out_features: int,
    threads: int,
    *,
    require_pack8: bool = False,
) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if in_features % _Q4_K_BLOCK != 0:
        raise ValueError("in_features must be divisible by GGUF Q4_K block size 256")
    if require_pack8 and out_features % 8 != 0:
        raise ValueError("out_features must be divisible by 8 for GGUF Q4_K pack8")
    allowed_threads = {32, 64, 128} if require_pack8 else _ALLOWED_THREADS
    if threads not in allowed_threads:
        allowed = ", ".join(str(value) for value in sorted(allowed_threads))
        raise ValueError(f"threads must be one of {allowed}")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _make_raw_wrapper(symbol: str):
    def wrapper(*args, **kwargs) -> None:
        kwargs.setdefault("threads", 128)
        kwargs.setdefault("stream", 0)
        kwargs.setdefault("library", None)
        kwargs.setdefault("runtime", None)
        _launch(symbol, *args, **kwargs)

    return wrapper


def _make_pack8_wrapper(symbol: str):
    def wrapper(*args, **kwargs) -> None:
        kwargs.setdefault("threads", 0)
        kwargs.setdefault("stream", 0)
        kwargs.setdefault("library", None)
        kwargs.setdefault("runtime", None)
        _launch_pack8(symbol, *args, **kwargs)

    return wrapper


def gguf_q4_k_selected_dual_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate_selected(x_rows, rows, num_experts, in_features, out_features, threads)
    library = library or build_gguf_q4_k_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SELECTED_DUAL_BF16_BF16_OUT)
    fn.argtypes = [
        ctypes.c_void_p,
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_a_ptr),
        ctypes.c_void_p(out_b_ptr),
        ctypes.c_int64(x_rows),
        ctypes.c_int64(rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _make_selected_wrapper(symbol: str):
    def wrapper(*args, **kwargs) -> None:
        kwargs.setdefault("threads", 128)
        kwargs.setdefault("stream", 0)
        kwargs.setdefault("library", None)
        kwargs.setdefault("runtime", None)
        _launch_selected(symbol, *args, **kwargs)

    return wrapper


gguf_q4_k_selected_gemv_bf16_bf16_out = _make_selected_wrapper(_SYMBOL_SELECTED_BF16_BF16_OUT)
gguf_q4_k_selected_pack8_gemv_bf16_bf16_out = _make_selected_wrapper(_SYMBOL_SELECTED_PACK8_BF16_BF16_OUT)
gguf_q4_k_gemv_f32_fp16_out = _make_raw_wrapper(_SYMBOL_F32_FP16_OUT)
gguf_q4_k_gemv_fp16_fp16_out = _make_raw_wrapper(_SYMBOL_FP16_FP16_OUT)
gguf_q4_k_gemv_bf16_fp16_out = _make_raw_wrapper(_SYMBOL_BF16_FP16_OUT)
gguf_q4_k_prefill_f32_f32_out = _make_raw_wrapper(_SYMBOL_F32_F32_OUT)
gguf_q4_k_prefill_f32_fp16_out = _make_raw_wrapper(_SYMBOL_F32_FP16_OUT)
gguf_q4_k_prefill_fp16_f32_out = _make_raw_wrapper(_SYMBOL_FP16_F32_OUT)
gguf_q4_k_prefill_fp16_fp16_out = _make_raw_wrapper(_SYMBOL_FP16_FP16_OUT)
gguf_q4_k_prefill_bf16_f32_out = _make_raw_wrapper(_SYMBOL_BF16_F32_OUT)
gguf_q4_k_prefill_bf16_fp16_out = _make_raw_wrapper(_SYMBOL_BF16_FP16_OUT)
gguf_q4_k_prefill_bf16_bf16_out = _make_raw_wrapper(_SYMBOL_BF16_BF16_OUT)
gguf_q4_k_pack8_gemv_f32_fp16_out = _make_pack8_wrapper(_SYMBOL_PACK8_F32_FP16_OUT)
gguf_q4_k_pack8_gemv_fp16_fp16_out = _make_pack8_wrapper(_SYMBOL_PACK8_FP16_FP16_OUT)
gguf_q4_k_pack8_gemv_bf16_fp16_out = _make_pack8_wrapper(_SYMBOL_PACK8_BF16_FP16_OUT)
gguf_q4_k_pack8_prefill_f32_f32_out = _make_pack8_wrapper(_SYMBOL_PACK8_F32_F32_OUT)
gguf_q4_k_pack8_prefill_f32_fp16_out = _make_pack8_wrapper(_SYMBOL_PACK8_F32_FP16_OUT)
gguf_q4_k_pack8_prefill_fp16_f32_out = _make_pack8_wrapper(_SYMBOL_PACK8_FP16_F32_OUT)
gguf_q4_k_pack8_prefill_fp16_fp16_out = _make_pack8_wrapper(_SYMBOL_PACK8_FP16_FP16_OUT)
gguf_q4_k_pack8_prefill_bf16_f32_out = _make_pack8_wrapper(_SYMBOL_PACK8_BF16_F32_OUT)
gguf_q4_k_pack8_prefill_bf16_fp16_out = _make_pack8_wrapper(_SYMBOL_PACK8_BF16_FP16_OUT)
gguf_q4_k_pack8_prefill_bf16_bf16_out = _make_pack8_wrapper(_SYMBOL_PACK8_BF16_BF16_OUT)

_EXTRA_Q4_K_WRAPPERS = {
    "gemv_f32_fp16_out": gguf_q4_k_gemv_f32_fp16_out,
    "gemv_fp16_fp16_out": gguf_q4_k_gemv_fp16_fp16_out,
    "gemv_bf16_fp16_out": gguf_q4_k_gemv_bf16_fp16_out,
    "selected_dual_gemv_bf16_bf16_out": gguf_q4_k_selected_dual_gemv_bf16_bf16_out,
    "selected_pack8_gemv_bf16_bf16_out": gguf_q4_k_selected_pack8_gemv_bf16_bf16_out,
    "prefill_f32_f32_out": gguf_q4_k_prefill_f32_f32_out,
    "prefill_f32_fp16_out": gguf_q4_k_prefill_f32_fp16_out,
    "prefill_fp16_f32_out": gguf_q4_k_prefill_fp16_f32_out,
    "prefill_fp16_fp16_out": gguf_q4_k_prefill_fp16_fp16_out,
    "prefill_bf16_f32_out": gguf_q4_k_prefill_bf16_f32_out,
    "prefill_bf16_fp16_out": gguf_q4_k_prefill_bf16_fp16_out,
    "prefill_bf16_bf16_out": gguf_q4_k_prefill_bf16_bf16_out,
    "pack8_f32_fp16_out": gguf_q4_k_pack8_gemv_f32_fp16_out,
    "pack8_fp16_fp16_out": gguf_q4_k_pack8_gemv_fp16_fp16_out,
    "pack8_bf16_fp16_out": gguf_q4_k_pack8_gemv_bf16_fp16_out,
    "pack8_prefill_f32_f32_out": gguf_q4_k_pack8_prefill_f32_f32_out,
    "pack8_prefill_f32_fp16_out": gguf_q4_k_pack8_prefill_f32_fp16_out,
    "pack8_prefill_fp16_f32_out": gguf_q4_k_pack8_prefill_fp16_f32_out,
    "pack8_prefill_fp16_fp16_out": gguf_q4_k_pack8_prefill_fp16_fp16_out,
    "pack8_prefill_bf16_f32_out": gguf_q4_k_pack8_prefill_bf16_f32_out,
    "pack8_prefill_bf16_fp16_out": gguf_q4_k_pack8_prefill_bf16_fp16_out,
    "pack8_prefill_bf16_bf16_out": gguf_q4_k_pack8_prefill_bf16_bf16_out,
}

register_gguf_q4_k_gemv_kernels()


__all__ = [
    "build_gguf_q4_k_gemv",
    "gguf_q4_k_gemv_bf16_bf16_out",
    "gguf_q4_k_gemv_bf16_f32_out",
    "gguf_q4_k_gemv_bf16_fp16_out",
    "gguf_q4_k_selected_gemv_bf16_bf16_out",
    "gguf_q4_k_selected_dual_gemv_bf16_bf16_out",
    "gguf_q4_k_selected_pack8_gemv_bf16_bf16_out",
    "gguf_q4_k_gemv_f32_f32_out",
    "gguf_q4_k_gemv_f32_fp16_out",
    "gguf_q4_k_gemv_fp16_f32_out",
    "gguf_q4_k_gemv_fp16_fp16_out",
    "gguf_q4_k_pack8_dual_prefill_bf16_bf16_out",
    "gguf_q4_k_pack8_gemv_bf16_bf16_out",
    "gguf_q4_k_pack8_gemv_bf16_f32_out",
    "gguf_q4_k_pack8_gemv_bf16_fp16_out",
    "gguf_q4_k_pack8_gemv_f32_f32_out",
    "gguf_q4_k_pack8_gemv_f32_fp16_out",
    "gguf_q4_k_pack8_gemv_fp16_f32_out",
    "gguf_q4_k_pack8_gemv_fp16_fp16_out",
    "gguf_q4_k_pack8_prefill_bf16_bf16_out",
    "gguf_q4_k_pack8_prefill_bf16_f32_out",
    "gguf_q4_k_pack8_prefill_bf16_fp16_out",
    "gguf_q4_k_pack8_prefill_f32_f32_out",
    "gguf_q4_k_pack8_prefill_f32_fp16_out",
    "gguf_q4_k_pack8_prefill_fp16_f32_out",
    "gguf_q4_k_pack8_prefill_fp16_fp16_out",
    "gguf_q4_k_prefill_bf16_bf16_out",
    "gguf_q4_k_prefill_bf16_f32_out",
    "gguf_q4_k_prefill_bf16_fp16_out",
    "gguf_q4_k_prefill_f32_f32_out",
    "gguf_q4_k_prefill_f32_fp16_out",
    "gguf_q4_k_prefill_fp16_f32_out",
    "gguf_q4_k_prefill_fp16_fp16_out",
    "plan_gguf_q4_k_gemv_build",
    "register_gguf_q4_k_gemv_kernels",
]
