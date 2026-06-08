"""Raw-pointer wrappers for GGUF Q8_0/Q5_K/Q6_K GEMV kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_k_gemv.hip")
_OUTPUT_NAME = "gguf_k_gemv.so"
_ALLOWED_THREADS = {64, 128, 256}
_QTYPE_BLOCK_SIZE = {"gguf_q8_0": 32, "gguf_q5_k": 256, "gguf_q6_k": 256}


def plan_gguf_k_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_k_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gguf_k_gemv(
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
        family="gguf_k_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _make_wrapper(quant: str, symbol: str):
    def wrapper(*args, **kwargs) -> None:
        _launch(quant, symbol, *args, **kwargs)

    return wrapper


def _make_selected_wrapper(quant: str, symbol: str):
    def wrapper(*args, **kwargs) -> None:
        _launch_selected(quant, symbol, *args, **kwargs)

    return wrapper


def _make_dual_wrapper(quant: str, symbol: str):
    def wrapper(*args, **kwargs) -> None:
        _launch_dual(quant, symbol, *args, **kwargs)

    return wrapper


def _make_pack8_wrapper(quant: str, symbol: str):
    def wrapper(*args, **kwargs) -> None:
        _launch(quant, symbol, *args, require_pack8=True, **kwargs)

    return wrapper


def _make_selected_pack8_wrapper(quant: str, symbol: str):
    def wrapper(*args, **kwargs) -> None:
        _launch_selected(quant, symbol, *args, require_pack8=True, **kwargs)

    return wrapper


def _symbol(quant: str, variant: str) -> str:
    return f"hipengine_{quant}_{variant}"


gguf_q8_0_gemv_f32_f32_out = _make_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "gemv_f32_f32_out"))
gguf_q8_0_gemv_f32_fp16_out = _make_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "gemv_f32_fp16_out"))
gguf_q8_0_gemv_fp16_f32_out = _make_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "gemv_fp16_f32_out"))
gguf_q8_0_gemv_fp16_fp16_out = _make_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "gemv_fp16_fp16_out"))
gguf_q8_0_gemv_bf16_f32_out = _make_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "gemv_bf16_f32_out"))
gguf_q8_0_gemv_bf16_fp16_out = _make_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "gemv_bf16_fp16_out"))
gguf_q8_0_gemv_bf16_bf16_out = _make_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "gemv_bf16_bf16_out"))
gguf_q8_0_dual_gemv_bf16_bf16_out = _make_dual_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "dual_gemv_bf16_bf16_out"))
gguf_q8_0_pack8_gemv_bf16_f32_out = _make_pack8_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "pack8_gemv_bf16_f32_out"))
gguf_q8_0_pack8_gemv_bf16_bf16_out = _make_pack8_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "pack8_gemv_bf16_bf16_out"))
gguf_q8_0_selected_gemv_bf16_bf16_out = _make_selected_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "selected_gemv_bf16_bf16_out"))
gguf_q8_0_selected_pack8_gemv_bf16_bf16_out = _make_selected_pack8_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "selected_pack8_gemv_bf16_bf16_out"))
gguf_q8_0_prefill_f32_f32_out = gguf_q8_0_gemv_f32_f32_out
gguf_q8_0_prefill_f32_fp16_out = gguf_q8_0_gemv_f32_fp16_out
gguf_q8_0_prefill_fp16_f32_out = gguf_q8_0_gemv_fp16_f32_out
gguf_q8_0_prefill_fp16_fp16_out = gguf_q8_0_gemv_fp16_fp16_out
gguf_q8_0_prefill_bf16_f32_out = gguf_q8_0_gemv_bf16_f32_out
gguf_q8_0_prefill_bf16_fp16_out = gguf_q8_0_gemv_bf16_fp16_out
gguf_q8_0_prefill_bf16_bf16_out = gguf_q8_0_gemv_bf16_bf16_out

gguf_q5_k_gemv_f32_f32_out = _make_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "gemv_f32_f32_out"))
gguf_q5_k_gemv_f32_fp16_out = _make_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "gemv_f32_fp16_out"))
gguf_q5_k_gemv_fp16_f32_out = _make_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "gemv_fp16_f32_out"))
gguf_q5_k_gemv_fp16_fp16_out = _make_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "gemv_fp16_fp16_out"))
gguf_q5_k_gemv_bf16_f32_out = _make_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "gemv_bf16_f32_out"))
gguf_q5_k_gemv_bf16_fp16_out = _make_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "gemv_bf16_fp16_out"))
gguf_q5_k_gemv_bf16_bf16_out = _make_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "gemv_bf16_bf16_out"))
gguf_q5_k_pack8_gemv_bf16_f32_out = _make_pack8_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "pack8_gemv_bf16_f32_out"))
gguf_q5_k_pack8_gemv_bf16_bf16_out = _make_pack8_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "pack8_gemv_bf16_bf16_out"))
gguf_q5_k_selected_gemv_bf16_bf16_out = _make_selected_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "selected_gemv_bf16_bf16_out"))
gguf_q5_k_selected_pack8_gemv_bf16_bf16_out = _make_selected_pack8_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "selected_pack8_gemv_bf16_bf16_out"))
gguf_q5_k_prefill_f32_f32_out = gguf_q5_k_gemv_f32_f32_out
gguf_q5_k_prefill_f32_fp16_out = gguf_q5_k_gemv_f32_fp16_out
gguf_q5_k_prefill_fp16_f32_out = gguf_q5_k_gemv_fp16_f32_out
gguf_q5_k_prefill_fp16_fp16_out = gguf_q5_k_gemv_fp16_fp16_out
gguf_q5_k_prefill_bf16_f32_out = gguf_q5_k_gemv_bf16_f32_out
gguf_q5_k_prefill_bf16_fp16_out = gguf_q5_k_gemv_bf16_fp16_out
gguf_q5_k_prefill_bf16_bf16_out = gguf_q5_k_gemv_bf16_bf16_out

gguf_q6_k_gemv_f32_f32_out = _make_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "gemv_f32_f32_out"))
gguf_q6_k_gemv_f32_fp16_out = _make_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "gemv_f32_fp16_out"))
gguf_q6_k_gemv_fp16_f32_out = _make_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "gemv_fp16_f32_out"))
gguf_q6_k_gemv_fp16_fp16_out = _make_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "gemv_fp16_fp16_out"))
gguf_q6_k_gemv_bf16_f32_out = _make_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "gemv_bf16_f32_out"))
gguf_q6_k_gemv_bf16_fp16_out = _make_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "gemv_bf16_fp16_out"))
gguf_q6_k_gemv_bf16_bf16_out = _make_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "gemv_bf16_bf16_out"))
gguf_q6_k_pack8_gemv_bf16_f32_out = _make_pack8_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "pack8_gemv_bf16_f32_out"))
gguf_q6_k_pack8_gemv_bf16_bf16_out = _make_pack8_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "pack8_gemv_bf16_bf16_out"))
gguf_q6_k_selected_gemv_bf16_bf16_out = _make_selected_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "selected_gemv_bf16_bf16_out"))
gguf_q6_k_selected_pack8_gemv_bf16_bf16_out = _make_selected_pack8_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "selected_pack8_gemv_bf16_bf16_out"))
gguf_q6_k_prefill_f32_f32_out = gguf_q6_k_gemv_f32_f32_out
gguf_q6_k_prefill_f32_fp16_out = gguf_q6_k_gemv_f32_fp16_out
gguf_q6_k_prefill_fp16_f32_out = gguf_q6_k_gemv_fp16_f32_out
gguf_q6_k_prefill_fp16_fp16_out = gguf_q6_k_gemv_fp16_fp16_out
gguf_q6_k_prefill_bf16_f32_out = gguf_q6_k_gemv_bf16_f32_out
gguf_q6_k_prefill_bf16_fp16_out = gguf_q6_k_gemv_bf16_fp16_out
gguf_q6_k_prefill_bf16_bf16_out = gguf_q6_k_gemv_bf16_bf16_out


def register_gguf_k_gemv_kernels(*, replace: bool = True) -> None:
    for quant in ("gguf_q8_0", "gguf_q5_k", "gguf_q6_k"):
        for variant, fn in _WRAPPERS[quant].items():
            register(KernelKey("hip_gfx1100", "linear", quant, variant), fn, replace=replace)


def _launch(
    quant: str,
    symbol: str,
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
    require_pack8: bool = False,
) -> None:
    _validate(quant, rows, in_features, out_features, threads, require_pack8=require_pack8)
    library = library or build_gguf_k_gemv(load=True)
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


def _launch_dual(
    quant: str,
    symbol: str,
    x_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate(quant, rows, in_features, out_features, threads)
    library = library or build_gguf_k_gemv(load=True)
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
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_a_ptr),
        ctypes.c_void_p(out_b_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_selected(
    quant: str,
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
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    require_pack8: bool = False,
) -> None:
    if x_rows <= 0:
        raise ValueError("x_rows must be positive")
    if rows <= 0 or rows % x_rows != 0:
        raise ValueError("rows must be positive and divisible by x_rows")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    _validate(quant, rows, in_features, out_features, threads, require_pack8=require_pack8)
    library = library or build_gguf_k_gemv(load=True)
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


def _validate(
    quant: str,
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
    block_size = _QTYPE_BLOCK_SIZE[quant]
    if in_features % block_size != 0:
        raise ValueError(f"in_features must be divisible by GGUF {quant} block size {block_size}")
    if require_pack8 and out_features % 8 != 0:
        raise ValueError("out_features must be divisible by 8 for GGUF K pack8")
    if threads not in _ALLOWED_THREADS:
        allowed = ", ".join(str(value) for value in sorted(_ALLOWED_THREADS))
        raise ValueError(f"threads must be one of {allowed}")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


_WRAPPERS = {
    "gguf_q8_0": {
        "gemv_f32_f32_out": gguf_q8_0_gemv_f32_f32_out,
        "gemv_f32_fp16_out": gguf_q8_0_gemv_f32_fp16_out,
        "gemv_fp16_f32_out": gguf_q8_0_gemv_fp16_f32_out,
        "gemv_fp16_fp16_out": gguf_q8_0_gemv_fp16_fp16_out,
        "gemv_bf16_f32_out": gguf_q8_0_gemv_bf16_f32_out,
        "gemv_bf16_fp16_out": gguf_q8_0_gemv_bf16_fp16_out,
        "gemv_bf16_bf16_out": gguf_q8_0_gemv_bf16_bf16_out,
        "dual_gemv_bf16_bf16_out": gguf_q8_0_dual_gemv_bf16_bf16_out,
        "pack8_gemv_bf16_f32_out": gguf_q8_0_pack8_gemv_bf16_f32_out,
        "pack8_gemv_bf16_bf16_out": gguf_q8_0_pack8_gemv_bf16_bf16_out,
        "selected_gemv_bf16_bf16_out": gguf_q8_0_selected_gemv_bf16_bf16_out,
        "selected_pack8_gemv_bf16_bf16_out": gguf_q8_0_selected_pack8_gemv_bf16_bf16_out,
        "prefill_f32_f32_out": gguf_q8_0_prefill_f32_f32_out,
        "prefill_f32_fp16_out": gguf_q8_0_prefill_f32_fp16_out,
        "prefill_fp16_f32_out": gguf_q8_0_prefill_fp16_f32_out,
        "prefill_fp16_fp16_out": gguf_q8_0_prefill_fp16_fp16_out,
        "prefill_bf16_f32_out": gguf_q8_0_prefill_bf16_f32_out,
        "prefill_bf16_fp16_out": gguf_q8_0_prefill_bf16_fp16_out,
        "prefill_bf16_bf16_out": gguf_q8_0_prefill_bf16_bf16_out,
    },
    "gguf_q5_k": {
        "gemv_f32_f32_out": gguf_q5_k_gemv_f32_f32_out,
        "gemv_f32_fp16_out": gguf_q5_k_gemv_f32_fp16_out,
        "gemv_fp16_f32_out": gguf_q5_k_gemv_fp16_f32_out,
        "gemv_fp16_fp16_out": gguf_q5_k_gemv_fp16_fp16_out,
        "gemv_bf16_f32_out": gguf_q5_k_gemv_bf16_f32_out,
        "gemv_bf16_fp16_out": gguf_q5_k_gemv_bf16_fp16_out,
        "gemv_bf16_bf16_out": gguf_q5_k_gemv_bf16_bf16_out,
        "pack8_gemv_bf16_f32_out": gguf_q5_k_pack8_gemv_bf16_f32_out,
        "pack8_gemv_bf16_bf16_out": gguf_q5_k_pack8_gemv_bf16_bf16_out,
        "selected_gemv_bf16_bf16_out": gguf_q5_k_selected_gemv_bf16_bf16_out,
        "selected_pack8_gemv_bf16_bf16_out": gguf_q5_k_selected_pack8_gemv_bf16_bf16_out,
        "prefill_f32_f32_out": gguf_q5_k_prefill_f32_f32_out,
        "prefill_f32_fp16_out": gguf_q5_k_prefill_f32_fp16_out,
        "prefill_fp16_f32_out": gguf_q5_k_prefill_fp16_f32_out,
        "prefill_fp16_fp16_out": gguf_q5_k_prefill_fp16_fp16_out,
        "prefill_bf16_f32_out": gguf_q5_k_prefill_bf16_f32_out,
        "prefill_bf16_fp16_out": gguf_q5_k_prefill_bf16_fp16_out,
        "prefill_bf16_bf16_out": gguf_q5_k_prefill_bf16_bf16_out,
    },
    "gguf_q6_k": {
        "gemv_f32_f32_out": gguf_q6_k_gemv_f32_f32_out,
        "gemv_f32_fp16_out": gguf_q6_k_gemv_f32_fp16_out,
        "gemv_fp16_f32_out": gguf_q6_k_gemv_fp16_f32_out,
        "gemv_fp16_fp16_out": gguf_q6_k_gemv_fp16_fp16_out,
        "gemv_bf16_f32_out": gguf_q6_k_gemv_bf16_f32_out,
        "gemv_bf16_fp16_out": gguf_q6_k_gemv_bf16_fp16_out,
        "gemv_bf16_bf16_out": gguf_q6_k_gemv_bf16_bf16_out,
        "pack8_gemv_bf16_f32_out": gguf_q6_k_pack8_gemv_bf16_f32_out,
        "pack8_gemv_bf16_bf16_out": gguf_q6_k_pack8_gemv_bf16_bf16_out,
        "selected_gemv_bf16_bf16_out": gguf_q6_k_selected_gemv_bf16_bf16_out,
        "selected_pack8_gemv_bf16_bf16_out": gguf_q6_k_selected_pack8_gemv_bf16_bf16_out,
        "prefill_f32_f32_out": gguf_q6_k_prefill_f32_f32_out,
        "prefill_f32_fp16_out": gguf_q6_k_prefill_f32_fp16_out,
        "prefill_fp16_f32_out": gguf_q6_k_prefill_fp16_f32_out,
        "prefill_fp16_fp16_out": gguf_q6_k_prefill_fp16_fp16_out,
        "prefill_bf16_f32_out": gguf_q6_k_prefill_bf16_f32_out,
        "prefill_bf16_fp16_out": gguf_q6_k_prefill_bf16_fp16_out,
        "prefill_bf16_bf16_out": gguf_q6_k_prefill_bf16_bf16_out,
    },
}

register_gguf_k_gemv_kernels()


__all__ = [
    "build_gguf_k_gemv",
    "gguf_q5_k_gemv_f32_f32_out",
    "gguf_q5_k_gemv_f32_fp16_out",
    "gguf_q5_k_gemv_fp16_f32_out",
    "gguf_q5_k_gemv_fp16_fp16_out",
    "gguf_q5_k_gemv_bf16_f32_out",
    "gguf_q5_k_gemv_bf16_fp16_out",
    "gguf_q5_k_gemv_bf16_bf16_out",
    "gguf_q5_k_selected_gemv_bf16_bf16_out",
    "gguf_q5_k_prefill_f32_f32_out",
    "gguf_q5_k_prefill_f32_fp16_out",
    "gguf_q5_k_prefill_fp16_f32_out",
    "gguf_q5_k_prefill_fp16_fp16_out",
    "gguf_q5_k_prefill_bf16_f32_out",
    "gguf_q5_k_prefill_bf16_fp16_out",
    "gguf_q5_k_prefill_bf16_bf16_out",
    "gguf_q6_k_gemv_f32_f32_out",
    "gguf_q6_k_gemv_f32_fp16_out",
    "gguf_q6_k_gemv_fp16_f32_out",
    "gguf_q6_k_gemv_fp16_fp16_out",
    "gguf_q6_k_gemv_bf16_f32_out",
    "gguf_q6_k_gemv_bf16_fp16_out",
    "gguf_q6_k_gemv_bf16_bf16_out",
    "gguf_q6_k_selected_gemv_bf16_bf16_out",
    "gguf_q6_k_prefill_f32_f32_out",
    "gguf_q6_k_prefill_f32_fp16_out",
    "gguf_q6_k_prefill_fp16_f32_out",
    "gguf_q6_k_prefill_fp16_fp16_out",
    "gguf_q6_k_prefill_bf16_f32_out",
    "gguf_q6_k_prefill_bf16_fp16_out",
    "gguf_q6_k_prefill_bf16_bf16_out",
    "gguf_q8_0_gemv_f32_f32_out",
    "gguf_q8_0_gemv_f32_fp16_out",
    "gguf_q8_0_gemv_fp16_f32_out",
    "gguf_q8_0_gemv_fp16_fp16_out",
    "gguf_q8_0_gemv_bf16_f32_out",
    "gguf_q8_0_gemv_bf16_fp16_out",
    "gguf_q8_0_gemv_bf16_bf16_out",
    "gguf_q8_0_dual_gemv_bf16_bf16_out",
    "gguf_q8_0_prefill_f32_f32_out",
    "gguf_q8_0_prefill_f32_fp16_out",
    "gguf_q8_0_prefill_fp16_f32_out",
    "gguf_q8_0_prefill_fp16_fp16_out",
    "gguf_q8_0_prefill_bf16_f32_out",
    "gguf_q8_0_prefill_bf16_fp16_out",
    "gguf_q8_0_prefill_bf16_bf16_out",
    "plan_gguf_k_gemv_build",
    "register_gguf_k_gemv_kernels",
]
