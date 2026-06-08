"""Wrappers for qwen35moe GGUF expert pack8 selected GEMV kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_expert_pack8_gemv.hip")
_OUTPUT_NAME = "gguf_expert_pack8_gemv.so"
_SYMBOLS = {
    "gguf_q4_k": "hipengine_gguf_q4_k_expert_pack8_selected_bf16_bf16_out",
    "gguf_q5_k": "hipengine_gguf_q5_k_expert_pack8_selected_bf16_bf16_out",
    "gguf_q6_k": "hipengine_gguf_q6_k_expert_pack8_selected_bf16_bf16_out",
}
_SYMBOL_Q4_DUAL = "hipengine_gguf_q4_k_expert_pack8_dual_selected_bf16_bf16_out"
_ALLOWED_THREADS = {32, 64, 128}
_QTYPE_GROUP_SIZE = {"gguf_q4_k": 32, "gguf_q5_k": 32, "gguf_q6_k": 16}


def plan_gguf_expert_pack8_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_expert_pack8_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gguf_expert_pack8_gemv(
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
        family="gguf_expert_pack8_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gguf_q4_k_expert_pack8_selected_bf16_bf16_out(*args, **kwargs) -> None:
    _launch_selected("gguf_q4_k", _SYMBOLS["gguf_q4_k"], *args, **kwargs)


def gguf_q5_k_expert_pack8_selected_bf16_bf16_out(*args, **kwargs) -> None:
    _launch_selected("gguf_q5_k", _SYMBOLS["gguf_q5_k"], *args, **kwargs)


def gguf_q6_k_expert_pack8_selected_bf16_bf16_out(*args, **kwargs) -> None:
    _launch_selected("gguf_q6_k", _SYMBOLS["gguf_q6_k"], *args, **kwargs)


def gguf_q4_k_expert_pack8_dual_selected_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    qweight_a_ptr: int,
    scales_a_ptr: int,
    mins_a_ptr: int,
    qweight_b_ptr: int,
    scales_b_ptr: int,
    mins_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    launch_threads = _select_threads(threads)
    _validate("gguf_q4_k", x_rows, rows, num_experts, in_features, out_features, launch_threads)
    library = library or build_gguf_expert_pack8_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_Q4_DUAL)
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
        ctypes.c_void_p(scales_a_ptr),
        ctypes.c_void_p(mins_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(scales_b_ptr),
        ctypes.c_void_p(mins_b_ptr),
        ctypes.c_void_p(out_a_ptr),
        ctypes.c_void_p(out_b_ptr),
        ctypes.c_int64(x_rows),
        ctypes.c_int64(rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(launch_threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def register_gguf_expert_pack8_gemv_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "moe_linear", "gguf_q4_k", "expert_pack8_selected_bf16_bf16_out"),
        gguf_q4_k_expert_pack8_selected_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "moe_linear", "gguf_q5_k", "expert_pack8_selected_bf16_bf16_out"),
        gguf_q5_k_expert_pack8_selected_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "moe_linear", "gguf_q6_k", "expert_pack8_selected_bf16_bf16_out"),
        gguf_q6_k_expert_pack8_selected_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "moe_linear", "gguf_q4_k", "expert_pack8_dual_selected_bf16_bf16_out"),
        gguf_q4_k_expert_pack8_dual_selected_bf16_bf16_out,
        replace=replace,
    )


def _launch_selected(
    quant: str,
    symbol: str,
    x_ptr: int,
    selected_ptr: int,
    qweight_low_ptr: int,
    qweight_high_ptr: int,
    scales_ptr: int,
    mins_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    launch_threads = _select_threads(threads)
    _validate(quant, x_rows, rows, num_experts, in_features, out_features, launch_threads)
    library = library or build_gguf_expert_pack8_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(qweight_low_ptr),
        ctypes.c_void_p(qweight_high_ptr),
        ctypes.c_void_p(scales_ptr),
        ctypes.c_void_p(mins_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(x_rows),
        ctypes.c_int64(rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(launch_threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _select_threads(threads: int) -> int:
    return 128 if threads == 0 else int(threads)


def _validate(
    quant: str,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    threads: int,
) -> None:
    if quant not in _QTYPE_GROUP_SIZE:
        raise ValueError(f"unsupported GGUF expert pack8 quant {quant!r}")
    if x_rows <= 0:
        raise ValueError("x_rows must be positive")
    if rows <= 0 or rows % x_rows != 0:
        raise ValueError("rows must be positive and divisible by x_rows")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if in_features <= 0 or in_features % _QTYPE_GROUP_SIZE[quant] != 0:
        raise ValueError(f"in_features must be positive and divisible by {_QTYPE_GROUP_SIZE[quant]}")
    if out_features <= 0 or out_features % 8 != 0:
        raise ValueError("out_features must be positive and divisible by 8")
    if threads not in _ALLOWED_THREADS:
        allowed = ", ".join(str(value) for value in sorted(_ALLOWED_THREADS))
        raise ValueError(f"threads must be one of {allowed}")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_gguf_expert_pack8_gemv_kernels()


__all__ = [
    "build_gguf_expert_pack8_gemv",
    "gguf_q4_k_expert_pack8_dual_selected_bf16_bf16_out",
    "gguf_q4_k_expert_pack8_selected_bf16_bf16_out",
    "gguf_q5_k_expert_pack8_selected_bf16_bf16_out",
    "gguf_q6_k_expert_pack8_selected_bf16_bf16_out",
    "plan_gguf_expert_pack8_gemv_build",
    "register_gguf_expert_pack8_gemv_kernels",
]
