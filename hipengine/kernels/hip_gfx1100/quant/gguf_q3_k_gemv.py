"""Raw-pointer selected GGUF Q3_K MoE GEMV kernels for NextN experts."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q3_k_gemv.hip")
_OUTPUT_NAME = "gguf_q3_k_gemv.so"
_QK_K = 256
_ALLOWED_THREADS = {64, 128, 256}
_SINGLE = "hipengine_gguf_q3_k_selected_gemv_bf16_bf16_out"
_DUAL_SILU = "hipengine_gguf_q3_k_selected_dual_silu_gemv_bf16_bf16_out"


def plan_gguf_q3_k_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q3_k_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q3_k_gemv(
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
        family="gguf_q3_k_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gguf_q3_k_selected_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate_selected(
        x_rows=x_rows,
        rows=rows,
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        threads=threads,
    )
    library = library or build_gguf_q3_k_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SINGLE)
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


def gguf_q3_k_selected_dual_silu_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    *,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate_selected(
        x_rows=x_rows,
        rows=rows,
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        threads=threads,
    )
    library = library or build_gguf_q3_k_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _DUAL_SILU)
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(gate_weight_ptr),
        ctypes.c_void_p(up_weight_ptr),
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


def _validate_selected(
    *,
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
    if in_features <= 0 or in_features % _QK_K != 0:
        raise ValueError("in_features must be positive and divisible by 256")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if threads not in _ALLOWED_THREADS:
        allowed = ", ".join(str(value) for value in sorted(_ALLOWED_THREADS))
        raise ValueError(f"threads must be one of {allowed}")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_gguf_q3_k_gemv_kernels(*, replace: bool = True) -> None:
    for variant, fn in (
        ("selected_gemv_decode_bf16_bf16_out", gguf_q3_k_selected_gemv_bf16_bf16_out),
        (
            "selected_dual_silu_gemv_decode_bf16_bf16_out",
            gguf_q3_k_selected_dual_silu_gemv_bf16_bf16_out,
        ),
    ):
        register(
            KernelKey("hip_gfx1100", "moe_linear", "gguf_q3_k", variant),
            fn,
            replace=replace,
        )


register_gguf_q3_k_gemv_kernels()


__all__ = [
    "build_gguf_q3_k_gemv",
    "gguf_q3_k_selected_dual_silu_gemv_bf16_bf16_out",
    "gguf_q3_k_selected_gemv_bf16_bf16_out",
    "plan_gguf_q3_k_gemv_build",
    "register_gguf_q3_k_gemv_kernels",
]
