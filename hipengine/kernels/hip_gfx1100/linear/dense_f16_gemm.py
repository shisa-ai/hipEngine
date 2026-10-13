"""F16-operand WMMA GEMM for small dense F32/BF16 weight matrices.

FP32 activations and FP32 or BF16 weights are rounded to F16 while staged,
multiplied with F16 WMMA and F32 accumulation, and written as unrounded FP32.
It replaces the scalar FP32 GEMV owners for multi-row prefill projections
whose weights are stored uncompressed (the indexer projections, the GDN gate
projections, and the hyper-connection inject), where a row-at-a-time GEMV
leaves the matrix units idle.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("dense_f16_gemm.hip")
_FAMILY = "dense_f16_gemm"
_OUTPUT_NAME = "dense_f16_gemm.so"
_SYMBOLS = {
    "f32": "hipengine_dense_f16_gemm_f32_f32w_f32_out",
    "bf16": "hipengine_dense_f16_gemm_f32_bf16w_f32_out",
}
DENSE_F16_GEMM_VARIANT = "f16_wmma_f32_hidden_f32_out"
DENSE_F16_GEMM_K_MULTIPLE = 32
# The same FP32-weight GEMM as a multi-row router-logits producer: the launch
# ABI (hidden rows, router weight, logits, rows, hidden, experts) is identical.
ROUTER_F16_GEMM_VARIANT = "f32_hidden_f16_wmma_gemm"


def plan_dense_f16_gemm_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family=_FAMILY,
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_dense_f16_gemm(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_hip(
        sources=[_SOURCE],
        family=_FAMILY,
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _make(weight_dtype: str):
    symbol = _SYMBOLS[weight_dtype]

    def gemm(
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
        """out[rows, out_features] (FP32) = x[rows, in_features] @ weight.T."""

        del threads  # launch geometry is fixed; accepted for ABI parity
        if rows <= 0 or out_features <= 0 or in_features <= 0:
            raise ValueError("dense F16 GEMM needs positive sizes")
        if in_features % DENSE_F16_GEMM_K_MULTIPLE:
            raise ValueError("dense F16 GEMM needs in_features % 32 == 0")
        lib = library or build_dense_f16_gemm(load=True)
        active = runtime or get_hip_runtime()
        fn = getattr(lib, symbol)
        fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3 + [ctypes.c_void_p]
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
        if err != HIP_SUCCESS:
            raise RuntimeError(f"{symbol} failed: {active.error_string(err)}")

    gemm.__name__ = f"dense_f16_gemm_f32_{weight_dtype}w_f32_out"
    return gemm


dense_f16_gemm_f32_f32w_f32_out = _make("f32")
dense_f16_gemm_f32_bf16w_f32_out = _make("bf16")


def register_dense_f16_gemm_kernels(*, replace: bool = True) -> None:
    for quant, fn in (
        ("f32", dense_f16_gemm_f32_f32w_f32_out),
        ("bf16", dense_f16_gemm_f32_bf16w_f32_out),
    ):
        register(
            KernelKey("hip_gfx1100", "dense_gemv", quant, DENSE_F16_GEMM_VARIANT),
            fn,
            replace=replace,
        )
    register(
        KernelKey("hip_gfx1100", "router_logits", "f32", ROUTER_F16_GEMM_VARIANT),
        dense_f16_gemm_f32_f32w_f32_out,
        replace=replace,
    )


register_dense_f16_gemm_kernels()

__all__ = [
    "DENSE_F16_GEMM_K_MULTIPLE",
    "DENSE_F16_GEMM_VARIANT",
    "ROUTER_F16_GEMM_VARIANT",
    "build_dense_f16_gemm",
    "dense_f16_gemm_f32_bf16w_f32_out",
    "dense_f16_gemm_f32_f32w_f32_out",
    "plan_dense_f16_gemm_build",
    "register_dense_f16_gemm_kernels",
]
