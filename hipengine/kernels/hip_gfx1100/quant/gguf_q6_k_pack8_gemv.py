"""Raw-pointer wrappers for GGUF Q6_K dense pack8 GEMV decode.

P9.B4b: dense single-output decode-shaped pack8 GEMV for raw GGUF Q6_K
weights. The qwen35moe Qwen3.6-35B-A3B-UD-Q4_K_M lm-head ties to a Q6_K
output projection, so the F32 output variant is the production path the
sampler reads logits from. Mirrors the Q4_K dense kernel (P9.B4) with the
inner k loop swapped for raw GGUF Q6_K block dequant (`int8` per-16-K
scales x `fp16` super-scale, with 2 high bits per element).

Four launch entry points are registered: BF16/BF16, FP16/FP16, BF16/F32,
FP16/F32. No new ABI and no resident weight sidecar/repack.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q6_k_pack8_gemv.hip")
_OUTPUT_NAME = "gguf_q6_k_pack8_gemv.so"
_SYM_BF16_BF16 = "hipengine_gguf_q6_k_pack8_gemv_decode_bf16_bf16_out"
_SYM_FP16_FP16 = "hipengine_gguf_q6_k_pack8_gemv_decode_fp16_fp16_out"
_SYM_BF16_F32 = "hipengine_gguf_q6_k_pack8_gemv_decode_bf16_f32_out"
_SYM_FP16_F32 = "hipengine_gguf_q6_k_pack8_gemv_decode_fp16_f32_out"
_Q6_K_BLOCK = 256


def plan_gguf_q6_k_pack8_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q6_k_pack8_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q6_k_pack8_gemv(
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
        family="gguf_q6_k_pack8_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _make_launch(symbol: str):
    def launch(
        x_ptr: int,
        qweight_ptr: int,
        out_ptr: int,
        rows: int,
        in_features: int,
        out_features: int,
        *,
        stream: int = 0,
        library: ctypes.CDLL | None = None,
        runtime: HipRuntime | None = None,
    ) -> None:
        _check_common(rows, in_features, out_features)
        library = library or build_gguf_q6_k_pack8_gemv(load=True)
        runtime = runtime or get_hip_runtime()
        fn = getattr(library, symbol)
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
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
            ctypes.c_void_p(stream),
        )
        if int(err) != HIP_SUCCESS:
            runtime.check(int(err))

    launch.__name__ = symbol[len("hipengine_") :]
    return launch


gguf_q6_k_pack8_gemv_decode_bf16_bf16_out = _make_launch(_SYM_BF16_BF16)
gguf_q6_k_pack8_gemv_decode_fp16_fp16_out = _make_launch(_SYM_FP16_FP16)
gguf_q6_k_pack8_gemv_decode_bf16_f32_out = _make_launch(_SYM_BF16_F32)
gguf_q6_k_pack8_gemv_decode_fp16_f32_out = _make_launch(_SYM_FP16_F32)


def _check_common(rows: int, in_features: int, out_features: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if in_features % _Q6_K_BLOCK != 0:
        raise ValueError("in_features must be divisible by GGUF Q6_K block size 256")
    if out_features % 8 != 0:
        raise ValueError("out_features must be a multiple of 8 (pack8 lane)")


def register_gguf_q6_k_pack8_gemv_kernels(*, replace: bool = True) -> None:
    """Register P9.B4b dense raw-Q6_K pack8 GEMV decode kernels."""

    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q6_k", "pack8_gemv_decode_bf16_bf16_out"),
        gguf_q6_k_pack8_gemv_decode_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q6_k", "pack8_gemv_decode_fp16_fp16_out"),
        gguf_q6_k_pack8_gemv_decode_fp16_fp16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q6_k", "pack8_gemv_decode_bf16_f32_out"),
        gguf_q6_k_pack8_gemv_decode_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q6_k", "pack8_gemv_decode_fp16_f32_out"),
        gguf_q6_k_pack8_gemv_decode_fp16_f32_out,
        replace=replace,
    )


register_gguf_q6_k_pack8_gemv_kernels()


__all__ = [
    "build_gguf_q6_k_pack8_gemv",
    "gguf_q6_k_pack8_gemv_decode_bf16_bf16_out",
    "gguf_q6_k_pack8_gemv_decode_bf16_f32_out",
    "gguf_q6_k_pack8_gemv_decode_fp16_f32_out",
    "gguf_q6_k_pack8_gemv_decode_fp16_fp16_out",
    "plan_gguf_q6_k_pack8_gemv_build",
    "register_gguf_q6_k_pack8_gemv_kernels",
]
