"""Raw-pointer wrappers for GGUF Q4_K dense pack8 GEMV decode.

P9.B4 single-output decode-shaped pack8 GEMV for raw GGUF Q4_K weights,
covering the qwen35moe dense surfaces that materialize Q4_K as pack8 (the
attention Q/K/V/O projections, plus the lm-head logits projection when the
tied output weight is stored as Q4_K).

Mirrors the AWQ PARO ``gemv_awq_pack8_kernel`` structure
(``__launch_bounds__(128, 4)``, 4 wave32 wave-level reduction, per-block
scale/min hoist into shared memory) but with the inner k loop swapped for
raw GGUF Q4_K block dequant.

Four launch entry points are registered:

* ``pack8_gemv_decode_bf16_bf16_out`` and ``pack8_gemv_decode_fp16_fp16_out``
  for the attention QKV/O surfaces (BF16-only hidden in qwen35moe today;
  FP16 included for completeness with the other GEMV decode families).
* ``pack8_gemv_decode_bf16_f32_out`` and ``pack8_gemv_decode_fp16_f32_out``
  for the lm-head logits projection (F32 output feeds straight into the
  sampler).

No new ABI and no resident weight sidecar/repack: raw GGUF Q4_K bytes stay
on device.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q4_k_pack8_gemv.hip")
_OUTPUT_NAME = "gguf_q4_k_pack8_gemv.so"
_SYM_BF16_BF16 = "hipengine_gguf_q4_k_pack8_gemv_decode_bf16_bf16_out"
_SYM_FP16_FP16 = "hipengine_gguf_q4_k_pack8_gemv_decode_fp16_fp16_out"
_SYM_BF16_F32 = "hipengine_gguf_q4_k_pack8_gemv_decode_bf16_f32_out"
_SYM_FP16_F32 = "hipengine_gguf_q4_k_pack8_gemv_decode_fp16_f32_out"
_Q4_K_BLOCK = 256


def plan_gguf_q4_k_pack8_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q4_k_pack8_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q4_k_pack8_gemv(
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
        family="gguf_q4_k_pack8_gemv",
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
        library = library or build_gguf_q4_k_pack8_gemv(load=True)
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


gguf_q4_k_pack8_gemv_decode_bf16_bf16_out = _make_launch(_SYM_BF16_BF16)
gguf_q4_k_pack8_gemv_decode_fp16_fp16_out = _make_launch(_SYM_FP16_FP16)
gguf_q4_k_pack8_gemv_decode_bf16_f32_out = _make_launch(_SYM_BF16_F32)
gguf_q4_k_pack8_gemv_decode_fp16_f32_out = _make_launch(_SYM_FP16_F32)


def _check_common(rows: int, in_features: int, out_features: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if in_features % _Q4_K_BLOCK != 0:
        raise ValueError("in_features must be divisible by GGUF Q4_K block size 256")
    if out_features % 8 != 0:
        raise ValueError("out_features must be a multiple of 8 (pack8 lane)")


def register_gguf_q4_k_pack8_gemv_kernels(*, replace: bool = True) -> None:
    """Register P9.B4 dense raw-Q4_K pack8 GEMV decode kernels."""

    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "pack8_gemv_decode_bf16_bf16_out"),
        gguf_q4_k_pack8_gemv_decode_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "pack8_gemv_decode_fp16_fp16_out"),
        gguf_q4_k_pack8_gemv_decode_fp16_fp16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "pack8_gemv_decode_bf16_f32_out"),
        gguf_q4_k_pack8_gemv_decode_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "pack8_gemv_decode_fp16_f32_out"),
        gguf_q4_k_pack8_gemv_decode_fp16_f32_out,
        replace=replace,
    )


register_gguf_q4_k_pack8_gemv_kernels()


__all__ = [
    "build_gguf_q4_k_pack8_gemv",
    "gguf_q4_k_pack8_gemv_decode_bf16_bf16_out",
    "gguf_q4_k_pack8_gemv_decode_bf16_f32_out",
    "gguf_q4_k_pack8_gemv_decode_fp16_f32_out",
    "gguf_q4_k_pack8_gemv_decode_fp16_fp16_out",
    "plan_gguf_q4_k_pack8_gemv_build",
    "register_gguf_q4_k_pack8_gemv_kernels",
]
