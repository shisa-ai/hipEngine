"""Raw-pointer wrappers for GGUF Q8_0 dense pack8 GEMV decode (single + dual).

P9.B3 grouped/dense decode-shaped pack8 GEMV for raw GGUF Q8_0 weights. Two
launch entry points:

* ``gguf_q8_0_pack8_gemv_decode_*`` -- single output (drop-in replacement for
  the decode-shaped ``gguf_k_pack8_prefill_out_kernel<unsigned short,
  unsigned short, 8>`` family at ``rows == 1`` shapes).
* ``gguf_q8_0_pack8_dual_gate_up_gemv_decode_*`` -- fused dense gate+up, with
  one row-major concatenated output ``[rows, out_features_a + out_features_b]``
  that ``silu_mul_dual_out_*`` consumes directly. The qwen35moe shared-expert
  decode bundle is the primary consumer once task #25 wires it.

Mirrors the AWQ PARO ``gemv_awq_pack8_kernel`` / ``gemv_awq_dual_pack8_kernel``
structure (``__launch_bounds__(128, 4)``, 8-K-per-thread vec_stride loop, 4
wave32 wave-level reduction). The inner k loop swaps AWQ pack8 dequant for
raw GGUF Q8_0 (``d * int8``) and amortises the per-block ``d`` load across
the 8 inner ``j`` lanes (one Q8_0 block is 32 K's, so 8 aligned K's always
share a block).

No new ABI and no resident weight sidecar/repack: raw GGUF Q8_0 bytes stay
on device.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q8_0_pack8_gemv.hip")
_OUTPUT_NAME = "gguf_q8_0_pack8_gemv.so"
_Q8_0_SINGLE_BF16 = "hipengine_gguf_q8_0_pack8_gemv_decode_bf16_bf16_out"
_Q8_0_SINGLE_FP16 = "hipengine_gguf_q8_0_pack8_gemv_decode_fp16_fp16_out"
_Q8_0_DUAL_BF16 = "hipengine_gguf_q8_0_pack8_dual_gate_up_gemv_decode_bf16_bf16_out"
_Q8_0_DUAL_FP16 = "hipengine_gguf_q8_0_pack8_dual_gate_up_gemv_decode_fp16_fp16_out"
_Q8_0_BLOCK = 32


def plan_gguf_q8_0_pack8_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q8_0_pack8_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q8_0_pack8_gemv(
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
        family="gguf_q8_0_pack8_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gguf_q8_0_pack8_gemv_decode_bf16_bf16_out(
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
    """Launch BF16 dense single-output raw-Q8_0 pack8 GEMV decode."""

    _launch_single(
        _Q8_0_SINGLE_BF16,
        x_ptr,
        qweight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q8_0_pack8_gemv_decode_fp16_fp16_out(
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
    """Launch FP16 dense single-output raw-Q8_0 pack8 GEMV decode."""

    _launch_single(
        _Q8_0_SINGLE_FP16,
        x_ptr,
        qweight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q8_0_pack8_dual_gate_up_gemv_decode_bf16_bf16_out(
    x_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 dense fused gate+up raw-Q8_0 pack8 GEMV decode."""

    _launch_dual(
        _Q8_0_DUAL_BF16,
        x_ptr,
        qweight_a_ptr,
        qweight_b_ptr,
        out_ptr,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q8_0_pack8_dual_gate_up_gemv_decode_fp16_fp16_out(
    x_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 dense fused gate+up raw-Q8_0 pack8 GEMV decode."""

    _launch_dual(
        _Q8_0_DUAL_FP16,
        x_ptr,
        qweight_a_ptr,
        qweight_b_ptr,
        out_ptr,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _launch_single(
    symbol: str,
    x_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_common(rows, in_features)
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if out_features % 8 != 0:
        raise ValueError("out_features must be a multiple of 8 (pack8 lane)")
    library = library or build_gguf_q8_0_pack8_gemv(load=True)
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


def _launch_dual(
    symbol: str,
    x_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_common(rows, in_features)
    if out_features_a <= 0 or out_features_b <= 0:
        raise ValueError("out_features_a and out_features_b must be positive")
    if out_features_a % 8 != 0 or out_features_b % 8 != 0:
        raise ValueError("out_features_a and out_features_b must be multiples of 8")
    library = library or build_gguf_q8_0_pack8_gemv(load=True)
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _check_common(rows: int, in_features: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if in_features % _Q8_0_BLOCK != 0:
        raise ValueError("in_features must be divisible by GGUF Q8_0 block size 32")


def register_gguf_q8_0_pack8_gemv_kernels(*, replace: bool = True) -> None:
    """Register P9.B3 dense raw-Q8_0 pack8 GEMV decode kernels."""

    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "pack8_gemv_decode_bf16_bf16_out"),
        gguf_q8_0_pack8_gemv_decode_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "pack8_gemv_decode_fp16_fp16_out"),
        gguf_q8_0_pack8_gemv_decode_fp16_fp16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q8_0",
            "pack8_dual_gate_up_gemv_decode_bf16_bf16_out",
        ),
        gguf_q8_0_pack8_dual_gate_up_gemv_decode_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q8_0",
            "pack8_dual_gate_up_gemv_decode_fp16_fp16_out",
        ),
        gguf_q8_0_pack8_dual_gate_up_gemv_decode_fp16_fp16_out,
        replace=replace,
    )


register_gguf_q8_0_pack8_gemv_kernels()


__all__ = [
    "build_gguf_q8_0_pack8_gemv",
    "gguf_q8_0_pack8_dual_gate_up_gemv_decode_bf16_bf16_out",
    "gguf_q8_0_pack8_dual_gate_up_gemv_decode_fp16_fp16_out",
    "gguf_q8_0_pack8_gemv_decode_bf16_bf16_out",
    "gguf_q8_0_pack8_gemv_decode_fp16_fp16_out",
    "plan_gguf_q8_0_pack8_gemv_build",
    "register_gguf_q8_0_pack8_gemv_kernels",
]
