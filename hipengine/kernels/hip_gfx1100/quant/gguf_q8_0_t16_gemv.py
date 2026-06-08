"""Raw-pointer wrappers for GGUF Q8_0 T16 GEMV decode kernels.

P9.H3 Q8T16 implementation slice.  These wrappers consume the replacement
layout produced by :func:`hipengine.quant.gguf_t16.repack_gguf_q8_0_tile16`:
``tiles[out_tiles16, blocks_per_row, 544]``.  The dense single-output kernel is
for ordinary Q8_0 projections; the dual variant emits concatenated gate/up
outputs for shared-expert decode once runtime dispatch is wired.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q8_0_t16_gemv.hip")
_OUTPUT_NAME = "gguf_q8_0_t16_gemv.so"
_Q8_0_SINGLE_BF16 = "hipengine_gguf_q8_0_t16_gemv_decode_bf16_bf16_out"
_Q8_0_SINGLE_FP16 = "hipengine_gguf_q8_0_t16_gemv_decode_fp16_fp16_out"
_Q8_0_SINGLE_F32_BF16 = "hipengine_gguf_q8_0_t16_gemv_decode_f32_bf16_out"
_Q8_0_DUAL_BF16 = "hipengine_gguf_q8_0_t16_dual_gate_up_gemv_decode_bf16_bf16_out"
_Q8_0_DUAL_FP16 = "hipengine_gguf_q8_0_t16_dual_gate_up_gemv_decode_fp16_fp16_out"
_Q8_0_DUAL_SPLIT_BF16 = "hipengine_gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out"
_Q8_0_DUAL_SPLIT_FP16 = "hipengine_gguf_q8_0_t16_dual_gemv_decode_fp16_fp16_out"
_Q8_0_TRIPLE_SPLIT_BF16 = "hipengine_gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out"
_Q8_0_TRIPLE_SPLIT_FP16 = "hipengine_gguf_q8_0_t16_triple_gemv_decode_fp16_fp16_out"
_Q8_0_BLOCK = 32
_T16_COLS = 16


def plan_gguf_q8_0_t16_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q8_0_t16_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q8_0_t16_gemv(
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
        family="gguf_q8_0_t16_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gguf_q8_0_t16_gemv_decode_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
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
    """Launch BF16 dense single-output Q8T16 GEMV decode."""

    del threads

    _launch_single(
        _Q8_0_SINGLE_BF16,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q8_0_t16_gemv_decode_fp16_fp16_out(
    x_ptr: int,
    tiles_ptr: int,
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
    """Launch FP16 dense single-output Q8T16 GEMV decode."""

    del threads

    _launch_single(
        _Q8_0_SINGLE_FP16,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q8_0_t16_gemv_decode_f32_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
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
    """Launch F32-input, BF16-output dense single Q8T16 GEMV decode."""

    del threads

    _launch_single(
        _Q8_0_SINGLE_F32_BF16,
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q8_0_t16_dual_gate_up_gemv_decode_bf16_bf16_out(
    x_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    *,
    threads: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 dense fused gate+up Q8T16 GEMV decode."""

    del threads

    _launch_dual(
        _Q8_0_DUAL_BF16,
        x_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_ptr,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q8_0_t16_dual_gate_up_gemv_decode_fp16_fp16_out(
    x_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    *,
    threads: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 dense fused gate+up Q8T16 GEMV decode."""

    del threads

    _launch_dual(
        _Q8_0_DUAL_FP16,
        x_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_ptr,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out(
    x_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    *,
    threads: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 dense dual Q8T16 GEMV decode into separate outputs."""

    del threads

    _launch_dual_split(
        _Q8_0_DUAL_SPLIT_BF16,
        x_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_a_ptr,
        out_b_ptr,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q8_0_t16_dual_gemv_decode_fp16_fp16_out(
    x_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    *,
    threads: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 dense dual Q8T16 GEMV decode into separate outputs."""

    del threads

    _launch_dual_split(
        _Q8_0_DUAL_SPLIT_FP16,
        x_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        out_a_ptr,
        out_b_ptr,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out(
    x_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    tiles_c_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    out_c_ptr: int,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    out_features_c: int,
    *,
    threads: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 dense triple Q8T16 GEMV decode into separate outputs."""

    del threads

    _launch_triple_split(
        _Q8_0_TRIPLE_SPLIT_BF16,
        x_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        tiles_c_ptr,
        out_a_ptr,
        out_b_ptr,
        out_c_ptr,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        out_features_c,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q8_0_t16_triple_gemv_decode_fp16_fp16_out(
    x_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    tiles_c_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    out_c_ptr: int,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    out_features_c: int,
    *,
    threads: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 dense triple Q8T16 GEMV decode into separate outputs."""

    del threads

    _launch_triple_split(
        _Q8_0_TRIPLE_SPLIT_FP16,
        x_ptr,
        tiles_a_ptr,
        tiles_b_ptr,
        tiles_c_ptr,
        out_a_ptr,
        out_b_ptr,
        out_c_ptr,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        out_features_c,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _launch_single(
    symbol: str,
    x_ptr: int,
    tiles_ptr: int,
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
    if out_features % _T16_COLS != 0:
        raise ValueError("out_features must be a multiple of 16 (T16 tile)")
    library = library or build_gguf_q8_0_t16_gemv(load=True)
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
        ctypes.c_void_p(tiles_ptr),
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
    tiles_a_ptr: int,
    tiles_b_ptr: int,
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
        raise ValueError("out_features_a/out_features_b must be positive")
    if out_features_a % _T16_COLS != 0 or out_features_b % _T16_COLS != 0:
        raise ValueError("out_features_a/out_features_b must be multiples of 16 (T16 tile)")
    library = library or build_gguf_q8_0_t16_gemv(load=True)
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
        ctypes.c_void_p(tiles_a_ptr),
        ctypes.c_void_p(tiles_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_dual_split(
    symbol: str,
    x_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
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
        raise ValueError("out_features_a/out_features_b must be positive")
    if out_features_a % _T16_COLS != 0 or out_features_b % _T16_COLS != 0:
        raise ValueError("out_features_a/out_features_b must be multiples of 16 (T16 tile)")
    library = library or build_gguf_q8_0_t16_gemv(load=True)
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
        ctypes.c_void_p(tiles_a_ptr),
        ctypes.c_void_p(tiles_b_ptr),
        ctypes.c_void_p(out_a_ptr),
        ctypes.c_void_p(out_b_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_triple_split(
    symbol: str,
    x_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    tiles_c_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    out_c_ptr: int,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    out_features_c: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_common(rows, in_features)
    if out_features_a <= 0 or out_features_b <= 0 or out_features_c <= 0:
        raise ValueError("out_features_a/out_features_b/out_features_c must be positive")
    if out_features_a % _T16_COLS != 0 or out_features_b % _T16_COLS != 0 or out_features_c % _T16_COLS != 0:
        raise ValueError("out_features_a/out_features_b/out_features_c must be multiples of 16 (T16 tile)")
    library = library or build_gguf_q8_0_t16_gemv(load=True)
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(tiles_a_ptr),
        ctypes.c_void_p(tiles_b_ptr),
        ctypes.c_void_p(tiles_c_ptr),
        ctypes.c_void_p(out_a_ptr),
        ctypes.c_void_p(out_b_ptr),
        ctypes.c_void_p(out_c_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(out_features_c),
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


def register_gguf_q8_0_t16_gemv_kernels(*, replace: bool = True) -> None:
    """Register P9.H3 Q8T16 GEMV decode kernels."""

    for variant, fn in (
        ("t16_gemv_decode_bf16_bf16_out", gguf_q8_0_t16_gemv_decode_bf16_bf16_out),
        ("t16_gemv_decode_fp16_fp16_out", gguf_q8_0_t16_gemv_decode_fp16_fp16_out),
        ("t16_gemv_decode_f32_bf16_out", gguf_q8_0_t16_gemv_decode_f32_bf16_out),
        ("t16_dual_gate_up_gemv_decode_bf16_bf16_out", gguf_q8_0_t16_dual_gate_up_gemv_decode_bf16_bf16_out),
        ("t16_dual_gate_up_gemv_decode_fp16_fp16_out", gguf_q8_0_t16_dual_gate_up_gemv_decode_fp16_fp16_out),
        ("t16_dual_gemv_decode_bf16_bf16_out", gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out),
        ("t16_dual_gemv_decode_fp16_fp16_out", gguf_q8_0_t16_dual_gemv_decode_fp16_fp16_out),
        ("t16_triple_gemv_decode_bf16_bf16_out", gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out),
        ("t16_triple_gemv_decode_fp16_fp16_out", gguf_q8_0_t16_triple_gemv_decode_fp16_fp16_out),
    ):
        register(
            KernelKey("hip_gfx1100", "linear", "gguf_q8_0_t16_v1", variant),
            fn,
            replace=replace,
        )


register_gguf_q8_0_t16_gemv_kernels()


__all__ = [
    "build_gguf_q8_0_t16_gemv",
    "gguf_q8_0_t16_dual_gate_up_gemv_decode_bf16_bf16_out",
    "gguf_q8_0_t16_dual_gate_up_gemv_decode_fp16_fp16_out",
    "gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out",
    "gguf_q8_0_t16_dual_gemv_decode_fp16_fp16_out",
    "gguf_q8_0_t16_gemv_decode_bf16_bf16_out",
    "gguf_q8_0_t16_gemv_decode_f32_bf16_out",
    "gguf_q8_0_t16_gemv_decode_fp16_fp16_out",
    "gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out",
    "gguf_q8_0_t16_triple_gemv_decode_fp16_fp16_out",
    "plan_gguf_q8_0_t16_gemv_build",
    "register_gguf_q8_0_t16_gemv_kernels",
]
