"""Wrappers for selected GGUF X8 replacement-layout GEMV kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_x8_selected_gemv.hip")
_OUTPUT_NAME = "gguf_x8_selected_gemv.so"
_QK_K = 256
_X8_COLS = 8
_ALLOWED_THREADS = {64, 128, 256}

_Q5_DIRECT_BF16 = "hipengine_gguf_q5_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out"
_Q6_DIRECT_BF16 = "hipengine_gguf_q6_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out"
_Q5_COMPACT_BF16 = "hipengine_gguf_q5_k_x8_selected_q8_1_dp4a_gemv_decode_compact_bf16_bf16_out"
_Q6_COMPACT_BF16 = "hipengine_gguf_q6_k_x8_selected_q8_1_dp4a_gemv_decode_compact_bf16_bf16_out"


def plan_gguf_x8_selected_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_x8_selected_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_x8_selected_gemv(
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
        family="gguf_x8_selected_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gguf_q5_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out(
    xq_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
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
) -> None:
    """Launch direct selected Q5X8 q8_1+sudot4 GEMV."""

    _launch_direct(
        _Q5_DIRECT_BF16,
        xq_ptr,
        selected_ptr,
        tiles_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out(
    xq_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
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
) -> None:
    """Launch direct selected Q6X8 q8_1+sudot4 GEMV."""

    _launch_direct(
        _Q6_DIRECT_BF16,
        xq_ptr,
        selected_ptr,
        tiles_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q5_k_x8_selected_q8_1_dp4a_gemv_decode_compact_bf16_bf16_out(
    xq_ptr: int,
    expert_start_compact_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch compact selected Q5X8 q8_1+sudot4 GEMV."""

    _launch_compact(
        _Q5_COMPACT_BF16,
        xq_ptr,
        expert_start_compact_ptr,
        tiles_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features,
        num_experts,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_x8_selected_q8_1_dp4a_gemv_decode_compact_bf16_bf16_out(
    xq_ptr: int,
    expert_start_compact_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch compact selected Q6X8 q8_1+sudot4 GEMV."""

    _launch_compact(
        _Q6_COMPACT_BF16,
        xq_ptr,
        expert_start_compact_ptr,
        tiles_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features,
        num_experts,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _launch_direct(
    symbol: str,
    xq_ptr: int,
    selected_ptr: int,
    tiles_ptr: int,
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
    if x_rows <= 0:
        raise ValueError("x_rows must be positive")
    if rows <= 0 or rows % x_rows != 0:
        raise ValueError("rows must be positive and divisible by x_rows")
    _check_common(rows, in_features, out_features, num_experts, threads)
    library = library or build_gguf_x8_selected_gemv(load=True)
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
        ctypes.c_void_p(xq_ptr),
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(x_rows),
        ctypes.c_int64(rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_compact(
    symbol: str,
    xq_ptr: int,
    expert_start_compact_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_common(compact_rows, in_features, out_features, num_experts, threads)
    library = library or build_gguf_x8_selected_gemv(load=True)
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(xq_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _check_common(rows: int, in_features: int, out_features: int, num_experts: int, threads: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0 or in_features % _QK_K != 0:
        raise ValueError("in_features must be positive and divisible by GGUF K block size 256")
    if out_features <= 0 or out_features % _X8_COLS != 0:
        raise ValueError("out_features must be positive and divisible by 8")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, or 256")


def register_gguf_x8_selected_gemv_kernels(*, replace: bool = True) -> None:
    """Register selected X8 q8_1+sudot4 GEMV kernels."""

    for quant_key, direct, compact in (
        (
            "gguf_q5_k_x8_v1",
            gguf_q5_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out,
            gguf_q5_k_x8_selected_q8_1_dp4a_gemv_decode_compact_bf16_bf16_out,
        ),
        (
            "gguf_q6_k_x8_v1",
            gguf_q6_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out,
            gguf_q6_k_x8_selected_q8_1_dp4a_gemv_decode_compact_bf16_bf16_out,
        ),
    ):
        register(
            KernelKey("hip_gfx1100", "moe_linear", quant_key, "selected_x8_q8_1_dp4a_gemv_decode_bf16_bf16_out"),
            direct,
            replace=replace,
        )
        register(
            KernelKey(
                "hip_gfx1100",
                "moe_linear",
                quant_key,
                "selected_x8_q8_1_dp4a_gemv_decode_compact_bf16_bf16_out",
            ),
            compact,
            replace=replace,
        )


register_gguf_x8_selected_gemv_kernels()


__all__ = [
    "build_gguf_x8_selected_gemv",
    "gguf_q5_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out",
    "gguf_q5_k_x8_selected_q8_1_dp4a_gemv_decode_compact_bf16_bf16_out",
    "gguf_q6_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out",
    "gguf_q6_k_x8_selected_q8_1_dp4a_gemv_decode_compact_bf16_bf16_out",
    "plan_gguf_x8_selected_gemv_build",
    "register_gguf_x8_selected_gemv_kernels",
]
