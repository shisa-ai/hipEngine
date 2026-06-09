"""Raw-pointer wrapper for the B1 fused selected-expert GGUF Q4_K MoE FFN.

One block per selected (token, expert) row computes the whole expert FFN
(``gate_up GEMV -> silu*mul -> down GEMV``) with the ffn_len-wide intermediate
kept on-chip, producing the per-selected-row down projection ``out[rows, hidden]``.
The routing-weighted combine remains a separate kernel.

Registered under the four-axis key
``(hip_gfx1100, moe_ffn_selected, gguf_q4_k, fused_dual_silu_down_*)``. The
numerically-equivalent unfused fallback is the existing primitive chain
(``gguf_q4_k_selected_dual_gemv`` -> ``silu_mul`` -> ``gguf_q4_k_selected_gemv``),
which the runner keeps using when this fused variant is unavailable.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q4_k_moe_ffn_fused.hip")
_OUTPUT_NAME = "gguf_q4_k_moe_ffn_fused.so"
_FAMILY = "gguf_q4_k_moe_ffn_fused"
_SYMBOL_BF16_BF16_OUT = "hipengine_gguf_q4_k_selected_ffn_fused_bf16_bf16_out"
_SYMBOL_F32_F32_OUT = "hipengine_gguf_q4_k_selected_ffn_fused_f32_f32_out"
_ALLOWED_THREADS = {64, 128, 256}
_Q4_K_BLOCK = 256


def plan_gguf_q4_k_moe_ffn_fused_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family=_FAMILY,
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q4_k_moe_ffn_fused(
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
        family=_FAMILY,
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _launch_fused(
    symbol: str,
    x_ptr: int,
    selected_ptr: int,
    gate_ptr: int,
    up_ptr: int,
    down_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    hidden: int,
    ffn_len: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _validate(x_rows, rows, num_experts, hidden, ffn_len, threads)
    library = library or build_gguf_q4_k_moe_ffn_fused(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,  # x
        ctypes.c_void_p,  # selected
        ctypes.c_void_p,  # gate_w
        ctypes.c_void_p,  # up_w
        ctypes.c_void_p,  # down_w
        ctypes.c_void_p,  # out
        ctypes.c_int64,  # x_rows
        ctypes.c_int64,  # rows
        ctypes.c_int64,  # num_experts
        ctypes.c_int64,  # hidden
        ctypes.c_int64,  # ffn_len
        ctypes.c_int64,  # threads
        ctypes.c_void_p,  # stream
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(up_ptr),
        ctypes.c_void_p(down_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(x_rows),
        ctypes.c_int64(rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(hidden),
        ctypes.c_int64(ffn_len),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_selected_ffn_fused_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    gate_ptr: int,
    up_ptr: int,
    down_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    hidden: int,
    ffn_len: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Fused selected-expert Q4_K MoE FFN with BF16 activation and BF16 output."""

    _launch_fused(
        _SYMBOL_BF16_BF16_OUT,
        x_ptr,
        selected_ptr,
        gate_ptr,
        up_ptr,
        down_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        hidden,
        ffn_len,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_selected_ffn_fused_f32_f32_out(
    x_ptr: int,
    selected_ptr: int,
    gate_ptr: int,
    up_ptr: int,
    down_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    hidden: int,
    ffn_len: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Fused selected-expert Q4_K MoE FFN with FP32 activation and FP32 output."""

    _launch_fused(
        _SYMBOL_F32_F32_OUT,
        x_ptr,
        selected_ptr,
        gate_ptr,
        up_ptr,
        down_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        hidden,
        ffn_len,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def register_gguf_q4_k_moe_ffn_fused_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "moe_ffn_selected", "gguf_q4_k", "fused_dual_silu_down_bf16_bf16_out"),
        gguf_q4_k_selected_ffn_fused_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "moe_ffn_selected", "gguf_q4_k", "fused_dual_silu_down_f32_f32_out"),
        gguf_q4_k_selected_ffn_fused_f32_f32_out,
        replace=replace,
    )


def _validate(
    x_rows: int,
    rows: int,
    num_experts: int,
    hidden: int,
    ffn_len: int,
    threads: int,
) -> None:
    if x_rows <= 0:
        raise ValueError("x_rows must be positive")
    if rows <= 0 or rows % x_rows != 0:
        raise ValueError("rows must be positive and divisible by x_rows")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if hidden <= 0 or ffn_len <= 0:
        raise ValueError("hidden and ffn_len must be positive")
    if hidden % _Q4_K_BLOCK != 0:
        raise ValueError("hidden must be divisible by GGUF Q4_K block size 256")
    if ffn_len % _Q4_K_BLOCK != 0:
        raise ValueError("ffn_len must be divisible by GGUF Q4_K block size 256")
    if threads not in _ALLOWED_THREADS:
        allowed = ", ".join(str(value) for value in sorted(_ALLOWED_THREADS))
        raise ValueError(f"threads must be one of {allowed}")


register_gguf_q4_k_moe_ffn_fused_kernels()


__all__ = [
    "build_gguf_q4_k_moe_ffn_fused",
    "plan_gguf_q4_k_moe_ffn_fused_build",
    "gguf_q4_k_selected_ffn_fused_bf16_bf16_out",
    "gguf_q4_k_selected_ffn_fused_f32_f32_out",
    "register_gguf_q4_k_moe_ffn_fused_kernels",
]
