"""Raw-pointer wrapper for the B3 fused selected-expert PARO MoE FFN megakernel.

One block per selected (token, expert) row computes the whole PARO expert FFN
(``rotate1 -> AWQ gate_up GEMV -> silu*mul -> down_rotate -> AWQ down GEMV``)
with both incoherence rotations and the ffn_len-wide intermediate kept on-chip,
producing the per-selected-row down projection ``out[rows, hidden]``. The
routing-weighted combine remains a separate kernel.

Registered under the four-axis key
``(hip_gfx1100, moe_ffn_selected, w4_paro, fused_rotate_dual_silu_rotate_down_*)``.
The numerically-equivalent unfused fallback is the existing PARO primitive chain
(``paro_rotate1`` -> ``gemv_awq_dual_pack8`` -> ``silu_mul_dual_rotate_out`` ->
``gemv_awq_pack8``), which the runner keeps using when this fused variant is
unavailable.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("paro_moe_ffn_fused.hip")
_OUTPUT_NAME = "paro_moe_ffn_fused.so"
_FAMILY = "paro_moe_ffn_fused"
_SYMBOL_BF16_BF16_OUT = "hipengine_paro_selected_ffn_fused_bf16_bf16_out"
_SYMBOL_FP16_FP16_OUT = "hipengine_paro_selected_ffn_fused_fp16_fp16_out"
_SYMBOL_F32_F32_OUT = "hipengine_paro_selected_ffn_fused_f32_f32_out"
_ALLOWED_THREADS = {64, 128, 256}

# 18 device pointers (17 inputs + out) + 8 int64 scalars + stream.
_ARGTYPES = (
    [ctypes.c_void_p] * 18
    + [ctypes.c_int64] * 8
    + [ctypes.c_void_p]
)


def plan_paro_moe_ffn_fused_build(
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


def build_paro_moe_ffn_fused(
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


def _validate(
    x_rows: int,
    rows: int,
    num_experts: int,
    hidden: int,
    ffn_len: int,
    group_size: int,
    krot: int,
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
    if group_size <= 0 or hidden % group_size != 0 or ffn_len % group_size != 0:
        raise ValueError("hidden and ffn_len must be divisible by group_size")
    if hidden % 8 != 0 or ffn_len % 8 != 0:
        raise ValueError("hidden and ffn_len must be divisible by 8 (AWQ pack8)")
    if krot <= 0:
        raise ValueError("krot must be positive")
    if threads not in _ALLOWED_THREADS:
        allowed = ", ".join(str(value) for value in sorted(_ALLOWED_THREADS))
        raise ValueError(f"threads must be one of {allowed}")


def _launch_fused(
    symbol: str,
    ptrs: tuple[int, ...],
    x_rows: int,
    rows: int,
    num_experts: int,
    hidden: int,
    ffn_len: int,
    group_size: int,
    krot: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _validate(x_rows, rows, num_experts, hidden, ffn_len, group_size, krot, threads)
    if len(ptrs) != 18:
        raise ValueError("expected 18 device pointers (17 inputs + out)")
    library = library or build_paro_moe_ffn_fused(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = _ARGTYPES
    fn.restype = ctypes.c_int
    err = fn(
        *[ctypes.c_void_p(p) for p in ptrs],
        ctypes.c_int64(x_rows),
        ctypes.c_int64(rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(hidden),
        ctypes.c_int64(ffn_len),
        ctypes.c_int64(group_size),
        ctypes.c_int64(krot),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def paro_selected_ffn_fused_f32_f32_out(
    x_ptr: int,
    selected_ptr: int,
    gate_qw_ptr: int,
    gate_qz_ptr: int,
    gate_sc_ptr: int,
    up_qw_ptr: int,
    up_qz_ptr: int,
    up_sc_ptr: int,
    down_qw_ptr: int,
    down_qz_ptr: int,
    down_sc_ptr: int,
    r1_pairs_ptr: int,
    r1_theta_ptr: int,
    r1_cscale_ptr: int,
    dr_pairs_ptr: int,
    dr_theta_ptr: int,
    dr_cscale_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    hidden: int,
    ffn_len: int,
    group_size: int,
    krot: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Fused selected-expert PARO MoE FFN with FP32 activation and FP32 output."""

    _launch_fused(
        _SYMBOL_F32_F32_OUT,
        (
            x_ptr, selected_ptr,
            gate_qw_ptr, gate_qz_ptr, gate_sc_ptr,
            up_qw_ptr, up_qz_ptr, up_sc_ptr,
            down_qw_ptr, down_qz_ptr, down_sc_ptr,
            r1_pairs_ptr, r1_theta_ptr, r1_cscale_ptr,
            dr_pairs_ptr, dr_theta_ptr, dr_cscale_ptr,
            out_ptr,
        ),
        x_rows, rows, num_experts, hidden, ffn_len, group_size, krot,
        threads=threads, stream=stream, library=library, runtime=runtime,
    )


def paro_selected_ffn_fused_fp16_fp16_out(
    x_ptr: int,
    selected_ptr: int,
    gate_qw_ptr: int,
    gate_qz_ptr: int,
    gate_sc_ptr: int,
    up_qw_ptr: int,
    up_qz_ptr: int,
    up_sc_ptr: int,
    down_qw_ptr: int,
    down_qz_ptr: int,
    down_sc_ptr: int,
    r1_pairs_ptr: int,
    r1_theta_ptr: int,
    r1_cscale_ptr: int,
    dr_pairs_ptr: int,
    dr_theta_ptr: int,
    dr_cscale_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    hidden: int,
    ffn_len: int,
    group_size: int,
    krot: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Fused selected-expert PARO MoE FFN with FP16 activation and FP16 output."""

    _launch_fused(
        _SYMBOL_FP16_FP16_OUT,
        (
            x_ptr, selected_ptr,
            gate_qw_ptr, gate_qz_ptr, gate_sc_ptr,
            up_qw_ptr, up_qz_ptr, up_sc_ptr,
            down_qw_ptr, down_qz_ptr, down_sc_ptr,
            r1_pairs_ptr, r1_theta_ptr, r1_cscale_ptr,
            dr_pairs_ptr, dr_theta_ptr, dr_cscale_ptr,
            out_ptr,
        ),
        x_rows, rows, num_experts, hidden, ffn_len, group_size, krot,
        threads=threads, stream=stream, library=library, runtime=runtime,
    )


def paro_selected_ffn_fused_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    gate_qw_ptr: int,
    gate_qz_ptr: int,
    gate_sc_ptr: int,
    up_qw_ptr: int,
    up_qz_ptr: int,
    up_sc_ptr: int,
    down_qw_ptr: int,
    down_qz_ptr: int,
    down_sc_ptr: int,
    r1_pairs_ptr: int,
    r1_theta_ptr: int,
    r1_cscale_ptr: int,
    dr_pairs_ptr: int,
    dr_theta_ptr: int,
    dr_cscale_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    hidden: int,
    ffn_len: int,
    group_size: int,
    krot: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Fused selected-expert PARO MoE FFN with BF16 activation and BF16 output."""

    _launch_fused(
        _SYMBOL_BF16_BF16_OUT,
        (
            x_ptr, selected_ptr,
            gate_qw_ptr, gate_qz_ptr, gate_sc_ptr,
            up_qw_ptr, up_qz_ptr, up_sc_ptr,
            down_qw_ptr, down_qz_ptr, down_sc_ptr,
            r1_pairs_ptr, r1_theta_ptr, r1_cscale_ptr,
            dr_pairs_ptr, dr_theta_ptr, dr_cscale_ptr,
            out_ptr,
        ),
        x_rows, rows, num_experts, hidden, ffn_len, group_size, krot,
        threads=threads, stream=stream, library=library, runtime=runtime,
    )


def register_paro_moe_ffn_fused_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "moe_ffn_selected", "w4_paro", "fused_rotate_dual_silu_rotate_down_f32_f32_out"),
        paro_selected_ffn_fused_f32_f32_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "moe_ffn_selected", "w4_paro", "fused_rotate_dual_silu_rotate_down_bf16_bf16_out"),
        paro_selected_ffn_fused_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "moe_ffn_selected", "w4_paro", "fused_rotate_dual_silu_rotate_down_fp16_fp16_out"),
        paro_selected_ffn_fused_fp16_fp16_out,
        replace=replace,
    )


register_paro_moe_ffn_fused_kernels()


__all__ = [
    "build_paro_moe_ffn_fused",
    "plan_paro_moe_ffn_fused_build",
    "paro_selected_ffn_fused_f32_f32_out",
    "paro_selected_ffn_fused_bf16_bf16_out",
    "paro_selected_ffn_fused_fp16_fp16_out",
    "register_paro_moe_ffn_fused_kernels",
]
