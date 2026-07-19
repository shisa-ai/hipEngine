"""Raw selected IQ3_XXS/IQ4_XS MoE decode kernels.

The fused gate/up wrappers preserve the existing BF16 selected-expert ABI and
materialized rounding boundary while reading the resident GGUF blocks directly.
The IQ4_XS single-output wrapper serves the routed down projection.  All
launches are stream-ordered and contain no host reads, allocations, or device
synchronization, so they can participate in resident graph capture/replay.
"""

from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_iq_selected_gemv.hip")
_TABLES = Path(__file__).with_name("gguf_iq_tables.h")
_OUTPUT_NAME = "gguf_iq_selected_gemv.so"
_IQ3_GATE_UP = "hipengine_gguf_iq3_xxs_selected_fused_gate_up_silu_bf16_bf16_out"
_IQ4_GATE_UP = "hipengine_gguf_iq4_xs_selected_fused_gate_up_silu_bf16_bf16_out"
_IQ4_SELECTED = "hipengine_gguf_iq4_xs_selected_gemv_bf16_bf16_out"
_QK_K = 256


def plan_gguf_iq_selected_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_iq_selected_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=(_table_revision_flag(),),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_iq_selected_gemv(
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
        family="gguf_iq_selected_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=(_table_revision_flag(),),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _table_revision_flag() -> str:
    """Make the generated IQ-table include part of the JIT cache identity."""

    digest = hashlib.sha256(_TABLES.read_bytes()).hexdigest()[:16]
    return f"-DHIPENGINE_IQ_TABLES_REV=0x{digest}ULL"


def gguf_iq3_xxs_selected_fused_gate_up_silu_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch fused selected IQ3_XXS gate/up + SiLU decode."""

    _launch_pair(
        _IQ3_GATE_UP,
        x_ptr,
        selected_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq4_xs_selected_fused_gate_up_silu_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch fused selected IQ4_XS gate/up + SiLU decode."""

    _launch_pair(
        _IQ4_GATE_UP,
        x_ptr,
        selected_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_iq4_xs_selected_gemv_bf16_bf16_out(
    x_ptr: int,
    selected_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch selected raw IQ4_XS BF16 down-projection GEMV."""

    _validate(x_rows, rows, num_experts, in_features, out_features)
    library = library or build_gguf_iq_selected_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _IQ4_SELECTED)
    _set_signature(fn, pointer_count=4)
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(weight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(x_rows),
        ctypes.c_int64(rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_pair(
    symbol: str,
    x_ptr: int,
    selected_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _validate(x_rows, rows, num_experts, in_features, out_features)
    library = library or build_gguf_iq_selected_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    _set_signature(fn, pointer_count=5)
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
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _set_signature(fn, *, pointer_count: int) -> None:
    fn.argtypes = [
        *([ctypes.c_void_p] * pointer_count),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int


def _validate(
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
) -> None:
    if x_rows <= 0:
        raise ValueError("x_rows must be positive")
    if rows <= 0:
        raise ValueError("rows must be positive")
    if rows % x_rows != 0:
        raise ValueError("rows must be divisible by x_rows")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if in_features <= 0 or in_features % _QK_K != 0:
        raise ValueError("in_features must be divisible by GGUF IQ block size 256")
    if out_features <= 0:
        raise ValueError("out_features must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_gguf_iq_selected_gemv_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_fused_gate_up_silu_bf16_bf16_out",
        ),
        gguf_iq3_xxs_selected_fused_gate_up_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_iq4_xs",
            "selected_fused_gate_up_silu_bf16_bf16_out",
        ),
        gguf_iq4_xs_selected_fused_gate_up_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_iq4_xs",
            "selected_gemv_bf16_bf16_out",
        ),
        gguf_iq4_xs_selected_gemv_bf16_bf16_out,
        replace=replace,
    )


register_gguf_iq_selected_gemv_kernels()


__all__ = [
    "build_gguf_iq_selected_gemv",
    "gguf_iq3_xxs_selected_fused_gate_up_silu_bf16_bf16_out",
    "gguf_iq4_xs_selected_fused_gate_up_silu_bf16_bf16_out",
    "gguf_iq4_xs_selected_gemv_bf16_bf16_out",
    "plan_gguf_iq_selected_gemv_build",
    "register_gguf_iq_selected_gemv_kernels",
]
