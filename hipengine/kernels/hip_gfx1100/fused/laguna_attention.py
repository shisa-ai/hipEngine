"""Raw-pointer wrappers for Laguna attention-side unfused primitives."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("laguna_attention.hip")
_OUTPUT_NAME = "laguna_attention.so"
_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_PACKED_ARGS = (
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
)
_ALLOWED_THREADS = {32, 64, 128, 256}


def plan_laguna_attention_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="laguna_attention",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_laguna_attention(
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
        family="laguna_attention",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _launch(
    symbol,
    context_ptr,
    gate_ptr,
    out_ptr,
    rows,
    heads,
    head_dim,
    *,
    threads,
    stream,
    library,
    runtime,
):
    if rows <= 0 or heads <= 0 or head_dim <= 0:
        raise ValueError("rows, heads, and head_dim must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 32, 64, 128, 256")
    library = library or build_laguna_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, symbol, _ARGS, ctypes.c_int)
    err = fn(context_ptr, gate_ptr, out_ptr, rows, heads, head_dim, threads, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def laguna_softplus_head_gate_f32_out(
    context_ptr,
    gate_ptr,
    out_ptr,
    rows,
    heads,
    head_dim,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    """Compute FP32 ``context * softplus(gate)`` with one gate per head."""

    _launch(
        "hipengine_laguna_softplus_head_gate_f32_out",
        context_ptr,
        gate_ptr,
        out_ptr,
        rows,
        heads,
        head_dim,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_softplus_head_gate_f32_bf16_out(
    context_ptr,
    gate_ptr,
    out_ptr,
    rows,
    heads,
    head_dim,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    """Compute FP32 softplus gating and round the result to BF16."""

    _launch(
        "hipengine_laguna_softplus_head_gate_f32_bf16_out",
        context_ptr,
        gate_ptr,
        out_ptr,
        rows,
        heads,
        head_dim,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_softplus_head_gate_f32_fp16_via_bf16_out(
    context_ptr,
    gate_ptr,
    out_ptr,
    rows,
    heads,
    head_dim,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    """Gate directly to the FP16 representation of the BF16 boundary."""

    _launch(
        "hipengine_laguna_softplus_head_gate_f32_fp16_via_bf16_out",
        context_ptr,
        gate_ptr,
        out_ptr,
        rows,
        heads,
        head_dim,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _launch_packed_tiles(
    symbol,
    context_ptr,
    gate_ptr,
    out_ptr,
    rows,
    heads,
    head_dim,
    packed_begin,
    packed_end,
    *,
    threads,
    stream,
    library,
    runtime,
):
    if rows <= 0 or heads <= 0 or head_dim <= 0:
        raise ValueError("rows, heads, and head_dim must be positive")
    if (
        packed_begin < 0
        or packed_begin >= packed_end
        or packed_end > rows
        or packed_begin % 128
        or packed_end % 128
    ):
        raise ValueError("packed tile range must be aligned within rows")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 32, 64, 128, 256")
    library = library or build_laguna_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, symbol, _PACKED_ARGS, ctypes.c_int)
    err = fn(
        context_ptr,
        gate_ptr,
        out_ptr,
        rows,
        heads,
        head_dim,
        packed_begin,
        packed_end,
        threads,
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def laguna_softplus_head_gate_f32_bf16_packed_tiles_out(
    context_ptr,
    gate_ptr,
    out_ptr,
    rows,
    heads,
    head_dim,
    packed_begin,
    packed_end,
    *,
    threads=128,
    stream=0,
    library=None,
    runtime=None,
):
    """Gate mixed generic/head-major 128-row tiles and emit generic BF16."""

    _launch_packed_tiles(
        "hipengine_laguna_softplus_head_gate_f32_bf16_packed_tiles_out",
        context_ptr,
        gate_ptr,
        out_ptr,
        rows,
        heads,
        head_dim,
        packed_begin,
        packed_end,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_softplus_head_gate_f32_fp16_via_bf16_packed_tiles_out(
    context_ptr,
    gate_ptr,
    out_ptr,
    rows,
    heads,
    head_dim,
    packed_begin,
    packed_end,
    *,
    threads=128,
    stream=0,
    library=None,
    runtime=None,
):
    """Gate mixed tiles directly to the FP16 representation of BF16."""

    _launch_packed_tiles(
        "hipengine_laguna_softplus_head_gate_f32_fp16_via_bf16_packed_tiles_out",
        context_ptr,
        gate_ptr,
        out_ptr,
        rows,
        heads,
        head_dim,
        packed_begin,
        packed_end,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def register_laguna_attention_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "attention_gate", "f32", "softplus_broadcast_f32_out"),
        laguna_softplus_head_gate_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "attention_gate",
            "f32",
            "softplus_broadcast_bf16_packed_tiles_out",
        ),
        laguna_softplus_head_gate_f32_bf16_packed_tiles_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "attention_gate",
            "f32",
            "softplus_broadcast_fp16_via_bf16_packed_tiles_out",
        ),
        laguna_softplus_head_gate_f32_fp16_via_bf16_packed_tiles_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "attention_gate", "f32", "softplus_broadcast_bf16_out"),
        laguna_softplus_head_gate_f32_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "attention_gate",
            "f32",
            "softplus_broadcast_fp16_via_bf16_out",
        ),
        laguna_softplus_head_gate_f32_fp16_via_bf16_out,
        replace=replace,
    )


register_laguna_attention_kernels()

__all__ = [
    "build_laguna_attention",
    "laguna_softplus_head_gate_f32_bf16_out",
    "laguna_softplus_head_gate_f32_bf16_packed_tiles_out",
    "laguna_softplus_head_gate_f32_fp16_via_bf16_out",
    "laguna_softplus_head_gate_f32_fp16_via_bf16_packed_tiles_out",
    "laguna_softplus_head_gate_f32_out",
    "plan_laguna_attention_build",
    "register_laguna_attention_kernels",
]
