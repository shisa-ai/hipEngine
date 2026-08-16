"""Exact mixed-grid standard-Q6 QKV plus Q4 gate decode pair."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q6_q4_pair.hip")
_OUTPUT_NAME = "gguf_q6_q4_pair.so"
_SYMBOL = "hipengine_gguf_q6_q4_t16_mixed_grid_pair_bf16_bf16_out"
_QUANT = "gguf_q6_k_t16_v1+gguf_q4_k_t16_v1"
_VARIANT = "mixed_grid_bf16_bf16_out"
_ARGTYPES = [
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


def plan_gguf_q6_q4_pair_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q6_q4_pair",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q6_q4_pair(
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
        family="gguf_q6_q4_pair",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gguf_q6_q4_t16_mixed_grid_pair_bf16_bf16_out(
    x_ptr: int,
    q6_tiles_ptr: int,
    q4_tiles_ptr: int,
    q6_out_ptr: int,
    q4_out_ptr: int,
    rows: int,
    in_features: int,
    q6_out_features: int,
    q4_out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the rows1 block-parallel standard-Q6/Q4 projection pair."""

    if int(rows) != 1:
        raise ValueError("Q6/Q4 mixed-grid decode requires rows == 1")
    if int(in_features) <= 0 or int(in_features) % 256:
        raise ValueError("in_features must be a positive multiple of 256")
    if int(q6_out_features) <= 0 or int(q6_out_features) % 16:
        raise ValueError("Q6 output features must be a positive multiple of 16")
    if int(q4_out_features) <= 0 or int(q4_out_features) % 32:
        raise ValueError("Q4 output features must be a positive multiple of 32")

    lib = library or build_gguf_q6_q4_pair(load=True)
    rt = runtime or get_hip_runtime()
    fn = signed_kernel_fn(lib, _SYMBOL, _ARGTYPES, ctypes.c_int)
    status = fn(
        int(x_ptr),
        int(q6_tiles_ptr),
        int(q4_tiles_ptr),
        int(q6_out_ptr),
        int(q4_out_ptr),
        int(rows),
        int(in_features),
        int(q6_out_features),
        int(q4_out_features),
        int(stream),
    )
    rt.check(int(status))


def register_gguf_q6_q4_pair_kernels(*, replace: bool = False) -> None:
    register(
        KernelKey("hip_gfx1100", "linear_pair", _QUANT, _VARIANT),
        gguf_q6_q4_t16_mixed_grid_pair_bf16_bf16_out,
        replace=replace,
    )


register_gguf_q6_q4_pair_kernels()


__all__ = [
    "build_gguf_q6_q4_pair",
    "gguf_q6_q4_t16_mixed_grid_pair_bf16_bf16_out",
    "plan_gguf_q6_q4_pair_build",
    "register_gguf_q6_q4_pair_kernels",
]
