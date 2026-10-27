"""Raw-pointer wrappers for standard rotate-half RoPE over prefill rows.

This is the dense Llama-family rotary contract: full or partial rotate-half RoPE
on BF16 Q/K with no per-head RMSNorm, driven by per-row cos/sin tables. Models
that fuse a head norm into the rotation use the Qwen3.5 or Laguna owners instead.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("dense_rope.hip")
_FAMILY = "dense_rope"
_OUTPUT_NAME = "dense_rope.so"
_SYMBOL_POSITIONS_INPLACE = "hipengine_dense_rope_positions_bf16_inplace"
_ALLOWED_THREADS = {64, 128, 256}
_MAX_BLOCK = 256


def plan_dense_rope_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family=_FAMILY,
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_dense_rope(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
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


def _dense_rope_library() -> ctypes.CDLL:
    return build_dense_rope(load=True)


def dense_rope_positions_bf16_inplace(
    query_ptr: int,
    key_ptr: int,
    cos_ptr: int,
    sin_ptr: int,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Rotate BF16 Q/K in place over ``rows`` prefill rows.

    ``cos_ptr``/``sin_ptr`` address FP32 ``[rows, rotary_dim]`` tables. Only the
    first ``rotary_dim`` channels of each head are rotated; the remainder are left
    untouched.
    """

    parsed_rows = int(rows)
    parsed_q_heads = int(num_q_heads)
    parsed_kv_heads = int(num_kv_heads)
    parsed_head_dim = int(head_dim)
    parsed_rotary_dim = int(rotary_dim)
    if parsed_rows <= 0:
        raise ValueError("rows must be positive")
    if parsed_q_heads <= 0 or parsed_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if parsed_head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if parsed_rotary_dim <= 0 or parsed_rotary_dim > parsed_head_dim:
        raise ValueError("rotary_dim must be positive and no larger than head_dim")
    if parsed_rotary_dim % 2:
        raise ValueError("rotary_dim must be even for rotate-half RoPE")
    if int(threads) not in _ALLOWED_THREADS:
        raise ValueError(f"threads must be one of {sorted(_ALLOWED_THREADS)}")
    for name, pointer in (
        ("query_ptr", query_ptr),
        ("key_ptr", key_ptr),
        ("cos_ptr", cos_ptr),
        ("sin_ptr", sin_ptr),
    ):
        if int(pointer) <= 0:
            raise ValueError(f"{name} must be non-zero")

    library = library or _dense_rope_library()
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_POSITIONS_INPLACE)
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
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_ptr),
        ctypes.c_void_p(cos_ptr),
        ctypes.c_void_p(sin_ptr),
        ctypes.c_int64(parsed_q_heads),
        ctypes.c_int64(parsed_kv_heads),
        ctypes.c_int64(parsed_head_dim),
        ctypes.c_int64(parsed_rotary_dim),
        ctypes.c_int64(parsed_rows),
        ctypes.c_int64(int(threads)),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_dense_rope_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "rotate", "bf16", "positions_inplace"),
        dense_rope_positions_bf16_inplace,
        replace=replace,
    )


__all__ = [
    "build_dense_rope",
    "dense_rope_positions_bf16_inplace",
    "plan_dense_rope_build",
    "register_dense_rope_kernels",
]
