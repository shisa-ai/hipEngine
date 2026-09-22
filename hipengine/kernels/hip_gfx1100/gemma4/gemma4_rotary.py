"""Raw-pointer wrappers for the Gemma 4 partial rotary and K-to-V copy.

See ``gemma4_rope.py`` for the table layout these kernels index. The rotated
elements are two spans, not a contiguous prefix, so ``rotary_dim`` is always
``head_dim`` here; a smaller value would rotate the wrong pairs.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_ARGTYPES_ROTARY = (
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
)
_ARGTYPES_K_TO_V = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_void_p,
)

_SOURCE = Path(__file__).with_name("gemma4_rotary.hip")
_OUTPUT_NAME = "gemma4_rotary.so"

_SYMBOL_ROTARY_BF16 = "hipengine_gemma4_partial_rotary_bf16"
_SYMBOL_ROTARY_F32 = "hipengine_gemma4_partial_rotary_f32"
_SYMBOL_K_TO_V_BF16 = "hipengine_gemma4_k_to_v_bf16"


def plan_gemma4_rotary_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gemma4_rotary",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gemma4_rotary(
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
        family="gemma4_rotary",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gemma4_partial_rotary_bf16(
    query_ptr: int,
    key_ptr: int,
    cos_ptr: int,
    sin_ptr: int,
    query_out_ptr: int,
    key_out_ptr: int,
    tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    rotary_dim: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Rotate BF16 query and key heads with a per-token doubled angle table.

    ``query`` is ``(tokens, num_q_heads, head_dim)`` and ``key`` is
    ``(tokens, num_kv_heads, head_dim)``, both BF16. ``cos``/``sin`` are F32
    ``(tokens, head_dim)``. Pass ``key_ptr = 0`` to rotate queries only.
    """

    rotary = _check_rotary_shape(tokens, num_q_heads, num_kv_heads, head_dim, rotary_dim)
    library = library or build_gemma4_rotary(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_ROTARY_BF16, _ARGTYPES_ROTARY, ctypes.c_int)
    err = fn(
        query_ptr,
        key_ptr,
        cos_ptr,
        sin_ptr,
        query_out_ptr,
        key_out_ptr,
        tokens,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary,
        stream,
    )
    _check_launch(runtime, err)


def gemma4_partial_rotary_f32(
    query_ptr: int,
    key_ptr: int,
    cos_ptr: int,
    sin_ptr: int,
    query_out_ptr: int,
    key_out_ptr: int,
    tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    rotary_dim: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """F32 twin of :func:`gemma4_partial_rotary_bf16`."""

    rotary = _check_rotary_shape(tokens, num_q_heads, num_kv_heads, head_dim, rotary_dim)
    library = library or build_gemma4_rotary(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_ROTARY_F32, _ARGTYPES_ROTARY, ctypes.c_int)
    err = fn(
        query_ptr,
        key_ptr,
        cos_ptr,
        sin_ptr,
        query_out_ptr,
        key_out_ptr,
        tokens,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary,
        stream,
    )
    _check_launch(runtime, err)


def gemma4_k_to_v_bf16(
    key_ptr: int,
    value_ptr: int,
    total: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Copy a raw K projection into the V slot for ``attention_k_eq_v`` layers."""

    if total <= 0:
        raise ValueError("total must be positive")
    library = library or build_gemma4_rotary(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_K_TO_V_BF16, _ARGTYPES_K_TO_V, ctypes.c_int)
    err = fn(key_ptr, value_ptr, total, stream)
    _check_launch(runtime, err)


def register_gemma4_rotary_kernels(*, replace: bool = False) -> None:
    """Register the Gemma 4 rotary family against the four-axis registry."""

    for quant in ("gguf_q4_k_m", "gguf_q4_k_xl", "gguf_q8_0", "gguf"):
        register(
            KernelKey("hip_gfx1100", "partial_rotary", quant, "gemma4_plain"),
            gemma4_partial_rotary_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "k_to_v", quant, "gemma4_plain"),
            gemma4_k_to_v_bf16,
            replace=replace,
        )


def _check_rotary_shape(
    tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int | None,
) -> int:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if num_q_heads <= 0:
        raise ValueError("num_q_heads must be positive")
    if num_kv_heads < 0:
        raise ValueError("num_kv_heads must not be negative")
    if head_dim <= 0 or head_dim % 2:
        raise ValueError("head_dim must be a positive even number")
    rotary = head_dim if rotary_dim is None else int(rotary_dim)
    if rotary != head_dim:
        raise ValueError(
            "Gemma 4 rotates pairs (i, i + head_dim // 2) and leaves a gap, so "
            f"rotary_dim must be head_dim ({head_dim}), got {rotary}"
        )
    return rotary


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_gemma4_rotary_kernels()
