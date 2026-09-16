"""YuE2 NAR kernel build + launch wrappers (nar.hip, nar_wmma.hip).

Raw device pointers only; the host runtime converts. The attention kernel is
bidirectional over the concatenated ``[AR cache | NAR]`` key space and uses a
tiled online softmax, so a chunk's key count is bounded by the model context
rather than by shared memory.

``nar_wmma.hip`` adds a tensor-core attention for the production head geometry
(16 query heads over 8 key/value heads, head_dim 128). It changes arithmetic, so
``nar_attention_f32`` remains the strict fallback and the caller chooses.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime

_SOURCE = Path(__file__).with_name("nar.hip")
_SOURCE_WMMA = Path(__file__).with_name("nar_wmma.hip")
_OUTPUT_NAME = "yue2_nar"

_P = ctypes.c_void_p
_F = ctypes.c_float
_I = ctypes.c_int64
_S = ctypes.c_void_p

_ARGTYPES_GATHER_ADD = (_P, _P, _P, _P, _I, _I, _S)
_ARGTYPES_ADD_BROADCAST = (_P, _P, _P, _I, _I, _S)
_ARGTYPES_STATE_UPDATE = (_P, _P, _F, _P, _I, _I, _S)
_ARGTYPES_ATTENTION = (_P, _P, _P, _P, _P, _P, _I, _I, _I, _I, _I, _F, _S)
_ARGTYPES_ATTENTION_WMMA = (_P, _P, _P, _P, _P, _P, _I, _I, _I, _I, _I, _F, _S)
_ARGTYPES_ROPE = (_P, _P, _P, _P, _P, _P, _P, _I, _I, _I, _I, _S)


def plan_yue2_nar_build(**kwargs):
    return plan_hip_build(
        sources=[_SOURCE, _SOURCE_WMMA],
        family="yue2_nar",
        output_name=_OUTPUT_NAME,
        **kwargs,
    )


def build_yue2_nar(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: str = "decode",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | None:
    return build_hip(
        sources=[_SOURCE, _SOURCE_WMMA],
        family="yue2_nar",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _library() -> ctypes.CDLL:
    library = build_yue2_nar()
    if library is None:
        raise RuntimeError("yue2_nar build returned no library")
    return library


def nar_gather_add_bf16(
    x_ptr: int,
    table_ptr: int,
    positions_ptr: int,
    out_ptr: int,
    rows: int,
    width: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """``out[row] = bf16(x[row] + table[positions[row]])``."""
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_yue2_nar_gather_add_bf16", _ARGTYPES_GATHER_ADD, ctypes.c_int
    )
    err = fn(x_ptr, table_ptr, positions_ptr, out_ptr, rows, width, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def nar_add_broadcast_bf16(
    x_ptr: int,
    vector_ptr: int,
    out_ptr: int,
    rows: int,
    width: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """``out[row, c] = bf16(x[row, c] + vector[c])``."""
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_yue2_nar_add_broadcast_bf16", _ARGTYPES_ADD_BROADCAST, ctypes.c_int
    )
    err = fn(x_ptr, vector_ptr, out_ptr, rows, width, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def nar_state_update_bf16(
    state_ptr: int,
    velocity_ptr: int,
    scale: float,
    out_ptr: int,
    rows: int,
    width: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """``out = bf16(state - bf16(velocity * scale))``: two roundings, as torch.

    ``scale`` is FP32 because the reference multiplies by a Python float, which
    torch evaluates in the tensor's FP32 compute type before rounding to BF16.
    """
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_yue2_nar_state_update_bf16", _ARGTYPES_STATE_UPDATE, ctypes.c_int
    )
    err = fn(state_ptr, velocity_ptr, scale, out_ptr, rows, width, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def nar_attention_f32(
    q_ptr: int,
    nar_k_ptr: int,
    nar_v_ptr: int,
    ar_k_ptr: int,
    ar_v_ptr: int,
    out_ptr: int,
    rows: int,
    ar_rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Bidirectional GQA over ``[AR cache | NAR]`` keys; Q and out are FP32."""
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_yue2_nar_attention_f32", _ARGTYPES_ATTENTION, ctypes.c_int
    )
    err = fn(
        q_ptr, nar_k_ptr, nar_v_ptr, ar_k_ptr, ar_v_ptr, out_ptr,
        rows, ar_rows, num_q_heads, num_kv_heads, head_dim, scale, stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def nar_attention_wmma(
    q_ptr: int,
    nar_k_ptr: int,
    nar_v_ptr: int,
    ar_k_ptr: int,
    ar_v_ptr: int,
    out_ptr: int,
    rows: int,
    ar_rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Tensor-core path for the production head geometry.

    Same ABI as :func:`nar_attention_f32`. The device entry point rejects any
    geometry it does not implement, so the caller can fall back on its error
    code rather than duplicating the guards.
    """
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_yue2_nar_attention_wmma",
        _ARGTYPES_ATTENTION_WMMA, ctypes.c_int,
    )
    err = fn(
        q_ptr, nar_k_ptr, nar_v_ptr, ar_k_ptr, ar_v_ptr, out_ptr,
        rows, ar_rows, num_q_heads, num_kv_heads, head_dim, scale, stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def nar_rope_f32(
    q_ptr: int,
    k_ptr: int,
    cos_table_ptr: int,
    sin_table_ptr: int,
    positions_ptr: int,
    q_out_ptr: int,
    k_out_ptr: int,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Rotate-half RoPE with per-row positions; either head count may be zero."""
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_yue2_nar_rope_f32", _ARGTYPES_ROPE, ctypes.c_int)
    err = fn(
        q_ptr, k_ptr, cos_table_ptr, sin_table_ptr, positions_ptr, q_out_ptr, k_out_ptr,
        rows, num_q_heads, num_kv_heads, head_dim, stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))
