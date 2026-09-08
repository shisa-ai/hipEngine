"""Raw-pointer launchers for the TimesFM 2.5 FP32 HIP kernel family.

Importing this module registers ctypes launch wrappers but does not build or
load ROCm until a wrapper is called.  Kernel semantics mirror the NumPy CPU
reference in ``hipengine/kernels/cpu_reference/timesfm.py``.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime

_SOURCE = Path(__file__).with_name("timesfm.hip")
_OUTPUT_NAME = "timesfm_f32.so"

_FAMILY = "timesfm_f32"

_ROWS_FEAT_EPS = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int32,
    ctypes.c_float, ctypes.c_void_p,
)
_NORM_ADD = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int32,
    ctypes.c_float, ctypes.c_void_p,
)
_ROWS_FEAT = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32,
    ctypes.c_void_p,
)
_1PTR_COUNT = (ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p)
_ADD = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
    ctypes.c_void_p,
)
_ROPE = (
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p,
)
_HEAD_RMSNORM = (
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_void_p,
)
_HEAD_PERDIM = (
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p,
)
_SCATTER = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p,
)
_ATTENTION = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.c_void_p,
)


def plan_timesfm_build(
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


def build_timesfm(
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


def _library() -> ctypes.CDLL:
    return build_timesfm(load=True)


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def timesfm_rmsnorm_f32(
    x_ptr: int,
    scale_ptr: int,
    out_ptr: int,
    rows: int,
    features: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """out = rmsnorm(x) * scale for row-major [rows, features]."""

    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "timesfm_rmsnorm_f32", _ROWS_FEAT_EPS, ctypes.c_int)
    err = fn(x_ptr, scale_ptr, out_ptr, rows, features, float(eps), stream)
    _check_launch(runtime, err)


def timesfm_norm_add_f32(
    x_ptr: int,
    other_ptr: int,
    scale_ptr: int,
    out_ptr: int,
    rows: int,
    features: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """out = rmsnorm(x) * scale + other (TimesFM post-norm residual)."""

    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "timesfm_norm_add_f32", _NORM_ADD, ctypes.c_int)
    err = fn(x_ptr, other_ptr, scale_ptr, out_ptr, rows, features, float(eps), stream)
    _check_launch(runtime, err)


def timesfm_bias_f32(
    x_ptr: int,
    bias_ptr: int,
    rows: int,
    features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """x += bias, in place."""

    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "timesfm_bias_f32", _ROWS_FEAT, ctypes.c_int)
    err = fn(x_ptr, bias_ptr, rows, features, stream)
    _check_launch(runtime, err)


def timesfm_bias_swish_f32(
    x_ptr: int,
    bias_ptr: int,
    rows: int,
    features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """x = swish(x + bias), in place."""

    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "timesfm_bias_swish_f32", _ROWS_FEAT, ctypes.c_int)
    err = fn(x_ptr, bias_ptr, rows, features, stream)
    _check_launch(runtime, err)


def timesfm_swish_f32(
    x_ptr: int,
    count: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """x = swish(x), in place."""

    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "timesfm_swish_f32", _1PTR_COUNT, ctypes.c_int)
    err = fn(x_ptr, count, stream)
    _check_launch(runtime, err)


def timesfm_add_f32(
    a_ptr: int,
    b_ptr: int,
    out_ptr: int,
    count: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """out = a + b, elementwise."""

    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "timesfm_add_f32", _ADD, ctypes.c_int)
    err = fn(a_ptr, b_ptr, out_ptr, count, stream)
    _check_launch(runtime, err)


def timesfm_rope_f32(
    x_ptr: int,
    pos_ptr: int,
    batch: int,
    patches: int,
    heads: int,
    head_dim: int,
    patch_stride: int,
    base: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply non-interleaved-halves RoPE in place to a packed-heads view."""

    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "timesfm_rope_f32", _ROPE, ctypes.c_int)
    err = fn(x_ptr, pos_ptr, batch, patches, heads, head_dim, patch_stride, base, stream)
    _check_launch(runtime, err)


def timesfm_head_rmsnorm_f32(
    x_ptr: int,
    scale_ptr: int,
    batch: int,
    patches: int,
    heads: int,
    head_dim: int,
    patch_stride: int,
    base: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Per-head RMSNorm in place on a packed-heads view (q or k)."""

    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "timesfm_head_rmsnorm_f32", _HEAD_RMSNORM, ctypes.c_int)
    err = fn(
        x_ptr, scale_ptr, batch, patches, heads, head_dim, patch_stride, base,
        float(eps), stream,
    )
    _check_launch(runtime, err)


def timesfm_head_perdim_scale_f32(
    x_ptr: int,
    scale_ptr: int,
    batch: int,
    patches: int,
    heads: int,
    head_dim: int,
    patch_stride: int,
    base: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Per-dim softplus query scaling in place on a packed-heads view (q)."""

    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "timesfm_head_perdim_scale_f32", _HEAD_PERDIM, ctypes.c_int)
    err = fn(
        x_ptr, scale_ptr, batch, patches, heads, head_dim, patch_stride, base, stream
    )
    _check_launch(runtime, err)


def timesfm_scatter_kv_f32(
    qkv_ptr: int,
    cache_k_ptr: int,
    cache_v_ptr: int,
    batch: int,
    patches: int,
    cache_size: int,
    heads: int,
    head_dim: int,
    patch_stride: int,
    start: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Copy packed k/v rows into the contiguous decode cache at `start`."""

    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "timesfm_scatter_kv_f32", _SCATTER, ctypes.c_int)
    err = fn(
        qkv_ptr, cache_k_ptr, cache_v_ptr,
        batch, patches, cache_size, heads, head_dim, patch_stride, start,
        stream,
    )
    _check_launch(runtime, err)


def timesfm_attention_f32(
    q_ptr: int,
    k_ptr: int,
    v_ptr: int,
    num_masked_ptr: int,
    q_offset_ptr: int,
    out_ptr: int,
    batch: int,
    queries: int,
    cache_size: int,
    heads: int,
    head_dim: int,
    q_patch_stride: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Unscaled masked attention; one block per (b, h, q) row.

    ``q`` is read from the packed QKV GEMM output via ``q_patch_stride``; k/v
    are the contiguous per-layer decode caches.
    """

    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "timesfm_attention_f32", _ATTENTION, ctypes.c_int)
    err = fn(
        q_ptr, k_ptr, v_ptr, num_masked_ptr, q_offset_ptr, out_ptr,
        batch, queries, cache_size, heads, head_dim, q_patch_stride, stream,
    )
    _check_launch(runtime, err)


__all__ = [
    "build_timesfm",
    "plan_timesfm_build",
    "timesfm_add_f32",
    "timesfm_attention_f32",
    "timesfm_bias_f32",
    "timesfm_bias_swish_f32",
    "timesfm_head_perdim_scale_f32",
    "timesfm_head_rmsnorm_f32",
    "timesfm_norm_add_f32",
    "timesfm_rmsnorm_f32",
    "timesfm_rope_f32",
    "timesfm_scatter_kv_f32",
    "timesfm_swish_f32",
]
