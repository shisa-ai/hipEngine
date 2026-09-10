"""Raw-pointer launchers for the TimesFM 3.0 HIP kernel family.

Importing this module registers ctypes launch wrappers but does not build or
load ROCm until a wrapper is called.  Kernel semantics mirror the NumPy CPU
reference in ``hipengine/kernels/cpu_reference/timesfm3.py`` (see the .hip
header for the pieces that differ from the 2.5 family).
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime

_SOURCE = Path(__file__).with_name("timesfm3.hip")
_OUTPUT_NAME = "timesfm3.so"

_FAMILY = "timesfm3"

_1PTR_COUNT = (ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p)

_QKV_SCATTER = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_float,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
)

_VAR_ATTENTION = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_void_p,
)


def plan_timesfm3_build(
    *, cache_root: str | Path | None = None, profile: ProfileName = "decode"
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family=_FAMILY,
        profile=profile,
        cache_root=cache_root,
        output_name=_OUTPUT_NAME,
    )


def build_timesfm3(
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
    return build_timesfm3(load=True)


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def timesfm3_relu(
    x_ptr: int,
    count: int,
    *,
    dtype: str = "f16",
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """out = relu(x) elementwise, in place."""

    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, f"timesfm3_relu_{dtype}", _1PTR_COUNT, ctypes.c_int)
    err = fn(x_ptr, count, stream)
    _check_launch(runtime, err)


def timesfm3_qkv_norm_scatter_f16(
    qkv_ptr: int,
    pos_ptr: int,
    timescale_ptr: int,
    qscale_ptr: int,
    k_ln_ptr: int,
    batch: int,
    n: int,
    cache_size: int,
    heads: int,
    head_dim: int,
    patch_stride: int,
    start: int,
    eps: float,
    qt_ptr: int,
    cache_k_ptr: int,
    cache_v_ptr: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Fused rope + QK rmsnorm (eps) + per-dim scale + head-major scatter."""

    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "timesfm3_qkv_norm_scatter_f16", _QKV_SCATTER, ctypes.c_int
    )
    err = fn(
        qkv_ptr, pos_ptr, timescale_ptr, qscale_ptr, k_ln_ptr,
        batch, n, cache_size, heads, head_dim, patch_stride, start, eps,
        qt_ptr, cache_k_ptr, cache_v_ptr, stream,
    )
    _check_launch(runtime, err)


def timesfm3_var_attention(
    q_ptr: int,
    k_ptr: int,
    v_ptr: int,
    front_masked_ptr: int,
    out_ptr: int,
    batch: int,
    variates: int,
    n: int,
    heads: int,
    head_dim: int,
    *,
    dtype: str = "f16",
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Non-causal variate attention over the variate axis.

    q/k/v/out are [batch*variates, n, heads*head_dim] row-major with row
    index ((b*V + v)*N + n); ``front_masked`` is [batch*variates] int32.
    Fully-masked query rows produce zeros.
    """

    if dtype not in ("f16", "f32"):
        raise ValueError("dtype must be 'f16' or 'f32'")
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, f"timesfm3_var_attention_{dtype}", _VAR_ATTENTION, ctypes.c_int
    )
    err = fn(
        q_ptr, k_ptr, v_ptr, front_masked_ptr, out_ptr,
        batch, variates, n, heads, head_dim, stream,
    )
    _check_launch(runtime, err)


__all__ = [
    "build_timesfm3",
    "plan_timesfm3_build",
    "timesfm3_qkv_norm_scatter_f16",
    "timesfm3_relu",
    "timesfm3_var_attention",
]
