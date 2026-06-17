"""Raw-pointer + numpy wrappers for native MTP NextN draft-head kernels (GGUF path).

M3 deliverable: a *real* GPU NextN ``eh_proj`` sub-kernel registered under
``KernelKey(backend, "mtp_nextn_eh_proj", "gguf_f32", "qwen35")`` for both
``hip_gfx1100`` and ``hip_gfx1151``.  Without this, the registry silently falls
back to the ``cpu_reference`` numpy oracle (``registry._candidate_keys`` appends
``cpu_reference`` last), so M3 had no native runtime kernel.

These F32 kernels are correctness-first and size-agnostic: they mirror
``cpu_reference`` math exactly so the M3 fixture gate runs on a real GPU.  M6
swaps the inner GEMVs for WMMA / K-quant tuned kernels on real shapes; these
remain the correctness baseline.

Importing this module registers the wrappers but does not build or load ROCm
until a wrapper is called.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("mtp_nextn.hip")
_OUTPUT_NAME = "mtp_nextn.so"
_SYMBOL_RMSNORM_F32 = "hipengine_mtp_rmsnorm_f32"
_SYMBOL_EH_PROJ_F32 = "hipengine_mtp_eh_proj_f32"

# ptr(s) + rows(int64) + hidden(int64) + eps(float) + stream
_ARGTYPES_RMSNORM_F32 = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_void_p,
)
# ptr(s) + rows(int64) + hidden(int64) + stream
_ARGTYPES_EH_PROJ_F32 = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p,
)


def plan_mtp_nextn_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="mtp_nextn",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_mtp_nextn(
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
        family="mtp_nextn",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _check_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def mtp_rmsnorm_f32(
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    hidden: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch F32 RMSNorm, one block per row.  ``out = x * rsqrt(mean(x^2)+eps) * weight``."""

    _check_positive("rows", rows)
    _check_positive("hidden", hidden)
    library = library or build_mtp_nextn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_RMSNORM_F32, _ARGTYPES_RMSNORM_F32, ctypes.c_int)
    err = fn(x_ptr, weight_ptr, out_ptr, rows, hidden, float(eps), stream)
    _check_launch(runtime, err)


def mtp_eh_proj_f32(
    e_norm_ptr: int,
    h_norm_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch F32 eh_proj GEMV.

    ``out[row, j] = sum_k e_norm[row,k]*weight[j,k] + h_norm[row,k]*weight[j,k+hidden]``
    with ``weight`` row-major ``[hidden, 2*hidden]`` (matches ``fused @ weight.T``).
    """

    _check_positive("rows", rows)
    _check_positive("hidden", hidden)
    library = library or build_mtp_nextn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_EH_PROJ_F32, _ARGTYPES_EH_PROJ_F32, ctypes.c_int)
    err = fn(e_norm_ptr, h_norm_ptr, weight_ptr, out_ptr, rows, hidden, stream)
    _check_launch(runtime, err)


def qwen35_gguf_mtp_eh_proj_f32(
    hidden_seed: "np.ndarray",
    token_embedding: "np.ndarray",
    eh_proj_weight: "np.ndarray",
    hnorm_weight: "np.ndarray",
    enorm_weight: "np.ndarray",
    eps: float = 1e-6,
) -> np.ndarray:
    """Numpy-in/out wrapper matching ``cpu_reference.qwen35_gguf_mtp_eh_proj``.

    enorm(embed) + hnorm(hidden) -> eh_proj F32 GEMV.  Returns ``[rows, hidden]`` F32.
    """

    hidden_arr = np.ascontiguousarray(hidden_seed, dtype=np.float32)
    embed_arr = np.ascontiguousarray(token_embedding, dtype=np.float32)
    weight = np.ascontiguousarray(eh_proj_weight, dtype=np.float32)
    hnorm = np.ascontiguousarray(hnorm_weight, dtype=np.float32)
    enorm = np.ascontiguousarray(enorm_weight, dtype=np.float32)
    if hidden_arr.ndim != 2:
        raise ValueError("hidden_seed must have shape [rows, hidden]")
    rows, hidden = hidden_arr.shape
    if embed_arr.shape != hidden_arr.shape:
        raise ValueError("token_embedding must match hidden_seed shape")
    if weight.shape != (hidden, hidden * 2):
        raise ValueError(
            f"eh_proj_weight must have shape [hidden, 2*hidden]=[{hidden}, {hidden * 2}]; "
            f"got {weight.shape}"
        )
    if hnorm.shape != (hidden,):
        raise ValueError("hnorm_weight must have shape [hidden]")
    if enorm.shape != (hidden,):
        raise ValueError("enorm_weight must have shape [hidden]")

    runtime = get_hip_runtime()
    e_norm_dev = malloc(embed_arr.nbytes, runtime=runtime)
    h_norm_dev = malloc(hidden_arr.nbytes, runtime=runtime)
    out_dev = malloc(hidden_arr.nbytes, runtime=runtime)
    buffers = [e_norm_dev, h_norm_dev, out_dev]
    try:
        embed_dev = malloc(embed_arr.nbytes, runtime=runtime); buffers.append(embed_dev)
        hidden_dev = malloc(hidden_arr.nbytes, runtime=runtime); buffers.append(hidden_dev)
        weight_dev = malloc(weight.nbytes, runtime=runtime); buffers.append(weight_dev)
        hnorm_dev = malloc(hnorm.nbytes, runtime=runtime); buffers.append(hnorm_dev)
        enorm_dev = malloc(enorm.nbytes, runtime=runtime); buffers.append(enorm_dev)
        copy_host_to_device(embed_dev, host_array_ptr(embed_arr), runtime=runtime)
        copy_host_to_device(hidden_dev, host_array_ptr(hidden_arr), runtime=runtime)
        copy_host_to_device(weight_dev, host_array_ptr(weight), runtime=runtime)
        copy_host_to_device(hnorm_dev, host_array_ptr(hnorm), runtime=runtime)
        copy_host_to_device(enorm_dev, host_array_ptr(enorm), runtime=runtime)
        # h_norm = rmsnorm(hidden, hnorm); e_norm = rmsnorm(embed, enorm)
        mtp_rmsnorm_f32(embed_dev.ptr, enorm_dev.ptr, e_norm_dev.ptr, rows, hidden, eps=eps,
                        runtime=runtime)
        mtp_rmsnorm_f32(hidden_dev.ptr, hnorm_dev.ptr, h_norm_dev.ptr, rows, hidden, eps=eps,
                        runtime=runtime)
        mtp_eh_proj_f32(e_norm_dev.ptr, h_norm_dev.ptr, weight_dev.ptr, out_dev.ptr,
                        rows, hidden, runtime=runtime)
        runtime.device_synchronize()
        out = np.empty((rows, hidden), dtype=np.float32)
        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
        return out
    finally:
        for buf in buffers:
            free(buf, runtime=runtime)


def register_mtp_nextn_kernels(*, replace: bool = True) -> None:
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        register(
            KernelKey(backend, "mtp_nextn_eh_proj", "gguf_f32", "qwen35"),
            qwen35_gguf_mtp_eh_proj_f32,
            replace=replace,
        )


register_mtp_nextn_kernels()


__all__ = [
    "build_mtp_nextn",
    "mtp_eh_proj_f32",
    "mtp_rmsnorm_f32",
    "plan_mtp_nextn_build",
    "qwen35_gguf_mtp_eh_proj_f32",
    "register_mtp_nextn_kernels",
]
