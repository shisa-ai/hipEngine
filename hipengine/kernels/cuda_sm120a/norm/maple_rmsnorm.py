"""Raw-pointer CUDA wrappers for the Maple/PARO RMSNorm family.

The source-faithful kernels preserve the in-tree BF16 and residual boundaries.
Importing this module registers ctypes launch wrappers but does not build or load
CUDA until a wrapper is called.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_cuda, plan_cuda_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.cuda import CUDA_SUCCESS, CudaRuntime, get_cuda_runtime
from hipengine.kernels.backends import cuda_target_arch_for_backend
from hipengine.kernels.registry import KernelKey, register

# Cached argtypes tuples for the RMSNorm launchers used by the verifier.
# Shape: ptr(s) + rows/hidden_size + eps (float!) + stream.
_ARGTYPES_RMSNORM_3PTR = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_void_p,
)
_ARGTYPES_RMSNORM_5PTR = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_void_p,
)

_SOURCE = Path(__file__).with_name("maple_rmsnorm.cu")
_OUTPUT_NAME = "maple_rmsnorm.so"
_BACKEND = "cuda_sm120a"
_TARGET_ARCH = cuda_target_arch_for_backend(_BACKEND)
_SYMBOL_RMSNORM = "hipengine_qwen35_rmsnorm_bf16"
_SYMBOL_ADD_RMSNORM = "hipengine_qwen35_add_rmsnorm_bf16"
_SYMBOL_ADD_RMSNORM_F32 = "hipengine_qwen35_add_rmsnorm_f32_bf16"
_SYMBOL_HEAD_RMSNORM = "hipengine_qwen35_head_rmsnorm_f32_bf16"
_SYMBOL_PARO_RMSNORM_OUT = "hipengine_paro_rmsnorm_out_bf16"
_SYMBOL_PARO_ADD_RMSNORM_OUT = "hipengine_paro_add_rmsnorm_out_bf16"
_SYMBOL_PARO_RMSNORM_OUT_FP16 = "hipengine_paro_rmsnorm_out_fp16"
_SYMBOL_PARO_ADD_RMSNORM_OUT_FP16 = "hipengine_paro_add_rmsnorm_out_fp16"


def plan_qwen35_rmsnorm_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_cuda_build(
        sources=[_SOURCE],
        family="cuda_sm120a_maple_rmsnorm",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        output_name=_OUTPUT_NAME,
    )


def build_qwen35_rmsnorm(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_cuda(
        sources=[_SOURCE],
        family="cuda_sm120a_maple_rmsnorm",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def qwen35_rmsnorm_bf16(
    hidden_states_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    hidden_size: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Launch ``qwen35_rmsnorm_kernel`` for BF16-bit input/weight/output buffers."""

    _check_positive_shape(rows, hidden_size, "rows", "hidden_size")
    library = library or build_qwen35_rmsnorm(load=True)
    runtime = runtime or get_cuda_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_RMSNORM, _ARGTYPES_RMSNORM_3PTR, ctypes.c_int)
    err = fn(hidden_states_ptr, weight_ptr, out_ptr,
             rows, hidden_size, float(eps), stream)
    _check_launch(runtime, err)


def qwen35_add_rmsnorm_bf16(
    hidden_states_ptr: int,
    residual_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    residual_out_ptr: int,
    rows: int,
    hidden_size: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Launch BF16 residual-add + RMSNorm, writing normalized and residual outputs."""

    _check_positive_shape(rows, hidden_size, "rows", "hidden_size")
    library = library or build_qwen35_rmsnorm(load=True)
    runtime = runtime or get_cuda_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_ADD_RMSNORM, _ARGTYPES_RMSNORM_5PTR, ctypes.c_int)
    err = fn(hidden_states_ptr, residual_ptr, weight_ptr, out_ptr, residual_out_ptr,
             rows, hidden_size, float(eps), stream)
    _check_launch(runtime, err)


def qwen35_add_rmsnorm_f32_bf16(
    hidden_states_ptr: int,
    residual_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    residual_out_ptr: int,
    rows: int,
    hidden_size: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Launch F32 hidden + BF16 residual RMSNorm, writing BF16-bit outputs."""

    _check_positive_shape(rows, hidden_size, "rows", "hidden_size")
    library = library or build_qwen35_rmsnorm(load=True)
    runtime = runtime or get_cuda_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_ADD_RMSNORM_F32, _ARGTYPES_RMSNORM_5PTR, ctypes.c_int)
    err = fn(hidden_states_ptr, residual_ptr, weight_ptr, out_ptr, residual_out_ptr,
             rows, hidden_size, float(eps), stream)
    _check_launch(runtime, err)


def qwen35_head_rmsnorm_f32_bf16(
    hidden_states_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    heads: int,
    head_dim: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Launch F32 per-head RMSNorm with BF16-bit weight deltas and F32 output."""

    _check_positive_shape(heads, head_dim, "heads", "head_dim")
    library = library or build_qwen35_rmsnorm(load=True)
    runtime = runtime or get_cuda_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_HEAD_RMSNORM, _ARGTYPES_RMSNORM_3PTR, ctypes.c_int)
    err = fn(hidden_states_ptr, weight_ptr, out_ptr, heads, head_dim, float(eps), stream)
    _check_launch(runtime, err)


def paro_rmsnorm_out_bf16(
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    hidden_size: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Launch PARO BF16 RMSNorm into a caller-owned output buffer.

    Unlike the Qwen3.5 delta-weight kernels, PARO norm weights are direct scale values.
    """

    _check_positive_shape(rows, hidden_size, "rows", "hidden_size")
    library = library or build_qwen35_rmsnorm(load=True)
    runtime = runtime or get_cuda_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_PARO_RMSNORM_OUT, _ARGTYPES_RMSNORM_3PTR, ctypes.c_int)
    err = fn(x_ptr, weight_ptr, out_ptr, rows, hidden_size, float(eps), stream)
    _check_launch(runtime, err)


def paro_add_rmsnorm_out_bf16(
    x_ptr: int,
    add_ptr: int,
    weight_ptr: int,
    norm_out_ptr: int,
    residual_out_ptr: int,
    rows: int,
    hidden_size: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Launch PARO BF16 residual-add + RMSNorm into caller-owned output buffers."""

    _check_positive_shape(rows, hidden_size, "rows", "hidden_size")
    library = library or build_qwen35_rmsnorm(load=True)
    runtime = runtime or get_cuda_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_PARO_ADD_RMSNORM_OUT, _ARGTYPES_RMSNORM_5PTR, ctypes.c_int)
    err = fn(x_ptr, add_ptr, weight_ptr, norm_out_ptr, residual_out_ptr,
             rows, hidden_size, float(eps), stream)
    _check_launch(runtime, err)


def paro_rmsnorm_out_fp16(
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    hidden_size: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Launch PARO FP16 RMSNorm into a caller-owned output buffer."""

    _check_positive_shape(rows, hidden_size, "rows", "hidden_size")
    library = library or build_qwen35_rmsnorm(load=True)
    runtime = runtime or get_cuda_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_PARO_RMSNORM_OUT_FP16, _ARGTYPES_RMSNORM_3PTR, ctypes.c_int)
    err = fn(x_ptr, weight_ptr, out_ptr, rows, hidden_size, float(eps), stream)
    _check_launch(runtime, err)


def paro_add_rmsnorm_out_fp16(
    x_ptr: int,
    add_ptr: int,
    weight_ptr: int,
    norm_out_ptr: int,
    residual_out_ptr: int,
    rows: int,
    hidden_size: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Launch PARO FP16 residual-add + RMSNorm into caller-owned output buffers."""

    _check_positive_shape(rows, hidden_size, "rows", "hidden_size")
    library = library or build_qwen35_rmsnorm(load=True)
    runtime = runtime or get_cuda_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_PARO_ADD_RMSNORM_OUT_FP16, _ARGTYPES_RMSNORM_5PTR, ctypes.c_int)
    err = fn(x_ptr, add_ptr, weight_ptr, norm_out_ptr, residual_out_ptr,
             rows, hidden_size, float(eps), stream)
    _check_launch(runtime, err)


def register_qwen35_rmsnorm_kernels(*, replace: bool = True) -> None:
    register(KernelKey(_BACKEND, "rmsnorm", "bf16"), qwen35_rmsnorm_bf16, replace=replace)
    register(
        KernelKey(_BACKEND, "add_rmsnorm", "bf16"),
        qwen35_add_rmsnorm_bf16,
        replace=replace,
    )
    register(
        KernelKey(_BACKEND, "add_rmsnorm_f32", "bf16"),
        qwen35_add_rmsnorm_f32_bf16,
        replace=replace,
    )
    register(
        KernelKey(_BACKEND, "head_rmsnorm", "bf16"),
        qwen35_head_rmsnorm_f32_bf16,
        replace=replace,
    )
    for quant in ("bf16", "w4_paro"):
        register(
            KernelKey(_BACKEND, "rmsnorm", quant, "paro_out"),
            paro_rmsnorm_out_bf16,
            replace=replace,
        )
        register(
            KernelKey(_BACKEND, "add_rmsnorm", quant, "paro_out"),
            paro_add_rmsnorm_out_bf16,
            replace=replace,
        )
    register(
        KernelKey(_BACKEND, "rmsnorm", "w4_paro", "paro_out_fp16"),
        paro_rmsnorm_out_fp16,
        replace=replace,
    )
    register(
        KernelKey(_BACKEND, "add_rmsnorm", "w4_paro", "paro_out_fp16"),
        paro_add_rmsnorm_out_fp16,
        replace=replace,
    )


def _check_positive_shape(outer: int, inner: int, outer_name: str, inner_name: str) -> None:
    if outer <= 0:
        raise ValueError(f"{outer_name} must be positive")
    if inner <= 0:
        raise ValueError(f"{inner_name} must be positive")


def _check_launch(runtime: CudaRuntime, err: int) -> None:
    if int(err) != CUDA_SUCCESS:
        runtime.check(int(err))


register_qwen35_rmsnorm_kernels()
