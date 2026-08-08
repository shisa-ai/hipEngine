"""Lazy wrapper for the CUDA ``sm_120a`` build/launch smoke."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, build_cuda, plan_cuda_build
from hipengine.core.cuda import CudaRuntime, get_cuda_runtime
from hipengine.kernels.backends import cuda_target_arch_for_backend
from hipengine.kernels.registry import KernelKey, register

BACKEND = "cuda_sm120a"
TARGET_ARCH = cuda_target_arch_for_backend(BACKEND)
_SOURCE = Path(__file__).with_name("smoke_add.cu")
_SYMBOL = "hipengine_cuda_sm120a_smoke_add_f32"


def plan_smoke_add_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    target_arch: str = TARGET_ARCH,
) -> BuildArtifact:
    return plan_cuda_build(
        sources=[_SOURCE],
        family="cuda_sm120a_smoke",
        profile="baseline",
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=target_arch,
        output_name="smoke_add.so",
    )


def build_smoke_add(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    target_arch: str = TARGET_ARCH,
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_cuda(
        sources=[_SOURCE],
        family="cuda_sm120a_smoke",
        profile="baseline",
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=target_arch,
        output_name="smoke_add.so",
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def smoke_add_f32(
    a_ptr: int,
    b_ptr: int,
    out_ptr: int,
    n: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Launch the raw-pointer CUDA C ABI smoke kernel."""

    library = library or build_smoke_add(load=True)
    runtime = runtime or get_cuda_runtime()
    function = getattr(library, _SYMBOL)
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    runtime.check(
        function(
            ctypes.c_void_p(a_ptr),
            ctypes.c_void_p(b_ptr),
            ctypes.c_void_p(out_ptr),
            ctypes.c_int64(n),
            ctypes.c_void_p(stream),
        )
    )


def register_smoke_add_kernel(*, replace: bool = True) -> None:
    register(KernelKey(BACKEND, "smoke_add", "fp32"), smoke_add_f32, replace=replace)


register_smoke_add_kernel()
