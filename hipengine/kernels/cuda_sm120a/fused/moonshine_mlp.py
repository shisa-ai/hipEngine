"""Raw-pointer Moonshine decoder gated-SiLU FP16 primitive for CUDA ``sm_120a``."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_cuda, plan_cuda_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.cuda import CUDA_SUCCESS, CudaRuntime, get_cuda_runtime
from hipengine.kernels.backends import cuda_target_arch_for_backend
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("moonshine_mlp.cu")
_OUTPUT_NAME = "moonshine_mlp.so"
_BACKEND = "cuda_sm120a"
_TARGET_ARCH = cuda_target_arch_for_backend(_BACKEND)
_ALLOWED_THREADS = {32, 64, 128, 256}
_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)


def plan_moonshine_mlp_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_cuda_build(
        sources=[_SOURCE],
        family="cuda_sm120a_moonshine_mlp",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        output_name=_OUTPUT_NAME,
    )


def build_moonshine_mlp(
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
        family="cuda_sm120a_moonshine_mlp",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def moonshine_gated_silu_fp16(
    fc1_output_ptr: int,
    output_ptr: int,
    rows: int,
    intermediate_size: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if intermediate_size <= 0:
        raise ValueError("intermediate_size must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 32, 64, 128, 256")
    library = library or build_moonshine_mlp(load=True)
    runtime = runtime or get_cuda_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_cuda_sm120a_moonshine_gated_silu_fp16",
        _ARGS,
        ctypes.c_int,
    )
    error = fn(
        fc1_output_ptr,
        output_ptr,
        rows,
        intermediate_size,
        threads,
        stream,
    )
    if int(error) != CUDA_SUCCESS:
        runtime.check(int(error))


def register_moonshine_mlp_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey(
            _BACKEND,
            "moonshine_gated_silu",
            "fp16",
            "value_gate_split",
        ),
        moonshine_gated_silu_fp16,
        replace=replace,
    )


register_moonshine_mlp_kernels()

__all__ = [
    "build_moonshine_mlp",
    "moonshine_gated_silu_fp16",
    "plan_moonshine_mlp_build",
    "register_moonshine_mlp_kernels",
]
