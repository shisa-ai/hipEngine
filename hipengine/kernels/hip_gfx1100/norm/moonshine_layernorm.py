"""Raw-pointer Moonshine FP16 LayerNorm with FP32 statistics."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("moonshine_layernorm.hip")
_OUTPUT_NAME = "moonshine_layernorm.so"
_ALLOWED_THREADS = {32, 64, 128, 256}
_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_int64,
    ctypes.c_void_p,
)


def plan_moonshine_layernorm_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="moonshine_layernorm",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_moonshine_layernorm(
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
        family="moonshine_layernorm",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def moonshine_layernorm_fp16(
    input_ptr: int,
    weight_ptr: int,
    output_ptr: int,
    rows: int,
    hidden_size: int,
    *,
    eps: float = 1.0e-5,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 32, 64, 128, 256")
    library = library or build_moonshine_layernorm(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_moonshine_layernorm_fp16",
        _ARGS,
        ctypes.c_int,
    )
    error = fn(
        input_ptr,
        weight_ptr,
        output_ptr,
        rows,
        hidden_size,
        eps,
        threads,
        stream,
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def register_moonshine_layernorm_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey(
            "hip_gfx1100",
            "moonshine_layernorm",
            "fp16",
            "fp32_stats",
        ),
        moonshine_layernorm_fp16,
        replace=replace,
    )


register_moonshine_layernorm_kernels()

__all__ = [
    "build_moonshine_layernorm",
    "moonshine_layernorm_fp16",
    "plan_moonshine_layernorm_build",
    "register_moonshine_layernorm_kernels",
]
