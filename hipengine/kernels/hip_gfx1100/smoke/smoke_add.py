"""Lazy wrapper and registry entry for the first HIP smoke kernel.

Importing this module registers the wrapper but does not invoke hipcc, load ROCm, or touch the
GPU. ``plan_smoke_add_build`` is safe in CPU-only CI.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("smoke_add.hip")
_SYMBOL = "hipengine_smoke_add_f32"


def plan_smoke_add_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="smoke",
        profile="baseline",
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name="smoke_add.so",
    )


def build_smoke_add(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_hip(
        sources=[_SOURCE],
        family="smoke",
        profile="baseline",
        cache_root=cache_root,
        compiler_version=compiler_version,
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
    runtime: HipRuntime | None = None,
) -> None:
    """Launch smoke_add through the C ABI wrapper.

    This is the first GPU-touching function in the smoke path. It is not called by CPU-only
    tests; the real smoke run happens after the user clears the GPU.
    """

    library = library or build_smoke_add(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(n),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_smoke_add_kernel(*, replace: bool = True) -> None:
    register(KernelKey("hip_gfx1100", "smoke_add", "fp16"), smoke_add_f32, replace=replace)


register_smoke_add_kernel()
