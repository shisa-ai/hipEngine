"""Raw-pointer wrappers for PARO pairwise rotation kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("paro_rotate.hip")
_OUTPUT_NAME = "paro_rotate.so"
_SYMBOL_ROTATE2 = "hipengine_paro_rotate2_bf16"
_SYMBOL_ROTATE3 = "hipengine_paro_rotate3_bf16"


def plan_paro_rotate_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="paro_rotate",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_paro_rotate(
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
        family="paro_rotate",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def paro_rotate2_bf16(
    x_ptr: int,
    out0_ptr: int,
    out1_ptr: int,
    pairs0_ptr: int,
    pairs1_ptr: int,
    theta0_ptr: int,
    theta1_ptr: int,
    scales0_ptr: int,
    scales1_ptr: int,
    tokens: int,
    hidden: int,
    group_size: int,
    krot: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch parent PARO two-output pairwise rotation kernel for BF16 buffers."""

    _check_rotate_shape(tokens, hidden, group_size, krot)
    library = library or build_paro_rotate(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_ROTATE2)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(out0_ptr),
        ctypes.c_void_p(out1_ptr),
        ctypes.c_void_p(pairs0_ptr),
        ctypes.c_void_p(pairs1_ptr),
        ctypes.c_void_p(theta0_ptr),
        ctypes.c_void_p(theta1_ptr),
        ctypes.c_void_p(scales0_ptr),
        ctypes.c_void_p(scales1_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(hidden),
        ctypes.c_int64(group_size),
        ctypes.c_int64(krot),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def paro_rotate3_bf16(
    x_ptr: int,
    out0_ptr: int,
    out1_ptr: int,
    out2_ptr: int,
    pairs0_ptr: int,
    pairs1_ptr: int,
    pairs2_ptr: int,
    theta0_ptr: int,
    theta1_ptr: int,
    theta2_ptr: int,
    scales0_ptr: int,
    scales1_ptr: int,
    scales2_ptr: int,
    tokens: int,
    hidden: int,
    group_size: int,
    krot: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch parent PARO three-output pairwise rotation kernel for BF16 buffers."""

    _check_rotate_shape(tokens, hidden, group_size, krot)
    library = library or build_paro_rotate(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_ROTATE3)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(out0_ptr),
        ctypes.c_void_p(out1_ptr),
        ctypes.c_void_p(out2_ptr),
        ctypes.c_void_p(pairs0_ptr),
        ctypes.c_void_p(pairs1_ptr),
        ctypes.c_void_p(pairs2_ptr),
        ctypes.c_void_p(theta0_ptr),
        ctypes.c_void_p(theta1_ptr),
        ctypes.c_void_p(theta2_ptr),
        ctypes.c_void_p(scales0_ptr),
        ctypes.c_void_p(scales1_ptr),
        ctypes.c_void_p(scales2_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(hidden),
        ctypes.c_int64(group_size),
        ctypes.c_int64(krot),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def register_paro_rotate_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "paro_rotate2", "w4_paro", "bf16"),
        paro_rotate2_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paro_rotate3", "w4_paro", "bf16"),
        paro_rotate3_bf16,
        replace=replace,
    )


def _check_rotate_shape(tokens: int, hidden: int, group_size: int, krot: int) -> None:
    _check_positive(tokens, "tokens")
    _check_positive(hidden, "hidden")
    _check_positive(group_size, "group_size")
    if krot < 0:
        raise ValueError("krot must be non-negative")
    if group_size % 2 != 0:
        raise ValueError("group_size must be even")
    if hidden % group_size != 0:
        raise ValueError("hidden must be divisible by group_size")
    if group_size // 2 > 1024:
        raise ValueError("group_size / 2 must fit in one HIP block")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_paro_rotate_kernels()
