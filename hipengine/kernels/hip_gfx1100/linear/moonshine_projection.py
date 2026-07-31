"""Raw-pointer Moonshine source-F16 projection baselines."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("moonshine_projection.hip")
_OUTPUT_NAME = "moonshine_projection.so"
_ALLOWED_THREADS = {32, 64, 128, 256}
_SINGLE_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_BIAS_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_PAIR_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_TRIPLE_ARGS = (
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
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)


def plan_moonshine_projection_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="moonshine_projection",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_moonshine_projection(
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
        family="moonshine_projection",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _check_launch(runtime: HipRuntime, error: int) -> None:
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


def _validate(
    rows: int,
    in_features: int,
    outputs: tuple[int, ...],
    threads: int,
) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if any(width <= 0 for width in outputs):
        raise ValueError("out_features must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 32, 64, 128, 256")


def moonshine_f16_projection(
    input_ptr: int,
    weight_ptr: int,
    output_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate(rows, in_features, (out_features,), threads)
    library = library or build_moonshine_projection(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_moonshine_f16_projection",
        _SINGLE_ARGS,
        ctypes.c_int,
    )
    error = fn(
        input_ptr,
        weight_ptr,
        output_ptr,
        rows,
        in_features,
        out_features,
        threads,
        stream,
    )
    _check_launch(runtime, error)


def moonshine_f16_projection_bias(
    input_ptr: int,
    weight_ptr: int,
    bias_ptr: int,
    output_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate(rows, in_features, (out_features,), threads)
    library = library or build_moonshine_projection(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_moonshine_f16_projection_bias",
        _BIAS_ARGS,
        ctypes.c_int,
    )
    error = fn(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        rows,
        in_features,
        out_features,
        threads,
        stream,
    )
    _check_launch(runtime, error)


def moonshine_f16_projection_pair(
    input_ptr: int,
    weight_a_ptr: int,
    weight_b_ptr: int,
    output_a_ptr: int,
    output_b_ptr: int,
    rows: int,
    in_features: int,
    out_a_features: int,
    out_b_features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate(rows, in_features, (out_a_features, out_b_features), threads)
    library = library or build_moonshine_projection(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_moonshine_f16_projection_pair",
        _PAIR_ARGS,
        ctypes.c_int,
    )
    error = fn(
        input_ptr,
        weight_a_ptr,
        weight_b_ptr,
        output_a_ptr,
        output_b_ptr,
        rows,
        in_features,
        out_a_features,
        out_b_features,
        threads,
        stream,
    )
    _check_launch(runtime, error)


def moonshine_f16_projection_triple(
    input_ptr: int,
    weight_a_ptr: int,
    weight_b_ptr: int,
    weight_c_ptr: int,
    output_a_ptr: int,
    output_b_ptr: int,
    output_c_ptr: int,
    rows: int,
    in_features: int,
    out_a_features: int,
    out_b_features: int,
    out_c_features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate(
        rows,
        in_features,
        (out_a_features, out_b_features, out_c_features),
        threads,
    )
    library = library or build_moonshine_projection(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_moonshine_f16_projection_triple",
        _TRIPLE_ARGS,
        ctypes.c_int,
    )
    error = fn(
        input_ptr,
        weight_a_ptr,
        weight_b_ptr,
        weight_c_ptr,
        output_a_ptr,
        output_b_ptr,
        output_c_ptr,
        rows,
        in_features,
        out_a_features,
        out_b_features,
        out_c_features,
        threads,
        stream,
    )
    _check_launch(runtime, error)


def register_moonshine_projection_kernels(*, replace: bool = True) -> None:
    registrations = (
        (
            KernelKey(
                "hip_gfx1100",
                "moonshine_projection",
                "fp16",
                "single_fp32_accum",
            ),
            moonshine_f16_projection,
        ),
        (
            KernelKey(
                "hip_gfx1100",
                "moonshine_projection_rows",
                "fp16",
                "single_fp32_accum",
            ),
            moonshine_f16_projection,
        ),
        (
            KernelKey(
                "hip_gfx1100",
                "moonshine_projection_bias",
                "fp16",
                "single_fp32_accum",
            ),
            moonshine_f16_projection_bias,
        ),
        (
            KernelKey(
                "hip_gfx1100",
                "moonshine_projection_pair",
                "fp16",
                "pair_fp32_accum",
            ),
            moonshine_f16_projection_pair,
        ),
        (
            KernelKey(
                "hip_gfx1100",
                "moonshine_qkv_proj",
                "fp16",
                "triple_fp32_accum",
            ),
            moonshine_f16_projection_triple,
        ),
    )
    for key, kernel in registrations:
        register(key, kernel, replace=replace)


register_moonshine_projection_kernels()

__all__ = [
    "build_moonshine_projection",
    "moonshine_f16_projection",
    "moonshine_f16_projection_bias",
    "moonshine_f16_projection_pair",
    "moonshine_f16_projection_triple",
    "plan_moonshine_projection_build",
    "register_moonshine_projection_kernels",
]
