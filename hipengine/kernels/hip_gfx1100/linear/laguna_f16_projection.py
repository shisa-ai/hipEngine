"""Raw-pointer wrappers for Laguna source-F16 mixed dense projections."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("laguna_f16_projection.hip")
_OUTPUT_NAME = "laguna_f16_projection.so"
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
_DUAL_ARGS = (
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
_QUAD_ARGS = (
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
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_OUTPUT_ADD_RMSNORM_ARGS = (
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
    ctypes.c_float,
    ctypes.c_int64,
    ctypes.c_void_p,
)


def plan_laguna_f16_projection_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="laguna_f16_projection",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_laguna_f16_projection(
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
        family="laguna_f16_projection",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def build_laguna_f16_projection_prefill(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    """Build the rows>1 tiled variants with the dedicated prefill profile."""

    return build_laguna_f16_projection(
        cache_root=cache_root,
        compiler_version=compiler_version,
        profile="prefill",
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _single(
    symbol,
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    *,
    threads,
    stream,
    library,
    runtime,
):
    _validate(rows, in_features, (out_features,), threads)
    library = library or build_laguna_f16_projection(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, symbol, _SINGLE_ARGS, ctypes.c_int)
    err = fn(x_ptr, weight_ptr, out_ptr, rows, in_features, out_features, threads, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def laguna_f16w_gemv_bf16_f32_out(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _single(
        "hipengine_laguna_f16w_gemv_bf16_f32_out",
        x_ptr,
        weight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_f16w_gemv_bf16_bf16_out(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _single(
        "hipengine_laguna_f16w_gemv_bf16_bf16_out",
        x_ptr,
        weight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_f16w_onebarrier_gemv_bf16_f32_out(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _validate_decode(rows, in_features, (out_features,), threads)
    _single(
        "hipengine_laguna_f16w_onebarrier_gemv_bf16_f32_out",
        x_ptr,
        weight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_f16w_onebarrier_gemv_bf16_bf16_out(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _validate_decode(rows, in_features, (out_features,), threads)
    _single(
        "hipengine_laguna_f16w_onebarrier_gemv_bf16_bf16_out",
        x_ptr,
        weight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_f16w_fixedk_onebarrier_gemv_bf16_f32_out(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _validate_fixedk_decode(rows, in_features, (out_features,), threads)
    _single(
        "hipengine_laguna_f16w_fixedk_onebarrier_gemv_bf16_f32_out",
        x_ptr,
        weight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_f16w_fixedk_onebarrier_gemv_bf16_bf16_out(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _validate_fixedk_decode(rows, in_features, (out_features,), threads)
    _single(
        "hipengine_laguna_f16w_fixedk_onebarrier_gemv_bf16_bf16_out",
        x_ptr,
        weight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_f16w_fixedk_nontemporal_gemv_bf16_f32_out(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _validate_fixedk_decode(rows, in_features, (out_features,), threads)
    _single(
        "hipengine_laguna_f16w_fixedk_nontemporal_gemv_bf16_f32_out",
        x_ptr,
        weight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_f16w_fixedk_nontemporal_gemv_bf16_bf16_out(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _validate_fixedk_decode(rows, in_features, (out_features,), threads)
    _single(
        "hipengine_laguna_f16w_fixedk_nontemporal_gemv_bf16_bf16_out",
        x_ptr,
        weight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_f16w_fixedk_output_add_rmsnorm_bf16(
    x_ptr,
    weight_ptr,
    projection_out_ptr,
    residual_ptr,
    norm_weight_ptr,
    norm_out_ptr,
    residual_out_ptr,
    completion_counter_ptr,
    rows,
    in_features,
    out_features,
    eps,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _validate_fixedk_decode(rows, in_features, (out_features,), threads)
    if in_features not in {6144, 9216} or out_features != 3072:
        raise ValueError(
            "fixed-K output add/RMSNorm requires K6144/K9216 and N3072"
        )
    if not completion_counter_ptr:
        raise ValueError("completion_counter_ptr must be nonzero")
    library = library or build_laguna_f16_projection(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_laguna_f16w_fixedk_output_add_rmsnorm_bf16",
        _OUTPUT_ADD_RMSNORM_ARGS,
        ctypes.c_int,
    )
    err = fn(
        x_ptr,
        weight_ptr,
        projection_out_ptr,
        residual_ptr,
        norm_weight_ptr,
        norm_out_ptr,
        residual_out_ptr,
        completion_counter_ptr,
        rows,
        in_features,
        out_features,
        eps,
        threads,
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def laguna_f16w_fixedk_nontemporal_output_add_rmsnorm_bf16(
    x_ptr,
    weight_ptr,
    projection_out_ptr,
    residual_ptr,
    norm_weight_ptr,
    norm_out_ptr,
    residual_out_ptr,
    completion_counter_ptr,
    rows,
    in_features,
    out_features,
    eps,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _validate_fixedk_decode(rows, in_features, (out_features,), threads)
    if in_features not in {6144, 9216} or out_features != 3072:
        raise ValueError(
            "fixed-K output add/RMSNorm requires K6144/K9216 and N3072"
        )
    if not completion_counter_ptr:
        raise ValueError("completion_counter_ptr must be nonzero")
    library = library or build_laguna_f16_projection(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_laguna_f16w_fixedk_nontemporal_output_add_rmsnorm_bf16",
        _OUTPUT_ADD_RMSNORM_ARGS,
        ctypes.c_int,
    )
    err = fn(
        x_ptr,
        weight_ptr,
        projection_out_ptr,
        residual_ptr,
        norm_weight_ptr,
        norm_out_ptr,
        residual_out_ptr,
        completion_counter_ptr,
        rows,
        in_features,
        out_features,
        eps,
        threads,
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def laguna_f16w_gemv_f32_f32_out(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _single(
        "hipengine_laguna_f16w_gemv_f32_f32_out",
        x_ptr,
        weight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_f16w_gemv_f32_bf16_out(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _single(
        "hipengine_laguna_f16w_gemv_f32_bf16_out",
        x_ptr,
        weight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_f16w_tiled_bf16_f32_out(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _single(
        "hipengine_laguna_f16w_tiled_bf16_f32_out",
        x_ptr,
        weight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library or build_laguna_f16_projection_prefill(load=True),
        runtime=runtime,
    )


def laguna_f16w_tiled_bf16_bf16_out(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _single(
        "hipengine_laguna_f16w_tiled_bf16_bf16_out",
        x_ptr,
        weight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library or build_laguna_f16_projection_prefill(load=True),
        runtime=runtime,
    )


def laguna_f16w_wmma_bf16_f32_out(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _validate_wmma(rows, in_features, (out_features,), threads)
    _single(
        "hipengine_laguna_f16w_wmma_bf16_f32_out",
        x_ptr,
        weight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library or build_laguna_f16_projection_prefill(load=True),
        runtime=runtime,
    )


def laguna_f16w_wmma_bf16_bf16_out(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _validate_wmma(rows, in_features, (out_features,), threads)
    _single(
        "hipengine_laguna_f16w_wmma_bf16_bf16_out",
        x_ptr,
        weight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library or build_laguna_f16_projection_prefill(load=True),
        runtime=runtime,
    )


def _compensated_wmma_single(
    symbol,
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    *,
    threads,
    stream,
    library,
    runtime,
):
    _validate_wmma(rows, in_features, (out_features,), threads)
    _single(
        symbol,
        x_ptr,
        weight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library or build_laguna_f16_projection_prefill(load=True),
        runtime=runtime,
    )


def laguna_f16w_wmma_comp_bf16_f32_out(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _compensated_wmma_single(
        "hipengine_laguna_f16w_wmma_comp_bf16_f32_out",
        x_ptr,
        weight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_f16w_wmma_comp_bf16_bf16_out(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _compensated_wmma_single(
        "hipengine_laguna_f16w_wmma_comp_bf16_bf16_out",
        x_ptr,
        weight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_f16w_dual_gemv_bf16_f32_out(
    x_ptr,
    weight_a_ptr,
    weight_b_ptr,
    out_a_ptr,
    out_b_ptr,
    rows,
    in_features,
    out_a_features,
    out_b_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _validate(rows, in_features, (out_a_features, out_b_features), threads)
    library = library or build_laguna_f16_projection(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_laguna_f16w_dual_gemv_bf16_f32_out", _DUAL_ARGS, ctypes.c_int
    )
    err = fn(
        x_ptr,
        weight_a_ptr,
        weight_b_ptr,
        out_a_ptr,
        out_b_ptr,
        rows,
        in_features,
        out_a_features,
        out_b_features,
        threads,
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def laguna_f16w_triple_gemv_bf16_f32_out(
    x_ptr,
    weight_a_ptr,
    weight_b_ptr,
    weight_c_ptr,
    out_a_ptr,
    out_b_ptr,
    out_c_ptr,
    rows,
    in_features,
    out_a_features,
    out_b_features,
    out_c_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _validate(rows, in_features, (out_a_features, out_b_features, out_c_features), threads)
    library = library or build_laguna_f16_projection(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_laguna_f16w_triple_gemv_bf16_f32_out", _TRIPLE_ARGS, ctypes.c_int
    )
    err = fn(
        x_ptr,
        weight_a_ptr,
        weight_b_ptr,
        weight_c_ptr,
        out_a_ptr,
        out_b_ptr,
        out_c_ptr,
        rows,
        in_features,
        out_a_features,
        out_b_features,
        out_c_features,
        threads,
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def laguna_f16w_triple_onebarrier_gemv_bf16_f32_out(
    x_ptr,
    weight_a_ptr,
    weight_b_ptr,
    weight_c_ptr,
    out_a_ptr,
    out_b_ptr,
    out_c_ptr,
    rows,
    in_features,
    out_a_features,
    out_b_features,
    out_c_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _validate_decode(
        rows,
        in_features,
        (out_a_features, out_b_features, out_c_features),
        threads,
    )
    library = library or build_laguna_f16_projection(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_laguna_f16w_triple_onebarrier_gemv_bf16_f32_out",
        _TRIPLE_ARGS,
        ctypes.c_int,
    )
    err = fn(
        x_ptr,
        weight_a_ptr,
        weight_b_ptr,
        weight_c_ptr,
        out_a_ptr,
        out_b_ptr,
        out_c_ptr,
        rows,
        in_features,
        out_a_features,
        out_b_features,
        out_c_features,
        threads,
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def laguna_f16w_triple_fixedk_onebarrier_gemv_bf16_f32_out(
    x_ptr,
    weight_a_ptr,
    weight_b_ptr,
    weight_c_ptr,
    out_a_ptr,
    out_b_ptr,
    out_c_ptr,
    rows,
    in_features,
    out_a_features,
    out_b_features,
    out_c_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _validate_fixedk_decode(
        rows,
        in_features,
        (out_a_features, out_b_features, out_c_features),
        threads,
    )
    library = library or build_laguna_f16_projection(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_laguna_f16w_triple_fixedk_onebarrier_gemv_bf16_f32_out",
        _TRIPLE_ARGS,
        ctypes.c_int,
    )
    err = fn(
        x_ptr,
        weight_a_ptr,
        weight_b_ptr,
        weight_c_ptr,
        out_a_ptr,
        out_b_ptr,
        out_c_ptr,
        rows,
        in_features,
        out_a_features,
        out_b_features,
        out_c_features,
        threads,
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def laguna_f16w_triple_fixedk_nontemporal_gemv_bf16_f32_out(
    x_ptr,
    weight_a_ptr,
    weight_b_ptr,
    weight_c_ptr,
    out_a_ptr,
    out_b_ptr,
    out_c_ptr,
    rows,
    in_features,
    out_a_features,
    out_b_features,
    out_c_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _validate_fixedk_decode(
        rows,
        in_features,
        (out_a_features, out_b_features, out_c_features),
        threads,
    )
    library = library or build_laguna_f16_projection(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_laguna_f16w_triple_fixedk_nontemporal_gemv_bf16_f32_out",
        _TRIPLE_ARGS,
        ctypes.c_int,
    )
    err = fn(
        x_ptr,
        weight_a_ptr,
        weight_b_ptr,
        weight_c_ptr,
        out_a_ptr,
        out_b_ptr,
        out_c_ptr,
        rows,
        in_features,
        out_a_features,
        out_b_features,
        out_c_features,
        threads,
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def laguna_f16w_quad_fixedk_onebarrier_gemv_bf16_f32_out(
    x_ptr,
    weight_a_ptr,
    weight_b_ptr,
    weight_c_ptr,
    weight_d_ptr,
    out_a_ptr,
    out_b_ptr,
    out_c_ptr,
    out_d_ptr,
    rows,
    in_features,
    out_a_features,
    out_b_features,
    out_c_features,
    out_d_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _validate_fixedk_decode(
        rows,
        in_features,
        (out_a_features, out_b_features, out_c_features, out_d_features),
        threads,
    )
    library = library or build_laguna_f16_projection(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_laguna_f16w_quad_fixedk_onebarrier_gemv_bf16_f32_out",
        _QUAD_ARGS,
        ctypes.c_int,
    )
    err = fn(
        x_ptr,
        weight_a_ptr,
        weight_b_ptr,
        weight_c_ptr,
        weight_d_ptr,
        out_a_ptr,
        out_b_ptr,
        out_c_ptr,
        out_d_ptr,
        rows,
        in_features,
        out_a_features,
        out_b_features,
        out_c_features,
        out_d_features,
        threads,
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def laguna_f16w_quad_fixedk_nontemporal_gemv_bf16_f32_out(
    x_ptr,
    weight_a_ptr,
    weight_b_ptr,
    weight_c_ptr,
    weight_d_ptr,
    out_a_ptr,
    out_b_ptr,
    out_c_ptr,
    out_d_ptr,
    rows,
    in_features,
    out_a_features,
    out_b_features,
    out_c_features,
    out_d_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _validate_fixedk_decode(
        rows,
        in_features,
        (out_a_features, out_b_features, out_c_features, out_d_features),
        threads,
    )
    library = library or build_laguna_f16_projection(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_laguna_f16w_quad_fixedk_nontemporal_gemv_bf16_f32_out",
        _QUAD_ARGS,
        ctypes.c_int,
    )
    err = fn(
        x_ptr,
        weight_a_ptr,
        weight_b_ptr,
        weight_c_ptr,
        weight_d_ptr,
        out_a_ptr,
        out_b_ptr,
        out_c_ptr,
        out_d_ptr,
        rows,
        in_features,
        out_a_features,
        out_b_features,
        out_c_features,
        out_d_features,
        threads,
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def laguna_f16w_triple_tiled_bf16_f32_out(
    x_ptr,
    weight_a_ptr,
    weight_b_ptr,
    weight_c_ptr,
    out_a_ptr,
    out_b_ptr,
    out_c_ptr,
    rows,
    in_features,
    out_a_features,
    out_b_features,
    out_c_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _validate(rows, in_features, (out_a_features, out_b_features, out_c_features), threads)
    library = library or build_laguna_f16_projection_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_laguna_f16w_triple_tiled_bf16_f32_out", _TRIPLE_ARGS, ctypes.c_int
    )
    err = fn(
        x_ptr,
        weight_a_ptr,
        weight_b_ptr,
        weight_c_ptr,
        out_a_ptr,
        out_b_ptr,
        out_c_ptr,
        rows,
        in_features,
        out_a_features,
        out_b_features,
        out_c_features,
        threads,
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _triple_wmma(
    symbol,
    x_ptr,
    weight_a_ptr,
    weight_b_ptr,
    weight_c_ptr,
    out_a_ptr,
    out_b_ptr,
    out_c_ptr,
    rows,
    in_features,
    out_a_features,
    out_b_features,
    out_c_features,
    *,
    threads,
    stream,
    library,
    runtime,
):
    outputs = (out_a_features, out_b_features, out_c_features)
    _validate_wmma(rows, in_features, outputs, threads)
    library = library or build_laguna_f16_projection_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, symbol, _TRIPLE_ARGS, ctypes.c_int)
    err = fn(
        x_ptr,
        weight_a_ptr,
        weight_b_ptr,
        weight_c_ptr,
        out_a_ptr,
        out_b_ptr,
        out_c_ptr,
        rows,
        in_features,
        out_a_features,
        out_b_features,
        out_c_features,
        threads,
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def laguna_f16w_triple_wmma_bf16_f32_out(
    x_ptr,
    weight_a_ptr,
    weight_b_ptr,
    weight_c_ptr,
    out_a_ptr,
    out_b_ptr,
    out_c_ptr,
    rows,
    in_features,
    out_a_features,
    out_b_features,
    out_c_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _triple_wmma(
        "hipengine_laguna_f16w_triple_wmma_bf16_f32_out",
        x_ptr,
        weight_a_ptr,
        weight_b_ptr,
        weight_c_ptr,
        out_a_ptr,
        out_b_ptr,
        out_c_ptr,
        rows,
        in_features,
        out_a_features,
        out_b_features,
        out_c_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_f16w_triple_wmma_comp_bf16_f32_out(
    x_ptr,
    weight_a_ptr,
    weight_b_ptr,
    weight_c_ptr,
    out_a_ptr,
    out_b_ptr,
    out_c_ptr,
    rows,
    in_features,
    out_a_features,
    out_b_features,
    out_c_features,
    *,
    threads=256,
    stream=0,
    library=None,
    runtime=None,
):
    _triple_wmma(
        "hipengine_laguna_f16w_triple_wmma_comp_bf16_f32_out",
        x_ptr,
        weight_a_ptr,
        weight_b_ptr,
        weight_c_ptr,
        out_a_ptr,
        out_b_ptr,
        out_c_ptr,
        rows,
        in_features,
        out_a_features,
        out_b_features,
        out_c_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _validate(rows: int, in_features: int, outputs: tuple[int, ...], threads: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if any(value <= 0 for value in outputs):
        raise ValueError("out_features must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 32, 64, 128, 256")


def _validate_decode(
    rows: int, in_features: int, outputs: tuple[int, ...], threads: int
) -> None:
    _validate(rows, in_features, outputs, threads)
    if rows != 1:
        raise ValueError("decode GEMV requires exactly one row")
    if threads != 256:
        raise ValueError("decode GEMV compatibility threads must be 256")


def _validate_fixedk_decode(
    rows: int, in_features: int, outputs: tuple[int, ...], threads: int
) -> None:
    _validate_decode(rows, in_features, outputs, threads)
    if in_features not in {3072, 6144, 9216}:
        raise ValueError("fixed-K decode GEMV requires K3072, K6144, or K9216")


def _validate_wmma(
    rows: int, in_features: int, outputs: tuple[int, ...], threads: int
) -> None:
    _validate(rows, in_features, outputs, threads)
    if in_features % 16:
        raise ValueError("WMMA in_features must be a multiple of 16")
    if threads != 256:
        raise ValueError("WMMA compatibility threads must be 256")


def register_laguna_f16_projection_kernels(*, replace: bool = True) -> None:
    variants = (
        ("bf16_f32_out", laguna_f16w_gemv_bf16_f32_out),
        ("bf16_bf16_out", laguna_f16w_gemv_bf16_bf16_out),
        ("f32_f32_out", laguna_f16w_gemv_f32_f32_out),
        ("f32_bf16_out", laguna_f16w_gemv_f32_bf16_out),
    )
    for variant, fn in variants:
        register(KernelKey("hip_gfx1100", "linear", "fp16_weight", variant), fn, replace=replace)
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "fp16_weight",
            "onebarrier_bf16_f32_out",
        ),
        laguna_f16w_onebarrier_gemv_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "fp16_weight",
            "onebarrier_bf16_bf16_out",
        ),
        laguna_f16w_onebarrier_gemv_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "fp16_weight",
            "fixedk_onebarrier_bf16_f32_out",
        ),
        laguna_f16w_fixedk_onebarrier_gemv_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "fp16_weight",
            "fixedk_onebarrier_bf16_bf16_out",
        ),
        laguna_f16w_fixedk_onebarrier_gemv_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear+add+rmsnorm",
            "fp16_weight+gguf_f32_weight",
            "fixedk_onebarrier_bf16_out",
        ),
        laguna_f16w_fixedk_output_add_rmsnorm_bf16,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear+add+rmsnorm",
            "fp16_weight+gguf_f32_weight",
            "fixedk_nontemporal_bf16_out",
        ),
        laguna_f16w_fixedk_nontemporal_output_add_rmsnorm_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "fp16_weight", "tiled_bf16_f32_out"),
        laguna_f16w_tiled_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "fp16_weight", "tiled_bf16_bf16_out"),
        laguna_f16w_tiled_bf16_bf16_out,
        replace=replace,
    )
    wmma_variants = (
        (
            "wmma",
            laguna_f16w_wmma_bf16_f32_out,
            laguna_f16w_wmma_bf16_bf16_out,
            laguna_f16w_triple_wmma_bf16_f32_out,
        ),
        (
            "wmma_comp",
            laguna_f16w_wmma_comp_bf16_f32_out,
            laguna_f16w_wmma_comp_bf16_bf16_out,
            laguna_f16w_triple_wmma_comp_bf16_f32_out,
        ),
    )
    for prefix, single_f32, single_bf16, triple_f32 in wmma_variants:
        register(
            KernelKey(
                "hip_gfx1100", "linear", "fp16_weight", f"{prefix}_bf16_f32_out"
            ),
            single_f32,
            replace=replace,
        )
        register(
            KernelKey(
                "hip_gfx1100", "linear", "fp16_weight", f"{prefix}_bf16_bf16_out"
            ),
            single_bf16,
            replace=replace,
        )
        register(
            KernelKey(
                "hip_gfx1100",
                "linear_triple",
                "fp16_weight",
                f"{prefix}_bf16_f32_out",
            ),
            triple_f32,
            replace=replace,
        )
    register(
        KernelKey("hip_gfx1100", "linear_pair", "fp16_weight", "bf16_f32_out"),
        laguna_f16w_dual_gemv_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear_triple", "fp16_weight", "bf16_f32_out"),
        laguna_f16w_triple_gemv_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_triple",
            "fp16_weight",
            "onebarrier_bf16_f32_out",
        ),
        laguna_f16w_triple_onebarrier_gemv_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_triple",
            "fp16_weight",
            "fixedk_onebarrier_bf16_f32_out",
        ),
        laguna_f16w_triple_fixedk_onebarrier_gemv_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_triple",
            "fp16_weight",
            "fixedk_nontemporal_bf16_f32_out",
        ),
        laguna_f16w_triple_fixedk_nontemporal_gemv_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_quad",
            "fp16_weight",
            "fixedk_onebarrier_bf16_f32_out",
        ),
        laguna_f16w_quad_fixedk_onebarrier_gemv_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_quad",
            "fp16_weight",
            "fixedk_nontemporal_bf16_f32_out",
        ),
        laguna_f16w_quad_fixedk_nontemporal_gemv_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear_triple", "fp16_weight", "tiled_bf16_f32_out"),
        laguna_f16w_triple_tiled_bf16_f32_out,
        replace=replace,
    )


register_laguna_f16_projection_kernels()

__all__ = [
    "build_laguna_f16_projection",
    "build_laguna_f16_projection_prefill",
    "laguna_f16w_dual_gemv_bf16_f32_out",
    "laguna_f16w_fixedk_onebarrier_gemv_bf16_bf16_out",
    "laguna_f16w_fixedk_onebarrier_gemv_bf16_f32_out",
    "laguna_f16w_fixedk_nontemporal_gemv_bf16_bf16_out",
    "laguna_f16w_fixedk_nontemporal_gemv_bf16_f32_out",
    "laguna_f16w_fixedk_nontemporal_output_add_rmsnorm_bf16",
    "laguna_f16w_fixedk_output_add_rmsnorm_bf16",
    "laguna_f16w_gemv_bf16_bf16_out",
    "laguna_f16w_gemv_bf16_f32_out",
    "laguna_f16w_gemv_f32_bf16_out",
    "laguna_f16w_gemv_f32_f32_out",
    "laguna_f16w_onebarrier_gemv_bf16_bf16_out",
    "laguna_f16w_onebarrier_gemv_bf16_f32_out",
    "laguna_f16w_quad_fixedk_onebarrier_gemv_bf16_f32_out",
    "laguna_f16w_quad_fixedk_nontemporal_gemv_bf16_f32_out",
    "laguna_f16w_tiled_bf16_bf16_out",
    "laguna_f16w_tiled_bf16_f32_out",
    "laguna_f16w_triple_fixedk_onebarrier_gemv_bf16_f32_out",
    "laguna_f16w_triple_fixedk_nontemporal_gemv_bf16_f32_out",
    "laguna_f16w_triple_gemv_bf16_f32_out",
    "laguna_f16w_triple_onebarrier_gemv_bf16_f32_out",
    "laguna_f16w_triple_tiled_bf16_f32_out",
    "laguna_f16w_triple_wmma_bf16_f32_out",
    "laguna_f16w_triple_wmma_comp_bf16_f32_out",
    "laguna_f16w_wmma_bf16_bf16_out",
    "laguna_f16w_wmma_bf16_f32_out",
    "laguna_f16w_wmma_comp_bf16_bf16_out",
    "laguna_f16w_wmma_comp_bf16_f32_out",
    "plan_laguna_f16_projection_build",
    "register_laguna_f16_projection_kernels",
]
