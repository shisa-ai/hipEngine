"""Raw-pointer wrappers for PARO AWQ pack8 GEMV kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("paro_awq_gemv.hip")
_OUTPUT_NAME = "paro_awq_gemv.so"
_SYMBOL_PACK8_STRIDED = "hipengine_gemv_awq_pack8_strided_bf16"
_SYMBOL_PACK8_TRANSPOSED = "hipengine_gemv_awq_pack8_transposed_bf16"
_SYMBOL_DUAL_PACK8_STRIDED = "hipengine_gemv_awq_dual_pack8_strided_bf16"
_SYMBOL_DUAL_PACK8_TRANSPOSED = "hipengine_gemv_awq_dual_pack8_transposed_bf16"
_SYMBOL_PACK8_STRIDED_FP16 = "hipengine_gemv_awq_pack8_strided_fp16"
_SYMBOL_PACK8_TRANSPOSED_FP16 = "hipengine_gemv_awq_pack8_transposed_fp16"
_SYMBOL_DUAL_PACK8_STRIDED_FP16 = "hipengine_gemv_awq_dual_pack8_strided_fp16"
_SYMBOL_DUAL_PACK8_TRANSPOSED_FP16 = "hipengine_gemv_awq_dual_pack8_transposed_fp16"
_SYMBOL_SELECTED_DUAL_ROTATE_STRIDED = "hipengine_gemv_awq_selected_dual_pack8_strided_rotate_out_bf16"
_SYMBOL_SELECTED_DUAL_STRIDED = "hipengine_gemv_awq_selected_dual_pack8_strided_bf16"
_SYMBOL_SELECTED_DUAL_TRANSPOSED = "hipengine_gemv_awq_selected_dual_pack8_transposed_bf16"
_SYMBOL_SELECTED_STRIDED = "hipengine_gemv_awq_selected_pack8_strided_bf16"
_SYMBOL_SELECTED_TRANSPOSED = "hipengine_gemv_awq_selected_pack8_transposed_bf16"
_ALLOWED_THREADS = {64, 128}


def plan_paro_awq_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="paro_awq_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_paro_awq_gemv(
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
        family="paro_awq_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )



def gemv_awq_pack8_strided_bf16(
    x_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch generic single-projection pack8 GEMV with strided qweight layout."""

    _launch_pack8_single(
        _SYMBOL_PACK8_STRIDED,
        x_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_pack8_transposed_bf16(
    x_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch generic single-projection pack8 GEMV with transposed qweight layout."""

    _launch_pack8_single(
        _SYMBOL_PACK8_TRANSPOSED,
        x_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_dual_pack8_strided_bf16(
    x_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch generic dual pack8 GEMV with one shared input and strided qweights."""

    _launch_pack8_dual(
        _SYMBOL_DUAL_PACK8_STRIDED,
        (x_ptr,),
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_dual_pack8_transposed_bf16(
    x_a_ptr: int,
    x_b_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch generic dual pack8 GEMV with separate inputs and transposed qweights."""

    _launch_pack8_dual(
        _SYMBOL_DUAL_PACK8_TRANSPOSED,
        (x_a_ptr, x_b_ptr),
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_pack8_strided_fp16(
    x_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch generic single-projection pack8 GEMV for FP16 buffers."""

    _launch_pack8_single(
        _SYMBOL_PACK8_STRIDED_FP16,
        x_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_pack8_transposed_fp16(
    x_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch generic single-projection transposed pack8 GEMV for FP16 buffers."""

    _launch_pack8_single(
        _SYMBOL_PACK8_TRANSPOSED_FP16,
        x_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_dual_pack8_strided_fp16(
    x_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch generic dual pack8 GEMV with one shared FP16 input."""

    _launch_pack8_dual(
        _SYMBOL_DUAL_PACK8_STRIDED_FP16,
        (x_ptr,),
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_dual_pack8_transposed_fp16(
    x_a_ptr: int,
    x_b_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch generic dual transposed pack8 GEMV for FP16 buffers."""

    _launch_pack8_dual(
        _SYMBOL_DUAL_PACK8_TRANSPOSED_FP16,
        (x_a_ptr, x_b_ptr),
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_selected_dual_pack8_strided_rotate_out_bf16(
    x_ptr: int,
    selected_ptr: int,
    pairs_ptr: int,
    theta_ptr: int,
    channel_scales_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    krot: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch parent fused rotate + selected dual pack8 GEMV strided kernel."""

    _launch_selected_dual_rotate(
        _SYMBOL_SELECTED_DUAL_ROTATE_STRIDED,
        x_ptr,
        selected_ptr,
        pairs_ptr,
        theta_ptr,
        channel_scales_ptr,
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        x_rows,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        num_experts,
        group_size,
        krot,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )

def gemv_awq_selected_dual_pack8_strided_bf16(
    x_ptr: int,
    selected_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch selected-expert dual gate/up pack8 GEMV with strided qweight layout."""

    _launch_selected_dual(
        _SYMBOL_SELECTED_DUAL_STRIDED,
        x_ptr,
        selected_ptr,
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        x_rows,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        num_experts,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_selected_dual_pack8_transposed_bf16(
    x_ptr: int,
    selected_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch selected-expert dual gate/up pack8 GEMV with transposed qweight layout."""

    _launch_selected_dual(
        _SYMBOL_SELECTED_DUAL_TRANSPOSED,
        x_ptr,
        selected_ptr,
        qweight_a_ptr,
        qzeros_a_ptr,
        scales_a_ptr,
        qweight_b_ptr,
        qzeros_b_ptr,
        scales_b_ptr,
        out_ptr,
        x_rows,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        num_experts,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_selected_pack8_strided_bf16(
    x_ptr: int,
    selected_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    num_experts: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch selected-expert single/down pack8 GEMV with strided qweight layout."""

    _launch_selected_single(
        _SYMBOL_SELECTED_STRIDED,
        x_ptr,
        selected_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        num_experts,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gemv_awq_selected_pack8_transposed_bf16(
    x_ptr: int,
    selected_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    num_experts: int,
    group_size: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch selected-expert single/down pack8 GEMV with transposed qweight layout."""

    _launch_selected_single(
        _SYMBOL_SELECTED_TRANSPOSED,
        x_ptr,
        selected_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        out_ptr,
        rows,
        in_features,
        out_packed,
        num_experts,
        group_size,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def register_paro_awq_gemv_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "pack8_gemv", "w4_paro", "strided"),
        gemv_awq_pack8_strided_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "pack8_gemv", "w4_paro", "transposed"),
        gemv_awq_pack8_transposed_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dual_pack8_gemv", "w4_paro", "strided"),
        gemv_awq_dual_pack8_strided_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dual_pack8_gemv", "w4_paro", "transposed"),
        gemv_awq_dual_pack8_transposed_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "pack8_gemv", "w4_paro", "strided_fp16"),
        gemv_awq_pack8_strided_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "pack8_gemv", "w4_paro", "transposed_fp16"),
        gemv_awq_pack8_transposed_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dual_pack8_gemv", "w4_paro", "strided_fp16"),
        gemv_awq_dual_pack8_strided_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dual_pack8_gemv", "w4_paro", "transposed_fp16"),
        gemv_awq_dual_pack8_transposed_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "rotate+selected_dual_pack8_gemv", "w4_paro", "strided"),
        gemv_awq_selected_dual_pack8_strided_rotate_out_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "selected_dual_pack8_gemv", "w4_paro", "strided"),
        gemv_awq_selected_dual_pack8_strided_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "selected_dual_pack8_gemv", "w4_paro", "transposed"),
        gemv_awq_selected_dual_pack8_transposed_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "selected_pack8_gemv", "w4_paro", "strided"),
        gemv_awq_selected_pack8_strided_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "selected_pack8_gemv", "w4_paro", "transposed"),
        gemv_awq_selected_pack8_transposed_bf16,
        replace=replace,
    )



def _launch_pack8_single(
    symbol: str,
    x_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_pack8_single_shape(rows, in_features, out_packed, group_size, threads)
    library = library or build_paro_awq_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
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
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(qzeros_ptr),
        ctypes.c_void_p(scales_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_packed),
        ctypes.c_int64(group_size),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_pack8_dual(
    symbol: str,
    input_ptrs: tuple[int, ...],
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    if len(input_ptrs) not in (1, 2):
        raise ValueError("input_ptrs must contain one or two pointers")
    _check_pack8_dual_shape(rows, in_features, out_packed_a, out_packed_b, group_size, threads)
    library = library or build_paro_awq_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [*(ctypes.c_void_p for _ in input_ptrs)] + [
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
    ]
    fn.restype = ctypes.c_int
    err = fn(
        *(ctypes.c_void_p(ptr) for ptr in input_ptrs),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qzeros_a_ptr),
        ctypes.c_void_p(scales_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(qzeros_b_ptr),
        ctypes.c_void_p(scales_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_packed_a),
        ctypes.c_int64(out_packed_b),
        ctypes.c_int64(group_size),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_selected_dual_rotate(
    symbol: str,
    x_ptr: int,
    selected_ptr: int,
    pairs_ptr: int,
    theta_ptr: int,
    channel_scales_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    krot: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_selected_dual_rotate_shape(
        x_rows,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        num_experts,
        group_size,
        krot,
        threads,
    )
    library = library or build_paro_awq_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(pairs_ptr),
        ctypes.c_void_p(theta_ptr),
        ctypes.c_void_p(channel_scales_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qzeros_a_ptr),
        ctypes.c_void_p(scales_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(qzeros_b_ptr),
        ctypes.c_void_p(scales_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(x_rows),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_packed_a),
        ctypes.c_int64(out_packed_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(group_size),
        ctypes.c_int64(krot),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)

def _launch_selected_dual(
    symbol: str,
    x_ptr: int,
    selected_ptr: int,
    qweight_a_ptr: int,
    qzeros_a_ptr: int,
    scales_a_ptr: int,
    qweight_b_ptr: int,
    qzeros_b_ptr: int,
    scales_b_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_selected_dual_shape(
        x_rows,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        num_experts,
        group_size,
        threads,
    )
    library = library or build_paro_awq_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qzeros_a_ptr),
        ctypes.c_void_p(scales_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(qzeros_b_ptr),
        ctypes.c_void_p(scales_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(x_rows),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_packed_a),
        ctypes.c_int64(out_packed_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(group_size),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_selected_single(
    symbol: str,
    x_ptr: int,
    selected_ptr: int,
    qweight_ptr: int,
    qzeros_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_packed: int,
    num_experts: int,
    group_size: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_selected_single_shape(rows, in_features, out_packed, num_experts, group_size, threads)
    library = library or build_paro_awq_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
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
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(qzeros_ptr),
        ctypes.c_void_p(scales_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_packed),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(group_size),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)



def _check_pack8_single_shape(
    rows: int,
    in_features: int,
    out_packed: int,
    group_size: int,
    threads: int,
) -> None:
    _check_positive(rows, "rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_packed, "out_packed")
    _check_pack8_common(in_features, group_size, threads)


def _check_pack8_dual_shape(
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    group_size: int,
    threads: int,
) -> None:
    _check_positive(rows, "rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_packed_a, "out_packed_a")
    _check_positive(out_packed_b, "out_packed_b")
    _check_pack8_common(in_features, group_size, threads)


def _check_pack8_common(in_features: int, group_size: int, threads: int) -> None:
    _check_positive(group_size, "group_size")
    if in_features % group_size != 0:
        raise ValueError("in_features must be divisible by group_size")
    if group_size % 8 != 0:
        raise ValueError("group_size must be a multiple of 8")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64 or 128")


def _check_selected_dual_rotate_shape(
    x_rows: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    krot: int,
    threads: int,
) -> None:
    _check_selected_dual_shape(
        x_rows,
        rows,
        in_features,
        out_packed_a,
        out_packed_b,
        num_experts,
        group_size,
        threads,
    )
    if krot < 0:
        raise ValueError("krot must be non-negative")

def _check_selected_dual_shape(
    x_rows: int,
    rows: int,
    in_features: int,
    out_packed_a: int,
    out_packed_b: int,
    num_experts: int,
    group_size: int,
    threads: int,
) -> None:
    _check_positive(x_rows, "x_rows")
    _check_positive(rows, "rows")
    if rows % x_rows != 0:
        raise ValueError("rows must be divisible by x_rows")
    _check_common_quant_shape(in_features, num_experts, group_size, threads)
    _check_positive(out_packed_a, "out_packed_a")
    _check_positive(out_packed_b, "out_packed_b")


def _check_selected_single_shape(
    rows: int,
    in_features: int,
    out_packed: int,
    num_experts: int,
    group_size: int,
    threads: int,
) -> None:
    _check_positive(rows, "rows")
    _check_common_quant_shape(in_features, num_experts, group_size, threads)
    _check_positive(out_packed, "out_packed")


def _check_common_quant_shape(
    in_features: int,
    num_experts: int,
    group_size: int,
    threads: int,
) -> None:
    _check_positive(in_features, "in_features")
    _check_positive(num_experts, "num_experts")
    _check_positive(group_size, "group_size")
    if in_features % group_size != 0:
        raise ValueError("in_features must be divisible by group_size")
    if group_size % 8 != 0:
        raise ValueError("group_size must be a multiple of 8")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64 or 128")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_paro_awq_gemv_kernels()
