"""Raw-pointer wrappers for PARO SiLU and down-rotation kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

# rotate_out argtypes vary by number of input ptrs (1 for dual rotate gate_up,
# 2 for pair rotate gate+up); tail is the same.
_ARGTYPES_SILU_ROTATE_1 = (
    ctypes.c_void_p,                                                  # input(s)
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # pairs, theta, scales, out
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,    # rows, features, group_size, krot
    ctypes.c_void_p,                                                  # stream
)
_ARGTYPES_SILU_ROTATE_2 = (
    ctypes.c_void_p, ctypes.c_void_p,                                  # gate, up
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # pairs, theta, scales, out
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGTYPES_SILU_SEPARATE = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gate, up, out
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,      # rows, features, threads
    ctypes.c_void_p,                                    # stream
)

_SOURCE = Path(__file__).with_name("paro_silu.hip")
_OUTPUT_NAME = "paro_silu.so"
_SYMBOL_DUAL_OUT = "hipengine_silu_mul_dual_out_bf16"
_SYMBOL_DUAL_OUT_FP16 = "hipengine_silu_mul_dual_out_fp16"
_SYMBOL_SEPARATE_OUT = "hipengine_silu_mul_separate_out_bf16"
_SYMBOL_SEPARATE_OUT_FP16 = "hipengine_silu_mul_separate_out_fp16"
_SYMBOL_DUAL_ROTATE_OUT = "hipengine_silu_mul_dual_rotate_out_bf16"
_SYMBOL_DUAL_ROTATE_OUT_FP16 = "hipengine_silu_mul_dual_rotate_out_fp16"
_SYMBOL_PAIR_ROTATE_OUT = "hipengine_silu_mul_pair_rotate_out_bf16"
_SYMBOL_PAIR_ROTATE_OUT_FP16 = "hipengine_silu_mul_pair_rotate_out_fp16"
_ALLOWED_THREADS = {64, 128, 256}


def plan_paro_silu_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="paro_silu",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_paro_silu(
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
        family="paro_silu",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def silu_mul_dual_out_bf16(
    gate_up_ptr: int,
    out_ptr: int,
    rows: int,
    features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch SiLU(gate) * up over packed ``[rows, 2 * features]`` input."""

    _check_activation_shape(rows, features)
    _check_threads(threads)
    library = library or build_paro_silu(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DUAL_OUT)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(gate_up_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def silu_mul_dual_out_fp16(
    gate_up_ptr: int,
    out_ptr: int,
    rows: int,
    features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 SiLU(gate) * up over packed ``[rows, 2 * features]`` input."""

    _check_activation_shape(rows, features)
    _check_threads(threads)
    library = library or build_paro_silu(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DUAL_OUT_FP16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(gate_up_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def silu_mul_separate_out_bf16(
    gate_ptr: int,
    up_ptr: int,
    out_ptr: int,
    rows: int,
    features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch SiLU(gate) * up where ``gate`` and ``up`` are separate ``[rows, features]`` buffers."""

    _launch_separate(
        _SYMBOL_SEPARATE_OUT,
        gate_ptr,
        up_ptr,
        out_ptr,
        rows,
        features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def silu_mul_separate_out_fp16(
    gate_ptr: int,
    up_ptr: int,
    out_ptr: int,
    rows: int,
    features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 SiLU(gate) * up where ``gate`` and ``up`` are separate ``[rows, features]`` buffers."""

    _launch_separate(
        _SYMBOL_SEPARATE_OUT_FP16,
        gate_ptr,
        up_ptr,
        out_ptr,
        rows,
        features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def silu_mul_dual_rotate_out_bf16(
    gate_up_ptr: int,
    pairs_ptr: int,
    theta_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    features: int,
    group_size: int,
    krot: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch packed gate/up SiLU, channel scale, and pairwise down rotation."""

    _launch_rotate(
        _SYMBOL_DUAL_ROTATE_OUT,
        (gate_up_ptr,),
        pairs_ptr,
        theta_ptr,
        scales_ptr,
        out_ptr,
        rows,
        features,
        group_size,
        krot,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def silu_mul_dual_rotate_out_fp16(
    gate_up_ptr: int,
    pairs_ptr: int,
    theta_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    features: int,
    group_size: int,
    krot: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 packed gate/up SiLU, channel scale, and pairwise down rotation."""

    _launch_rotate(
        _SYMBOL_DUAL_ROTATE_OUT_FP16,
        (gate_up_ptr,),
        pairs_ptr,
        theta_ptr,
        scales_ptr,
        out_ptr,
        rows,
        features,
        group_size,
        krot,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def silu_mul_pair_rotate_out_bf16(
    gate_ptr: int,
    up_ptr: int,
    pairs_ptr: int,
    theta_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    features: int,
    group_size: int,
    krot: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch separate gate/up SiLU, channel scale, and pairwise down rotation."""

    _launch_rotate(
        _SYMBOL_PAIR_ROTATE_OUT,
        (gate_ptr, up_ptr),
        pairs_ptr,
        theta_ptr,
        scales_ptr,
        out_ptr,
        rows,
        features,
        group_size,
        krot,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def silu_mul_pair_rotate_out_fp16(
    gate_ptr: int,
    up_ptr: int,
    pairs_ptr: int,
    theta_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    features: int,
    group_size: int,
    krot: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 separate gate/up SiLU, channel scale, and pairwise down rotation."""

    _launch_rotate(
        _SYMBOL_PAIR_ROTATE_OUT_FP16,
        (gate_ptr, up_ptr),
        pairs_ptr,
        theta_ptr,
        scales_ptr,
        out_ptr,
        rows,
        features,
        group_size,
        krot,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def register_paro_silu_kernels(*, replace: bool = True) -> None:
    for quant in ("bf16", "w4_paro"):
        register(
            KernelKey("hip_gfx1100", "silu_mul_dual", quant, "out"),
            silu_mul_dual_out_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "silu_mul_dual", quant, "out_fp16"),
            silu_mul_dual_out_fp16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "silu_mul_separate", quant, "out"),
            silu_mul_separate_out_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "silu_mul_separate", quant, "out_fp16"),
            silu_mul_separate_out_fp16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "silu_mul_dual_rotate", quant, "out"),
            silu_mul_dual_rotate_out_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "silu_mul_dual_rotate", quant, "out_fp16"),
            silu_mul_dual_rotate_out_fp16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "silu_mul_pair_rotate", quant, "out"),
            silu_mul_pair_rotate_out_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "silu_mul_pair_rotate", quant, "out_fp16"),
            silu_mul_pair_rotate_out_fp16,
            replace=replace,
        )
    register(
        KernelKey("hip_gfx1100", "silu_mul_dual", "fp16", "out"),
        silu_mul_dual_out_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "silu_mul_separate", "fp16", "out"),
        silu_mul_separate_out_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "silu_mul_dual_rotate", "fp16", "out"),
        silu_mul_dual_rotate_out_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "silu_mul_pair_rotate", "fp16", "out"),
        silu_mul_pair_rotate_out_fp16,
        replace=replace,
    )


def _launch_rotate(
    symbol: str,
    input_ptrs: tuple[int, ...],
    pairs_ptr: int,
    theta_ptr: int,
    scales_ptr: int,
    out_ptr: int,
    rows: int,
    features: int,
    group_size: int,
    krot: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_rotate_shape(rows, features, group_size, krot)
    library = library or build_paro_silu(load=True)
    runtime = runtime or get_hip_runtime()
    argtypes = _ARGTYPES_SILU_ROTATE_1 if len(input_ptrs) == 1 else _ARGTYPES_SILU_ROTATE_2
    fn = signed_kernel_fn(library, symbol, argtypes, ctypes.c_int)
    err = fn(*input_ptrs, pairs_ptr, theta_ptr, scales_ptr, out_ptr,
             rows, features, group_size, krot, stream)
    _check_launch(runtime, err)


def _launch_separate(
    symbol: str,
    gate_ptr: int,
    up_ptr: int,
    out_ptr: int,
    rows: int,
    features: int,
    *,
    threads: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_activation_shape(rows, features)
    _check_threads(threads)
    library = library or build_paro_silu(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, symbol, _ARGTYPES_SILU_SEPARATE, ctypes.c_int)
    err = fn(gate_ptr, up_ptr, out_ptr, rows, features, threads, stream)
    _check_launch(runtime, err)


def _check_rotate_shape(rows: int, features: int, group_size: int, krot: int) -> None:
    _check_activation_shape(rows, features)
    _check_positive(group_size, "group_size")
    _check_positive(krot, "krot")
    if group_size % 2 != 0:
        raise ValueError("group_size must be even")
    if features % group_size != 0:
        raise ValueError("features must be divisible by group_size")


def _check_activation_shape(rows: int, features: int) -> None:
    _check_positive(rows, "rows")
    _check_positive(features, "features")


def _check_threads(threads: int) -> None:
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64, 128, or 256")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_paro_silu_kernels()
