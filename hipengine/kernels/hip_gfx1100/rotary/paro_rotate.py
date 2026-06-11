"""Raw-pointer wrappers for PARO pairwise rotation kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("paro_rotate.hip")
_OUTPUT_NAME = "paro_rotate.so"
_SYMBOL_ROTATE1 = "hipengine_paro_rotate1_bf16"
_SYMBOL_ROTATE2 = "hipengine_paro_rotate2_bf16"
_SYMBOL_ROTATE3 = "hipengine_paro_rotate3_bf16"
_SYMBOL_ROTATE1_FP16 = "hipengine_paro_rotate1_fp16"
_SYMBOL_ROTATE1_F32_TO_FP16 = "hipengine_paro_rotate1_f32_to_fp16"
_SYMBOL_ROTATE2_FP16 = "hipengine_paro_rotate2_fp16"
_SYMBOL_ROTATE3_FP16 = "hipengine_paro_rotate3_fp16"
_SYMBOL_ROTATE1_BF16_GATE_FP16 = "hipengine_paro_rotate1_bf16_gate_fp16"
_SYMBOL_RMSNORM_ROTATE2_FP16 = "hipengine_paro_rmsnorm_rotate2_fp16"

# rmsnorm_rotate2: x, ln_weight, out_norm, out0, out1, pairs0, pairs1, theta0,
# theta1, scales0, scales1 + eps, tokens, hidden, group_size, krot, stream
_ARGTYPES_RMSNORM_ROTATE2 = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_float,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p,
)

# rotate1: x, out, pairs, theta, scales + tokens, hidden, group_size, krot, stream
_ARGTYPES_ROTATE1 = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p,
)
# rotate2: x, out0, out1, pairs0, pairs1, theta0, theta1, scales0, scales1 + tokens, hidden, group_size, krot, stream
_ARGTYPES_ROTATE2 = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p,
)
# rotate3: x, out0, out1, out2, pairs0, pairs1, pairs2, theta0, theta1, theta2, scales0, scales1, scales2 + tokens, hidden, group_size, krot, stream
_ARGTYPES_ROTATE3 = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p,
)
# rotate1_gate: x, gate, out, pairs, theta, scales + tokens, hidden, group_size, krot, stream
_ARGTYPES_ROTATE1_GATE = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p,
)


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


def paro_rotate1_bf16(
    x_ptr: int,
    out_ptr: int,
    pairs_ptr: int,
    theta_ptr: int,
    scales_ptr: int,
    tokens: int,
    hidden: int,
    group_size: int,
    krot: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch parent PARO single-output pairwise rotation kernel for BF16 buffers."""

    _check_rotate_shape(tokens, hidden, group_size, krot)
    library = library or build_paro_rotate(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_ROTATE1, _ARGTYPES_ROTATE1, ctypes.c_int)
    err = fn(x_ptr, out_ptr, pairs_ptr, theta_ptr, scales_ptr,
             tokens, hidden, group_size, krot, stream)
    _check_launch(runtime, err)


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
    fn = signed_kernel_fn(library, _SYMBOL_ROTATE2, _ARGTYPES_ROTATE2, ctypes.c_int)
    err = fn(x_ptr, out0_ptr, out1_ptr, pairs0_ptr, pairs1_ptr,
             theta0_ptr, theta1_ptr, scales0_ptr, scales1_ptr,
             tokens, hidden, group_size, krot, stream)
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
    fn = signed_kernel_fn(library, _SYMBOL_ROTATE3, _ARGTYPES_ROTATE3, ctypes.c_int)
    err = fn(x_ptr, out0_ptr, out1_ptr, out2_ptr,
             pairs0_ptr, pairs1_ptr, pairs2_ptr,
             theta0_ptr, theta1_ptr, theta2_ptr,
             scales0_ptr, scales1_ptr, scales2_ptr,
             tokens, hidden, group_size, krot, stream)
    _check_launch(runtime, err)


def paro_rotate1_fp16(
    x_ptr: int,
    out_ptr: int,
    pairs_ptr: int,
    theta_ptr: int,
    scales_ptr: int,
    tokens: int,
    hidden: int,
    group_size: int,
    krot: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch parent PARO single-output pairwise rotation kernel for FP16 buffers."""

    _check_rotate_shape(tokens, hidden, group_size, krot)
    library = library or build_paro_rotate(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_ROTATE1_FP16, _ARGTYPES_ROTATE1, ctypes.c_int)
    err = fn(x_ptr, out_ptr, pairs_ptr, theta_ptr, scales_ptr,
             tokens, hidden, group_size, krot, stream)
    _check_launch(runtime, err)


def paro_rotate1_f32_to_fp16(
    x_ptr: int,
    out_ptr: int,
    pairs_ptr: int,
    theta_ptr: int,
    scales_ptr: int,
    tokens: int,
    hidden: int,
    group_size: int,
    krot: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Round FP32 input to FP16, then launch PARO rotate1 into FP16 output."""

    _check_rotate_shape(tokens, hidden, group_size, krot)
    library = library or build_paro_rotate(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_ROTATE1_F32_TO_FP16, _ARGTYPES_ROTATE1, ctypes.c_int)
    err = fn(x_ptr, out_ptr, pairs_ptr, theta_ptr, scales_ptr,
             tokens, hidden, group_size, krot, stream)
    _check_launch(runtime, err)


def paro_rotate1_bf16_gate_fp16(
    x_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    pairs_ptr: int,
    theta_ptr: int,
    scales_ptr: int,
    tokens: int,
    hidden: int,
    group_size: int,
    krot: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Gate BF16 attention with FP16 gate, then launch PARO rotate1 into FP16 output."""

    _check_rotate_shape(tokens, hidden, group_size, krot)
    library = library or build_paro_rotate(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_ROTATE1_BF16_GATE_FP16, _ARGTYPES_ROTATE1_GATE, ctypes.c_int)
    err = fn(x_ptr, gate_ptr, out_ptr, pairs_ptr, theta_ptr, scales_ptr,
             tokens, hidden, group_size, krot, stream)
    _check_launch(runtime, err)


def paro_rotate2_fp16(
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
    """Launch parent PARO two-output pairwise rotation kernel for FP16 buffers."""

    _check_rotate_shape(tokens, hidden, group_size, krot)
    library = library or build_paro_rotate(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_ROTATE2_FP16, _ARGTYPES_ROTATE2, ctypes.c_int)
    err = fn(x_ptr, out0_ptr, out1_ptr, pairs0_ptr, pairs1_ptr,
             theta0_ptr, theta1_ptr, scales0_ptr, scales1_ptr,
             tokens, hidden, group_size, krot, stream)
    _check_launch(runtime, err)


def paro_rotate3_fp16(
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
    """Launch parent PARO three-output pairwise rotation kernel for FP16 buffers."""

    _check_rotate_shape(tokens, hidden, group_size, krot)
    library = library or build_paro_rotate(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_ROTATE3_FP16, _ARGTYPES_ROTATE3, ctypes.c_int)
    err = fn(x_ptr, out0_ptr, out1_ptr, out2_ptr,
             pairs0_ptr, pairs1_ptr, pairs2_ptr,
             theta0_ptr, theta1_ptr, theta2_ptr,
             scales0_ptr, scales1_ptr, scales2_ptr,
             tokens, hidden, group_size, krot, stream)
    _check_launch(runtime, err)


def paro_rmsnorm_rotate2_fp16(
    x_ptr: int,
    ln_weight_ptr: int,
    out_norm_ptr: int,
    out0_ptr: int,
    out1_ptr: int,
    pairs0_ptr: int,
    pairs1_ptr: int,
    theta0_ptr: int,
    theta1_ptr: int,
    scales0_ptr: int,
    scales1_ptr: int,
    eps: float,
    tokens: int,
    hidden: int,
    group_size: int,
    krot: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """M15.4: fused RMSNorm + paro_rotate2 (FP16).

    Bit-identical to ``paro_rmsnorm_out_fp16`` followed by ``paro_rotate2_fp16``
    (same 256-thread reduction, fp16 normed round-trip, and per-group butterfly),
    in one launch.  ``x``/``ln_weight`` feed RMSNorm; ``out_norm`` receives the
    (unrotated) RMSNorm output, and ``out0``/``out1`` the two rotated
    projections.  Pass ``out_norm_ptr=0`` to skip the RMSNorm write-back.
    Requires the M15.4 shape constraints (256-thread block;
    ``(hidden/group_size) % (256/(group_size/2)) == 0``).
    """

    _check_rotate_shape(tokens, hidden, group_size, krot)
    library = library or build_paro_rotate(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_RMSNORM_ROTATE2_FP16, _ARGTYPES_RMSNORM_ROTATE2, ctypes.c_int)
    err = fn(x_ptr, ln_weight_ptr, out_norm_ptr, out0_ptr, out1_ptr, pairs0_ptr, pairs1_ptr,
             theta0_ptr, theta1_ptr, scales0_ptr, scales1_ptr,
             float(eps), tokens, hidden, group_size, krot, stream)
    _check_launch(runtime, err)


def register_paro_rotate_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "paro_rotate1", "w4_paro", "bf16"),
        paro_rotate1_bf16,
        replace=replace,
    )
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
    register(
        KernelKey("hip_gfx1100", "paro_rotate1", "w4_paro", "fp16"),
        paro_rotate1_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paro_rotate1", "w4_paro", "f32_to_fp16"),
        paro_rotate1_f32_to_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paro_rotate1", "w4_paro", "bf16_gate_fp16"),
        paro_rotate1_bf16_gate_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paro_rotate2", "w4_paro", "fp16"),
        paro_rotate2_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paro_rotate3", "w4_paro", "fp16"),
        paro_rotate3_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "paro_rmsnorm_rotate2", "w4_paro", "fp16"),
        paro_rmsnorm_rotate2_fp16,
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
