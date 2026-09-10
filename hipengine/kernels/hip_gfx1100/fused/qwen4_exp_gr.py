"""Raw-pointer wrappers for strict Qwen4Exp gated-residual HIP primitives."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("qwen4_exp_gr.hip")
_OUTPUT_NAME = "qwen4_exp_gr.so"
_THREADS = 256
_ARGS_REPEAT = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGS_GROUPED_NORM = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_void_p,
)
_ARGS_MEAN = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGS_WRITE = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGS_SCALED_SILU = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_void_p,
)
_ARGS_SILU_MUL = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGS_SIGMOID = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGS_SIGMOID_NORM = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_void_p,
)


def plan_qwen4_exp_gr_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="qwen4_exp_gr",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_qwen4_exp_gr(
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
        family="qwen4_exp_gr",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def qwen4_exp_repeat_bf16_branches(
    input_ptr: int,
    output_ptr: int,
    branches: int,
    hidden: int,
    *,
    rows: int = 1,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if rows <= 0 or branches <= 0 or hidden <= 0:
        raise ValueError("rows, branches, and hidden must be positive")
    library = library or build_qwen4_exp_gr(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_repeat_bf16_branches",
        _ARGS_REPEAT,
        ctypes.c_int,
    )
    _check_launch(runtime, fn(input_ptr, output_ptr, rows, branches, hidden, stream))


def qwen4_exp_grouped_rmsnorm_bf16_f32(
    residual_ptr: int,
    weight_ptr: int,
    output_ptr: int,
    rows: int,
    branches: int,
    hidden: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Normalize each BF16 residual branch with direct F32 folded gamma."""

    _check_shape(rows, branches, hidden)
    library = library or build_qwen4_exp_gr(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_grouped_rmsnorm_bf16_f32",
        _ARGS_GROUPED_NORM,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            residual_ptr,
            weight_ptr,
            output_ptr,
            rows,
            branches,
            hidden,
            float(eps),
            stream,
        ),
    )


def qwen4_exp_grouped_rmsnorm_f32(
    input_ptr: int,
    weight_ptr: int,
    output_ptr: int,
    rows: int,
    branches: int,
    hidden: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_shape(rows, branches, hidden)
    library = library or build_qwen4_exp_gr(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_grouped_rmsnorm_f32",
        _ARGS_GROUPED_NORM,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(input_ptr, weight_ptr, output_ptr, rows, branches, hidden, float(eps), stream),
    )


def qwen4_exp_gated_mean_f32(
    normalized_ptr: int,
    gate_ptr: int,
    output_ptr: int,
    rows: int,
    branches: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Collapse gated F32 residual branches by their arithmetic mean."""

    _check_shape(rows, branches, hidden)
    library = library or build_qwen4_exp_gr(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_gated_mean_f32",
        _ARGS_MEAN,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(normalized_ptr, gate_ptr, output_ptr, rows, branches, hidden, stream),
    )


def qwen4_exp_gated_mean_sigmoid_unfused_f32(
    normalized_ptr: int,
    gate_logits_ptr: int,
    output_ptr: int,
    rows: int,
    branches: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Strict unfused sigmoid + gated-mean fallback with the fused ABI."""

    _check_shape(rows, branches, hidden)
    library = library or build_qwen4_exp_gr(load=True)
    runtime = runtime or get_hip_runtime()
    qwen4_exp_sigmoid_f32(
        gate_logits_ptr,
        gate_logits_ptr,
        rows * branches * hidden,
        stream=stream,
        library=library,
        runtime=runtime,
    )
    qwen4_exp_gated_mean_f32(
        normalized_ptr,
        gate_logits_ptr,
        output_ptr,
        rows,
        branches,
        hidden,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen4_exp_gated_mean_sigmoid_f32(
    normalized_ptr: int,
    gate_logits_ptr: int,
    output_ptr: int,
    rows: int,
    branches: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Collapse normalized branches with sigmoid-transformed gate logits."""

    _check_shape(rows, branches, hidden)
    library = library or build_qwen4_exp_gr(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_gated_mean_sigmoid_f32",
        _ARGS_MEAN,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            normalized_ptr,
            gate_logits_ptr,
            output_ptr,
            rows,
            branches,
            hidden,
            stream,
        ),
    )


def qwen4_exp_gr_write_bf16_f32(
    residual_ptr: int,
    block_output_ptr: int,
    inject_logits_ptr: int,
    output_ptr: int,
    rows: int,
    branches: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Write F32 block output into every BF16 residual branch."""

    _check_shape(rows, branches, hidden)
    library = library or build_qwen4_exp_gr(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_gr_write_bf16_f32",
        _ARGS_WRITE,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            residual_ptr,
            block_output_ptr,
            inject_logits_ptr,
            output_ptr,
            rows,
            branches,
            hidden,
            stream,
        ),
    )


def qwen4_exp_scaled_silu_f32(
    input_ptr: int,
    output_ptr: int,
    elements: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if elements <= 0:
        raise ValueError("elements must be positive")
    library = library or build_qwen4_exp_gr(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_scaled_silu_f32",
        _ARGS_SCALED_SILU,
        ctypes.c_int,
    )
    _check_launch(runtime, fn(input_ptr, output_ptr, elements, float(scale), stream))


def qwen4_exp_silu_mul_f32(
    gate_ptr: int,
    up_ptr: int,
    output_ptr: int,
    elements: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if elements <= 0:
        raise ValueError("elements must be positive")
    library = library or build_qwen4_exp_gr(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_silu_mul_f32",
        _ARGS_SILU_MUL,
        ctypes.c_int,
    )
    _check_launch(runtime, fn(gate_ptr, up_ptr, output_ptr, elements, stream))


def qwen4_exp_sigmoid_f32(
    input_ptr: int,
    output_ptr: int,
    elements: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if elements <= 0:
        raise ValueError("elements must be positive")
    library = library or build_qwen4_exp_gr(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_sigmoid_f32",
        _ARGS_SIGMOID,
        ctypes.c_int,
    )
    _check_launch(runtime, fn(input_ptr, output_ptr, elements, stream))


def qwen4_exp_sigmoid_gated_rmsnorm_f32(
    input_ptr: int,
    weight_ptr: int,
    gate_ptr: int,
    output_ptr: int,
    heads: int,
    head_dim: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply per-head F32 RMSNorm and Qwen4Exp sigmoid output gating."""

    if heads <= 0 or head_dim <= 0:
        raise ValueError("heads and head_dim must be positive")
    library = library or build_qwen4_exp_gr(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_sigmoid_gated_rmsnorm_f32",
        _ARGS_SIGMOID_NORM,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            input_ptr,
            weight_ptr,
            gate_ptr,
            output_ptr,
            heads,
            head_dim,
            float(eps),
            stream,
        ),
    )


def register_qwen4_exp_gr_kernels(*, replace: bool = True) -> None:
    registrations = {
        KernelKey(
            "hip_gfx1100",
            "repeat_branches",
            "bf16",
            "strict",
        ): qwen4_exp_repeat_bf16_branches,
        KernelKey(
            "hip_gfx1100",
            "gr_grouped_rmsnorm",
            "bf16_f32",
            "strict",
        ): qwen4_exp_grouped_rmsnorm_bf16_f32,
        KernelKey(
            "hip_gfx1100",
            "gr_grouped_rmsnorm",
            "f32",
            "strict",
        ): qwen4_exp_grouped_rmsnorm_f32,
        KernelKey(
            "hip_gfx1100",
            "gr_gated_mean",
            "f32",
            "strict",
        ): qwen4_exp_gated_mean_f32,
        KernelKey(
            "hip_gfx1100",
            "gr_gated_mean_sigmoid",
            "f32",
            "strict",
        ): qwen4_exp_gated_mean_sigmoid_f32,
        KernelKey(
            "hip_gfx1100",
            "gr_gated_mean_sigmoid",
            "f32",
            "strict_unfused",
        ): qwen4_exp_gated_mean_sigmoid_unfused_f32,
        KernelKey(
            "hip_gfx1100",
            "gr_write",
            "bf16_f32",
            "strict",
        ): qwen4_exp_gr_write_bf16_f32,
        KernelKey(
            "hip_gfx1100",
            "scaled_silu",
            "f32",
            "strict",
        ): qwen4_exp_scaled_silu_f32,
        KernelKey(
            "hip_gfx1100",
            "silu_mul",
            "f32",
            "strict",
        ): qwen4_exp_silu_mul_f32,
        KernelKey(
            "hip_gfx1100",
            "sigmoid",
            "f32",
            "strict",
        ): qwen4_exp_sigmoid_f32,
        KernelKey(
            "hip_gfx1100",
            "gdn_sigmoid_gated_rmsnorm",
            "f32",
            "strict",
        ): qwen4_exp_sigmoid_gated_rmsnorm_f32,
    }
    for key, function in registrations.items():
        register(key, function, replace=replace)


def _check_shape(rows: int, branches: int, hidden: int) -> None:
    if rows <= 0 or branches <= 0 or hidden <= 0:
        raise ValueError("rows, branches, and hidden must be positive")


def _check_launch(runtime: HipRuntime, error: int) -> None:
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


register_qwen4_exp_gr_kernels()


__all__ = [
    "build_qwen4_exp_gr",
    "plan_qwen4_exp_gr_build",
    "qwen4_exp_gated_mean_f32",
    "qwen4_exp_gated_mean_sigmoid_f32",
    "qwen4_exp_gated_mean_sigmoid_unfused_f32",
    "qwen4_exp_repeat_bf16_branches",
    "qwen4_exp_gr_write_bf16_f32",
    "qwen4_exp_grouped_rmsnorm_bf16_f32",
    "qwen4_exp_grouped_rmsnorm_f32",
    "qwen4_exp_scaled_silu_f32",
    "qwen4_exp_silu_mul_f32",
    "qwen4_exp_sigmoid_f32",
    "qwen4_exp_sigmoid_gated_rmsnorm_f32",
    "register_qwen4_exp_gr_kernels",
]
