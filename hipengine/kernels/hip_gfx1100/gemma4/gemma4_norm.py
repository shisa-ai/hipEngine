"""Raw-pointer wrappers for the Gemma 4 norm and layer-output HIP family.

Gemma 4's RMSNorm applies its weight as-is. The Qwen3.5 kernels next door apply
``1.0f + weight``, so the two are not interchangeable and Gemma 4 needs its own
entry points rather than a variant of that family.

Importing this module registers ctypes launch wrappers but does not build or
load ROCm until a wrapper is called.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_ARGTYPES_NORM_4PTR = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_void_p,
)
_ARGTYPES_WEIGHTLESS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_void_p,
)
_ARGTYPES_ROUTER_PRESCALE = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_void_p,
)
_ARGTYPES_ADD_RMSNORM_SCALE = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_void_p,
)
_ARGTYPES_EXPERT_WEIGHT_SCALE = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGTYPES_BRANCH_ADD = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGTYPES_SCALE = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_void_p,
)

_SOURCE = Path(__file__).with_name("gemma4_norm.hip")
_OUTPUT_NAME = "gemma4_norm.so"

_SYMBOL_RMSNORM_BF16 = "hipengine_gemma4_rmsnorm_f32w_bf16"
_SYMBOL_RMSNORM_F32 = "hipengine_gemma4_rmsnorm_f32w_f32"
_SYMBOL_WEIGHTLESS_BF16 = "hipengine_gemma4_rmsnorm_weightless_bf16"
_SYMBOL_HEAD_RMSNORM_BF16 = "hipengine_gemma4_head_rmsnorm_f32w_bf16"
_SYMBOL_ROUTER_PRESCALE_BF16 = "hipengine_gemma4_router_prescale_bf16"
_SYMBOL_ADD_RMSNORM_SCALE_BF16 = "hipengine_gemma4_add_rmsnorm_scale_bf16"
_SYMBOL_EXPERT_WEIGHT_SCALE_F32 = "hipengine_gemma4_expert_weight_scale_f32"
_SYMBOL_BRANCH_ADD_BF16 = "hipengine_gemma4_branch_add_bf16"
_SYMBOL_SCALE_BF16 = "hipengine_gemma4_scale_bf16"


def plan_gemma4_norm_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gemma4_norm",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gemma4_norm(
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
        family="gemma4_norm",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gemma4_rmsnorm_f32w_bf16(
    hidden_states_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    hidden_size: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the plain-weight RMSNorm over BF16-bit rows with an F32 weight."""

    _check_positive_shape(rows, hidden_size)
    library = library or build_gemma4_norm(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_RMSNORM_BF16, _ARGTYPES_NORM_4PTR, ctypes.c_int)
    err = fn(hidden_states_ptr, weight_ptr, out_ptr, rows, hidden_size, float(eps), stream)
    _check_launch(runtime, err)


def gemma4_rmsnorm_f32w_f32(
    hidden_states_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    hidden_size: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the plain-weight RMSNorm over F32 rows with an F32 weight."""

    _check_positive_shape(rows, hidden_size)
    library = library or build_gemma4_norm(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_RMSNORM_F32, _ARGTYPES_NORM_4PTR, ctypes.c_int)
    err = fn(hidden_states_ptr, weight_ptr, out_ptr, rows, hidden_size, float(eps), stream)
    _check_launch(runtime, err)


def gemma4_rmsnorm_weightless_bf16(
    hidden_states_ptr: int,
    out_ptr: int,
    rows: int,
    hidden_size: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the weightless RMSNorm, used by the value norm and the router."""

    _check_positive_shape(rows, hidden_size)
    library = library or build_gemma4_norm(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_WEIGHTLESS_BF16, _ARGTYPES_WEIGHTLESS, ctypes.c_int)
    err = fn(hidden_states_ptr, out_ptr, rows, hidden_size, float(eps), stream)
    _check_launch(runtime, err)


def gemma4_head_rmsnorm_f32w_bf16(
    hidden_states_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    head_dim: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the per-head RMSNorm over ``rows`` heads of ``head_dim`` each.

    Pass ``weight_ptr = 0`` for the weightless form used by the value norm.
    """

    _check_positive_shape(rows, head_dim)
    library = library or build_gemma4_norm(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_HEAD_RMSNORM_BF16, _ARGTYPES_NORM_4PTR, ctypes.c_int)
    err = fn(hidden_states_ptr, weight_ptr, out_ptr, rows, head_dim, float(eps), stream)
    _check_launch(runtime, err)


def gemma4_router_prescale_bf16(
    hidden_states_ptr: int,
    scale_ptr: int,
    out_ptr: int,
    rows: int,
    hidden_size: int,
    eps: float = 1e-6,
    *,
    root_size: float,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the router's weightless norm plus ``scale * hidden_size**-0.5``."""

    _check_positive_shape(rows, hidden_size)
    library = library or build_gemma4_norm(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, _SYMBOL_ROUTER_PRESCALE_BF16, _ARGTYPES_ROUTER_PRESCALE, ctypes.c_int
    )
    err = fn(
        hidden_states_ptr,
        scale_ptr,
        out_ptr,
        rows,
        hidden_size,
        float(eps),
        float(root_size),
        stream,
    )
    _check_launch(runtime, err)


def gemma4_add_rmsnorm_scale_bf16(
    hidden_states_ptr: int,
    residual_ptr: int,
    weight_ptr: int,
    layer_scalar_ptr: int,
    out_ptr: int,
    rows: int,
    hidden_size: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch ``out = (residual + rmsnorm(x, weight)) * layer_scalar``.

    Pass ``layer_scalar_ptr = 0`` to skip the scale, which makes the kernel a
    plain add-norm.
    """

    _check_positive_shape(rows, hidden_size)
    library = library or build_gemma4_norm(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, _SYMBOL_ADD_RMSNORM_SCALE_BF16, _ARGTYPES_ADD_RMSNORM_SCALE, ctypes.c_int
    )
    err = fn(
        hidden_states_ptr,
        residual_ptr,
        weight_ptr,
        layer_scalar_ptr,
        out_ptr,
        rows,
        hidden_size,
        float(eps),
        stream,
    )
    _check_launch(runtime, err)


def gemma4_expert_weight_scale_f32(
    weights_ptr: int,
    selected_ptr: int,
    per_expert_scale_ptr: int,
    rows: int,
    top_k: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Scale each selected routing weight by its expert's ``per_expert_scale``."""

    _check_positive_shape(rows, top_k)
    library = library or build_gemma4_norm(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        _SYMBOL_EXPERT_WEIGHT_SCALE_F32,
        _ARGTYPES_EXPERT_WEIGHT_SCALE,
        ctypes.c_int,
    )
    err = fn(weights_ptr, selected_ptr, per_expert_scale_ptr, rows, top_k, stream)
    _check_launch(runtime, err)


def gemma4_branch_add_bf16(
    a_ptr: int,
    b_ptr: int,
    out_ptr: int,
    total: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Add the dense and expert branch outputs elementwise over a flat buffer."""

    if total <= 0:
        raise ValueError("total must be positive")
    library = library or build_gemma4_norm(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_BRANCH_ADD_BF16, _ARGTYPES_BRANCH_ADD, ctypes.c_int)
    err = fn(a_ptr, b_ptr, out_ptr, total, stream)
    _check_launch(runtime, err)


def gemma4_scale_bf16(
    x_ptr: int,
    out_ptr: int,
    rows: int,
    hidden_size: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch ``out = x * scale`` elementwise over ``rows * hidden_size`` BF16 values.

    Gemma 4 multiplies its embedding by ``sqrt(hidden_size)``. That cannot be
    folded into the RMSNorm that follows it: the norm would not see the factor,
    but the residual stream the layer adds back into would be wrong by exactly
    that much. In-place is allowed (``out_ptr == x_ptr``); every thread reads its
    own element before writing it.
    """

    _check_positive_shape(rows, hidden_size)
    library = library or build_gemma4_norm(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_SCALE_BF16, _ARGTYPES_SCALE, ctypes.c_int)
    err = fn(x_ptr, out_ptr, rows * hidden_size, float(scale), stream)
    _check_launch(runtime, err)


def register_gemma4_norm_kernels(*, replace: bool = False) -> None:
    """Register the Gemma 4 norm family against the four-axis registry."""

    for quant in ("gguf_q4_k_m", "gguf_q4_k_xl", "gguf_q8_0", "gguf"):
        register(
            KernelKey("hip_gfx1100", "rmsnorm", quant, "gemma4_plain"),
            gemma4_rmsnorm_f32w_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "rmsnorm_weightless", quant, "gemma4_plain"),
            gemma4_rmsnorm_weightless_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "head_rmsnorm", quant, "gemma4_plain"),
            gemma4_head_rmsnorm_f32w_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "router_prescale", quant, "gemma4_plain"),
            gemma4_router_prescale_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "add_rmsnorm_scale", quant, "gemma4_plain"),
            gemma4_add_rmsnorm_scale_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "branch_add", quant, "gemma4_plain"),
            gemma4_branch_add_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "scale", quant, "gemma4_plain"),
            gemma4_scale_bf16,
            replace=replace,
        )


def _check_positive_shape(outer: int, inner: int) -> None:
    if outer <= 0:
        raise ValueError("rows must be positive")
    if inner <= 0:
        raise ValueError("hidden size must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_gemma4_norm_kernels()
