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

# fused, stride, rows, q_width, kv_width, parts, q, k, v, stream
_ARGTYPES_QKV_SPLIT = (
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
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
_SYMBOL_LOGIT_SOFTCAP = "hipengine_gemma4_logit_softcap_f32"
_SYMBOL_LOGIT_ARGMAX = "hipengine_gemma4_logit_argmax_f32"
_SYMBOL_QKV_SPLIT = "hipengine_gemma4_qkv_split_bf16"


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


def gemma4_qkv_split_bf16(
    fused_ptr: int,
    q_ptr: int,
    k_ptr: int,
    v_ptr: int,
    rows: int,
    q_width: int,
    kv_width: int,
    parts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Split one fused q/k/v projection output into the three attention buffers.

    ``fused_ptr`` is the row-major
    ``[rows, q_width + (parts - 1) * kv_width]`` result of a single fused ``gemma4_project`` call, which is how P6 replaces
    three projection launches with one. ``parts`` is 2 on k_eq_v layers -- their
    artifact carries no ``attn_v`` -- and 3 otherwise; with ``parts == 2`` the
    ``v_ptr`` region is never addressed.

    The device launcher re-derives ``stride`` from the three widths and
    rejects a mismatch, so a fused buffer sized wrongly fails loudly instead of
    landing in the wrong array. Values are copied as raw BF16 bits, so the
    split is bitwise transparent: this cannot change a single output element,
    which is what makes an end-to-end bitwise gate against the unfused path a
    meaningful gate rather than a tolerance check.
    """
    if parts not in (2, 3):
        raise ValueError(f"parts must be 2 or 3, got {parts!r}")
    _check_positive_shape(rows, q_width)
    if kv_width <= 0:
        raise ValueError("kv width must be positive")
    library = library or build_gemma4_norm(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_QKV_SPLIT, _ARGTYPES_QKV_SPLIT, ctypes.c_int)
    err = fn(
        fused_ptr,
        q_width + (parts - 1) * kv_width,
        rows,
        q_width,
        kv_width,
        parts,
        q_ptr,
        k_ptr,
        v_ptr,
        stream,
    )
    _check_launch(runtime, err)


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


# Multi-output norm (D6 fusion): in0..in2, w0..w2, out0..out2, count, rows,
# hidden_size, eps, stream.
_ARGTYPES_MULTI_NORM = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_void_p,
)
_SYMBOL_MULTI_BF16 = "hipengine_gemma4_multi_rmsnorm_bf16"


def gemma4_multi_rmsnorm_bf16(
    in0_ptr: int,
    in1_ptr: int,
    in2_ptr: int,
    w0_ptr: int,
    w1_ptr: int,
    w2_ptr: int,
    out0_ptr: int,
    out1_ptr: int,
    out2_ptr: int,
    count: int,
    rows: int,
    hidden_size: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch up to three RMSNorms over one block-per-row dispatch (D6).

    ``count`` outputs are computed sequentially: input ``k``, weight ``k``
    (``0`` selects the weightless expression), output ``k`` for
    ``k in 0..count-1``; slots beyond ``count`` are ignored but must be
    non-null for the slots in use. Each output is bit-identical to its
    standalone kernel, so the fused call may replace a chain of
    ``gemma4_rmsnorm_f32w_bf16`` / ``gemma4_rmsnorm_weightless_bf16`` calls
    with matching arguments without changing any value.
    """

    if count < 1 or count > 3:
        raise ValueError("count must be in 1..3")
    _check_positive_shape(rows, hidden_size)
    library = library or build_gemma4_norm(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_MULTI_BF16, _ARGTYPES_MULTI_NORM, ctypes.c_int)
    err = fn(
        in0_ptr, in1_ptr, in2_ptr,
        w0_ptr, w1_ptr, w2_ptr,
        out0_ptr, out1_ptr, out2_ptr,
        int(count), rows, hidden_size, float(eps), stream,
    )
    _check_launch(runtime, err)


# D6 tail fold: dense, experts, residual, dense_weight, tail_weight,
# layer_scalar, out, rows, hidden_size, eps, stream.
_ARGTYPES_DENSE_COMBINE = (
    ctypes.c_void_p,
    ctypes.c_void_p,
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
_SYMBOL_DENSE_COMBINE_BF16 = "hipengine_gemma4_dense_combine_rmsnorm_scale_bf16"


def gemma4_dense_combine_rmsnorm_scale_bf16(
    dense_ptr: int,
    experts_ptr: int,
    residual_ptr: int,
    dense_weight_ptr: int,
    tail_weight_ptr: int,
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
    """Fold post_ffw_norm_1 -> branch_add -> add_rmsnorm_scale into one launch.

    Bit-identical to the three-kernel chain it replaces: every intermediate
    passes through the same bf16 rounding and every reduction keeps its own
    tree. ``layer_scalar_ptr = 0`` is the null-scalar form. ``out_ptr`` may
    alias ``residual_ptr`` (the production call writes the residual buffer in
    place).
    """

    _check_positive_shape(rows, hidden_size)
    library = library or build_gemma4_norm(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, _SYMBOL_DENSE_COMBINE_BF16, _ARGTYPES_DENSE_COMBINE, ctypes.c_int
    )
    err = fn(
        dense_ptr, experts_ptr, residual_ptr,
        dense_weight_ptr, tail_weight_ptr, layer_scalar_ptr,
        out_ptr, rows, hidden_size, float(eps), stream,
    )
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


# x (in place), total, cap, stream
_ARGTYPES_LOGIT_SOFTCAP = (
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_void_p,
)


def gemma4_logit_softcap_f32(
    x_ptr: int,
    total: int,
    cap: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply ``x = tanh(x / cap) * cap`` in place over ``total`` f32 values.

    The device-side home of ``final_logit_softcapping``: the runner enqueues
    this on the logits device buffer *before* the D2H copy, which removes
    0.528 ms of ``np.tanh`` host time per decode step (X7 measurement, about
    a third of the measured host wall gap). ``apply_softcap=False`` callers
    simply do not launch it.

    ``tanhf`` on the device and numpy's libm tanh round independently, so
    consumers needing sampler guarantees go through the battery in
    ``tests/test_gpu_gemma4_softcap_kernel.py``: elementwise agreement
    within two f32 ulps of the capped range, greedy argmax exact on unique
    tops, tie sets preserved, and the saturating tail equal bitwise.
    """

    if total <= 0:
        raise ValueError("logit softcap needs at least one element")
    library = library or build_gemma4_norm(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, _SYMBOL_LOGIT_SOFTCAP, _ARGTYPES_LOGIT_SOFTCAP, ctypes.c_int
    )
    err = fn(x_ptr, int(total), float(cap), stream)
    _check_launch(runtime, err)


# x, total, scratch, scratch_blocks, out, stream
_ARGTYPES_LOGIT_ARGMAX = (
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_void_p,
    ctypes.c_void_p,
)


def gemma4_logit_argmax_scratch_bytes(total: int) -> int:
    """Device scratch bytes :func:`gemma4_logit_argmax_f32` needs for ``total``.

    Partial slots are ``min(65535, ceil(total / 256))``, each an f32 value
    plus an int64 index; the launcher never writes more than this.
    """

    if total <= 0:
        raise ValueError("logit argmax needs at least one element")
    blocks = min(65535, (total + 255) // 256)
    return max(1, blocks) * (4 + 8)


def gemma4_logit_argmax_f32(
    x_ptr: int,
    total: int,
    out_ptr: int,
    *,
    scratch_ptr: int,
    scratch_blocks: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Argmax over ``total`` f32 values in ``x_ptr``; write index + value.

    ``out_ptr`` must hold 16 bytes: bytes 0..7 the winning ``int64`` index,
    bytes 8..11 the ``f32`` value at that index (bitwise ``x[index]``).
    Semantics are ``np.argmax``'s exactly -- first maximum wins, first NaN
    beats every finite value -- which is what ``Gemma4Runner.next_token``
    runs on the host path today, so the greedy route can swap the transfer
    (12 bytes against the 1 MB vocab copy) without changing a token
    (D10 first half; ``tests/test_gpu_gemma4_argmax_kernel.py`` pins index,
    value, and the chained-after-softcap comparator).

    ``scratch_ptr`` / ``scratch_blocks`` come from
    :func:`gemma4_logit_argmax_scratch_bytes`; callers doing repeated steps
    should keep one scratch buffer alive rather than allocating per call.
    """

    if total <= 0:
        raise ValueError("logit argmax needs at least one element")
    if scratch_blocks <= 0:
        raise ValueError("logit argmax needs a non-empty scratch")
    library = library or build_gemma4_norm(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, _SYMBOL_LOGIT_ARGMAX, _ARGTYPES_LOGIT_ARGMAX, ctypes.c_int
    )
    err = fn(x_ptr, int(total), scratch_ptr, int(scratch_blocks), out_ptr, stream)
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
            KernelKey("hip_gfx1100", "multi_rmsnorm", quant, "gemma4_plain"),
            gemma4_multi_rmsnorm_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "dense_combine_rmsnorm_scale", quant, "gemma4_plain"),
            gemma4_dense_combine_rmsnorm_scale_bf16,
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
