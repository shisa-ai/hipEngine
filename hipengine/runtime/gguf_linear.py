"""Registry-driven GGUF linear dispatch helpers."""

from __future__ import annotations

import contextlib
import ctypes
import os
from dataclasses import dataclass
from typing import Iterator, Mapping

from hipengine.kernels.hip_gfx1100.linear.dense_gemv import register_dense_gemv_kernels
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    gguf_q8_0_dual_gemv_bf16_bf16_out,
    register_gguf_k_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    gguf_q4_k_pack8_dual_prefill_bf16_bf16_out,
    register_gguf_q4_k_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_prefill import (
    gguf_q4_k_wmma_prefill_dual_bf16_bf16_out,
    register_gguf_q4_k_prefill_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
    register_gguf_q6_k_t16_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_prefill import (
    gguf_q8_0_wmma_prefill_dual_gate_up_bf16_bf16_out,
    register_gguf_q8_0_prefill_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_gemv import (
    gguf_q8_0_t16_dual_gate_up_gemv_decode_bf16_bf16_out,
    gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out,
    gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out,
    register_gguf_q8_0_t16_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_prefill import (
    register_gguf_q8_0_t16_prefill_kernels,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_GGUF_Q6_K_T16,
    LAYOUT_GGUF_Q8_0_T16,
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
    Qwen35GGUFDeviceWeight,
)

GGUF_ACTIVATION_BF16 = "bf16"
GGUF_ACTIVATION_F32 = "f32"
GGUF_OUTPUT_BF16 = "bf16"
GGUF_OUTPUT_FP16 = "fp16"
GGUF_OUTPUT_F32 = "f32"

# Opt-in env var for the GGUF WMMA batched prefill family (P8). See
# docs/GGUF.md "P8: real batched prefill GEMM" for the wider plan.
_WMMA_PREFILL_ENV = "HIPENGINE_GGUF_WMMA_PREFILL"

# Session-scoped override; runners can flip this on entry to their bulk
# prefill paths (e.g. from ``PrefillConfig.use_wmma_prefill``). Stays
# ``None`` until set, so the env var still controls the default for plain
# bench/diagnostic invocations.
_wmma_prefill_session_enabled: bool | None = None

# Opt-in env var for the GGUF pack8 GEMV decode family (P9.B). See
# docs/GGUF.md "P9: closing the qwen35moe gap to PARO" for the wider
# plan. This toggles the ``rows == 1`` decode rewrite that routes single-
# token projections through the new ``pack8_gemv_decode_*`` kernels
# (P9.B1-P9.B4b) instead of the legacy ``pack8_gemv_*`` decoders.
_GEMV_DECODE_ENV = "HIPENGINE_GGUF_GEMV_DECODE"
_gemv_decode_session_enabled: bool | None = None

# Quants currently shipping a batched ``wmma_prefill_*`` family. Values are
# the raw GGUF K-block alignment constraints enforced before dispatching to
# the WMMA wrappers. Q4_K is raw-layout only for now: dense 2D Q4_K resident
# weights are still materialized as the lossless pack8 fallback layout, so
# they never reach the raw WMMA ABI unless a caller explicitly has raw bytes.
_WMMA_PREFILL_QUANT_BLOCKS: Mapping[str, int] = {
    "gguf_q8_0": 32,
    "gguf_q4_k": 256,
    # P10.B4: Q8T16 dense WMMA prefill consumes T16 tiles with 32 K-values
    # per tile slab. Same block alignment as raw Q8_0.
    "gguf_q8_0_t16_v1": 32,
}


@dataclass(frozen=True)
class GGUFLinearDispatch:
    """Resolved kernel key and ABI family for one GGUF linear launch."""

    key: KernelKey
    abi: str


_DISPATCH_TABLE: Mapping[tuple[str, str, str], GGUFLinearDispatch] = {
    (LAYOUT_Q4_K_PACK8, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "pack8_bf16_bf16_out"),
        "pack8",
    ),
    (LAYOUT_Q4_K_PACK8, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_FP16): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "pack8_bf16_fp16_out"),
        "pack8",
    ),
    (LAYOUT_Q4_K_PACK8, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_F32): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "pack8_bf16_f32_out"),
        "pack8",
    ),
    (LAYOUT_RAW_GGUF, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "<from-weight>", "gemv_bf16_bf16_out"),
        "raw",
    ),
    (LAYOUT_RAW_GGUF, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_FP16): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "<from-weight>", "gemv_bf16_fp16_out"),
        "raw",
    ),
    (LAYOUT_RAW_GGUF, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_F32): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "<from-weight>", "gemv_bf16_f32_out"),
        "raw",
    ),
    (LAYOUT_DENSE_BF16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "dense_gemv", "bf16", "out"),
        "dense_bf16",
    ),
    (LAYOUT_GGUF_Q6_K_T16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_F32): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q6_k_t16_v1", "t16_gemv_decode_bf16_f32_out"),
        "t16",
    ),
    (LAYOUT_GGUF_Q8_0_T16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_bf16_bf16_out"),
        "t16",
    ),
    (LAYOUT_GGUF_Q8_0_T16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_FP16): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_fp16_fp16_out"),
        "t16",
    ),
    (LAYOUT_GGUF_Q8_0_T16, GGUF_ACTIVATION_F32, GGUF_OUTPUT_BF16): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_f32_bf16_out"),
        "t16",
    ),
}


def set_gemv_decode_enabled(enabled: bool | None) -> None:
    """Set the session-scoped opt-in for the GGUF pack8 GEMV decode family.

    Pass ``True`` / ``False`` to override env + per-call kwargs for this
    process. Pass ``None`` to clear the override and fall back to the env
    var (``HIPENGINE_GGUF_GEMV_DECODE``). Intended to be called once by a
    runner that drives ``Qwen35GGUFResidentSession.use_gemv_decode`` from
    its public API. The kwarg path remains available for ad-hoc bisects.
    """

    global _gemv_decode_session_enabled
    _gemv_decode_session_enabled = None if enabled is None else bool(enabled)


@contextlib.contextmanager
def gemv_decode_session(enabled: bool | None) -> Iterator[None]:
    """Context manager wrapper around :func:`set_gemv_decode_enabled`."""

    previous = _gemv_decode_session_enabled
    set_gemv_decode_enabled(enabled)
    try:
        yield
    finally:
        set_gemv_decode_enabled(previous)


def _env_gemv_decode_enabled() -> bool:
    raw = os.environ.get(_GEMV_DECODE_ENV, "")
    if not raw:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def gguf_gemv_decode_enabled(use_gemv_decode: bool | None = None) -> bool:
    """Return the resolved GGUF pack8 GEMV decode opt-in state.

    Precedence (highest first): explicit kwarg, session toggle, env var.
    Mirrors :func:`gguf_wmma_prefill_enabled` for the decode-side rewrite.
    """

    return _resolve_use_gemv_decode(use_gemv_decode)


def _resolve_use_gemv_decode(kwarg: bool | None) -> bool:
    if kwarg is not None:
        return bool(kwarg)
    if _gemv_decode_session_enabled is not None:
        return _gemv_decode_session_enabled
    return _env_gemv_decode_enabled()


def set_wmma_prefill_enabled(enabled: bool | None) -> None:
    """Set the session-scoped opt-in for the GGUF WMMA prefill family.

    Pass ``True`` / ``False`` to override env + per-call kwargs for this
    process. Pass ``None`` to clear the override and fall back to the env
    var (``HIPENGINE_GGUF_WMMA_PREFILL``). Intended to be called once by a
    runner that drives ``PrefillConfig.use_wmma_prefill`` from its public
    API. The kwarg path remains available for ad-hoc bisects.
    """

    global _wmma_prefill_session_enabled
    _wmma_prefill_session_enabled = None if enabled is None else bool(enabled)


@contextlib.contextmanager
def wmma_prefill_session(enabled: bool | None) -> Iterator[None]:
    """Context manager wrapper around :func:`set_wmma_prefill_enabled`."""

    previous = _wmma_prefill_session_enabled
    set_wmma_prefill_enabled(enabled)
    try:
        yield
    finally:
        set_wmma_prefill_enabled(previous)


def _env_wmma_prefill_enabled() -> bool:
    raw = os.environ.get(_WMMA_PREFILL_ENV, "")
    if not raw:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def gguf_wmma_prefill_enabled(use_wmma_prefill: bool | None = None) -> bool:
    """Return the resolved GGUF WMMA prefill opt-in state.

    This exposes the same precedence used by :func:`launch_gguf_linear` so
    higher-level runners can route composite GGUF prefill paths without
    duplicating env-var or session-toggle checks.
    """

    return _resolve_use_wmma_prefill(use_wmma_prefill)


def _resolve_use_wmma_prefill(kwarg: bool | None) -> bool:
    """Combine per-call kwarg + session toggle + env var.

    Precedence (highest first): explicit kwarg, session toggle, env var.
    """

    if kwarg is not None:
        return bool(kwarg)
    if _wmma_prefill_session_enabled is not None:
        return _wmma_prefill_session_enabled
    return _env_wmma_prefill_enabled()


def resolve_gguf_linear_dispatch(
    weight: Qwen35GGUFDeviceWeight,
    *,
    activation_dtype: str = GGUF_ACTIVATION_BF16,
    output_dtype: str = GGUF_OUTPUT_BF16,
    backend: str = "hip_gfx1100",
    rows: int = 1,
) -> GGUFLinearDispatch:
    """Resolve a GGUF linear launch without model/engine quant branches."""

    table_key = (weight.spec.layout, activation_dtype, output_dtype)
    try:
        dispatch = _DISPATCH_TABLE[table_key]
    except KeyError as exc:
        raise ValueError(
            "unsupported GGUF linear dispatch: "
            f"layout={weight.spec.layout!r}, activation={activation_dtype!r}, output={output_dtype!r}"
        ) from exc
    quant = weight.spec.quant_key if dispatch.key.quant == "<from-weight>" else dispatch.key.quant
    variant = _variant_for_rows(dispatch.key.variant, rows=rows)
    return GGUFLinearDispatch(
        KernelKey(backend, dispatch.key.layer, quant, variant),
        dispatch.abi,
    )


def launch_gguf_linear(
    weight: Qwen35GGUFDeviceWeight,
    x_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    activation_dtype: str = GGUF_ACTIVATION_BF16,
    output_dtype: str = GGUF_OUTPUT_BF16,
    backend: str = "hip_gfx1100",
    threads: int = 0,
    stream: int = 0,
    libraries: Mapping[str, ctypes.CDLL] | None = None,
    runtime=None,
    use_wmma_prefill: bool | None = None,
    use_gemv_decode: bool | None = None,
) -> None:
    """Launch a GGUF resident linear projection through the kernel registry.

    Hidden projections use ``output_dtype='bf16'``. The tied Q6_K lm-head path
    uses ``output_dtype='f32'`` to produce logits.

    When ``rows > 1`` and the raw-layout quant has a WMMA prefill kernel
    registered (currently ``gguf_q8_0`` and raw ``gguf_q4_k``), the dispatch
    rewrites to the ``wmma_prefill_*`` family if any of these is true:

    * ``use_wmma_prefill=True`` is passed explicitly,
    * a runner has called :func:`set_wmma_prefill_enabled` with ``True``,
    * the env var ``HIPENGINE_GGUF_WMMA_PREFILL`` is set.

    Otherwise the existing decode-shaped ``prefill_*`` aliases run.
    """

    dispatch = resolve_gguf_linear_dispatch(
        weight,
        activation_dtype=activation_dtype,
        output_dtype=output_dtype,
        backend=backend,
        rows=rows,
    )
    dispatch = _pack8_decode_dispatch(dispatch, rows=rows, out_features=out_features)
    dispatch = _gemv_decode_dispatch(
        dispatch,
        rows=rows,
        use_gemv_decode=_resolve_use_gemv_decode(use_gemv_decode),
    )
    dispatch = _wmma_prefill_dispatch(
        dispatch,
        rows=rows,
        in_features=in_features,
        use_wmma=_resolve_use_wmma_prefill(use_wmma_prefill),
    )
    _ensure_linear_kernel_registered(dispatch.key)
    fn = resolve(
        backend=dispatch.key.backend,
        layer=dispatch.key.layer,
        quant=dispatch.key.quant,
        variant=dispatch.key.variant,
    )
    library = None if libraries is None else libraries.get(dispatch.key.quant)
    kwargs = {"stream": stream, "runtime": runtime}
    if threads:
        kwargs["threads"] = threads
    if library is not None:
        kwargs["library"] = library
    _LAUNCH_ABI[dispatch.abi](fn, weight, x_ptr, out_ptr, rows, in_features, out_features, kwargs)


def launch_gguf_linear_raw_ptr(
    weight: Qwen35GGUFDeviceWeight,
    qweight_ptr: int,
    x_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    activation_dtype: str = GGUF_ACTIVATION_BF16,
    output_dtype: str = GGUF_OUTPUT_BF16,
    backend: str = "hip_gfx1100",
    threads: int = 0,
    stream: int = 0,
    libraries: Mapping[str, ctypes.CDLL] | None = None,
    runtime=None,
    use_wmma_prefill: bool | None = None,
) -> None:
    """Launch a raw GGUF linear using an already offset qweight pointer.

    Rank-3 MoE expert tensors are materialized as one contiguous raw GGUF
    allocation.  The caller selects an expert by offsetting into that allocation,
    while dispatch still resolves from the original logical weight spec.
    """

    dispatch = resolve_gguf_linear_dispatch(
        weight,
        activation_dtype=activation_dtype,
        output_dtype=output_dtype,
        backend=backend,
        rows=rows,
    )
    if dispatch.abi != "raw":
        raise ValueError(f"raw-pointer GGUF launch requires raw layout, got {weight.spec.layout!r}")
    dispatch = _wmma_prefill_dispatch(
        dispatch,
        rows=rows,
        in_features=in_features,
        use_wmma=_resolve_use_wmma_prefill(use_wmma_prefill),
    )
    _ensure_linear_kernel_registered(dispatch.key)
    fn = resolve(
        backend=dispatch.key.backend,
        layer=dispatch.key.layer,
        quant=dispatch.key.quant,
        variant=dispatch.key.variant,
    )
    library = None if libraries is None else libraries.get(dispatch.key.quant)
    kwargs = {"stream": stream, "runtime": runtime}
    if threads and dispatch.abi != "wmma_raw":
        # The WMMA wrapper takes (tile_m, tile_n) instead of (threads); the
        # caller-supplied ``threads`` value applies to the decode-shaped path
        # only and is silently dropped on the WMMA path.
        kwargs["threads"] = threads
    if library is not None:
        kwargs["library"] = library
    fn(x_ptr, int(qweight_ptr), out_ptr, rows, in_features, out_features, **kwargs)


def launch_gguf_linear_pair(
    weight_a: Qwen35GGUFDeviceWeight,
    weight_b: Qwen35GGUFDeviceWeight,
    x_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    out_features_b: int | None = None,
    stream: int = 0,
    runtime=None,
    use_wmma_prefill: bool | None = None,
    use_gemv_decode: bool | None = None,
) -> bool:
    """Launch a supported pair of GGUF projections, returning True when fused.

    The pair fast paths cover Q8_0 dual decode GEMV, Q4_K pack8 dual prefill,
    and the P8.2 raw-Q4_K dual WMMA prefill. There is still no Q8_0 dual WMMA
    prefill; when ``use_wmma_prefill`` would otherwise route Q8_0 rows>1 to
    the WMMA family, the pair function returns ``False`` so the caller falls
    back to two singletons that each take the WMMA path via
    :func:`launch_gguf_linear`.

    When ``use_gemv_decode`` is enabled (kwarg / session / env opt-in) and
    ``rows == 1`` with a registered Q8_0 dual gate+up GEMV decode kernel,
    the pair is fused through :func:`gguf_q8_0_pack8_dual_gate_up_gemv_decode_bf16_bf16_out`
    (P9.B3); the output layout matches the legacy ``gguf_q8_0_dual_gemv``
    concatenated layout that ``silu_mul_dual_out_*`` consumes downstream.
    """

    use_wmma = _resolve_use_wmma_prefill(use_wmma_prefill)
    use_gemv = _resolve_use_gemv_decode(use_gemv_decode)
    out_features_b = out_features if out_features_b is None else int(out_features_b)
    dispatch_a = _pack8_decode_dispatch(
        resolve_gguf_linear_dispatch(weight_a, rows=rows),
        rows=rows,
        out_features=out_features,
    )
    dispatch_b = _pack8_decode_dispatch(
        resolve_gguf_linear_dispatch(weight_b, rows=rows),
        rows=rows,
        out_features=out_features,
    )
    if use_wmma and rows > 1:
        q4_prefill_raw = KernelKey(
            "hip_gfx1100", "linear", "gguf_q4_k", "prefill_bf16_bf16_out"
        )
        if (
            out_features_b == out_features
            and dispatch_a.abi == "raw"
            and dispatch_b.abi == "raw"
            and dispatch_a.key == q4_prefill_raw
            and dispatch_b.key == q4_prefill_raw
            and _wmma_prefill_shape_supported("gguf_q4_k", in_features)
        ):
            gguf_q4_k_wmma_prefill_dual_bf16_bf16_out(
                x_ptr,
                weight_a.allocation("raw").tensor.ptr,
                weight_b.allocation("raw").tensor.ptr,
                out_a_ptr,
                out_b_ptr,
                rows,
                in_features,
                out_features,
                stream=stream,
                runtime=runtime,
            )
            return True

        # If either side would be routed to a WMMA prefill singleton that does
        # not have a dual pair path here (currently Q8_0), decline the pair
        # fusion so the caller falls back to two singletons (each picks up the
        # WMMA family via launch_gguf_linear).
        for d in (dispatch_a, dispatch_b):
            if _dispatch_can_use_wmma_prefill(d, rows=rows, in_features=in_features):
                return False
    q8_t16_dual = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q8_0_t16_v1",
        "t16_dual_gemv_decode_bf16_bf16_out",
    )
    if (
        dispatch_a.abi == "t16"
        and dispatch_b.abi == "t16"
        and dispatch_a.key.quant == "gguf_q8_0_t16_v1"
        and dispatch_b.key.quant == "gguf_q8_0_t16_v1"
        and is_registered(q8_t16_dual)
    ):
        # P10.B4: decline the Q8T16 dual GEMV fusion at rows>1 when WMMA
        # prefill is opted in, so the caller falls back to two singletons
        # that each take the dense Q8T16 WMMA prefill path.
        if use_wmma and rows > 1 and (
            _dispatch_can_use_t16_wmma_prefill(dispatch_a, rows=rows, in_features=in_features)
            or _dispatch_can_use_t16_wmma_prefill(dispatch_b, rows=rows, in_features=in_features)
        ):
            return False
        gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out(
            x_ptr,
            weight_a.allocation("tiles").tensor.ptr,
            weight_b.allocation("tiles").tensor.ptr,
            out_a_ptr,
            out_b_ptr,
            rows,
            in_features,
            out_features,
            out_features_b,
            stream=stream,
            runtime=runtime,
        )
        return True

    q8_decode = KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "pack8_gemv_bf16_bf16_out")
    if rows == 1 and out_features_b == out_features and dispatch_a.key == q8_decode and dispatch_b.key == q8_decode:
        gguf_q8_0_dual_gemv_bf16_bf16_out(
            x_ptr,
            weight_a.allocation("raw").tensor.ptr,
            weight_b.allocation("raw").tensor.ptr,
            out_a_ptr,
            out_b_ptr,
            rows,
            in_features,
            out_features,
            stream=stream,
            runtime=runtime,
        )
        return True

    q4_prefill = KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "pack8_prefill_bf16_bf16_out")
    if rows > 1 and out_features_b == out_features and dispatch_a.key == q4_prefill and dispatch_b.key == q4_prefill:
        gguf_q4_k_pack8_dual_prefill_bf16_bf16_out(
            x_ptr,
            weight_a.allocation("qweight").tensor.ptr,
            weight_a.allocation("scales").tensor.ptr,
            weight_a.allocation("mins").tensor.ptr,
            weight_b.allocation("qweight").tensor.ptr,
            weight_b.allocation("scales").tensor.ptr,
            weight_b.allocation("mins").tensor.ptr,
            out_a_ptr,
            out_b_ptr,
            rows,
            in_features,
            out_features,
            stream=stream,
            runtime=runtime,
        )
        return True
    return False


def launch_gguf_linear_triple(
    weight_a: Qwen35GGUFDeviceWeight,
    weight_b: Qwen35GGUFDeviceWeight,
    weight_c: Qwen35GGUFDeviceWeight,
    x_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    out_c_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    out_features_b: int | None = None,
    out_features_c: int | None = None,
    stream: int = 0,
    runtime=None,
) -> bool:
    """Launch a supported same-input triple of GGUF projections."""

    out_features_b = out_features if out_features_b is None else int(out_features_b)
    out_features_c = out_features if out_features_c is None else int(out_features_c)
    dispatch_a = _pack8_decode_dispatch(
        resolve_gguf_linear_dispatch(weight_a, rows=rows),
        rows=rows,
        out_features=out_features,
    )
    dispatch_b = _pack8_decode_dispatch(
        resolve_gguf_linear_dispatch(weight_b, rows=rows),
        rows=rows,
        out_features=out_features_b,
    )
    dispatch_c = _pack8_decode_dispatch(
        resolve_gguf_linear_dispatch(weight_c, rows=rows),
        rows=rows,
        out_features=out_features_c,
    )
    use_wmma = _resolve_use_wmma_prefill(None)
    q8_t16_triple = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q8_0_t16_v1",
        "t16_triple_gemv_decode_bf16_bf16_out",
    )
    if use_wmma and rows > 1 and (
        _dispatch_can_use_t16_wmma_prefill(dispatch_a, rows=rows, in_features=in_features)
        or _dispatch_can_use_t16_wmma_prefill(dispatch_b, rows=rows, in_features=in_features)
        or _dispatch_can_use_t16_wmma_prefill(dispatch_c, rows=rows, in_features=in_features)
    ):
        # P10.B4: decline Q8T16 triple GEMV fusion at rows>1 when WMMA
        # prefill is opted in, so the caller falls back to singletons that
        # each take the dense Q8T16 WMMA prefill path.
        return False
    if (
        dispatch_a.abi == "t16"
        and dispatch_b.abi == "t16"
        and dispatch_c.abi == "t16"
        and dispatch_a.key.quant == "gguf_q8_0_t16_v1"
        and dispatch_b.key.quant == "gguf_q8_0_t16_v1"
        and dispatch_c.key.quant == "gguf_q8_0_t16_v1"
        and is_registered(q8_t16_triple)
    ):
        gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out(
            x_ptr,
            weight_a.allocation("tiles").tensor.ptr,
            weight_b.allocation("tiles").tensor.ptr,
            weight_c.allocation("tiles").tensor.ptr,
            out_a_ptr,
            out_b_ptr,
            out_c_ptr,
            rows,
            in_features,
            out_features,
            out_features_b,
            out_features_c,
            stream=stream,
            runtime=runtime,
        )
        return True
    return False


def launch_gguf_linear_pair_concat(
    weight_a: Qwen35GGUFDeviceWeight,
    weight_b: Qwen35GGUFDeviceWeight,
    x_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    runtime=None,
    use_wmma_prefill: bool | None = None,
    use_gemv_decode: bool | None = None,
) -> bool:
    """Launch a supported projection pair into one concatenated output buffer.

    This is the prefill-side companion to :func:`launch_gguf_linear_pair` for
    kernels whose natural ABI is ``[rows, out_a + out_b]``. P9.C1 uses it for
    the Q8_0 shared-expert gate+up WMMA prefill path so the downstream
    ``silu_mul_dual_out_*`` kernel can consume the same layout as the selected
    MoE gate+up path. P9.H3 also uses it for resident Q8T16 shared gate/up
    decode so the two Q8_0 projections share one T16 kernel launch family.
    """

    use_wmma = _resolve_use_wmma_prefill(use_wmma_prefill)
    dispatch_a = resolve_gguf_linear_dispatch(weight_a, rows=rows)
    dispatch_b = resolve_gguf_linear_dispatch(weight_b, rows=rows)
    q8_t16_dual = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q8_0_t16_v1",
        "t16_dual_gate_up_gemv_decode_bf16_bf16_out",
    )
    if (
        dispatch_a.abi == "t16"
        and dispatch_b.abi == "t16"
        and dispatch_a.key.quant == "gguf_q8_0_t16_v1"
        and dispatch_b.key.quant == "gguf_q8_0_t16_v1"
        and is_registered(q8_t16_dual)
    ):
        # P10.B4: decline the Q8T16 dual-gate-up GEMV fusion at rows>1 when
        # WMMA prefill is opted in, so the caller falls back through
        # ``launch_gguf_linear_pair`` (which itself declines T16 fusion when
        # WMMA prefill is on) all the way down to two singletons that each
        # take the dense Q8T16 WMMA prefill path.
        if use_wmma and rows > 1 and (
            _dispatch_can_use_t16_wmma_prefill(dispatch_a, rows=rows, in_features=in_features)
            or _dispatch_can_use_t16_wmma_prefill(dispatch_b, rows=rows, in_features=in_features)
        ):
            return False
        gguf_q8_0_t16_dual_gate_up_gemv_decode_bf16_bf16_out(
            x_ptr,
            weight_a.allocation("tiles").tensor.ptr,
            weight_b.allocation("tiles").tensor.ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            out_features,
            stream=stream,
            runtime=runtime,
        )
        return True

    if not use_wmma or rows <= 1:
        return False
    q8_prefill_raw = KernelKey(
        "hip_gfx1100", "linear", "gguf_q8_0", "prefill_bf16_bf16_out"
    )
    q8_dual = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q8_0",
        "wmma_prefill_dual_gate_up_bf16_bf16_out",
    )
    if (
        dispatch_a.abi == "raw"
        and dispatch_b.abi == "raw"
        and dispatch_a.key == q8_prefill_raw
        and dispatch_b.key == q8_prefill_raw
        and _wmma_prefill_shape_supported("gguf_q8_0", in_features)
        and is_registered(q8_dual)
    ):
        gguf_q8_0_wmma_prefill_dual_gate_up_bf16_bf16_out(
            x_ptr,
            weight_a.allocation("raw").tensor.ptr,
            weight_b.allocation("raw").tensor.ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            out_features,
            tile_m=16,
            tile_n=32,
            stream=stream,
            runtime=runtime,
        )
        return True
    return False


def _launch_pack8(fn, weight, x_ptr, out_ptr, rows, in_features, out_features, kwargs) -> None:
    fn(
        x_ptr,
        weight.allocation("qweight").tensor.ptr,
        weight.allocation("scales").tensor.ptr,
        weight.allocation("mins").tensor.ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    )


def _launch_raw(fn, weight, x_ptr, out_ptr, rows, in_features, out_features, kwargs) -> None:
    fn(
        x_ptr,
        weight.allocation("raw").tensor.ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    )


def _launch_dense_bf16(fn, weight, x_ptr, out_ptr, rows, in_features, out_features, kwargs) -> None:
    fn(
        x_ptr,
        weight.allocation("raw").tensor.ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    )


def _launch_t16(fn, weight, x_ptr, out_ptr, rows, in_features, out_features, kwargs) -> None:
    fn(
        x_ptr,
        weight.allocation("tiles").tensor.ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    )


def _pack8_decode_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    out_features: int,
) -> GGUFLinearDispatch:
    if (
        dispatch.abi == "raw"
        and rows == 1
        and out_features % 8 == 0
        and dispatch.key.quant in {"gguf_q8_0", "gguf_q5_k", "gguf_q6_k"}
        and dispatch.key.variant in {"gemv_bf16_bf16_out", "gemv_bf16_f32_out"}
    ):
        return GGUFLinearDispatch(
            KernelKey(
                dispatch.key.backend,
                dispatch.key.layer,
                dispatch.key.quant,
                f"pack8_{dispatch.key.variant}",
            ),
            dispatch.abi,
        )
    return dispatch


def _gemv_decode_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    use_gemv_decode: bool,
) -> GGUFLinearDispatch:
    """Rewrite ``pack8_gemv_*`` -> ``pack8_gemv_decode_*`` for supported quants.

    A no-op unless all of the following hold:

    * ``use_gemv_decode`` is ``True`` (kwarg / session / env opt-in resolved).
    * ``rows == 1`` (prefill / bulk paths are not affected).
    * ``dispatch.abi == "raw"`` (the new GEMV decode kernel reads raw GGUF
      bytes via the same single ``raw`` allocation as the legacy decoder).
    * ``dispatch.key.quant`` ships a registered ``pack8_gemv_decode_*`` family
      (currently P9.B3 ``gguf_q8_0``; the Q5_K/Q6_K dense decode variants
      added in P9.B4b cover the lm-head case via separate runner wiring).
    * ``dispatch.key.variant`` is one of the ``pack8_gemv_*`` aliases
      (i.e. ``_pack8_decode_dispatch`` already rewrote the raw decoder).
    * The rewritten registry key is actually registered. If not, the
      function returns the original ``dispatch`` unchanged so the runtime
      transparently falls back to the legacy decoder.
    """

    if not use_gemv_decode or rows != 1:
        return dispatch
    if dispatch.abi != "raw":
        return dispatch
    variant = dispatch.key.variant
    if not variant.startswith("pack8_gemv_") or variant.startswith("pack8_gemv_decode_"):
        return dispatch
    rewritten_variant = f"pack8_gemv_decode_{variant[len('pack8_gemv_') :]}"
    rewritten_key = KernelKey(
        dispatch.key.backend,
        dispatch.key.layer,
        dispatch.key.quant,
        rewritten_variant,
    )
    if not is_registered(rewritten_key):
        # Registry miss: fall back to the legacy decoder without raising.
        # ``is_registered`` is an exact-key check so the cpu_reference fp16
        # ``linear`` catch-all does not silently route to a kernel whose
        # ABI does not match the GGUF launcher.
        return dispatch
    return GGUFLinearDispatch(rewritten_key, dispatch.abi)


def _wmma_prefill_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
    use_wmma: bool,
) -> GGUFLinearDispatch:
    """Rewrite decode-shape variants -> ``wmma_prefill_*`` for supported quants.

    A no-op unless all of the following hold:

    * ``use_wmma`` is ``True`` (kwarg / session / env opt-in resolved).
    * ``rows > 1`` (decode is not affected).
    * ``dispatch.abi`` is one of the supported source ABIs:
      - ``"raw"`` -> ``"wmma_raw"`` (the legacy raw-GGUF WMMA prefill
        family for ``gguf_q8_0`` and raw-layout ``gguf_q4_k``).
      - ``"t16"`` -> ``"t16"`` (P10.B4: ``gguf_q8_0_t16_v1`` rewrites the
        ``t16_gemv_decode_*`` variant to ``t16_wmma_prefill_*`` and keeps
        the same allocation name + launch signature, so the existing
        ``_launch_t16`` ABI helper is reused).
    * ``dispatch.key.quant`` ships a registered WMMA prefill family.
    * ``dispatch.key.variant`` is the rows>1 alias produced by
      ``_variant_for_rows``.
    * ``in_features`` satisfies the quant's K-block alignment constraint.
    """

    if not use_wmma or rows <= 1:
        return dispatch
    if dispatch.abi == "raw":
        if not _dispatch_can_use_wmma_prefill(dispatch, rows=rows, in_features=in_features):
            return dispatch
        variant = dispatch.key.variant
        return GGUFLinearDispatch(
            KernelKey(
                dispatch.key.backend,
                dispatch.key.layer,
                dispatch.key.quant,
                f"wmma_{variant}",
            ),
            "wmma_raw",
        )
    if dispatch.abi == "t16":
        if not _dispatch_can_use_t16_wmma_prefill(dispatch, rows=rows, in_features=in_features):
            return dispatch
        # The T16 decode variant is named ``t16_gemv_decode_<in>_<out>_out``;
        # the rewrite swaps that for ``t16_wmma_prefill_<in>_<out>_out`` while
        # keeping the ``t16`` ABI (same (x, tiles, out, rows, in_f, out_f)
        # signature, additional (tile_m, tile_n) kwargs).
        variant = dispatch.key.variant
        if not variant.startswith("t16_gemv_decode_"):
            return dispatch
        suffix = variant[len("t16_gemv_decode_") :]
        return GGUFLinearDispatch(
            KernelKey(
                dispatch.key.backend,
                dispatch.key.layer,
                dispatch.key.quant,
                f"t16_wmma_prefill_{suffix}",
            ),
            "t16",
        )
    return dispatch


def _dispatch_can_use_t16_wmma_prefill(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
) -> bool:
    """P10.B4 gate: T16 dense rows>1 only rewrites when the kernel is wired."""

    return (
        rows > 1
        and dispatch.abi == "t16"
        and dispatch.key.variant.startswith("t16_gemv_decode_")
        and dispatch.key.quant in _WMMA_PREFILL_QUANT_BLOCKS
        and dispatch.key.quant.endswith("_t16_v1")
        and _wmma_prefill_shape_supported(dispatch.key.quant, in_features)
    )


def _wmma_prefill_shape_supported(quant: str, in_features: int) -> bool:
    block = _WMMA_PREFILL_QUANT_BLOCKS.get(quant)
    return block is not None and in_features % block == 0


def _dispatch_can_use_wmma_prefill(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
) -> bool:
    return (
        rows > 1
        and dispatch.abi == "raw"
        and dispatch.key.variant.startswith("prefill_")
        and _wmma_prefill_shape_supported(dispatch.key.quant, in_features)
    )


def _variant_for_rows(variant: str, *, rows: int) -> str:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if rows == 1:
        return variant
    if variant.startswith("pack8_"):
        return f"pack8_prefill_{variant[len('pack8_') :]}"
    if variant.startswith("gemv_"):
        return f"prefill_{variant[len('gemv_') :]}"
    if variant == "out":
        return "prefill_out"
    return variant


def _launch_wmma_raw(fn, weight, x_ptr, out_ptr, rows, in_features, out_features, kwargs) -> None:
    # The WMMA prefill wrapper has the same (x, qweight, out, rows, in_f, out_f)
    # raw-pointer signature as _launch_raw, but accepts (tile_m, tile_n, stream)
    # in place of (threads, stream). Strip ``threads`` if the caller set it.
    wmma_kwargs = {k: v for k, v in kwargs.items() if k != "threads"}
    fn(
        x_ptr,
        weight.allocation("raw").tensor.ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **wmma_kwargs,
    )


def _ensure_linear_kernel_registered(key: KernelKey) -> None:
    # Registry plan tests clear global registrations; keep GGUF runtime dispatch
    # independent of previous test/import order without overwriting tests that
    # deliberately replace one dispatch key with a fixture kernel.
    if resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
        missing="none",
    ) is not None:
        return
    register_dense_gemv_kernels()
    register_gguf_k_gemv_kernels()
    register_gguf_q4_k_gemv_kernels()
    register_gguf_q4_k_prefill_kernels()
    register_gguf_q6_k_t16_gemv_kernels()
    register_gguf_q8_0_prefill_kernels()
    register_gguf_q8_0_t16_gemv_kernels()
    register_gguf_q8_0_t16_prefill_kernels()


_LAUNCH_ABI = {
    "dense_bf16": _launch_dense_bf16,
    "pack8": _launch_pack8,
    "raw": _launch_raw,
    "t16": _launch_t16,
    "wmma_raw": _launch_wmma_raw,
}


__all__ = [
    "GGUF_ACTIVATION_BF16",
    "GGUF_ACTIVATION_F32",
    "GGUF_OUTPUT_BF16",
    "GGUF_OUTPUT_FP16",
    "GGUF_OUTPUT_F32",
    "GGUFLinearDispatch",
    "gguf_wmma_prefill_enabled",
    "launch_gguf_linear",
    "launch_gguf_linear_pair",
    "launch_gguf_linear_pair_concat",
    "launch_gguf_linear_raw_ptr",
    "launch_gguf_linear_triple",
    "resolve_gguf_linear_dispatch",
    "set_wmma_prefill_enabled",
    "wmma_prefill_session",
]
