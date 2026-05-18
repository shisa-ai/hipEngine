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
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_prefill import (
    register_gguf_q8_0_prefill_kernels,
)
from hipengine.kernels.registry import KernelKey, resolve
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
    Qwen35GGUFDeviceWeight,
)

GGUF_ACTIVATION_BF16 = "bf16"
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

# Quants currently shipping a batched ``wmma_prefill_*`` family. Values are
# the raw GGUF K-block alignment constraints enforced before dispatching to
# the WMMA wrappers. Q4_K is raw-layout only for now: dense 2D Q4_K resident
# weights are still materialized as the lossless pack8 fallback layout, so
# they never reach the raw WMMA ABI unless a caller explicitly has raw bytes.
_WMMA_PREFILL_QUANT_BLOCKS: Mapping[str, int] = {
    "gguf_q8_0": 32,
    "gguf_q4_k": 256,
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
}


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
    stream: int = 0,
    runtime=None,
    use_wmma_prefill: bool | None = None,
) -> bool:
    """Launch a supported pair of GGUF projections, returning True when fused.

    The pair fast paths cover Q8_0 dual decode GEMV, Q4_K pack8 dual prefill,
    and the P8.2 raw-Q4_K dual WMMA prefill. There is still no Q8_0 dual WMMA
    prefill; when ``use_wmma_prefill`` would otherwise route Q8_0 rows>1 to
    the WMMA family, the pair function returns ``False`` so the caller falls
    back to two singletons that each take the WMMA path via
    :func:`launch_gguf_linear`.
    """

    use_wmma = _resolve_use_wmma_prefill(use_wmma_prefill)
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
            dispatch_a.abi == "raw"
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
    q8_decode = KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "pack8_gemv_bf16_bf16_out")
    if rows == 1 and dispatch_a.key == q8_decode and dispatch_b.key == q8_decode:
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
    if rows > 1 and dispatch_a.key == q4_prefill and dispatch_b.key == q4_prefill:
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


def _wmma_prefill_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
    use_wmma: bool,
) -> GGUFLinearDispatch:
    """Rewrite ``prefill_*`` -> ``wmma_prefill_*`` for supported quants.

    A no-op unless all of the following hold:

    * ``use_wmma`` is ``True`` (kwarg / session / env opt-in resolved).
    * ``rows > 1`` (decode is not affected).
    * ``dispatch.abi == "raw"`` (the WMMA kernel consumes raw GGUF bytes
      via the same single ``raw`` allocation as the decode-shaped path).
    * ``dispatch.key.quant`` ships a registered WMMA prefill family
      (``gguf_q8_0`` or raw-layout ``gguf_q4_k``).
    * ``dispatch.key.variant`` is one of the ``prefill_*`` aliases (i.e.
      the rows>1 rewrite from ``_variant_for_rows`` already happened).
    * ``in_features`` satisfies the quant's raw block-size constraint
      (32 for Q8_0, 256 for Q4_K).
    """

    if not use_wmma or rows <= 1:
        return dispatch
    if dispatch.abi != "raw":
        return dispatch
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
    register_gguf_q8_0_prefill_kernels()


_LAUNCH_ABI = {
    "dense_bf16": _launch_dense_bf16,
    "pack8": _launch_pack8,
    "raw": _launch_raw,
    "wmma_raw": _launch_wmma_raw,
}


__all__ = [
    "GGUF_ACTIVATION_BF16",
    "GGUF_OUTPUT_BF16",
    "GGUF_OUTPUT_FP16",
    "GGUF_OUTPUT_F32",
    "GGUFLinearDispatch",
    "launch_gguf_linear",
    "launch_gguf_linear_pair",
    "launch_gguf_linear_raw_ptr",
    "resolve_gguf_linear_dispatch",
    "set_wmma_prefill_enabled",
    "wmma_prefill_session",
]
