"""Registry-driven GGUF linear dispatch helpers."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Mapping

from hipengine.kernels.hip_gfx1100.linear.dense_gemv import register_dense_gemv_kernels
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    gguf_q8_0_dual_gemv_bf16_bf16_out,
    register_gguf_k_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    gguf_q4_k_pack8_dual_prefill_bf16_bf16_out,
    register_gguf_q4_k_gemv_kernels,
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
) -> None:
    """Launch a GGUF resident linear projection through the kernel registry.

    Hidden projections use ``output_dtype='bf16'``. The tied Q6_K lm-head path
    uses ``output_dtype='f32'`` to produce logits.
    """

    dispatch = resolve_gguf_linear_dispatch(
        weight,
        activation_dtype=activation_dtype,
        output_dtype=output_dtype,
        backend=backend,
        rows=rows,
    )
    dispatch = _pack8_decode_dispatch(dispatch, rows=rows, out_features=out_features)
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
) -> bool:
    """Launch a supported pair of GGUF projections, returning True when fused."""

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


_LAUNCH_ABI = {
    "dense_bf16": _launch_dense_bf16,
    "pack8": _launch_pack8,
    "raw": _launch_raw,
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
]
