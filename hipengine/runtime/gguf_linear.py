"""Registry-driven GGUF linear dispatch helpers."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Mapping

from hipengine.kernels.registry import KernelKey, resolve
from hipengine.loading.qwen35_gguf_materialize import (
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


def _variant_for_rows(variant: str, *, rows: int) -> str:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if rows == 1:
        return variant
    if variant.startswith("pack8_"):
        return f"pack8_prefill_{variant[len('pack8_') :]}"
    if variant.startswith("gemv_"):
        return f"prefill_{variant[len('gemv_') :]}"
    return variant


_LAUNCH_ABI = {
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
    "resolve_gguf_linear_dispatch",
]
