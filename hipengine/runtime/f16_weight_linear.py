"""Registry-driven mixed-activation source-F16 projection dispatch."""

from __future__ import annotations

import ctypes
import os
from typing import Mapping

from hipengine.kernels.backends import backend_package_capability, load_backend_kernel_package
from hipengine.kernels.hip_gfx1100.linear.laguna_f16_projection import (
    register_laguna_f16_projection_kernels,
)
from hipengine.kernels.registry import KernelKey, resolve
from hipengine.runtime.gguf_weight import GGUFDeviceWeight

LAYOUT_DENSE_F16 = "dense_f16"
SOURCE_QUANT_FP16 = "fp16"
F16_WEIGHT = "fp16_weight"
_ENV_PREFILL_MODE = "HIPENGINE_LAGUNA_F16_PREFILL"
_PREFILL_MODES = frozenset({"auto", "gemv", "tiled", "wmma_comp_swa"})


def _variant(activation_dtype: str, output_dtype: str) -> str:
    if activation_dtype not in {"bf16", "f32"}:
        raise ValueError(f"unsupported activation dtype {activation_dtype!r}")
    if output_dtype not in {"bf16", "f32"}:
        raise ValueError(f"unsupported output dtype {output_dtype!r}")
    return f"{activation_dtype}_{output_dtype}_out"


def _prefill_strategy(
    *,
    rows: int,
    activation_dtype: str,
    backend: str,
    compensated_wmma_eligible: bool = False,
) -> str | None:
    if rows <= 1 or activation_dtype != "bf16":
        return None
    mode = os.environ.get(_ENV_PREFILL_MODE, "auto").strip().lower() or "auto"
    if mode not in _PREFILL_MODES:
        expected = ", ".join(sorted(_PREFILL_MODES))
        raise ValueError(f"{_ENV_PREFILL_MODE} must be one of: {expected}")
    if mode == "gemv":
        return None
    if mode == "tiled":
        return mode
    if mode == "wmma_comp_swa":
        return "wmma_comp" if compensated_wmma_eligible and rows >= 16 else "tiled"
    strategy = backend_package_capability(
        backend, "LAGUNA_F16_PREFILL_STRATEGY", None
    )
    minimum = int(
        backend_package_capability(backend, "LAGUNA_F16_PREFILL_MIN_ROWS", 0) or 0
    )
    return str(strategy) if strategy == "tiled" and rows >= minimum > 0 else None


def _backend(weights: tuple[GGUFDeviceWeight, ...], backend: str | None) -> str:
    resident = {weight.backend for weight in weights}
    if len(resident) != 1:
        raise ValueError("F16 projection weights must share one backend")
    value = next(iter(resident))
    if backend is not None and backend != value:
        raise ValueError(f"dispatch backend {backend!r} does not match resident backend {value!r}")
    for weight in weights:
        if weight.spec.layout != LAYOUT_DENSE_F16 or weight.spec.quant_key != SOURCE_QUANT_FP16:
            raise ValueError(
                f"F16 projection requires layout={LAYOUT_DENSE_F16!r}, "
                f"source quant={SOURCE_QUANT_FP16!r}"
            )
    return value


def _resolve(key: KernelKey):
    fn = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
        missing="none",
    )
    if fn is None:
        register_laguna_f16_projection_kernels()
        load_backend_kernel_package(key.backend)
        fn = resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
    return fn


def _kwargs(key: KernelKey, libraries: Mapping[str, ctypes.CDLL] | None, **kwargs):
    library = None
    if libraries is not None:
        library = libraries.get(f"{key.quant}:{key.variant}")
        if library is None:
            library = libraries.get(key.quant)
    if library is not None:
        kwargs["library"] = library
    return kwargs


def launch_f16_weight_linear(
    weight: GGUFDeviceWeight,
    x_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    activation_dtype: str = "bf16",
    output_dtype: str = "f32",
    backend: str | None = None,
    threads: int = 256,
    stream: int = 0,
    libraries: Mapping[str, ctypes.CDLL] | None = None,
    runtime=None,
    compensated_wmma_eligible: bool = False,
) -> None:
    resolved_backend = _backend((weight,), backend)
    variant = _variant(activation_dtype, output_dtype)
    strategy = _prefill_strategy(
        rows=rows,
        activation_dtype=activation_dtype,
        backend=resolved_backend,
        compensated_wmma_eligible=compensated_wmma_eligible,
    )
    if strategy is not None:
        variant = f"{strategy}_{variant}"
    key = KernelKey(resolved_backend, "linear", F16_WEIGHT, variant)
    fn = _resolve(key)
    fn(
        x_ptr,
        weight.allocation("raw").tensor.ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **_kwargs(
            key,
            libraries,
            threads=threads,
            stream=stream,
            runtime=runtime,
        ),
    )


def launch_f16_weight_linear_pair(
    weight_a: GGUFDeviceWeight,
    weight_b: GGUFDeviceWeight,
    x_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_a_features: int,
    out_b_features: int,
    *,
    backend: str | None = None,
    threads: int = 256,
    stream: int = 0,
    libraries: Mapping[str, ctypes.CDLL] | None = None,
    runtime=None,
) -> None:
    resolved_backend = _backend((weight_a, weight_b), backend)
    key = KernelKey(resolved_backend, "linear_pair", F16_WEIGHT, "bf16_f32_out")
    fn = _resolve(key)
    fn(
        x_ptr,
        weight_a.allocation("raw").tensor.ptr,
        weight_b.allocation("raw").tensor.ptr,
        out_a_ptr,
        out_b_ptr,
        rows,
        in_features,
        out_a_features,
        out_b_features,
        **_kwargs(key, libraries, threads=threads, stream=stream, runtime=runtime),
    )


def launch_f16_weight_linear_triple(
    weight_a: GGUFDeviceWeight,
    weight_b: GGUFDeviceWeight,
    weight_c: GGUFDeviceWeight,
    x_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    out_c_ptr: int,
    rows: int,
    in_features: int,
    out_a_features: int,
    out_b_features: int,
    out_c_features: int,
    *,
    backend: str | None = None,
    threads: int = 256,
    stream: int = 0,
    libraries: Mapping[str, ctypes.CDLL] | None = None,
    runtime=None,
    compensated_wmma_eligible: bool = False,
) -> None:
    resolved_backend = _backend((weight_a, weight_b, weight_c), backend)
    variant = "bf16_f32_out"
    strategy = _prefill_strategy(
        rows=rows,
        activation_dtype="bf16",
        backend=resolved_backend,
        compensated_wmma_eligible=compensated_wmma_eligible,
    )
    if strategy is not None:
        variant = f"{strategy}_{variant}"
    key = KernelKey(resolved_backend, "linear_triple", F16_WEIGHT, variant)
    fn = _resolve(key)
    fn(
        x_ptr,
        weight_a.allocation("raw").tensor.ptr,
        weight_b.allocation("raw").tensor.ptr,
        weight_c.allocation("raw").tensor.ptr,
        out_a_ptr,
        out_b_ptr,
        out_c_ptr,
        rows,
        in_features,
        out_a_features,
        out_b_features,
        out_c_features,
        **_kwargs(key, libraries, threads=threads, stream=stream, runtime=runtime),
    )


__all__ = [
    "F16_WEIGHT",
    "LAYOUT_DENSE_F16",
    "SOURCE_QUANT_FP16",
    "launch_f16_weight_linear",
    "launch_f16_weight_linear_pair",
    "launch_f16_weight_linear_triple",
]
