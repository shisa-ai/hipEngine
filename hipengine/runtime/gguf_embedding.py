"""Registry-driven GGUF token embedding dispatch helpers."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Mapping

from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_embedding import (
    register_gguf_q6_k_embedding_kernels,
)
from hipengine.kernels.hip_gfx1100.runtime.state import register_runtime_state_kernels
from hipengine.kernels.registry import KernelKey, resolve
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_RAW_GGUF,
    Qwen35GGUFDeviceWeight,
)

GGUF_EMBEDDING_OUTPUT_BF16 = "bf16"


@dataclass(frozen=True)
class GGUFEmbeddingDispatch:
    """Resolved kernel key and ABI family for one GGUF embedding launch."""

    key: KernelKey
    abi: str


_RAW_EMBEDDING_QUANTS = frozenset({"gguf_q6_k", "gguf_q8_0"})


def resolve_gguf_embedding_dispatch(
    weight: Qwen35GGUFDeviceWeight,
    *,
    output_dtype: str = GGUF_EMBEDDING_OUTPUT_BF16,
    backend: str = "hip_gfx1100",
) -> GGUFEmbeddingDispatch:
    if output_dtype != GGUF_EMBEDDING_OUTPUT_BF16:
        raise ValueError(f"unsupported GGUF embedding output dtype {output_dtype!r}")
    if weight.spec.layout == LAYOUT_RAW_GGUF and weight.spec.quant_key in _RAW_EMBEDDING_QUANTS:
        return GGUFEmbeddingDispatch(
            KernelKey(backend, "embedding", weight.spec.quant_key, "lookup_bf16_out"),
            "raw",
        )
    if weight.spec.layout == LAYOUT_DENSE_BF16:
        return GGUFEmbeddingDispatch(
            KernelKey(backend, "embedding", "bf16", "lookup_bf16_out"),
            "dense_bf16",
        )
    raise ValueError(
        "unsupported GGUF embedding dispatch: "
        f"layout={weight.spec.layout!r}, quant={weight.spec.quant_key!r}, output={output_dtype!r}"
    )


def launch_gguf_embedding(
    weight: Qwen35GGUFDeviceWeight,
    token_ids_ptr: int,
    out_ptr: int,
    rows: int,
    hidden_size: int,
    vocab_size: int,
    *,
    output_dtype: str = GGUF_EMBEDDING_OUTPUT_BF16,
    backend: str = "hip_gfx1100",
    threads: int = 0,
    stream: int = 0,
    libraries: Mapping[str, ctypes.CDLL] | None = None,
    runtime=None,
) -> None:
    dispatch = resolve_gguf_embedding_dispatch(
        weight,
        output_dtype=output_dtype,
        backend=backend,
    )
    _ensure_embedding_kernel_registered(dispatch.key)
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
    _LAUNCH_ABI[dispatch.abi](fn, weight, token_ids_ptr, out_ptr, rows, hidden_size, vocab_size, kwargs)


def _launch_raw(fn, weight, token_ids_ptr, out_ptr, rows, hidden_size, vocab_size, kwargs) -> None:
    fn(
        token_ids_ptr,
        weight.allocation("raw").tensor.ptr,
        out_ptr,
        rows,
        hidden_size,
        vocab_size,
        **kwargs,
    )


def _launch_dense_bf16(fn, weight, token_ids_ptr, out_ptr, rows, hidden_size, vocab_size, kwargs) -> None:
    fn(
        weight.allocation("raw").tensor.ptr,
        token_ids_ptr,
        out_ptr,
        hidden_size,
        vocab_size,
        **kwargs,
    )


def _ensure_embedding_kernel_registered(key: KernelKey) -> None:
    # Some registry-focused tests intentionally clear global registrations after
    # module import. Re-register at dispatch time so the runtime path does not
    # depend on import order, but do not overwrite tests that deliberately
    # replace one dispatch key with a fixture kernel.
    if resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
        missing="none",
    ) is not None:
        return
    register_gguf_q6_k_embedding_kernels()
    register_runtime_state_kernels()


_LAUNCH_ABI = {
    "dense_bf16": _launch_dense_bf16,
    "raw": _launch_raw,
}


__all__ = [
    "GGUF_EMBEDDING_OUTPUT_BF16",
    "GGUFEmbeddingDispatch",
    "launch_gguf_embedding",
    "resolve_gguf_embedding_dispatch",
]
