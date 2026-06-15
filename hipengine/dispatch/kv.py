"""KV-cache kernel selection helpers.

The KV cache has its own storage axis (for example BF16 versus
``int8_per_token_head``) that is independent of the model weight quantization
preset.  Runtime code should derive paged-KV write / attention kernel keys from
``KVLiveSpans`` metadata and resolve those keys through the registry instead of
adding backend- or quant-specific branches in the dispatch path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from hipengine.core.dtype import DType
from hipengine.dispatch.fusion import BoundKernel, KernelPlanStep
from hipengine.kernels.registry import KernelKey, MissingPolicy, resolve
from hipengine.kvcache import KVLiveSpans


class PagedKVWriteKind(str, Enum):
    """Paged-KV append/write launch shapes."""

    DECODE = "decode"
    PROMPT = "prompt"
    BATCH = "batch"


class PagedAttnDecodeKind(str, Enum):
    """Paged-attention decode launch variants selected by KV storage metadata."""

    GQA_SPLITK = "gqa_splitk"
    GQA_SPLITK_GATE_BF16 = "gqa_splitk_gate_bf16"
    GQA_SPLITK_GATE_FP16 = "gqa_splitk_gate_fp16"


class PagedAttnPrefillKind(str, Enum):
    """Paged-attention prefill launch variants selected by KV storage metadata."""

    GQA_GATE_FP16 = "gqa_gate_fp16"
    GQA_GATE_BF16 = "gqa_gate_bf16"


@dataclass(frozen=True)
class KVKernelSelection:
    """Layer/quant/variant tuple selected from KV policy/span metadata."""

    layer: str
    quant: str
    variant: str

    def key(self, backend: str) -> KernelKey:
        return KernelKey(backend, self.layer, self.quant, self.variant)

    def step(self, backend: str) -> KernelPlanStep:
        return KernelPlanStep(backend=backend, layer=self.layer, quant=self.quant, variant=self.variant)


@dataclass(frozen=True)
class _RouteTemplate:
    layer: str
    variant: str
    quant: str | None = None

    def instantiate(self, *, model_quant: str) -> KVKernelSelection:
        quant = self.quant if self.quant is not None else model_quant
        return KVKernelSelection(layer=self.layer, quant=quant, variant=self.variant)


def _enum_value(enum_type: type[Enum], value: Enum | str, name: str) -> Enum:
    try:
        return value if isinstance(value, enum_type) else enum_type(str(value))
    except ValueError as exc:
        valid = ", ".join(item.value for item in enum_type)
        raise ValueError(f"unknown {name} {value!r}; expected one of: {valid}") from exc


_PAGED_KV_WRITE_ROUTES: dict[tuple[DType, PagedKVWriteKind, DType], _RouteTemplate] = {
    (DType.BF16, PagedKVWriteKind.DECODE, DType.BF16): _RouteTemplate(
        "paged_kv_write", "mixed_bf16_spans"
    ),
    (DType.BF16, PagedKVWriteKind.BATCH, DType.BF16): _RouteTemplate(
        "paged_kv_write", "mixed_bf16_batch_spans"
    ),
    (DType.BF16, PagedKVWriteKind.DECODE, DType.FP16): _RouteTemplate(
        "paged_kv_write", "mixed_fp16_spans"
    ),
    (DType.BF16, PagedKVWriteKind.BATCH, DType.FP16): _RouteTemplate(
        "paged_kv_write", "mixed_fp16_batch_spans"
    ),
    (DType.BF16, PagedKVWriteKind.PROMPT, DType.FP16): _RouteTemplate(
        "paged_kv_write", "mixed_fp16_prompt_spans"
    ),
    (DType.BF16, PagedKVWriteKind.DECODE, DType.FP32): _RouteTemplate(
        "paged_kv_write", "f32_spans"
    ),
    (DType.INT8_PER_TOKEN_HEAD, PagedKVWriteKind.DECODE, DType.FP32): _RouteTemplate(
        "paged_kv_write", "per_token_head_spans", quant=DType.INT8_PER_TOKEN_HEAD.value
    ),
    (DType.INT8_PER_TOKEN_HEAD, PagedKVWriteKind.PROMPT, DType.FP32): _RouteTemplate(
        "paged_kv_write", "per_token_head_prompt_spans", quant=DType.INT8_PER_TOKEN_HEAD.value
    ),
    (DType.INT8_PER_TOKEN_HEAD, PagedKVWriteKind.BATCH, DType.FP32): _RouteTemplate(
        "paged_kv_write", "per_token_head_batch_spans", quant=DType.INT8_PER_TOKEN_HEAD.value
    ),
}


_PAGED_ATTN_PREFILL_ROUTES: dict[tuple[DType, PagedAttnPrefillKind], _RouteTemplate] = {
    (DType.BF16, PagedAttnPrefillKind.GQA_GATE_FP16): _RouteTemplate(
        "paged_attn_prefill", "bf16_gqa_gate_fp16_spans"
    ),
    (DType.BF16, PagedAttnPrefillKind.GQA_GATE_BF16): _RouteTemplate(
        "paged_attn_prefill", "bf16_gqa_gate_bf16_spans"
    ),
    (DType.INT8_PER_TOKEN_HEAD, PagedAttnPrefillKind.GQA_GATE_FP16): _RouteTemplate(
        "paged_attn_prefill",
        "per_token_head_gqa_gate_fp16_spans",
        quant=DType.INT8_PER_TOKEN_HEAD.value,
    ),
}


_PAGED_ATTN_DECODE_ROUTES: dict[tuple[DType, PagedAttnDecodeKind], _RouteTemplate] = {
    (DType.BF16, PagedAttnDecodeKind.GQA_SPLITK): _RouteTemplate(
        "paged_attn_decode", "bf16_split_k_gqa_spans"
    ),
    (DType.BF16, PagedAttnDecodeKind.GQA_SPLITK_GATE_BF16): _RouteTemplate(
        "paged_attn_decode", "bf16_split_k_gqa_gate_bf16_spans"
    ),
    (DType.BF16, PagedAttnDecodeKind.GQA_SPLITK_GATE_FP16): _RouteTemplate(
        "paged_attn_decode", "bf16_split_k_gqa_gate_fp16_spans"
    ),
    (DType.INT8_PER_TOKEN_HEAD, PagedAttnDecodeKind.GQA_SPLITK): _RouteTemplate(
        "paged_attn_decode",
        "per_token_head_gqa_splitk_spans",
        quant=DType.INT8_PER_TOKEN_HEAD.value,
    ),
    (DType.INT8_PER_TOKEN_HEAD, PagedAttnDecodeKind.GQA_SPLITK_GATE_BF16): _RouteTemplate(
        "paged_attn_decode",
        "per_token_head_gqa_splitk_gate_bf16_spans",
        quant=DType.INT8_PER_TOKEN_HEAD.value,
    ),
    (DType.INT8_PER_TOKEN_HEAD, PagedAttnDecodeKind.GQA_SPLITK_GATE_FP16): _RouteTemplate(
        "paged_attn_decode",
        "per_token_head_gqa_splitk_gate_fp16_spans",
        quant=DType.INT8_PER_TOKEN_HEAD.value,
    ),
}


def plan_paged_kv_write(
    spans: KVLiveSpans,
    *,
    kind: PagedKVWriteKind | str = PagedKVWriteKind.DECODE,
    source_dtype: DType | str = DType.FP32,
    model_quant: str = "w4_paro",
) -> KVKernelSelection:
    """Select a paged-KV write key from span storage metadata.

    ``model_quant`` is used only for BF16-storage routes whose kernel keys stay
    under the model's weight-quant axis (for example ``w4_paro``).  INT8 KV
    storage routes override the key quant to ``int8_per_token_head``.
    """

    write_kind = _enum_value(PagedKVWriteKind, kind, "paged KV write kind")
    source = DType.parse(source_dtype)
    route = _PAGED_KV_WRITE_ROUTES.get((spans.storage_dtype, write_kind, source))
    if route is None:
        raise ValueError(
            "no paged KV write route for "
            f"storage_dtype={spans.storage_dtype.value!r}, kind={write_kind.value!r}, "
            f"source_dtype={source.value!r}"
        )
    return route.instantiate(model_quant=model_quant)


def resolve_paged_kv_write(
    *,
    backend: str,
    spans: KVLiveSpans,
    kind: PagedKVWriteKind | str = PagedKVWriteKind.DECODE,
    source_dtype: DType | str = DType.FP32,
    model_quant: str = "w4_paro",
    missing: MissingPolicy = "error",
) -> Callable[..., Any] | None:
    """Resolve a paged-KV write kernel selected by ``plan_paged_kv_write``."""

    selection = plan_paged_kv_write(
        spans,
        kind=kind,
        source_dtype=source_dtype,
        model_quant=model_quant,
    )
    return resolve(
        backend=backend,
        layer=selection.layer,
        quant=selection.quant,
        variant=selection.variant,
        missing=missing,
    )


def bind_paged_kv_write(
    *,
    backend: str,
    spans: KVLiveSpans,
    kind: PagedKVWriteKind | str = PagedKVWriteKind.DECODE,
    source_dtype: DType | str = DType.FP32,
    model_quant: str = "w4_paro",
) -> BoundKernel:
    selection = plan_paged_kv_write(
        spans,
        kind=kind,
        source_dtype=source_dtype,
        model_quant=model_quant,
    )
    kernel = resolve(
        backend=backend,
        layer=selection.layer,
        quant=selection.quant,
        variant=selection.variant,
    )
    return BoundKernel(step=selection.step(backend), kernel=kernel)


def plan_paged_attn_prefill(
    spans: KVLiveSpans,
    *,
    kind: PagedAttnPrefillKind | str = PagedAttnPrefillKind.GQA_GATE_FP16,
    model_quant: str = "w4_paro",
) -> KVKernelSelection:
    """Select a paged-attention prefill key from span storage metadata."""

    prefill_kind = _enum_value(PagedAttnPrefillKind, kind, "paged attention prefill kind")
    route = _PAGED_ATTN_PREFILL_ROUTES.get((spans.storage_dtype, prefill_kind))
    if route is None:
        raise ValueError(
            "no paged attention prefill route for "
            f"storage_dtype={spans.storage_dtype.value!r}, kind={prefill_kind.value!r}"
        )
    return route.instantiate(model_quant=model_quant)


def resolve_paged_attn_prefill(
    *,
    backend: str,
    spans: KVLiveSpans,
    kind: PagedAttnPrefillKind | str = PagedAttnPrefillKind.GQA_GATE_FP16,
    model_quant: str = "w4_paro",
    missing: MissingPolicy = "error",
) -> Callable[..., Any] | None:
    """Resolve a paged-attention prefill kernel selected by ``plan_paged_attn_prefill``."""

    selection = plan_paged_attn_prefill(spans, kind=kind, model_quant=model_quant)
    return resolve(
        backend=backend,
        layer=selection.layer,
        quant=selection.quant,
        variant=selection.variant,
        missing=missing,
    )


def bind_paged_attn_prefill(
    *,
    backend: str,
    spans: KVLiveSpans,
    kind: PagedAttnPrefillKind | str = PagedAttnPrefillKind.GQA_GATE_FP16,
    model_quant: str = "w4_paro",
) -> BoundKernel:
    selection = plan_paged_attn_prefill(spans, kind=kind, model_quant=model_quant)
    kernel = resolve(
        backend=backend,
        layer=selection.layer,
        quant=selection.quant,
        variant=selection.variant,
    )
    return BoundKernel(step=selection.step(backend), kernel=kernel)


def plan_paged_attn_decode(
    spans: KVLiveSpans,
    *,
    kind: PagedAttnDecodeKind | str = PagedAttnDecodeKind.GQA_SPLITK,
    model_quant: str = "w4_paro",
) -> KVKernelSelection:
    """Select a paged-attention decode key from span storage metadata."""

    decode_kind = _enum_value(PagedAttnDecodeKind, kind, "paged attention decode kind")
    route = _PAGED_ATTN_DECODE_ROUTES.get((spans.storage_dtype, decode_kind))
    if route is None:
        raise ValueError(
            "no paged attention decode route for "
            f"storage_dtype={spans.storage_dtype.value!r}, kind={decode_kind.value!r}"
        )
    return route.instantiate(model_quant=model_quant)


def resolve_paged_attn_decode(
    *,
    backend: str,
    spans: KVLiveSpans,
    kind: PagedAttnDecodeKind | str = PagedAttnDecodeKind.GQA_SPLITK,
    model_quant: str = "w4_paro",
    missing: MissingPolicy = "error",
) -> Callable[..., Any] | None:
    """Resolve a paged-attention decode kernel selected by ``plan_paged_attn_decode``."""

    selection = plan_paged_attn_decode(spans, kind=kind, model_quant=model_quant)
    return resolve(
        backend=backend,
        layer=selection.layer,
        quant=selection.quant,
        variant=selection.variant,
        missing=missing,
    )


def bind_paged_attn_decode(
    *,
    backend: str,
    spans: KVLiveSpans,
    kind: PagedAttnDecodeKind | str = PagedAttnDecodeKind.GQA_SPLITK,
    model_quant: str = "w4_paro",
) -> BoundKernel:
    selection = plan_paged_attn_decode(spans, kind=kind, model_quant=model_quant)
    kernel = resolve(
        backend=backend,
        layer=selection.layer,
        quant=selection.quant,
        variant=selection.variant,
    )
    return BoundKernel(step=selection.step(backend), kernel=kernel)
