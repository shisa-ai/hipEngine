"""Separate GGUF tensor map for trailing Qwen3.5/Qwen3.6 NextN blocks.

The autoregressive Qwen35 GGUF map deliberately stops before trailing blocks
that carry ``nextn.*`` tensors.  This module maps those blocks as a separate
draft model so they can never be mistaken for an additional AR layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from hipengine.loading.gguf import GGUFModelInfo, GGUFTensorInfo, MissingGGUFTensorError
from hipengine.loading.qwen35_gguf import Qwen35GGUFConfig, qwen35_gguf_config_from_metadata
from hipengine.quant.gguf import GGMLQuantizationType

_NEXTN_COMMON_LAYER_SLOTS: Mapping[str, str] = MappingProxyType(
    {
        "attn_norm": "attn_norm.weight",
        "post_attention_norm": "post_attention_norm.weight",
        "attn_q": "attn_q.weight",
        "attn_k": "attn_k.weight",
        "attn_v": "attn_v.weight",
        "attn_output": "attn_output.weight",
        "attn_q_norm": "attn_q_norm.weight",
        "attn_k_norm": "attn_k_norm.weight",
    }
)

_NEXTN_MOE_LAYER_SLOTS: Mapping[str, str] = MappingProxyType(
    {
        "ffn_gate_inp": "ffn_gate_inp.weight",
        "ffn_gate_inp_shexp": "ffn_gate_inp_shexp.weight",
        "ffn_gate_exps": "ffn_gate_exps.weight",
        "ffn_up_exps": "ffn_up_exps.weight",
        "ffn_down_exps": "ffn_down_exps.weight",
        "ffn_gate_shexp": "ffn_gate_shexp.weight",
        "ffn_up_shexp": "ffn_up_shexp.weight",
        "ffn_down_shexp": "ffn_down_shexp.weight",
    }
)

_NEXTN_DENSE_LAYER_SLOTS: Mapping[str, str] = MappingProxyType(
    {
        "ffn_gate": "ffn_gate.weight",
        "ffn_up": "ffn_up.weight",
        "ffn_down": "ffn_down.weight",
    }
)

_NEXTN_SLOTS: Mapping[str, str] = MappingProxyType(
    {
        "eh_proj": "nextn.eh_proj.weight",
        "enorm": "nextn.enorm.weight",
        "hnorm": "nextn.hnorm.weight",
        "shared_head_norm": "nextn.shared_head_norm.weight",
    }
)

_OPTIONAL_NEXTN_SLOTS: Mapping[str, str] = MappingProxyType(
    {
        "embed_tokens": "nextn.embed_tokens.weight",
        "shared_head_head": "nextn.shared_head_head.weight",
    }
)

_EXPECTED_COMMON_QTYPES: Mapping[str, GGMLQuantizationType] = MappingProxyType(
    {
        "attn_norm": GGMLQuantizationType.F32,
        "post_attention_norm": GGMLQuantizationType.F32,
        "attn_q_norm": GGMLQuantizationType.F32,
        "attn_k_norm": GGMLQuantizationType.F32,
        "eh_proj": GGMLQuantizationType.Q8_0,
        "enorm": GGMLQuantizationType.F32,
        "hnorm": GGMLQuantizationType.F32,
        "shared_head_norm": GGMLQuantizationType.F32,
    }
)

_EXPECTED_MOE_QTYPES: Mapping[str, GGMLQuantizationType] = MappingProxyType(
    {
        "attn_q": GGMLQuantizationType.Q8_0,
        "attn_k": GGMLQuantizationType.Q8_0,
        "attn_v": GGMLQuantizationType.Q8_0,
        "attn_output": GGMLQuantizationType.Q8_0,
        "ffn_gate_inp": GGMLQuantizationType.BF16,
        "ffn_gate_inp_shexp": GGMLQuantizationType.BF16,
        "ffn_gate_exps": GGMLQuantizationType.Q3_K,
        "ffn_up_exps": GGMLQuantizationType.Q3_K,
        "ffn_down_exps": GGMLQuantizationType.Q4_K,
        "ffn_gate_shexp": GGMLQuantizationType.Q8_0,
        "ffn_up_shexp": GGMLQuantizationType.Q8_0,
        "ffn_down_shexp": GGMLQuantizationType.Q8_0,
    }
)

_EXPECTED_DENSE_QTYPES: Mapping[str, GGMLQuantizationType] = MappingProxyType(
    {
        "attn_q": GGMLQuantizationType.Q4_K,
        "attn_k": GGMLQuantizationType.Q4_K,
        "attn_v": GGMLQuantizationType.Q6_K,
        "attn_output": GGMLQuantizationType.Q4_K,
        "ffn_gate": GGMLQuantizationType.Q4_K,
        "ffn_up": GGMLQuantizationType.Q4_K,
        "ffn_down": GGMLQuantizationType.Q6_K,
    }
)


@dataclass(frozen=True)
class Qwen35GGUFNextNValidation:
    """Validation result for one trailing GGUF draft block."""

    config: Qwen35GGUFConfig
    block_id: int
    present: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    dtype_errors: tuple[str, ...]
    shape_errors: tuple[str, ...]
    embedding_fallback: str
    head_fallback: str
    head_norm_source: str

    @property
    def passed(self) -> bool:
        return not (self.missing or self.unexpected or self.dtype_errors or self.shape_errors)

    def raise_for_errors(self) -> None:
        if self.passed:
            return
        parts: list[str] = []
        for label, values in (
            ("missing tensors", self.missing),
            ("unexpected tensors", self.unexpected),
            ("dtype errors", self.dtype_errors),
            ("shape errors", self.shape_errors),
        ):
            if values:
                preview = "; ".join(values[:6])
                more = "" if len(values) <= 6 else f" (+{len(values) - 6} more)"
                parts.append(f"{label}: {preview}{more}")
        raise MissingGGUFTensorError("; ".join(parts))


@dataclass(frozen=True)
class Qwen35GGUFNextNMap:
    """Canonical tensors for one target-attached GGUF NextN draft block."""

    config: Qwen35GGUFConfig
    block_id: int
    layer_tensors: Mapping[str, GGUFTensorInfo]
    nextn_tensors: Mapping[str, GGUFTensorInfo]
    fallback_tensors: Mapping[str, GGUFTensorInfo]
    validation: Qwen35GGUFNextNValidation

    def tensor(self, slot: str) -> GGUFTensorInfo:
        tensor = self.layer_tensors.get(slot)
        if tensor is None:
            tensor = self.nextn_tensors.get(slot)
        if tensor is None:
            raise MissingGGUFTensorError(f"NextN block {self.block_id} has no tensor slot {slot!r}")
        return tensor

    def fallback(self, slot: str) -> GGUFTensorInfo:
        try:
            return self.fallback_tensors[slot]
        except KeyError as exc:
            raise MissingGGUFTensorError(f"NextN block {self.block_id} has no fallback slot {slot!r}") from exc

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(tensor.name for tensor in (*self.layer_tensors.values(), *self.nextn_tensors.values()))


def required_qwen35_gguf_nextn_tensor_names(
    block_id: int,
    *,
    config: Qwen35GGUFConfig | None = None,
) -> tuple[str, ...]:
    """Return architecture-shaped required tensor names for one NextN block.

    ``config=None`` preserves the historical twenty-tensor MoE contract for
    callers that only have a block id. Dense callers pass the decoded config
    and receive the real fifteen-tensor gate/up/down contract.
    """

    prefix = f"blk.{int(block_id)}."
    layer_slots = _nextn_layer_slots(config)
    return tuple(prefix + suffix for suffix in (*layer_slots.values(), *_NEXTN_SLOTS.values()))


def validate_qwen35_gguf_nextn_tensor_map(info: GGUFModelInfo) -> Qwen35GGUFNextNValidation:
    config = qwen35_gguf_config_from_metadata(info)
    if len(config.ignored_block_ids) != 1:
        block_id = config.block_count
        return Qwen35GGUFNextNValidation(
            config=config,
            block_id=block_id,
            present=(),
            missing=("expected exactly one trailing NextN block",),
            unexpected=(),
            dtype_errors=(),
            shape_errors=(),
            embedding_fallback="token_embd.weight",
            head_fallback="output.weight" if config.is_moe else "token_embd.weight",
            head_norm_source="output_norm.weight",
        )
    block_id = int(config.ignored_block_ids[0])
    actual = {tensor.name: tensor for tensor in info.tensors}
    required = set(required_qwen35_gguf_nextn_tensor_names(block_id, config=config))
    prefix = f"blk.{block_id}."
    optional = {prefix + suffix for suffix in _OPTIONAL_NEXTN_SLOTS.values()}
    block_names = {name for name in actual if name.startswith(prefix)}
    present = tuple(sorted(required & block_names))
    missing = tuple(sorted(required - block_names))
    unexpected = tuple(sorted(block_names - required - optional))

    slot_names = _slot_names(block_id, config=config)
    dtype_errors: list[str] = []
    for slot, expected in _expected_qtypes(config).items():
        tensor = actual.get(slot_names[slot])
        if tensor is not None and int(tensor.ggml_type) != int(expected):
            dtype_errors.append(
                f"{tensor.name}: expected {expected.name}, got {tensor.ggml_type_name}"
            )

    shape_errors: list[str] = []
    for slot, expected in _expected_shapes(config).items():
        tensor = actual.get(slot_names[slot])
        if tensor is not None and tuple(tensor.shape) != expected:
            shape_errors.append(f"{tensor.name}: expected shape {expected}, got {tuple(tensor.shape)}")

    embed_name = prefix + _OPTIONAL_NEXTN_SLOTS["embed_tokens"]
    head_name = prefix + _OPTIONAL_NEXTN_SLOTS["shared_head_head"]
    norm_name = prefix + _NEXTN_SLOTS["shared_head_norm"]
    embedding_fallback = embed_name if embed_name in actual else "token_embd.weight"
    target_head = config.lm_head_tensor_name
    head_fallback = head_name if head_name in actual else target_head
    head_norm_source = norm_name if norm_name in actual else "output_norm.weight"
    for fallback in (embedding_fallback, head_fallback, head_norm_source):
        if fallback not in actual and fallback not in missing:
            missing = (*missing, fallback)

    return Qwen35GGUFNextNValidation(
        config=config,
        block_id=block_id,
        present=present,
        missing=tuple(sorted(set(missing))),
        unexpected=unexpected,
        dtype_errors=tuple(dtype_errors),
        shape_errors=tuple(shape_errors),
        embedding_fallback=embedding_fallback,
        head_fallback=head_fallback,
        head_norm_source=head_norm_source,
    )


def build_qwen35_gguf_nextn_tensor_map(
    info: GGUFModelInfo,
    *,
    strict: bool = True,
) -> Qwen35GGUFNextNMap:
    validation = validate_qwen35_gguf_nextn_tensor_map(info)
    if strict:
        validation.raise_for_errors()
    actual = {tensor.name: tensor for tensor in info.tensors}
    prefix = f"blk.{validation.block_id}."
    layer = {
        slot: actual[prefix + suffix]
        for slot, suffix in _nextn_layer_slots(validation.config).items()
        if prefix + suffix in actual
    }
    nextn = {
        slot: actual[prefix + suffix]
        for slot, suffix in {**_NEXTN_SLOTS, **_OPTIONAL_NEXTN_SLOTS}.items()
        if prefix + suffix in actual
    }
    fallbacks = {
        "token_embedding": actual[validation.embedding_fallback],
        "lm_head": actual[validation.head_fallback],
        "output_norm": actual[validation.head_norm_source],
    }
    return Qwen35GGUFNextNMap(
        config=validation.config,
        block_id=validation.block_id,
        layer_tensors=MappingProxyType(layer),
        nextn_tensors=MappingProxyType(nextn),
        fallback_tensors=MappingProxyType(fallbacks),
        validation=validation,
    )


def _nextn_layer_slots(
    config: Qwen35GGUFConfig | None,
) -> dict[str, str]:
    ffn_slots = (
        _NEXTN_MOE_LAYER_SLOTS
        if config is None or config.is_moe
        else _NEXTN_DENSE_LAYER_SLOTS
    )
    return {**_NEXTN_COMMON_LAYER_SLOTS, **ffn_slots}


def _expected_qtypes(
    config: Qwen35GGUFConfig,
) -> dict[str, GGMLQuantizationType]:
    architecture_qtypes = (
        _EXPECTED_MOE_QTYPES if config.is_moe else _EXPECTED_DENSE_QTYPES
    )
    return {**_EXPECTED_COMMON_QTYPES, **architecture_qtypes}


def _slot_names(
    block_id: int,
    *,
    config: Qwen35GGUFConfig,
) -> dict[str, str]:
    prefix = f"blk.{block_id}."
    return {
        slot: prefix + suffix
        for slot, suffix in {**_nextn_layer_slots(config), **_NEXTN_SLOTS}.items()
    }


def _expected_shapes(config: Qwen35GGUFConfig) -> dict[str, tuple[int, ...]]:
    hidden = int(config.hidden_size)
    q_width = int(config.head_count) * int(config.key_length)
    kv_width = int(config.head_count_kv) * int(config.key_length)
    shapes: dict[str, tuple[int, ...]] = {
        "attn_norm": (hidden,),
        "post_attention_norm": (hidden,),
        "attn_q": (2 * q_width, hidden),
        "attn_k": (kv_width, hidden),
        "attn_v": (kv_width, hidden),
        "attn_output": (hidden, q_width),
        "attn_q_norm": (int(config.key_length),),
        "attn_k_norm": (int(config.key_length),),
        "eh_proj": (hidden, 2 * hidden),
        "enorm": (hidden,),
        "hnorm": (hidden,),
        "shared_head_norm": (hidden,),
    }
    if config.is_moe:
        experts = int(config.expert_count)
        expert_ffn = int(config.expert_feed_forward_length)
        shared_ffn = int(config.expert_shared_feed_forward_length)
        shapes.update(
            ffn_gate_inp=(experts, hidden),
            ffn_gate_inp_shexp=(hidden,),
            ffn_gate_exps=(experts, expert_ffn, hidden),
            ffn_up_exps=(experts, expert_ffn, hidden),
            ffn_down_exps=(experts, hidden, expert_ffn),
            ffn_gate_shexp=(shared_ffn, hidden),
            ffn_up_shexp=(shared_ffn, hidden),
            ffn_down_shexp=(hidden, shared_ffn),
        )
    else:
        dense_ffn = int(config.feed_forward_length)
        shapes.update(
            ffn_gate=(dense_ffn, hidden),
            ffn_up=(dense_ffn, hidden),
            ffn_down=(hidden, dense_ffn),
        )
    return shapes


__all__ = [
    "Qwen35GGUFNextNMap",
    "Qwen35GGUFNextNValidation",
    "build_qwen35_gguf_nextn_tensor_map",
    "required_qwen35_gguf_nextn_tensor_names",
    "validate_qwen35_gguf_nextn_tensor_map",
]
