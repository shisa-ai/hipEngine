"""Qwen3.5 GGUF tensor-name mapping and layout validation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from hipengine.loading.gguf import GGUFModelInfo, GGUFTensorInfo, MissingGGUFTensorError

FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"

_ROOT_SLOTS = {
    "token_embedding": "token_embd.weight",
    "output_norm": "output_norm.weight",
    # Qwen3.5 GGUF omits a separate output tensor for this local target. The
    # lm-head is tied to token_embd.weight and materialization should alias it.
    "lm_head": "token_embd.weight",
}

_COMMON_LAYER_SLOTS = {
    "attn_norm": "attn_norm.weight",
    "post_attention_norm": "post_attention_norm.weight",
    "ffn_gate": "ffn_gate.weight",
    "ffn_up": "ffn_up.weight",
    "ffn_down": "ffn_down.weight",
}

_LINEAR_LAYER_SLOTS = {
    "attn_gate": "attn_gate.weight",
    "attn_qkv": "attn_qkv.weight",
    "ssm_a": "ssm_a",
    "ssm_alpha": "ssm_alpha.weight",
    "ssm_beta": "ssm_beta.weight",
    "ssm_conv1d": "ssm_conv1d.weight",
    "ssm_dt_bias": "ssm_dt.bias",
    "ssm_norm": "ssm_norm.weight",
    "ssm_out": "ssm_out.weight",
}

_FULL_LAYER_SLOTS = {
    "attn_q": "attn_q.weight",
    "attn_k": "attn_k.weight",
    "attn_v": "attn_v.weight",
    "attn_output": "attn_output.weight",
    "attn_q_norm": "attn_q_norm.weight",
    "attn_k_norm": "attn_k_norm.weight",
}


@dataclass(frozen=True)
class Qwen35GGUFConfig:
    """Qwen3.5 dimensions decoded from GGUF metadata."""

    architecture: str
    block_count: int
    hidden_size: int
    vocab_size: int
    feed_forward_length: int
    context_length: int
    head_count: int
    head_count_kv: int
    key_length: int
    value_length: int
    full_attention_interval: int
    layer_types: tuple[str, ...]
    rms_norm_eps: float
    rope_dimension_count: int
    rope_dimension_sections: tuple[int, ...]
    rope_freq_base: float
    ssm_inner_size: int
    ssm_group_count: int
    ssm_state_size: int
    ssm_conv_kernel: int
    ssm_time_step_rank: int


@dataclass(frozen=True)
class Qwen35GGUFMappingValidation:
    config: Qwen35GGUFConfig
    present: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.missing and not self.unexpected and not self.shape_errors

    def raise_for_errors(self) -> None:
        if self.passed:
            return
        parts: list[str] = []
        if self.missing:
            preview = ", ".join(self.missing[:8])
            more = "" if len(self.missing) <= 8 else f" (+{len(self.missing) - 8} more)"
            parts.append(f"missing tensors: {preview}{more}")
        if self.unexpected:
            preview = ", ".join(self.unexpected[:8])
            more = "" if len(self.unexpected) <= 8 else f" (+{len(self.unexpected) - 8} more)"
            parts.append(f"unexpected tensors: {preview}{more}")
        if self.shape_errors:
            preview = "; ".join(self.shape_errors[:4])
            more = "" if len(self.shape_errors) <= 4 else f" (+{len(self.shape_errors) - 4} more)"
            parts.append(f"shape errors: {preview}{more}")
        raise MissingGGUFTensorError("; ".join(parts))


@dataclass(frozen=True)
class Qwen35GGUFLayerMap:
    """Canonical tensor slots for one Qwen3.5 GGUF layer."""

    layer_id: int
    layer_type: str
    tensors: Mapping[str, GGUFTensorInfo]

    def tensor(self, slot: str) -> GGUFTensorInfo:
        try:
            return self.tensors[slot]
        except KeyError as exc:
            raise MissingGGUFTensorError(
                f"layer {self.layer_id} has no GGUF tensor slot {slot!r}"
            ) from exc

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(tensor.name for tensor in self.tensors.values())


@dataclass(frozen=True)
class Qwen35GGUFModelMap:
    """Canonical root/layer tensor map for a Qwen3.5 GGUF file."""

    config: Qwen35GGUFConfig
    root_tensors: Mapping[str, GGUFTensorInfo]
    layers: tuple[Qwen35GGUFLayerMap, ...]
    validation: Qwen35GGUFMappingValidation

    def root(self, slot: str) -> GGUFTensorInfo:
        try:
            return self.root_tensors[slot]
        except KeyError as exc:
            raise MissingGGUFTensorError(f"model has no GGUF root tensor slot {slot!r}") from exc

    def layer(self, layer_id: int) -> Qwen35GGUFLayerMap:
        return self.layers[layer_id]

    @property
    def tensor_names(self) -> tuple[str, ...]:
        names: list[str] = []
        seen: set[str] = set()
        for tensor in self.root_tensors.values():
            if tensor.name not in seen:
                seen.add(tensor.name)
                names.append(tensor.name)
        for layer in self.layers:
            for name in layer.tensor_names:
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        return tuple(names)


def qwen35_gguf_config_from_metadata(info: GGUFModelInfo) -> Qwen35GGUFConfig:
    metadata = info.metadata
    architecture = str(metadata.get("general.architecture", ""))
    if architecture != "qwen35":
        raise ValueError(f"expected GGUF architecture 'qwen35', got {architecture!r}")
    block_count = _int_metadata(metadata, "qwen35.block_count")
    full_interval = int(metadata.get("qwen35.full_attention_interval", 0) or 0)
    layer_types = tuple(
        FULL_ATTENTION if full_interval and (layer_id + 1) % full_interval == 0 else LINEAR_ATTENTION
        for layer_id in range(block_count)
    )
    token_embedding = info.tensor("token_embd.weight")
    return Qwen35GGUFConfig(
        architecture=architecture,
        block_count=block_count,
        hidden_size=_int_metadata(metadata, "qwen35.embedding_length"),
        vocab_size=int(token_embedding.shape[0]),
        feed_forward_length=_int_metadata(metadata, "qwen35.feed_forward_length"),
        context_length=_int_metadata(metadata, "qwen35.context_length"),
        head_count=_int_metadata(metadata, "qwen35.attention.head_count"),
        head_count_kv=_int_metadata(metadata, "qwen35.attention.head_count_kv"),
        key_length=_int_metadata(metadata, "qwen35.attention.key_length"),
        value_length=_int_metadata(metadata, "qwen35.attention.value_length"),
        full_attention_interval=full_interval,
        layer_types=layer_types,
        rms_norm_eps=float(metadata.get("qwen35.attention.layer_norm_rms_epsilon", 1.0e-6)),
        rope_dimension_count=_int_metadata(metadata, "qwen35.rope.dimension_count"),
        rope_dimension_sections=tuple(int(item) for item in metadata.get("qwen35.rope.dimension_sections", ())),
        rope_freq_base=float(metadata.get("qwen35.rope.freq_base", 10000000.0)),
        ssm_inner_size=_int_metadata(metadata, "qwen35.ssm.inner_size"),
        ssm_group_count=_int_metadata(metadata, "qwen35.ssm.group_count"),
        ssm_state_size=_int_metadata(metadata, "qwen35.ssm.state_size"),
        ssm_conv_kernel=_int_metadata(metadata, "qwen35.ssm.conv_kernel"),
        ssm_time_step_rank=_int_metadata(metadata, "qwen35.ssm.time_step_rank"),
    )


def required_qwen35_gguf_tensor_names(config: Qwen35GGUFConfig) -> tuple[str, ...]:
    names: list[str] = ["output_norm.weight", "token_embd.weight"]
    for layer_id, layer_type in enumerate(config.layer_types):
        names.extend(_layer_required_tensor_names(layer_id, layer_type))
    return tuple(names)


def validate_qwen35_gguf_tensor_map(info: GGUFModelInfo) -> Qwen35GGUFMappingValidation:
    config = qwen35_gguf_config_from_metadata(info)
    actual = {tensor.name: tensor for tensor in info.tensors}
    required = set(required_qwen35_gguf_tensor_names(config))
    actual_names = set(actual)
    missing = tuple(sorted(required - actual_names))
    unexpected = tuple(sorted(actual_names - required))
    shape_errors = tuple(_shape_errors(config, actual))
    present = tuple(sorted(required & actual_names))
    return Qwen35GGUFMappingValidation(
        config=config,
        present=present,
        missing=missing,
        unexpected=unexpected,
        shape_errors=shape_errors,
    )


def build_qwen35_gguf_tensor_map(info: GGUFModelInfo, *, strict: bool = True) -> Qwen35GGUFModelMap:
    validation = validate_qwen35_gguf_tensor_map(info)
    if strict:
        validation.raise_for_errors()
    actual = {tensor.name: tensor for tensor in info.tensors}
    root_tensors = MappingProxyType(
        {slot: actual[name] for slot, name in _ROOT_SLOTS.items() if name in actual}
    )
    layers = tuple(
        _build_layer_map(validation.config, actual, layer_id)
        for layer_id in range(validation.config.block_count)
    )
    return Qwen35GGUFModelMap(
        config=validation.config,
        root_tensors=root_tensors,
        layers=layers,
        validation=validation,
    )


def _build_layer_map(
    config: Qwen35GGUFConfig,
    actual: Mapping[str, GGUFTensorInfo],
    layer_id: int,
) -> Qwen35GGUFLayerMap:
    layer_type = config.layer_types[layer_id]
    slot_suffixes = dict(_COMMON_LAYER_SLOTS)
    slot_suffixes.update(_FULL_LAYER_SLOTS if layer_type == FULL_ATTENTION else _LINEAR_LAYER_SLOTS)
    tensors = {
        slot: actual[f"blk.{layer_id}.{suffix}"]
        for slot, suffix in slot_suffixes.items()
        if f"blk.{layer_id}.{suffix}" in actual
    }
    return Qwen35GGUFLayerMap(
        layer_id=layer_id,
        layer_type=layer_type,
        tensors=MappingProxyType(tensors),
    )


def _layer_required_tensor_names(layer_id: int, layer_type: str) -> tuple[str, ...]:
    slot_suffixes = dict(_COMMON_LAYER_SLOTS)
    if layer_type == FULL_ATTENTION:
        slot_suffixes.update(_FULL_LAYER_SLOTS)
    elif layer_type == LINEAR_ATTENTION:
        slot_suffixes.update(_LINEAR_LAYER_SLOTS)
    else:
        raise ValueError(f"unknown Qwen3.5 GGUF layer type {layer_type!r}")
    return tuple(f"blk.{layer_id}.{suffix}" for suffix in slot_suffixes.values())


def _shape_errors(config: Qwen35GGUFConfig, actual: Mapping[str, GGUFTensorInfo]) -> list[str]:
    expected = {
        "output_norm.weight": (config.hidden_size,),
        "token_embd.weight": (config.vocab_size, config.hidden_size),
    }
    for layer_id, layer_type in enumerate(config.layer_types):
        prefix = f"blk.{layer_id}"
        expected.update(
            {
                f"{prefix}.attn_norm.weight": (config.hidden_size,),
                f"{prefix}.post_attention_norm.weight": (config.hidden_size,),
                f"{prefix}.ffn_gate.weight": (config.feed_forward_length, config.hidden_size),
                f"{prefix}.ffn_up.weight": (config.feed_forward_length, config.hidden_size),
                f"{prefix}.ffn_down.weight": (config.hidden_size, config.feed_forward_length),
            }
        )
        if layer_type == LINEAR_ATTENTION:
            expected.update(
                {
                    f"{prefix}.attn_gate.weight": (config.ssm_inner_size, config.hidden_size),
                    f"{prefix}.attn_qkv.weight": (3 * config.ssm_inner_size, config.hidden_size),
                    f"{prefix}.ssm_a": (config.ssm_time_step_rank,),
                    f"{prefix}.ssm_alpha.weight": (config.ssm_time_step_rank, config.hidden_size),
                    f"{prefix}.ssm_beta.weight": (config.ssm_time_step_rank, config.hidden_size),
                    f"{prefix}.ssm_conv1d.weight": (3 * config.ssm_inner_size, config.ssm_conv_kernel),
                    f"{prefix}.ssm_dt.bias": (config.ssm_time_step_rank,),
                    f"{prefix}.ssm_norm.weight": (config.ssm_state_size,),
                    f"{prefix}.ssm_out.weight": (config.hidden_size, config.ssm_inner_size),
                }
            )
        else:
            expected.update(
                {
                    f"{prefix}.attn_q.weight": (2 * config.head_count * config.key_length, config.hidden_size),
                    f"{prefix}.attn_k.weight": (config.head_count_kv * config.key_length, config.hidden_size),
                    f"{prefix}.attn_v.weight": (config.head_count_kv * config.value_length, config.hidden_size),
                    f"{prefix}.attn_output.weight": (config.hidden_size, config.ssm_inner_size),
                    f"{prefix}.attn_q_norm.weight": (config.key_length,),
                    f"{prefix}.attn_k_norm.weight": (config.key_length,),
                }
            )
    errors: list[str] = []
    for name, shape in expected.items():
        tensor = actual.get(name)
        if tensor is not None and tensor.shape != shape:
            errors.append(f"{name}: expected shape {shape}, got {tensor.shape}")
    return errors


def _int_metadata(metadata: Mapping[str, Any], key: str) -> int:
    try:
        return int(metadata[key])
    except KeyError as exc:
        raise KeyError(f"missing required Qwen3.5 GGUF metadata key {key!r}") from exc


__all__ = [
    "FULL_ATTENTION",
    "LINEAR_ATTENTION",
    "Qwen35GGUFConfig",
    "Qwen35GGUFLayerMap",
    "Qwen35GGUFMappingValidation",
    "Qwen35GGUFModelMap",
    "build_qwen35_gguf_tensor_map",
    "qwen35_gguf_config_from_metadata",
    "required_qwen35_gguf_tensor_names",
    "validate_qwen35_gguf_tensor_map",
]
