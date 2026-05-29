"""StepFun Step 3.x GGUF metadata and tensor-name validation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from hipengine.loading.gguf import GGUFSplitModelInfo, GGUFSplitTensorInfo, MissingGGUFTensorError

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"
DENSE_MLP = "dense_mlp"
MOE = "moe"

_ROOT_SLOTS = {
    "token_embedding": "token_embd.weight",
    "lm_head": "output.weight",
    "output_norm": "output_norm.weight",
    "rope_freqs": "rope_freqs.weight",
}

_COMMON_LAYER_SLOTS = {
    "attn_gate": "attn_gate.weight",
    "attn_k": "attn_k.weight",
    "attn_k_norm": "attn_k_norm.weight",
    "attn_norm": "attn_norm.weight",
    "attn_output": "attn_output.weight",
    "attn_q": "attn_q.weight",
    "attn_q_norm": "attn_q_norm.weight",
    "attn_v": "attn_v.weight",
    "ffn_norm": "ffn_norm.weight",
}

_DENSE_LAYER_SLOTS = {
    "ffn_gate": "ffn_gate.weight",
    "ffn_up": "ffn_up.weight",
    "ffn_down": "ffn_down.weight",
}

_MOE_LAYER_SLOTS = {
    "ffn_gate_inp": "ffn_gate_inp.weight",
    "exp_probs_bias": "exp_probs_b.bias",
    "ffn_gate_exps": "ffn_gate_exps.weight",
    "ffn_up_exps": "ffn_up_exps.weight",
    "ffn_down_exps": "ffn_down_exps.weight",
    "ffn_gate_shexp": "ffn_gate_shexp.weight",
    "ffn_up_shexp": "ffn_up_shexp.weight",
    "ffn_down_shexp": "ffn_down_shexp.weight",
}


class StepFunUnsupportedFeatureError(RuntimeError):
    """Raised when a requested StepFun capability is absent from local assets."""


class _StepGGUFInfo(Protocol):
    metadata: Mapping[str, Any]
    tensors: tuple[GGUFSplitTensorInfo, ...]
    architecture: str | None

    def tensor(self, name: str) -> GGUFSplitTensorInfo: ...


@dataclass(frozen=True)
class StepFunGGUFConfig:
    """StepFun Step 3.5/3.7 GGUF dimensions decoded from metadata."""

    architecture: str
    block_count: int
    hidden_size: int
    vocab_size: int
    context_length: int
    dense_block_count: int
    feed_forward_length: int
    expert_count: int
    expert_used_count: int
    expert_feed_forward_length: int
    expert_shared_feed_forward_length: int
    expert_weights_norm: bool
    expert_weights_scale: float
    rms_norm_eps: float
    head_dim: int
    value_dim: int
    sliding_window: int
    head_counts: tuple[int, ...]
    kv_head_counts: tuple[int, ...]
    sliding_window_pattern: tuple[bool, ...]
    rope_freq_base: float
    rope_freq_base_swa: float
    swiglu_clamp_exp: tuple[float, ...]
    swiglu_clamp_shexp: tuple[float, ...]
    tokenizer_model: str
    tokenizer_pre: str
    bos_token_id: int
    eos_token_id: int
    padding_token_id: int

    @property
    def layer_attention_types(self) -> tuple[str, ...]:
        return tuple(
            SLIDING_ATTENTION if is_sliding else FULL_ATTENTION
            for is_sliding in self.sliding_window_pattern
        )

    @property
    def layer_mlp_types(self) -> tuple[str, ...]:
        return tuple(
            DENSE_MLP if layer_id < self.dense_block_count else MOE
            for layer_id in range(self.block_count)
        )


@dataclass(frozen=True)
class StepFunGGUFMappingValidation:
    config: StepFunGGUFConfig
    present: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_errors: tuple[str, ...]
    type_errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not (
            self.missing or self.unexpected or self.shape_errors or self.type_errors
        )

    def raise_for_errors(self) -> None:
        if self.passed:
            return
        parts: list[str] = []
        if self.missing:
            parts.append(_preview("missing tensors", self.missing))
        if self.unexpected:
            parts.append(_preview("unexpected tensors", self.unexpected))
        if self.shape_errors:
            parts.append(_preview("shape errors", self.shape_errors, separator="; ", limit=4))
        if self.type_errors:
            parts.append(_preview("type errors", self.type_errors, separator="; ", limit=4))
        raise MissingGGUFTensorError("; ".join(parts))


@dataclass(frozen=True)
class StepFunGGUFLayerMap:
    layer_id: int
    attention_type: str
    mlp_type: str
    tensors: Mapping[str, GGUFSplitTensorInfo]

    def tensor(self, slot: str) -> GGUFSplitTensorInfo:
        try:
            return self.tensors[slot]
        except KeyError as exc:
            raise MissingGGUFTensorError(
                f"layer {self.layer_id} has no StepFun GGUF tensor slot {slot!r}"
            ) from exc

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(tensor.name for tensor in self.tensors.values())


@dataclass(frozen=True)
class StepFunGGUFModelMap:
    config: StepFunGGUFConfig
    root_tensors: Mapping[str, GGUFSplitTensorInfo]
    layers: tuple[StepFunGGUFLayerMap, ...]
    validation: StepFunGGUFMappingValidation

    def root(self, slot: str) -> GGUFSplitTensorInfo:
        try:
            return self.root_tensors[slot]
        except KeyError as exc:
            raise MissingGGUFTensorError(f"model has no StepFun GGUF root slot {slot!r}") from exc

    def layer(self, layer_id: int) -> StepFunGGUFLayerMap:
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


def stepfun_gguf_config_from_metadata(info: GGUFSplitModelInfo) -> StepFunGGUFConfig:
    metadata = info.metadata
    architecture = str(metadata.get("general.architecture", ""))
    if architecture != "step35":
        raise ValueError(f"expected GGUF architecture 'step35', got {architecture!r}")
    prefix = architecture
    token_embedding = info.tensor("token_embd.weight")
    block_count = _int_metadata(metadata, f"{prefix}.block_count")
    head_counts = tuple(int(item) for item in metadata.get(f"{prefix}.attention.head_count", ()))
    kv_head_counts = tuple(
        int(item) for item in metadata.get(f"{prefix}.attention.head_count_kv", ())
    )
    sliding_window_pattern = tuple(
        bool(item) for item in metadata.get(f"{prefix}.attention.sliding_window_pattern", ())
    )
    if len(head_counts) != block_count:
        raise ValueError(f"expected {block_count} StepFun head counts, got {len(head_counts)}")
    if len(kv_head_counts) != block_count:
        raise ValueError(
            f"expected {block_count} StepFun KV head counts, got {len(kv_head_counts)}"
        )
    if len(sliding_window_pattern) != block_count:
        count = len(sliding_window_pattern)
        raise ValueError(f"expected {block_count} StepFun sliding-window flags, got {count}")
    return StepFunGGUFConfig(
        architecture=architecture,
        block_count=block_count,
        hidden_size=_int_metadata(metadata, f"{prefix}.embedding_length"),
        vocab_size=int(token_embedding.shape[0]),
        context_length=_int_metadata(metadata, f"{prefix}.context_length"),
        dense_block_count=_int_metadata(metadata, f"{prefix}.leading_dense_block_count"),
        feed_forward_length=_int_metadata(metadata, f"{prefix}.feed_forward_length"),
        expert_count=_int_metadata(metadata, f"{prefix}.expert_count"),
        expert_used_count=_int_metadata(metadata, f"{prefix}.expert_used_count"),
        expert_feed_forward_length=_int_metadata(
            metadata, f"{prefix}.expert_feed_forward_length"
        ),
        expert_shared_feed_forward_length=_int_metadata(
            metadata, f"{prefix}.expert_shared_feed_forward_length"
        ),
        expert_weights_norm=bool(metadata.get(f"{prefix}.expert_weights_norm", False)),
        expert_weights_scale=float(metadata.get(f"{prefix}.expert_weights_scale", 1.0)),
        rms_norm_eps=float(metadata.get(f"{prefix}.attention.layer_norm_rms_epsilon", 1.0e-5)),
        head_dim=_int_metadata(metadata, f"{prefix}.attention.key_length"),
        value_dim=_int_metadata(metadata, f"{prefix}.attention.value_length"),
        sliding_window=_int_metadata(metadata, f"{prefix}.attention.sliding_window"),
        head_counts=head_counts,
        kv_head_counts=kv_head_counts,
        sliding_window_pattern=sliding_window_pattern,
        rope_freq_base=float(metadata.get(f"{prefix}.rope.freq_base", 5_000_000.0)),
        rope_freq_base_swa=float(metadata.get(f"{prefix}.rope.freq_base_swa", 10_000.0)),
        swiglu_clamp_exp=tuple(
            float(item) for item in metadata.get(f"{prefix}.swiglu_clamp_exp", ())
        ),
        swiglu_clamp_shexp=tuple(
            float(item) for item in metadata.get(f"{prefix}.swiglu_clamp_shexp", ())
        ),
        tokenizer_model=str(metadata.get("tokenizer.ggml.model", "")),
        tokenizer_pre=str(metadata.get("tokenizer.ggml.pre", "")),
        bos_token_id=_int_metadata(metadata, "tokenizer.ggml.bos_token_id"),
        eos_token_id=_int_metadata(metadata, "tokenizer.ggml.eos_token_id"),
        padding_token_id=_int_metadata(metadata, "tokenizer.ggml.padding_token_id"),
    )


def required_stepfun_gguf_tensor_names(config: StepFunGGUFConfig) -> tuple[str, ...]:
    names = list(_ROOT_SLOTS.values())
    for layer_id in range(config.block_count):
        names.extend(_layer_required_tensor_names(config, layer_id))
    return tuple(dict.fromkeys(names))


def validate_stepfun_gguf_tensor_map(info: GGUFSplitModelInfo) -> StepFunGGUFMappingValidation:
    config = stepfun_gguf_config_from_metadata(info)
    actual = {tensor.name: tensor for tensor in info.tensors}
    required = set(required_stepfun_gguf_tensor_names(config))
    actual_names = set(actual)
    missing = tuple(sorted(required - actual_names))
    unexpected = tuple(sorted(actual_names - required))
    shape_errors, type_errors = _layout_errors(config, actual)
    present = tuple(sorted(required & actual_names))
    return StepFunGGUFMappingValidation(
        config=config,
        present=present,
        missing=missing,
        unexpected=unexpected,
        shape_errors=tuple(shape_errors),
        type_errors=tuple(type_errors),
    )


def validate_stepfun_multimodal_projector_assets(info: GGUFSplitModelInfo) -> None:
    """Fail clearly when text-only GGUF shards lack projector/vision tensors."""

    projector_names = tuple(
        tensor.name
        for tensor in info.tensors
        if "projector" in tensor.name or "mmproj" in tensor.name or "vision" in tensor.name
    )
    if not projector_names:
        raise StepFunUnsupportedFeatureError(
            "StepFun multimodal projector/vision assets are not present in these GGUF shards; "
            "run text-only mode or provide the separate projector/vision files before requesting "
            "image inputs"
        )


def build_stepfun_gguf_tensor_map(
    info: GGUFSplitModelInfo, *, strict: bool = True
) -> StepFunGGUFModelMap:
    validation = validate_stepfun_gguf_tensor_map(info)
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
    return StepFunGGUFModelMap(
        config=validation.config,
        root_tensors=root_tensors,
        layers=layers,
        validation=validation,
    )


def _build_layer_map(
    config: StepFunGGUFConfig,
    actual: Mapping[str, GGUFSplitTensorInfo],
    layer_id: int,
) -> StepFunGGUFLayerMap:
    attention_type = config.layer_attention_types[layer_id]
    mlp_type = config.layer_mlp_types[layer_id]
    tensors = {
        slot: actual[f"blk.{layer_id}.{suffix}"]
        for slot, suffix in _layer_slot_suffixes(config, layer_id).items()
        if f"blk.{layer_id}.{suffix}" in actual
    }
    return StepFunGGUFLayerMap(
        layer_id=layer_id,
        attention_type=attention_type,
        mlp_type=mlp_type,
        tensors=MappingProxyType(tensors),
    )


def _layer_slot_suffixes(config: StepFunGGUFConfig, layer_id: int) -> dict[str, str]:
    suffixes = dict(_COMMON_LAYER_SLOTS)
    suffixes.update(_DENSE_LAYER_SLOTS if layer_id < config.dense_block_count else _MOE_LAYER_SLOTS)
    return suffixes


def _layer_required_tensor_names(config: StepFunGGUFConfig, layer_id: int) -> tuple[str, ...]:
    return tuple(
        f"blk.{layer_id}.{suffix}"
        for suffix in _layer_slot_suffixes(config, layer_id).values()
    )


def _layout_errors(
    config: StepFunGGUFConfig,
    actual: Mapping[str, GGUFSplitTensorInfo],
) -> tuple[list[str], list[str]]:
    expected_shapes: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (config.vocab_size, config.hidden_size),
        "output.weight": (config.vocab_size, config.hidden_size),
        "output_norm.weight": (config.hidden_size,),
        "rope_freqs.weight": (config.head_dim // 2,),
    }
    expected_types: dict[str, str] = {
        "token_embd.weight": "Q8_0",
        "output.weight": "Q8_0",
        "output_norm.weight": "F32",
        "rope_freqs.weight": "F32",
    }
    for layer_id in range(config.block_count):
        head_count = config.head_counts[layer_id]
        kv_head_count = config.kv_head_counts[layer_id]
        prefix = f"blk.{layer_id}"
        q_width = head_count * config.head_dim
        kv_width = kv_head_count * config.head_dim
        expected_shapes.update(
            {
                f"{prefix}.attn_gate.weight": (head_count, config.hidden_size),
                f"{prefix}.attn_k.weight": (kv_width, config.hidden_size),
                f"{prefix}.attn_k_norm.weight": (config.head_dim,),
                f"{prefix}.attn_norm.weight": (config.hidden_size,),
                f"{prefix}.attn_output.weight": (config.hidden_size, q_width),
                f"{prefix}.attn_q.weight": (q_width, config.hidden_size),
                f"{prefix}.attn_q_norm.weight": (config.head_dim,),
                f"{prefix}.attn_v.weight": (kv_head_count * config.value_dim, config.hidden_size),
                f"{prefix}.ffn_norm.weight": (config.hidden_size,),
            }
        )
        expected_types.update(
            {
                f"{prefix}.attn_gate.weight": "Q3_K",
                f"{prefix}.attn_k.weight": "Q3_K",
                f"{prefix}.attn_k_norm.weight": "F32",
                f"{prefix}.attn_norm.weight": "F32",
                f"{prefix}.attn_output.weight": "Q5_K",
                f"{prefix}.attn_q.weight": "Q3_K",
                f"{prefix}.attn_q_norm.weight": "F32",
                f"{prefix}.attn_v.weight": "Q5_K",
                f"{prefix}.ffn_norm.weight": "F32",
            }
        )
        if layer_id < config.dense_block_count:
            expected_shapes.update(
                {
                    f"{prefix}.ffn_gate.weight": (config.feed_forward_length, config.hidden_size),
                    f"{prefix}.ffn_up.weight": (config.feed_forward_length, config.hidden_size),
                    f"{prefix}.ffn_down.weight": (config.hidden_size, config.feed_forward_length),
                }
            )
            expected_types.update(
                {
                    f"{prefix}.ffn_gate.weight": "Q3_K",
                    f"{prefix}.ffn_up.weight": "Q3_K",
                    f"{prefix}.ffn_down.weight": "Q5_K",
                }
            )
        else:
            expected_shapes.update(
                {
                    f"{prefix}.ffn_gate_inp.weight": (config.expert_count, config.hidden_size),
                    f"{prefix}.exp_probs_b.bias": (config.expert_count,),
                    f"{prefix}.ffn_gate_exps.weight": (
                        config.expert_count,
                        config.expert_feed_forward_length,
                        config.hidden_size,
                    ),
                    f"{prefix}.ffn_up_exps.weight": (
                        config.expert_count,
                        config.expert_feed_forward_length,
                        config.hidden_size,
                    ),
                    f"{prefix}.ffn_down_exps.weight": (
                        config.expert_count,
                        config.hidden_size,
                        config.expert_feed_forward_length,
                    ),
                    f"{prefix}.ffn_gate_shexp.weight": (
                        config.expert_shared_feed_forward_length,
                        config.hidden_size,
                    ),
                    f"{prefix}.ffn_up_shexp.weight": (
                        config.expert_shared_feed_forward_length,
                        config.hidden_size,
                    ),
                    f"{prefix}.ffn_down_shexp.weight": (
                        config.hidden_size,
                        config.expert_shared_feed_forward_length,
                    ),
                }
            )
            expected_types.update(
                {
                    f"{prefix}.ffn_gate_inp.weight": "F32",
                    f"{prefix}.exp_probs_b.bias": "F32",
                    f"{prefix}.ffn_gate_exps.weight": "Q3_K",
                    f"{prefix}.ffn_up_exps.weight": "Q3_K",
                    f"{prefix}.ffn_down_exps.weight": "Q5_K",
                    f"{prefix}.ffn_gate_shexp.weight": "Q3_K",
                    f"{prefix}.ffn_up_shexp.weight": "Q3_K",
                    f"{prefix}.ffn_down_shexp.weight": "Q5_K",
                }
            )

    shape_errors: list[str] = []
    type_errors: list[str] = []
    for name, expected_shape in expected_shapes.items():
        tensor = actual.get(name)
        if tensor is None:
            continue
        if tensor.shape != expected_shape:
            shape_errors.append(f"{name}: expected shape {expected_shape}, got {tensor.shape}")
    for name, expected_type in expected_types.items():
        tensor = actual.get(name)
        if tensor is None:
            continue
        if tensor.ggml_type_name != expected_type:
            type_errors.append(
                f"{name}: expected type {expected_type}, got {tensor.ggml_type_name}"
            )
    return shape_errors, type_errors


def _int_metadata(metadata: Mapping[str, Any], key: str) -> int:
    if key not in metadata:
        raise KeyError(f"missing required StepFun GGUF metadata key {key!r}")
    return int(metadata[key])


def _preview(
    label: str,
    items: tuple[str, ...],
    *,
    separator: str = ", ",
    limit: int = 8,
) -> str:
    preview = separator.join(items[:limit])
    more = "" if len(items) <= limit else f" (+{len(items) - limit} more)"
    return f"{label}: {preview}{more}"


__all__ = [
    "DENSE_MLP",
    "FULL_ATTENTION",
    "MOE",
    "SLIDING_ATTENTION",
    "StepFunGGUFConfig",
    "StepFunGGUFLayerMap",
    "StepFunGGUFMappingValidation",
    "StepFunGGUFModelMap",
    "StepFunUnsupportedFeatureError",
    "build_stepfun_gguf_tensor_map",
    "required_stepfun_gguf_tensor_names",
    "stepfun_gguf_config_from_metadata",
    "validate_stepfun_gguf_tensor_map",
    "validate_stepfun_multimodal_projector_assets",
]
