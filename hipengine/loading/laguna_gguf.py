"""Torch-free Laguna GGUF metadata and tensor-contract loading."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from hipengine.loading.gguf import (
    GGUFModelInfo,
    GGUFTensorInfo,
    MissingGGUFTensorError,
)
from hipengine.quant.gguf import GGMLQuantizationType

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"
DENSE_MLP = "dense_mlp"
SPARSE_MOE = "sparse_moe"
PER_HEAD_GATE = "per_head"
PER_ELEMENT_GATE = "per_element"

_LAGUNA_ARCHITECTURE = "laguna"
_INVALID_GATE = "invalid"
_LAGUNA_SIGMOID_GATING_ID = 2
_DEFAULT_SWA_PATTERN = 4

_ROOT_SLOTS = {
    "token_embedding": "token_embd.weight",
    "output_norm": "output_norm.weight",
    "lm_head": "output.weight",
}
_COMMON_LAYER_SLOTS = {
    "attn_norm": "attn_norm.weight",
    "attn_q": "attn_q.weight",
    "attn_k": "attn_k.weight",
    "attn_v": "attn_v.weight",
    "attn_gate": "attn_gate.weight",
    "attn_q_norm": "attn_q_norm.weight",
    "attn_k_norm": "attn_k_norm.weight",
    "attn_output": "attn_output.weight",
    "ffn_norm": "ffn_norm.weight",
}
_DENSE_LAYER_SLOTS = {
    "ffn_gate": "ffn_gate.weight",
    "ffn_up": "ffn_up.weight",
    "ffn_down": "ffn_down.weight",
}
_SPARSE_LAYER_SLOTS = {
    "ffn_gate_inp": "ffn_gate_inp.weight",
    "exp_probs_b": "exp_probs_b.bias",
    "ffn_gate_exps": "ffn_gate_exps.weight",
    "ffn_up_exps": "ffn_up_exps.weight",
    "ffn_down_exps": "ffn_down_exps.weight",
    "ffn_gate_shexp": "ffn_gate_shexp.weight",
    "ffn_up_shexp": "ffn_up_shexp.weight",
    "ffn_down_shexp": "ffn_down_shexp.weight",
}


@dataclass(frozen=True)
class LagunaRoPEConfig:
    """One layer-family RoPE contract decoded from GGUF metadata."""

    rope_type: str
    dimension_count: int
    freq_base: float
    scaling_factor: float = 1.0
    original_context_length: int = 0
    yarn_attn_factor: float = 1.0
    yarn_beta_fast: float = 0.0
    yarn_beta_slow: float = 0.0


@dataclass(frozen=True)
class LagunaGGUFConfig:
    """Validated architecture dimensions for a Laguna GGUF model."""

    architecture: str
    block_count: int
    hidden_size: int
    vocab_size: int
    feed_forward_length: int
    context_length: int
    head_counts: tuple[int, ...]
    head_count_kv: int
    key_length: int
    value_length: int
    rms_norm_eps: float
    sliding_window: int
    sliding_window_pattern: int
    layer_types: tuple[str, ...]
    mlp_layer_types: tuple[str, ...]
    full_rope: LagunaRoPEConfig
    swa_rope: LagunaRoPEConfig | None
    expert_count: int
    expert_used_count: int
    expert_feed_forward_length: int
    expert_shared_feed_forward_length: int
    expert_weights_norm: bool
    expert_weights_scale: float
    expert_gating_func: str
    leading_dense_block_count: int

    def _check_layer(self, layer_id: int) -> int:
        layer = int(layer_id)
        if layer < 0 or layer >= self.block_count:
            raise IndexError(f"layer_id {layer} outside [0, {self.block_count})")
        return layer

    def head_count(self, layer_id: int) -> int:
        return self.head_counts[self._check_layer(layer_id)]

    def layer_type(self, layer_id: int) -> str:
        return self.layer_types[self._check_layer(layer_id)]

    def mlp_type(self, layer_id: int) -> str:
        return self.mlp_layer_types[self._check_layer(layer_id)]

    def rope_for_layer(self, layer_id: int) -> LagunaRoPEConfig:
        layer = self._check_layer(layer_id)
        if self.layer_types[layer] == SLIDING_ATTENTION:
            if self.swa_rope is None:
                raise ValueError("sliding-attention layer has no SWA RoPE config")
            return self.swa_rope
        return self.full_rope


@dataclass(frozen=True)
class LagunaGGUFMappingValidation:
    """Result of validating the architecture's complete tensor inventory."""

    config: LagunaGGUFConfig
    present: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_errors: tuple[str, ...]
    type_errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not (
            self.missing
            or self.unexpected
            or self.shape_errors
            or self.type_errors
        )

    def raise_for_errors(self) -> None:
        if self.passed:
            return
        parts: list[str] = []
        for label, errors, limit in (
            ("missing tensors", self.missing, 8),
            ("unexpected tensors", self.unexpected, 8),
            ("shape errors", self.shape_errors, 4),
            ("type errors", self.type_errors, 4),
        ):
            if not errors:
                continue
            preview = "; ".join(errors[:limit])
            more = "" if len(errors) <= limit else f" (+{len(errors) - limit} more)"
            parts.append(f"{label}: {preview}{more}")
        raise MissingGGUFTensorError("; ".join(parts))


@dataclass(frozen=True)
class LagunaGGUFLayerMap:
    """Canonical Laguna tensor slots and execution kinds for one layer."""

    layer_id: int
    attention_type: str
    mlp_type: str
    attention_gate_type: str
    tensors: Mapping[str, GGUFTensorInfo]

    def tensor(self, slot: str) -> GGUFTensorInfo:
        try:
            return self.tensors[slot]
        except KeyError as exc:
            raise MissingGGUFTensorError(
                f"Laguna layer {self.layer_id} has no GGUF tensor slot {slot!r}"
            ) from exc

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(tensor.name for tensor in self.tensors.values())


@dataclass(frozen=True)
class LagunaGGUFModelMap:
    """Canonical root and layer tensor map for one Laguna GGUF artifact."""

    config: LagunaGGUFConfig
    root_tensors: Mapping[str, GGUFTensorInfo]
    layers: tuple[LagunaGGUFLayerMap, ...]
    validation: LagunaGGUFMappingValidation

    def root(self, slot: str) -> GGUFTensorInfo:
        try:
            return self.root_tensors[slot]
        except KeyError as exc:
            raise MissingGGUFTensorError(
                f"Laguna model has no GGUF root tensor slot {slot!r}"
            ) from exc

    def layer(self, layer_id: int) -> LagunaGGUFLayerMap:
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


def laguna_gguf_config_from_metadata(info: GGUFModelInfo) -> LagunaGGUFConfig:
    """Decode and validate Laguna architecture metadata without reading weights."""

    metadata = info.metadata
    architecture = str(metadata.get("general.architecture", ""))
    if architecture != _LAGUNA_ARCHITECTURE:
        raise ValueError(
            f"expected GGUF architecture 'laguna', got {architecture!r}"
        )

    prefix = _LAGUNA_ARCHITECTURE
    block_count = _positive_int(metadata, f"{prefix}.block_count")
    hidden_size = _positive_int(metadata, f"{prefix}.embedding_length")
    vocab_size = _positive_int(metadata, f"{prefix}.vocab_size")
    feed_forward_length = _positive_int(
        metadata,
        f"{prefix}.feed_forward_length",
    )
    context_length = _positive_int(metadata, f"{prefix}.context_length")
    head_count_kv = _positive_int(
        metadata,
        f"{prefix}.attention.head_count_kv",
    )
    head_counts = _head_counts(
        metadata,
        f"{prefix}.attention.head_count",
        block_count=block_count,
    )
    for layer_id, heads in enumerate(head_counts):
        if heads % head_count_kv:
            raise ValueError(
                f"laguna attention head_count {heads} at layer {layer_id} must be "
                f"divisible by head_count_kv {head_count_kv}"
            )

    key_length = _positive_int(metadata, f"{prefix}.attention.key_length")
    value_length = _positive_int(metadata, f"{prefix}.attention.value_length")
    rms_norm_eps = float(
        _required(metadata, f"{prefix}.attention.layer_norm_rms_epsilon")
    )
    if rms_norm_eps <= 0.0:
        raise ValueError("Laguna RMS epsilon must be positive")

    sliding_window = int(metadata.get(f"{prefix}.attention.sliding_window", 0) or 0)
    if sliding_window < 0:
        raise ValueError("Laguna sliding window must be non-negative")
    if sliding_window:
        sliding_pattern = int(
            metadata.get(
                f"{prefix}.attention.sliding_window_pattern",
                _DEFAULT_SWA_PATTERN,
            )
            or _DEFAULT_SWA_PATTERN
        )
        if sliding_pattern < 2:
            raise ValueError("Laguna sliding-window pattern must be at least 2")
        layer_types = tuple(
            FULL_ATTENTION if layer_id % sliding_pattern == 0 else SLIDING_ATTENTION
            for layer_id in range(block_count)
        )
    else:
        sliding_pattern = 0
        layer_types = (FULL_ATTENTION,) * block_count

    full_rope = _full_rope_config(metadata, prefix=prefix)
    swa_rope = (
        _swa_rope_config(metadata, prefix=prefix) if sliding_window else None
    )

    expert_count = _positive_int(metadata, f"{prefix}.expert_count")
    expert_used_count = _positive_int(metadata, f"{prefix}.expert_used_count")
    if expert_used_count > expert_count:
        raise ValueError(
            "Laguna expert_used_count must be <= expert_count"
        )
    expert_feed_forward_length = _positive_int(
        metadata,
        f"{prefix}.expert_feed_forward_length",
    )
    expert_shared_feed_forward_length = _positive_int(
        metadata,
        f"{prefix}.expert_shared_feed_forward_length",
    )
    expert_weights_norm = bool(
        metadata.get(f"{prefix}.expert_weights_norm", True)
    )
    expert_weights_scale = float(
        metadata.get(f"{prefix}.expert_weights_scale", 1.0)
    )
    if expert_weights_scale <= 0.0:
        raise ValueError("Laguna expert weights scale must be positive")
    gating_id = int(
        metadata.get(
            f"{prefix}.expert_gating_func",
            _LAGUNA_SIGMOID_GATING_ID,
        )
        or _LAGUNA_SIGMOID_GATING_ID
    )
    if gating_id != _LAGUNA_SIGMOID_GATING_ID:
        raise ValueError(
            "Laguna requires sigmoid expert gating "
            f"(GGUF id {_LAGUNA_SIGMOID_GATING_ID}), got {gating_id}"
        )

    leading_dense = int(
        metadata.get(f"{prefix}.leading_dense_block_count", 0) or 0
    )
    if leading_dense < 0 or leading_dense > block_count:
        raise ValueError(
            "Laguna leading_dense_block_count must be within the layer count"
        )
    mlp_layer_types = tuple(
        DENSE_MLP if layer_id < leading_dense else SPARSE_MOE
        for layer_id in range(block_count)
    )

    return LagunaGGUFConfig(
        architecture=architecture,
        block_count=block_count,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        feed_forward_length=feed_forward_length,
        context_length=context_length,
        head_counts=head_counts,
        head_count_kv=head_count_kv,
        key_length=key_length,
        value_length=value_length,
        rms_norm_eps=rms_norm_eps,
        sliding_window=sliding_window,
        sliding_window_pattern=sliding_pattern,
        layer_types=layer_types,
        mlp_layer_types=mlp_layer_types,
        full_rope=full_rope,
        swa_rope=swa_rope,
        expert_count=expert_count,
        expert_used_count=expert_used_count,
        expert_feed_forward_length=expert_feed_forward_length,
        expert_shared_feed_forward_length=expert_shared_feed_forward_length,
        expert_weights_norm=expert_weights_norm,
        expert_weights_scale=expert_weights_scale,
        expert_gating_func="sigmoid",
        leading_dense_block_count=leading_dense,
    )


def required_laguna_gguf_tensor_names(
    config: LagunaGGUFConfig,
) -> tuple[str, ...]:
    names = list(_ROOT_SLOTS.values())
    for layer_id in range(config.block_count):
        names.extend(
            f"blk.{layer_id}.{suffix}"
            for suffix in _layer_slots(config, layer_id).values()
        )
    return tuple(dict.fromkeys(names))


def validate_laguna_gguf_tensor_map(
    info: GGUFModelInfo,
) -> LagunaGGUFMappingValidation:
    config = laguna_gguf_config_from_metadata(info)
    actual = {tensor.name: tensor for tensor in info.tensors}
    required = set(required_laguna_gguf_tensor_names(config))
    actual_names = set(actual)
    return LagunaGGUFMappingValidation(
        config=config,
        present=tuple(sorted(required & actual_names)),
        missing=tuple(sorted(required - actual_names)),
        unexpected=tuple(sorted(actual_names - required)),
        shape_errors=tuple(_tensor_shape_errors(config, actual)),
        type_errors=tuple(_tensor_type_errors(config, actual)),
    )


def build_laguna_gguf_tensor_map(
    info: GGUFModelInfo,
    *,
    strict: bool = True,
) -> LagunaGGUFModelMap:
    validation = validate_laguna_gguf_tensor_map(info)
    if strict:
        validation.raise_for_errors()
    actual = {tensor.name: tensor for tensor in info.tensors}
    roots = MappingProxyType(
        {
            slot: actual[name]
            for slot, name in _ROOT_SLOTS.items()
            if name in actual
        }
    )
    layers = tuple(
        _build_layer_map(validation.config, actual, layer_id)
        for layer_id in range(validation.config.block_count)
    )
    return LagunaGGUFModelMap(
        config=validation.config,
        root_tensors=roots,
        layers=layers,
        validation=validation,
    )


def _build_layer_map(
    config: LagunaGGUFConfig,
    actual: Mapping[str, GGUFTensorInfo],
    layer_id: int,
) -> LagunaGGUFLayerMap:
    slots = _layer_slots(config, layer_id)
    tensors = MappingProxyType(
        {
            slot: actual[f"blk.{layer_id}.{suffix}"]
            for slot, suffix in slots.items()
            if f"blk.{layer_id}.{suffix}" in actual
        }
    )
    gate = tensors.get("attn_gate")
    per_element_shape = (
        config.head_count(layer_id) * config.key_length,
        config.hidden_size,
    )
    per_head_shape = (config.head_count(layer_id), config.hidden_size)
    if gate is not None and gate.shape == per_element_shape:
        gate_type = PER_ELEMENT_GATE
    elif gate is not None and gate.shape == per_head_shape:
        gate_type = PER_HEAD_GATE
    else:
        gate_type = _INVALID_GATE
    return LagunaGGUFLayerMap(
        layer_id=layer_id,
        attention_type=config.layer_type(layer_id),
        mlp_type=config.mlp_type(layer_id),
        attention_gate_type=gate_type,
        tensors=tensors,
    )


def _layer_slots(
    config: LagunaGGUFConfig,
    layer_id: int,
) -> dict[str, str]:
    slots = dict(_COMMON_LAYER_SLOTS)
    if config.mlp_type(layer_id) == DENSE_MLP:
        slots.update(_DENSE_LAYER_SLOTS)
    else:
        slots.update(_SPARSE_LAYER_SLOTS)
    return slots


def _tensor_shape_errors(
    config: LagunaGGUFConfig,
    actual: Mapping[str, GGUFTensorInfo],
) -> list[str]:
    expected: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (config.vocab_size, config.hidden_size),
        "output_norm.weight": (config.hidden_size,),
        "output.weight": (config.vocab_size, config.hidden_size),
    }
    gate_shapes: dict[str, tuple[tuple[int, ...], ...]] = {}
    for layer_id in range(config.block_count):
        prefix = f"blk.{layer_id}"
        heads = config.head_count(layer_id)
        q_width = heads * config.key_length
        output_width = heads * config.value_length
        expected.update(
            {
                f"{prefix}.attn_norm.weight": (config.hidden_size,),
                f"{prefix}.attn_q.weight": (q_width, config.hidden_size),
                f"{prefix}.attn_k.weight": (
                    config.head_count_kv * config.key_length,
                    config.hidden_size,
                ),
                f"{prefix}.attn_v.weight": (
                    config.head_count_kv * config.value_length,
                    config.hidden_size,
                ),
                f"{prefix}.attn_q_norm.weight": (config.key_length,),
                f"{prefix}.attn_k_norm.weight": (config.key_length,),
                f"{prefix}.attn_output.weight": (
                    config.hidden_size,
                    output_width,
                ),
                f"{prefix}.ffn_norm.weight": (config.hidden_size,),
            }
        )
        gate_shapes[f"{prefix}.attn_gate.weight"] = (
            (heads, config.hidden_size),
            (heads * config.key_length, config.hidden_size),
        )
        if config.mlp_type(layer_id) == DENSE_MLP:
            expected.update(
                {
                    f"{prefix}.ffn_gate.weight": (
                        config.feed_forward_length,
                        config.hidden_size,
                    ),
                    f"{prefix}.ffn_up.weight": (
                        config.feed_forward_length,
                        config.hidden_size,
                    ),
                    f"{prefix}.ffn_down.weight": (
                        config.hidden_size,
                        config.feed_forward_length,
                    ),
                }
            )
        else:
            expected.update(
                {
                    f"{prefix}.ffn_gate_inp.weight": (
                        config.expert_count,
                        config.hidden_size,
                    ),
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

    errors: list[str] = []
    for name, shape in expected.items():
        tensor = actual.get(name)
        if tensor is not None and tensor.shape != shape:
            errors.append(f"{name}: expected shape {shape}, got {tensor.shape}")
    for name, shapes in gate_shapes.items():
        tensor = actual.get(name)
        if tensor is not None and tensor.shape not in shapes:
            errors.append(
                f"{name}: expected per-head/per-element shape {shapes}, got {tensor.shape}"
            )
    return errors


def _tensor_type_errors(
    config: LagunaGGUFConfig,
    actual: Mapping[str, GGUFTensorInfo],
) -> list[str]:
    q = GGMLQuantizationType
    expected: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (int(q.Q4_K), int(q.Q5_K)),
        "output_norm.weight": (int(q.F32),),
        "output.weight": (int(q.Q4_K), int(q.Q6_K)),
    }
    for layer_id in range(config.block_count):
        prefix = f"blk.{layer_id}"
        expected.update(
            {
                f"{prefix}.attn_norm.weight": (int(q.F32),),
                f"{prefix}.attn_q.weight": (
                    int(q.F16),
                    int(q.Q5_K),
                    int(q.Q6_K),
                ),
                f"{prefix}.attn_k.weight": (
                    int(q.F16),
                    int(q.Q6_K),
                    int(q.Q8_0),
                ),
                f"{prefix}.attn_v.weight": (
                    int(q.F16),
                    int(q.Q6_K),
                    int(q.Q8_0),
                ),
                f"{prefix}.attn_gate.weight": (
                    int(q.F16),
                    int(q.Q5_K),
                    int(q.Q6_K),
                ),
                f"{prefix}.attn_q_norm.weight": (int(q.F32),),
                f"{prefix}.attn_k_norm.weight": (int(q.F32),),
                f"{prefix}.attn_output.weight": (
                    int(q.F16),
                    int(q.Q5_K),
                    int(q.Q6_K),
                ),
                f"{prefix}.ffn_norm.weight": (int(q.F32),),
            }
        )
        if config.mlp_type(layer_id) == DENSE_MLP:
            expected.update(
                {
                    f"{prefix}.ffn_gate.weight": (
                        int(q.Q4_K),
                        int(q.Q5_K),
                    ),
                    f"{prefix}.ffn_up.weight": (
                        int(q.Q4_K),
                        int(q.Q5_K),
                    ),
                    f"{prefix}.ffn_down.weight": (int(q.Q6_K),),
                }
            )
        else:
            expected.update(
                {
                    f"{prefix}.ffn_gate_inp.weight": (int(q.F32),),
                    f"{prefix}.exp_probs_b.bias": (int(q.F32),),
                    f"{prefix}.ffn_gate_exps.weight": (
                        int(q.Q4_K),
                        int(q.IQ2_XS),
                        int(q.IQ3_XXS),
                    ),
                    f"{prefix}.ffn_up_exps.weight": (
                        int(q.Q4_K),
                        int(q.IQ2_XS),
                        int(q.IQ3_XXS),
                    ),
                    f"{prefix}.ffn_down_exps.weight": (
                        int(q.Q4_K),
                        int(q.Q6_K),
                        int(q.IQ3_XXS),
                        int(q.IQ4_XS),
                    ),
                    f"{prefix}.ffn_gate_shexp.weight": (
                        int(q.Q4_K),
                        int(q.Q5_K),
                        int(q.Q6_K),
                    ),
                    f"{prefix}.ffn_up_shexp.weight": (
                        int(q.Q4_K),
                        int(q.Q5_K),
                        int(q.Q6_K),
                    ),
                    f"{prefix}.ffn_down_shexp.weight": (
                        int(q.Q4_K),
                        int(q.Q6_K),
                        int(q.Q8_0),
                    ),
                }
            )

    errors: list[str] = []
    for name, allowed in expected.items():
        tensor = actual.get(name)
        if tensor is not None and tensor.ggml_type not in allowed:
            names = tuple(GGMLQuantizationType(item).name for item in allowed)
            errors.append(
                f"{name}: expected GGML type {names}, got {tensor.ggml_type_name}"
            )
    return errors


def _full_rope_config(
    metadata: Mapping[str, Any],
    *,
    prefix: str,
) -> LagunaRoPEConfig:
    rope_type = str(metadata.get(f"{prefix}.rope.scaling.type", "default"))
    dimension = _positive_int(metadata, f"{prefix}.rope.dimension_count")
    freq_base = float(_required(metadata, f"{prefix}.rope.freq_base"))
    if freq_base <= 0.0:
        raise ValueError("Laguna RoPE frequency base must be positive")
    scaling_factor = float(metadata.get(f"{prefix}.rope.scaling.factor", 1.0))
    if scaling_factor <= 0.0:
        raise ValueError("Laguna RoPE scaling factor must be positive")
    original_context = int(
        metadata.get(f"{prefix}.rope.scaling.original_context_length", 0) or 0
    )
    if rope_type == "yarn" and original_context <= 0:
        raise ValueError("Laguna YaRN RoPE requires original_context_length")
    return LagunaRoPEConfig(
        rope_type=rope_type,
        dimension_count=dimension,
        freq_base=freq_base,
        scaling_factor=scaling_factor,
        original_context_length=original_context,
        yarn_attn_factor=float(
            metadata.get(f"{prefix}.rope.scaling.yarn_attn_factor", 1.0)
        ),
        yarn_beta_fast=float(
            metadata.get(f"{prefix}.rope.scaling.yarn_beta_fast", 0.0)
        ),
        yarn_beta_slow=float(
            metadata.get(f"{prefix}.rope.scaling.yarn_beta_slow", 0.0)
        ),
    )


def _swa_rope_config(
    metadata: Mapping[str, Any],
    *,
    prefix: str,
) -> LagunaRoPEConfig:
    dimension = _positive_int(metadata, f"{prefix}.rope.dimension_count_swa")
    freq_base = float(_required(metadata, f"{prefix}.rope.freq_base_swa"))
    if freq_base <= 0.0:
        raise ValueError("Laguna SWA RoPE frequency base must be positive")
    return LagunaRoPEConfig(
        rope_type="default",
        dimension_count=dimension,
        freq_base=freq_base,
    )


def _head_counts(
    metadata: Mapping[str, Any],
    key: str,
    *,
    block_count: int,
) -> tuple[int, ...]:
    value = _required(metadata, key)
    if isinstance(value, (list, tuple)):
        result = tuple(int(item) for item in value)
        if len(result) != block_count:
            raise ValueError(
                f"Laguna {key} array must have {block_count} entries, got {len(result)}"
            )
    else:
        result = (int(value),) * block_count
    if any(item <= 0 for item in result):
        raise ValueError(f"Laguna {key} values must be positive")
    return result


def _required(metadata: Mapping[str, Any], key: str) -> Any:
    try:
        return metadata[key]
    except KeyError as exc:
        raise KeyError(f"missing required Laguna GGUF metadata key {key!r}") from exc


def _positive_int(metadata: Mapping[str, Any], key: str) -> int:
    value = int(_required(metadata, key))
    if value <= 0:
        raise ValueError(f"Laguna metadata {key!r} must be positive")
    return value


__all__ = [
    "DENSE_MLP",
    "FULL_ATTENTION",
    "LagunaGGUFConfig",
    "LagunaGGUFLayerMap",
    "LagunaGGUFMappingValidation",
    "LagunaGGUFModelMap",
    "LagunaRoPEConfig",
    "PER_ELEMENT_GATE",
    "PER_HEAD_GATE",
    "SPARSE_MOE",
    "SLIDING_ATTENTION",
    "build_laguna_gguf_tensor_map",
    "laguna_gguf_config_from_metadata",
    "required_laguna_gguf_tensor_names",
    "validate_laguna_gguf_tensor_map",
]
