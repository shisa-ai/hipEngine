"""Torch-free Laguna GGUF metadata and tensor-contract loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from hipengine.loading.gguf import GGUFModelInfo

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"
DENSE_MLP = "dense_mlp"
SPARSE_MOE = "sparse_moe"

_LAGUNA_ARCHITECTURE = "laguna"
_LAGUNA_SIGMOID_GATING_ID = 2
_DEFAULT_SWA_PATTERN = 4


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
    "LagunaRoPEConfig",
    "SPARSE_MOE",
    "SLIDING_ATTENTION",
    "laguna_gguf_config_from_metadata",
]
