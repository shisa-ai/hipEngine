"""Dense Llama-family decoder plugin (HF ``LlamaForCausalLM``).

Covers the plain post-norm Llama decoder: RMSNorm before attention and before
the MLP, grouped-query attention with rotate-half RoPE over the full head
dimension, no attention or MLP bias, and a SiLU-gated dense MLP. Every layer is
full attention, so the whole stack is a single uniform layer sequence.

The plugin is deliberately architecture-level rather than checkpoint-level. A
checkpoint qualifies by what its ``config.json`` declares -- geometry, bias
flags, activation, rope scaling, and weight names -- not by its repository name
or file hash. A configuration this decoder cannot execute raises a named
capability miss instead of loading a partially correct model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hipengine.models.registry import register_model

LLAMA_ARCHITECTURE = "LlamaForCausalLM"
LLAMA_MODEL_TYPE = "llama"
LLAMA_FULL_ATTENTION = "full_attention"

#: Weight-name templates for one decoder layer, in load order.
LLAMA_LAYER_WEIGHT_TEMPLATES: tuple[str, ...] = (
    "model.layers.{layer}.input_layernorm.weight",
    "model.layers.{layer}.self_attn.q_proj.weight",
    "model.layers.{layer}.self_attn.k_proj.weight",
    "model.layers.{layer}.self_attn.v_proj.weight",
    "model.layers.{layer}.self_attn.o_proj.weight",
    "model.layers.{layer}.post_attention_layernorm.weight",
    "model.layers.{layer}.mlp.gate_proj.weight",
    "model.layers.{layer}.mlp.up_proj.weight",
    "model.layers.{layer}.mlp.down_proj.weight",
)

LLAMA_ROOT_WEIGHT_TEMPLATES: tuple[str, ...] = (
    "model.embed_tokens.weight",
    "model.norm.weight",
)


class LlamaCapabilityMiss(ValueError):
    """The checkpoint declares a decoder this implementation cannot execute."""


@dataclass(frozen=True)
class LlamaSpec:
    """Validated decoder geometry and numerics from a checkpoint ``config.json``."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool
    mlp_bias: bool
    hidden_act: str
    rope_scaling: Mapping[str, Any] | None

    @property
    def q_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def query_group_size(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def attention_scale(self) -> float:
        return float(self.head_dim) ** -0.5

    def layer_types(self) -> tuple[str, ...]:
        return tuple(LLAMA_FULL_ATTENTION for _ in range(self.num_hidden_layers))

    def full_attention_layer_ids(self) -> tuple[int, ...]:
        return tuple(range(self.num_hidden_layers))

    def compact_layer_index(self, physical_layer_id: int) -> int:
        """Map a physical layer id to its index among full-attention layers.

        Every Llama layer is full attention, so the mapping is the identity. The
        DMS sidecar still addresses layers through this map, which is what keeps
        the sidecar geometry contract identical to a hybrid model's.
        """

        layer = int(physical_layer_id)
        if layer < 0 or layer >= self.num_hidden_layers:
            raise ValueError(
                f"physical layer {layer} is outside the decoder's {self.num_hidden_layers} layers"
            )
        return layer

    def expected_weight_shapes(self, *, lm_head: bool = True) -> dict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {
            "model.embed_tokens.weight": (self.vocab_size, self.hidden_size),
            "model.norm.weight": (self.hidden_size,),
        }
        if lm_head:
            shapes["lm_head.weight"] = (self.vocab_size, self.hidden_size)
        for layer in range(self.num_hidden_layers):
            shapes[f"model.layers.{layer}.input_layernorm.weight"] = (self.hidden_size,)
            shapes[f"model.layers.{layer}.post_attention_layernorm.weight"] = (self.hidden_size,)
            shapes[f"model.layers.{layer}.self_attn.q_proj.weight"] = (
                self.q_width,
                self.hidden_size,
            )
            shapes[f"model.layers.{layer}.self_attn.k_proj.weight"] = (
                self.kv_width,
                self.hidden_size,
            )
            shapes[f"model.layers.{layer}.self_attn.v_proj.weight"] = (
                self.kv_width,
                self.hidden_size,
            )
            shapes[f"model.layers.{layer}.self_attn.o_proj.weight"] = (
                self.hidden_size,
                self.q_width,
            )
            shapes[f"model.layers.{layer}.mlp.gate_proj.weight"] = (
                self.intermediate_size,
                self.hidden_size,
            )
            shapes[f"model.layers.{layer}.mlp.up_proj.weight"] = (
                self.intermediate_size,
                self.hidden_size,
            )
            shapes[f"model.layers.{layer}.mlp.down_proj.weight"] = (
                self.hidden_size,
                self.intermediate_size,
            )
        return shapes


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise LlamaCapabilityMiss(f"checkpoint config is missing required field {key!r}")
    return mapping[key]


def _positive_int(mapping: Mapping[str, Any], key: str) -> int:
    value = _require(mapping, key)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LlamaCapabilityMiss(f"config field {key!r} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise LlamaCapabilityMiss(f"config field {key!r} must be positive, got {parsed}")
    return parsed


def parse_llama_spec(config: Mapping[str, Any]) -> LlamaSpec:
    """Validate a checkpoint config against what this decoder can execute.

    Raises :class:`LlamaCapabilityMiss` naming the unsupported declaration. A
    missing evidence row, benchmark, or model card never reaches this function:
    admission is decided by the declared geometry and numerics only.
    """

    model_type = str(config.get("model_type", ""))
    if model_type and model_type != LLAMA_MODEL_TYPE:
        raise LlamaCapabilityMiss(
            f"expected model_type {LLAMA_MODEL_TYPE!r} for a dense Llama decoder, "
            f"got {model_type!r}"
        )

    attention_bias = bool(config.get("attention_bias", False))
    if attention_bias:
        raise LlamaCapabilityMiss("attention_bias=true is not implemented by the dense Llama decoder")
    mlp_bias = bool(config.get("mlp_bias", False))
    if mlp_bias:
        raise LlamaCapabilityMiss("mlp_bias=true is not implemented by the dense Llama decoder")

    hidden_act = str(config.get("hidden_act", config.get("activation", "silu")))
    if hidden_act != "silu":
        raise LlamaCapabilityMiss(
            f"hidden_act={hidden_act!r} is not implemented; the dense Llama decoder runs SiLU-gated MLPs"
        )

    rope_scaling = config.get("rope_scaling")
    if rope_scaling:
        raise LlamaCapabilityMiss(
            "rope_scaling is not implemented; only unmodified rotate-half RoPE at rope_theta is supported"
        )

    hidden_size = _positive_int(config, "hidden_size")
    num_attention_heads = _positive_int(config, "num_attention_heads")
    num_key_value_heads = _positive_int(config, "num_key_value_heads")
    head_dim = int(config.get("head_dim") or 0) or hidden_size // num_attention_heads
    if head_dim <= 0:
        raise LlamaCapabilityMiss("head_dim could not be derived from the config")
    if num_attention_heads % num_key_value_heads:
        raise LlamaCapabilityMiss(
            f"num_attention_heads ({num_attention_heads}) must be divisible by "
            f"num_key_value_heads ({num_key_value_heads})"
        )
    if head_dim % 2:
        raise LlamaCapabilityMiss(f"head_dim must be even for rotate-half RoPE, got {head_dim}")
    declared_heads = int(config.get("num_query_groups") or 0)
    if declared_heads and declared_heads != num_key_value_heads:
        raise LlamaCapabilityMiss(
            f"num_query_groups ({declared_heads}) disagrees with "
            f"num_key_value_heads ({num_key_value_heads})"
        )

    intermediate_size = int(
        config.get("intermediate_size") or config.get("ffn_hidden_size") or 0
    )
    if intermediate_size <= 0:
        raise LlamaCapabilityMiss("config declares no positive intermediate_size")
    declared_ffn = int(config.get("ffn_hidden_size") or 0)
    if declared_ffn and declared_ffn != intermediate_size:
        raise LlamaCapabilityMiss(
            f"ffn_hidden_size ({declared_ffn}) disagrees with intermediate_size ({intermediate_size})"
        )

    rope_theta = float(config.get("rope_theta") or config.get("rotary_base") or 0.0)
    if rope_theta <= 0.0:
        raise LlamaCapabilityMiss("config declares no positive rope_theta")
    declared_rotary_base = float(config.get("rotary_base") or 0.0)
    if declared_rotary_base and declared_rotary_base != rope_theta:
        raise LlamaCapabilityMiss(
            f"rotary_base ({declared_rotary_base}) disagrees with rope_theta ({rope_theta})"
        )

    return LlamaSpec(
        hidden_size=hidden_size,
        num_hidden_layers=_positive_int(config, "num_hidden_layers"),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        vocab_size=_positive_int(config, "vocab_size"),
        rms_norm_eps=float(config.get("rms_norm_eps", 1.0e-6)),
        rope_theta=rope_theta,
        max_position_embeddings=int(config.get("max_position_embeddings", 0) or 0),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        attention_bias=attention_bias,
        mlp_bias=mlp_bias,
        hidden_act=hidden_act,
        rope_scaling=rope_scaling,
    )


@dataclass(frozen=True)
class LlamaModel:
    """Registry plugin for the dense Llama decoder."""

    name: str = "llama_dense"
    architectures: tuple[str, ...] = (LLAMA_ARCHITECTURE,)
    default_quant: str = "bf16"
    default_backend: str = "auto"
    weight_name_templates: tuple[str, ...] = (
        *LLAMA_ROOT_WEIGHT_TEMPLATES,
        "lm_head.weight",
        *LLAMA_LAYER_WEIGHT_TEMPLATES,
    )

    def layer_sequence(self) -> Sequence[str]:
        return (
            "embed",
            "rmsnorm",
            "qkv_proj",
            "rotate",
            "attention_prefill",
            "o_proj",
            "rmsnorm",
            "gate_up_proj",
            "silu_mul",
            "down_proj",
            "final_rmsnorm",
            "lm_head",
        )


LLAMA_DENSE = register_model(LlamaModel())


__all__ = [
    "LLAMA_ARCHITECTURE",
    "LLAMA_DENSE",
    "LLAMA_FULL_ATTENTION",
    "LLAMA_LAYER_WEIGHT_TEMPLATES",
    "LLAMA_MODEL_TYPE",
    "LLAMA_ROOT_WEIGHT_TEMPLATES",
    "LlamaCapabilityMiss",
    "LlamaModel",
    "LlamaSpec",
    "parse_llama_spec",
]
