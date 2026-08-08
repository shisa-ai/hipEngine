"""Pinned Maple ternary model contract and plugin metadata."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hipengine.models.registry import register_model

MAPLE_ARCHITECTURE = "MapleForCausalLM"
MAPLE_MODEL_TYPE = "maple"
MAPLE_LAYER_PATTERN = (
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
) * 6


@dataclass(frozen=True)
class MapleModelSpec:
    """Validated geometry and storage contract for Maple-Preview 2-bit MLX."""

    architecture: str
    model_type: str
    stored_dtype: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    partial_rotary_factor: float
    rotary_dim: int
    rope_theta: float
    rms_norm_eps: float
    max_position_embeddings: int
    sliding_window: int
    layer_types: tuple[str, ...]
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    vocab_size: int
    bos_token_id: int
    eos_token_id: int
    tie_word_embeddings: bool
    ternary_bits: int
    ternary_group_size: int
    embedding_bits: int
    embedding_group_size: int
    lm_head_bits: int
    lm_head_group_size: int

    @property
    def q_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def attention_kind(self, layer: int) -> str:
        if layer < 0 or layer >= self.num_hidden_layers:
            raise IndexError(f"Maple layer {layer} outside [0, {self.num_hidden_layers})")
        return self.layer_types[layer]

    def uses_rope(self, layer: int) -> bool:
        """Maple applies partial RoPE only on sliding-attention layers."""

        return self.attention_kind(layer) == "sliding_attention"


@dataclass(frozen=True)
class MapleModel:
    """Maple 20B-A1B ternary MoE plugin metadata.

    24 layers of GQA (16 Q / 4 KV heads, head_dim 128) with QK-RMSNorm,
    partial RoPE on the 3-of-4 sliding-window (512) layers only (NoPE on global
    layers), and a 256-expert top-8 MoE with clamped SwiGLU experts.
    """

    name: str = "maple"
    architectures: tuple[str, ...] = (MAPLE_ARCHITECTURE,)
    default_quant: str = "maple_ternary2"
    default_backend: str = "auto"
    weight_name_templates: tuple[str, ...] = (
        "model.word_embeddings.{param}",
        "model.layers.{layer}.input_layernorm.weight",
        "model.layers.{layer}.self_attn.{proj}.weight",
        "model.layers.{layer}.self_attn.{proj}.row_alpha",
        "model.layers.{layer}.self_attn.{norm}.weight",
        "model.layers.{layer}.post_attention_layernorm.weight",
        "model.layers.{layer}.mlp.gate.weight",
        "model.layers.{layer}.mlp.switch_mlp.{proj}.weight",
        "model.layers.{layer}.mlp.switch_mlp.{proj}.row_alpha",
        "model.norm.weight",
        "lm_head.{param}",
    )

    def layer_sequence(self) -> tuple[str, ...]:
        """Return a representative decode sequence for registry/fusion planning."""

        return (
            "embed",
            *self.decode_layer_sequence(attention_kind="sliding_attention"),
            "final_rmsnorm",
            "lm_head",
        )

    def decode_layer_sequence(self, *, attention_kind: str) -> tuple[str, ...]:
        """Return primitive layer keys for one Maple decode layer."""

        if attention_kind == "sliding_attention":
            attention_layers = (
                "rmsnorm",
                "ternary_qkv_proj",
                "qknorm_partial_rope",
                "kv_append",
                "swa_attention_decode",
                "ternary_o_proj",
            )
        elif attention_kind == "full_attention":
            attention_layers = (
                "rmsnorm",
                "ternary_qkv_proj",
                "qknorm",
                "kv_append",
                "full_attention_decode",
                "ternary_o_proj",
            )
        else:
            raise ValueError("attention_kind must be 'sliding_attention' or 'full_attention'")

        return (
            *attention_layers,
            "add_rmsnorm",
            "router_topk_renorm",
            "ternary_expert_gate_up",
            "clamped_swiglu",
            "ternary_expert_down",
            "weighted_sum+residual",
        )


def _positive_int(config: Mapping[str, Any], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Maple {name} must be a positive integer")
    return value


def _token_id(config: Mapping[str, Any], name: str, vocab_size: int) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < vocab_size:
        raise ValueError(f"Maple {name} must be an integer in [0, {vocab_size})")
    return value


def _quant_override(quantization: Mapping[str, Any], name: str) -> tuple[int, int]:
    value = quantization.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"Maple quantization.{name} must be an object")
    bits = value.get("bits")
    group_size = value.get("group_size")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits <= 0:
        raise ValueError(f"Maple quantization.{name}.bits must be a positive integer")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
        raise ValueError(
            f"Maple quantization.{name}.group_size must be a positive integer"
        )
    return bits, group_size


def parse_maple_model_spec(config: Mapping[str, Any]) -> MapleModelSpec:
    """Parse and reject drift from the official Maple-Preview 2-bit contract.

    The shipped MLX model resolves RoPE from ``layer_types``. Its legacy
    ``nope_on_global_attention`` field is not authoritative, so this parser
    intentionally derives NoPE/global behavior from the validated 3:1 pattern.
    """

    expected: dict[str, Any] = {
        "architectures": [MAPLE_ARCHITECTURE],
        "model_type": MAPLE_MODEL_TYPE,
        "dtype": "bfloat16",
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "num_shared_experts": 0,
        "first_k_dense_replace": 0,
        "vocab_size": 151_936,
        "max_position_embeddings": 128_000,
        "sliding_window": 512,
        "hidden_act": "silu",
        "use_bias": False,
        "use_qk_norm": True,
        "norm_topk_prob": True,
        "router_dtype": "fp32",
        "tie_word_embeddings": False,
        "rope_scaling": None,
    }
    for name, expected_value in expected.items():
        if config.get(name) != expected_value:
            raise ValueError(
                f"Maple {name}={config.get(name)!r}, expected {expected_value!r}"
            )

    layer_types_value = config.get("layer_types")
    if not isinstance(layer_types_value, (list, tuple)):
        raise TypeError("Maple layer_types must be an array")
    layer_types = tuple(str(kind) for kind in layer_types_value)
    if layer_types != MAPLE_LAYER_PATTERN:
        raise ValueError("Maple layer_types must use the pinned 3:1 SWA/full pattern")

    hidden_size = _positive_int(config, "hidden_size")
    num_heads = _positive_int(config, "num_attention_heads")
    num_kv_heads = _positive_int(config, "num_key_value_heads")
    head_dim = _positive_int(config, "head_dim")
    if hidden_size != num_heads * head_dim:
        raise ValueError("Maple hidden_size must equal num_attention_heads * head_dim")
    if num_heads % num_kv_heads:
        raise ValueError("Maple num_attention_heads must be divisible by num_key_value_heads")

    partial = float(config.get("partial_rotary_factor", float("nan")))
    rope_theta = float(config.get("rope_theta", float("nan")))
    eps = float(config.get("rms_norm_eps", float("nan")))
    rotary_dim = int(head_dim * partial) if math.isfinite(partial) else -1
    if not math.isclose(partial, 0.5, rel_tol=0.0, abs_tol=1.0e-12) or rotary_dim != 64:
        raise ValueError("Maple partial_rotary_factor must resolve to rotary_dim 64")
    if not math.isclose(rope_theta, 10_000.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Maple rope_theta must equal 10000")
    if not math.isclose(eps, 1.0e-6, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("Maple rms_norm_eps must equal 1e-6")

    quantization = config.get("quantization")
    if not isinstance(quantization, Mapping):
        raise TypeError("Maple quantization must be an object")
    if config.get("quantization_config") != quantization:
        raise ValueError("Maple quantization_config must equal quantization")
    if quantization.get("mode") != "affine":
        raise ValueError("Maple quantization.mode must equal 'affine'")
    ternary_bits = quantization.get("bits")
    ternary_group_size = quantization.get("group_size")
    if ternary_bits != 2 or ternary_group_size != 128:
        raise ValueError("Maple projections require 2-bit group-128 ternary storage")
    embedding_bits, embedding_group_size = _quant_override(
        quantization, "model.word_embeddings"
    )
    lm_head_bits, lm_head_group_size = _quant_override(quantization, "lm_head")
    if (embedding_bits, embedding_group_size) != (4, 64):
        raise ValueError("Maple embeddings require affine 4-bit group-64 storage")
    if (lm_head_bits, lm_head_group_size) != (4, 64):
        raise ValueError("Maple lm_head requires affine 4-bit group-64 storage")

    vocab_size = _positive_int(config, "vocab_size")
    return MapleModelSpec(
        architecture=MAPLE_ARCHITECTURE,
        model_type=MAPLE_MODEL_TYPE,
        stored_dtype="bfloat16",
        hidden_size=hidden_size,
        num_hidden_layers=_positive_int(config, "num_hidden_layers"),
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        partial_rotary_factor=partial,
        rotary_dim=rotary_dim,
        rope_theta=rope_theta,
        rms_norm_eps=eps,
        max_position_embeddings=_positive_int(config, "max_position_embeddings"),
        sliding_window=_positive_int(config, "sliding_window"),
        layer_types=layer_types,
        num_experts=_positive_int(config, "num_experts"),
        num_experts_per_tok=_positive_int(config, "num_experts_per_tok"),
        moe_intermediate_size=_positive_int(config, "moe_intermediate_size"),
        vocab_size=vocab_size,
        bos_token_id=_token_id(config, "bos_token_id", vocab_size),
        eos_token_id=_token_id(config, "eos_token_id", vocab_size),
        tie_word_embeddings=False,
        ternary_bits=int(ternary_bits),
        ternary_group_size=int(ternary_group_size),
        embedding_bits=embedding_bits,
        embedding_group_size=embedding_group_size,
        lm_head_bits=lm_head_bits,
        lm_head_group_size=lm_head_group_size,
    )


MAPLE = register_model(MapleModel())


__all__ = [
    "MAPLE",
    "MAPLE_ARCHITECTURE",
    "MAPLE_LAYER_PATTERN",
    "MAPLE_MODEL_TYPE",
    "MapleModel",
    "MapleModelSpec",
    "parse_maple_model_spec",
]
