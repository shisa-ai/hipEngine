"""Qwen3.5/PARO model plugin metadata."""

from __future__ import annotations

from dataclasses import dataclass

from hipengine.models.registry import register_model


@dataclass(frozen=True)
class Qwen35ParoMoeModel:
    """Qwen3.5 MoE decode metadata for the PARO/W4A16 path.

    This plugin is intentionally metadata-only: it gives the planner stable layer keys and
    records the canonical HF architecture/weight-name shape without loading tensors or
    importing torch. Config-driven layer repetition and attention-specific parameters will
    live in the loader/model-spec layer.
    """

    name: str = "qwen3_5_moe_paro"
    architectures: tuple[str, ...] = (
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
    )
    default_quant: str = "w4_paro"
    default_backend: str = "auto"
    weight_name_templates: tuple[str, ...] = (
        "model.embed_tokens.weight",
        "model.layers.{layer}.input_layernorm.weight",
        "model.layers.{layer}.self_attn.{proj}.qweight",
        "model.layers.{layer}.self_attn.{proj}.qzeros",
        "model.layers.{layer}.self_attn.{proj}.scales",
        "model.layers.{layer}.post_attention_layernorm.weight",
        "model.layers.{layer}.mlp.gate.weight",
        "model.layers.{layer}.mlp.experts.{expert}.{proj}.qweight",
        "model.layers.{layer}.mlp.experts.{expert}.{proj}.qzeros",
        "model.layers.{layer}.mlp.experts.{expert}.{proj}.scales",
        "model.layers.{layer}.mlp.shared_expert.{proj}.weight",
        "model.layers.{layer}.mlp.shared_expert_gate.weight",
        "model.norm.weight",
        "lm_head.weight",
    )

    def layer_sequence(self) -> tuple[str, ...]:
        """Return a representative decode sequence for registry/fusion planning."""

        return (
            "embed",
            *self.decode_layer_sequence(attention_kind="full_attention"),
            "final_rmsnorm",
            "lm_head",
        )

    def decode_layer_sequence(self, *, attention_kind: str) -> tuple[str, ...]:
        """Return primitive layer keys for one Qwen3.5 decode layer.

        ``attention_kind`` mirrors Qwen3.5's config-level ``layer_types`` entries.
        """

        if attention_kind == "full_attention":
            attention_layers = (
                "rmsnorm",
                "full_attention_qkv_proj",
                "rope",
                "paged_kv_write",
                "full_attention_decode",
                "full_attention_o_proj",
            )
        elif attention_kind == "linear_attention":
            attention_layers = (
                "rmsnorm",
                "linear_attention_qkvz_proj",
                "linear_attention_conv_decode",
                "linear_attention_recurrence",
                "linear_attention_o_proj",
            )
        else:
            raise ValueError("attention_kind must be 'full_attention' or 'linear_attention'")

        return (
            *attention_layers,
            "add_rmsnorm",
            "router_topk_shared",
            "selected_dual_pack8_gemv",
            "silu_mul_dual_rotate",
            "selected_pack8_gemv",
            "w8a16_linear",
            "weighted_sum+shared_gate+residual",
        )


@dataclass(frozen=True)
class Qwen35GGUFModel:
    """Qwen3.5 dense GGUF model plugin metadata."""

    name: str = "qwen3_5_gguf"
    architectures: tuple[str, ...] = ("qwen35",)
    default_quant: str = "gguf_q4_k_m"
    default_backend: str = "hip_gfx1100"
    weight_name_templates: tuple[str, ...] = (
        "token_embd.weight",
        "output_norm.weight",
        "blk.{layer}.attn_norm.weight",
        "blk.{layer}.post_attention_norm.weight",
        "blk.{layer}.attn_gate.weight",
        "blk.{layer}.attn_qkv.weight",
        "blk.{layer}.attn_q.weight",
        "blk.{layer}.attn_k.weight",
        "blk.{layer}.attn_v.weight",
        "blk.{layer}.attn_output.weight",
        "blk.{layer}.ffn_gate.weight",
        "blk.{layer}.ffn_up.weight",
        "blk.{layer}.ffn_down.weight",
    )


@dataclass(frozen=True)
class Qwen35MoeGGUFModel:
    """Qwen3.6/Qwen3.5 MoE GGUF model plugin metadata."""

    name: str = "qwen3_5_moe_gguf"
    architectures: tuple[str, ...] = ("qwen35moe",)
    default_quant: str = "gguf_q4_k_m"
    default_backend: str = "hip_gfx1100"
    weight_name_templates: tuple[str, ...] = (
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
        "blk.{layer}.attn_norm.weight",
        "blk.{layer}.post_attention_norm.weight",
        "blk.{layer}.attn_gate.weight",
        "blk.{layer}.attn_qkv.weight",
        "blk.{layer}.attn_q.weight",
        "blk.{layer}.attn_k.weight",
        "blk.{layer}.attn_v.weight",
        "blk.{layer}.attn_output.weight",
        "blk.{layer}.ffn_gate_inp.weight",
        "blk.{layer}.ffn_gate_inp_shexp.weight",
        "blk.{layer}.ffn_gate_exps.weight",
        "blk.{layer}.ffn_up_exps.weight",
        "blk.{layer}.ffn_down_exps.weight",
        "blk.{layer}.ffn_gate_shexp.weight",
        "blk.{layer}.ffn_up_shexp.weight",
        "blk.{layer}.ffn_down_shexp.weight",
    )


QWEN35_PARO_MOE = register_model(Qwen35ParoMoeModel())
QWEN35_GGUF = register_model(Qwen35GGUFModel())
QWEN35_MOE_GGUF = register_model(Qwen35MoeGGUFModel())
