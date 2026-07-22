"""Laguna model plugin metadata."""

from __future__ import annotations

from dataclasses import dataclass

from hipengine.models.registry import register_model

_FULL_ATTENTION = "full_attention"
_SLIDING_ATTENTION = "sliding_attention"
_DENSE_MLP = "dense_mlp"
_SPARSE_MOE = "sparse_moe"


@dataclass(frozen=True)
class LagunaGGUFModel:
    """Laguna GGUF/HF architecture metadata for registry and fusion planning.

    Runtime dimensions and the exact per-layer mix are decoded by the Laguna
    loader. This plugin intentionally contains no backend or tensor-layout
    branches.
    """

    name: str = "laguna_gguf"
    architectures: tuple[str, ...] = ("laguna", "LagunaForCausalLM")
    default_quant: str = "gguf_q4_k_m"
    default_backend: str = "auto"
    weight_name_templates: tuple[str, ...] = (
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
        "blk.{layer}.attn_norm.weight",
        "blk.{layer}.attn_q.weight",
        "blk.{layer}.attn_k.weight",
        "blk.{layer}.attn_v.weight",
        "blk.{layer}.attn_gate.weight",
        "blk.{layer}.attn_q_norm.weight",
        "blk.{layer}.attn_k_norm.weight",
        "blk.{layer}.attn_output.weight",
        "blk.{layer}.ffn_norm.weight",
        "blk.{layer}.ffn_gate.weight",
        "blk.{layer}.ffn_up.weight",
        "blk.{layer}.ffn_down.weight",
        "blk.{layer}.ffn_gate_inp.weight",
        "blk.{layer}.exp_probs_b.bias",
        "blk.{layer}.ffn_gate_exps.weight",
        "blk.{layer}.ffn_up_exps.weight",
        "blk.{layer}.ffn_down_exps.weight",
        "blk.{layer}.ffn_gate_shexp.weight",
        "blk.{layer}.ffn_up_shexp.weight",
        "blk.{layer}.ffn_down_shexp.weight",
    )

    def layer_sequence(self) -> tuple[str, ...]:
        """Return representative dense-full and sparse-SWA Laguna layers."""

        return (
            "embed",
            *self.decode_layer_sequence(
                attention_kind=_FULL_ATTENTION,
                mlp_kind=_DENSE_MLP,
            ),
            *self.decode_layer_sequence(
                attention_kind=_SLIDING_ATTENTION,
                mlp_kind=_SPARSE_MOE,
            ),
            "final_rmsnorm",
            "lm_head",
        )

    def decode_layer_sequence(
        self,
        *,
        attention_kind: str,
        mlp_kind: str,
    ) -> tuple[str, ...]:
        """Return the unfused primitive plan for one Laguna decoder layer."""

        if attention_kind == _FULL_ATTENTION:
            attention = (
                "rmsnorm",
                "full_attention_qkv_proj",
                "yarn_rope",
                "paged_kv_write",
                "full_attention_decode",
            )
        elif attention_kind == _SLIDING_ATTENTION:
            attention = (
                "rmsnorm",
                "sliding_attention_qkv_proj",
                "rope",
                "paged_kv_write",
                "sliding_attention_decode",
            )
        else:
            raise ValueError(
                "attention_kind must be 'full_attention' or 'sliding_attention'"
            )

        if mlp_kind == _DENSE_MLP:
            mlp = ("dense_mlp",)
        elif mlp_kind == _SPARSE_MOE:
            mlp = (
                "laguna_sigmoid_router_topk",
                "selected_expert_mlp",
                "laguna_shared_expert",
                "laguna_routed_shared_combine",
            )
        else:
            raise ValueError("mlp_kind must be 'dense_mlp' or 'sparse_moe'")

        return (
            *attention,
            "softplus_head_gate",
            "attention_o_proj",
            "add_rmsnorm",
            *mlp,
            "residual_add",
        )


LAGUNA_GGUF = register_model(LagunaGGUFModel())


__all__ = ["LAGUNA_GGUF", "LagunaGGUFModel"]
