"""Gemma 4 model plugin metadata."""

from __future__ import annotations

from dataclasses import dataclass

from hipengine.models.registry import register_model

_FULL_ATTENTION = "full_attention"
_SLIDING_ATTENTION = "sliding_attention"


@dataclass(frozen=True)
class Gemma4GGUFModel:
    """Gemma 4 GGUF architecture metadata for registry and fusion planning.

    Runtime dimensions and the per-layer attention geometry mix are decoded by
    the Gemma 4 loader. This plugin intentionally contains no backend or
    tensor-layout branches.
    """

    name: str = "gemma4_gguf"
    architectures: tuple[str, ...] = ("gemma4", "Gemma4ForCausalLM", "Gemma4ForConditionalGeneration")
    default_quant: str = "gguf_q4_k_m"
    default_backend: str = "auto"
    weight_name_templates: tuple[str, ...] = (
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
        "rope_freqs.weight",
        "blk.{layer}.attn_norm.weight",
        "blk.{layer}.attn_q.weight",
        "blk.{layer}.attn_k.weight",
        "blk.{layer}.attn_v.weight",
        "blk.{layer}.attn_q_norm.weight",
        "blk.{layer}.attn_k_norm.weight",
        "blk.{layer}.attn_output.weight",
        "blk.{layer}.post_attention_norm.weight",
        "blk.{layer}.ffn_norm.weight",
        "blk.{layer}.ffn_gate.weight",
        "blk.{layer}.ffn_up.weight",
        "blk.{layer}.ffn_down.weight",
        "blk.{layer}.ffn_gate_inp.weight",
        "blk.{layer}.ffn_gate_inp.scale",
        "blk.{layer}.pre_ffw_norm_2.weight",
        "blk.{layer}.post_ffw_norm.weight",
        "blk.{layer}.post_ffw_norm_1.weight",
        "blk.{layer}.post_ffw_norm_2.weight",
        "blk.{layer}.ffn_gate_up_exps.weight",
        "blk.{layer}.ffn_down_exps.weight",
        "blk.{layer}.ffn_down_exps.scale",
        "blk.{layer}.layer_output_scale.weight",
    )

    def layer_sequence(self) -> tuple[str, ...]:
        """Return one representative sliding and one representative global layer."""

        return (
            "embed",
            *self.decode_layer_sequence(attention_kind=_SLIDING_ATTENTION),
            *self.decode_layer_sequence(attention_kind=_FULL_ATTENTION),
            "final_rmsnorm",
            "lm_head",
        )

    def decode_layer_sequence(self, *, attention_kind: str) -> tuple[str, ...]:
        """Return the unfused primitive plan for one Gemma 4 decoder layer.

        Every Gemma 4 decoder layer runs both feed-forward branches in parallel
        and sums them after two separate post-norms, so ``dense_mlp`` and
        ``selected_expert_mlp`` are siblings rather than alternatives.
        """

        if attention_kind == _FULL_ATTENTION:
            attention = (
                "rmsnorm",
                "full_attention_qk_proj",
                "gemma4_proportional_partial_rope",
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

        return (
            *attention,
            "attention_o_proj",
            "post_attention_rmsnorm",
            "residual_add",
            "rmsnorm",
            "dense_mlp",
            "gemma4_dense_branch_rmsnorm",
            "gemma4_weightless_router_topk",
            "rmsnorm",
            "selected_expert_mlp",
            "gemma4_expert_branch_rmsnorm",
            "gemma4_parallel_ffn_combine",
            "gemma4_layer_scalar",
            "residual_add",
        )


GEMMA4_GGUF = register_model(Gemma4GGUFModel())


__all__ = ["GEMMA4_GGUF", "Gemma4GGUFModel"]
