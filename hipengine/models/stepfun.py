"""StepFun Step 3.5/3.7 model plugin metadata."""

from __future__ import annotations

from dataclasses import dataclass

from hipengine.models.registry import register_model


class StepFunUnsupportedCapabilityError(ValueError):
    pass


@dataclass(frozen=True)
class StepFunStep37GGUFModel:
    """Text-first StepFun Step 3.5/3.7 GGUF model plugin metadata.

    The plugin records architecture aliases, weight-name templates, and a
    representative decode sequence for registry/fusion planning. Vision,
    projector, MTP/speculative decode, and ModelOpt NVFP4 are explicit deferred
    capabilities until the text-only GGUF path is correct.
    """

    name: str = "stepfun_step3_7_gguf"
    architectures: tuple[str, ...] = (
        "step35",
        "step3p5",
        "step3p7",
        "Step3p5ForCausalLM",
        "Step3p7ForConditionalGeneration",
    )
    default_quant: str = "gguf_q3_k_l"
    default_backend: str = "hip_gfx1151"
    supported_capabilities: tuple[str, ...] = ("text_decode", "gguf_q3_k_l")
    deferred_capabilities: tuple[str, ...] = (
        "vision",
        "projector",
        "mtp",
        "speculative_decode",
        "modelopt_nvfp4",
    )
    weight_name_templates: tuple[str, ...] = (
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
        "rope_freqs.weight",
        "blk.{layer}.attn_gate.weight",
        "blk.{layer}.attn_k.weight",
        "blk.{layer}.attn_k_norm.weight",
        "blk.{layer}.attn_norm.weight",
        "blk.{layer}.attn_output.weight",
        "blk.{layer}.attn_q.weight",
        "blk.{layer}.attn_q_norm.weight",
        "blk.{layer}.attn_v.weight",
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
        """Return a representative text-only decode sequence."""

        return (
            "embed",
            *self.decode_layer_sequence(attention_kind="full_attention", mlp_kind="dense_mlp"),
            *self.decode_layer_sequence(attention_kind="sliding_attention", mlp_kind="moe"),
            "final_rmsnorm",
            "lm_head",
        )

    def decode_layer_sequence(self, *, attention_kind: str, mlp_kind: str) -> tuple[str, ...]:
        """Return primitive layer keys for one Step text decode layer."""

        if attention_kind == "full_attention":
            attention_layers = (
                "step_rmsnorm",
                "step_full_attention_qkv_proj",
                "step_rope_full_partial",
                "paged_kv_write",
                "full_attention_decode",
                "step_head_gate",
                "full_attention_o_proj",
            )
        elif attention_kind == "sliding_attention":
            attention_layers = (
                "step_rmsnorm",
                "step_sliding_attention_qkv_proj",
                "step_rope_sliding",
                "paged_kv_write_windowed",
                "sliding_attention_decode",
                "step_head_gate",
                "sliding_attention_o_proj",
            )
        else:
            raise ValueError("attention_kind must be 'full_attention' or 'sliding_attention'")

        if mlp_kind == "dense_mlp":
            mlp_layers = (
                "step_rmsnorm",
                "gguf_mixed_dense_swiglu",
                "residual_add",
            )
        elif mlp_kind == "moe":
            mlp_layers = (
                "step_rmsnorm",
                "step_router_topk8_sigmoid_bias",
                "step_selected_expert_swiglu",
                "step_shared_expert_swiglu",
                "step_weighted_sum_shared_residual",
            )
        else:
            raise ValueError("mlp_kind must be 'dense_mlp' or 'moe'")

        return (*attention_layers, *mlp_layers)

    def capability_status(self, capability: str) -> str:
        if capability in self.supported_capabilities:
            return "supported"
        if capability in self.deferred_capabilities:
            return "deferred"
        return "unknown"

    def require_capability(self, capability: str) -> None:
        status = self.capability_status(capability)
        if status == "supported":
            return
        if status == "deferred":
            raise StepFunUnsupportedCapabilityError(
                f"StepFun capability {capability!r} is deferred until text-only GGUF decode "
                "is correct"
            )
        raise StepFunUnsupportedCapabilityError(
            f"StepFun capability {capability!r} is not declared by this model plugin"
        )


STEPFUN_STEP37_GGUF = register_model(StepFunStep37GGUFModel())


__all__ = [
    "STEPFUN_STEP37_GGUF",
    "StepFunStep37GGUFModel",
    "StepFunUnsupportedCapabilityError",
]
