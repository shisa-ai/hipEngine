"""Qwen3.8-Flash-Next qwen4exp GGUF model plugin metadata."""

from __future__ import annotations

from dataclasses import dataclass

from hipengine.models.registry import register_model


@dataclass(frozen=True)
class Qwen4ExpGGUFModel:
    """Frozen architecture contract for the Qwen3.8-Flash-Next text target."""

    name: str = "qwen4_exp_gguf"
    architectures: tuple[str, ...] = ("qwen4exp",)
    default_quant: str = "gguf_q4_k_m"
    default_backend: str = "auto"
    block_count: int = 48
    hidden_size: int = 2560
    residual_branch_count: int = 4
    residual_width: int = 10240
    native_context_length: int = 262144
    qsa_dense_equivalent_max_tokens: int = 2051
    ple_device_resident: bool = False
    vision_supported: bool = False
    mtp_supported: bool = False
    weight_name_templates: tuple[str, ...] = (
        "token_embd.weight",
        "output.weight",
        "output_hc_norm.weight",
        "output_hc_down.weight",
        "output_hc_up.weight",
        "per_layer_token_embd.weight",
        "blk.{layer}.hc_attn_norm.weight",
        "blk.{layer}.hc_attn_down.weight",
        "blk.{layer}.hc_attn_up.weight",
        "blk.{layer}.hc_attn_inject.weight",
        "blk.{layer}.hc_ffn_norm.weight",
        "blk.{layer}.hc_ffn_down.weight",
        "blk.{layer}.hc_ffn_up.weight",
        "blk.{layer}.hc_ffn_inject.weight",
        "blk.{layer}.attn_q.weight",
        "blk.{layer}.attn_q_norm.weight",
        "blk.{layer}.attn_k.weight",
        "blk.{layer}.attn_k_norm.weight",
        "blk.{layer}.attn_v.weight",
        "blk.{layer}.attn_output.weight",
        "blk.{layer}.indexer.q_proj.weight",
        "blk.{layer}.indexer.k_proj.weight",
        "blk.{layer}.indexer.q_norm.weight",
        "blk.{layer}.indexer.k_norm.weight",
        "blk.{layer}.attn_qkv.weight",
        "blk.{layer}.attn_gate.weight",
        "blk.{layer}.ssm_a",
        "blk.{layer}.ssm_conv1d.weight",
        "blk.{layer}.ssm_dt.bias",
        "blk.{layer}.ssm_norm.weight",
        "blk.{layer}.ssm_beta.weight",
        "blk.{layer}.ssm_alpha.weight",
        "blk.{layer}.ssm_out.weight",
        "blk.{layer}.ffn_gate_inp.weight",
        "blk.{layer}.ffn_gate_inp_shexp.weight",
        "blk.{layer}.ffn_gate_exps.weight",
        "blk.{layer}.ffn_up_exps.weight",
        "blk.{layer}.ffn_down_exps.weight",
        "blk.{layer}.ffn_gate_shexp.weight",
        "blk.{layer}.ffn_up_shexp.weight",
        "blk.{layer}.ffn_down_shexp.weight",
        "blk.{layer}.ple_key.weight",
        "blk.{layer}.ple_value.weight",
        "blk.{layer}.ple_norm_key.weight",
        "blk.{layer}.ple_norm_query.weight",
        "blk.{layer}.ple_norm_conv.weight",
        "blk.{layer}.ple_conv1d.weight",
    )

    def layer_sequence(self) -> tuple[str, ...]:
        """Return representative GDN/PLE and QSA paths for fusion planning."""

        return (
            "embed",
            "widen_residual_4",
            *self.decode_layer_sequence(attention_kind="gdn", include_ple=True),
            *self.decode_layer_sequence(attention_kind="qsa", include_ple=False),
            "gr_head_read",
            "lm_head",
        )

    def attention_kind_for_layer(self, layer_id: int) -> str:
        """Return the frozen three-GDN/one-QSA layer pattern."""

        self._validate_layer_id(layer_id)
        return "qsa" if layer_id % 4 == 3 else "gdn"

    def has_ple(self, layer_id: int) -> bool:
        """Return whether the zero-based decoder layer owns PLE injection."""

        self._validate_layer_id(layer_id)
        return layer_id == 1

    def decode_layer_sequence(
        self,
        *,
        attention_kind: str,
        include_ple: bool,
    ) -> tuple[str, ...]:
        """Return one Qwen4Exp decode layer's primitive ownership sequence."""

        ple = ("ple_sparse_gather", "ple_inject") if include_ple else ()
        if attention_kind == "gdn":
            mixer = (
                "gdn_qkvz_proj",
                "gdn_conv_decode",
                "gdn_recurrence_fp32",
                "gdn_sigmoid_gate_norm",
                "gdn_o_proj",
            )
        elif attention_kind == "qsa":
            mixer = (
                "qsa_qkv_proj",
                "qsa_index_write",
                "qsa_index_select",
                "kv_live_spans",
                "paged_kv_write",
                "sparse_paged_attention_decode",
                "qsa_o_proj",
            )
        else:
            raise ValueError("attention_kind must be 'gdn' or 'qsa'")
        return (
            *ple,
            "gr_attn_read",
            *mixer,
            "gr_attn_write",
            "gr_ffn_read",
            "router_top10_shared",
            "selected_expert_swiglu",
            "moe_top10_shared",
            "gr_ffn_write",
            "residual_state_commit",
        )

    def _validate_layer_id(self, layer_id: int) -> None:
        if not 0 <= layer_id < self.block_count:
            raise ValueError(f"layer_id must be in [0, {self.block_count}), got {layer_id}")


QWEN4_EXP_GGUF = register_model(Qwen4ExpGGUFModel())


__all__ = ["QWEN4_EXP_GGUF", "Qwen4ExpGGUFModel"]
