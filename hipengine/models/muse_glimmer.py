"""Muse Glimmer model plugin metadata."""

from __future__ import annotations

from dataclasses import dataclass

from hipengine.models.registry import register_model

_SWA_ATTENTION = "sliding_attention"
_FULL_ATTENTION = "full_attention"


@dataclass(frozen=True)
class MuseGlimmerGGUFModel:
    """Muse Glimmer GGUF architecture metadata for registry and fusion planning.

    Runtime dimensions and the per-layer SWA/NoPE mix are decoded from the
    artifact by the loader. This plugin contains no backend or tensor-layout
    branches.
    """

    name: str = "muse_glimmer_gguf"
    architectures: tuple[str, ...] = (
        "muse-glimmer",
        "MuseGlimmerForConditionalGeneration",
    )
    default_quant: str = "gguf_q4_k_m"
    default_backend: str = "auto"
    speculative_mtp_provider: str = "dflash"

    @staticmethod
    def resolve_speculative_mtp_draft_model(model_path: str):
        """Discover the DFlash sidecar shipped beside the target artifact.

        Capability-based automatic selection: a sibling ``dflash-*.gguf`` is
        admitted by the provider's structural checks (architecture, hidden
        size, head_dim, target_layers), never by name/hash allowlists. Absent
        sidecar -> plain AR; ``speculative_mtp_serving='off'`` opts out.
        """

        from pathlib import Path

        directory = Path(model_path).expanduser().resolve().parent
        heads = tuple(sorted(directory.glob("dflash-*.gguf")))
        return heads[0] if len(heads) == 1 else None
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
        "blk.{layer}.post_attention_norm.weight",
        "blk.{layer}.ffn_norm.weight",
        "blk.{layer}.ffn_gate.weight",
        "blk.{layer}.ffn_up.weight",
        "blk.{layer}.ffn_down.weight",
        "blk.{layer}.post_ffw_norm.weight",
    )

    def layer_sequence(self) -> tuple[str, ...]:
        """Representative SWA and NoPE-full Glimmer layers, unfused."""

        return (
            "embed",
            "weightless_rmsnorm",
            *self.decode_layer_sequence(attention_kind=_SWA_ATTENTION),
            *self.decode_layer_sequence(attention_kind=_FULL_ATTENTION),
            "final_rmsnorm",
            "lm_head",
            "logit_scale_softcap",
        )

    def decode_layer_sequence(self, *, attention_kind: str) -> tuple[str, ...]:
        """Return the unfused primitive plan for one Glimmer decoder layer."""

        if attention_kind == _SWA_ATTENTION:
            rotary = ("head_rmsnorm", "rope")
        elif attention_kind == _FULL_ATTENTION:
            # Full layers are NoPE: the head norm runs, the rotation does not.
            rotary = ("head_rmsnorm",)
        else:
            raise ValueError(
                "attention_kind must be 'sliding_attention' or 'full_attention'"
            )
        return (
            "rmsnorm",
            "qkv_proj",
            *rotary,
            "kv_write",
            "gqa_attention",
            "sigmoid_gate_mul",
            "attention_o_proj",
            "post_rmsnorm",
            "residual_add",
            "rmsnorm",
            "swiglu_mlp",
            "post_rmsnorm",
            "residual_add",
        )


MUSE_GLIMMER_GGUF = register_model(MuseGlimmerGGUFModel())


__all__ = ["MUSE_GLIMMER_GGUF", "MuseGlimmerGGUFModel"]