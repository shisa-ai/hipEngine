"""Gemma 4 model plugin metadata."""

from __future__ import annotations

from dataclasses import dataclass

from hipengine.models.registry import register_model
from hipengine.speculative.serving import (
    SpeculativeMTPServingDecision,
    SpeculativeMTPServingImplementation,
    SpeculativeMTPServingKey,
    resolve_speculative_mtp_serving_plan,
    unsupported_contract,
)

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
    speculative_mtp_provider: str = "gemma4_mtp"

    @staticmethod
    def resolve_speculative_mtp_draft_model(model_path: str):
        """Find the external assistant head shipped with this Gemma artifact."""

        from pathlib import Path

        mtp_dir = Path(model_path).expanduser().resolve().parent / "MTP"
        heads = tuple(sorted(mtp_dir.glob("mtp-*.gguf")))
        return heads[0] if len(heads) == 1 else None

    # backbone's BF16 KV, drafts with its own 4-block head, and verifies the
    # whole draft in one backbone forward. It has no retained evidence rows, so
    # this declaration is the whole admission scope: an artifact whose kernels
    # can execute the route is admitted by capability, and only a genuine
    # capability miss (storage, backend, width, depth, or sampling mode) refuses.
    # There is no artifact hash, model name, or known-good list here. Depth is
    # the provider's structural maximum (``_MAX_CANDIDATES`` in
    # ``hipengine/generation/gemma4_mtp.py``) and the width is one row, because
    # the provider serves a single request at a time.
    speculative_mtp_serving_implementations: tuple[SpeculativeMTPServingImplementation, ...] = (
        SpeculativeMTPServingImplementation(
            name="gguf_gemma4_assistant_bf16_c1_chain",
            kv_storage="bf16",
            backends=(("hip_gfx1100", "gfx1100"), ("hip_gfx1151", "gfx1151")),
            max_candidate_count=8,
            max_group_rows=1,
            group_rejection_reason="gemma4_assistant_group_above_offered_width",
            sampling_modes=("greedy_fast", "sampled"),
            automatic_eligible=True,
        ),
    )
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

    def _speculative_mtp_serving_implementation(
        self,
        key: SpeculativeMTPServingKey,
    ) -> SpeculativeMTPServingImplementation | None:
        """The declaration covering this key's KV storage contract.

        Storage is matched first because that is the contract a declaration
        describes. A storage match on an unsupported backend or width is still
        returned so the refusal names the real axis rather than a storage
        mismatch.
        """

        for implementation in self.speculative_mtp_serving_implementations:
            if implementation.kv_storage == key.kv_storage:
                return implementation
        return None

    def resolve_speculative_mtp_serving_plan(
        self,
        *,
        key: SpeculativeMTPServingKey,
        request_mode: str = "automatic",
    ) -> SpeculativeMTPServingDecision:
        """Resolve the assistant-head serving scope before mutation.

        The assistant head has no retained evidence rows, so the implementation
        declaration is the admission scope in every request mode: a missing
        measurement never withholds a path the kernels support, and only a
        capability gap keeps it off. This is the same rule the Qwen plugins
        apply when no row covers the cell.
        """

        evidence_decision = resolve_speculative_mtp_serving_plan(
            (),
            key=key,
            request_mode=request_mode,
        )
        implementation = self._speculative_mtp_serving_implementation(key)
        if implementation is not None:
            return implementation.resolve(key, request_mode=request_mode)
        return unsupported_contract(evidence_decision)


GEMMA4_GGUF = register_model(Gemma4GGUFModel())


__all__ = ["GEMMA4_GGUF", "Gemma4GGUFModel"]
