"""Qwen3.5/PARO model plugin metadata."""

from __future__ import annotations

from dataclasses import dataclass

from hipengine.models.kv_capabilities import (
    KVCapabilityEvidence,
    KVCapabilityKey,
    KVCapabilityResolution,
    ModelArtifactIdentity,
    resolve_kv_capability,
)
from hipengine.models.registry import register_model
from hipengine.speculative.serving import (
    SpeculativeMTPServingDecision,
    SpeculativeMTPServingEvidence,
    SpeculativeMTPServingKey,
    resolve_speculative_mtp_serving_plan,
)


_QWEN36_MOE_Q4KM_MTP_SERVING_EVIDENCE = (
    SpeculativeMTPServingEvidence(
        evidence_key="qwen36-moe-q4km-gfx1100-production-bf16-c1-k2-d24",
        artifact_sha256=(
            "0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b"
        ),
        artifact_size_bytes=22_663_387_424,
        backend="hip_gfx1100",
        target_arch="gfx1100",
        weight_quant="gguf_q4_k_m",
        execution_profile="production",
        execution_profile_manifest_sha256=(
            "2b64229e062c85d08244149191f515c226c6897ecd753d86849dad9fe7c92ca9"
        ),
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=1,
        resident_capacity=1,
        candidate_budget=2,
        sampling_modes=("greedy_fast",),
        max_sequence_length=1024,
        min_context_tokens=4,
        max_context_tokens=95,
        min_output_horizon_tokens=24,
        max_output_horizon_tokens=24,
        reason="qualified_automatic_moe_c1_k2_d24",
        evidence_artifacts=(
            "benchmarks/results/2026-08-27-w7900-35b-moe-mtp2-production-quality.json",
            "benchmarks/results/2026-08-27-w7900-35b-moe-mtp2-production-performance.json",
            "benchmarks/results/2026-08-27-w7900-35b-moe-mtp2-production-serving.json",
        ),
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=True,
    ),
    SpeculativeMTPServingEvidence(
        evidence_key="qwen36-moe-q4km-gfx1100-production-bf16-c2-k2-d24",
        artifact_sha256=(
            "0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b"
        ),
        artifact_size_bytes=22_663_387_424,
        backend="hip_gfx1100",
        target_arch="gfx1100",
        weight_quant="gguf_q4_k_m",
        execution_profile="production",
        execution_profile_manifest_sha256=(
            "2b64229e062c85d08244149191f515c226c6897ecd753d86849dad9fe7c92ca9"
        ),
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=2,
        resident_capacity=2,
        candidate_budget=2,
        sampling_modes=("greedy_fast",),
        max_sequence_length=1024,
        min_context_tokens=4,
        max_context_tokens=95,
        min_output_horizon_tokens=24,
        max_output_horizon_tokens=24,
        reason="qualified_automatic_production_moe_c2_k2_d24",
        evidence_artifacts=(
            "benchmarks/results/2026-08-28-w7900-35b-moe-mtp2-c2-automatic-promotion.json",
        ),
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=True,
    ),
)


_QWEN36_DENSE_Q4KM_MTP_SERVING_EVIDENCE = (
    SpeculativeMTPServingEvidence(
        evidence_key="qwen36-dense-q4km-gfx1100-production-bf16-c1-k3-d24",
        artifact_sha256=(
            "a7cbd3ecc0e3f9b333edee61ae66bc87ed713c5d49587a8355814722ed329e0f"
        ),
        artifact_size_bytes=17_106_773_120,
        backend="hip_gfx1100",
        target_arch="gfx1100",
        weight_quant="gguf_q4_k_m",
        execution_profile="production",
        execution_profile_manifest_sha256=(
            "2adc137a32d65bc63619947577f5233548d5835a474713abe270d666122a1960"
        ),
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=1,
        resident_capacity=1,
        candidate_budget=3,
        sampling_modes=("greedy_fast",),
        max_sequence_length=1024,
        min_context_tokens=4,
        max_context_tokens=95,
        min_output_horizon_tokens=24,
        max_output_horizon_tokens=24,
        reason="qualified_automatic_dense_c1_k3_d24",
        evidence_artifacts=(
            "benchmarks/results/2026-08-27-w7900-27b-dense-mtp2-production-quality.json",
            "benchmarks/results/2026-08-27-w7900-27b-dense-mtp2-production-performance.json",
            "benchmarks/results/2026-08-27-w7900-27b-dense-mtp2-production-serving.json",
            "benchmarks/results/2026-09-05-w7900-q4km-k3-c8-automatic-promotion.json",
        ),
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=True,
    ),
    SpeculativeMTPServingEvidence(
        evidence_key="qwen36-dense-q4km-gfx1100-strict-bf16-c2-k2-d24",
        artifact_sha256=(
            "a7cbd3ecc0e3f9b333edee61ae66bc87ed713c5d49587a8355814722ed329e0f"
        ),
        artifact_size_bytes=17_106_773_120,
        backend="hip_gfx1100",
        target_arch="gfx1100",
        weight_quant="gguf_q4_k_m",
        execution_profile="strict",
        execution_profile_manifest_sha256=(
            "52a3d5b8b02c4dc8230c8c9dc8e43b01135db7ae1b44b027fc8915d66bedcdbb"
        ),
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=2,
        resident_capacity=2,
        candidate_budget=2,
        sampling_modes=("greedy_fast",),
        max_sequence_length=1024,
        min_context_tokens=4,
        max_context_tokens=95,
        min_output_horizon_tokens=24,
        max_output_horizon_tokens=24,
        reason="qualified_explicit_dense_c2_k2_d24",
        evidence_artifacts=(
            "benchmarks/results/2026-08-27-w7900-27b-dense-mtp2-c2-explicit-ownership.json",
            "benchmarks/results/2026-09-05-w7900-q4km-k3-c8-automatic-promotion.json",
        ),
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=False,
    ),
    SpeculativeMTPServingEvidence(
        evidence_key="qwen36-dense-q4km-gfx1100-production-bf16-c2-k2-d24",
        artifact_sha256=(
            "a7cbd3ecc0e3f9b333edee61ae66bc87ed713c5d49587a8355814722ed329e0f"
        ),
        artifact_size_bytes=17_106_773_120,
        backend="hip_gfx1100",
        target_arch="gfx1100",
        weight_quant="gguf_q4_k_m",
        execution_profile="production",
        execution_profile_manifest_sha256=(
            "2adc137a32d65bc63619947577f5233548d5835a474713abe270d666122a1960"
        ),
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=2,
        resident_capacity=2,
        candidate_budget=2,
        sampling_modes=("greedy_fast",),
        max_sequence_length=1024,
        min_context_tokens=4,
        max_context_tokens=95,
        min_output_horizon_tokens=24,
        max_output_horizon_tokens=24,
        reason="qualified_automatic_production_dense_c2_k2_d24",
        evidence_artifacts=(
            "benchmarks/results/2026-08-27-w7900-27b-dense-mtp2-c2-production-quality.json",
            "benchmarks/results/2026-08-27-w7900-27b-dense-mtp2-c2-explicit-ownership.json",
            "benchmarks/results/2026-08-27-w7900-27b-dense-mtp2-c2-automatic-promotion.json",
            "benchmarks/results/2026-09-05-w7900-q4km-k3-c8-automatic-promotion.json",
        ),
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=True,
    ),
)


_QWEN38_Q4KM_MTP_SERVING_EVIDENCE = (
    SpeculativeMTPServingEvidence(
        evidence_key="qwen38-q4km-gfx1151-strict-bf16-c1-b3-natural25-s0",
        artifact_sha256=(
            "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
        ),
        artifact_size_bytes=17_106_775_008,
        backend="hip_gfx1151",
        target_arch="gfx1151",
        weight_quant="gguf_q4_k_m",
        execution_profile="strict",
        execution_profile_manifest_sha256=(
            "393155123c5e09700ff017f949f338fb5f519579e2f05bea3ffef7a43a09a71b"
        ),
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=1,
        resident_capacity=1,
        candidate_budget=3,
        sampling_modes=("greedy_fast",),
        max_sequence_length=1024,
        min_context_tokens=1,
        max_context_tokens=67,
        min_output_horizon_tokens=25,
        max_output_horizon_tokens=25,
        reason="qualified_automatic_c1_b3",
        evidence_artifacts=(
            "benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s0.json",
            "benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s0-openai.json",
            "benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s1.json",
            "benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s2.json",
            "benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s3.json",
            "benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e0-current-baseline.json",
        ),
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=True,
    ),
    SpeculativeMTPServingEvidence(
        evidence_key="qwen38-q4km-gfx1151-strict-bf16-cap4-realized-c1-b3",
        artifact_sha256=(
            "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
        ),
        artifact_size_bytes=17_106_775_008,
        backend="hip_gfx1151",
        target_arch="gfx1151",
        weight_quant="gguf_q4_k_m",
        execution_profile="strict",
        execution_profile_manifest_sha256=(
            "393155123c5e09700ff017f949f338fb5f519579e2f05bea3ffef7a43a09a71b"
        ),
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=1,
        resident_capacity=4,
        candidate_budget=3,
        sampling_modes=("greedy_fast",),
        max_sequence_length=1024,
        min_context_tokens=1,
        max_context_tokens=67,
        min_output_horizon_tokens=25,
        max_output_horizon_tokens=25,
        reason="qualified_automatic_realized_singleton_c1_b3",
        evidence_artifacts=(
            "benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s4-auto.json",
            "benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s5-closure.json",
            "benchmarks/results/2026-08-27-gfx1151-qwen38-realized-singleton-auto.json",
            "benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e0-current-baseline.json",
        ),
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=True,
    ),
    SpeculativeMTPServingEvidence(
        evidence_key="qwen38-q4km-gfx1151-production-bf16-c1-b3-context128",
        artifact_sha256=(
            "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
        ),
        artifact_size_bytes=17_106_775_008,
        backend="hip_gfx1151",
        target_arch="gfx1151",
        weight_quant="gguf_q4_k_m",
        execution_profile="production",
        execution_profile_manifest_sha256=(
            "534a8bac3ca74428e3c1a60e9c3cbd91254f8963ddfcd678949052783331c565"
        ),
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=1,
        resident_capacity=1,
        candidate_budget=3,
        sampling_modes=("greedy_fast",),
        max_sequence_length=1024,
        min_context_tokens=68,
        max_context_tokens=128,
        min_output_horizon_tokens=24,
        max_output_horizon_tokens=24,
        reason="qualified_explicit_production_c1_b3_context128",
        evidence_artifacts=(
            "benchmarks/results/2026-08-27-gfx1151-qwen38-concurrency2-t04-production-suite.json",
            "benchmarks/results/2026-08-27-gfx1151-qwen38-concurrency2-t11-t13-ownership.json",
            "benchmarks/results/2026-08-27-gfx1151-qwen38-postcampaign-mtp-c1-c8.json",
            "benchmarks/results/2026-08-27-gfx1151-qwen38-c68-c128-production-explicit.json",
        ),
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=False,
    ),
    SpeculativeMTPServingEvidence(
        evidence_key="qwen38-q4km-gfx1151-production-bf16-cap4-c1-intent-k3-d24",
        artifact_sha256=(
            "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
        ),
        artifact_size_bytes=17_106_775_008,
        backend="hip_gfx1151",
        target_arch="gfx1151",
        weight_quant="gguf_q4_k_m",
        execution_profile="production",
        execution_profile_manifest_sha256=(
            "af20ee3b22921dc9a0c988dd1c3f5c471932f0ecda4e557ec2ba4bbc8ef5d95f"
        ),
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=1,
        resident_capacity=4,
        candidate_budget=3,
        sampling_modes=("greedy_fast",),
        max_sequence_length=1024,
        min_context_tokens=1,
        max_context_tokens=128,
        min_output_horizon_tokens=24,
        max_output_horizon_tokens=24,
        reason="diagnostic_production_cap4_c1_or_c2_after_ar_rebase",
        evidence_artifacts=(
            "benchmarks/results/2026-08-28-gfx1151-qwen38-c2-production-q4-rowtile-retained.json",
            "benchmarks/results/2026-08-27-gfx1151-qwen38-realized-singleton-auto.json",
            "benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e0-current-baseline.json",
        ),
        max_realized_group_rows=2,
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=False,
    ),
    SpeculativeMTPServingEvidence(
        evidence_key="qwen38-q4km-gfx1151-production-bf16-c2-k3-d24",
        artifact_sha256=(
            "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
        ),
        artifact_size_bytes=17_106_775_008,
        backend="hip_gfx1151",
        target_arch="gfx1151",
        weight_quant="gguf_q4_k_m",
        execution_profile="production",
        execution_profile_manifest_sha256=(
            "af20ee3b22921dc9a0c988dd1c3f5c471932f0ecda4e557ec2ba4bbc8ef5d95f"
        ),
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=2,
        resident_capacity=4,
        candidate_budget=3,
        sampling_modes=("greedy_fast",),
        max_sequence_length=1024,
        min_context_tokens=1,
        max_context_tokens=128,
        min_output_horizon_tokens=24,
        max_output_horizon_tokens=24,
        reason="diagnostic_production_c2_after_ar_rebase",
        evidence_artifacts=(
            "benchmarks/results/2026-08-28-gfx1151-qwen38-c2-production-q4-rowtile-retained.json",
            "benchmarks/results/2026-08-27-gfx1151-qwen38-dynamic-admission-d3-lifecycle.json",
            "benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e0-current-baseline.json",
        ),
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=False,
    ),
    SpeculativeMTPServingEvidence(
        evidence_key="qwen38-q4km-gfx1151-production-bf16-c8-k3-d24",
        artifact_sha256=(
            "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
        ),
        artifact_size_bytes=17_106_775_008,
        backend="hip_gfx1151",
        target_arch="gfx1151",
        weight_quant="gguf_q4_k_m",
        execution_profile="production",
        execution_profile_manifest_sha256=(
            "af20ee3b22921dc9a0c988dd1c3f5c471932f0ecda4e557ec2ba4bbc8ef5d95f"
        ),
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=8,
        resident_capacity=8,
        candidate_budget=3,
        sampling_modes=("greedy_fast",),
        max_sequence_length=1024,
        min_context_tokens=1,
        max_context_tokens=128,
        min_output_horizon_tokens=24,
        max_output_horizon_tokens=24,
        reason="qualified_explicit_production_c8_k3_after_q6_lm_head_rebase",
        evidence_artifacts=(
            "benchmarks/results/2026-09-05-gfx1151-qwen38-c8-k3-width-policy-retained.json",
            "benchmarks/results/2026-09-03-gfx1151-qwen38-b5-planar-q6-integer-mmq-retained.json",
            "benchmarks/results/2026-09-05-gfx1151-qwen38-q6-lm-head-row8-retained.json",
        ),
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=False,
    ),
    SpeculativeMTPServingEvidence(
        evidence_key="qwen38-q4km-gfx1100-production-bf16-c2-k2-d24",
        artifact_sha256=(
            "7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b"
        ),
        artifact_size_bytes=17_106_773_984,
        backend="hip_gfx1100",
        target_arch="gfx1100",
        weight_quant="gguf_q4_k_m",
        execution_profile="production",
        execution_profile_manifest_sha256=(
            "2adc137a32d65bc63619947577f5233548d5835a474713abe270d666122a1960"
        ),
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=2,
        resident_capacity=2,
        candidate_budget=2,
        sampling_modes=("greedy_fast",),
        max_sequence_length=1024,
        min_context_tokens=4,
        max_context_tokens=95,
        min_output_horizon_tokens=24,
        max_output_horizon_tokens=24,
        reason="qualified_automatic_gfx1100_production_c2_k2_d24",
        evidence_artifacts=(
            "benchmarks/results/2026-08-29-w7900-qwen38-q4km-p8-c2-correctness-closure.json",
            "benchmarks/results/2026-08-30-w7900-qwen38-q4km-p11-integrated-explicit-c2.json",
            "benchmarks/results/2026-08-30-w7900-qwen38-q4km-p12-c2-automatic-promotion.json",
            "benchmarks/results/2026-09-05-w7900-q4km-k3-c8-automatic-promotion.json",
        ),
        max_realized_group_rows=2,
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=True,
    ),
    SpeculativeMTPServingEvidence(
        evidence_key="qwen38-q4km-gfx1100-production-bf16-c8-k3-d24",
        artifact_sha256=(
            "7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b"
        ),
        artifact_size_bytes=17_106_773_984,
        backend="hip_gfx1100",
        target_arch="gfx1100",
        weight_quant="gguf_q4_k_m",
        execution_profile="production",
        execution_profile_manifest_sha256=(
            "2adc137a32d65bc63619947577f5233548d5835a474713abe270d666122a1960"
        ),
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=8,
        resident_capacity=8,
        candidate_budget=4,
        sampling_modes=("greedy_fast",),
        max_sequence_length=1024,
        min_context_tokens=4,
        max_context_tokens=95,
        min_output_horizon_tokens=24,
        max_output_horizon_tokens=24,
        reason="qualified_automatic_gfx1100_production_c8_k3_d24",
        evidence_artifacts=(
            "benchmarks/results/2026-09-04-w7900-q4km-k3-c8-p4-q6-dp4a-l4-numerics.json",
            "benchmarks/results/2026-09-04-w7900-q4km-k3-c8-p4-q6-dp4a-retention-e2e.json",
            "benchmarks/results/2026-09-05-w7900-q4km-k3-c8-automatic-promotion.json",
        ),
        max_realized_group_rows=8,
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=True,
    ),
)


_QWEN38_GGUF_KV_CAPABILITY_EVIDENCE = (
    KVCapabilityEvidence(
        key=KVCapabilityKey(
            artifact_sha256="7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b",
            artifact_size_bytes=17_106_773_984,
            backend="hip_gfx1100",
            target_arch="gfx1100",
            weight_quant="gguf_q4_k_m",
            kv_storage="int8_per_token_head",
            storage_layout="uniform",
            scale_dtype="fp32",
            scale_granularity="per_token_head",
        ),
        decision="qualified",
        scope="explicit_no_mirror_c1_direct_c4_serial",
        quality_artifact=(
            "benchmarks/results/"
            "2026-08-16-qwen38-27b-actual-context-quality-w7900.json"
        ),
        reason=(
            "complete 512/8 and 4K/16 plus bounded 129024/16 quality pass on "
            "gfx1100; direct compact execution is qualified at physical c1, with "
            "artifact-scoped serial c1-per-row residency through logical c4"
        ),
        max_direct_rows=1,
        max_serial_resident_rows=4,
        persistent_bf16_mirror=False,
        decode_batch_variant=(
            "per_token_head_gqa_splitk_gate_bf16_batch_strided_spans"
        ),
    ),
    KVCapabilityEvidence(
        key=KVCapabilityKey(
            artifact_sha256="7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169",
            artifact_size_bytes=17_106_775_008,
            backend="hip_gfx1151",
            target_arch="gfx1151",
            weight_quant="gguf_q4_k_m",
            kv_storage="int8_per_token_head",
            storage_layout="uniform",
            scale_dtype="fp32",
            scale_granularity="per_token_head",
        ),
        decision="rejected",
        scope="native_no_mirror_quality",
        quality_artifact=(
            "benchmarks/results/"
            "2026-08-15-gfx1151-qwen38-27b-int8-kv-quality-rejected.json"
        ),
        reason=(
            "complete 1K/8 transfer rejected: minimum-prompt top-1 agreement "
            "0.7778 is below the 0.90 gate"
        ),
        max_direct_rows=0,
        max_serial_resident_rows=0,
        persistent_bf16_mirror=False,
    ),
)


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
    default_backend: str = "auto"
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
    kv_capability_evidence: tuple[KVCapabilityEvidence, ...] = _QWEN38_GGUF_KV_CAPABILITY_EVIDENCE
    speculative_mtp_serving_evidence: tuple[SpeculativeMTPServingEvidence, ...] = (
        _QWEN38_Q4KM_MTP_SERVING_EVIDENCE
        + _QWEN36_DENSE_Q4KM_MTP_SERVING_EVIDENCE
    )
    speculative_mtp2_adapter: str = "dense_nextn"

    def resolve_speculative_mtp_serving_plan(
        self,
        *,
        key: SpeculativeMTPServingKey,
    ) -> SpeculativeMTPServingDecision:
        """Resolve the exact Qwen dense serving scope before mutation."""

        return resolve_speculative_mtp_serving_plan(
            self.speculative_mtp_serving_evidence,
            key=key,
        )

    def resolve_kv_capability(
        self,
        *,
        key: KVCapabilityKey,
        artifact: ModelArtifactIdentity,
    ) -> KVCapabilityResolution:
        """Resolve artifact/backend-specific KV evidence for this plugin."""

        return resolve_kv_capability(
            self.kv_capability_evidence,
            key=key,
            artifact=artifact,
        )


@dataclass(frozen=True)
class Qwen35MoeGGUFModel:
    """Qwen3.6/Qwen3.5 MoE GGUF model plugin metadata."""

    name: str = "qwen3_5_moe_gguf"
    architectures: tuple[str, ...] = ("qwen35moe",)
    default_quant: str = "gguf_q4_k_m"
    default_backend: str = "auto"
    speculative_mtp2_adapter: str = "moe_nextn"
    speculative_mtp_serving_evidence: tuple[SpeculativeMTPServingEvidence, ...] = (
        _QWEN36_MOE_Q4KM_MTP_SERVING_EVIDENCE
    )
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

    def resolve_speculative_mtp_serving_plan(
        self,
        *,
        key: SpeculativeMTPServingKey,
    ) -> SpeculativeMTPServingDecision:
        """Resolve the exact Qwen MoE serving scope before mutation."""

        return resolve_speculative_mtp_serving_plan(
            self.speculative_mtp_serving_evidence,
            key=key,
        )


QWEN35_PARO_MOE = register_model(Qwen35ParoMoeModel())
QWEN35_GGUF = register_model(Qwen35GGUFModel())
QWEN35_MOE_GGUF = register_model(Qwen35MoeGGUFModel())
