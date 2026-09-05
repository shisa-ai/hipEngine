from __future__ import annotations

from dataclasses import fields, replace
from types import SimpleNamespace

import pytest

from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator
from hipengine.llm import LLM
from hipengine.models.qwen35 import Qwen35GGUFModel, Qwen35MoeGGUFModel
from hipengine.speculative.serving import (
    SpeculativeMTPServingEvidence,
    SpeculativeMTPServingKey,
    SpeculativeMTPStaticState,
    resolve_speculative_mtp_serving_plan,
)


_MODEL_SHA256 = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
_STRICT_MANIFEST_SHA256 = "393155123c5e09700ff017f949f338fb5f519579e2f05bea3ffef7a43a09a71b"
_PRODUCTION_MANIFEST_SHA256 = "af20ee3b22921dc9a0c988dd1c3f5c471932f0ecda4e557ec2ba4bbc8ef5d95f"
_PRODUCTION_EXPLICIT_MANIFEST_SHA256 = (
    "534a8bac3ca74428e3c1a60e9c3cbd91254f8963ddfcd678949052783331c565"
)


def _key(**changes) -> SpeculativeMTPServingKey:
    key = SpeculativeMTPServingKey(
        artifact_sha256=_MODEL_SHA256,
        artifact_size_bytes=17_106_775_008,
        content_verified=True,
        backend="hip_gfx1151",
        target_arch="gfx1151",
        weight_quant="gguf_q4_k_m",
        execution_profile="strict",
        execution_profile_manifest_sha256=_STRICT_MANIFEST_SHA256,
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=1,
        resident_capacity=1,
        candidate_budget=3,
        sampling_mode="greedy_fast",
        max_sequence_length=1024,
        context_tokens=67,
        output_horizon_tokens=25,
        memory_fit=True,
    )
    return replace(key, **changes)


def _evidence() -> SpeculativeMTPServingEvidence:
    return Qwen35GGUFModel().speculative_mtp_serving_evidence[0]


def test_qwen38_q4km_strict_c1_b3_capacity4_realized_singleton_is_automatic() -> None:
    decision = resolve_speculative_mtp_serving_plan(
        Qwen35GGUFModel().speculative_mtp_serving_evidence,
        key=_key(resident_capacity=4),
    )

    assert decision.admitted is True
    assert decision.selected_route == "speculative_mtp"
    assert decision.selected_candidate_count == 3
    assert decision.reason == "qualified_automatic_realized_singleton_c1_b3"
    assert decision.automatic_eligible is True
    assert decision.strict_fallback_key == "gguf_target_ar"
    assert decision.static_eligibility.state is SpeculativeMTPStaticState.SPECULATIVE_CAPABLE
    assert decision.static_eligibility.max_candidate_count == 3
    assert decision.static_eligibility.max_realized_group_rows == 1
    assert "realized_group_rows" not in decision.static_eligibility.as_dict()
    assert decision.as_dict()["static_eligibility"]["eligible"] is True


def test_qwen38_q4km_production_c1_b3_context128_is_explicit_only() -> None:
    evidence = Qwen35GGUFModel().speculative_mtp_serving_evidence
    key = _key(
        execution_profile="production",
        execution_profile_manifest_sha256=_PRODUCTION_EXPLICIT_MANIFEST_SHA256,
        context_tokens=128,
        output_horizon_tokens=24,
    )

    decision = resolve_speculative_mtp_serving_plan(evidence, key=key)
    over = resolve_speculative_mtp_serving_plan(
        evidence,
        key=replace(key, context_tokens=129),
    )

    assert decision.admitted is True
    assert decision.selected_route == "speculative_mtp"
    assert decision.selected_candidate_count == 3
    assert decision.reason == "qualified_explicit_production_c1_b3_context128"
    assert decision.automatic_eligible is False
    assert decision.strict_fallback_key == "gguf_target_ar"
    assert decision.evidence_artifacts[-1] == (
        "benchmarks/results/"
        "2026-08-27-gfx1151-qwen38-c68-c128-production-explicit.json"
    )
    assert over.admitted is False
    assert over.reason == "context_bucket_not_qualified"


def test_qwen38_q4km_production_c2_k3_d24_is_explicit_after_ar_rebase() -> None:
    evidence = Qwen35GGUFModel().speculative_mtp_serving_evidence
    key = _key(
        execution_profile="production",
        execution_profile_manifest_sha256=_PRODUCTION_MANIFEST_SHA256,
        realized_group_rows=2,
        resident_capacity=4,
        context_tokens=128,
        output_horizon_tokens=24,
    )

    singleton = resolve_speculative_mtp_serving_plan(
        evidence,
        key=replace(key, realized_group_rows=1),
    )
    decision = resolve_speculative_mtp_serving_plan(evidence, key=key)
    over_horizon = resolve_speculative_mtp_serving_plan(
        evidence,
        key=replace(key, output_horizon_tokens=25),
    )

    assert singleton.admitted is True
    assert singleton.automatic_eligible is False
    assert singleton.static_eligibility.max_realized_group_rows == 2
    assert singleton.reason == "diagnostic_production_cap4_c1_or_c2_after_ar_rebase"
    assert decision.admitted is True
    assert decision.selected_route == "speculative_mtp"
    assert decision.selected_candidate_count == 3
    assert decision.reason == "diagnostic_production_c2_after_ar_rebase"
    assert decision.automatic_eligible is False
    assert decision.static_eligibility.max_realized_group_rows == 2
    assert decision.strict_fallback_key == "gguf_target_ar"
    assert over_horizon.admitted is False
    assert over_horizon.reason == "output_horizon_not_qualified"


def test_qwen38_q4km_production_c8_k3_d24_is_explicit_and_c7_is_not() -> None:
    evidence = Qwen35GGUFModel().speculative_mtp_serving_evidence
    key = _key(
        execution_profile="production",
        execution_profile_manifest_sha256=_PRODUCTION_MANIFEST_SHA256,
        realized_group_rows=8,
        resident_capacity=8,
        context_tokens=128,
        output_horizon_tokens=24,
    )

    decision = resolve_speculative_mtp_serving_plan(evidence, key=key)
    c7 = resolve_speculative_mtp_serving_plan(
        evidence,
        key=replace(key, realized_group_rows=7),
    )

    assert decision.admitted is True
    assert decision.selected_route == "speculative_mtp"
    assert decision.selected_candidate_count == 3
    assert decision.reason == (
        "qualified_explicit_production_c8_k3_after_q6_lm_head_rebase"
    )
    assert decision.automatic_eligible is False
    assert decision.static_eligibility.max_realized_group_rows == 8
    assert decision.strict_fallback_key == "gguf_target_ar"
    assert decision.evidence_artifacts[0] == (
        "benchmarks/results/"
        "2026-09-05-gfx1151-qwen38-c8-k3-width-policy-retained.json"
    )
    assert c7.admitted is False
    assert c7.reason == "physical_group_not_qualified"


def test_qwen38_q4km_strict_c1_b3_plan_is_automatic_product_scope() -> None:
    decision = resolve_speculative_mtp_serving_plan((_evidence(),), key=_key())

    assert decision.admitted is True
    assert decision.selected_route == "speculative_mtp"
    assert decision.selected_candidate_count == 3
    assert decision.reason == "qualified_automatic_c1_b3"
    assert decision.automatic_eligible is True
    assert decision.strict_fallback_key == "gguf_target_ar"
    assert decision.evidence_artifacts == (
        "benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s0.json",
        "benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s0-openai.json",
        "benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s1.json",
        "benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s2.json",
        "benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s3.json",
        "benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e0-current-baseline.json",
    )
    assert decision.plan_fingerprint.startswith("sha256:")
    assert decision.plan_fingerprint == resolve_speculative_mtp_serving_plan(
        (_evidence(),),
        key=_key(context_tokens=1),
    ).plan_fingerprint
    assert decision == resolve_speculative_mtp_serving_plan((_evidence(),), key=_key())


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"artifact_sha256": "0" * 64}, "artifact_not_qualified"),
        ({"artifact_size_bytes": 17_106_775_009}, "artifact_not_qualified"),
        ({"backend": "hip_gfx1100"}, "backend_not_qualified"),
        ({"target_arch": "gfx1100"}, "target_arch_not_qualified"),
        ({"weight_quant": "gguf_q4_k_s"}, "weight_quant_not_qualified"),
        ({"execution_profile": "production"}, "execution_profile_not_qualified"),
        (
            {"execution_profile_manifest_sha256": "1" * 64},
            "execution_profile_manifest_not_qualified",
        ),
        ({"kv_storage": "int8_per_token_head"}, "kv_storage_not_qualified"),
        ({"kv_layout": "paged_int8"}, "kv_layout_not_qualified"),
        ({"realized_group_rows": 2}, "physical_group_not_qualified"),
        ({"resident_capacity": 4}, "resident_capacity_not_qualified"),
        ({"candidate_budget": 2}, "candidate_budget_not_qualified"),
        ({"sampling_mode": "processed_argmax"}, "sampling_mode_not_qualified"),
        ({"max_sequence_length": 2048}, "max_sequence_length_not_qualified"),
        ({"context_tokens": 68}, "context_bucket_not_qualified"),
        ({"output_horizon_tokens": 24}, "output_horizon_not_qualified"),
        ({"memory_fit": False}, "insufficient_memory"),
    ],
)
def test_qwen38_candidate_plan_fails_closed_on_every_unqualified_axis(
    changes: dict[str, object],
    reason: str,
) -> None:
    decision = resolve_speculative_mtp_serving_plan((_evidence(),), key=_key(**changes))

    assert decision.admitted is False
    assert decision.selected_route == "default"
    assert decision.selected_candidate_count == 0
    assert decision.reason == reason
    assert decision.strict_fallback_key == "gguf_target_ar"


def test_qwen36_dense_production_row_resolves_after_qwen38_evidence() -> None:
    evidence = next(
        row
        for row in Qwen35GGUFModel().speculative_mtp_serving_evidence
        if row.evidence_key
        == "qwen36-dense-q4km-gfx1100-production-bf16-c1-k3-d24"
    )
    key = SpeculativeMTPServingKey(
        artifact_sha256=evidence.artifact_sha256,
        artifact_size_bytes=evidence.artifact_size_bytes,
        content_verified=True,
        backend=evidence.backend,
        target_arch=evidence.target_arch,
        weight_quant=evidence.weight_quant,
        execution_profile=evidence.execution_profile,
        execution_profile_manifest_sha256=evidence.execution_profile_manifest_sha256,
        kv_storage=evidence.kv_storage,
        kv_layout=evidence.kv_layout,
        realized_group_rows=1,
        resident_capacity=1,
        candidate_budget=3,
        sampling_mode="greedy_fast",
        max_sequence_length=1024,
        context_tokens=95,
        output_horizon_tokens=24,
        memory_fit=True,
    )

    decision = Qwen35GGUFModel().resolve_speculative_mtp_serving_plan(key=key)

    assert decision.admitted is True
    assert decision.automatic_eligible is True
    assert decision.reason == "qualified_automatic_dense_c1_k3_d24"
    assert decision.selected_candidate_count == 3


def test_qwen36_moe_production_c1_k2_plan_is_exact_automatic_scope() -> None:
    evidence = Qwen35MoeGGUFModel().speculative_mtp_serving_evidence[0]
    key = SpeculativeMTPServingKey(
        artifact_sha256=evidence.artifact_sha256,
        artifact_size_bytes=evidence.artifact_size_bytes,
        content_verified=True,
        backend=evidence.backend,
        target_arch=evidence.target_arch,
        weight_quant=evidence.weight_quant,
        execution_profile=evidence.execution_profile,
        execution_profile_manifest_sha256=(
            evidence.execution_profile_manifest_sha256
        ),
        kv_storage=evidence.kv_storage,
        kv_layout=evidence.kv_layout,
        realized_group_rows=1,
        resident_capacity=1,
        candidate_budget=2,
        sampling_mode="greedy_fast",
        max_sequence_length=1024,
        context_tokens=95,
        output_horizon_tokens=24,
        memory_fit=True,
    )

    decision = Qwen35MoeGGUFModel().resolve_speculative_mtp_serving_plan(key=key)

    assert decision.admitted is True
    assert decision.automatic_eligible is True
    assert decision.selected_candidate_count == 2
    assert decision.reason == "qualified_automatic_moe_c1_k2_d24"
    assert replace(key, context_tokens=96) != key
    assert Qwen35MoeGGUFModel().resolve_speculative_mtp_serving_plan(
        key=replace(key, context_tokens=96)
    ).reason == "context_bucket_not_qualified"
    assert Qwen35MoeGGUFModel().resolve_speculative_mtp_serving_plan(
        key=replace(key, output_horizon_tokens=25)
    ).reason == "output_horizon_not_qualified"


def test_rejected_serving_plan_exposes_permanent_ar_static_eligibility() -> None:
    decision = resolve_speculative_mtp_serving_plan(
        Qwen35GGUFModel().speculative_mtp_serving_evidence,
        key=_key(context_tokens=1024),
    )

    assert decision.admitted is False
    assert decision.static_eligibility.state is SpeculativeMTPStaticState.PERMANENT_AR
    assert decision.static_eligibility.eligible is False
    assert decision.static_eligibility.max_candidate_count == 0
    assert decision.static_eligibility.max_realized_group_rows == 0
    assert decision.static_eligibility.automatic_eligible is False


def test_unverified_artifact_and_generic_dense_inventory_cannot_admit() -> None:
    unverified = resolve_speculative_mtp_serving_plan(
        (_evidence(),),
        key=_key(
            artifact_sha256=None,
            artifact_size_bytes=None,
            content_verified=False,
        ),
    )
    generic = resolve_speculative_mtp_serving_plan((), key=_key())

    assert unverified.admitted is False
    assert unverified.reason == "artifact_identity_unverified"
    assert generic.admitted is False
    assert generic.reason == "no_model_plugin_evidence"


def test_unrelated_q4ks_artifact_keeps_legacy_explicit_compatibility(tmp_path) -> None:
    model_path = tmp_path / "qwen38-q4ks.gguf"
    with model_path.open("wb") as handle:
        handle.truncate(16_121_359_328)
    generator = Qwen35GGUFBringupGenerator.__new__(Qwen35GGUFBringupGenerator)
    generator.model_path = model_path
    generator.weight_index = SimpleNamespace(
        path=model_path,
        file_type_name="Q4_K_S",
    )
    generator.model_plugin = Qwen35GGUFModel()
    generator.backend = "hip_gfx1151"

    assert generator.resolve_speculative_mtp_serving_plan(
        execution_profile_manifest_sha256=_STRICT_MANIFEST_SHA256,
        realized_group_rows=1,
        resident_capacity=1,
        candidate_budget=3,
        sampling_mode="greedy_fast",
        max_sequence_length=1024,
        context_tokens=67,
        output_horizon_tokens=25,
        kv_storage="bf16",
        memory_fit=True,
    ) is None


def test_llm_delegates_mechanical_serving_identity_to_loaded_generator() -> None:
    calls = []

    class Generator:
        resident_capacity = 1

        def resolve_speculative_mtp_serving_plan(self, **kwargs):
            calls.append(kwargs)
            return "candidate"

    class LoadedLLM(LLM):
        @property
        def execution_profile_manifest_sha256(self):
            return _STRICT_MANIFEST_SHA256

        def _get_text_generator(self):
            return generator

    generator = Generator()
    llm = LoadedLLM(
        "fake.gguf",
        execution_profile="strict",
        max_active_requests=1,
        max_sequence_length=1024,
        speculative_candidate_budget=3,
    )

    decision = llm.resolve_speculative_mtp_serving_plan(
        realized_group_rows=1,
        sampling_mode="greedy_fast",
        context_tokens=67,
        output_horizon_tokens=25,
        kv_storage="auto",
        memory_fit=True,
    )

    assert decision == "candidate"
    assert calls == [
        {
            "execution_profile_manifest_sha256": _STRICT_MANIFEST_SHA256,
            "realized_group_rows": 1,
            "resident_capacity": 1,
            "candidate_budget": 3,
            "sampling_mode": "greedy_fast",
            "max_sequence_length": 1024,
            "context_tokens": 67,
            "output_horizon_tokens": 25,
            "kv_storage": "auto",
            "memory_fit": True,
        }
    ]


def test_serving_key_has_no_prompt_content_or_benchmark_identity_fields() -> None:
    names = {field.name for field in fields(SpeculativeMTPServingKey)}

    assert names.isdisjoint(
        {
            "prompt",
            "prompt_text",
            "prompt_hash",
            "prompt_token_ids",
            "category",
            "heldout",
            "task_result",
            "oracle",
        }
    )
