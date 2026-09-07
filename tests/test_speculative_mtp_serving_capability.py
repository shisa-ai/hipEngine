from __future__ import annotations

from dataclasses import fields, replace
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import resolve_runtime_profile
from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator
from hipengine.generation.qwen36_gguf_gfx1100_profiles import (
    QWEN36_DENSE_GGUF_BACKEND,
    QWEN36_DENSE_GGUF_MODEL,
    QWEN36_DENSE_GGUF_QUANT,
    register_qwen36_dense_gguf_gfx1100_profiles,
)
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
_W7900_MODEL_SHA256 = "7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b"
_W7900_PRODUCTION_MANIFEST_SHA256 = "2adc137a32d65bc63619947577f5233548d5835a474713abe270d666122a1960"


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
    return next(
        row for row in Qwen35GGUFModel().speculative_mtp_serving_evidence
        if row.evidence_key == "qwen38-q4km-gfx1151-strict-bf16-c1-b3-natural25-s0"
    )


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


def test_qwen38_q4km_gfx1100_production_c2_k2_d24_is_exact_automatic_key() -> None:
    evidence = Qwen35GGUFModel().speculative_mtp_serving_evidence
    key = _key(
        artifact_sha256=_W7900_MODEL_SHA256,
        artifact_size_bytes=17_106_773_984,
        backend="hip_gfx1100",
        target_arch="gfx1100",
        execution_profile="production",
        execution_profile_manifest_sha256=_W7900_PRODUCTION_MANIFEST_SHA256,
        realized_group_rows=2,
        resident_capacity=2,
        candidate_budget=2,
        context_tokens=95,
        output_horizon_tokens=24,
    )

    decision = resolve_speculative_mtp_serving_plan(evidence, key=key)
    assert decision.admitted is True
    # Qwen3.8 MTP is no longer automatic on gfx1100: the 2026-09-06 C x K
    # sweep measured every width below its own AR arm (best cell C8/K3 at
    # 0.9902x), so the rows stay admissible for explicit opt-in and
    # re-measurement but the automatic route selects AR.
    assert decision.automatic_eligible is False
    assert decision.selected_candidate_count == 2
    assert decision.reason == (
        "qualified_automatic_gfx1100_production_c2_k2_d24"
        "_measured_slower_than_ar_2026_09_06"
    )
    assert decision.static_eligibility.max_realized_group_rows == 2
    assert any(
        path.endswith("2026-08-30-w7900-qwen38-q4km-p12-c2-automatic-promotion.json")
        for path in decision.evidence_artifacts
    )
    assert decision.evidence_artifacts[-1].endswith(
        "2026-09-05-w7900-q4km-k3-c8-automatic-promotion.json"
    )

    # A row qualifies a maximum speculative depth, not one exact depth. A
    # shallower chain is strictly less speculative work through the same
    # verified path, so it must admit and must select the requested depth
    # rather than silently running the row's deeper budget.
    shallower = resolve_speculative_mtp_serving_plan(
        evidence,
        key=replace(key, candidate_budget=1),
    )
    assert shallower.admitted is True
    assert shallower.selected_candidate_count == 1

    for changes, reason in (
        ({"resident_capacity": 4}, "resident_capacity_not_qualified"),
        ({"realized_group_rows": 1}, "physical_group_not_qualified"),
        (
            {"realized_group_rows": 3, "resident_capacity": 3},
            "physical_group_not_qualified",
        ),
        ({"context_tokens": 96}, "context_bucket_not_qualified"),
        ({"context_tokens": 3}, "context_bucket_not_qualified"),
        ({"output_horizon_tokens": 25}, "output_horizon_not_qualified"),
        ({"sampling_mode": "sampled"}, "sampling_mode_not_qualified"),
    ):
        rejected = resolve_speculative_mtp_serving_plan(
            evidence,
            key=replace(key, **changes),
        )
        assert rejected.admitted is False
        assert rejected.automatic_eligible is False
        assert rejected.reason == reason


def test_qwen38_q4km_gfx1100_production_c2_k3_d24_is_explicit_packet6_selection() -> None:
    """The Packet 6 grid selection qualifies C2/K3 as an explicit row.

    The 56-cell diagnostic grid measured K3 as the best C2 depth (1.069x
    diagnostic; 1.067x retained vs the K2 row's 1.005x), so budget-3
    requests at C2 now admit through the registered row while budget-4
    still fails closed on the depth axis.
    """

    evidence = Qwen35GGUFModel().speculative_mtp_serving_evidence
    key = _key(
        artifact_sha256=_W7900_MODEL_SHA256,
        artifact_size_bytes=17_106_773_984,
        backend="hip_gfx1100",
        target_arch="gfx1100",
        execution_profile="production",
        execution_profile_manifest_sha256=_W7900_PRODUCTION_MANIFEST_SHA256,
        realized_group_rows=2,
        resident_capacity=2,
        candidate_budget=3,
        context_tokens=95,
        output_horizon_tokens=24,
    )

    decision = resolve_speculative_mtp_serving_plan(evidence, key=key)
    assert decision.admitted is True
    assert decision.automatic_eligible is False
    assert decision.selected_candidate_count == 3
    assert decision.reason == (
        "qualified_explicit_gfx1100_production_c2_k3_d24"
        "_packet6_grid_selection_2026_09_06"
    )
    assert decision.static_eligibility.max_realized_group_rows == 2
    assert decision.evidence_artifacts[-1].endswith(
        "2026-09-06-w7900-q4km-mtp-packet6-grid-and-c2k3.json"
    )

    deeper = resolve_speculative_mtp_serving_plan(
        evidence,
        key=replace(key, candidate_budget=4),
    )
    assert deeper.admitted is False
    assert deeper.reason == "candidate_budget_not_qualified"


def test_qwen38_q4km_gfx1100_production_c8_k3_d24_is_exact_automatic_key() -> None:
    evidence = Qwen35GGUFModel().speculative_mtp_serving_evidence
    key = _key(
        artifact_sha256=_W7900_MODEL_SHA256,
        artifact_size_bytes=17_106_773_984,
        backend="hip_gfx1100",
        target_arch="gfx1100",
        execution_profile="production",
        execution_profile_manifest_sha256=_W7900_PRODUCTION_MANIFEST_SHA256,
        realized_group_rows=8,
        resident_capacity=8,
        candidate_budget=3,
        context_tokens=95,
        output_horizon_tokens=24,
    )

    decision = resolve_speculative_mtp_serving_plan(evidence, key=key)
    assert decision.admitted is True
    # Qwen3.8 MTP is no longer automatic on gfx1100: the 2026-09-06 C x K
    # sweep measured every width below its own AR arm (best cell C8/K3 at
    # 0.9902x), so the rows stay admissible for explicit opt-in and
    # re-measurement but the automatic route selects AR.
    assert decision.automatic_eligible is False
    assert decision.selected_candidate_count == 3
    assert decision.reason == (
        "qualified_automatic_gfx1100_production_c8_k3_d24"
        "_measured_slower_than_ar_2026_09_06"
    )
    assert decision.static_eligibility.max_realized_group_rows == 8
    assert decision.evidence_artifacts[-1].endswith(
        "2026-09-05-w7900-q4km-k3-c8-automatic-promotion.json"
    )

    singleton = resolve_speculative_mtp_serving_plan(
        evidence, key=replace(key, realized_group_rows=1),
    )
    assert singleton.admitted is True
    assert singleton.automatic_eligible is False
    assert singleton.evidence_key == "qwen38-q4km-gfx1100-production-bf16-c1-k3-d24"

    # C1 has separate capacity-8 evidence; widths 2-7 do not.
    for realized_rows in range(2, 8):
        rejected = resolve_speculative_mtp_serving_plan(
            evidence,
            key=replace(key, realized_group_rows=realized_rows),
        )
        assert rejected.admitted is False
        assert rejected.automatic_eligible is False
        assert rejected.selected_candidate_count == 0


def test_w7900_dense_evidence_tracks_current_profile_manifests() -> None:
    register_qwen36_dense_gguf_gfx1100_profiles()
    evidence = Qwen35GGUFModel().speculative_mtp_serving_evidence
    for profile in ("strict", "production"):
        resolved = resolve_runtime_profile(
            model=QWEN36_DENSE_GGUF_MODEL,
            backend=QWEN36_DENSE_GGUF_BACKEND,
            quant=QWEN36_DENSE_GGUF_QUANT,
            profile=profile,
        )
        relevant = tuple(
            row
            for row in evidence
            if row.backend == "hip_gfx1100" and row.execution_profile == profile
        )
        assert relevant
        assert {
            row.execution_profile_manifest_sha256 for row in relevant
        } == {resolved.manifest_sha256}


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
        ({"candidate_budget": 4}, "candidate_budget_not_qualified"),
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


def test_serving_resolver_selects_exact_physical_width_among_same_artifact_rows() -> None:
    c1 = _evidence()
    c2 = replace(
        c1,
        evidence_key="qwen38-q4km-gfx1151-strict-bf16-c2-b3-natural25",
        realized_group_rows=2,
        resident_capacity=2,
        max_realized_group_rows=2,
        reason="qualified_explicit_c2_b3",
        automatic_eligible=False,
    )

    decision = resolve_speculative_mtp_serving_plan(
        (c1, c2),
        key=_key(realized_group_rows=2, resident_capacity=2),
    )

    assert decision.admitted is True
    assert decision.reason == "qualified_explicit_c2_b3"
    assert decision.evidence_key == c2.evidence_key
    assert decision.automatic_eligible is False

    c1_capacity2 = replace(
        c1,
        evidence_key="qwen38-q4km-gfx1151-strict-bf16-c1-cap2-b3-natural25",
        resident_capacity=2,
        reason="qualified_explicit_c1_cap2_b3",
        automatic_eligible=False,
    )
    c1_decision = resolve_speculative_mtp_serving_plan(
        (c1, c1_capacity2, c2),
        key=_key(realized_group_rows=1, resident_capacity=2),
    )
    assert c1_decision.admitted is True
    assert c1_decision.evidence_key == c1_capacity2.evidence_key


def test_serving_resolver_prefers_future_c2_intent_on_equal_score() -> None:
    c1 = _evidence()
    c2 = replace(
        c1,
        evidence_key="future-c2",
        realized_group_rows=2,
        resident_capacity=2,
        max_realized_group_rows=2,
        reason="qualified_future_c2",
        automatic_eligible=True,
    )

    decision = resolve_speculative_mtp_serving_plan(
        (c1, c2),
        key=_key(realized_group_rows=1, resident_capacity=2),
    )

    assert decision.admitted is False
    assert decision.reason == "physical_group_not_qualified"
    assert decision.static_eligibility.eligible is True
    assert decision.static_eligibility.max_realized_group_rows == 2
    assert decision.static_eligibility.evidence_key == "future-c2"


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


def test_qwen36_dense_strict_c2_k2_plan_is_explicit_only() -> None:
    evidence = next(
        row
        for row in Qwen35GGUFModel().speculative_mtp_serving_evidence
        if row.evidence_key
        == "qwen36-dense-q4km-gfx1100-strict-bf16-c2-k2-d24"
    )
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
        realized_group_rows=2,
        resident_capacity=2,
        candidate_budget=2,
        sampling_mode="greedy_fast",
        max_sequence_length=1024,
        context_tokens=95,
        output_horizon_tokens=24,
        memory_fit=True,
    )

    decision = Qwen35GGUFModel().resolve_speculative_mtp_serving_plan(key=key)

    assert decision.admitted is True
    assert decision.automatic_eligible is False
    assert decision.selected_candidate_count == 2
    assert decision.reason == "qualified_explicit_dense_c2_k2_d24"
    assert replace(key, realized_group_rows=1) != key
    assert Qwen35GGUFModel().resolve_speculative_mtp_serving_plan(
        key=replace(key, realized_group_rows=1)
    ).reason == "physical_group_not_qualified"


def test_qwen36_dense_production_c2_k2_plan_is_exact_automatic_scope() -> None:
    evidence = next(
        row
        for row in Qwen35GGUFModel().speculative_mtp_serving_evidence
        if row.evidence_key
        == "qwen36-dense-q4km-gfx1100-production-bf16-c2-k2-d24"
    )
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
        realized_group_rows=2,
        resident_capacity=2,
        candidate_budget=2,
        sampling_mode="greedy_fast",
        max_sequence_length=1024,
        context_tokens=95,
        output_horizon_tokens=24,
        memory_fit=True,
    )

    decision = Qwen35GGUFModel().resolve_speculative_mtp_serving_plan(key=key)

    assert decision.admitted is True
    assert decision.automatic_eligible is True
    assert decision.selected_candidate_count == 2
    assert decision.reason == "qualified_automatic_production_dense_c2_k2_d24"
    assert decision.strict_fallback_key == "gguf_target_ar"
    assert any(
        path.endswith("2026-08-27-w7900-27b-dense-mtp2-c2-automatic-promotion.json")
        for path in decision.evidence_artifacts
    )
    assert decision.evidence_artifacts[-1].endswith(
        "2026-09-05-w7900-q4km-k3-c8-automatic-promotion.json"
    )

    frontend_c1 = Qwen35GGUFModel().resolve_speculative_mtp_serving_plan(
        key=replace(key, realized_group_rows=1)
    )
    assert frontend_c1.admitted is False
    assert frontend_c1.reason == "physical_group_not_qualified"
    assert frontend_c1.static_eligibility.eligible is True
    assert frontend_c1.static_eligibility.automatic_eligible is True
    assert frontend_c1.static_eligibility.max_candidate_count == 2
    assert frontend_c1.static_eligibility.max_realized_group_rows == 2

    for overrides, reason in (
        # candidate_budget=3 ties the dense row (budget axis) with the
        # earlier-declared qwen38 C2/K3 row (artifact axis) at one failed
        # check each; the documented declaration-order tie-break attributes
        # the rejection to the qwen38 row.
        ({"candidate_budget": 3}, "artifact_not_qualified"),
        ({"output_horizon_tokens": 23}, "output_horizon_not_qualified"),
        ({"context_tokens": 96}, "context_bucket_not_qualified"),
        ({"sampling_mode": "processed_argmax"}, "sampling_mode_not_qualified"),
        ({"max_sequence_length": 2048}, "max_sequence_length_not_qualified"),
        ({"kv_storage": "int8"}, "kv_storage_not_qualified"),
        ({"realized_group_rows": 3}, "physical_group_not_qualified"),
        ({"resident_capacity": 3}, "resident_capacity_not_qualified"),
        ({"memory_fit": False}, "insufficient_memory"),
    ):
        rejected = Qwen35GGUFModel().resolve_speculative_mtp_serving_plan(
            key=replace(key, **overrides)
        )
        assert rejected.admitted is False
        assert rejected.static_eligibility.eligible is False
        assert rejected.reason == reason


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


def test_qwen36_moe_production_c2_k2_plan_is_exact_automatic_scope() -> None:
    evidence = Qwen35MoeGGUFModel().speculative_mtp_serving_evidence[1]
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
        realized_group_rows=2,
        resident_capacity=2,
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
    assert decision.reason == "qualified_automatic_production_moe_c2_k2_d24"
    for changed, reason in (
        ({"realized_group_rows": 3}, "physical_group_not_qualified"),
        ({"realized_group_rows": 1}, "physical_group_not_qualified"),
        ({"candidate_budget": 3}, "candidate_budget_not_qualified"),
        ({"output_horizon_tokens": 25}, "output_horizon_not_qualified"),
    ):
        assert Qwen35MoeGGUFModel().resolve_speculative_mtp_serving_plan(
            key=replace(key, **changed)
        ).reason == reason


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


def test_llm_capability_selects_matching_row_for_resident_capacity() -> None:
    c1 = _evidence()
    c2 = replace(
        c1,
        evidence_key="fake-c2-cap2",
        realized_group_rows=2,
        resident_capacity=2,
        max_realized_group_rows=2,
    )
    calls = []

    class Generator:
        resident_capacity = 2

        def resolve_speculative_mtp_serving_plan(self, **kwargs):
            calls.append(dict(kwargs))
            return SimpleNamespace(
                admitted=kwargs["realized_group_rows"] == 2,
                realized_group_rows=kwargs["realized_group_rows"],
            )

    class LoadedLLM(LLM):
        @property
        def execution_profile_manifest_sha256(self):
            return _STRICT_MANIFEST_SHA256

        def _load_model_metadata(self):
            return None, SimpleNamespace(
                speculative_mtp_serving_evidence=(c1, c2)
            )

        def _get_text_generator(self):
            return generator

    generator = Generator()
    llm = LoadedLLM(
        "fake.gguf",
        execution_profile="strict",
        max_active_requests=2,
        max_sequence_length=1024,
        speculative_candidate_budget=3,
    )

    decision = llm.speculative_mtp_serving_capability

    assert decision.admitted is True
    assert decision.realized_group_rows == 2
    assert [call["realized_group_rows"] for call in calls] == [1, 2]


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
