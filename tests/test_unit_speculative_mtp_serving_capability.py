from __future__ import annotations

from dataclasses import fields, replace
from types import SimpleNamespace

import pytest

from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator
from hipengine.llm import LLM
from hipengine.models.kv_capabilities import ModelArtifactIdentity
from hipengine.models.qwen35 import Qwen35GGUFModel, Qwen35MoeGGUFModel
from hipengine.server.api import (
    ServerConfig,
    _log_pretty_startup_summary,
    _speculation_startup_text,
)
from hipengine.speculative.serving import (
    STRUCTURAL_REJECTION_AXES,
    SpeculativeMTPServingEvidence,
    SpeculativeMTPServingKey,
    SpeculativeMTPStaticState,
    resolve_max_qualified_candidate_budget,
    resolve_speculative_mtp_serving_plan,
)


_MODEL_SHA256 = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
_W7900_MODEL_SHA256 = "7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b"
# Execution identity of both Qwen3.8-27B-Q4_K_M builds (see models/qwen35.py).
_PLAIN_FINGERPRINT = "4c4268886f225fba3675e32a521fba1d6ff1db4562bd06416c89fd22b6f90faa"
_UD_Q4KM_FINGERPRINT = "93fe11b8ac0f4696567cc123f2c08ea4bc7fcf615ed985d4fb11742df404f3ca"


def _key(**changes) -> SpeculativeMTPServingKey:
    key = SpeculativeMTPServingKey(
        artifact_sha256=_MODEL_SHA256,
        artifact_size_bytes=17_106_775_008,
        artifact_execution_fingerprint=_PLAIN_FINGERPRINT,
        content_verified=True,
        backend="hip_gfx1151",
        target_arch="gfx1151",
        weight_quant="gguf_q4_k_m",
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=1,
        resident_capacity=1,
        candidate_budget=3,
        sampling_mode="greedy_fast",
        memory_fit=True,
    )
    return replace(key, **changes)


def _evidence() -> SpeculativeMTPServingEvidence:
    return next(
        row for row in Qwen35GGUFModel().speculative_mtp_serving_evidence
        if row.evidence_key == "qwen38-q4km-gfx1151-strict-bf16-c1-b3-natural25-s0"
    )


# The Qwen3.6-27B-Q4_K_M artifact the three qwen36-dense rows measured is not on
# this host, so those rows record no execution identity and authorize nothing.
# Tests that exercise their scope bind them to this stand-in, which is the only
# missing piece; the real value comes from
# `python3 scripts/gguf_execution_identity.py /models/gguf/Qwen3.6-27B-Q4_K_M.gguf`.
_QWEN36_DENSE_FINGERPRINT = "b7" * 32


def _bound_rows(fingerprint: str = _QWEN36_DENSE_FINGERPRINT):
    """The dense plugin's rows with an identity supplied for the unbound ones."""

    return tuple(
        row
        if row.artifact_execution_fingerprint is not None
        else replace(row, artifact_execution_fingerprint=fingerprint)
        for row in Qwen35GGUFModel().speculative_mtp_serving_evidence
    )


def _row_key(row: SpeculativeMTPServingEvidence, **changes):
    """Build the serving key for one evidence row's own physical cell."""

    key = SpeculativeMTPServingKey(
        artifact_sha256=row.artifact_sha256,
        artifact_size_bytes=row.artifact_size_bytes,
        artifact_execution_fingerprint=row.artifact_execution_fingerprint,
        content_verified=True,
        backend=row.backend,
        target_arch=row.target_arch,
        weight_quant=row.weight_quant,
        kv_storage=row.kv_storage,
        kv_layout=row.kv_layout,
        realized_group_rows=row.realized_group_rows,
        resident_capacity=row.resident_capacity,
        candidate_budget=row.candidate_budget,
        sampling_mode=row.sampling_modes[0],
        memory_fit=True,
    )
    return replace(key, **changes)


@pytest.mark.parametrize("budget", [2, 3, 7])
@pytest.mark.parametrize("capacity", [1, 2, 8])
def test_w7900_c1_fails_closed_until_packed_target_is_qualified(budget, capacity) -> None:
    decision = resolve_speculative_mtp_serving_plan(
        Qwen35GGUFModel().speculative_mtp_serving_evidence,
        key=_key(
            artifact_sha256=_W7900_MODEL_SHA256,
            artifact_size_bytes=17_106_773_984,
            backend="hip_gfx1100", target_arch="gfx1100",
            realized_group_rows=1, resident_capacity=capacity,
            candidate_budget=budget,
        ),
    )
    assert decision.admitted is False
    assert decision.automatic_eligible is False
    assert decision.selected_candidate_count == 0


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


def test_qwen38_q4km_gfx1151_c1_b3_cell_takes_the_strongest_authorization() -> None:
    """Two rows describe this cell; the automatic authorization wins.

    The strict row was retained for the natural-25 scope and the production row
    for the 68-128 context scope.  Admission is now purely physical, so the cell
    resolves once and keeps the automatic eligibility rather than depending on
    declaration order.
    """

    evidence = Qwen35GGUFModel().speculative_mtp_serving_evidence

    decision = resolve_speculative_mtp_serving_plan(evidence, key=_key())

    assert decision.admitted is True
    assert decision.selected_route == "speculative_mtp"
    assert decision.selected_candidate_count == 3
    assert decision.evidence_key == (
        "qwen38-q4km-gfx1151-strict-bf16-c1-b3-natural25-s0"
    )
    assert decision.reason == "qualified_automatic_c1_b3"
    assert decision.automatic_eligible is True
    assert decision.strict_fallback_key == "gguf_target_ar"
    assert decision.evidence_artifacts[-1] == (
        "benchmarks/results/"
        "2026-08-29-gfx1151-qwen38-mtp-e0-current-baseline.json"
    )


def test_qwen38_q4km_production_c2_k3_d24_is_explicit_after_ar_rebase() -> None:
    evidence = Qwen35GGUFModel().speculative_mtp_serving_evidence
    key = _key(
        realized_group_rows=2,
        resident_capacity=4,
    )

    singleton = resolve_speculative_mtp_serving_plan(
        evidence,
        key=replace(key, realized_group_rows=1),
    )
    decision = resolve_speculative_mtp_serving_plan(evidence, key=key)
    deeper = resolve_speculative_mtp_serving_plan(
        evidence,
        key=replace(key, candidate_budget=4),
    )

    assert singleton.admitted is True
    assert singleton.automatic_eligible is True
    assert singleton.static_eligibility.max_realized_group_rows == 1
    assert singleton.reason == "qualified_automatic_realized_singleton_c1_b3"
    assert decision.admitted is True
    assert decision.selected_route == "speculative_mtp"
    assert decision.selected_candidate_count == 3
    assert decision.reason == "diagnostic_production_c2_after_ar_rebase"
    assert decision.automatic_eligible is False
    assert decision.static_eligibility.max_realized_group_rows == 2
    assert decision.strict_fallback_key == "gguf_target_ar"
    assert deeper.admitted is False
    assert deeper.reason == "candidate_budget_unmeasured"


def test_qwen38_q4km_gfx1100_production_c2_k2_d24_is_exact_automatic_key() -> None:
    evidence = Qwen35GGUFModel().speculative_mtp_serving_evidence
    key = _key(
        artifact_sha256=_W7900_MODEL_SHA256,
        artifact_size_bytes=17_106_773_984,
        backend="hip_gfx1100",
        target_arch="gfx1100",
        realized_group_rows=2,
        resident_capacity=2,
        candidate_budget=2,
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
        ({"resident_capacity": 4}, "resident_capacity_unmeasured"),
        ({"realized_group_rows": 1}, "physical_group_unmeasured"),
        (
            {"realized_group_rows": 3, "resident_capacity": 3},
            "physical_group_unmeasured",
        ),
        ({"sampling_mode": "sampled"}, "sampling_mode_unmeasured"),
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
        realized_group_rows=2,
        resident_capacity=2,
        candidate_budget=3,
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
    assert deeper.reason == "candidate_budget_unmeasured"


def test_qwen38_q4km_gfx1100_production_c8_k3_d24_is_exact_automatic_key() -> None:
    evidence = Qwen35GGUFModel().speculative_mtp_serving_evidence
    key = _key(
        artifact_sha256=_W7900_MODEL_SHA256,
        artifact_size_bytes=17_106_773_984,
        backend="hip_gfx1100",
        target_arch="gfx1100",
        realized_group_rows=8,
        resident_capacity=8,
        candidate_budget=3,
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

    # The C1 measurements used the legacy target and cannot qualify the
    # repaired packed target. Only the exact C8 key remains admitted here.
    for realized_rows in range(1, 8):
        rejected = resolve_speculative_mtp_serving_plan(
            evidence,
            key=replace(key, realized_group_rows=realized_rows),
        )
        assert rejected.admitted is False
        assert rejected.automatic_eligible is False
        assert rejected.selected_candidate_count == 0


def test_serving_evidence_rows_are_not_admitted_by_profile_or_request_shape() -> None:
    """Shape and profile describe a measurement, not an admission scope.

    They change with ordinary serving traffic and with any kernel or variant
    selection, so gating on them silently disables an already-qualified path.
    The provider keeps its own profile authorization.
    """

    removed_axes = {
        "execution_profile",
        "execution_profile_manifest_sha256",
        "max_sequence_length",
        "context_tokens",
        "output_horizon_tokens",
        "min_context_tokens",
        "max_context_tokens",
        "min_output_horizon_tokens",
        "max_output_horizon_tokens",
    }
    assert {field.name for field in fields(SpeculativeMTPServingKey)}.isdisjoint(
        removed_axes
    )
    assert {
        field.name for field in fields(SpeculativeMTPServingEvidence)
    }.isdisjoint(removed_axes)


def test_qwen38_q4km_gfx1151_production_c8_k3_d24_is_withdrawn() -> None:
    # The gfx1151 C8-K3 serving row was withdrawn on 2026-09-19: the backend's
    # prompt-streaming policy admits widths (1,2,3,4), so an eight-row pending set
    # is refused by the provider while the route still ran its cycles, and the
    # cell measured 46.88 tok/s against 50.10 true AR (0.936x) with 0 of 10 cells
    # reproducing the AR output. An explicit capacity-8 request now resolves to
    # the registered strict fallback instead of a wide speculative cell.
    evidence = Qwen35GGUFModel().speculative_mtp_serving_evidence
    key = _key(
        realized_group_rows=8,
        resident_capacity=8,
    )

    decision = resolve_speculative_mtp_serving_plan(evidence, key=key)
    c7 = resolve_speculative_mtp_serving_plan(
        evidence,
        key=replace(key, realized_group_rows=7),
    )

    assert decision.admitted is False
    assert decision.selected_route == "default"
    assert decision.selected_candidate_count == 0
    assert decision.strict_fallback_key == "gguf_target_ar"
    assert c7.admitted is False
    assert all(
        row.evidence_key != "qwen38-q4km-gfx1151-production-bf16-c8-k3-d24"
        for row in evidence
    )
    assert c7.admitted is False
    assert c7.reason == "physical_group_unmeasured"


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
        key=_key(sampling_mode="greedy_fast"),
    ).plan_fingerprint
    assert decision == resolve_speculative_mtp_serving_plan((_evidence(),), key=_key())


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"artifact_execution_fingerprint": "0" * 64}, "artifact_unmeasured"),
        ({"backend": "hip_gfx1100"}, "backend_unmeasured"),
        ({"target_arch": "gfx1100"}, "target_arch_unmeasured"),
        ({"weight_quant": "gguf_q4_k_s"}, "weight_quant_unmeasured"),
        ({"kv_storage": "int8_per_token_head"}, "kv_storage_unmeasured"),
        ({"kv_layout": "paged_int8"}, "kv_layout_unmeasured"),
        ({"realized_group_rows": 2}, "physical_group_unmeasured"),
        ({"resident_capacity": 4}, "resident_capacity_unmeasured"),
        ({"candidate_budget": 4}, "candidate_budget_unmeasured"),
        ({"sampling_mode": "processed_argmax"}, "sampling_mode_unmeasured"),
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
    # The summary reason is the first failed axis; the full set is what a
    # screening decision has to read, because the first axis can be screenable
    # while a later one is not.
    assert decision.failed_axes == (reason,)
    assert decision.structural_rejection == (
        reason if reason in STRUCTURAL_REJECTION_AXES else None
    )


def test_structural_rejection_axes_are_the_correctness_boundaries() -> None:
    """Pin the fail-closed set so widening it is a deliberate edit."""

    assert STRUCTURAL_REJECTION_AXES == frozenset(
        {
            "sampling_mode_unmeasured",
            "insufficient_memory",
        }
    )


def test_rejection_reports_every_failed_axis_not_just_the_summary_reason() -> None:
    """A screenable first axis must not hide a structural failure behind it.

    Reproduced finding: with real model evidence, ``candidate_budget=4`` (over
    the row's qualified depth) plus ``memory_fit=False`` reported
    ``candidate_budget_unmeasured`` as its reason, which is a screenable
    axis, so the screening helper granted override eligibility for a cell that
    did not fit in memory at all. Memory fit is a correctness boundary and must
    stay blocked for every request.
    """

    decision = resolve_speculative_mtp_serving_plan(
        (_evidence(),),
        key=_key(candidate_budget=4, memory_fit=False),
    )

    assert decision.admitted is False
    assert decision.reason == "candidate_budget_unmeasured"
    assert decision.failed_axes == (
        "candidate_budget_unmeasured",
        "insufficient_memory",
    )
    assert decision.structural_rejection == "insufficient_memory"
    assert "failed_axes" in decision.as_dict()
    assert decision.as_dict()["failed_axes"] == [
        "candidate_budget_unmeasured",
        "insufficient_memory",
    ]

    masked_sampling = resolve_speculative_mtp_serving_plan(
        (_evidence(),),
        key=_key(
            candidate_budget=4,
            sampling_mode="processed_argmax",
            memory_fit=False,
        ),
    )
    assert masked_sampling.reason == "candidate_budget_unmeasured"
    assert masked_sampling.failed_axes == (
        "candidate_budget_unmeasured",
        "sampling_mode_unmeasured",
        "insufficient_memory",
    )
    assert masked_sampling.structural_rejection == "sampling_mode_unmeasured"

    # The structural axes stay structural on their own too, and a plan that
    # fails only screenable axes reports no structural rejection.
    assert (
        resolve_speculative_mtp_serving_plan(
            (_evidence(),), key=_key(memory_fit=False)
        ).structural_rejection
        == "insufficient_memory"
    )
    assert (
        resolve_speculative_mtp_serving_plan(
            (_evidence(),), key=_key(candidate_budget=4)
        ).structural_rejection
        is None
    )


def test_screening_switch_does_not_widen_model_plugin_evidence(monkeypatch) -> None:
    """Screening is a serving-layer decision, never an evidence widening.

    The resolver keeps failing closed on every unqualified axis even with the
    operator screening switch on; only the serving layer may turn that into an
    explicit, marked screening admission.
    """

    monkeypatch.setenv("HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS", "1")
    for changes, reason in (
        ({"candidate_budget": 4}, "candidate_budget_unmeasured"),
        ({"sampling_mode": "processed_argmax"}, "sampling_mode_unmeasured"),
        ({"memory_fit": False}, "insufficient_memory"),
        ({"realized_group_rows": 2}, "physical_group_unmeasured"),
    ):
        decision = resolve_speculative_mtp_serving_plan(
            (_evidence(),), key=_key(**changes)
        )
        assert decision.admitted is False, reason
        assert decision.selected_route == "default", reason
        assert decision.selected_candidate_count == 0, reason
        assert decision.reason == reason
        assert decision.static_eligibility.automatic_eligible is False


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
    assert decision.reason == "physical_group_unmeasured"
    assert decision.static_eligibility.eligible is True
    assert decision.static_eligibility.max_realized_group_rows == 2
    assert decision.static_eligibility.evidence_key == "future-c2"


def test_qwen36_dense_production_row_resolves_after_qwen38_evidence() -> None:
    """Declaration order does not hide a later row that owns its own cell."""

    plugin = Qwen35GGUFModel(speculative_mtp_serving_evidence=_bound_rows())
    evidence = next(
        row
        for row in plugin.speculative_mtp_serving_evidence
        if row.evidence_key
        == "qwen36-dense-q4km-gfx1100-production-bf16-c1-k3-d24"
    )

    decision = plugin.resolve_speculative_mtp_serving_plan(
        key=_row_key(evidence),
    )

    assert decision.admitted is True
    assert decision.automatic_eligible is True
    assert decision.reason == "qualified_automatic_dense_c1_k3_d24"
    assert decision.selected_candidate_count == 3


def test_qwen36_dense_rows_without_identity_still_run_on_capability() -> None:
    """A row that measures nothing on this host never withholds the cell.

    The three qwen36-dense rows carry no execution identity because their
    artifact is absent here.  That makes them unable to *certify* the cell; it
    does not make them able to refuse it.  Both request modes run on the
    implementation declaration, and recording the identity later upgrades the
    basis to evidence without changing what executes.
    """

    plugin = Qwen35GGUFModel()
    unbound = [
        row
        for row in plugin.speculative_mtp_serving_evidence
        if row.evidence_key.startswith("qwen36-dense-q4km-gfx1100")
    ]
    assert unbound, "the qwen36-dense rows are part of the retained evidence"
    assert all(row.artifact_execution_fingerprint is None for row in unbound)

    for row in unbound:
        # The key stands for a resident file whose identity is computable; the
        # row is the side that cannot be matched.
        key = _row_key(row, artifact_execution_fingerprint=_QWEN36_DENSE_FINGERPRINT)
        for request_mode in ("automatic", "explicit"):
            decision = plugin.resolve_speculative_mtp_serving_plan(
                key=key,
                request_mode=request_mode,
            )
            assert decision.admitted is True, request_mode
            assert decision.as_dict()["admission_basis"] == "implementation"


def test_qwen36_dense_c2_k2_cell_prefers_the_automatic_row() -> None:
    """The strict row is shadowed by the production row for the same cell.

    Both rows describe identical physical axes.  The cell keeps the automatic
    authorization instead of taking whichever row happens to be declared first.
    """

    plugin = Qwen35GGUFModel(speculative_mtp_serving_evidence=_bound_rows())
    evidence = next(
        row
        for row in plugin.speculative_mtp_serving_evidence
        if row.evidence_key
        == "qwen36-dense-q4km-gfx1100-strict-bf16-c2-k2-d24"
    )

    decision = plugin.resolve_speculative_mtp_serving_plan(
        key=_row_key(evidence),
    )

    assert decision.admitted is True
    assert decision.evidence_key == (
        "qwen36-dense-q4km-gfx1100-production-bf16-c2-k2-d24"
    )
    assert decision.automatic_eligible is True
    assert decision.selected_candidate_count == 2
    assert decision.reason == "qualified_automatic_production_dense_c2_k2_d24"
    # A narrower physical cell is outside what this row measured, so it stops
    # selecting the row -- and runs on the implementation declaration instead
    # of being refused.
    narrower = plugin.resolve_speculative_mtp_serving_plan(
        key=replace(evidence and _row_key(evidence), realized_group_rows=1)
    )
    assert narrower.admitted is True
    assert narrower.as_dict()["admission_basis"] == "implementation"


def test_qwen36_dense_production_c2_k2_plan_is_exact_automatic_scope() -> None:
    plugin = Qwen35GGUFModel(speculative_mtp_serving_evidence=_bound_rows())
    evidence = next(
        row
        for row in plugin.speculative_mtp_serving_evidence
        if row.evidence_key
        == "qwen36-dense-q4km-gfx1100-production-bf16-c2-k2-d24"
    )
    key = _row_key(evidence)

    decision = plugin.resolve_speculative_mtp_serving_plan(key=key)

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

    # A cell this row did not measure still runs when the kernels implement it.
    for admitted_overrides in (
        {"realized_group_rows": 1},
        {"realized_group_rows": 3},
        {"resident_capacity": 3},
        {"candidate_budget": 3},
    ):
        widened = plugin.resolve_speculative_mtp_serving_plan(
            key=replace(key, **admitted_overrides)
        )
        assert widened.admitted is True, admitted_overrides
        assert widened.as_dict()["admission_basis"] == "implementation"

    # Every refusal names a capability or resource fact, never a missing
    # measurement.
    for overrides, reason in (
        ({"sampling_mode": "processed_argmax"}, "mtp_sampling_unsupported"),
        ({"kv_storage": "int8"}, "mtp_contract_unsupported"),
        ({"memory_fit": False}, "insufficient_memory"),
    ):
        rejected = plugin.resolve_speculative_mtp_serving_plan(
            key=replace(key, **overrides)
        )
        assert rejected.admitted is False, overrides
        assert rejected.reason == reason


def test_qwen36_moe_production_c1_k2_plan_is_exact_automatic_scope() -> None:
    evidence = Qwen35MoeGGUFModel().speculative_mtp_serving_evidence[0]
    key = SpeculativeMTPServingKey(
        artifact_sha256=evidence.artifact_sha256,
        artifact_size_bytes=evidence.artifact_size_bytes,
        artifact_execution_fingerprint=evidence.artifact_execution_fingerprint,
        content_verified=True,
        backend=evidence.backend,
        target_arch=evidence.target_arch,
        weight_quant=evidence.weight_quant,
        kv_storage=evidence.kv_storage,
        kv_layout=evidence.kv_layout,
        realized_group_rows=1,
        resident_capacity=1,
        candidate_budget=2,
        sampling_mode="greedy_fast",
        memory_fit=True,
    )

    decision = Qwen35MoeGGUFModel().resolve_speculative_mtp_serving_plan(key=key)

    assert decision.admitted is True
    assert decision.automatic_eligible is True
    assert decision.selected_candidate_count == 2
    assert decision.reason == "qualified_automatic_moe_c1_k2_d24"
    # Off-scope now reports the implementation's own width bound, which is a
    # capability fact, rather than the absence of a measurement.
    assert Qwen35MoeGGUFModel().resolve_speculative_mtp_serving_plan(
        key=replace(key, realized_group_rows=3, resident_capacity=3)
    ).reason == "moe_group_above_offered_width"


def test_qwen36_moe_production_c2_k2_plan_is_exact_automatic_scope() -> None:
    evidence = Qwen35MoeGGUFModel().speculative_mtp_serving_evidence[1]
    key = SpeculativeMTPServingKey(
        artifact_sha256=evidence.artifact_sha256,
        artifact_size_bytes=evidence.artifact_size_bytes,
        artifact_execution_fingerprint=evidence.artifact_execution_fingerprint,
        content_verified=True,
        backend=evidence.backend,
        target_arch=evidence.target_arch,
        weight_quant=evidence.weight_quant,
        kv_storage=evidence.kv_storage,
        kv_layout=evidence.kv_layout,
        realized_group_rows=2,
        resident_capacity=2,
        candidate_budget=2,
        sampling_mode="greedy_fast",
        memory_fit=True,
    )

    decision = Qwen35MoeGGUFModel().resolve_speculative_mtp_serving_plan(key=key)

    assert decision.admitted is True
    assert decision.automatic_eligible is True
    assert decision.selected_candidate_count == 2
    assert decision.reason == "qualified_automatic_production_moe_c2_k2_d24"
    # Every refusal below names a kernel bound the implementation declares.
    for changed, reason in (
        ({"realized_group_rows": 3}, "moe_group_above_offered_width"),
        ({"candidate_budget": 3}, "mtp_candidate_depth_unsupported"),
        ({"sampling_mode": "sampled"}, "mtp_sampling_unsupported"),
    ):
        rejected = Qwen35MoeGGUFModel().resolve_speculative_mtp_serving_plan(
            key=replace(key, **changed)
        )
        assert rejected.admitted is False, changed
        assert rejected.reason == reason

    # A narrower physical cell is inside what the kernels implement, so it runs
    # even though this row measured the wider one.
    narrower = Qwen35MoeGGUFModel().resolve_speculative_mtp_serving_plan(
        key=replace(key, realized_group_rows=1)
    )
    assert narrower.admitted is True
    assert narrower.as_dict()["admission_basis"] == "implementation"


def test_rejected_serving_plan_exposes_permanent_ar_static_eligibility() -> None:
    decision = resolve_speculative_mtp_serving_plan(
        Qwen35GGUFModel().speculative_mtp_serving_evidence,
        key=_key(memory_fit=False),
    )

    assert decision.admitted is False
    assert decision.static_eligibility.state is SpeculativeMTPStaticState.PERMANENT_AR
    assert decision.static_eligibility.eligible is False
    assert decision.static_eligibility.max_candidate_count == 0
    assert decision.static_eligibility.max_realized_group_rows == 0
    assert decision.static_eligibility.automatic_eligible is False


def test_evidence_only_resolution_reports_unmeasured_never_unverified_identity() -> None:
    """Evidence resolution scopes measurements; it does not police identity.

    An artifact whose identity is unverified simply matches no retained row, so
    the reason is that nothing measured it.  Admission is decided by the
    implementation declaration the plugin layers on top, not here.
    """

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
    assert unverified.reason == "artifact_unmeasured"
    assert "unverified" not in unverified.reason
    assert generic.admitted is False
    assert generic.reason == "no_model_plugin_evidence"


def test_generator_resolves_the_qualified_qwen_depth_from_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    """The Qwen3.8 gfx1151 cell resolves to its qualified B3, not a constant."""

    # This test owns the depth ladder, not the inventory: the artifact axis is
    # the resident file's execution identity, so pin the identity the rows bind.
    monkeypatch.setattr(
        "hipengine.loading.gguf.gguf_execution_fingerprint",
        lambda _info: _PLAIN_FINGERPRINT,
    )
    model_path = tmp_path / "qwen38-q4km.gguf"
    with model_path.open("wb") as handle:
        handle.truncate(17_106_775_008)
    generator = Qwen35GGUFBringupGenerator.__new__(Qwen35GGUFBringupGenerator)
    generator.model_path = model_path
    generator.weight_index = SimpleNamespace(path=model_path, file_type_name="Q4_K_M")
    generator.model_plugin = Qwen35GGUFModel()
    generator.backend = "hip_gfx1151"
    generator._kv_artifact_identity = ModelArtifactIdentity(
        path=str(model_path),
        size_bytes=17_106_775_008,
        sha256=_MODEL_SHA256,
        content_verified=True,
    )

    assert (
        generator.max_qualified_speculative_candidate_budget(
            realized_group_rows=1,
            resident_capacity=1,
            sampling_mode="greedy_fast",
        )
        == 3
    )
    assert (
        generator.max_qualified_speculative_candidate_budget(
            realized_group_rows=1,
            resident_capacity=8,
            sampling_mode="greedy_fast",
        )
        is None
    )


def test_unrelated_q4ks_artifact_runs_on_implementation_capability(
    tmp_path,
    monkeypatch,
) -> None:
    """A Q4_K_S artifact no retained row binds still executes when requested.

    Its execution identity is unknown to the evidence, so no measurement backs
    it, but both intents resolve through the dense BF16 implementation
    declaration rather than falling through to an untracked legacy route.  The
    declaration is automatic-eligible, so automatic policy needs no measurement
    either: the physical cell table is what bounds it.
    """

    monkeypatch.setattr(
        "hipengine.loading.gguf.gguf_execution_fingerprint",
        lambda _info: "9" * 64,
    )
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

    explicit = generator.resolve_speculative_mtp_serving_plan(
        realized_group_rows=1,
        resident_capacity=1,
        candidate_budget=3,
        sampling_mode="greedy_fast",
        kv_storage="bf16",
        memory_fit=True,
        request_mode="explicit",
    )
    assert explicit is not None
    assert explicit.admitted is True
    assert explicit.as_dict()["admission_basis"] == "implementation"
    assert explicit.automatic_eligible is True

    # Automatic intent takes the same capability route: an unrelated artifact
    # is unmeasured, and unmeasured never means refused.
    automatic = generator.resolve_speculative_mtp_serving_plan(
        realized_group_rows=1,
        resident_capacity=1,
        candidate_budget=3,
        sampling_mode="greedy_fast",
        kv_storage="bf16",
        memory_fit=True,
    )
    assert automatic is not None
    assert automatic.admitted is True
    assert automatic.as_dict()["admission_basis"] == "implementation"
    assert automatic.automatic_eligible is True

    # The physical cell table stays the gate: gfx1151 offers a multi-row chain
    # only up to C4, so a wider group is a capability miss rather than a
    # measurement miss, and it refuses in both intents.
    wider = generator.resolve_speculative_mtp_serving_plan(
        realized_group_rows=8,
        resident_capacity=8,
        candidate_budget=3,
        sampling_mode="greedy_fast",
        kv_storage="bf16",
        memory_fit=True,
    )
    assert wider is not None
    assert wider.admitted is False
    assert wider.reason == "dense_group_above_offered_width"


def test_llm_delegates_mechanical_serving_identity_to_loaded_generator() -> None:
    calls = []

    class Generator:
        resident_capacity = 1

        def resolve_speculative_mtp_serving_plan(self, **kwargs):
            calls.append(kwargs)
            return "candidate"

    class LoadedLLM(LLM):
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
        kv_storage="auto",
        memory_fit=True,
    )

    assert decision == "candidate"
    assert calls == [
        {
            "realized_group_rows": 1,
            "resident_capacity": 1,
            "candidate_budget": 3,
            "sampling_mode": "greedy_fast",
            "kv_storage": "auto",
            "memory_fit": True,
            "request_mode": "automatic",
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


def test_max_qualified_candidate_budget_reads_the_cell_not_the_request() -> None:
    """An omitted budget resolves to the deepest depth this cell qualifies."""

    evidence = Qwen35GGUFModel().speculative_mtp_serving_evidence

    assert (
        resolve_max_qualified_candidate_budget(
            evidence, key=_key(candidate_budget=1)
        )
        == 3
    )
    assert (
        resolve_max_qualified_candidate_budget(
            evidence, key=_key(resident_capacity=4, candidate_budget=1)
        )
        == 3
    )


def test_max_qualified_candidate_budget_fails_closed_for_unqualified_cells() -> None:
    """No matching physical row means no depth to resolve, not a default."""

    evidence = Qwen35GGUFModel().speculative_mtp_serving_evidence

    for changed in (
        {"artifact_sha256": _W7900_MODEL_SHA256, "artifact_size_bytes": 17_106_773_984,
         "backend": "hip_gfx1100", "target_arch": "gfx1100"},
        {"weight_quant": "gguf_q4_k_s"},
        {"realized_group_rows": 2, "resident_capacity": 2},
        {"sampling_mode": "processed_argmax"},
        {"memory_fit": False},
    ):
        assert (
            resolve_max_qualified_candidate_budget(
                evidence, key=_key(candidate_budget=1, **changed)
            )
            is None
        )


def test_max_qualified_candidate_budget_prefers_automatic_over_deeper_explicit() -> None:
    """A depth retained only for explicit use never becomes the default."""

    automatic = replace(_evidence(), candidate_budget=3, automatic_eligible=True)
    deeper_explicit = replace(
        _evidence(),
        evidence_key="fake-explicit-b4",
        candidate_budget=4,
        automatic_eligible=False,
    )

    assert (
        resolve_max_qualified_candidate_budget(
            (deeper_explicit, automatic), key=_key(candidate_budget=1)
        )
        == 3
    )
    # Without the automatic row the explicit-only depth is still the best
    # retained authorization for the cell.
    assert (
        resolve_max_qualified_candidate_budget(
            (deeper_explicit,), key=_key(candidate_budget=1)
        )
        == 4
    )


def _budget_resolution_llm(generator, **kwargs) -> LLM:
    class LoadedLLM(LLM):
        def _get_text_generator(self):
            return generator

    return LoadedLLM(
        "fake.gguf",
        execution_profile="strict",
        max_active_requests=1,
        max_sequence_length=1024,
        **kwargs,
    )


def test_omitted_budget_waits_for_the_resident_owner_before_reading_evidence() -> None:
    """Evidence keys on the served capacity, so it is not read pre-prepare."""

    calls = []

    class Generator:
        # No ``resident_capacity`` yet: the resident runner does not exist.
        def max_qualified_speculative_candidate_budget(self, **kwargs):
            calls.append(dict(kwargs))
            return 3 if kwargs["resident_capacity"] == 4 else None

        def speculative_candidate_budget_default(self):
            raise AssertionError("a pre-prepare lookup must not publish a default")

    generator = Generator()
    llm = _budget_resolution_llm(generator)

    llm._publish_candidate_budget(generator)
    assert calls == []
    assert llm.speculative_candidate_budget is None
    assert llm.speculative_candidate_budget_source == "pending"

    # ``prepare`` creates the resident runner, which publishes its capacity.
    generator.resident_capacity = 4
    llm._publish_candidate_budget(generator)

    assert calls == [
        {
            "realized_group_rows": 1,
            "resident_capacity": 4,
            "sampling_mode": "greedy_fast",
            "kv_storage": "auto",
            "memory_fit": True,
        }
    ]
    assert llm.speculative_candidate_budget == 3
    assert llm.speculative_candidate_budget_source == "model_plugin_evidence"
    assert generator.speculative_candidate_budget == 3


def test_omitted_budget_resolves_from_model_plugin_evidence() -> None:
    """The default server configuration must not need a budget flag at all."""

    calls = []

    class Generator:
        resident_capacity = 1

        def max_qualified_speculative_candidate_budget(self, **kwargs):
            calls.append(dict(kwargs))
            return 3

        def speculative_candidate_budget_default(self):
            raise AssertionError("evidence must win over the generator default")

    llm = _budget_resolution_llm(Generator())
    budget, source = llm._resolve_candidate_budget(Generator())

    assert (budget, source) == (3, "model_plugin_evidence")
    assert calls == [
        {
            "realized_group_rows": 1,
            "resident_capacity": 1,
            "sampling_mode": "greedy_fast",
            "kv_storage": "auto",
            "memory_fit": True,
        }
    ]
    assert llm.speculative_candidate_budget_requested is None


def test_omitted_budget_falls_back_to_the_generator_default() -> None:
    """An unmeasured cell keeps the dense MTP owner's own declared depth."""

    class Generator:
        resident_capacity = 1

        def max_qualified_speculative_candidate_budget(self, **kwargs):
            return None

        def speculative_candidate_budget_default(self):
            return 3

    llm = _budget_resolution_llm(Generator())

    assert llm._resolve_candidate_budget(Generator()) == (3, "generator_default")


def test_omitted_budget_without_capability_data_stays_unresolved() -> None:
    """No evidence and no declared default means no speculative admission."""

    class Generator:
        resident_capacity = 1

        def resolve_speculative_mtp_serving_plan(self, **kwargs):
            raise AssertionError("an unresolved budget must not reach the resolver")

    llm = _budget_resolution_llm(Generator())

    assert llm._resolve_candidate_budget(Generator()) == (None, "unresolved")
    llm.speculative_candidate_budget = None
    assert (
        llm.resolve_speculative_mtp_serving_plan(
            realized_group_rows=1, sampling_mode="greedy_fast"
        )
        is None
    )


def test_omitted_budget_keeps_the_provider_declared_depth() -> None:
    """A generic provider owns its own shape; the server must not invent one."""

    class Generator:
        resident_capacity = 1

        def max_qualified_speculative_candidate_budget(self, **kwargs):
            raise AssertionError("provider depth is not model-plugin evidence")

    llm = _budget_resolution_llm(
        Generator(),
        speculative_provider="dflash",
        draft_model="draft.gguf",
    )

    assert llm._resolve_candidate_budget(Generator()) == (4, "provider_default")


def test_explicit_budget_is_never_rewritten() -> None:
    """An operator-pinned depth passes through capability resolution intact."""

    class Generator:
        resident_capacity = 1

        def max_qualified_speculative_candidate_budget(self, **kwargs):
            raise AssertionError("an explicit budget is not resolved from evidence")

    llm = _budget_resolution_llm(Generator(), speculative_candidate_budget=7)
    generator = Generator()

    assert llm._resolve_candidate_budget(generator) == (7, "explicit")
    assert llm.speculative_candidate_budget_requested == 7

    # The pinned depth still reaches the resident owner that consumes it.
    llm._publish_candidate_budget(generator)
    assert generator.speculative_candidate_budget == 7
    assert llm.speculative_candidate_budget_resolution == {
        "requested_candidate_budget": 7,
        "candidate_budget": 7,
        "candidate_budget_source": "explicit",
    }


def test_budget_resolution_reports_requested_resolved_and_source() -> None:
    """Operators can see which depth the server actually runs."""

    class Generator:
        resident_capacity = 1

        def max_qualified_speculative_candidate_budget(self, **kwargs):
            return 3

    llm = _budget_resolution_llm(Generator())
    llm.speculative_candidate_budget, llm.speculative_candidate_budget_source = (
        llm._resolve_candidate_budget(Generator())
    )

    assert llm.speculative_candidate_budget_resolution == {
        "requested_candidate_budget": None,
        "candidate_budget": 3,
        "candidate_budget_source": "model_plugin_evidence",
    }


# ---------------------------------------------------------------------------
# Startup summary states the resolved route, not the configured policy.
#
# ``serving_route`` is true whenever MTP is configured and the engine can serve
# it, so a line keyed on it advertises a route the request path may refuse.
# That is invisible to an operator reading the banner, which is how a server
# came to report "MTP enabled" while every request ran plain AR.
# ---------------------------------------------------------------------------

_BANNER_BUDGET = {"requested": None, "resolved": 3, "source": "generator_default"}


def _banner_text(
    config: ServerConfig,
    *,
    serving_route: bool,
    plan: dict | None,
    budget: dict = _BANNER_BUDGET,
) -> str:
    return _speculation_startup_text(
        config,
        capability={"serving_route": serving_route},
        budget=budget,
        engine=SimpleNamespace(speculative_mtp_serving_capability=plan),
    )


def _banner_config(**changes) -> ServerConfig:
    return ServerConfig(model="/models/example.gguf", served_model_name="example", **changes)


def test_startup_speculation_line_never_advertises_a_refused_route() -> None:
    """A configured route the plan refuses must not read as enabled."""

    config = _banner_config(speculative_mtp_serving="auto")
    text = _banner_text(
        config,
        serving_route=True,
        plan={"admitted": False, "reason": "insufficient_memory"},
    )
    assert text == "MTP unavailable (insufficient_memory)"


def test_startup_speculation_line_marks_a_depth_no_measurement_backs() -> None:
    """Capability-only admission reads differently from a measured one."""

    config = _banner_config(speculative_mtp_serving="auto")
    unmeasured = _banner_text(
        config,
        serving_route=True,
        plan={
            "admitted": True,
            "automatic_eligible": True,
            "reason": "implemented_gguf_dense_bf16_gfx1151_c1_native_chain",
            "selected_candidate_count": 3,
        },
    )
    assert unmeasured == "MTP enabled, candidate budget 3 (unmeasured)"

    measured = _banner_text(
        config,
        serving_route=True,
        plan={
            "admitted": True,
            "automatic_eligible": True,
            "evidence_key": "measured-cell",
            "evidence_fingerprint": "sha256:measured-cell",
            "evidence_artifacts": ["measurements.json"],
            "reason": "qualified_automatic_realized_singleton_c1_b3",
            "selected_candidate_count": 3,
        },
    )
    assert measured == "MTP enabled, candidate budget 3"


def test_startup_speculation_line_reports_the_resolved_depth_over_the_configured_one() -> None:
    """The selected depth wins: a row may authorize less than the default."""

    config = _banner_config(speculative_mtp_serving="auto")
    text = _banner_text(
        config,
        serving_route=True,
        plan={
            "admitted": True,
            "automatic_eligible": True,
            "evidence_key": "measured-cell",
            "evidence_fingerprint": "sha256:measured-cell",
            "evidence_artifacts": ["measurements.json"],
            "reason": "qualified_automatic_realized_singleton_c1_b2",
            "selected_candidate_count": 2,
        },
    )
    assert text == "MTP enabled, candidate budget 2"


def test_startup_speculation_line_states_the_policy_when_no_route_is_configured() -> None:
    config = _banner_config(speculative_mtp_serving="off")
    assert _banner_text(config, serving_route=False, plan=None) == (
        "MTP unavailable (policy off)"
    )


def test_startup_speculation_line_opt_in_policy_requires_explicit_requests() -> None:
    plan = resolve_speculative_mtp_serving_plan((_evidence(),), key=_key()).as_dict()
    assert plan["admitted"] and plan["automatic_eligible"]
    text = _banner_text(
        _banner_config(speculative_mtp_serving="opt_in"), serving_route=True, plan=plan,
    )
    assert text == "MTP explicit-only (default AR; policy opt_in), candidate budget 3"


def test_startup_speculation_line_reports_an_unresolved_plan() -> None:
    """An engine that exposes support but resolves no plan is not enabled."""

    config = _banner_config(speculative_mtp_serving="auto")
    assert _banner_text(config, serving_route=True, plan=None) == (
        "MTP unavailable (unresolved)"
    )


def test_startup_speculation_line_reports_an_unresolved_depth() -> None:
    """A plan that admits without a depth states that rather than "None"."""

    config = _banner_config(speculative_mtp_serving="auto")
    text = _banner_text(
        config,
        serving_route=True,
        plan={
            "admitted": True, "automatic_eligible": True,
            "reason": "implemented_x", "selected_candidate_count": None,
        },
        budget={"requested": None, "resolved": None, "source": "unresolved"},
    )
    assert text == "MTP enabled, candidate budget unresolved (unmeasured)"


@pytest.mark.parametrize("policy", ["auto", "enabled"])
def test_startup_speculation_line_reports_explicit_only_evidence(policy) -> None:
    row = next(
        row for row in Qwen35GGUFModel().speculative_mtp_serving_evidence
        if "measured_slower_than_ar" in row.reason
    )
    plan = resolve_speculative_mtp_serving_plan((row,), key=_row_key(row)).as_dict()
    assert plan["admitted"] and not plan["automatic_eligible"]
    text = _banner_text(
        _banner_config(speculative_mtp_serving=policy), serving_route=True, plan=plan,
    )
    assert text == (
        f"MTP explicit-only (default AR; {row.reason}), "
        f"candidate budget {row.candidate_budget}"
    )


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_startup_speculation_line_recognizes_sampled_evidence(width) -> None:
    row = next(
        row for row in Qwen35GGUFModel().speculative_mtp_serving_evidence
        if row.evidence_key == f"qwen38-q4km-gfx1151-native-sampled-c{width}-k3"
    )
    plan = resolve_speculative_mtp_serving_plan((row,), key=_row_key(row)).as_dict()
    text = _banner_text(
        _banner_config(speculative_mtp_serving="auto"), serving_route=True, plan=plan,
    )
    assert text == "MTP enabled, candidate budget 3"


@pytest.mark.parametrize("reason", ["qualified_but_no_evidence", "implemented_chain"])
@pytest.mark.parametrize("automatic", [True, False, None])
def test_startup_speculation_line_does_not_infer_policy_or_evidence_from_reason(
    reason, automatic,
) -> None:
    plan = {"admitted": True, "reason": reason, "selected_candidate_count": 3}
    if automatic is not None:
        plan["automatic_eligible"] = automatic
    text = _banner_text(
        _banner_config(speculative_mtp_serving="auto"), serving_route=True, plan=plan,
    )
    status = "MTP enabled" if automatic else f"MTP explicit-only (default AR; {reason})"
    assert text == f"{status}, candidate budget 3 (unmeasured)"


def test_pretty_startup_summary_reads_the_resolved_speculation_route(monkeypatch) -> None:
    """The banner call site is wired to the resolved plan, not the policy."""

    messages: list[str] = []
    monkeypatch.setattr(
        "hipengine.server.api._LOGGER.info",
        lambda message, *args: messages.append(message % args),
    )
    config = _banner_config(speculative_mtp_serving="auto")
    engine = SimpleNamespace(
        speculative_mtp_serving_capability={
            "admitted": False,
            "reason": "mtp_backend_unsupported",
        },
        # Required for the route to read as configured at all, which is what
        # makes the resolved plan rather than the policy decide the line.
        generate_speculative_mtp_detailed=lambda *args, **kwargs: None,
        live_loop_snapshot=lambda: {},
    )
    _log_pretty_startup_summary(config, engine=engine, memory=None)

    rendered = messages[-1]
    assert "MTP unavailable (mtp_backend_unsupported)" in rendered
    assert "MTP enabled" not in rendered
