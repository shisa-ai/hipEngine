from __future__ import annotations

import pytest

from hipengine.speculative.policy import (
    DEFAULT_AUTO_DEPTH_POLICY,
    select_offline_speculative_depth,
)
from hipengine.speculative import (
    ProviderAttachment,
    ProviderCatchupMode,
    SpecK0Class,
    SpecPlanReason,
    SpecTransactionMode,
    SpeculativeCapability,
    SpeculativeRequestSemantics,
    plan_speculative_requests,
)


def _capability(**overrides) -> SpeculativeCapability:
    values = {
        "capability_key": "mtp2:qwen38:gfx1151:strict",
        "target_key": "qwen38_27b_q4ks",
        "provider_key": "qwen38_nextn",
        "method_key": "mtp2",
        "policy_fingerprint": "policy:v1",
        "execution_profile": "strict",
        "kv_backend_key": "paged_bf16",
        "attachment": ProviderAttachment.TARGET_ATTACHED,
        "catchup_mode": ProviderCatchupMode.TARGET_OUTPUT,
        "supported_modes": ("verify_chain",),
        "supported_sampling_modes": ("greedy",),
        "max_requests": 8,
        "max_candidates_per_request": 3,
        "max_frontier_rows": 32,
        "proposal_widths": (1, 2, 4, 8),
        "target_row_buckets": (2, 4, 8, 16),
        "target_transaction_mode": SpecTransactionMode.PACKED_SCRATCH,
        "provider_transaction_mode": SpecTransactionMode.REVERSIBLE_JOURNAL,
        "graph_supported": True,
        "eager_supported": True,
        "strict_fallback_key": "target_ar_strict",
        "max_context_tokens": 4096,
    }
    values.update(overrides)
    return SpeculativeCapability(**values)


def _semantics(
    request_id: int,
    *,
    sampling_mode: str = "greedy",
    context_tokens: int = 128,
    remaining_decode: int = 8,
) -> SpeculativeRequestSemantics:
    return SpeculativeRequestSemantics(
        request_id=request_id,
        sampling_mode=sampling_mode,
        mode="verify_chain",
        context_tokens=context_tokens,
        remaining_decode=remaining_decode,
    )


def _plan(capability, semantics, desired, **kwargs):
    return plan_speculative_requests(
        capability,
        semantics,
        resident_slots=tuple(reversed(range(len(semantics)))),
        desired_candidate_counts=desired,
        operation_id="policy-cycle:1",
        cycle_id=1,
        context_bucket_size=256,
        **kwargs,
    )


def test_missing_capability_selects_k0_before_provider_ownership() -> None:
    plan = _plan(None, (_semantics(1), _semantics(2)), (3, 3))

    assert plan.is_ar_only
    assert plan.reasons == (SpecPlanReason.NO_PROVIDER, SpecPlanReason.NO_PROVIDER)
    assert plan.provider_key is None
    assert plan.provider_transaction_mode is None
    assert plan.execution_route == "ar"


def test_plan_preserves_declared_logical_width_across_physical_subset() -> None:
    plan = _plan(
        _capability(),
        tuple(_semantics(index) for index in range(4)),
        (3, 3, 3, 3),
        declared_logical_c=8,
    )

    assert plan.declared_logical_c == 8
    assert len(plan.request_ids) == 4


def test_unsupported_sampling_selects_per_request_k0_in_mixed_plan() -> None:
    semantics = (_semantics(1), _semantics(2, sampling_mode="top_p"))

    plan = _plan(_capability(), semantics, (3, 3))

    assert plan.candidate_counts == (3, 0)
    assert plan.reasons == (
        SpecPlanReason.SPECULATIVE_QUALIFIED,
        SpecPlanReason.UNSUPPORTED_SAMPLING,
    )
    assert plan.speculative_request_ids == (1,)
    assert plan.proposal_widths == (1,)
    assert plan.logical_frontier_rows == 5


def test_context_and_output_room_misses_select_k0_without_mutation() -> None:
    capability = _capability(max_context_tokens=132)
    semantics = (
        _semantics(1, context_tokens=131, remaining_decode=8),
        _semantics(2, context_tokens=128, remaining_decode=1),
    )

    plan = _plan(capability, semantics, (3, 3))

    assert plan.candidate_counts == (0, 0)
    assert plan.reasons == (
        SpecPlanReason.TARGET_GRAPH_CONTEXT_BUCKET_MISS,
        SpecPlanReason.TARGET_GRAPH_OUTPUT_ROOM_MISS,
    )
    assert plan.is_ar_only


def test_terminal_zero_accept_capability_keeps_one_bounded_candidate() -> None:
    plan = _plan(
        _capability(
            max_requests=1,
            proposal_widths=(1,),
            terminal_zero_accept_supported=True,
        ),
        (_semantics(1, remaining_decode=1),),
        (3,),
    )

    assert plan.candidate_counts == (1,)
    assert plan.reasons == (SpecPlanReason.SPECULATIVE_QUALIFIED,)
    assert plan.logical_frontier_rows == 2


def test_claim_miss_and_circuit_breaker_select_stable_k0_reasons() -> None:
    semantics = (_semantics(1), _semantics(2))

    claim_miss = _plan(_capability(), semantics, (2, 2), claims_fit=False)
    assert claim_miss.candidate_counts == (0, 0)
    assert claim_miss.reasons == (
        SpecPlanReason.RESOURCE_CLAIM_MISS,
        SpecPlanReason.RESOURCE_CLAIM_MISS,
    )

    breaker = _plan(_capability(), semantics, (2, 2), circuit_breaker_open=True)
    assert breaker.candidate_counts == (0, 0)
    assert breaker.reasons == (
        SpecPlanReason.CIRCUIT_BREAKER_OPEN,
        SpecPlanReason.CIRCUIT_BREAKER_OPEN,
    )


def test_graph_miss_uses_qualified_eager_or_k0_when_no_route_exists() -> None:
    semantics = (_semantics(1),)

    eager = _plan(_capability(), semantics, (2,), graph_available=False)
    assert eager.candidate_counts == (2,)
    assert eager.execution_route == "eager"

    graph_only = _capability(eager_supported=False)
    k0 = _plan(graph_only, semantics, (2,), graph_available=False)
    assert k0.candidate_counts == (0,)
    assert k0.reasons == (SpecPlanReason.TARGET_PHYSICAL_BUCKET_MISS,)


@pytest.mark.parametrize(
    ("concurrency", "cell_key", "reason"),
    [
        (1, "auto-c1-product-pending-k0", "product_qualification_pending"),
        (2, "auto-c2-measured-k0", "measured_speedup_below_1p10"),
        (4, "auto-c4-measured-k0", "measured_speedup_below_1p10"),
        (8, "auto-c5-c8-unqualified-k0", "no_qualified_physical_frontier"),
        (17, "auto-c9-c17-unqualified-k0", "no_qualified_physical_frontier"),
        (32, "auto-c18-c32-unqualified-k0", "no_qualified_physical_frontier"),
    ],
)
def test_default_offline_depth_policy_selects_k0_with_stable_cell_reason(
    concurrency: int,
    cell_key: str,
    reason: str,
) -> None:
    decision = select_offline_speculative_depth(
        DEFAULT_AUTO_DEPTH_POLICY,
        concurrency=concurrency,
        output_horizon_tokens=24,
    )

    assert decision.selected_k == 0
    assert decision.cell_key == cell_key
    assert decision.reason == reason
    assert decision.policy_fingerprint.startswith("sha256:")
    assert DEFAULT_AUTO_DEPTH_POLICY.policy_key == (
        "specdec2:auto:qwen38-q4ks:production:p9-fixed-reseed:v4"
    )
    assert decision.evidence == (
        "benchmarks/results/2026-08-26-gfx1151-specdec2-perf-p9-fixed-policy.json"
    )


def test_offline_depth_policy_fails_closed_outside_qualified_concurrency() -> None:
    decision = select_offline_speculative_depth(
        DEFAULT_AUTO_DEPTH_POLICY,
        concurrency=33,
        output_horizon_tokens=24,
    )

    assert decision.selected_k == 0
    assert decision.cell_key == "auto-outside-qualified-concurrency-k0"
    assert decision.reason == "outside_qualified_concurrency"


def test_policy_caps_k_and_decomposes_c8_deterministically() -> None:
    counts = (0, 1, 2, 3, 0, 1, 2, 3)
    semantics = tuple(_semantics(100 + index) for index in range(8))

    first = _plan(_capability(), semantics, counts)
    second = _plan(_capability(), semantics, counts)

    assert first == second
    assert first.candidate_counts == counts
    assert first.proposal_widths == (4, 2)
    assert first.target_row_decomposition == (16, 4)
    assert first.logical_frontier_rows == 20
    assert first.execution_route == "graph"


def test_policy_reports_requested_depth_alongside_admitted_cap() -> None:
    """A bounded request must expose both its requested and admitted depth.

    The campaign contract: min(requested, qualified) admission may cap a
    request, but the cap must be visible and cannot silently masquerade
    as the requested depth.
    """

    semantics = (_semantics(1, remaining_decode=64),)
    # Requested K7; the capability qualifies at most 3.
    plan = _plan(_capability(max_candidates_per_request=3), semantics, (7,))

    assert plan.candidate_counts == (3,)
    assert plan.requested_candidate_counts == (7,)

    # Unbounded admission reports requested == admitted.
    plan = _plan(_capability(max_candidates_per_request=8), semantics, (7,))
    assert plan.candidate_counts == (7,)
    assert plan.requested_candidate_counts == (7,)

    # Suppression zeroes the admitted depth but preserves the request.
    plan = _plan(
        _capability(max_candidates_per_request=8),
        semantics,
        (7,),
        suppress_speculation=(True,),
    )
    assert plan.candidate_counts == (0,)
    assert plan.requested_candidate_counts == (7,)

    # A defaulted plan (legacy constructor path) reports requested == admitted.
    from hipengine.speculative import SpecRequestPlan

    legacy = SpecRequestPlan(
        operation_id="legacy:1",
        cycle_id=1,
        request_ids=(1,),
        resident_slots=(0,),
        candidate_counts=(0,),
        reasons=(SpecPlanReason.POLICY_SELECTED_AR,),
        k0_classes=(SpecK0Class.PURE,),
        mode="decode",
        capability_key=None,
        provider_key=None,
        target_transaction_mode=SpecTransactionMode.PACKED_SCRATCH,
        provider_transaction_mode=None,
        proposal_widths=(),
        target_row_decomposition=(1,),
        context_bucket_size=256,
        execution_route="ar",
    )
    assert legacy.requested_candidate_counts == (0,)


def test_policy_rejects_misaligned_or_negative_desired_depth() -> None:
    semantics = (_semantics(1), _semantics(2))
    with pytest.raises(ValueError, match="desired_candidate_counts"):
        _plan(_capability(), semantics, (2,))
    with pytest.raises(ValueError, match="desired_candidate_counts"):
        _plan(_capability(), semantics, (2, -1))
    with pytest.raises(ValueError, match="resident_slots"):
        plan_speculative_requests(
            _capability(),
            semantics,
            resident_slots=(0, 0),
            desired_candidate_counts=(2, 2),
            operation_id="bad-slots",
            cycle_id=1,
            context_bucket_size=256,
        )


def test_suppressed_request_plans_k0_transitional_for_catchup_repair() -> None:
    semantics = (_semantics(1), _semantics(2))
    plan = _plan(
        _capability(),
        semantics,
        (2, 2),
        suppress_speculation=(True, False),
    )
    assert plan.candidate_counts == (0, 2)
    assert plan.reasons[0] is SpecPlanReason.POLICY_SELECTED_AR
    assert plan.reasons[1] is SpecPlanReason.SPECULATIVE_QUALIFIED
    # desired stayed positive for the suppressed request, so its K0 class is
    # TRANSITIONAL: prepare_k0 must run the target-hidden catchup advance.
    assert plan.k0_classes[0] is SpecK0Class.TRANSITIONAL
    assert plan.k0_classes[1] is SpecK0Class.NOT_K0


def test_suppress_speculation_must_align_with_semantics() -> None:
    semantics = (_semantics(1), _semantics(2))
    with pytest.raises(ValueError, match="suppress_speculation"):
        _plan(_capability(), semantics, (2, 2), suppress_speculation=(True,))
