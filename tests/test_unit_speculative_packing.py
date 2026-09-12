from __future__ import annotations

import pytest

from hipengine.speculative.packing import (
    SpeculativePackingBudget,
    SpeculativePackingRequest,
    VerifierCostMap,
    VerifierCostRecord,
    pack_speculative_requests,
)


def _request(
    request_id: int,
    *,
    provider: str = "nextn",
    rows: int = 4,
    transaction_bytes: int = 1024,
    deadline: float | None = None,
) -> SpeculativePackingRequest:
    return SpeculativePackingRequest(
        request_id=request_id,
        target_key="qwen38-q4km",
        provider_key=provider,
        policy_fingerprint="greedy-v1",
        mode="verify_chain",
        context_bucket=768,
        transaction_mode="journal",
        execution_profile="strict",
        candidate_rows=rows,
        transaction_bytes=transaction_bytes,
        deadline_at=deadline,
    )


def test_verifier_cost_record_rejects_ar_d2_identity_and_prices_c_v_shape() -> None:
    record = VerifierCostRecord(
        target_key="qwen38-q4km",
        provider_key="nextn",
        mode="verify_chain",
        request_count=2,
        verifier_rows=8,
        tree_shape=(0, 1, 2, 0, 1, 2),
        context_bucket=768,
        transaction_mode="journal",
        execution_profile="strict",
        predicted_microseconds=2400.0,
        source="fixture:verify-chain-c2-v8",
    )
    assert record.identity[0] == "verifier_cost_v1"
    assert "ar_d2" not in repr(record.identity).lower()
    table = VerifierCostMap((record,))
    assert table.resolve(record.identity) is record
    with pytest.raises(KeyError, match="verifier cost"):
        table.resolve((*record.identity[:-2], "production", record.identity[-1]))
    with pytest.raises(ValueError, match="verifier"):
        VerifierCostRecord(
            target_key="qwen38-q4km", provider_key="nextn", mode="decode",
            request_count=2, verifier_rows=2, tree_shape=(), context_bucket=768,
            transaction_mode="journal", execution_profile="strict",
            predicted_microseconds=1.0, source="ar_d2:c2",
        )


def test_speculative_packer_groups_only_identical_compatibility_and_bounds() -> None:
    requests = (
        _request(1), _request(2), _request(3, provider="dflash"), _request(4),
    )
    budget = SpeculativePackingBudget(
        max_draft_rows_per_round=12,
        max_verify_rows_per_round=12,
        max_speculative_cycles_per_round=3,
        max_spec_transaction_bytes=4096,
        max_spec_work_items_per_round=3,
    )

    plan = pack_speculative_requests(requests, budget=budget)

    assert tuple(group.request_ids for group in plan.groups) == ((1, 2), (3,))
    assert plan.deferred_request_ids == (4,)
    assert plan.charged_draft_rows == 12
    assert plan.charged_verify_rows == 12
    assert plan.charged_transaction_bytes == 3072
    assert all(len(set(group.compatibility_keys)) == 1 for group in plan.groups)


def test_speculative_packer_fairness_one_cycle_before_repeat_and_refill() -> None:
    budget = SpeculativePackingBudget(
        max_draft_rows_per_round=8,
        max_verify_rows_per_round=8,
        max_speculative_cycles_per_round=2,
        max_spec_transaction_bytes=4096,
        max_spec_work_items_per_round=2,
    )
    first = pack_speculative_requests((_request(10), _request(20), _request(30)), budget=budget)
    assert tuple(group.request_ids for group in first.groups) == ((10, 20),)
    assert first.deferred_request_ids == (30,)

    # Request 10 completes; the next fairness pass admits waiting 30 before 20
    # can consume a second speculative cycle.
    second = pack_speculative_requests(
        (_request(30), _request(20)), budget=budget,
        cycles_already_served={20: 1, 30: 0},
    )
    assert tuple(group.request_ids for group in second.groups) == ((30, 20),)


def test_speculative_packer_mixed_ar_pressure_and_slo_fallback_are_prelaunch() -> None:
    budget = SpeculativePackingBudget(
        max_draft_rows_per_round=4,
        max_verify_rows_per_round=4,
        max_speculative_cycles_per_round=1,
        max_spec_transaction_bytes=1024,
        max_spec_work_items_per_round=1,
        deadline_guard_seconds=0.050,
    )
    plan = pack_speculative_requests(
        (_request(1, transaction_bytes=2048), _request(2, deadline=10.02), _request(3)),
        budget=budget,
        now=10.0,
        ar_due_request_ids=(100, 101),
    )

    assert tuple(group.request_ids for group in plan.groups) == ((3,),)
    assert plan.ar_fallbacks == {
        1: "spec_transaction_budget",
        2: "deadline_guard",
    }
    assert plan.deferred_request_ids == ()
    assert plan.ar_due_request_ids == (100, 101)
    assert plan.provisional_mutations == 0
