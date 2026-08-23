from __future__ import annotations

import pytest

from hipengine.kvcache import ClaimLifetime, KVPoolPlan, KVPoolSpec, ResourceClaimSet
from hipengine.speculative import DraftBatch
from hipengine.speculative.simulator import (
    SpecCycleStage,
    SpeculativeCycleSimulator,
    SpeculativeRequestState,
    compose_speculative_claims,
)
from hipengine.generation.concurrency2_simulator import (
    SimulatedResourceCapacityError,
    SimulatedResourceLedger,
)


def _plan() -> KVPoolPlan:
    return KVPoolPlan(
        backend_fingerprint="spec-c0-fixture",
        generation=1,
        pools=(
            KVPoolSpec("target.txn", 64, lifetimes=(ClaimLifetime.TRANSACTION,)),
            KVPoolSpec("provider.kv", 64, lifetimes=(ClaimLifetime.TRANSACTION,)),
            KVPoolSpec("spec.rows", 64, lifetimes=(ClaimLifetime.WORK_ITEM,)),
            KVPoolSpec("spec.results", 16, lifetimes=(ClaimLifetime.WORK_ITEM,)),
        ),
    )


def _claims(prefix: str, *, target: int = 8, provider: int = 8, rows: int = 8, results: int = 2):
    return {
        "target": ResourceClaimSet.from_mapping(
            f"{prefix}:target", {"target.txn": target}, lifetime=ClaimLifetime.TRANSACTION
        ),
        "provider": ResourceClaimSet.from_mapping(
            f"{prefix}:provider", {"provider.kv": provider}, lifetime=ClaimLifetime.TRANSACTION
        ),
        "transient": ResourceClaimSet.from_mapping(
            f"{prefix}:transient",
            {"spec.rows": rows, "spec.results": results},
            lifetime=ClaimLifetime.WORK_ITEM,
        ),
    }


def _states() -> tuple[SpeculativeRequestState, ...]:
    return (
        SpeculativeRequestState(
            method_key="mtp_chain",
            provider_key="nextn",
            policy_fingerprint="policy:v1",
            target_request_id=10,
            resident_slot=2,
            target_cursor=100,
            provider_cursor=100,
            provider_state_lease="provider:10",
            output_limit=8,
        ),
        SpeculativeRequestState(
            method_key="mtp_chain",
            provider_key="nextn",
            policy_fingerprint="policy:v1",
            target_request_id=20,
            resident_slot=5,
            target_cursor=200,
            provider_cursor=200,
            provider_state_lease="provider:20",
            output_limit=8,
        ),
    )


def _draft(*, tree: bool = False) -> DraftBatch:
    return DraftBatch(
        request_ids=(10, 20),
        candidate_tokens=(101, 102, 103, 201, 202),
        parent_positions=(100, 101, 102, 200, 201),
        draft_depths=(1, 2, 3, 1, 2),
        row_to_request=(10, 10, 10, 20, 20),
        tree_parents=(-1, 0, 1, -1, 3),
        active_mask=(True,) * 5,
        mode="verify_tree" if tree else "verify_chain",
        cycle_id=1,
        resident_slots=(2, 2, 2, 5, 5),
        candidate_ids=(0, 1, 2, 0, 1),
        provider_metadata=(("candidate_budget", 3),),
    )


def test_spec_c0_draft_batch_carries_one_to_many_device_materializable_maps() -> None:
    draft = _draft(tree=True)

    assert draft.verifier_request_ids == (10, 10, 10, 20, 20)
    assert draft.resident_slots == (2, 2, 2, 5, 5)
    assert draft.candidate_ids == (0, 1, 2, 0, 1)
    assert draft.provider_metadata_dict() == {"candidate_budget": 3}
    assert draft.draft_rows == 5

    with pytest.raises(ValueError, match="resident_slots"):
        DraftBatch(
            request_ids=(10,), candidate_tokens=(1,), parent_positions=(0,),
            draft_depths=(1,), row_to_request=(10,), resident_slots=(2, 3),
        )


def test_spec_c0_claim_composition_is_atomic_across_provider_target_and_transients() -> None:
    claims = compose_speculative_claims("spec:1", _claims("spec:1"))
    assert claims.units_by_pool() == {
        "provider.kv": 8,
        "spec.results": 2,
        "spec.rows": 8,
        "target.txn": 8,
    }
    assert claims.metadata_dict() == {
        "component_count": 3,
        "components": "provider,target,transient",
    }

    ledger = SimulatedResourceLedger(_plan())
    oversized = _claims("oversized", provider=65)
    with pytest.raises(SimulatedResourceCapacityError, match="provider.kv"):
        ledger.reserve("spec-cycle:oversized", compose_speculative_claims("oversized", oversized))
    ledger.assert_conserved()
    assert all(row["used"] == 0 for row in ledger.snapshot().values())


@pytest.mark.parametrize(
    ("accepted", "corrections", "expected_visible", "expected_deltas"),
    [
        ((0, 0), (901, 902), ((901,), (902,)), (1, 1)),
        ((1, 1), (911, 912), ((101, 911), (201, 912)), (2, 2)),
        ((3, 2), (921, 922), ((101, 102, 103, 921), (201, 202, 922)), (4, 3)),
    ],
)
def test_spec_c0_reject_partial_full_transactions_commit_exact_visible_ranges(
    accepted, corrections, expected_visible, expected_deltas
) -> None:
    ledger = SimulatedResourceLedger(_plan())
    simulator = SpeculativeCycleSimulator(ledger, _states())

    result = simulator.run_cycle(
        _draft(),
        component_claims=_claims("cycle:1"),
        accepted_counts=accepted,
        correction_or_bonus_tokens=corrections,
    )

    assert result.stage is SpecCycleStage.COMMITTED
    assert result.accept_result.accepted_counts == accepted
    assert result.accept_result.correction_or_bonus_tokens == corrections
    assert result.accept_result.target_cursor_deltas == expected_deltas
    assert result.accept_result.provider_cursor_deltas == accepted
    assert tuple(simulator.state(rid).visible_tokens for rid in (10, 20)) == expected_visible
    assert tuple(simulator.state(rid).target_cursor for rid in (10, 20)) == tuple(
        base + delta for base, delta in zip((100, 200), expected_deltas, strict=True)
    )
    assert tuple(simulator.state(rid).provider_cursor for rid in (10, 20)) == tuple(
        base + delta for base, delta in zip((100, 200), accepted, strict=True)
    )
    simulator.assert_conserved()
    assert all(row["used"] == 0 for row in ledger.snapshot().values())


@pytest.mark.parametrize(
    "stage",
    [
        SpecCycleStage.RESERVED,
        SpecCycleStage.TARGET_OPEN,
        SpecCycleStage.PROVIDER_OPEN,
        SpecCycleStage.DRAFTED,
        SpecCycleStage.VERIFIED,
        SpecCycleStage.ACCEPTED,
    ],
)
def test_spec_c0_cancellation_at_every_stage_rolls_back_both_transactions_and_claims(stage) -> None:
    ledger = SimulatedResourceLedger(_plan())
    simulator = SpeculativeCycleSimulator(ledger, _states())

    result = simulator.run_cycle(
        _draft(tree=True),
        component_claims=_claims(f"cancel:{stage.value}"),
        accepted_counts=(1, 2),
        correction_or_bonus_tokens=(901, 902),
        cancel_at=stage,
        cancel_request_id=10,
    )

    assert result.stage is SpecCycleStage.CANCELLED
    assert result.transaction.rolled_back is True
    assert result.transaction.target_committed is False
    assert result.transaction.provider_committed is False
    cancelled = simulator.state(10)
    peer = simulator.state(20)
    assert cancelled.cancelled and cancelled.finished
    assert cancelled.target_cursor == cancelled.provider_cursor == 100
    assert peer.target_cursor == peer.provider_cursor == 200
    assert peer.cancelled is False and peer.finished is False
    assert cancelled.pending_transaction_id is None
    assert peer.pending_transaction_id is None
    simulator.assert_conserved()
    assert all(row["used"] == 0 for row in ledger.snapshot().values())
