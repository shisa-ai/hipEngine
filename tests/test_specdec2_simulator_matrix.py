from __future__ import annotations

from dataclasses import replace

import pytest

from hipengine.generation.concurrency2_simulator import SimulatedResourceLedger
from hipengine.kvcache import ClaimLifetime, ResourceClaimSet
from hipengine.speculative import (
    DraftBatch,
    SpecCycleStage,
    SpecPlanReason,
    SpeculativeCycleSimulator,
    SpeculativeRequestState,
)
from tests.test_speculative_cycle_simulator import _claims, _draft, _plan, _states


def _matrix_states(candidate_counts: tuple[int, ...]) -> tuple[SpeculativeRequestState, ...]:
    width = len(candidate_counts)
    slots = tuple(reversed(range(width)))
    return tuple(
        SpeculativeRequestState(
            method_key="mtp2",
            provider_key="nextn",
            policy_fingerprint="policy:v2",
            target_request_id=100 + index,
            resident_slot=slots[index],
            target_cursor=1000 + 10 * index,
            provider_cursor=2000 + 10 * index,
            provider_state_lease=f"provider:{100 + index}",
            output_limit=64,
        )
        for index in range(width)
    )


def _matrix_draft(
    candidate_counts: tuple[int, ...],
    states: tuple[SpeculativeRequestState, ...],
    *,
    cycle_id: int = 1,
) -> DraftBatch:
    request_ids = tuple(state.target_request_id for state in states)
    tokens: list[int] = []
    positions: list[int] = []
    depths: list[int] = []
    owners: list[int] = []
    parents: list[int] = []
    slots: list[int] = []
    offset = 0
    for state, count in zip(states, candidate_counts, strict=True):
        for depth in range(1, count + 1):
            tokens.append(state.target_request_id * 10 + depth)
            positions.append(state.target_cursor + depth - 1)
            depths.append(depth)
            owners.append(state.target_request_id)
            parents.append(-1 if depth == 1 else offset + depth - 2)
            slots.append(state.resident_slot)
        offset += count
    return DraftBatch(
        request_ids=request_ids,
        candidate_tokens=tuple(tokens),
        parent_positions=tuple(positions),
        draft_depths=tuple(depths),
        row_to_request=tuple(owners),
        tree_parents=tuple(parents),
        active_mask=(True,) * len(tokens),
        mode="verify_chain",
        cycle_id=cycle_id,
        resident_slots=tuple(slots),
    )


@pytest.mark.parametrize(
    "candidate_counts",
    [
        (3,),
        (0, 1),
        (0, 1, 2, 3),
        (0, 1, 2, 3, 0, 1, 2, 3),
    ],
)
def test_simulator_covers_c1_c2_c4_c8_with_mixed_k0_to_k3(candidate_counts) -> None:
    states = _matrix_states(candidate_counts)
    simulator = SpeculativeCycleSimulator(SimulatedResourceLedger(_plan()), states)
    draft = _matrix_draft(candidate_counts, states)
    accepted = tuple(count // 2 for count in candidate_counts)
    corrections = tuple(9000 + index for index in range(len(states)))

    result = simulator.run_cycle(
        draft,
        component_claims=_claims(f"matrix:c{len(states)}", rows=32, results=8),
        accepted_counts=accepted,
        correction_or_bonus_tokens=corrections,
    )

    assert result.stage is SpecCycleStage.COMMITTED
    assert result.telemetry is not None
    assert result.telemetry.candidate_counts == candidate_counts
    assert result.telemetry.logical_request_count == len(states)
    assert result.telemetry.logical_frontier_rows == len(states) + sum(candidate_counts)
    assert result.transaction.provider_request_ids == tuple(
        state.target_request_id
        for state, count in zip(states, candidate_counts, strict=True)
        if count > 0
    )
    for state, count, accepted_count in zip(
        states, candidate_counts, accepted, strict=True
    ):
        updated = simulator.state(state.target_request_id)
        assert updated.target_cursor == state.target_cursor + accepted_count + 1
        assert updated.provider_cursor == state.provider_cursor + accepted_count
        if count == 0:
            assert updated.provider_cursor == state.provider_cursor
    simulator.assert_conserved()


def test_simulator_runs_target_only_k0_without_provider_claim_or_cursor_update() -> None:
    states = _matrix_states((0, 0, 0, 0))
    simulator = SpeculativeCycleSimulator(SimulatedResourceLedger(_plan()), states)
    claims = {
        "target": ResourceClaimSet.from_mapping(
            "k0:target",
            {"target.txn": 4},
            lifetime=ClaimLifetime.TRANSACTION,
        ),
        "transient": ResourceClaimSet.from_mapping(
            "k0:transient",
            {"spec.results": 4},
            lifetime=ClaimLifetime.WORK_ITEM,
        ),
    }

    result = simulator.run_k0_cycle(
        tuple(state.target_request_id for state in states),
        component_claims=claims,
        output_tokens=(41, 42, 43, 44),
        reasons=(
            SpecPlanReason.NO_PROVIDER,
            SpecPlanReason.POLICY_SELECTED_AR,
            SpecPlanReason.UNSUPPORTED_SAMPLING,
            SpecPlanReason.TARGET_PHYSICAL_BUCKET_MISS,
        ),
    )

    assert result.stage is SpecCycleStage.COMMITTED
    assert not result.transaction.has_provider
    assert result.telemetry is not None
    assert result.telemetry.execution_route == "ar"
    assert result.telemetry.candidate_counts == (0, 0, 0, 0)
    for state, token in zip(states, (41, 42, 43, 44), strict=True):
        updated = simulator.state(state.target_request_id)
        assert updated.target_cursor == state.target_cursor + 1
        assert updated.provider_cursor == state.provider_cursor
        assert updated.visible_tokens == (token,)
    simulator.assert_conserved()


@pytest.mark.parametrize(
    "stage",
    [
        SpecCycleStage.RESERVED,
        SpecCycleStage.TARGET_OPEN,
        SpecCycleStage.PROVIDER_OPEN,
        SpecCycleStage.DRAFTED,
        SpecCycleStage.VERIFIED,
        SpecCycleStage.ACCEPTED,
        SpecCycleStage.READBACK,
        SpecCycleStage.COMMITTING,
    ],
)
def test_injected_failure_rolls_back_every_precommit_stage(stage) -> None:
    simulator = SpeculativeCycleSimulator(SimulatedResourceLedger(_plan()), _states())

    result = simulator.run_cycle(
        _draft(tree=True),
        component_claims=_claims(f"failure:{stage.value}"),
        accepted_counts=(1, 2),
        correction_or_bonus_tokens=(901, 902),
        fail_at=stage,
        failure_message=f"failed at {stage.value}",
    )

    assert result.stage is SpecCycleStage.FAILED
    assert result.error == f"failed at {stage.value}"
    assert result.transaction.rolled_back
    assert simulator.state(10).target_cursor == simulator.state(10).provider_cursor == 100
    assert simulator.state(20).target_cursor == simulator.state(20).provider_cursor == 200
    simulator.assert_conserved()


@pytest.mark.parametrize("stage", [SpecCycleStage.READBACK, SpecCycleStage.COMMITTING])
def test_late_cancellation_rolls_back_before_visible_commit(stage) -> None:
    simulator = SpeculativeCycleSimulator(SimulatedResourceLedger(_plan()), _states())

    result = simulator.run_cycle(
        _draft(),
        component_claims=_claims(f"cancel:{stage.value}"),
        accepted_counts=(3, 2),
        correction_or_bonus_tokens=(901, 902),
        cancel_at=stage,
        cancel_request_id=10,
    )

    assert result.stage is SpecCycleStage.CANCELLED
    assert simulator.state(10).visible_tokens == ()
    assert simulator.state(20).visible_tokens == ()
    assert simulator.state(20).finished is False
    simulator.assert_conserved()


def test_permuted_slots_reclaim_and_refill_preserve_survivor_ownership() -> None:
    first, survivor = _states()
    first = replace(first, output_limit=1)
    simulator = SpeculativeCycleSimulator(
        SimulatedResourceLedger(_plan()),
        (first, survivor),
    )
    k0_claims = {
        "target": ResourceClaimSet.from_mapping(
            "refill:target",
            {"target.txn": 2},
            lifetime=ClaimLifetime.TRANSACTION,
        )
    }
    simulator.run_k0_cycle(
        (10, 20),
        component_claims=k0_claims,
        output_tokens=(51, 52),
    )
    assert simulator.state(10).finished
    assert simulator.reclaim_finished(10).resident_slot == 2

    refill = SpeculativeRequestState(
        method_key="mtp2",
        provider_key="nextn",
        policy_fingerprint="policy:v1",
        target_request_id=30,
        resident_slot=2,
        target_cursor=300,
        provider_cursor=300,
        provider_state_lease="provider:30",
        cycle_id=1,
        output_limit=8,
    )
    simulator.admit_states((refill,))
    joined_states = (simulator.state(20), simulator.state(30))
    draft = _matrix_draft((0, 1), joined_states, cycle_id=2)
    result = simulator.run_cycle(
        draft,
        component_claims=_claims("refill:joined"),
        accepted_counts=(0, 1),
        correction_or_bonus_tokens=(61, 62),
    )

    assert result.stage is SpecCycleStage.COMMITTED
    assert simulator.state(20).resident_slot == 5
    assert simulator.state(20).visible_tokens == (52, 61)
    assert simulator.state(30).resident_slot == 2
    assert simulator.state(30).visible_tokens[-2:] == (301, 62)
    simulator.assert_conserved()
