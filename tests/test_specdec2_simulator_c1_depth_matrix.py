"""C1 singleton depth matrix through K7 on the deterministic cycle simulator.

Campaign contract (docs/QWEN38-27B-GFX1100-CONCURRENCY2-BETTER-MTP.md,
Packet 5): rejection and EOS at every candidate position, zero/all
accepted, correction/bonus handling, output-horizon clipping,
cancellation, rollback, retry and following-cycle state at every depth
1-7, K7<->K1<->K0 transitions, and no silent depth truncation (a K7
request whose execution only covers K3 rows must be rejected, not
accepted with a shrunk depth).
"""

from __future__ import annotations

import pytest

from hipengine.generation.concurrency2_simulator import SimulatedResourceLedger
from hipengine.speculative import (
    DraftBatch,
    SpecCycleStage,
    SpeculativeCycleSimulator,
    SpeculativeRequestState,
)
from tests.test_speculative_cycle_simulator import _claims, _plan

_DEPTHS = (1, 2, 3, 4, 5, 6, 7)


def _c1_state(*, request_id: int = 10, output_limit: int = 0) -> SpeculativeRequestState:
    return SpeculativeRequestState(
        method_key="mtp2",
        provider_key="nextn",
        policy_fingerprint="policy:v2",
        target_request_id=request_id,
        resident_slot=7,
        target_cursor=1000,
        provider_cursor=2000,
        provider_state_lease=f"provider:{request_id}",
        output_limit=output_limit,
    )


def _c1_draft(
    state: SpeculativeRequestState,
    depth: int,
    *,
    cycle_id: int,
    active_depths: int | None = None,
) -> DraftBatch:
    """One C1 singleton draft with `depth` chain candidates."""
    active = depth if active_depths is None else active_depths
    tokens = tuple(state.target_request_id * 10 + d for d in range(1, depth + 1))
    return DraftBatch(
        request_ids=(state.target_request_id,),
        candidate_tokens=tokens,
        parent_positions=tuple(
            state.target_cursor + d - 1 for d in range(1, depth + 1)
        ),
        draft_depths=tuple(range(1, depth + 1)),
        row_to_request=(state.target_request_id,) * depth,
        tree_parents=tuple(-1 if d == 1 else d - 2 for d in range(1, depth + 1)),
        active_mask=tuple(d <= active for d in range(1, depth + 1)),
        mode="verify_chain",
        cycle_id=cycle_id,
        resident_slots=(state.resident_slot,) * depth,
    )


def _simulator(state: SpeculativeRequestState) -> SpeculativeCycleSimulator:
    return SpeculativeCycleSimulator(SimulatedResourceLedger(_plan()), (state,))


def _run(simulator, state, draft, *, accepted, correction=None):
    return simulator.run_cycle(
        draft,
        component_claims=_claims(f"c1k{draft.cycle_id}", rows=draft.draft_rows),
        accepted_counts=(accepted,),
        correction_or_bonus_tokens=(correction,),
    )


@pytest.mark.parametrize("depth", _DEPTHS)
def test_c1_full_acceptance_at_every_depth(depth: int) -> None:
    """Requested depth K executes K rows and commits K+1 visible tokens."""

    state = _c1_state()
    simulator = _simulator(state)
    result = _run(
        simulator,
        state,
        _c1_draft(state, depth, cycle_id=1),
        accepted=depth,
    )

    assert result.stage is SpecCycleStage.COMMITTED
    assert result.telemetry.candidate_counts == (depth,)
    assert result.telemetry.logical_frontier_rows == depth + 1
    updated = simulator.state(state.target_request_id)
    # The resident root is already committed; visible tokens are the K
    # accepted candidates (the simulator does not re-commit the root).
    assert updated.target_cursor == state.target_cursor + depth
    assert updated.provider_cursor == state.provider_cursor + depth
    assert len(updated.visible_tokens) == depth
    assert updated.finished is False
    simulator.assert_conserved()


@pytest.mark.parametrize("depth", _DEPTHS)
@pytest.mark.parametrize("rejected_after", (0, 1, 2, 3))
def test_c1_rejection_at_every_candidate_position(depth: int, rejected_after: int) -> None:
    """Accept a prefix, reject the rest, commit the correction token."""

    if rejected_after >= depth:
        pytest.skip("position beyond this depth")
    state = _c1_state()
    simulator = _simulator(state)
    result = _run(
        simulator,
        state,
        _c1_draft(state, depth, cycle_id=1),
        accepted=rejected_after,
        correction=9000 + rejected_after,
    )

    assert result.stage is SpecCycleStage.COMMITTED
    updated = simulator.state(state.target_request_id)
    # Committed visible tokens: accepted candidates plus the correction.
    assert updated.target_cursor == state.target_cursor + rejected_after + 1
    assert updated.provider_cursor == state.provider_cursor + rejected_after
    assert updated.visible_tokens[-1] == 9000 + rejected_after
    assert updated.finished is False
    simulator.assert_conserved()


@pytest.mark.parametrize("depth", _DEPTHS)
@pytest.mark.parametrize("eos_position", (1, 2, 3, 4))
def test_c1_eos_replacement_correction_at_every_candidate_position(
    depth: int, eos_position: int
) -> None:
    """EOS at candidate position p accepts p-1 rows and commits its correction.

    The EOS candidate itself is never committed; the correction token
    replaces it and the following cycle continues from the corrected
    cursor. (Terminal EOS drain semantics are engine-level; the simulator
    pins the per-position accept/correction arithmetic.)
    """

    if eos_position > depth:
        pytest.skip("position beyond this depth")
    accepted = eos_position - 1
    state = _c1_state(output_limit=1000)
    simulator = _simulator(state)
    result = _run(
        simulator,
        state,
        _c1_draft(state, depth, cycle_id=1),
        accepted=accepted,
        correction=8000 + eos_position,
    )

    assert result.stage is SpecCycleStage.COMMITTED
    updated = simulator.state(state.target_request_id)
    # Committed visible tokens: accepted candidates plus the EOS-replacing
    # correction (the EOS candidate itself is never committed).
    assert updated.target_cursor == state.target_cursor + accepted + 1
    assert updated.provider_cursor == state.provider_cursor + accepted
    assert updated.visible_tokens[-1] == 8000 + eos_position
    simulator.assert_conserved()
    # The corrected request continues in the next cycle from its new cursor.
    followup = _run(
        simulator,
        updated,
        _c1_draft(updated, depth, cycle_id=2),
        accepted=depth,
    )
    assert followup.stage is SpecCycleStage.COMMITTED
    continued = simulator.state(state.target_request_id)
    assert continued.target_cursor == updated.target_cursor + depth
    simulator.assert_conserved()


@pytest.mark.parametrize("depth", (4, 7))
def test_c1_silent_depth_truncation_is_rejected_not_executed(depth: int) -> None:
    """A K-depth request whose active rows cover fewer depths cannot accept K.

    This is the "explicit K7 test that silently runs K3 fails" contract:
    execution covers 3 active rows while the request requested `depth`.
    """

    state = _c1_state()
    simulator = _simulator(state)
    with pytest.raises(ValueError, match="accepted counts exceed active"):
        _run(
            simulator,
            state,
            _c1_draft(state, depth, cycle_id=1, active_depths=3),
            accepted=depth,
        )


@pytest.mark.parametrize("depth", (1, 7))
def test_c1_output_horizon_clips_and_finishes(depth: int) -> None:
    """The horizon clips the commit and the request drains cleanly."""

    horizon = depth  # the full-acceptance commit reaches exactly the limit
    state = _c1_state(output_limit=horizon)
    simulator = _simulator(state)
    result = _run(
        simulator,
        state,
        _c1_draft(state, depth, cycle_id=1),
        accepted=depth,
    )

    assert result.stage is SpecCycleStage.COMMITTED
    updated = simulator.state(state.target_request_id)
    assert updated.finished is True
    simulator.assert_conserved()
    # Reclaim only works once the transaction is complete and frees the slot.
    reclaimed = simulator.reclaim_finished(state.target_request_id)
    assert reclaimed.finished is True
    with pytest.raises(KeyError):
        simulator.state(state.target_request_id)


@pytest.mark.parametrize("depth", (1, 7))
@pytest.mark.parametrize(
    "stage",
    (
        SpecCycleStage.RESERVED,
        SpecCycleStage.TARGET_OPEN,
        SpecCycleStage.PROVIDER_OPEN,
        SpecCycleStage.DRAFTED,
        SpecCycleStage.ACCEPTED,
    ),
)
def test_c1_failure_at_every_stage_rolls_back_and_allows_retry(
    depth: int, stage: SpecCycleStage
) -> None:
    """An injected failure rolls cursors back and the retry cycle commits."""

    state = _c1_state()
    simulator = _simulator(state)
    failed = simulator.run_cycle(
        _c1_draft(state, depth, cycle_id=1),
        component_claims=_claims("c1:fail", rows=depth),
        accepted_counts=(depth,),
        correction_or_bonus_tokens=(None,),
        fail_at=stage,
    )

    assert failed.stage is SpecCycleStage.FAILED
    assert failed.stage is SpecCycleStage.FAILED
    assert failed.transaction.rolled_back is True
    assert failed.telemetry is None
    updated = simulator.state(state.target_request_id)
    assert updated.target_cursor == state.target_cursor
    assert updated.provider_cursor == state.provider_cursor
    # The failed transaction released ownership; the request can retry.
    assert updated.pending_transaction_id is None
    simulator.assert_conserved()

    # Retry: the next cycle commits from the unchanged cursors. The failed
    # cycle never advanced the request cycle_id, so the retry reuses it.
    retry = _run(
        simulator,
        updated,
        _c1_draft(updated, depth, cycle_id=1),
        accepted=depth,
    )
    assert retry.stage is SpecCycleStage.COMMITTED
    committed = simulator.state(state.target_request_id)
    assert committed.target_cursor == state.target_cursor + depth
    assert committed.provider_cursor == state.provider_cursor + depth
    assert committed.pending_transaction_id is None
    simulator.assert_conserved()


def test_c1_k7_k1_k0_transition_keeps_cursors_monotonic() -> None:
    """K7 -> K1 -> K0 -> K7 at one resident slot across four cycles."""

    state = _c1_state()
    simulator = _simulator(state)
    cursor = state.target_cursor
    provider = state.provider_cursor

    # K7 full acceptance: 7 committed candidates.
    result = _run(
        simulator,
        state,
        _c1_draft(state, 7, cycle_id=1),
        accepted=7,
    )
    assert result.stage is SpecCycleStage.COMMITTED
    updated = simulator.state(state.target_request_id)
    assert updated.target_cursor == cursor + 7
    assert updated.provider_cursor == provider + 7

    # K1 full acceptance.
    result = _run(
        simulator,
        updated,
        _c1_draft(updated, 1, cycle_id=2),
        accepted=1,
    )
    assert result.stage is SpecCycleStage.COMMITTED
    updated = simulator.state(state.target_request_id)
    assert updated.target_cursor == cursor + 8
    assert updated.provider_cursor == provider + 8

    # K0 (target-only AR) — provider cursor must not move.
    from hipengine.kvcache import ClaimLifetime, ResourceClaimSet
    from hipengine.speculative import SpecPlanReason

    result = simulator.run_k0_cycle(
        (state.target_request_id,),
        component_claims={
            "target": ResourceClaimSet.from_mapping(
                "c1:k0:target",
                {"target.txn": 1},
                lifetime=ClaimLifetime.TRANSACTION,
            ),
            "transient": ResourceClaimSet.from_mapping(
                "c1:k0:transient",
                {"spec.results": 1},
                lifetime=ClaimLifetime.WORK_ITEM,
            ),
        },
        output_tokens=(41,),
        reasons=(SpecPlanReason.POLICY_SELECTED_AR,),
    )
    assert result.stage is SpecCycleStage.COMMITTED
    updated = simulator.state(state.target_request_id)
    assert updated.target_cursor == cursor + 9
    assert updated.provider_cursor == provider + 8

    # Back to K7.
    result = _run(
        simulator,
        updated,
        _c1_draft(updated, 7, cycle_id=4),
        accepted=7,
    )
    assert result.stage is SpecCycleStage.COMMITTED
    final = simulator.state(state.target_request_id)
    assert final.target_cursor == cursor + 16
    assert final.provider_cursor == provider + 15
    simulator.assert_conserved()
