from __future__ import annotations

import pytest

from hipengine.kvcache import ClaimLifetime, ResourceClaimSet
from hipengine.speculative import (
    AcceptResult,
    SpecCycleResult,
    SpecCycleStage,
    SpecCycleTelemetry,
    SpecCycleTransaction,
    SpecPlanReason,
    SpecTransactionMode,
)
from hipengine.speculative.simulator import (
    SpecTransaction,
    SpeculativeCycleResult,
    SpeculativeCycleSimulator,
)
from hipengine.generation.concurrency2_simulator import SimulatedResourceLedger
from tests.test_speculative_cycle_simulator import _claims, _draft, _plan, _states


def _reserved_claims() -> ResourceClaimSet:
    return ResourceClaimSet.from_mapping(
        "spec-cycle:1:1",
        {"target.txn": 4, "provider.kv": 3, "spec.rows": 5},
        lifetime=ClaimLifetime.TRANSACTION,
    )


def _transaction(**overrides) -> SpecCycleTransaction:
    values = {
        "operation_id": "spec-cycle:1:1",
        "transaction_id": 1,
        "cycle_id": 1,
        "request_ids": (10, 20),
        "reserved_claims": _reserved_claims(),
        "pre_target_cursors": (100, 200),
        "pre_provider_cursors": (100, 200),
        "pre_rng_counters": (0, 0),
        "target_transaction_mode": SpecTransactionMode.PACKED_SCRATCH,
        "provider_transaction_mode": SpecTransactionMode.REVERSIBLE_JOURNAL,
        "target_owner": "target:cycle:1",
        "provider_owner": "provider:cycle:1",
        "target_checkpoint_ids": ("target:10:100", "target:20:200"),
        "provider_checkpoint_ids": ("provider:10:100", "provider:20:200"),
    }
    values.update(overrides)
    return SpecCycleTransaction(**values)


def test_transaction_requires_atomic_target_and_provider_commit() -> None:
    transaction = _transaction(
        target_open=True,
        provider_open=True,
        target_committed=True,
        provider_committed=True,
    )
    assert transaction.has_provider
    assert transaction.committed

    with pytest.raises(ValueError, match="atomic"):
        _transaction(target_committed=True, provider_committed=False)
    with pytest.raises(ValueError, match="rolled-back"):
        _transaction(
            target_committed=True,
            provider_committed=True,
            rolled_back=True,
        )


def test_k0_transaction_has_target_owner_without_provider_owner() -> None:
    transaction = SpecCycleTransaction(
        operation_id="ar-cycle:2",
        transaction_id=2,
        cycle_id=2,
        request_ids=(7,),
        reserved_claims=ResourceClaimSet.from_mapping(
            "ar-cycle:2",
            {"target.txn": 1},
            lifetime=ClaimLifetime.TRANSACTION,
        ),
        pre_target_cursors=(44,),
        pre_rng_counters=(9,),
        target_transaction_mode=SpecTransactionMode.RESERVED_APPEND,
        provider_transaction_mode=None,
        target_owner="target:ar:2",
        provider_owner=None,
        target_checkpoint_ids=("target:7:44",),
        provider_checkpoint_ids=(),
        target_open=True,
        target_committed=True,
    )

    assert not transaction.has_provider
    assert transaction.committed
    assert transaction.pre_provider_cursors == ()
    with pytest.raises(ValueError, match="provider ownership"):
        SpecCycleTransaction(
            operation_id="ar-cycle:2",
            transaction_id=3,
            cycle_id=3,
            request_ids=(7,),
            reserved_claims=transaction.reserved_claims,
            pre_target_cursors=(44,),
            pre_rng_counters=(9,),
            target_transaction_mode=SpecTransactionMode.RESERVED_APPEND,
            provider_transaction_mode=None,
            target_owner="target:ar:3",
            provider_owner="stale-provider",
            target_checkpoint_ids=("target:7:44",),
        )


def test_cycle_telemetry_reports_logical_and_physical_work() -> None:
    telemetry = SpecCycleTelemetry(
        operation_id="spec-cycle:1:1",
        request_ids=(10, 20),
        candidate_counts=(2, 1),
        plan_reasons=(
            SpecPlanReason.SPECULATIVE_QUALIFIED,
            SpecPlanReason.SPECULATIVE_QUALIFIED,
        ),
        proposal_widths=(2,),
        target_row_decomposition=(3, 2),
        execution_route="graph",
        proposal_seconds=0.001,
        target_seconds=0.004,
        accept_commit_seconds=0.0005,
        provider_update_seconds=0.0002,
        scheduler_readback_seconds=0.0001,
        weight_sweeps=2,
        result_bytes=64,
    )

    assert telemetry.logical_request_count == 2
    assert telemetry.logical_frontier_rows == 5
    assert telemetry.complete_seconds == pytest.approx(0.0058)
    assert telemetry.weight_sweeps == 2
    with pytest.raises(ValueError, match="target_row_decomposition"):
        SpecCycleTelemetry(
            operation_id="bad",
            request_ids=(10,),
            candidate_counts=(2,),
            plan_reasons=(SpecPlanReason.SPECULATIVE_QUALIFIED,),
            proposal_widths=(1,),
            target_row_decomposition=(2,),
            execution_route="eager",
        )


def test_committed_cycle_result_derives_visible_ids_from_accept_result() -> None:
    transaction = _transaction(
        target_open=True,
        provider_open=True,
        target_committed=True,
        provider_committed=True,
    )
    accept = AcceptResult(
        request_ids=(10, 20),
        accepted_counts=(2, 0),
        accepted_tokens=((101, 102), ()),
        transaction_id=1,
        selected_candidate_rows=(3, 1),
        correction_or_bonus_tokens=(103, 203),
        target_cursor_deltas=(3, 1),
        provider_cursor_deltas=(2, 0),
        finish_reasons=(None, "length"),
    )

    result = SpecCycleResult.committed(transaction, accept)

    assert result.stage is SpecCycleStage.COMMITTED
    assert result.committed_output_ids == ((101, 102, 103), (203,))
    assert result.committed_output_lengths == (3, 1)
    assert result.selected_rows == (3, 1)
    assert result.finish_reasons == (None, "length")
    with pytest.raises(ValueError, match="terminal"):
        SpecCycleResult(
            stage=SpecCycleStage.VERIFIED,
            transaction=transaction,
        )


def test_simulator_uses_production_transaction_and_result_records() -> None:
    simulator = SpeculativeCycleSimulator(SimulatedResourceLedger(_plan()), _states())

    result = simulator.run_cycle(
        _draft(),
        component_claims=_claims("production-records"),
        accepted_counts=(1, 2),
        correction_or_bonus_tokens=(901, 902),
    )

    assert type(result) is SpecCycleResult
    assert type(result.transaction) is SpecCycleTransaction
    assert SpeculativeCycleResult is SpecCycleResult
    assert SpecTransaction is SpecCycleTransaction
    assert result.committed_output_ids == ((101, 901), (201, 202, 902))
    assert result.telemetry is not None
    assert result.telemetry.logical_frontier_rows == 7
    simulator.assert_conserved()
