from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from hipengine.generation import (
    GeneratedToken,
    GenerationAdmissionRejected,
    ResidentEngineLoop,
)
from hipengine.kvcache import (    ClaimLifetime,
    FitAwareAdmissionController,
    KVPoolPlan,
    LedgerAdmissionCoordinator,
    KVPoolSpec,
    ReservationState,
    ResourceChange,
    ResourceClaim,
    ResourceClaimSet,
    ResourceDelta,
    ResourceLedger,
    ResourceUnavailable,
)
from hipengine.kvcache.simulated import (
    FAKE_KV_BACKEND_KINDS,
    create_fake_kv_backend,
)


def _plan(*, payload: int = 10, scales: int = 5, workspace: int = 2) -> KVPoolPlan:
    return KVPoolPlan(
        backend_fingerprint="test-backend",
        generation=1,
        pools=(
            KVPoolSpec(
                "kv.payload",
                payload,
                unit="pages",
                lifetimes=(ClaimLifetime.LOAD, ClaimLifetime.LEASE, ClaimLifetime.CACHE),
            ),
            KVPoolSpec(
                "kv.scales",
                scales,
                unit="pages",
                lifetimes=(ClaimLifetime.LEASE, ClaimLifetime.CACHE),
            ),
            KVPoolSpec(
                "kv.workspace",
                workspace,
                unit="slots",
                lifetimes=(ClaimLifetime.WORK_ITEM, ClaimLifetime.TRANSACTION),
            ),
        ),
    )


def _ownership(snapshot: dict) -> dict:
    return {
        "pools": snapshot["pools"],
        "owners": snapshot["owners"],
        "provisional_reservations": snapshot["provisional_reservations"],
    }


def _claims(
    claim_id: str,
    *,
    payload: int = 0,
    scales: int = 0,
    workspace: int = 0,
    request_id: int | None = None,
) -> ResourceClaimSet:
    claims = []
    if payload:
        claims.append(ResourceClaim("kv.payload", payload, ClaimLifetime.LEASE))
    if scales:
        claims.append(ResourceClaim("kv.scales", scales, ClaimLifetime.LEASE))
    if workspace:
        claims.append(ResourceClaim("kv.workspace", workspace, ClaimLifetime.WORK_ITEM))
    return ResourceClaimSet(claim_id, tuple(claims), request_id=request_id)


@pytest.mark.parametrize("kind", FAKE_KV_BACKEND_KINDS)
def test_production_ledger_consumes_backend_estimators_without_format_formulas(kind: str) -> None:
    backend = create_fake_kv_backend(kind, capacity_tokens=4096)
    ledger = ResourceLedger(backend.plan_pools(None))
    request = SimpleNamespace(
        request_id=7,
        prompt_tokens=tuple(range(8)),
        max_new_tokens=8,
    )
    claims = backend.estimate(
        request,
        None,
        {"kind": "admission", "tokens": len(request.prompt_tokens)},
    )
    reservation = ledger.reserve_provisional(claims)
    ledger.commit(reservation, owner_id="lease:7")

    work_claims = backend.estimate(
        request,
        None,
        {"kind": "work_item", "current_tokens": 8, "next_tokens": 9},
    )
    if work_claims.claims:
        work = ledger.reserve_provisional(work_claims)
        ledger.rollback(work)
    ledger.release("lease:7")

    snapshot = ledger.snapshot()
    assert set(snapshot["pools"]) == {pool.pool_id for pool in backend.plan_pools(None).pools}
    assert all(pool["used"] == 0 for pool in snapshot["pools"].values())
    ledger.assert_conserved()


def test_resource_ledger_provisional_commit_delta_release_and_load_baseline() -> None:
    plan = replace(
        _plan(),
        load_claims=ResourceClaimSet(
            "load",
            (ResourceClaim("kv.payload", 1, ClaimLifetime.LOAD),),
        ),
    )
    ledger = ResourceLedger(plan)
    reservation = ledger.reserve_provisional(
        _claims("admit:7", payload=3, scales=2, request_id=7),
        reservation_id="reservation:7",
    )

    assert reservation.state is ReservationState.PROVISIONAL
    assert ledger.snapshot()["pools"]["kv.payload"]["used"] == 4
    committed = ledger.commit(reservation, owner_id="lease:7")
    assert committed.state is ReservationState.COMMITTED
    assert ledger.has_owner("lease:7")

    workspace = ledger.reserve_provisional(
        _claims("work:7", workspace=1, request_id=7),
        reservation_id="work:7",
    )
    ledger.commit(workspace, owner_id="operation:7")
    ledger.release("operation:7", operation_id="work-release:7")

    delta = ResourceDelta(
        operation_id="grow:7",
        lease_id="lease:7",
        request_id=7,
        changes=(
            ResourceChange("kv.payload", 1, ClaimLifetime.LEASE),
            ResourceChange("kv.scales", 1, ClaimLifetime.LEASE),
        ),
    )
    ledger.apply_delta("lease:7", delta)
    released = ledger.release("lease:7", operation_id="reclaim:7")

    assert released.units_by_pool() == {"kv.payload": -4, "kv.scales": -3}
    final = ledger.snapshot()
    assert final["pools"]["kv.payload"]["used"] == 1
    assert final["pools"]["kv.scales"]["used"] == 0
    assert final["pools"]["kv.workspace"]["used"] == 0
    assert final["owners"] == {"load:test-backend": {"kv.payload": 1}}
    ledger.assert_conserved()


def test_resource_ledger_reserve_and_delta_failures_are_atomic() -> None:
    ledger = ResourceLedger(_plan(payload=4, scales=4, workspace=1))
    before = ledger.snapshot()

    with pytest.raises(ResourceUnavailable) as reserve_error:
        ledger.reserve_provisional(_claims("too-large", payload=2, scales=5))
    assert reserve_error.value.resource == "kv.scales"
    assert reserve_error.value.impossible is True
    rejected = ledger.snapshot()
    assert _ownership(rejected) == _ownership(before)
    assert rejected["blocking_counts"] == {"kv.scales": 1}

    reservation = ledger.reserve_provisional(_claims("owner", payload=3, scales=2))
    ledger.commit(reservation, owner_id="lease:1")
    blocker = ledger.reserve_provisional(_claims("blocker", scales=2))
    ledger.commit(blocker, owner_id="lease:blocker")
    committed = ledger.snapshot()
    with pytest.raises(ResourceUnavailable) as delta_error:
        ledger.apply_delta(
            "lease:1",
            ResourceDelta(
                "grow",
                "lease:1",
                (
                    ResourceChange("kv.payload", 1),
                    ResourceChange("kv.scales", 2),
                ),
            ),
        )
    assert delta_error.value.resource == "kv.scales"
    assert delta_error.value.impossible is False
    assert _ownership(ledger.snapshot()) == _ownership(committed)

    with pytest.raises(ValueError, match="underflows"):
        ledger.apply_delta(
            "lease:1",
            ResourceDelta(
                "bad-release",
                "lease:1",
                (ResourceChange("kv.payload", -4),),
            ),
        )
    assert _ownership(ledger.snapshot()) == _ownership(committed)
    ledger.assert_conserved()


def test_resource_ledger_rollback_and_lifetime_validation_restore_exact_state() -> None:
    ledger = ResourceLedger(_plan())
    reservation = ledger.reserve_provisional(_claims("temporary", payload=2, scales=1))
    reserved = ledger.snapshot()
    assert reserved["provisional_reservations"] == 1

    rolled_back = ledger.rollback(reservation)

    assert rolled_back.state is ReservationState.ROLLED_BACK
    assert all(pool["used"] == 0 for pool in ledger.snapshot()["pools"].values())
    with pytest.raises(ValueError, match="does not accept lifetime"):
        ledger.reserve_provisional(
            ResourceClaimSet(
                "bad-lifetime",
                (ResourceClaim("kv.workspace", 1, ClaimLifetime.LEASE),),
            )
        )
    assert all(pool["used"] == 0 for pool in ledger.snapshot()["pools"].values())
    ledger.assert_conserved()


@pytest.mark.parametrize("rejected_pool", ("kv.payload", "kv.scales", "kv.workspace"))
def test_resource_ledger_injected_failure_at_each_claim_boundary(rejected_pool: str) -> None:
    ledger = ResourceLedger(_plan(payload=4, scales=4, workspace=4))
    claims = ResourceClaimSet(
        f"inject:{rejected_pool}",
        (
            ResourceClaim(
                "kv.payload",
                5 if rejected_pool == "kv.payload" else 1,
                ClaimLifetime.LEASE,
            ),
            ResourceClaim(
                "kv.scales",
                5 if rejected_pool == "kv.scales" else 1,
                ClaimLifetime.LEASE,
            ),
            ResourceClaim(
                "kv.workspace",
                5 if rejected_pool == "kv.workspace" else 1,
                ClaimLifetime.WORK_ITEM,
            ),
        ),
    )
    before = ledger.snapshot()

    with pytest.raises(ResourceUnavailable) as error:
        ledger.reserve_provisional(claims)

    assert error.value.resource == rejected_pool
    rejected = ledger.snapshot()
    assert _ownership(rejected) == _ownership(before)
    assert rejected["blocking_counts"] == {rejected_pool: 1}
    ledger.assert_conserved()


def test_fit_aware_admission_bypasses_temporarily_blocked_then_enters_drain_mode() -> None:
    ledger = ResourceLedger(_plan(payload=10, scales=10, workspace=2))
    active = ledger.reserve_provisional(_claims("active", payload=7))
    ledger.commit(active, owner_id="lease:active")
    controller = FitAwareAdmissionController(
        ledger,
        lookahead=8,
        max_bypasses=2,
    )
    controller.enqueue(1, _claims("long", payload=4, request_id=1))
    controller.enqueue(2, _claims("short-2", payload=2, request_id=2))

    [short_two] = controller.admit(max_items=1)
    assert short_two.request_id == 2
    assert controller.pending_state(1).bypass_count == 1
    ledger.release(short_two.owner_id)

    controller.enqueue(3, _claims("short-3", payload=2, request_id=3))
    [short_three] = controller.admit(max_items=1)
    assert short_three.request_id == 3
    assert controller.pending_state(1).bypass_count == 2
    ledger.release(short_three.owner_id)

    controller.enqueue(4, _claims("short-4", payload=2, request_id=4))
    assert controller.admit(max_items=1) == ()
    snapshot = controller.snapshot()
    assert snapshot["drain_mode_request_id"] == 1
    assert snapshot["pending"][0]["blocking_resources"] == ["kv.payload"]

    ledger.release("lease:active")
    [long_request] = controller.admit(max_items=1)
    assert long_request.request_id == 1
    assert controller.pending_count == 1
    ledger.release(long_request.owner_id)
    assert controller.cancel(4) is True
    assert controller.pending_count == 0
    ledger.assert_conserved()


def test_resident_loop_consumes_only_format_neutral_fit_aware_request_ids() -> None:
    ledger = ResourceLedger(_plan(payload=10, scales=10, workspace=2))
    blocker = ledger.reserve_provisional(_claims("blocker", payload=7))
    ledger.commit(blocker, owner_id="lease:blocker")
    controller = FitAwareAdmissionController(ledger, lookahead=8, max_bypasses=2)
    coordinator = LedgerAdmissionCoordinator(
        controller,
        estimator=lambda request: _claims(
            f"request:{request.request_id}",
            payload=int(request.prompt_tokens[0]),
            request_id=int(request.request_id),
        ),
    )

    class Runner:
        def plan_admission(self, pending_requests, *, max_items):
            return coordinator.plan_admission(pending_requests, max_items=max_items)

        def reserve_admission(self, request):
            coordinator.reserve_admission(request)

        def rollback_admission(self, request):
            coordinator.rollback_admission(request)

        def prefill_batch(self, work, *, commit: bool):
            assert commit is True

        def decode_batch(self, work, *, commit: bool):
            assert commit is True
            return tuple(GeneratedToken(request_id, 900 + request_id, finished=True) for request_id in work.request_ids)

        def compact_batch(self, moves):
            del moves

        def reclaim(self, completed):
            coordinator.reclaim_request(completed)

        def resource_observability_snapshot(self):
            return coordinator.snapshot()

    loop = ResidentEngineLoop(Runner(), capacity=2, prefill_chunk_size=8)
    long_request = loop.submit([4], max_new_tokens=1)
    short_request = loop.submit([2], max_new_tokens=1)

    first = loop.tick()
    assert [event.request_id for event in first if event.kind == "admitted"] == [short_request]
    assert loop.scheduler.active_batch.active_request_ids == (short_request,)
    assert loop.pending_count == 1
    assert coordinator.controller.pending_state(long_request).bypass_count == 1

    loop.tick()
    assert loop.active_count == 0
    assert ledger.snapshot()["pools"]["kv.payload"]["used"] == 7
    assert loop.observability_snapshot()["resources"]["admission"]["pending_count"] == 1

    ledger.release("lease:blocker")
    admitted = loop.tick()
    assert [event.request_id for event in admitted if event.kind == "admitted"] == [long_request]
    loop.tick()
    assert loop.active_count == 0
    assert all(pool["used"] == 0 for pool in ledger.snapshot()["pools"].values())
    ledger.assert_conserved()


def test_resident_loop_surfaces_impossible_claim_as_retryable_named_overload() -> None:
    ledger = ResourceLedger(_plan(payload=4, scales=4, workspace=1))
    coordinator = LedgerAdmissionCoordinator(
        FitAwareAdmissionController(ledger),
        estimator=lambda request: _claims(
            f"request:{request.request_id}",
            payload=int(request.prompt_tokens[0]),
            request_id=int(request.request_id),
        ),
    )

    class Runner:
        def plan_admission(self, pending_requests, *, max_items):
            return coordinator.plan_admission(pending_requests, max_items=max_items)

        def reserve_admission(self, request):
            coordinator.reserve_admission(request)

        def rollback_admission(self, request):
            coordinator.rollback_admission(request)

    loop = ResidentEngineLoop(Runner(), capacity=2, prefill_chunk_size=8)
    request_id = loop.submit([5], max_new_tokens=1)

    with pytest.raises(GenerationAdmissionRejected) as error:
        loop.tick()

    assert error.value.resource == "kv.payload"
    assert error.value.requested_units == 5
    assert error.value.current_units == 0
    assert error.value.capacity_units == 4
    assert loop.pending_count == 1
    assert loop.scheduler.pending_requests[0].request_id == request_id
    assert all(pool["used"] == 0 for pool in ledger.snapshot()["pools"].values())


def test_fit_aware_admission_rejects_impossible_and_has_o1_cancel_tombstones() -> None:
    ledger = ResourceLedger(_plan(payload=4, scales=4, workspace=1))
    controller = FitAwareAdmissionController(ledger, lookahead=4, max_bypasses=1)

    with pytest.raises(ResourceUnavailable) as error:
        controller.enqueue(9, _claims("impossible", payload=5, request_id=9))
    assert error.value.impossible is True
    assert error.value.resource == "kv.payload"

    controller.enqueue(10, _claims("queued", payload=1, request_id=10))
    assert controller.cancel(10) is True
    assert controller.cancel(10) is False
    assert controller.pending_count == 0
    assert controller.snapshot()["cancelled_total"] == 1
    ledger.assert_conserved()
