from __future__ import annotations

import random

import pytest

from hipengine.kvcache.backend import (
    ClaimConfidence,
    ClaimLifetime,
    KVPoolPlan,
    KVPoolSpec,
    LeaseState,
    ResourceClaim,
    ResourceClaimSet,
    ResourceChange,
    ResourceDelta,
)
from hipengine.kvcache.ledger import (
    DuplicateOperationError,
    InjectedLedgerFailure,
    InvalidOperationStateError,
    ResourceCapacityError,
    ResourceLedger,
    UnknownResourceClaimError,
)


def _plan() -> KVPoolPlan:
    return KVPoolPlan(
        backend_fingerprint="backend-v1",
        generation=4,
        pools=(
            KVPoolSpec("payload", 1000, unit="bytes"),
            KVPoolSpec("scales", 100, unit="bytes"),
            KVPoolSpec("rows", 8, unit="rows"),
            KVPoolSpec("workspace", 256, unit="bytes"),
        ),
    )


def _claims(
    payload: int,
    scales: int,
    *,
    rows: int = 1,
    workspace: int = 0,
    confidence: ClaimConfidence = ClaimConfidence.EXACT,
) -> ResourceClaimSet:
    values = [
        ResourceClaim("payload", payload, unit="bytes", confidence=confidence),
        ResourceClaim("scales", scales, unit="bytes", confidence=confidence),
        ResourceClaim("rows", rows, unit="rows", confidence=confidence),
    ]
    if workspace:
        values.append(
            ResourceClaim(
                "workspace",
                workspace,
                unit="bytes",
                lifetime=ClaimLifetime.WORK_ITEM,
                confidence=confidence,
            )
        )
    return ResourceClaimSet(stage="prefill", claims=tuple(values))


def _assert_conserved(ledger: ResourceLedger) -> None:
    snapshot = ledger.snapshot()
    for pool in snapshot.pools:
        assert pool.available >= 0
        assert pool.provisional >= 0
        assert pool.committed >= 0
        assert pool.available + pool.provisional + pool.committed == pool.capacity


def test_reserve_commit_release_tracks_each_pool_and_conserves_capacity() -> None:
    ledger = ResourceLedger(_plan())

    provisional = ledger.reserve("op-a", request_id=11, claims=_claims(400, 40, workspace=64))
    assert provisional.state is LeaseState.PROVISIONAL
    snapshot = ledger.snapshot()
    assert snapshot.pool("payload").provisional == 400
    assert snapshot.pool("payload").committed == 0
    assert snapshot.pool("workspace").provisional == 64

    committed = ledger.commit(provisional.lease_id)
    assert committed.state is LeaseState.COMMITTED
    assert ledger.snapshot().pool("payload").committed == 400
    assert ledger.snapshot().pool("workspace").committed == 64

    released = ledger.release(committed.lease_id, reason="terminal")
    assert {change.pool_id: change.amount for change in released.changes} == {
        "payload": -400,
        "scales": -40,
        "rows": -1,
        "workspace": -64,
    }
    assert ledger.snapshot().active_operations == 0
    _assert_conserved(ledger)


def test_multi_pool_capacity_failure_is_atomic() -> None:
    ledger = ResourceLedger(_plan())
    before = ledger.snapshot()

    with pytest.raises(ResourceCapacityError, match="scales") as exc_info:
        ledger.reserve("too-large", request_id=1, claims=_claims(10, 101))

    assert exc_info.value.pool_id == "scales"
    assert ledger.snapshot() == before
    _assert_conserved(ledger)


def test_same_pool_claims_with_distinct_lifetimes_are_aggregated_atomically() -> None:
    ledger = ResourceLedger(_plan())
    claims = ResourceClaimSet(
        stage="decode",
        claims=(
            ResourceClaim("payload", 600, lifetime=ClaimLifetime.LEASE),
            ResourceClaim("payload", 401, lifetime=ClaimLifetime.WORK_ITEM),
        ),
    )
    before = ledger.snapshot()

    with pytest.raises(ResourceCapacityError, match="payload") as exc_info:
        ledger.reserve("too-large", request_id=1, claims=claims)

    assert exc_info.value.requested == 1001
    assert ledger.snapshot() == before


def test_injected_failure_at_every_claim_boundary_rolls_back_all_mutations() -> None:
    claims = _claims(400, 40, workspace=64)
    for boundary in range(1, len(claims.claims) + 1):
        ledger = ResourceLedger(_plan())
        before = ledger.snapshot()
        with pytest.raises(InjectedLedgerFailure, match=f"boundary {boundary}"):
            ledger.reserve(
                f"inject-{boundary}",
                request_id=boundary,
                claims=claims,
                fail_after_claims=boundary,
            )
        assert ledger.snapshot() == before
        _assert_conserved(ledger)


def test_invalid_request_or_release_metadata_cannot_partially_mutate_ownership() -> None:
    ledger = ResourceLedger(_plan())
    before = ledger.snapshot()
    with pytest.raises(ValueError, match="request_id must be non-negative"):
        ledger.reserve("invalid-request", request_id=-1, claims=_claims(10, 1))
    assert ledger.snapshot() == before

    lease = ledger.reserve("valid", request_id=1, claims=_claims(10, 1))
    before_release = ledger.snapshot()
    with pytest.raises(ValueError, match="reason must not be empty"):
        ledger.release(lease.lease_id, reason="")
    assert ledger.snapshot() == before_release
    assert ledger.lease(lease.lease_id) == lease
    ledger.release(lease.lease_id)
    _assert_conserved(ledger)


def test_provisional_rollback_and_invalid_state_transitions_fail_closed() -> None:
    ledger = ResourceLedger(_plan())
    lease = ledger.reserve("op-a", request_id=1, claims=_claims(100, 10))
    delta = ledger.rollback(lease.lease_id, reason="cancel before admission")

    assert all(change.amount < 0 for change in delta.changes)
    assert ledger.snapshot().active_operations == 0
    with pytest.raises(InvalidOperationStateError, match="unknown operation"):
        ledger.commit(lease.lease_id)

    committed = ledger.commit(
        ledger.reserve("op-b", request_id=2, claims=_claims(100, 10)).lease_id
    )
    with pytest.raises(InvalidOperationStateError, match="committed"):
        ledger.rollback(committed.lease_id)
    _assert_conserved(ledger)


def test_delta_growth_and_release_are_atomic_and_operation_scoped() -> None:
    ledger = ResourceLedger(_plan())
    lease = ledger.commit(
        ledger.reserve("op-a", request_id=1, claims=_claims(800, 80)).lease_id
    )

    updated = ledger.apply(
        ResourceDelta(
            operation_id=lease.lease_id,
            changes=(
                ResourceChange("payload", -300, unit="bytes", reason="unused provisional pages"),
                ResourceChange(
                    "payload",
                    400,
                    unit="bytes",
                    lifetime=ClaimLifetime.WORK_ITEM,
                    reason="temporary decode ownership",
                ),
                ResourceChange("scales", -30, unit="bytes", reason="unused scale rows"),
                ResourceChange(
                    "workspace",
                    128,
                    unit="bytes",
                    lifetime=ClaimLifetime.WORK_ITEM,
                    reason="split-k workspace",
                ),
            ),
        )
    )
    assert updated.claims.amount("payload") == 900
    assert updated.claims.amount("payload", lifetime=ClaimLifetime.LEASE) == 500
    assert updated.claims.amount("payload", lifetime=ClaimLifetime.WORK_ITEM) == 400
    assert updated.claims.amount("scales") == 50
    assert updated.claims.amount("workspace") == 128
    before_failure = ledger.snapshot()

    with pytest.raises(ResourceCapacityError, match="workspace"):
        ledger.apply(
            ResourceDelta(
                operation_id=lease.lease_id,
                changes=(
                    ResourceChange(
                        "workspace",
                        129,
                        unit="bytes",
                        lifetime=ClaimLifetime.WORK_ITEM,
                    ),
                ),
            )
        )
    assert ledger.snapshot() == before_failure

    with pytest.raises(InvalidOperationStateError, match="does not own"):
        ledger.apply(
            ResourceDelta(
                operation_id=lease.lease_id,
                changes=(ResourceChange("payload", -501, unit="bytes"),),
            )
        )
    assert ledger.snapshot() == before_failure
    _assert_conserved(ledger)


def test_unknown_confidence_and_duplicate_operation_ids_fail_before_mutation() -> None:
    ledger = ResourceLedger(_plan())
    before = ledger.snapshot()
    with pytest.raises(UnknownResourceClaimError, match="unknown confidence"):
        ledger.reserve(
            "unknown",
            request_id=1,
            claims=_claims(10, 1, confidence=ClaimConfidence.UNKNOWN),
        )
    assert ledger.snapshot() == before

    ledger.reserve("same", request_id=2, claims=_claims(10, 1))
    after_first = ledger.snapshot()
    with pytest.raises(DuplicateOperationError, match="same"):
        ledger.reserve("same", request_id=3, claims=_claims(10, 1))
    assert ledger.snapshot() == after_first
    _assert_conserved(ledger)


def test_deterministic_random_reserve_commit_adjust_release_preserves_conservation() -> None:
    rng = random.Random(0xC203)
    ledger = ResourceLedger(_plan())
    live: dict[str, LeaseState] = {}
    next_id = 0

    for _ in range(500):
        choices = ["reserve"]
        if live:
            choices.extend(("commit", "rollback", "release", "adjust"))
        action = rng.choice(choices)
        if action == "reserve":
            operation_id = f"op-{next_id}"
            next_id += 1
            try:
                lease = ledger.reserve(
                    operation_id,
                    request_id=next_id,
                    claims=_claims(
                        rng.randint(1, 120),
                        rng.randint(1, 12),
                        workspace=rng.choice((0, 8, 16)),
                    ),
                )
            except ResourceCapacityError:
                pass
            else:
                live[operation_id] = lease.state
        else:
            operation_id = rng.choice(tuple(live))
            state = live[operation_id]
            if action == "commit" and state is LeaseState.PROVISIONAL:
                live[operation_id] = ledger.commit(operation_id).state
            elif action == "rollback" and state is LeaseState.PROVISIONAL:
                ledger.rollback(operation_id)
                live.pop(operation_id)
            elif action == "release":
                ledger.release(operation_id)
                live.pop(operation_id)
            elif action == "adjust" and state is LeaseState.COMMITTED:
                lease = ledger.lease(operation_id)
                payload = lease.claims.amount("payload")
                if payload > 1:
                    ledger.apply(
                        ResourceDelta(
                            operation_id=operation_id,
                            changes=(ResourceChange("payload", -1),),
                        )
                    )
        _assert_conserved(ledger)

    for operation_id in tuple(live):
        ledger.release(operation_id)
    assert ledger.snapshot().active_operations == 0
    _assert_conserved(ledger)
