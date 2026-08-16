"""Format-neutral named-resource ledger and fit-aware admission control."""

from __future__ import annotations

import threading
from collections import Counter, deque
from dataclasses import dataclass, replace
from enum import Enum
from collections.abc import Callable, Sequence
from typing import Any

from hipengine.kvcache.backend import (
    ClaimLifetime,
    KVPoolPlan,
    ResourceChange,
    ResourceClaimSet,
    ResourceDelta,
)


class ReservationState(str, Enum):
    PROVISIONAL = "provisional"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True, slots=True)
class ResourceBlock:
    resource: str
    requested_units: int
    current_units: int
    capacity_units: int
    lifetime: ClaimLifetime
    impossible: bool


@dataclass(frozen=True, slots=True)
class ResourceFit:
    fits: bool
    impossible: bool
    blocking: tuple[ResourceBlock, ...] = ()

    @property
    def blocking_resources(self) -> tuple[str, ...]:
        return tuple(block.resource for block in self.blocking)


class ResourceUnavailable(MemoryError):
    """Atomic admission failure with one named blocking resource."""

    def __init__(self, block: ResourceBlock, message: str | None = None) -> None:
        self.resource = block.resource
        self.requested_units = block.requested_units
        self.current_units = block.current_units
        self.capacity_units = block.capacity_units
        self.lifetime = block.lifetime
        self.impossible = bool(block.impossible)
        super().__init__(
            message
            or (
                f"resource {block.resource} cannot fit {block.requested_units} units "
                f"with {block.current_units}/{block.capacity_units} used"
            )
        )


@dataclass(frozen=True, slots=True)
class ResourceReservation:
    reservation_id: str
    claims: ResourceClaimSet
    state: ReservationState = ReservationState.PROVISIONAL
    owner_id: str | None = None

    def __post_init__(self) -> None:
        reservation_id = str(self.reservation_id)
        if not reservation_id or reservation_id != reservation_id.strip():
            raise ValueError("reservation_id must be a non-empty trimmed string")
        object.__setattr__(self, "reservation_id", reservation_id)
        object.__setattr__(self, "state", ReservationState(self.state))
        if self.owner_id is not None:
            owner = str(self.owner_id)
            if not owner or owner != owner.strip():
                raise ValueError("owner_id must be a non-empty trimmed string")
            object.__setattr__(self, "owner_id", owner)


class ResourceLedger:
    """Atomic ownership accounting over a backend-declared stable pool plan."""

    def __init__(self, plan: KVPoolPlan) -> None:
        self.plan = plan
        self._pool_by_id = {pool.pool_id: pool for pool in plan.pools}
        self._used = {pool.pool_id: 0 for pool in plan.pools}
        self._high_water = dict(self._used)
        self._used_by_lifetime: dict[str, Counter[ClaimLifetime]] = {
            pool.pool_id: Counter() for pool in plan.pools
        }
        self._owners: dict[str, ResourceClaimSet] = {}
        self._reservations: dict[str, ResourceReservation] = {}
        self._next_reservation_id = 0
        self._stats: Counter[str] = Counter()
        self._blocking_counts: Counter[str] = Counter()
        self._lock = threading.RLock()
        if plan.load_claims is not None:
            owner_id = f"load:{plan.backend_fingerprint}"
            self._validate_claims(plan.load_claims)
            fit = self._fit_locked(plan.load_claims)
            if not fit.fits:
                raise ResourceUnavailable(fit.blocking[0], "load claims exceed the pool plan")
            self._owners[owner_id] = plan.load_claims
            self._apply_claim_totals(plan.load_claims, sign=1)

    def fit(self, claims: ResourceClaimSet) -> ResourceFit:
        with self._lock:
            return self._fit_locked(claims)

    def reserve_provisional(
        self,
        claims: ResourceClaimSet,
        *,
        reservation_id: str | None = None,
    ) -> ResourceReservation:
        with self._lock:
            identifier = (
                f"reservation:{self._next_reservation_id}"
                if reservation_id is None
                else str(reservation_id)
            )
            if reservation_id is None:
                self._next_reservation_id += 1
            if identifier in self._reservations or identifier in self._owners:
                raise ValueError(f"resource reservation {identifier!r} already exists")
            self._validate_claims(claims)
            fit = self._fit_locked(claims)
            if not fit.fits:
                self._record_rejection(fit)
                raise ResourceUnavailable(fit.blocking[0])
            reservation = ResourceReservation(identifier, claims)
            self._reservations[identifier] = reservation
            self._apply_claim_totals(claims, sign=1)
            self._stats["provisional_reserves"] += 1
            return reservation

    def commit(
        self,
        reservation: ResourceReservation,
        *,
        owner_id: str,
    ) -> ResourceReservation:
        owner = str(owner_id)
        if not owner or owner != owner.strip():
            raise ValueError("owner_id must be a non-empty trimmed string")
        with self._lock:
            current = self._current_reservation(reservation)
            if current.state is not ReservationState.PROVISIONAL:
                raise ValueError("only a provisional reservation can commit")
            if owner in self._owners:
                raise ValueError(f"resource owner {owner!r} already exists")
            committed = replace(current, state=ReservationState.COMMITTED, owner_id=owner)
            self._reservations[current.reservation_id] = committed
            self._owners[owner] = current.claims
            self._stats["commits"] += 1
            return committed

    def rollback(self, reservation: ResourceReservation) -> ResourceReservation:
        with self._lock:
            current = self._current_reservation(reservation)
            if current.state is not ReservationState.PROVISIONAL:
                raise ValueError("only a provisional reservation can roll back")
            self._apply_claim_totals(current.claims, sign=-1)
            rolled_back = replace(current, state=ReservationState.ROLLED_BACK)
            self._reservations[current.reservation_id] = rolled_back
            self._stats["rollbacks"] += 1
            return rolled_back

    def apply_delta(self, owner_id: str, delta: ResourceDelta) -> ResourceClaimSet:
        owner = str(owner_id)
        if delta.lease_id != owner:
            raise ValueError("resource delta lease_id does not match owner_id")
        with self._lock:
            try:
                current = self._owners[owner]
            except KeyError as exc:
                raise KeyError(f"unknown resource owner {owner!r}") from exc
            updated = current.apply(delta, claim_id=f"{current.claim_id}@{delta.operation_id}")
            self._validate_claims(updated)
            changed = _changes_by_pool_and_lifetime(delta)
            for (pool_id, lifetime), units in sorted(
                changed.items(), key=lambda item: (item[0][0], item[0][1].value)
            ):
                current_entry = current.entries().get((pool_id, lifetime))
                current_owner_units = 0 if current_entry is None else current_entry.units
                if current_owner_units + units < 0:
                    raise ValueError(f"resource delta underflows {pool_id}/{lifetime.value}")
                next_used = self._used[pool_id] + units
                capacity = self._pool_by_id[pool_id].capacity
                if next_used > capacity:
                    block = ResourceBlock(
                        pool_id,
                        max(0, units),
                        self._used[pool_id],
                        capacity,
                        lifetime,
                        current_owner_units + units > capacity,
                    )
                    self._record_rejection(ResourceFit(False, block.impossible, (block,)))
                    raise ResourceUnavailable(block)
                if next_used < 0:
                    raise ValueError(f"resource delta underflows ledger pool {pool_id}")
            for (pool_id, lifetime), units in changed.items():
                self._used[pool_id] += units
                self._used_by_lifetime[pool_id][lifetime] += units
                if self._used_by_lifetime[pool_id][lifetime] == 0:
                    del self._used_by_lifetime[pool_id][lifetime]
                self._high_water[pool_id] = max(self._high_water[pool_id], self._used[pool_id])
            self._owners[owner] = updated
            self._stats["deltas"] += 1
            return updated

    def release(self, owner_id: str, *, operation_id: str | None = None) -> ResourceDelta:
        owner = str(owner_id)
        with self._lock:
            try:
                claims = self._owners.pop(owner)
            except KeyError as exc:
                raise KeyError(f"unknown resource owner {owner!r}") from exc
            self._apply_claim_totals(claims, sign=-1)
            self._stats["releases"] += 1
            return ResourceDelta(
                operation_id=(f"release:{owner}" if operation_id is None else str(operation_id)),
                lease_id=owner,
                request_id=claims.request_id,
                changes=tuple(
                    ResourceChange(claim.pool_id, -claim.units, claim.lifetime)
                    for claim in claims.claims
                ),
            )

    def has_owner(self, owner_id: str) -> bool:
        with self._lock:
            return str(owner_id) in self._owners

    def owner_claims(self, owner_id: str) -> ResourceClaimSet:
        with self._lock:
            return self._owners[str(owner_id)]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            provisional = sum(
                reservation.state is ReservationState.PROVISIONAL
                for reservation in self._reservations.values()
            )
            return {
                "backend_fingerprint": self.plan.backend_fingerprint,
                "generation": self.plan.generation,
                "pools": {
                    pool_id: {
                        "capacity": pool.capacity,
                        "used": self._used[pool_id],
                        "free": pool.capacity - self._used[pool_id],
                        "high_water": self._high_water[pool_id],
                        "unit": pool.unit,
                        "plane_role": pool.plane_role,
                        "used_by_lifetime": {
                            lifetime.value: units
                            for lifetime, units in sorted(
                                self._used_by_lifetime[pool_id].items(),
                                key=lambda item: item[0].value,
                            )
                        },
                    }
                    for pool_id, pool in sorted(self._pool_by_id.items())
                },
                "owners": {
                    owner: claims.units_by_pool()
                    for owner, claims in sorted(self._owners.items())
                },
                "provisional_reservations": provisional,
                "stats": dict(sorted(self._stats.items())),
                "blocking_counts": dict(sorted(self._blocking_counts.items())),
            }

    def assert_conserved(self) -> None:
        with self._lock:
            expected = {pool_id: 0 for pool_id in self._pool_by_id}
            expected_by_lifetime: dict[str, Counter[ClaimLifetime]] = {
                pool_id: Counter() for pool_id in self._pool_by_id
            }
            claim_sets = list(self._owners.values())
            claim_sets.extend(
                reservation.claims
                for reservation in self._reservations.values()
                if reservation.state is ReservationState.PROVISIONAL
            )
            for claims in claim_sets:
                self._validate_claims(claims)
                for claim in claims.claims:
                    expected[claim.pool_id] += claim.units
                    expected_by_lifetime[claim.pool_id][claim.lifetime] += claim.units
            if expected != self._used:
                raise AssertionError(
                    f"resource conservation mismatch: expected={expected}, used={self._used}"
                )
            if expected_by_lifetime != self._used_by_lifetime:
                raise AssertionError("resource lifetime conservation mismatch")
            for pool_id, used in self._used.items():
                if used < 0 or used > self._pool_by_id[pool_id].capacity:
                    raise AssertionError(f"pool {pool_id} used={used} is outside capacity")

    def _fit_locked(self, claims: ResourceClaimSet) -> ResourceFit:
        blocks: list[ResourceBlock] = []
        requested_by_pool: Counter[str] = Counter()
        requested_by_key: Counter[tuple[str, ClaimLifetime]] = Counter()
        for claim in claims.claims:
            pool = self._pool_by_id.get(claim.pool_id)
            if pool is None:
                blocks.append(
                    ResourceBlock(
                        claim.pool_id,
                        claim.units,
                        0,
                        0,
                        claim.lifetime,
                        True,
                    )
                )
                continue
            if claim.lifetime not in pool.lifetimes:
                blocks.append(
                    ResourceBlock(
                        claim.pool_id,
                        claim.units,
                        self._used[claim.pool_id],
                        pool.capacity,
                        claim.lifetime,
                        True,
                    )
                )
                continue
            requested_by_pool[claim.pool_id] += claim.units
            requested_by_key[(claim.pool_id, claim.lifetime)] += claim.units
        for pool_id, units in sorted(requested_by_pool.items()):
            pool = self._pool_by_id[pool_id]
            if self._used[pool_id] + units <= pool.capacity:
                continue
            lifetime = next(
                lifetime
                for candidate_pool, lifetime in requested_by_key
                if candidate_pool == pool_id
            )
            blocks.append(
                ResourceBlock(
                    pool_id,
                    units,
                    self._used[pool_id],
                    pool.capacity,
                    lifetime,
                    units > pool.capacity,
                )
            )
        return ResourceFit(
            fits=not blocks,
            impossible=any(block.impossible for block in blocks),
            blocking=tuple(blocks),
        )

    def _validate_claims(self, claims: ResourceClaimSet) -> None:
        for claim in claims.claims:
            pool = self._pool_by_id.get(claim.pool_id)
            if pool is None:
                raise ResourceUnavailable(
                    ResourceBlock(
                        claim.pool_id,
                        claim.units,
                        0,
                        0,
                        claim.lifetime,
                        True,
                    ),
                    f"claim references unknown resource {claim.pool_id!r}",
                )
            if claim.lifetime not in pool.lifetimes:
                raise ValueError(
                    f"resource {claim.pool_id} does not accept lifetime {claim.lifetime.value}"
                )

    def _apply_claim_totals(self, claims: ResourceClaimSet, *, sign: int) -> None:
        for claim in claims.claims:
            self._used[claim.pool_id] += sign * claim.units
            self._used_by_lifetime[claim.pool_id][claim.lifetime] += sign * claim.units
            if self._used_by_lifetime[claim.pool_id][claim.lifetime] == 0:
                del self._used_by_lifetime[claim.pool_id][claim.lifetime]
            if self._used[claim.pool_id] < 0:
                raise AssertionError(f"resource pool {claim.pool_id} underflowed")
            self._high_water[claim.pool_id] = max(
                self._high_water[claim.pool_id], self._used[claim.pool_id]
            )

    def _record_rejection(self, fit: ResourceFit) -> None:
        self._stats["rejections"] += 1
        for block in fit.blocking:
            self._blocking_counts[block.resource] += 1

    def _current_reservation(self, reservation: ResourceReservation) -> ResourceReservation:
        try:
            return self._reservations[reservation.reservation_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown resource reservation {reservation.reservation_id!r}"
            ) from exc


@dataclass(slots=True)
class PendingAdmission:
    request_id: int
    claims: ResourceClaimSet
    owner_id: str
    bypass_count: int = 0
    blocking_resources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AdmissionGrant:
    request_id: int
    owner_id: str
    reservation: ResourceReservation


class FitAwareAdmissionController:
    """Bounded-lookahead admission with deterministic starvation drain mode."""

    def __init__(
        self,
        ledger: ResourceLedger,
        *,
        lookahead: int = 32,
        max_bypasses: int = 8,
    ) -> None:
        if int(lookahead) <= 0:
            raise ValueError("lookahead must be positive")
        if int(max_bypasses) <= 0:
            raise ValueError("max_bypasses must be positive")
        self.ledger = ledger
        self.lookahead = int(lookahead)
        self.max_bypasses = int(max_bypasses)
        self._order: deque[int] = deque()
        self._pending: dict[int, PendingAdmission] = {}
        self._drain_mode_request_id: int | None = None
        self._admitted_total = 0
        self._cancelled_total = 0
        self._bypass_total = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def pending_request_ids(self) -> tuple[int, ...]:
        return tuple(self._ordered_pending_ids())

    def enqueue(
        self,
        request_id: int,
        claims: ResourceClaimSet,
        *,
        owner_id: str | None = None,
    ) -> PendingAdmission:
        rid = int(request_id)
        if rid < 0:
            raise ValueError("request_id must be non-negative")
        if rid in self._pending:
            raise ValueError(f"request_id {rid} is already pending")
        fit = self.ledger.fit(claims)
        if fit.impossible:
            raise ResourceUnavailable(fit.blocking[0])
        pending = PendingAdmission(
            request_id=rid,
            claims=claims,
            owner_id=f"request:{rid}" if owner_id is None else str(owner_id),
            blocking_resources=fit.blocking_resources,
        )
        self._pending[rid] = pending
        self._order.append(rid)
        return pending

    def cancel(self, request_id: int) -> bool:
        rid = int(request_id)
        if self._pending.pop(rid, None) is None:
            return False
        if self._drain_mode_request_id == rid:
            self._drain_mode_request_id = None
        self._cancelled_total += 1
        return True

    def pending_state(self, request_id: int) -> PendingAdmission:
        return self._pending[int(request_id)]

    def admit(self, *, max_items: int) -> tuple[AdmissionGrant, ...]:
        limit = int(max_items)
        if limit <= 0:
            raise ValueError("max_items must be positive")
        admitted: list[AdmissionGrant] = []
        while len(admitted) < limit and self._pending:
            ordered = self._ordered_pending_ids()
            if not ordered:
                break
            head = self._pending[ordered[0]]
            head_fit = self.ledger.fit(head.claims)
            head.blocking_resources = head_fit.blocking_resources
            if head.bypass_count >= self.max_bypasses and not head_fit.fits:
                self._drain_mode_request_id = head.request_id
                break
            selected: PendingAdmission | None = None
            for request_id in ordered[: self.lookahead]:
                candidate = self._pending[request_id]
                fit = self.ledger.fit(candidate.claims)
                candidate.blocking_resources = fit.blocking_resources
                if fit.fits:
                    selected = candidate
                    break
            if selected is None:
                break
            reservation = self.ledger.reserve_provisional(
                selected.claims,
                reservation_id=f"admission:{selected.request_id}",
            )
            committed = self.ledger.commit(reservation, owner_id=selected.owner_id)
            self._pending.pop(selected.request_id)
            if selected.request_id != head.request_id:
                head.bypass_count += 1
                self._bypass_total += 1
            else:
                self._drain_mode_request_id = None
            admitted.append(
                AdmissionGrant(selected.request_id, selected.owner_id, committed)
            )
            self._admitted_total += 1
        return tuple(admitted)

    def snapshot(self) -> dict[str, Any]:
        ordered = self._ordered_pending_ids()
        pending_payload = []
        for request_id in ordered:
            pending = self._pending[request_id]
            fit = self.ledger.fit(pending.claims)
            pending.blocking_resources = fit.blocking_resources
            pending_payload.append(
                {
                    "request_id": pending.request_id,
                    "owner_id": pending.owner_id,
                    "bypass_count": pending.bypass_count,
                    "blocking_resources": list(pending.blocking_resources),
                    "claim_units": pending.claims.units_by_pool(),
                }
            )
        return {
            "pending_count": len(pending_payload),
            "lookahead": self.lookahead,
            "max_bypasses": self.max_bypasses,
            "drain_mode_request_id": self._drain_mode_request_id,
            "admitted_total": self._admitted_total,
            "cancelled_total": self._cancelled_total,
            "bypass_total": self._bypass_total,
            "pending": pending_payload,
        }

    def _ordered_pending_ids(self) -> list[int]:
        while self._order and self._order[0] not in self._pending:
            self._order.popleft()
        return [request_id for request_id in self._order if request_id in self._pending]


class LedgerAdmissionCoordinator:
    """Bind backend estimates and fit-aware grants to scheduler callbacks."""

    def __init__(
        self,
        controller: FitAwareAdmissionController,
        estimator: Callable[[Any], ResourceClaimSet],
    ) -> None:
        self.controller = controller
        self.ledger = controller.ledger
        self.estimator = estimator
        self._grants: dict[int, AdmissionGrant] = {}

    def plan_admission(
        self,
        pending_requests: Sequence[Any],
        *,
        max_items: int,
    ) -> tuple[int, ...]:
        pending_by_id = {
            int(request.request_id): request for request in pending_requests
        }
        for request_id in self.controller.pending_request_ids:
            if request_id not in pending_by_id:
                self.controller.cancel(request_id)
        known = set(self.controller.pending_request_ids) | set(self._grants)
        for request_id, request in pending_by_id.items():
            if request_id in known:
                continue
            claims = self.estimator(request)
            self.controller.enqueue(
                request_id,
                claims,
                owner_id=f"request:{request_id}",
            )
        grants = self.controller.admit(max_items=max_items)
        for grant in grants:
            self._grants[grant.request_id] = grant
        return tuple(grant.request_id for grant in grants)

    def reserve_admission(self, request: Any) -> None:
        request_id = int(request.request_id)
        if request_id not in self._grants:
            raise RuntimeError(
                f"request_id {request_id} has no committed resource grant"
            )

    def rollback_admission(self, request: Any) -> None:
        request_id = int(request.request_id)
        grant = self._grants.pop(request_id, None)
        if grant is not None and self.ledger.has_owner(grant.owner_id):
            self.ledger.release(
                grant.owner_id,
                operation_id=f"admission-rollback:{request_id}",
            )

    def reclaim_request(self, request: Any) -> ResourceDelta | None:
        request_id = int(request.request_id)
        grant = self._grants.pop(request_id, None)
        if grant is None:
            self.controller.cancel(request_id)
            return None
        if not self.ledger.has_owner(grant.owner_id):
            return None
        return self.ledger.release(
            grant.owner_id,
            operation_id=f"request-reclaim:{request_id}",
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "admission": self.controller.snapshot(),
            "ledger": self.ledger.snapshot(),
            "granted_request_ids": sorted(self._grants),
        }


def _changes_by_pool_and_lifetime(
    delta: ResourceDelta,
) -> Counter[tuple[str, ClaimLifetime]]:
    changed: Counter[tuple[str, ClaimLifetime]] = Counter()
    for change in delta.changes:
        changed[(change.pool_id, change.lifetime)] += change.units
    return changed


__all__ = [
    "AdmissionGrant",
    "FitAwareAdmissionController",
    "LedgerAdmissionCoordinator",
    "PendingAdmission",
    "ReservationState",
    "ResourceBlock",
    "ResourceFit",
    "ResourceLedger",
    "ResourceReservation",
    "ResourceUnavailable",
]
