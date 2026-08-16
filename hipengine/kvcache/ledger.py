"""Deterministic host resource ledger for Generation-2 KV backends.

The ledger understands only backend-declared pool IDs, units, capacities,
claim lifetimes, and operation deltas.  It deliberately contains no storage
codec formulas.  One engine-service owner is expected to call it at commit
barriers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from hipengine.kvcache.backend import (
    ClaimConfidence,
    KVLease,
    KVPoolPlan,
    KVPoolSpec,
    LeaseState,
    ResourceClaim,
    ResourceClaimSet,
    ResourceChange,
    ResourceDelta,
)


class ResourceLedgerError(RuntimeError):
    """Base class for deterministic admission/accounting failures."""


class DuplicateOperationError(ResourceLedgerError):
    pass


class UnknownPoolError(ResourceLedgerError):
    def __init__(self, pool_id: str) -> None:
        self.pool_id = str(pool_id)
        super().__init__(f"unknown resource pool {self.pool_id!r}")


class ResourceCapacityError(ResourceLedgerError):
    def __init__(self, pool_id: str, *, requested: int, available: int) -> None:
        self.pool_id = str(pool_id)
        self.requested = int(requested)
        self.available = int(available)
        super().__init__(
            f"resource pool {self.pool_id!r} has {self.available} available; "
            f"requested {self.requested}"
        )


class ResourceUnitError(ResourceLedgerError):
    pass


class UnknownResourceClaimError(ResourceLedgerError):
    pass


class InvalidOperationStateError(ResourceLedgerError):
    pass


class InjectedLedgerFailure(ResourceLedgerError):
    pass


@dataclass(frozen=True, slots=True)
class ResourcePoolSnapshot:
    pool_id: str
    unit: str
    capacity: int
    provisional: int
    committed: int
    available: int


@dataclass(frozen=True, slots=True)
class ResourceLedgerSnapshot:
    backend_fingerprint: str
    generation: int
    pools: tuple[ResourcePoolSnapshot, ...]
    active_operations: int
    provisional_operations: int
    committed_operations: int

    def pool(self, pool_id: str) -> ResourcePoolSnapshot:
        wanted = str(pool_id)
        for pool in self.pools:
            if pool.pool_id == wanted:
                return pool
        raise KeyError(wanted)


@dataclass(slots=True)
class _PoolState:
    spec: KVPoolSpec
    provisional: int = 0
    committed: int = 0

    @property
    def available(self) -> int:
        return self.spec.capacity - self.provisional - self.committed


@dataclass(slots=True)
class _LeaseRecord:
    lease: KVLease


class ResourceLedger:
    """Atomic multi-pool ownership ledger for one resolved backend plan."""

    def __init__(self, plan: KVPoolPlan) -> None:
        self.plan = plan
        self._pools = {pool.pool_id: _PoolState(pool) for pool in plan.pools}
        self._operations: dict[str, _LeaseRecord] = {}

    def snapshot(self) -> ResourceLedgerSnapshot:
        pools = tuple(
            ResourcePoolSnapshot(
                pool_id=spec.pool_id,
                unit=spec.unit,
                capacity=spec.capacity,
                provisional=self._pools[spec.pool_id].provisional,
                committed=self._pools[spec.pool_id].committed,
                available=self._pools[spec.pool_id].available,
            )
            for spec in self.plan.pools
        )
        provisional = sum(
            record.lease.state is LeaseState.PROVISIONAL
            for record in self._operations.values()
        )
        committed = sum(
            record.lease.state is LeaseState.COMMITTED
            for record in self._operations.values()
        )
        return ResourceLedgerSnapshot(
            backend_fingerprint=self.plan.backend_fingerprint,
            generation=self.plan.generation,
            pools=pools,
            active_operations=len(self._operations),
            provisional_operations=provisional,
            committed_operations=committed,
        )

    def lease(self, operation_id: str) -> KVLease:
        return self._record(operation_id).lease

    def reserve(
        self,
        operation_id: str,
        *,
        request_id: int,
        claims: ResourceClaimSet,
        fail_after_claims: int | None = None,
    ) -> KVLease:
        """Provisionally reserve every claim or leave the ledger unchanged."""

        operation = self._operation_id(operation_id)
        if operation in self._operations:
            raise DuplicateOperationError(f"operation_id {operation!r} already exists")
        if fail_after_claims is not None:
            boundary = int(fail_after_claims)
            if boundary <= 0 or boundary > len(claims.claims):
                raise ValueError("fail_after_claims must identify a claim boundary")
        else:
            boundary = None
        self._validate_claims(claims)
        requested_by_pool: dict[str, int] = {}
        for claim in claims.claims:
            requested_by_pool[claim.pool_id] = (
                requested_by_pool.get(claim.pool_id, 0) + claim.amount
            )
        for pool_id, requested in requested_by_pool.items():
            pool = self._pools[pool_id]
            if requested > pool.available:
                raise ResourceCapacityError(
                    pool_id,
                    requested=requested,
                    available=pool.available,
                )

        lease = KVLease(
            lease_id=operation,
            request_id=request_id,
            backend_fingerprint=self.plan.backend_fingerprint,
            claims=claims,
            generation=self.plan.generation,
            state=LeaseState.PROVISIONAL,
        )
        mutated: list[ResourceClaim] = []
        try:
            for index, claim in enumerate(claims.claims, start=1):
                self._pools[claim.pool_id].provisional += claim.amount
                mutated.append(claim)
                if boundary == index:
                    raise InjectedLedgerFailure(f"injected failure at claim boundary {index}")
        except BaseException:
            for claim in reversed(mutated):
                self._pools[claim.pool_id].provisional -= claim.amount
            raise

        self._operations[operation] = _LeaseRecord(lease)
        self._assert_conserved()
        return lease

    def commit(self, operation_id: str) -> KVLease:
        """Atomically publish one provisional claim set as committed ownership."""

        record = self._record(operation_id)
        if record.lease.state is not LeaseState.PROVISIONAL:
            raise InvalidOperationStateError(
                f"operation {record.lease.lease_id!r} is already committed"
            )
        for claim in record.lease.claims.claims:
            pool = self._pools[claim.pool_id]
            pool.provisional -= claim.amount
            pool.committed += claim.amount
        record.lease = replace(record.lease, state=LeaseState.COMMITTED)
        self._assert_conserved()
        return record.lease

    def rollback(self, operation_id: str, *, reason: str = "rollback") -> ResourceDelta:
        """Release a provisional operation; committed ownership cannot roll back."""

        record = self._record(operation_id)
        if record.lease.state is LeaseState.COMMITTED:
            raise InvalidOperationStateError(
                f"operation {record.lease.lease_id!r} is committed and must be released"
            )
        return self._release_record(record, reason=reason)

    def release(self, operation_id: str, *, reason: str = "release") -> ResourceDelta:
        """Release every holding for a provisional or committed operation."""

        return self._release_record(self._record(operation_id), reason=reason)

    def apply(self, delta: ResourceDelta) -> KVLease:
        """Apply an atomic signed growth/release delta to committed ownership."""

        record = self._record(delta.operation_id)
        if record.lease.state is not LeaseState.COMMITTED:
            raise InvalidOperationStateError(
                f"operation {record.lease.lease_id!r} must be committed before applying a delta"
            )
        current = {
            (claim.pool_id, claim.lifetime): claim
            for claim in record.lease.claims.claims
        }
        updated_amounts = {
            identity: claim.amount for identity, claim in current.items()
        }
        metadata = dict(current)
        net_changes_by_pool: dict[str, int] = {}

        for change in delta.changes:
            self._validate_change(change)
            identity = (change.pool_id, change.lifetime)
            existing = current.get(identity)
            if existing is not None and existing.unit != change.unit:
                raise ResourceUnitError(
                    f"resource change for {change.pool_id!r} does not match the held unit"
                )
            held = updated_amounts.get(identity, 0)
            next_amount = held + change.amount
            if next_amount < 0:
                raise InvalidOperationStateError(
                    f"operation {record.lease.lease_id!r} does not own "
                    f"{abs(change.amount)} units from pool {change.pool_id!r} "
                    f"at lifetime {change.lifetime.value!r}"
                )
            updated_amounts[identity] = next_amount
            net_changes_by_pool[change.pool_id] = (
                net_changes_by_pool.get(change.pool_id, 0) + change.amount
            )
            if existing is None and change.amount > 0:
                metadata[identity] = ResourceClaim(
                    change.pool_id,
                    change.amount,
                    unit=change.unit,
                    lifetime=change.lifetime,
                    confidence=change.confidence,
                )

        for pool_id, net_change in net_changes_by_pool.items():
            if net_change > self._pools[pool_id].available:
                raise ResourceCapacityError(
                    pool_id,
                    requested=net_change,
                    available=self._pools[pool_id].available,
                )

        if not any(updated_amounts.values()):
            raise InvalidOperationStateError(
                "a delta cannot release the complete lease; use release()"
            )

        for change in delta.changes:
            self._pools[change.pool_id].committed += change.amount

        updated_claims = tuple(
            ResourceClaim(
                identity[0],
                amount,
                unit=metadata[identity].unit,
                lifetime=identity[1],
                confidence=metadata[identity].confidence,
            )
            for identity, amount in updated_amounts.items()
            if amount > 0
        )
        record.lease = replace(
            record.lease,
            claims=ResourceClaimSet(
                stage=record.lease.claims.stage,
                claims=updated_claims,
            ),
        )
        self._assert_conserved()
        return record.lease

    def _release_record(self, record: _LeaseRecord, *, reason: str) -> ResourceDelta:
        changes = tuple(
            ResourceChange(
                claim.pool_id,
                -claim.amount,
                unit=claim.unit,
                lifetime=claim.lifetime,
                confidence=claim.confidence,
                reason=reason,
            )
            for claim in record.lease.claims.claims
        )
        delta = ResourceDelta(
            operation_id=record.lease.lease_id,
            changes=changes,
        )
        category = (
            "provisional"
            if record.lease.state is LeaseState.PROVISIONAL
            else "committed"
        )
        for claim in record.lease.claims.claims:
            pool = self._pools[claim.pool_id]
            setattr(pool, category, getattr(pool, category) - claim.amount)
        del self._operations[record.lease.lease_id]
        self._assert_conserved()
        return delta

    def _record(self, operation_id: str) -> _LeaseRecord:
        operation = self._operation_id(operation_id)
        try:
            return self._operations[operation]
        except KeyError as exc:
            raise InvalidOperationStateError(
                f"unknown operation {operation!r}"
            ) from exc

    @staticmethod
    def _operation_id(operation_id: str) -> str:
        operation = str(operation_id).strip()
        if not operation:
            raise ValueError("operation_id must not be empty")
        return operation

    def _validate_claims(self, claims: ResourceClaimSet) -> None:
        for claim in claims.claims:
            if claim.confidence is ClaimConfidence.UNKNOWN:
                raise UnknownResourceClaimError(
                    f"pool {claim.pool_id!r} has unknown confidence"
                )
            self._validate_pool_unit(claim.pool_id, claim.unit)

    def _validate_change(self, change: ResourceChange) -> None:
        if change.confidence is ClaimConfidence.UNKNOWN:
            raise UnknownResourceClaimError(
                f"pool {change.pool_id!r} has unknown confidence"
            )
        self._validate_pool_unit(change.pool_id, change.unit)

    def _validate_pool_unit(self, pool_id: str, unit: str) -> None:
        pool = self._pools.get(pool_id)
        if pool is None:
            raise UnknownPoolError(pool_id)
        if pool.spec.unit != unit:
            raise ResourceUnitError(
                f"resource pool {pool_id!r} uses unit {pool.spec.unit!r}, not {unit!r}"
            )

    def _assert_conserved(self) -> None:
        for pool in self._pools.values():
            if pool.provisional < 0 or pool.committed < 0 or pool.available < 0:
                raise AssertionError(f"resource pool {pool.spec.pool_id!r} has negative ownership")
            if pool.available + pool.provisional + pool.committed != pool.spec.capacity:
                raise AssertionError(f"resource pool {pool.spec.pool_id!r} violates conservation")
