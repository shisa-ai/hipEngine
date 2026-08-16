"""Deterministic host-only Generation-2 engine and resource simulator.

This is an executable architecture oracle, not a production model runner.  It
exercises independent child completion, refill, physical-width lowering, and
format-neutral per-pool conservation against several structurally distinct fake
KV backends without requiring ROCm.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, replace
from typing import Any, Sequence

from hipengine.generation.concurrency2 import (
    ChildPhase,
    ChildRequest,
    CollectedOutput,
    EngineOutput,
    OutputCollector,
    OutputKind,
)
from hipengine.kvcache.backend import (
    KVBatchView,
    KVCacheBackend,
    KVLease,
    KVPoolPlan,
    ResourceClaimSet,
    ResourceDelta,
)
from hipengine.kvcache.simulated import FAKE_KV_BACKEND_KINDS, create_fake_kv_backend


class SimulatedResourceCapacityError(MemoryError):
    """Atomic fake-ledger rejection naming the first unavailable pool."""

    def __init__(self, pool_id: str, *, requested: int, used: int, capacity: int) -> None:
        self.pool_id = str(pool_id)
        self.requested = int(requested)
        self.used = int(used)
        self.capacity = int(capacity)
        super().__init__(
            f"resource {self.pool_id} cannot fit {self.requested} units "
            f"with {self.used}/{self.capacity} already used"
        )


class SimulatedResourceLedger:
    """Atomic named-capacity ledger used by the C2-0 conformance simulator."""

    def __init__(self, plan: KVPoolPlan) -> None:
        self.plan = plan
        self._pool_by_id = {pool.pool_id: pool for pool in plan.pools}
        self._used = {pool.pool_id: 0 for pool in plan.pools}
        self._high_water = dict(self._used)
        self._owners: dict[str, ResourceClaimSet] = {}
        self.reserve_count = 0
        self.release_count = 0
        self.delta_count = 0
        self.rejection_count = 0

    def reserve(self, owner_id: str, claims: ResourceClaimSet) -> None:
        owner = str(owner_id)
        if not owner:
            raise ValueError("ledger owner_id must not be empty")
        if owner in self._owners:
            raise ValueError(f"ledger owner {owner!r} already exists")
        requested = claims.units_by_pool()
        self._validate_claim_entries(claims)
        for pool_id in sorted(requested):
            amount = requested[pool_id]
            pool = self._pool_by_id[pool_id]
            if self._used[pool_id] + amount > pool.capacity:
                self.rejection_count += 1
                raise SimulatedResourceCapacityError(
                    pool_id,
                    requested=amount,
                    used=self._used[pool_id],
                    capacity=pool.capacity,
                )
        self._owners[owner] = claims
        for pool_id, amount in requested.items():
            self._used[pool_id] += amount
            self._high_water[pool_id] = max(self._high_water[pool_id], self._used[pool_id])
        self.reserve_count += 1

    def apply_delta(self, owner_id: str, delta: ResourceDelta) -> ResourceClaimSet:
        owner = str(owner_id)
        if delta.lease_id != owner:
            raise ValueError("resource delta lease_id does not match ledger owner")
        try:
            current = self._owners[owner]
        except KeyError as exc:
            raise KeyError(f"unknown ledger owner {owner!r}") from exc
        updated = current.apply(delta, claim_id=f"{current.claim_id}@{delta.operation_id}")
        self._validate_claim_entries(updated)
        changed_by_pool = delta.units_by_pool()
        for pool_id in sorted(changed_by_pool):
            change = changed_by_pool[pool_id]
            if pool_id not in self._pool_by_id:
                raise KeyError(f"resource delta references unknown pool {pool_id!r}")
            next_used = self._used[pool_id] + change
            if next_used < 0:
                raise ValueError(f"resource delta underflows ledger pool {pool_id}")
            capacity = self._pool_by_id[pool_id].capacity
            if next_used > capacity:
                self.rejection_count += 1
                raise SimulatedResourceCapacityError(
                    pool_id,
                    requested=max(0, change),
                    used=self._used[pool_id],
                    capacity=capacity,
                )
        self._owners[owner] = updated
        for pool_id, change in changed_by_pool.items():
            self._used[pool_id] += change
            self._high_water[pool_id] = max(self._high_water[pool_id], self._used[pool_id])
        self.delta_count += 1
        return updated

    def release(self, owner_id: str) -> ResourceClaimSet:
        owner = str(owner_id)
        claims = self._owners.pop(owner)
        for pool_id, amount in claims.units_by_pool().items():
            self._used[pool_id] -= amount
            if self._used[pool_id] < 0:  # defensive: assert_conserved gives a richer report.
                raise AssertionError(f"ledger pool {pool_id} underflowed during release")
        self.release_count += 1
        return claims

    def drop_empty(self, owner_id: str) -> None:
        owner = str(owner_id)
        claims = self._owners.get(owner)
        if claims is None:
            return
        if claims.claims:
            raise ValueError(f"cannot drop non-empty ledger owner {owner!r}")
        self._owners.pop(owner)
        self.release_count += 1

    def has_owner(self, owner_id: str) -> bool:
        return str(owner_id) in self._owners

    def owner_claims(self, owner_id: str) -> ResourceClaimSet:
        return self._owners[str(owner_id)]

    def assert_conserved(self) -> None:
        expected = {pool_id: 0 for pool_id in self._pool_by_id}
        for owner, claims in self._owners.items():
            self._validate_claim_entries(claims)
            for pool_id, amount in claims.units_by_pool().items():
                expected[pool_id] += amount
                if expected[pool_id] > self._pool_by_id[pool_id].capacity:
                    raise AssertionError(f"owner aggregation exceeds {pool_id} capacity at {owner}")
        if expected != self._used:
            raise AssertionError(f"resource conservation mismatch: owners={expected}, used={self._used}")
        for pool_id, used in self._used.items():
            if used < 0 or used > self._pool_by_id[pool_id].capacity:
                raise AssertionError(f"pool {pool_id} used={used} is outside capacity")

    def snapshot(self) -> dict[str, dict[str, int | str]]:
        return {
            pool_id: {
                "capacity": pool.capacity,
                "used": self._used[pool_id],
                "free": pool.capacity - self._used[pool_id],
                "high_water": self._high_water[pool_id],
                "unit": pool.unit,
            }
            for pool_id, pool in sorted(self._pool_by_id.items())
        }

    def _validate_claim_entries(self, claims: ResourceClaimSet) -> None:
        for claim in claims.claims:
            try:
                pool = self._pool_by_id[claim.pool_id]
            except KeyError as exc:
                raise KeyError(f"claim references unknown pool {claim.pool_id!r}") from exc
            if claim.lifetime not in pool.lifetimes:
                raise ValueError(
                    f"pool {claim.pool_id} does not accept lifetime {claim.lifetime.value}"
                )


@dataclass(frozen=True, slots=True)
class SimulatedWorkItem:
    request_ids: tuple[int, ...]
    context_lengths: tuple[int, ...]
    physical_width: int


@dataclass(frozen=True, slots=True)
class SimulatedCommitOperation:
    """Format-neutral scheduler commit metadata consumed by a fake backend."""

    operation_id: str
    lease: KVLease
    current_tokens: int
    next_tokens: int


@dataclass(slots=True)
class _RequestRecord:
    request: ChildRequest
    collector: OutputCollector
    phase: ChildPhase = ChildPhase.QUEUED
    resident_slot: int | None = None
    lease: KVLease | None = None
    generated_tokens: tuple[int, ...] = ()


class DeterministicEngineSimulator:
    """One queue/scheduler/output lifecycle shared by all fake KV backends."""

    command_queue_type = deque

    def __init__(
        self,
        backend: KVCacheBackend,
        *,
        resident_capacity: int,
        physical_widths: Sequence[int] | None = None,
    ) -> None:
        capacity = int(resident_capacity)
        if capacity <= 0:
            raise ValueError("resident_capacity must be positive")
        if not isinstance(backend, KVCacheBackend):
            raise TypeError("backend must implement KVCacheBackend")
        self.backend = backend
        self.resident_capacity = capacity
        configured_widths = (
            backend.spec.physical_widths
            if physical_widths is None
            else tuple(int(width) for width in physical_widths)
        )
        if not configured_widths or configured_widths[0] != 1:
            raise ValueError("physical widths must include c1 as their first route")
        if tuple(sorted(set(configured_widths))) != tuple(configured_widths):
            raise ValueError("physical widths must be unique and increasing")
        if any(width not in backend.spec.physical_widths for width in configured_widths):
            raise ValueError("physical width is not declared by the backend")
        self.physical_widths = tuple(configured_widths)
        plan = backend.plan_pools(None)
        if plan.backend_fingerprint != backend.spec.fingerprint:
            raise ValueError("backend pool plan fingerprint does not match resolved spec")
        self.ledger = SimulatedResourceLedger(plan)
        self.command_queue: deque[int] = deque()
        self._queued: dict[int, None] = {}
        self._records: dict[int, _RequestRecord] = {}
        self._resident_by_slot: dict[int, int] = {}
        self._admission_order: list[int] = []
        self._terminal_order: list[int] = []
        self._scheduled_rows = 0
        self._scheduled_steps = 0
        self._physical_width_counts: Counter[int] = Counter()
        self._last_physical_groups: tuple[int, ...] = ()
        self._last_batch_views: tuple[KVBatchView, ...] = ()
        self._operation_counts: Counter[str] = Counter()
        self._operation_sequence = 0
        self._compaction_count = 0

    @property
    def queued_count(self) -> int:
        return len(self._queued)

    @property
    def resident_count(self) -> int:
        return len(self._resident_by_slot)

    @property
    def resident_slots(self) -> tuple[int, ...]:
        return tuple(sorted(self._resident_by_slot))

    @property
    def resident_request_ids(self) -> tuple[int, ...]:
        return tuple(self._resident_by_slot[slot] for slot in sorted(self._resident_by_slot))

    @property
    def admission_order(self) -> tuple[int, ...]:
        return tuple(self._admission_order)

    @property
    def terminal_order(self) -> tuple[int, ...]:
        return tuple(self._terminal_order)

    @property
    def operation_counts(self) -> dict[str, int]:
        return dict(self._operation_counts)

    @property
    def idle(self) -> bool:
        return not self._queued and not self._resident_by_slot

    def phase(self, request_id: int) -> ChildPhase:
        return self._records[int(request_id)].phase

    def submit(self, request: ChildRequest, collector: OutputCollector) -> None:
        if not isinstance(request, ChildRequest):
            raise TypeError("submit requires ChildRequest")
        if not isinstance(collector, OutputCollector):
            raise TypeError("collector must implement OutputCollector")
        if request.request_id in self._records:
            raise ValueError(f"request_id {request.request_id} already exists")
        collector.bind(request.request_id)
        self._records[request.request_id] = _RequestRecord(request=request, collector=collector)
        self._queued[request.request_id] = None
        self.command_queue.append(request.request_id)

    def cancel(self, request_id: int, *, reason: str = "cancel") -> bool:
        rid = int(request_id)
        finish_reason = str(reason)
        if not finish_reason or finish_reason != finish_reason.strip():
            raise ValueError("cancel reason must be a non-empty trimmed string")
        record = self._records.get(rid)
        if record is None or record.phase is ChildPhase.TERMINAL:
            return False
        if record.phase is ChildPhase.QUEUED:
            self._queued.pop(rid, None)
            self._finish(record, reason=finish_reason)
        else:
            self._finish(record, reason=finish_reason)
        self._admit()
        self.assert_invariants()
        return True

    def step(self) -> bool:
        """Admit, execute every current resident once, commit, reclaim, refill."""

        self._admit()
        request_ids = self.resident_request_ids
        if not request_ids:
            return False
        groups = _decompose_requests(request_ids, self.physical_widths)
        batch_views: list[KVBatchView] = []
        self._last_physical_groups = tuple(width for width, _ in groups)
        for physical_width, group_ids in groups:
            self._physical_width_counts[physical_width] += 1
            contexts = tuple(
                len(self._records[request_id].request.prompt_tokens)
                + len(self._records[request_id].generated_tokens)
                for request_id in group_ids
            )
            view = self.backend.prepare(
                SimulatedWorkItem(
                    request_ids=group_ids,
                    context_lengths=contexts,
                    physical_width=physical_width,
                )
            )
            if view.storage_view.generation != self.ledger.plan.generation:
                raise RuntimeError("backend prepared a stale storage-view generation")
            if view.kernel_bundle_key != self.backend.spec.kernel_bundle_key:
                raise RuntimeError("backend prepared the wrong kernel bundle")
            batch_views.append(view)
            for request_id in group_ids:
                record = self._records[request_id]
                if record.phase is ChildPhase.DECODE:
                    self._advance(record)
                    self._scheduled_rows += 1
        self._last_batch_views = tuple(batch_views)
        self._scheduled_steps += 1
        self._admit()
        self.assert_invariants()
        return True

    def run_until_idle(self, *, max_steps: int = 100_000) -> int:
        steps = 0
        while not self.idle:
            progressed = self.step()
            steps += 1
            if not progressed:
                raise RuntimeError("simulated engine stalled with queued work")
            if steps > int(max_steps):
                raise RuntimeError("simulated engine exceeded max_steps")
        return steps

    def compact(self) -> tuple[tuple[int, int, int], ...]:
        """Compact resident slots at a deterministic commit barrier."""

        moves: list[tuple[int, int, int]] = []
        ordered = tuple(
            self._resident_by_slot[slot]
            for slot in sorted(self._resident_by_slot)
        )
        replacement: dict[int, int] = {}
        for new_slot, request_id in enumerate(ordered):
            record = self._records[request_id]
            assert record.resident_slot is not None
            old_slot = record.resident_slot
            replacement[new_slot] = request_id
            if old_slot != new_slot:
                moves.append((request_id, old_slot, new_slot))
                record.resident_slot = new_slot
        self._resident_by_slot = replacement
        if moves:
            self._compaction_count += 1
        self.assert_invariants()
        return tuple(moves)

    def take_result(self, request_id: int) -> CollectedOutput:
        rid = int(request_id)
        record = self._records[rid]
        if record.phase is not ChildPhase.TERMINAL or record.collector.result is None:
            raise RuntimeError("request result is not terminal")
        result = record.collector.result
        assert result is not None
        self._records.pop(rid)
        return result

    def snapshot(self) -> dict[str, Any]:
        return {
            "backend": {
                "topology": self.backend.spec.topology_key,
                "codec": self.backend.spec.hot_codec_key,
                "tier": self.backend.spec.tier_key,
                "fingerprint": self.backend.spec.fingerprint,
                "pool_generation": self.ledger.plan.generation,
            },
            "counters": {
                "queued": self.queued_count,
                "resident": self.resident_count,
                "scheduled_rows": self._scheduled_rows,
                "scheduled_steps": self._scheduled_steps,
                "physical_widths": dict(sorted(self._physical_width_counts.items())),
                "last_physical_groups": self._last_physical_groups,
                "completion_records": sum(
                    record.phase is ChildPhase.TERMINAL for record in self._records.values()
                ),
                "compactions": self._compaction_count,
                "operations": dict(self._operation_counts),
            },
            "pools": self.ledger.snapshot(),
            "requests": {
                request_id: {
                    "phase": record.phase.value,
                    "resident_slot": record.resident_slot,
                    "generated_tokens": len(record.generated_tokens),
                }
                for request_id, record in sorted(self._records.items())
            },
        }

    def assert_invariants(self) -> None:
        queued_records = {
            request_id
            for request_id, record in self._records.items()
            if record.phase is ChildPhase.QUEUED
        }
        if queued_records != set(self._queued):
            raise AssertionError(
                f"queued request mismatch: records={queued_records}, queue={set(self._queued)}"
            )
        if len(self._resident_by_slot) > self.resident_capacity:
            raise AssertionError("resident capacity exceeded")
        if len(set(self._resident_by_slot.values())) != len(self._resident_by_slot):
            raise AssertionError("request appears in multiple resident slots")
        for slot, request_id in self._resident_by_slot.items():
            if slot < 0 or slot >= self.resident_capacity:
                raise AssertionError("resident slot is outside capacity")
            record = self._records[request_id]
            if record.phase not in {ChildPhase.PREFILL, ChildPhase.DECODE, ChildPhase.VERIFY}:
                raise AssertionError("non-resident phase owns a resident slot")
            if record.resident_slot != slot or record.lease is None:
                raise AssertionError("resident record slot/lease mismatch")
            if not self.ledger.has_owner(record.lease.lease_id):
                raise AssertionError("resident lease is absent from resource ledger")
            if self.ledger.owner_claims(record.lease.lease_id).claims != record.lease.claims.claims:
                raise AssertionError("record lease claims differ from ledger ownership")
        for request_id, record in self._records.items():
            if record.phase is ChildPhase.TERMINAL:
                if record.resident_slot is not None or record.lease is not None:
                    raise AssertionError("terminal record retains backend resources")
                if record.collector.result is None:
                    raise AssertionError("terminal record has no published result")
            elif record.phase is ChildPhase.QUEUED:
                if record.resident_slot is not None or record.lease is not None:
                    raise AssertionError("queued record owns resident resources")
            elif request_id not in self._resident_by_slot.values():
                raise AssertionError("resident phase is absent from slot map")
        self.ledger.assert_conserved()

    def _admit(self) -> None:
        while self.resident_count < self.resident_capacity and self._queued:
            request_id = self._next_queued_request_id()
            if request_id is None:
                return
            record = self._records[request_id]
            current_tokens = len(record.request.prompt_tokens)
            claims = self.backend.estimate(
                record.request,
                None,
                {"kind": "admission", "tokens": current_tokens},
            )
            lease = self.backend.reserve(claims)
            if lease.request_id != request_id:
                raise ValueError("backend lease request_id does not match admission request")
            if lease.backend_fingerprint != self.backend.spec.fingerprint:
                raise ValueError("backend lease fingerprint does not match resolved backend")
            if lease.generation != self.ledger.plan.generation:
                raise ValueError("backend lease generation does not match current pool generation")
            try:
                self.ledger.reserve(lease.lease_id, lease.claims)
            except SimulatedResourceCapacityError:
                self.command_queue.appendleft(request_id)
                self._queued[request_id] = None
                return
            slot = self._first_free_slot()
            self._queued.pop(request_id, None)
            record.phase = ChildPhase.DECODE
            record.resident_slot = slot
            record.lease = lease
            self._resident_by_slot[slot] = request_id
            self._admission_order.append(request_id)
            if record.request.max_new_tokens == 0:
                self._finish(record, reason="length")

    def _next_queued_request_id(self) -> int | None:
        while self.command_queue:
            request_id = self.command_queue.popleft()
            if request_id in self._queued:
                return request_id
        return None

    def _first_free_slot(self) -> int:
        for slot in range(self.resident_capacity):
            if slot not in self._resident_by_slot:
                return slot
        raise AssertionError("no free resident slot")

    def _advance(self, record: _RequestRecord) -> None:
        lease = record.lease
        if lease is None:
            raise AssertionError("decode record has no lease")
        current_tokens = len(record.request.prompt_tokens) + len(record.generated_tokens)
        next_tokens = current_tokens + 1
        self._operation_sequence += 1
        operation = SimulatedCommitOperation(
            operation_id=f"commit:{record.request.request_id}:{self._operation_sequence}",
            lease=lease,
            current_tokens=current_tokens,
            next_tokens=next_tokens,
        )
        workspace = self.backend.estimate(
            record.request,
            None,
            {
                "kind": "work_item",
                "current_tokens": current_tokens,
                "next_tokens": next_tokens,
            },
        )
        workspace_owner = f"operation:{operation.operation_id}"
        if workspace.claims:
            self.ledger.reserve(workspace_owner, workspace)
        try:
            delta = self.backend.commit(operation, None)
            operation_kind = delta.operation_id.split(":", 1)[0]
            self._operation_counts[operation_kind] += 1
            updated_claims = self.ledger.apply_delta(lease.lease_id, delta)
            record.lease = replace(lease, claims=updated_claims)
        finally:
            if workspace.claims and self.ledger.has_owner(workspace_owner):
                self.ledger.release(workspace_owner)

        token_index = len(record.generated_tokens)
        token_id = _deterministic_token(token_index)
        accepted = record.collector.publish(
            EngineOutput(
                kind=OutputKind.TOKEN,
                request_id=record.request.request_id,
                token_id=token_id,
                token_index=token_index,
            )
        )
        if not accepted:
            self._finish(record, reason="client_backpressure")
            return
        record.generated_tokens = (*record.generated_tokens, token_id)
        if len(record.generated_tokens) >= record.request.max_new_tokens:
            self._finish(record, reason="length")

    def _finish(self, record: _RequestRecord, *, reason: str) -> None:
        if record.phase is ChildPhase.TERMINAL:
            return
        if record.lease is not None:
            lease = record.lease
            delta = self.backend.reclaim(lease)
            updated = self.ledger.apply_delta(lease.lease_id, delta)
            if updated.claims:
                raise AssertionError("backend reclaim did not release every lease claim")
            self.ledger.drop_empty(lease.lease_id)
        if record.resident_slot is not None:
            owner = self._resident_by_slot.pop(record.resident_slot, None)
            if owner != record.request.request_id:
                raise AssertionError("resident slot ownership changed before terminal reclaim")
        record.lease = None
        record.resident_slot = None
        record.phase = ChildPhase.TERMINAL
        published = record.collector.publish(
            EngineOutput(
                kind=OutputKind.TERMINAL,
                request_id=record.request.request_id,
                generated_token_ids=record.generated_tokens,
                finish_reason=str(reason),
            )
        )
        if not published:
            raise AssertionError("terminal publication must use a non-blocking reserved slot")
        self._terminal_order.append(record.request.request_id)


def _decompose_requests(
    request_ids: Sequence[int],
    widths: Sequence[int],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    remaining = tuple(int(request_id) for request_id in request_ids)
    groups: list[tuple[int, tuple[int, ...]]] = []
    offset = 0
    descending = tuple(reversed(tuple(int(width) for width in widths)))
    while offset < len(remaining):
        count = len(remaining) - offset
        physical_width = next(width for width in descending if width <= count)
        group = remaining[offset : offset + physical_width]
        groups.append((physical_width, group))
        offset += physical_width
    return tuple(groups)


def _deterministic_token(token_index: int) -> int:
    return 10_000 + int(token_index)


def independent_token_ids(request: ChildRequest) -> tuple[int, ...]:
    """Independent c1 oracle for the fake logical model."""

    return tuple(_deterministic_token(index) for index in range(request.max_new_tokens))


__all__ = [
    "FAKE_KV_BACKEND_KINDS",
    "DeterministicEngineSimulator",
    "SimulatedCommitOperation",
    "SimulatedResourceCapacityError",
    "SimulatedResourceLedger",
    "SimulatedWorkItem",
    "create_fake_kv_backend",
    "independent_token_ids",
]
