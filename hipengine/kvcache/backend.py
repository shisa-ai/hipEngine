"""Format-neutral Generation-2 KV-cache backend host contracts.

The scheduler sees named resource vectors and opaque storage views.  Codec,
retention, and tier implementations own every format-specific estimate and
plane layout; none of those formulas belong in the engine or scheduler.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence, runtime_checkable

from hipengine.kvcache.spans import KVLiveSpans


class ClaimLifetime(str, Enum):
    """Ownership interval for one named resource claim."""

    LOAD = "load"
    LEASE = "lease"
    WORK_ITEM = "work_item"
    TRANSACTION = "transaction"
    CACHE = "cache"


class ClaimConfidence(str, Enum):
    """How tightly a backend can bound one claim."""

    EXACT = "exact"
    BOUNDED = "bounded"
    UNKNOWN = "unknown"


def _required_text(value: object, label: str) -> str:
    text = str(value)
    if not text or text != text.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return text


@dataclass(frozen=True, slots=True)
class ResourceClaim:
    """One non-negative quantity in an atomic resource claim set."""

    pool_id: str
    units: int
    lifetime: ClaimLifetime = ClaimLifetime.LEASE
    confidence: ClaimConfidence = ClaimConfidence.EXACT

    def __post_init__(self) -> None:
        object.__setattr__(self, "pool_id", _required_text(self.pool_id, "pool_id"))
        units = int(self.units)
        if units <= 0:
            raise ValueError("resource claim units must be positive")
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "lifetime", ClaimLifetime(self.lifetime))
        object.__setattr__(self, "confidence", ClaimConfidence(self.confidence))

    @property
    def key(self) -> tuple[str, ClaimLifetime]:
        return self.pool_id, self.lifetime


@dataclass(frozen=True, slots=True)
class ResourceClaimSet:
    """An all-or-nothing vector of backend-produced resource quantities."""

    claim_id: str
    claims: tuple[ResourceClaim, ...] = ()
    request_id: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _required_text(self.claim_id, "claim_id"))
        normalized = tuple(
            claim if isinstance(claim, ResourceClaim) else ResourceClaim(*claim)
            for claim in self.claims
        )
        keys = tuple(claim.key for claim in normalized)
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate resource claim for pool_id/lifetime")
        if self.request_id is not None and int(self.request_id) < 0:
            raise ValueError("request_id must be non-negative when set")
        object.__setattr__(self, "request_id", None if self.request_id is None else int(self.request_id))
        object.__setattr__(self, "claims", normalized)

    @classmethod
    def from_mapping(
        cls,
        claim_id: str,
        units: dict[str, int],
        *,
        request_id: int | None = None,
        lifetime: ClaimLifetime = ClaimLifetime.LEASE,
        confidence: ClaimConfidence = ClaimConfidence.EXACT,
    ) -> "ResourceClaimSet":
        claims: list[ResourceClaim] = []
        for pool_id, raw_amount in sorted(units.items()):
            amount = int(raw_amount)
            if amount < 0:
                raise ValueError("resource mapping units must be non-negative")
            if amount:
                claims.append(ResourceClaim(pool_id, amount, lifetime, confidence))
        return cls(
            claim_id=claim_id,
            request_id=request_id,
            claims=tuple(claims),
        )

    def units_by_pool(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for claim in self.claims:
            result[claim.pool_id] = result.get(claim.pool_id, 0) + claim.units
        return result

    def entries(self) -> dict[tuple[str, ClaimLifetime], ResourceClaim]:
        return {claim.key: claim for claim in self.claims}

    def with_claim_id(self, claim_id: str) -> "ResourceClaimSet":
        return ResourceClaimSet(claim_id=claim_id, claims=self.claims, request_id=self.request_id)

    def apply(self, delta: "ResourceDelta", *, claim_id: str | None = None) -> "ResourceClaimSet":
        """Return this ownership vector after one validated signed delta."""

        if (
            self.request_id is not None
            and delta.request_id is not None
            and self.request_id != delta.request_id
        ):
            raise ValueError("resource delta request_id does not match claim ownership")
        entries = self.entries()
        for change in delta.changes:
            current = entries.get(change.key)
            current_units = 0 if current is None else current.units
            updated_units = current_units + change.units
            if updated_units < 0:
                raise ValueError(
                    f"resource delta underflows {change.pool_id}/{change.lifetime.value}"
                )
            if updated_units == 0:
                entries.pop(change.key, None)
            else:
                confidence = ClaimConfidence.EXACT if current is None else current.confidence
                entries[change.key] = ResourceClaim(
                    change.pool_id,
                    updated_units,
                    change.lifetime,
                    confidence,
                )
        return ResourceClaimSet(
            claim_id=self.claim_id if claim_id is None else claim_id,
            request_id=self.request_id,
            claims=tuple(entries[key] for key in sorted(entries, key=lambda item: (item[0], item[1].value))),
        )


@dataclass(frozen=True, slots=True)
class ResourceChange:
    """One signed ownership change; positive acquires and negative releases."""

    pool_id: str
    units: int
    lifetime: ClaimLifetime = ClaimLifetime.LEASE

    def __post_init__(self) -> None:
        object.__setattr__(self, "pool_id", _required_text(self.pool_id, "pool_id"))
        units = int(self.units)
        if units == 0:
            raise ValueError("resource change units must be non-zero")
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "lifetime", ClaimLifetime(self.lifetime))

    @property
    def key(self) -> tuple[str, ClaimLifetime]:
        return self.pool_id, self.lifetime


@dataclass(frozen=True, slots=True)
class ResourceDelta:
    """Atomic signed changes produced by one backend operation."""

    operation_id: str
    lease_id: str
    changes: tuple[ResourceChange, ...] = ()
    request_id: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _required_text(self.operation_id, "operation_id"))
        object.__setattr__(self, "lease_id", _required_text(self.lease_id, "lease_id"))
        normalized = tuple(
            change if isinstance(change, ResourceChange) else ResourceChange(*change)
            for change in self.changes
        )
        keys = tuple(change.key for change in normalized)
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate resource change for pool_id/lifetime")
        if self.request_id is not None and int(self.request_id) < 0:
            raise ValueError("request_id must be non-negative when set")
        object.__setattr__(self, "request_id", None if self.request_id is None else int(self.request_id))
        object.__setattr__(self, "changes", normalized)

    @classmethod
    def between(
        cls,
        *,
        operation_id: str,
        lease_id: str,
        before: ResourceClaimSet,
        after: ResourceClaimSet,
        request_id: int | None = None,
    ) -> "ResourceDelta":
        if (
            before.request_id is not None
            and after.request_id is not None
            and before.request_id != after.request_id
        ):
            raise ValueError("before/after resource claims belong to different requests")
        before_entries = before.entries()
        after_entries = after.entries()
        keys = sorted(set(before_entries) | set(after_entries), key=lambda item: (item[0], item[1].value))
        changes = []
        for key in keys:
            prior = 0 if key not in before_entries else before_entries[key].units
            current = 0 if key not in after_entries else after_entries[key].units
            if current != prior:
                changes.append(ResourceChange(key[0], current - prior, key[1]))
        inferred_request_id = (
            before.request_id if before.request_id is not None else after.request_id
        )
        return cls(
            operation_id=operation_id,
            lease_id=lease_id,
            request_id=inferred_request_id if request_id is None else request_id,
            changes=tuple(changes),
        )

    def units_by_pool(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for change in self.changes:
            result[change.pool_id] = result.get(change.pool_id, 0) + change.units
        return result


@dataclass(frozen=True, slots=True)
class KVBackendSpec:
    """Immutable resolved retention + codec + tier identity for one replica."""

    topology_key: str
    hot_codec_key: str
    tier_key: str
    layout_fingerprint: str
    artifact_fingerprint: str
    prefix_mode: str
    transaction_mode: str
    kernel_bundle_key: str
    physical_widths: tuple[int, ...] = (1,)
    max_context_tokens: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "topology_key",
            "hot_codec_key",
            "tier_key",
            "layout_fingerprint",
            "artifact_fingerprint",
            "kernel_bundle_key",
        ):
            object.__setattr__(self, field_name, _required_text(getattr(self, field_name), field_name))
        prefix_mode = _required_text(self.prefix_mode, "prefix_mode")
        if prefix_mode not in {"unsupported", "immutable_pages", "snapshot_overlay"}:
            raise ValueError("prefix_mode must be unsupported, immutable_pages, or snapshot_overlay")
        transaction_mode = _required_text(self.transaction_mode, "transaction_mode")
        if transaction_mode not in {"journal", "scratch", "snapshot", "unsupported"}:
            raise ValueError("transaction_mode must be journal, scratch, snapshot, or unsupported")
        widths = tuple(int(width) for width in self.physical_widths)
        if not widths or any(width <= 0 for width in widths):
            raise ValueError("physical_widths must contain positive widths")
        if tuple(sorted(set(widths))) != widths:
            raise ValueError("physical_widths must be unique and strictly increasing")
        if widths[0] != 1:
            raise ValueError("physical_widths must retain a c1 route")
        if self.max_context_tokens is not None and int(self.max_context_tokens) <= 0:
            raise ValueError("max_context_tokens must be positive when set")
        object.__setattr__(self, "prefix_mode", prefix_mode)
        object.__setattr__(self, "transaction_mode", transaction_mode)
        object.__setattr__(self, "physical_widths", widths)
        object.__setattr__(
            self,
            "max_context_tokens",
            None if self.max_context_tokens is None else int(self.max_context_tokens),
        )

    @property
    def compatibility_key(self) -> tuple[str, ...]:
        return (
            self.topology_key,
            self.hot_codec_key,
            self.tier_key,
            self.layout_fingerprint,
            self.artifact_fingerprint,
            self.kernel_bundle_key,
        )

    @property
    def fingerprint(self) -> str:
        payload = {
            "compatibility_key": self.compatibility_key,
            "prefix_mode": self.prefix_mode,
            "transaction_mode": self.transaction_mode,
            "physical_widths": self.physical_widths,
            "max_context_tokens": self.max_context_tokens,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class KVPoolSpec:
    """One stable named capacity in a backend pool set."""

    pool_id: str
    capacity: int
    unit: str = "units"
    plane_role: str = "payload"
    lifetimes: tuple[ClaimLifetime, ...] = (
        ClaimLifetime.LOAD,
        ClaimLifetime.LEASE,
        ClaimLifetime.WORK_ITEM,
        ClaimLifetime.TRANSACTION,
        ClaimLifetime.CACHE,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "pool_id", _required_text(self.pool_id, "pool_id"))
        capacity = int(self.capacity)
        if capacity <= 0:
            raise ValueError("pool capacity must be positive")
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "unit", _required_text(self.unit, "unit"))
        object.__setattr__(self, "plane_role", _required_text(self.plane_role, "plane_role"))
        lifetimes = tuple(ClaimLifetime(lifetime) for lifetime in self.lifetimes)
        if not lifetimes or len(lifetimes) != len(set(lifetimes)):
            raise ValueError("pool lifetimes must be non-empty and unique")
        object.__setattr__(self, "lifetimes", lifetimes)


@dataclass(frozen=True, slots=True)
class KVPoolPlan:
    """Complete stable pool set declared before engine admission starts."""

    backend_fingerprint: str
    generation: int
    pools: tuple[KVPoolSpec, ...]
    load_claims: ResourceClaimSet | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend_fingerprint",
            _required_text(self.backend_fingerprint, "backend_fingerprint"),
        )
        generation = int(self.generation)
        if generation <= 0:
            raise ValueError("pool generation must be positive")
        if not self.pools:
            raise ValueError("KVPoolPlan must declare at least one pool")
        normalized = tuple(pool if isinstance(pool, KVPoolSpec) else KVPoolSpec(*pool) for pool in self.pools)
        pool_ids = tuple(pool.pool_id for pool in normalized)
        if len(pool_ids) != len(set(pool_ids)):
            raise ValueError("duplicate pool_id in KVPoolPlan")
        if self.load_claims is not None:
            pool_by_id = {pool.pool_id: pool for pool in normalized}
            for claim in self.load_claims.claims:
                if claim.pool_id not in pool_by_id:
                    raise ValueError("load claim references a pool outside KVPoolPlan")
                pool = pool_by_id[claim.pool_id]
                if claim.lifetime is not ClaimLifetime.LOAD:
                    raise ValueError("KVPoolPlan load claims must use load lifetime")
                if ClaimLifetime.LOAD not in pool.lifetimes:
                    raise ValueError("load claim references a pool that rejects load lifetime")
                if claim.units > pool.capacity:
                    raise ValueError("load claim exceeds its planned pool capacity")
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "pools", normalized)

    def pool(self, pool_id: str) -> KVPoolSpec:
        wanted = str(pool_id)
        for pool in self.pools:
            if pool.pool_id == wanted:
                return pool
        raise KeyError(wanted)


@dataclass(frozen=True, slots=True)
class KVLease:
    """Typed backend ownership held by one independently schedulable child."""

    lease_id: str
    request_id: int
    backend_fingerprint: str
    generation: int
    claims: ResourceClaimSet
    shared_handles: tuple[str, ...] = ()
    private_handles: tuple[str, ...] = ()
    writable_tail_handle: str | None = None
    metadata_handles: tuple[str, ...] = ()
    growth_credits: ResourceClaimSet | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_id", _required_text(self.lease_id, "lease_id"))
        request_id = int(self.request_id)
        if request_id < 0:
            raise ValueError("request_id must be non-negative")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(
            self,
            "backend_fingerprint",
            _required_text(self.backend_fingerprint, "backend_fingerprint"),
        )
        generation = int(self.generation)
        if generation <= 0:
            raise ValueError("lease generation must be positive")
        object.__setattr__(self, "generation", generation)
        if self.claims.request_id is not None and self.claims.request_id != request_id:
            raise ValueError("lease claims request_id must match lease request_id")
        handle_fields = (
            ("shared_handles", self.shared_handles),
            ("private_handles", self.private_handles),
            ("metadata_handles", self.metadata_handles),
        )
        all_handles: list[str] = []
        for field_name, raw_handles in handle_fields:
            handles = tuple(
                _required_text(handle, field_name.removesuffix("s").replace("_", " "))
                for handle in raw_handles
            )
            if len(handles) != len(set(handles)):
                raise ValueError(f"{field_name} must be unique")
            object.__setattr__(self, field_name, handles)
            all_handles.extend(handles)
        tail = (
            None
            if self.writable_tail_handle is None
            else _required_text(self.writable_tail_handle, "writable tail handle")
        )
        if tail is not None:
            all_handles.append(tail)
        if len(all_handles) != len(set(all_handles)):
            raise ValueError("KV lease handles must not alias across ownership roles")
        object.__setattr__(self, "writable_tail_handle", tail)
        if self.growth_credits is not None:
            if (
                self.growth_credits.request_id is not None
                and self.growth_credits.request_id != request_id
            ):
                raise ValueError("growth-credit request_id must match lease request_id")
            if any(
                claim.lifetime is not ClaimLifetime.LEASE
                for claim in self.growth_credits.claims
            ):
                raise ValueError("growth credits must use lease lifetime")


@dataclass(frozen=True, slots=True)
class KVPlaneView:
    """Stable raw-pointer view for one backend-defined payload/metadata plane."""

    role: str
    dtype: str
    ptr: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _required_text(self.role, "plane role"))
        object.__setattr__(self, "dtype", _required_text(self.dtype, "plane dtype"))
        ptr = int(self.ptr)
        if ptr <= 0:
            raise ValueError("plane ptr must be positive")
        shape = tuple(int(item) for item in self.shape)
        strides = tuple(int(item) for item in self.strides)
        if not shape or any(item <= 0 for item in shape):
            raise ValueError("plane shape must contain positive dimensions")
        if len(strides) != len(shape) or any(item < 0 for item in strides):
            raise ValueError("plane strides must align with shape and be non-negative")
        object.__setattr__(self, "ptr", ptr)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "strides", strides)


@dataclass(frozen=True, slots=True)
class KVStorageView:
    """Opaque, generation-checked plane bundle selected by layout key."""

    layout_key: str
    generation: int
    planes: tuple[KVPlaneView, ...]
    artifact_fingerprint: str
    metadata_descriptor_ptr: int = 0
    metadata_descriptor_bytes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout_key", _required_text(self.layout_key, "layout_key"))
        generation = int(self.generation)
        if generation <= 0:
            raise ValueError("storage-view generation must be positive")
        if not self.planes:
            raise ValueError("KVStorageView must expose at least one plane")
        normalized = tuple(
            plane if isinstance(plane, KVPlaneView) else KVPlaneView(*plane)
            for plane in self.planes
        )
        roles = tuple(plane.role for plane in normalized)
        if len(roles) != len(set(roles)):
            raise ValueError("duplicate plane role in KVStorageView")
        descriptor_ptr = int(self.metadata_descriptor_ptr)
        descriptor_bytes = int(self.metadata_descriptor_bytes)
        if descriptor_ptr < 0 or descriptor_bytes < 0:
            raise ValueError("metadata descriptor pointer/size must be non-negative")
        if bool(descriptor_ptr) != bool(descriptor_bytes):
            raise ValueError("metadata descriptor pointer and size must both be zero or non-zero")
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "planes", normalized)
        object.__setattr__(
            self,
            "artifact_fingerprint",
            _required_text(self.artifact_fingerprint, "artifact_fingerprint"),
        )
        object.__setattr__(self, "metadata_descriptor_ptr", descriptor_ptr)
        object.__setattr__(self, "metadata_descriptor_bytes", descriptor_bytes)

    def plane(self, role: str) -> KVPlaneView:
        wanted = str(role)
        for plane in self.planes:
            if plane.role == wanted:
                return plane
        raise KeyError(wanted)


@dataclass(frozen=True, slots=True)
class KVBatchView:
    """Liveness, storage planes, and registered kernels for one work item."""

    live_spans: KVLiveSpans
    storage_view: KVStorageView
    kernel_bundle_key: str
    execution_compatibility_key: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kernel_bundle_key",
            _required_text(self.kernel_bundle_key, "kernel_bundle_key"),
        )
        key = tuple(_required_text(item, "execution compatibility key item") for item in self.execution_compatibility_key)
        if not key:
            raise ValueError("execution_compatibility_key must not be empty")
        object.__setattr__(self, "execution_compatibility_key", key)


@runtime_checkable
class KVCacheBackend(Protocol):
    """Scheduler-facing protocol for one resolved KV backend composition."""

    spec: KVBackendSpec

    def plan_pools(self, load_plan: Any) -> KVPoolPlan: ...

    def estimate(self, request: Any, prefix: Any, stage: Any) -> ResourceClaimSet: ...

    def reserve(self, claims: ResourceClaimSet) -> KVLease: ...

    def prepare(self, work_item: Any) -> KVBatchView: ...

    def begin_transaction(self, rows: Sequence[Any], draft: Any) -> Any: ...

    def commit(self, operation: Any, result: Any) -> ResourceDelta: ...

    def rollback(self, operation: Any) -> ResourceDelta: ...

    def reclaim(self, lease: KVLease) -> ResourceDelta: ...

    def prefix_lookup(self, tokens: Sequence[int]) -> Any: ...

    def maintenance(self, budget: Any) -> list[Any]: ...


__all__ = [
    "ClaimConfidence",
    "ClaimLifetime",
    "KVBackendSpec",
    "KVBatchView",
    "KVCacheBackend",
    "KVLease",
    "KVPlaneView",
    "KVPoolPlan",
    "KVPoolSpec",
    "KVStorageView",
    "ResourceChange",
    "ResourceClaim",
    "ResourceClaimSet",
    "ResourceDelta",
]
