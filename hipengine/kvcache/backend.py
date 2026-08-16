"""Format-neutral Generation-2 KV-cache backend host contracts.

These value objects describe storage and resource ownership without teaching the
scheduler how any codec computes bytes.  Concrete backends produce pool plans,
claims, storage views, and registered kernel bundles; the engine consumes the
common protocol.
"""

from __future__ import annotations

import hashlib
import json
import operator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence, runtime_checkable

from hipengine.kvcache.spans import KVLiveSpans


class PrefixMode(str, Enum):
    UNSUPPORTED = "unsupported"
    IMMUTABLE_PAGES = "immutable_pages"
    SNAPSHOT_OVERLAY = "snapshot_overlay"


class TransactionMode(str, Enum):
    JOURNAL = "journal"
    SCRATCH = "scratch"
    SNAPSHOT = "snapshot"
    UNSUPPORTED = "unsupported"


class ClaimLifetime(str, Enum):
    LOAD = "load"
    LEASE = "lease"
    WORK_ITEM = "work_item"
    TRANSACTION = "transaction"
    CACHE = "cache"


class ClaimConfidence(str, Enum):
    EXACT = "exact"
    BOUNDED = "bounded"
    UNKNOWN = "unknown"


class LeaseState(str, Enum):
    PROVISIONAL = "provisional"
    COMMITTED = "committed"


def _required_text(value: object, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _non_negative_int(value: object, name: str) -> int:
    parsed = _integer(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _positive_int(value: object, name: str) -> int:
    parsed = _non_negative_int(value, name)
    if parsed == 0:
        raise ValueError(f"{name} must be positive")
    return parsed


@dataclass(frozen=True, slots=True)
class KVBackendSpec:
    """Immutable identity for one validated topology/codec/tier composition."""

    topology_key: str
    hot_codec_key: str
    tier_key: str
    layout_fingerprint: str
    artifact_fingerprint: str
    prefix_mode: PrefixMode | str
    transaction_mode: TransactionMode | str
    kernel_bundle_key: str

    def __post_init__(self) -> None:
        for name in (
            "topology_key",
            "hot_codec_key",
            "tier_key",
            "layout_fingerprint",
            "artifact_fingerprint",
            "kernel_bundle_key",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(self, "prefix_mode", PrefixMode(self.prefix_mode))
        object.__setattr__(self, "transaction_mode", TransactionMode(self.transaction_mode))

    @property
    def execution_compatibility_prefix(self) -> tuple[str, ...]:
        return (
            self.topology_key,
            self.hot_codec_key,
            self.tier_key,
            self.layout_fingerprint,
            self.artifact_fingerprint,
            self.kernel_bundle_key,
        )

    @property
    def identity_fingerprint(self) -> str:
        payload = {
            "artifact_fingerprint": self.artifact_fingerprint,
            "hot_codec_key": self.hot_codec_key,
            "kernel_bundle_key": self.kernel_bundle_key,
            "layout_fingerprint": self.layout_fingerprint,
            "prefix_mode": self.prefix_mode.value,
            "tier_key": self.tier_key,
            "topology_key": self.topology_key,
            "transaction_mode": self.transaction_mode.value,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class KVPoolSpec:
    """One named capacity domain whose unit is declared by its backend."""

    pool_id: str
    capacity: int
    unit: str = "bytes"

    def __post_init__(self) -> None:
        object.__setattr__(self, "pool_id", _required_text(self.pool_id, "pool_id"))
        object.__setattr__(self, "capacity", _positive_int(self.capacity, "capacity"))
        object.__setattr__(self, "unit", _required_text(self.unit, "unit"))


@dataclass(frozen=True, slots=True)
class KVPoolPlan:
    """Complete load-time set of backend-owned resource pools."""

    backend_fingerprint: str
    pools: tuple[KVPoolSpec, ...]
    generation: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend_fingerprint",
            _required_text(self.backend_fingerprint, "backend_fingerprint"),
        )
        object.__setattr__(self, "pools", tuple(self.pools))
        if not self.pools:
            raise ValueError("KVPoolPlan must contain at least one pool")
        pool_ids = tuple(pool.pool_id for pool in self.pools)
        if len(set(pool_ids)) != len(pool_ids):
            raise ValueError("KVPoolPlan contains duplicate pool_id values")
        object.__setattr__(self, "generation", _non_negative_int(self.generation, "generation"))

    @property
    def pool_ids(self) -> tuple[str, ...]:
        return tuple(pool.pool_id for pool in self.pools)

    def pool(self, pool_id: str) -> KVPoolSpec:
        wanted = str(pool_id)
        for pool in self.pools:
            if pool.pool_id == wanted:
                return pool
        raise KeyError(wanted)


@dataclass(frozen=True, slots=True)
class ResourceClaim:
    """Positive capacity requested from one named pool."""

    pool_id: str
    amount: int
    unit: str = "bytes"
    lifetime: ClaimLifetime | str = ClaimLifetime.LEASE
    confidence: ClaimConfidence | str = ClaimConfidence.EXACT

    def __post_init__(self) -> None:
        object.__setattr__(self, "pool_id", _required_text(self.pool_id, "pool_id"))
        object.__setattr__(self, "amount", _positive_int(self.amount, "amount"))
        object.__setattr__(self, "unit", _required_text(self.unit, "unit"))
        object.__setattr__(self, "lifetime", ClaimLifetime(self.lifetime))
        object.__setattr__(self, "confidence", ClaimConfidence(self.confidence))


@dataclass(frozen=True, slots=True)
class ResourceClaimSet:
    """Atomic named-resource vector for one backend stage estimate."""

    stage: str
    claims: tuple[ResourceClaim, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _required_text(self.stage, "stage"))
        object.__setattr__(self, "claims", tuple(self.claims))
        if not self.claims:
            raise ValueError("ResourceClaimSet must contain at least one claim")
        identities = tuple((claim.pool_id, claim.lifetime) for claim in self.claims)
        if len(set(identities)) != len(identities):
            raise ValueError("ResourceClaimSet contains a duplicate pool/lifetime claim")

    @property
    def pool_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(claim.pool_id for claim in self.claims))

    def amount(
        self,
        pool_id: str,
        *,
        lifetime: ClaimLifetime | str | None = None,
    ) -> int:
        wanted = str(pool_id)
        parsed_lifetime = None if lifetime is None else ClaimLifetime(lifetime)
        return sum(
            claim.amount
            for claim in self.claims
            if claim.pool_id == wanted
            and (parsed_lifetime is None or claim.lifetime is parsed_lifetime)
        )

    def claim(
        self,
        pool_id: str,
        *,
        lifetime: ClaimLifetime | str = ClaimLifetime.LEASE,
    ) -> ResourceClaim:
        wanted = str(pool_id)
        wanted_lifetime = ClaimLifetime(lifetime)
        for claim in self.claims:
            if claim.pool_id == wanted and claim.lifetime is wanted_lifetime:
                return claim
        raise KeyError((wanted, wanted_lifetime.value))


@dataclass(frozen=True, slots=True)
class KVLease:
    """Immutable snapshot of one operation's provisional or committed holdings."""

    lease_id: str
    request_id: int
    backend_fingerprint: str
    claims: ResourceClaimSet
    generation: int
    state: LeaseState | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_id", _required_text(self.lease_id, "lease_id"))
        object.__setattr__(self, "request_id", _non_negative_int(self.request_id, "request_id"))
        object.__setattr__(
            self,
            "backend_fingerprint",
            _required_text(self.backend_fingerprint, "backend_fingerprint"),
        )
        object.__setattr__(self, "generation", _non_negative_int(self.generation, "generation"))
        object.__setattr__(self, "state", LeaseState(self.state))


@dataclass(frozen=True, slots=True)
class ResourceChange:
    """Signed ownership change; positive acquires and negative releases."""

    pool_id: str
    amount: int
    unit: str = "bytes"
    lifetime: ClaimLifetime | str = ClaimLifetime.LEASE
    confidence: ClaimConfidence | str = ClaimConfidence.EXACT
    reason: str = "unspecified"

    def __post_init__(self) -> None:
        object.__setattr__(self, "pool_id", _required_text(self.pool_id, "pool_id"))
        parsed = _integer(self.amount, "ResourceChange amount")
        if parsed == 0:
            raise ValueError("ResourceChange amount must be non-zero")
        object.__setattr__(self, "amount", parsed)
        object.__setattr__(self, "unit", _required_text(self.unit, "unit"))
        object.__setattr__(self, "lifetime", ClaimLifetime(self.lifetime))
        object.__setattr__(self, "confidence", ClaimConfidence(self.confidence))
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))


@dataclass(frozen=True, slots=True)
class ResourceDelta:
    """Atomic ownership delta tied to one lease/operation ID."""

    operation_id: str
    changes: tuple[ResourceChange, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _required_text(self.operation_id, "operation_id"))
        object.__setattr__(self, "changes", tuple(self.changes))
        if not self.changes:
            raise ValueError("ResourceDelta must contain at least one change")
        identities = tuple((change.pool_id, change.lifetime) for change in self.changes)
        if len(set(identities)) != len(identities):
            raise ValueError("ResourceDelta contains a duplicate pool/lifetime change")


@dataclass(frozen=True, slots=True)
class KVStoragePlane:
    """Stable raw-pointer view of one backend-owned storage plane."""

    role: str
    ptr: int
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _required_text(self.role, "role"))
        object.__setattr__(self, "ptr", _positive_int(self.ptr, "ptr"))
        object.__setattr__(self, "dtype", _required_text(self.dtype, "dtype"))
        shape = tuple(_positive_int(value, "shape dimension") for value in self.shape)
        strides = tuple(_positive_int(value, "stride") for value in self.strides)
        if not shape:
            raise ValueError("storage plane shape must not be empty")
        if len(shape) != len(strides):
            raise ValueError("storage plane strides must match shape rank")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "strides", strides)


@dataclass(frozen=True, slots=True)
class KVStorageView:
    """Generation-checked set of backend storage planes and device metadata."""

    layout_key: str
    generation: int
    planes: tuple[KVStoragePlane, ...]
    artifact_fingerprint: str
    device_metadata_ptr: int = 0
    device_metadata_nbytes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout_key", _required_text(self.layout_key, "layout_key"))
        object.__setattr__(self, "generation", _non_negative_int(self.generation, "generation"))
        object.__setattr__(self, "planes", tuple(self.planes))
        if not self.planes:
            raise ValueError("KVStorageView must contain at least one storage plane")
        roles = tuple(plane.role for plane in self.planes)
        if len(set(roles)) != len(roles):
            raise ValueError("KVStorageView contains duplicate storage plane role values")
        object.__setattr__(
            self,
            "artifact_fingerprint",
            _required_text(self.artifact_fingerprint, "artifact_fingerprint"),
        )
        metadata_ptr = _non_negative_int(self.device_metadata_ptr, "device_metadata_ptr")
        metadata_nbytes = _non_negative_int(
            self.device_metadata_nbytes,
            "device_metadata_nbytes",
        )
        if (metadata_ptr == 0) != (metadata_nbytes == 0):
            raise ValueError("device metadata pointer and size must either both be zero or both be positive")
        object.__setattr__(self, "device_metadata_ptr", metadata_ptr)
        object.__setattr__(self, "device_metadata_nbytes", metadata_nbytes)

    def plane(self, role: str) -> KVStoragePlane:
        wanted = str(role)
        for plane in self.planes:
            if plane.role == wanted:
                return plane
        raise KeyError(wanted)


@dataclass(frozen=True, slots=True)
class KVBatchView:
    """Liveness spans plus storage and registered-kernel compatibility identity."""

    spans: KVLiveSpans
    storage: KVStorageView
    kernel_bundle_key: str
    execution_compatibility_key: tuple[Any, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kernel_bundle_key",
            _required_text(self.kernel_bundle_key, "kernel_bundle_key"),
        )
        key = tuple(self.execution_compatibility_key)
        if not key:
            raise ValueError("execution_compatibility_key must not be empty")
        object.__setattr__(self, "execution_compatibility_key", key)


@runtime_checkable
class KVCacheBackend(Protocol):
    """Scheduler-facing contract implemented by every topology/codec composition."""

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
