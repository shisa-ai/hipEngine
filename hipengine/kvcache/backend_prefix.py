"""Generation-checked radix snapshots backed by a global KV pool."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping

from hipengine.kvcache.backend import (
    ClaimLifetime,
    KVBackendSpec,
    KVLease,
    ResourceChange,
    ResourceClaim,
    ResourceClaimSet,
    ResourceDelta,
)
from hipengine.kvcache.global_pool import GlobalKVPoolSet
from hipengine.kvcache.ledger import ResourceLedger
from hipengine.kvcache.radix import RadixCache


@dataclass(frozen=True, slots=True)
class PrefixCompatibilityKey:
    model_artifact_fingerprint: str
    model_revision: str
    adapter_identity: str
    hardware_backend: str
    model_key: str
    weight_quant: str
    backend_fingerprint: str
    backend_artifact_fingerprint: str
    rope_fingerprint: str
    multimodal_input_hash: str

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            value = str(getattr(self, field_name))
            if not value or value != value.strip():
                raise ValueError(
                    f"prefix compatibility {field_name} must be non-empty and trimmed"
                )
            object.__setattr__(self, field_name, value)

    @property
    def fingerprint(self) -> str:
        payload = {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def for_backend(
        cls,
        spec: KVBackendSpec,
        *,
        model_artifact_fingerprint: str,
        model_revision: str,
        adapter_identity: str = "none",
        hardware_backend: str,
        model_key: str,
        weight_quant: str,
        rope_fingerprint: str,
        multimodal_input_hash: str = "none",
    ) -> "PrefixCompatibilityKey":
        return cls(
            model_artifact_fingerprint=model_artifact_fingerprint,
            model_revision=model_revision,
            adapter_identity=adapter_identity,
            hardware_backend=hardware_backend,
            model_key=model_key,
            weight_quant=weight_quant,
            backend_fingerprint=spec.fingerprint,
            backend_artifact_fingerprint=spec.artifact_fingerprint,
            rope_fingerprint=rope_fingerprint,
            multimodal_input_hash=multimodal_input_hash,
        )


@dataclass(frozen=True, slots=True)
class KVSnapshotHandle:
    snapshot_id: str
    scope_fingerprint: str
    backend_fingerprint: str
    artifact_fingerprint: str
    generation: int
    matched_tokens: tuple[int, ...]
    page_ids: tuple[int, ...]
    tenant_id: str
    created_at: float
    last_access_at: float
    expires_at: float | None

    @property
    def matched_token_count(self) -> int:
        return len(self.matched_tokens)


@dataclass(frozen=True, slots=True)
class BackendPrefixMatch:
    snapshot: KVSnapshotHandle | None
    remaining_tokens: tuple[int, ...]
    miss_reason: str | None = None

    @property
    def hit(self) -> bool:
        return self.snapshot is not None

    @property
    def matched_token_count(self) -> int:
        return 0 if self.snapshot is None else self.snapshot.matched_token_count


class BackendRadixCache:
    """Radix index whose durable entries own global-pool pages and ledger units."""

    def __init__(
        self,
        *,
        spec: KVBackendSpec,
        generation: int,
        block_size: int,
        pool: GlobalKVPoolSet,
        ledger: ResourceLedger,
        page_pool_ids: tuple[str, ...],
        max_cached_pages: int,
        ttl_seconds: float | None = None,
        tenant_page_quotas: Mapping[str, int] | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        if int(generation) <= 0:
            raise ValueError("prefix cache generation must be positive")
        if int(block_size) <= 0:
            raise ValueError("prefix cache block_size must be positive")
        if int(max_cached_pages) <= 0:
            raise ValueError("max_cached_pages must be positive")
        if ttl_seconds is not None and float(ttl_seconds) <= 0:
            raise ValueError("ttl_seconds must be positive when set")
        if pool.generation != int(generation):
            raise ValueError("prefix cache generation must match global pool")
        if pool.backend_fingerprint != spec.artifact_fingerprint:
            raise ValueError("prefix cache pool artifact does not match backend")
        pool_ids = tuple(str(pool_id) for pool_id in page_pool_ids)
        if not pool_ids or len(pool_ids) != len(set(pool_ids)):
            raise ValueError("page_pool_ids must be non-empty and unique")
        for pool_id in pool_ids:
            pool_spec = ledger.plan.pool(pool_id)
            if ClaimLifetime.CACHE not in pool_spec.lifetimes:
                raise ValueError(f"resource pool {pool_id} does not support cache lifetime")
        quotas = {
            str(tenant): int(pages)
            for tenant, pages in (tenant_page_quotas or {}).items()
        }
        if any(not tenant or pages <= 0 for tenant, pages in quotas.items()):
            raise ValueError("tenant cache quotas require non-empty names and positive pages")
        self.spec = spec
        self.generation = int(generation)
        self.block_size = int(block_size)
        self.pool = pool
        self.ledger = ledger
        self.page_pool_ids = pool_ids
        self.max_cached_pages = int(max_cached_pages)
        self.ttl_seconds = None if ttl_seconds is None else float(ttl_seconds)
        self.tenant_page_quotas = quotas
        self._clock = clock
        self._radices: dict[str, RadixCache] = {}
        self._snapshots: dict[tuple[str, tuple[int, ...]], KVSnapshotHandle] = {}
        self._next_snapshot_id = 0
        self._hits = 0
        self._misses = 0
        self._stale_misses = 0
        self._expired_evictions = 0
        self._pressure_evictions = 0
        self._quota_evictions = 0
        self._lock = threading.RLock()
        self._cache_owner_id = f"prefix-cache:{self.spec.fingerprint}"

    def publish(
        self,
        scope: PrefixCompatibilityKey,
        lease: KVLease,
        tokens: tuple[int, ...] | list[int],
        *,
        tenant_id: str = "default",
        now: float | None = None,
    ) -> KVSnapshotHandle | None:
        token_tuple = tuple(int(token) for token in tokens)
        tenant = str(tenant_id)
        if not tenant:
            raise ValueError("tenant_id must be non-empty")
        self._validate_scope(scope)
        self._validate_lease(lease)
        complete_pages = min(
            len(token_tuple) // self.block_size,
            len(self.pool.lease(lease.lease_id).logical_page_ids),
        )
        if complete_pages <= 0:
            return None
        matched_tokens = token_tuple[: complete_pages * self.block_size]
        page_ids = self.pool.lease(lease.lease_id).logical_page_ids[:complete_pages]
        key = (scope.fingerprint, matched_tokens)
        timestamp = self._now(now)
        with self._lock:
            existing = self._snapshots.get(key)
            if existing is not None:
                if existing.page_ids != page_ids:
                    raise ValueError("prefix snapshot token key conflicts with page identity")
                touched = replace(existing, last_access_at=timestamp)
                self._snapshots[key] = touched
                return touched
            for page_id in page_ids:
                page = self.pool.page(page_id)
                if lease.lease_id not in page.active_lease_ids:
                    raise ValueError(f"KV page {page_id} is not active in source lease")
            for existing_snapshot in self._snapshots.values():
                if existing_snapshot.scope_fingerprint != scope.fingerprint:
                    continue
                common_tokens = min(
                    len(existing_snapshot.matched_tokens),
                    len(matched_tokens),
                )
                if (
                    existing_snapshot.matched_tokens[:common_tokens]
                    != matched_tokens[:common_tokens]
                ):
                    continue
                common_pages = common_tokens // self.block_size
                if existing_snapshot.page_ids[:common_pages] != page_ids[:common_pages]:
                    raise ValueError("prefix snapshot lineage has conflicting page IDs")
            newly_cached = tuple(
                page_id
                for page_id in page_ids
                if self.pool.page(page_id).cache_references == 0
            )
            transfer_claims = self._page_claims(
                len(newly_cached),
                lifetime=ClaimLifetime.LEASE,
                claim_id=f"cache-publish:{lease.request_id}",
            )
            if transfer_claims.claims:
                self.ledger.transfer(
                    lease.lease_id,
                    self._cache_owner_id,
                    transfer_claims,
                    destination_lifetime=ClaimLifetime.CACHE,
                    operation_id=f"cache-publish:{lease.request_id}:{self._next_snapshot_id}",
                )
            retained = False
            try:
                self.pool.retain_cache(page_ids)
                retained = True
                radix = self._radices.setdefault(
                    scope.fingerprint,
                    RadixCache(block_size=self.block_size),
                )
                radix.insert(lease.request_id, matched_tokens, page_ids)
                radix.retain_entry(matched_tokens, page_ids)
                radix.cancel(lease.request_id)
                snapshot = KVSnapshotHandle(
                    snapshot_id=f"snapshot:{self._next_snapshot_id}",
                    scope_fingerprint=scope.fingerprint,
                    backend_fingerprint=self.spec.fingerprint,
                    artifact_fingerprint=self.spec.artifact_fingerprint,
                    generation=self.generation,
                    matched_tokens=matched_tokens,
                    page_ids=page_ids,
                    tenant_id=tenant,
                    created_at=timestamp,
                    last_access_at=timestamp,
                    expires_at=(
                        None
                        if self.ttl_seconds is None
                        else timestamp + self.ttl_seconds
                    ),
                )
                self._next_snapshot_id += 1
                self._snapshots[key] = snapshot
            except Exception:
                if retained:
                    self.pool.release_cache(page_ids)
                if transfer_claims.claims:
                    self.ledger.transfer(
                        self._cache_owner_id,
                        lease.lease_id,
                        self._page_claims(
                            len(newly_cached),
                            lifetime=ClaimLifetime.CACHE,
                            claim_id=f"cache-publish-rollback:{lease.request_id}",
                        ),
                        destination_lifetime=ClaimLifetime.LEASE,
                        operation_id=f"cache-publish-rollback:{lease.request_id}",
                    )
                    if not self.ledger.owner_claims(self._cache_owner_id).claims:
                        self.ledger.release(
                            self._cache_owner_id,
                            operation_id="cache-publish-rollback-empty",
                        )
                raise
            self._enforce_limits(timestamp)
            return self._snapshots.get(key)

    def lookup(
        self,
        scope: PrefixCompatibilityKey,
        tokens: tuple[int, ...] | list[int],
        *,
        now: float | None = None,
    ) -> BackendPrefixMatch:
        token_tuple = tuple(int(token) for token in tokens)
        timestamp = self._now(now)
        with self._lock:
            if not self._scope_compatible(scope):
                self._misses += 1
                return BackendPrefixMatch(None, token_tuple, "incompatible_scope")
            self._expire(timestamp)
            radix = self._radices.get(scope.fingerprint)
            if radix is None:
                self._misses += 1
                return BackendPrefixMatch(None, token_tuple, "scope_miss")
            match = radix.match(token_tuple)
            if not match.hit:
                self._misses += 1
                return BackendPrefixMatch(None, token_tuple, "token_miss")
            key = (scope.fingerprint, match.matched_tokens)
            snapshot = self._snapshots.get(key)
            if snapshot is None:
                self._misses += 1
                return BackendPrefixMatch(None, token_tuple, "snapshot_missing")
            if not self._snapshot_current(snapshot):
                self._evict_key(key, reason="stale")
                self._stale_misses += 1
                self._misses += 1
                return BackendPrefixMatch(None, token_tuple, "stale_generation")
            touched = replace(snapshot, last_access_at=timestamp)
            self._snapshots[key] = touched
            self._hits += 1
            return BackendPrefixMatch(
                touched,
                token_tuple[len(match.matched_tokens) :],
            )

    def evict(self, snapshot: KVSnapshotHandle, *, reason: str = "explicit") -> bool:
        with self._lock:
            key = (snapshot.scope_fingerprint, snapshot.matched_tokens)
            current = self._snapshots.get(key)
            if current is None or current.snapshot_id != snapshot.snapshot_id:
                return False
            self._evict_key(key, reason=reason)
            return True

    def evict_for_pressure(self, required_free_pages: int) -> int:
        target = int(required_free_pages)
        if target < 0:
            raise ValueError("required_free_pages must be non-negative")
        with self._lock:
            before = self.pool.free_pages
            for key in self._lru_keys():
                if self.pool.free_pages >= target:
                    break
                self._evict_key(key, reason="pressure")
            return self.pool.free_pages - before

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            tenant_pages = self._tenant_page_usage()
            unique_pages = self._unique_cached_pages()
            return {
                "backend_fingerprint": self.spec.fingerprint,
                "artifact_fingerprint": self.spec.artifact_fingerprint,
                "generation": self.generation,
                "entries": len(self._snapshots),
                "cached_pages": len(unique_pages),
                "max_cached_pages": self.max_cached_pages,
                "tenant_pages": dict(sorted(tenant_pages.items())),
                "tenant_page_quotas": dict(sorted(self.tenant_page_quotas.items())),
                "hits": self._hits,
                "misses": self._misses,
                "stale_misses": self._stale_misses,
                "expired_evictions": self._expired_evictions,
                "pressure_evictions": self._pressure_evictions,
                "quota_evictions": self._quota_evictions,
                "snapshots": [
                    {
                        "snapshot_id": snapshot.snapshot_id,
                        "scope_fingerprint": snapshot.scope_fingerprint,
                        "matched_tokens": len(snapshot.matched_tokens),
                        "page_ids": list(snapshot.page_ids),
                        "tenant_id": snapshot.tenant_id,
                        "last_access_at": snapshot.last_access_at,
                        "expires_at": snapshot.expires_at,
                    }
                    for snapshot in sorted(
                        self._snapshots.values(),
                        key=lambda item: item.snapshot_id,
                    )
                ],
            }

    def assert_conserved(self) -> None:
        with self._lock:
            expected_refs: dict[int, int] = {}
            for snapshot in self._snapshots.values():
                for page_id in snapshot.page_ids:
                    expected_refs[page_id] = expected_refs.get(page_id, 0) + 1
            for page_id in range(self.pool.page_capacity):
                actual = self.pool.page(page_id).cache_references
                if actual != expected_refs.get(page_id, 0):
                    raise AssertionError(
                        f"prefix cache ref mismatch for page {page_id}: "
                        f"expected={expected_refs.get(page_id, 0)}, actual={actual}"
                    )
            cache_units = (
                {}
                if not self.ledger.has_owner(self._cache_owner_id)
                else self.ledger.owner_claims(self._cache_owner_id).units_by_pool()
            )
            unique_count = len(expected_refs)
            expected_units = {
                pool_id: unique_count for pool_id in self.page_pool_ids if unique_count
            }
            if cache_units != expected_units:
                raise AssertionError(
                    f"prefix cache ledger mismatch: expected={expected_units}, actual={cache_units}"
                )
            self.pool.assert_conserved()
            self.ledger.assert_conserved()

    def _evict_key(
        self,
        key: tuple[str, tuple[int, ...]],
        *,
        reason: str,
    ) -> None:
        snapshot = self._snapshots.pop(key)
        radix = self._radices[snapshot.scope_fingerprint]
        radix.evict_entry(snapshot.matched_tokens)
        self.pool.release_cache(snapshot.page_ids)
        uncached_pages = tuple(
            page_id
            for page_id in snapshot.page_ids
            if self.pool.page(page_id).cache_references == 0
        )
        by_destination: dict[str, list[int]] = {}
        unowned: list[int] = []
        for page_id in uncached_pages:
            active = self.pool.page(page_id).active_lease_ids
            if active:
                by_destination.setdefault(active[0], []).append(page_id)
            else:
                unowned.append(page_id)
        for destination, page_ids in by_destination.items():
            self.ledger.transfer(
                self._cache_owner_id,
                destination,
                self._page_claims(
                    len(page_ids),
                    lifetime=ClaimLifetime.CACHE,
                    claim_id=f"cache-evict-transfer:{snapshot.snapshot_id}",
                ),
                destination_lifetime=ClaimLifetime.LEASE,
                operation_id=f"cache-evict-transfer:{snapshot.snapshot_id}:{destination}",
            )
        if unowned:
            self.ledger.apply_delta(
                self._cache_owner_id,
                ResourceDelta(
                    operation_id=f"cache-evict-release:{snapshot.snapshot_id}",
                    lease_id=self._cache_owner_id,
                    changes=tuple(
                        ResourceChange(pool_id, -len(unowned), ClaimLifetime.CACHE)
                        for pool_id in self.page_pool_ids
                    ),
                ),
            )
        if self.ledger.has_owner(self._cache_owner_id):
            cache_claims = self.ledger.owner_claims(self._cache_owner_id)
            if not cache_claims.claims:
                self.ledger.release(
                    self._cache_owner_id,
                    operation_id="cache-owner-empty",
                )
        if reason == "expired":
            self._expired_evictions += 1
        elif reason == "pressure":
            self._pressure_evictions += 1
        elif reason in {"global_quota", "tenant_quota"}:
            self._quota_evictions += 1

    def _expire(self, now: float) -> None:
        for key, snapshot in tuple(self._snapshots.items()):
            if snapshot.expires_at is not None and snapshot.expires_at <= now:
                self._evict_key(key, reason="expired")

    def _enforce_limits(self, now: float) -> None:
        self._expire(now)
        while len(self._unique_cached_pages()) > self.max_cached_pages:
            keys = self._lru_keys()
            if not keys:
                break
            self._evict_key(keys[0], reason="global_quota")
        while True:
            usage = self._tenant_page_usage()
            over = [
                tenant
                for tenant, pages in usage.items()
                if pages > self.tenant_page_quotas.get(tenant, self.max_cached_pages)
            ]
            if not over:
                break
            tenant = sorted(over)[0]
            tenant_keys = [
                key
                for key in self._lru_keys()
                if self._snapshots[key].tenant_id == tenant
            ]
            if not tenant_keys:
                break
            self._evict_key(tenant_keys[0], reason="tenant_quota")

    def _lru_keys(self) -> list[tuple[str, tuple[int, ...]]]:
        return sorted(
            self._snapshots,
            key=lambda key: (
                self._snapshots[key].last_access_at,
                self._snapshots[key].created_at,
                self._snapshots[key].snapshot_id,
            ),
        )

    def _unique_cached_pages(self) -> set[int]:
        return {
            page_id
            for snapshot in self._snapshots.values()
            for page_id in snapshot.page_ids
        }

    def _tenant_page_usage(self) -> dict[str, int]:
        pages_by_tenant: dict[str, set[int]] = {}
        for snapshot in self._snapshots.values():
            pages_by_tenant.setdefault(snapshot.tenant_id, set()).update(snapshot.page_ids)
        return {tenant: len(pages) for tenant, pages in pages_by_tenant.items()}

    def _page_claims(
        self,
        pages: int,
        *,
        lifetime: ClaimLifetime,
        claim_id: str,
    ) -> ResourceClaimSet:
        count = int(pages)
        return ResourceClaimSet(
            claim_id=claim_id,
            claims=tuple(
                ResourceClaim(pool_id, count, lifetime)
                for pool_id in self.page_pool_ids
                if count
            ),
        )

    def _validate_scope(self, scope: PrefixCompatibilityKey) -> None:
        if not self._scope_compatible(scope):
            raise ValueError("prefix compatibility scope does not match backend")

    def _scope_compatible(self, scope: PrefixCompatibilityKey) -> bool:
        return (
            scope.backend_fingerprint == self.spec.fingerprint
            and scope.backend_artifact_fingerprint == self.spec.artifact_fingerprint
        )

    def _validate_lease(self, lease: KVLease) -> None:
        if lease.backend_fingerprint != self.spec.fingerprint:
            raise ValueError("prefix snapshot lease backend mismatch")
        if lease.generation != self.generation:
            raise ValueError("prefix snapshot lease generation mismatch")
        if not self.ledger.has_owner(lease.lease_id):
            raise ValueError("prefix snapshot lease has no ledger ownership")

    def _snapshot_current(self, snapshot: KVSnapshotHandle) -> bool:
        return (
            snapshot.backend_fingerprint == self.spec.fingerprint
            and snapshot.artifact_fingerprint == self.spec.artifact_fingerprint
            and snapshot.generation == self.generation
            and self.pool.generation == self.generation
            and self.ledger.plan.generation == self.generation
        )

    def _now(self, supplied: float | None) -> float:
        return float(self._clock() if supplied is None else supplied)


__all__ = [
    "BackendPrefixMatch",
    "BackendRadixCache",
    "KVSnapshotHandle",
    "PrefixCompatibilityKey",
]
