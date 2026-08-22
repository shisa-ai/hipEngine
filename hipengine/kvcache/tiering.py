"""Optional host/NVMe cold tier composition for Generation-2 KV backends."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
import zlib
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from hipengine.kvcache.backend import (
    ClaimLifetime,
    KVBackendSpec,
    KVBatchView,
    KVLease,
    KVPoolPlan,
    KVPoolSpec,
    ResourceClaim,
    ResourceClaimSet,
    ResourceDelta,
)
from hipengine.kvcache.ledger import ResourceLedger

_COLD_MAGIC = b"HEKVTC1\0"
_COLD_SCHEMA_VERSION = 1
_TIER_IDS = {"host": "tier.host_bytes", "nvme": "tier.nvme_bytes"}


@dataclass(frozen=True, slots=True)
class ColdObjectKey:
    """Complete hot-backend/artifact/token identity for one cold object."""

    hot_backend_fingerprint: str
    artifact_fingerprint: str
    hot_generation: int
    hot_codec: str
    cold_codec: str
    token_ids_sha256: str
    request_scope: str
    state_fingerprint: str
    schema_version: int = _COLD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "hot_backend_fingerprint",
            "artifact_fingerprint",
            "hot_codec",
            "cold_codec",
            "token_ids_sha256",
            "request_scope",
            "state_fingerprint",
        ):
            value = str(getattr(self, name))
            if not value or value != value.strip():
                raise ValueError(f"cold object {name} must be non-empty")
            object.__setattr__(self, name, value)
        if int(self.hot_generation) <= 0:
            raise ValueError("cold object hot_generation must be positive")
        if int(self.schema_version) != _COLD_SCHEMA_VERSION:
            raise ValueError("unsupported cold object schema")

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            {
                "schema_version": self.schema_version,
                "hot_backend_fingerprint": self.hot_backend_fingerprint,
                "artifact_fingerprint": self.artifact_fingerprint,
                "hot_generation": self.hot_generation,
                "hot_codec": self.hot_codec,
                "cold_codec": self.cold_codec,
                "token_ids_sha256": self.token_ids_sha256,
                "request_scope": self.request_scope,
                "state_fingerprint": self.state_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_tokens(
        cls,
        *,
        hot_backend_fingerprint: str,
        artifact_fingerprint: str,
        hot_generation: int,
        hot_codec: str,
        cold_codec: str,
        token_ids: Sequence[int],
        request_scope: str,
        state_fingerprint: str,
    ) -> "ColdObjectKey":
        digest = hashlib.sha256()
        for token in token_ids:
            digest.update(int(token).to_bytes(8, "little", signed=True))
        return cls(
            hot_backend_fingerprint=hot_backend_fingerprint,
            artifact_fingerprint=artifact_fingerprint,
            hot_generation=hot_generation,
            hot_codec=hot_codec,
            cold_codec=cold_codec,
            token_ids_sha256=digest.hexdigest(),
            request_scope=request_scope,
            state_fingerprint=state_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class ColdCodecResult:
    encoded: bytes
    original_bytes: int
    encoded_bytes: int
    payload_sha256: str


class KVTCColdCodec:
    """Deterministic KVTC-style lossless cold codec; never an attention layout."""

    name = "kvtc_zlib_v1"

    def __init__(self, *, level: int = 6) -> None:
        if not 0 <= int(level) <= 9:
            raise ValueError("KVTC zlib level must be in [0, 9]")
        self.level = int(level)

    def encode(self, key: ColdObjectKey, payload: bytes) -> ColdCodecResult:
        raw = bytes(payload)
        payload_digest = hashlib.sha256(raw).hexdigest()
        header = json.dumps(
            {
                "schema_version": _COLD_SCHEMA_VERSION,
                "key_fingerprint": key.fingerprint,
                "payload_sha256": payload_digest,
                "original_bytes": len(raw),
                "codec": self.name,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        compressed = zlib.compress(raw, self.level)
        encoded = b"".join(
            (_COLD_MAGIC, struct.pack("<I", len(header)), header, compressed)
        )
        return ColdCodecResult(
            encoded=encoded,
            original_bytes=len(raw),
            encoded_bytes=len(encoded),
            payload_sha256=payload_digest,
        )

    def decode(self, key: ColdObjectKey, encoded: bytes) -> bytes:
        blob = bytes(encoded)
        if not blob.startswith(_COLD_MAGIC) or len(blob) < len(_COLD_MAGIC) + 4:
            raise ValueError("invalid KVTC cold object header")
        offset = len(_COLD_MAGIC)
        header_bytes = struct.unpack("<I", blob[offset : offset + 4])[0]
        offset += 4
        header_end = offset + header_bytes
        if header_end > len(blob):
            raise ValueError("truncated KVTC cold object header")
        header = json.loads(blob[offset:header_end].decode("utf-8"))
        if header.get("key_fingerprint") != key.fingerprint:
            raise ValueError("cold object key fingerprint mismatch")
        if header.get("codec") != self.name:
            raise ValueError("cold object codec mismatch")
        payload = zlib.decompress(blob[header_end:])
        if len(payload) != int(header.get("original_bytes", -1)):
            raise ValueError("cold object restored byte length mismatch")
        if hashlib.sha256(payload).hexdigest() != header.get("payload_sha256"):
            raise ValueError("cold object checksum mismatch")
        return payload


@dataclass(frozen=True, slots=True)
class ColdStoredObject:
    key: ColdObjectKey
    tenant_id: str
    tier: Literal["host", "nvme"]
    encoded_bytes: int
    original_bytes: int
    created_sequence: int
    last_access_sequence: int
    pinned: bool = False
    path: str | None = None


class ColdTierStore:
    """Bounded deterministic host/NVMe store with explicit LRU eviction."""

    def __init__(
        self,
        *,
        host_capacity_bytes: int,
        nvme_capacity_bytes: int = 0,
        nvme_directory: str | Path | None = None,
        tenant_quota_bytes: Mapping[str, int] | None = None,
    ) -> None:
        if int(host_capacity_bytes) <= 0 or int(nvme_capacity_bytes) < 0:
            raise ValueError("cold tier capacities are invalid")
        if int(nvme_capacity_bytes) and nvme_directory is None:
            raise ValueError("NVMe capacity requires nvme_directory")
        self.host_capacity_bytes = int(host_capacity_bytes)
        self.nvme_capacity_bytes = int(nvme_capacity_bytes)
        self.nvme_directory = (
            None
            if nvme_directory is None
            else Path(nvme_directory).expanduser().resolve()
        )
        if self.nvme_directory is not None:
            self.nvme_directory.mkdir(parents=True, exist_ok=True)
        self.tenant_quota_bytes = {
            str(tenant): int(capacity)
            for tenant, capacity in (tenant_quota_bytes or {}).items()
        }
        if any(capacity <= 0 for capacity in self.tenant_quota_bytes.values()):
            raise ValueError("tenant cold quotas must be positive")
        self._objects: dict[str, ColdStoredObject] = {}
        self._host_payloads: dict[str, bytes] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._sequence = 0
        self._host_bytes = 0
        self._nvme_bytes = 0
        self._tenant_bytes: dict[str, int] = {}
        self.evictions = 0

    def choose_tier(self, encoded_bytes: int, *, tenant_id: str) -> Literal["host", "nvme"]:
        size = int(encoded_bytes)
        tenant = str(tenant_id)
        if size <= 0:
            raise ValueError("cold object size must be positive")
        quota = self.tenant_quota_bytes.get(tenant)
        if quota is not None and self._tenant_bytes.get(tenant, 0) + size > quota:
            raise MemoryError("cold tenant quota exceeded")
        if self._host_bytes + size <= self.host_capacity_bytes:
            return "host"
        if self._nvme_bytes + size <= self.nvme_capacity_bytes:
            return "nvme"
        raise MemoryError("cold host/NVMe capacity exhausted")

    def put(
        self,
        key: ColdObjectKey,
        encoded: bytes,
        *,
        original_bytes: int,
        tenant_id: str = "default",
        tier: Literal["host", "nvme"] | None = None,
    ) -> ColdStoredObject:
        fingerprint = key.fingerprint
        if fingerprint in self._objects:
            raise ValueError("cold object key already exists")
        payload = bytes(encoded)
        selected = self.choose_tier(len(payload), tenant_id=tenant_id) if tier is None else tier
        if selected not in {"host", "nvme"}:
            raise ValueError("cold tier must be host or nvme")
        tenant = str(tenant_id)
        # Revalidate a caller-selected tier.
        if selected == "host" and self._host_bytes + len(payload) > self.host_capacity_bytes:
            raise MemoryError("cold host capacity exhausted")
        if selected == "nvme" and self._nvme_bytes + len(payload) > self.nvme_capacity_bytes:
            raise MemoryError("cold NVMe capacity exhausted")
        quota = self.tenant_quota_bytes.get(tenant)
        if quota is not None and self._tenant_bytes.get(tenant, 0) + len(payload) > quota:
            raise MemoryError("cold tenant quota exceeded")
        self._sequence += 1
        path: Path | None = None
        if selected == "host":
            self._host_payloads[fingerprint] = payload
            self._host_bytes += len(payload)
        else:
            assert self.nvme_directory is not None
            path = self.nvme_directory / f"{fingerprint}.kvtc"
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, path)
            self._nvme_bytes += len(payload)
        record = ColdStoredObject(
            key=key,
            tenant_id=tenant,
            tier=selected,
            encoded_bytes=len(payload),
            original_bytes=int(original_bytes),
            created_sequence=self._sequence,
            last_access_sequence=self._sequence,
            path=None if path is None else str(path),
        )
        self._objects[fingerprint] = record
        self._tenant_bytes[tenant] = self._tenant_bytes.get(tenant, 0) + len(payload)
        self._lru[fingerprint] = None
        return record

    def get(self, key: ColdObjectKey) -> tuple[ColdStoredObject, bytes]:
        fingerprint = key.fingerprint
        try:
            record = self._objects[fingerprint]
        except KeyError as exc:
            raise KeyError("cold object does not exist") from exc
        if record.key != key:
            raise ValueError("cold object identity collision")
        payload = (
            self._host_payloads[fingerprint]
            if record.tier == "host"
            else Path(str(record.path)).read_bytes()
        )
        self._sequence += 1
        updated = replace(record, last_access_sequence=self._sequence)
        self._objects[fingerprint] = updated
        self._lru.move_to_end(fingerprint)
        return updated, payload

    def pin(self, key: ColdObjectKey, *, pinned: bool = True) -> None:
        fingerprint = key.fingerprint
        record = self._objects.get(fingerprint)
        if record is None:
            raise KeyError("cold object does not exist")
        self._objects[fingerprint] = replace(record, pinned=bool(pinned))

    def delete(self, key: ColdObjectKey) -> ColdStoredObject:
        fingerprint = key.fingerprint
        try:
            record = self._objects.pop(fingerprint)
        except KeyError as exc:
            raise KeyError("cold object does not exist") from exc
        if record.tier == "host":
            self._host_payloads.pop(fingerprint)
            self._host_bytes -= record.encoded_bytes
        else:
            path = Path(str(record.path))
            path.unlink(missing_ok=True)
            self._nvme_bytes -= record.encoded_bytes
        self._tenant_bytes[record.tenant_id] -= record.encoded_bytes
        if self._tenant_bytes[record.tenant_id] == 0:
            self._tenant_bytes.pop(record.tenant_id)
        self._lru.pop(fingerprint, None)
        return record

    def evict_lru(self, *, required_bytes: int = 0, max_objects: int | None = None) -> tuple[ColdStoredObject, ...]:
        required = max(0, int(required_bytes))
        limit = None if max_objects is None else int(max_objects)
        if limit is not None and limit < 0:
            raise ValueError("max_objects must be non-negative")
        evicted: list[ColdStoredObject] = []
        freed = 0
        for fingerprint in tuple(self._lru):
            if limit is not None and len(evicted) >= limit:
                break
            record = self._objects[fingerprint]
            if record.pinned:
                continue
            evicted.append(self.delete(record.key))
            freed += record.encoded_bytes
            self.evictions += 1
            if freed >= required and required > 0:
                break
        return tuple(evicted)

    def contains(self, key: ColdObjectKey) -> bool:
        return key.fingerprint in self._objects

    def snapshot(self) -> dict[str, Any]:
        return {
            "host_capacity_bytes": self.host_capacity_bytes,
            "host_used_bytes": self._host_bytes,
            "nvme_capacity_bytes": self.nvme_capacity_bytes,
            "nvme_used_bytes": self._nvme_bytes,
            "tenant_used_bytes": dict(sorted(self._tenant_bytes.items())),
            "object_count": len(self._objects),
            "pinned_objects": sum(record.pinned for record in self._objects.values()),
            "evictions": self.evictions,
            "lru": list(self._lru),
            "objects": {
                fingerprint: {
                    "tier": record.tier,
                    "tenant_id": record.tenant_id,
                    "encoded_bytes": record.encoded_bytes,
                    "original_bytes": record.original_bytes,
                    "last_access_sequence": record.last_access_sequence,
                    "pinned": record.pinned,
                }
                for fingerprint, record in sorted(self._objects.items())
            },
        }

    def assert_conserved(self) -> None:
        host = sum(
            record.encoded_bytes
            for record in self._objects.values()
            if record.tier == "host"
        )
        nvme = sum(
            record.encoded_bytes
            for record in self._objects.values()
            if record.tier == "nvme"
        )
        tenants: dict[str, int] = {}
        for record in self._objects.values():
            tenants[record.tenant_id] = tenants.get(record.tenant_id, 0) + record.encoded_bytes
        if host != self._host_bytes or nvme != self._nvme_bytes:
            raise AssertionError("cold tier byte accounting drift")
        if tenants != self._tenant_bytes:
            raise AssertionError("cold tier tenant accounting drift")
        if set(self._lru) != set(self._objects):
            raise AssertionError("cold tier LRU/object membership drift")
        if set(self._host_payloads) != {
            fingerprint
            for fingerprint, record in self._objects.items()
            if record.tier == "host"
        }:
            raise AssertionError("cold host payload membership drift")


@dataclass(frozen=True, slots=True)
class TierMaintenanceWork:
    work_id: str
    kind: Literal["offload", "restore", "evict"]
    key: ColdObjectKey | None = None
    lease: KVLease | None = None
    request: Any = None
    payload: bytes | None = None
    tenant_id: str = "default"
    restore_callback: Callable[[KVLease, bytes], None] | None = None
    required_bytes: int = 0


@dataclass(frozen=True, slots=True)
class TierMaintenanceResult:
    work_id: str
    kind: str
    passed: bool
    bytes_moved: int = 0
    tier: str | None = None
    lease: KVLease | None = None
    evicted_keys: tuple[str, ...] = ()
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class RestoreEconomics:
    restore_seconds: float
    recompute_seconds: float
    savings_seconds: float
    use_restore: bool


def evaluate_restore_economics(
    *,
    restore_seconds: float,
    recompute_seconds: float,
    minimum_savings_seconds: float = 0.0,
) -> RestoreEconomics:
    restore = float(restore_seconds)
    recompute = float(recompute_seconds)
    minimum = float(minimum_savings_seconds)
    if min(restore, recompute, minimum) < 0.0:
        raise ValueError("restore economics durations must be non-negative")
    savings = recompute - restore
    return RestoreEconomics(
        restore_seconds=restore,
        recompute_seconds=recompute,
        savings_seconds=savings,
        use_restore=bool(savings >= minimum and restore < recompute),
    )


class TieredKVCacheBackend:
    """Delegate hot execution and add optional cold maintenance/resource pools."""

    def __init__(
        self,
        hot_backend: Any,
        *,
        store: ColdTierStore,
        codec: KVTCColdCodec | None = None,
        transfer_workspace_bytes: int = 64 * 1024 * 1024,
        maintenance_budget_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if int(transfer_workspace_bytes) <= 0 or int(maintenance_budget_bytes) <= 0:
            raise ValueError("tier workspace/budget must be positive")
        self.hot_backend = hot_backend
        self.store = store
        self.codec = codec or KVTCColdCodec()
        self.transfer_workspace_bytes = int(transfer_workspace_bytes)
        self.maintenance_budget_bytes = int(maintenance_budget_bytes)
        hot_spec = hot_backend.spec
        self.spec = KVBackendSpec(
            topology_key=hot_spec.topology_key,
            hot_codec_key=hot_spec.hot_codec_key,
            tier_key=f"optional_{self.codec.name}",
            layout_fingerprint=hot_spec.layout_fingerprint,
            artifact_fingerprint=hot_spec.artifact_fingerprint,
            prefix_mode=hot_spec.prefix_mode,
            transaction_mode=hot_spec.transaction_mode,
            kernel_bundle_key=hot_spec.kernel_bundle_key,
            physical_widths=hot_spec.physical_widths,
            max_context_tokens=hot_spec.max_context_tokens,
        )
        self._plan = self._build_tier_plan()
        self.tier_ledger = ResourceLedger(self._plan)
        self._queue: deque[TierMaintenanceWork] = deque()
        self._cold_owner_by_key: dict[str, str] = {}
        self._sequence = 0
        self._results: list[TierMaintenanceResult] = []

    def plan_pools(self, load_plan: Any) -> KVPoolPlan:
        del load_plan
        return self._plan

    def estimate(self, request: Any, prefix: Any, stage: Any) -> ResourceClaimSet:
        return self.hot_backend.estimate(request, prefix, stage)

    def reserve(self, claims: ResourceClaimSet) -> KVLease:
        return self.hot_backend.reserve(claims)

    def prepare(self, work_item: Any) -> KVBatchView:
        # Cold bytes are never visible to attention; only restored hot storage is.
        return self.hot_backend.prepare(work_item)

    def begin_transaction(self, rows: Sequence[Any], draft: Any) -> Any:
        return self.hot_backend.begin_transaction(rows, draft)

    def commit(self, operation: Any, result: Any) -> ResourceDelta:
        return self.hot_backend.commit(operation, result)

    def rollback(self, operation: Any) -> ResourceDelta:
        return self.hot_backend.rollback(operation)

    def reclaim(self, lease: KVLease) -> ResourceDelta:
        return self.hot_backend.reclaim(lease)

    def prefix_lookup(self, tokens: Sequence[int]) -> Any:
        return self.hot_backend.prefix_lookup(tokens)

    def enqueue_offload(
        self,
        *,
        key: ColdObjectKey,
        lease: KVLease,
        hot_payload: bytes,
        tenant_id: str = "default",
    ) -> str:
        self._validate_key(key)
        self._sequence += 1
        work_id = f"tier-offload:{self._sequence}:{key.fingerprint}"
        self._queue.append(
            TierMaintenanceWork(
                work_id=work_id,
                kind="offload",
                key=key,
                lease=lease,
                payload=bytes(hot_payload),
                tenant_id=str(tenant_id),
            )
        )
        return work_id

    def enqueue_restore(
        self,
        *,
        key: ColdObjectKey,
        request: Any,
        restore_callback: Callable[[KVLease, bytes], None],
    ) -> str:
        self._validate_key(key)
        if not callable(restore_callback):
            raise TypeError("restore_callback must be callable")
        self._sequence += 1
        work_id = f"tier-restore:{self._sequence}:{key.fingerprint}"
        self._queue.append(
            TierMaintenanceWork(
                work_id=work_id,
                kind="restore",
                key=key,
                request=request,
                restore_callback=restore_callback,
            )
        )
        return work_id

    def enqueue_evict(self, *, required_bytes: int = 0) -> str:
        self._sequence += 1
        work_id = f"tier-evict:{self._sequence}"
        self._queue.append(
            TierMaintenanceWork(
                work_id=work_id,
                kind="evict",
                required_bytes=max(0, int(required_bytes)),
            )
        )
        return work_id

    def cancel_request(self, request_id: int) -> tuple[str, ...]:
        """Cancel queued maintenance owned by ``request_id`` before execution.

        Restore cancellation leaves the cold object available for another
        request and has no provisional workspace to release. A queued offload
        already owns its hot lease, so cancellation reclaims that lease instead
        of leaking hot capacity. Completed maintenance is unaffected.
        """

        target = int(request_id)
        kept: deque[TierMaintenanceWork] = deque()
        cancelled: list[str] = []
        while self._queue:
            work = self._queue.popleft()
            owner = None
            if work.lease is not None:
                owner = int(work.lease.request_id)
            elif work.request is not None and hasattr(work.request, "request_id"):
                owner = int(work.request.request_id)
            if owner != target:
                kept.append(work)
                continue
            if work.kind == "offload" and work.lease is not None:
                self.hot_backend.reclaim(work.lease)
            cancelled.append(work.work_id)
            self._results.append(
                TierMaintenanceResult(
                    work_id=work.work_id,
                    kind=work.kind,
                    passed=False,
                    error="cancelled before maintenance",
                )
            )
        self._queue = kept
        return tuple(cancelled)

    def maintenance(self, budget: Any = None) -> list[TierMaintenanceResult]:
        limit = self.maintenance_budget_bytes
        if isinstance(budget, Mapping):
            limit = int(budget.get("bytes", limit))
        elif budget is not None:
            limit = int(budget)
        if limit <= 0:
            return []
        results: list[TierMaintenanceResult] = []
        consumed = 0
        while self._queue:
            work = self._queue[0]
            estimated = len(work.payload or b"") or max(1, int(work.required_bytes))
            if results and consumed + estimated > limit:
                break
            self._queue.popleft()
            started = time.perf_counter()
            workspace_reservation = None
            try:
                workspace_units = 0
                if work.kind == "offload":
                    workspace_units = len(work.payload or b"")
                elif work.kind == "restore" and work.key is not None:
                    record = self.store._objects.get(work.key.fingerprint)
                    workspace_units = 0 if record is None else record.original_bytes
                if workspace_units:
                    workspace_reservation = self.tier_ledger.reserve_provisional(
                        ResourceClaimSet(
                            claim_id=f"tier-workspace:{work.work_id}",
                            claims=(
                                ResourceClaim(
                                    "tier.transfer_workspace_bytes",
                                    workspace_units,
                                    ClaimLifetime.WORK_ITEM,
                                ),
                            ),
                        )
                    )
                if work.kind == "offload":
                    result = self._execute_offload(work)
                elif work.kind == "restore":
                    result = self._execute_restore(work)
                else:
                    result = self._execute_evict(work)
            except Exception as exc:
                result = TierMaintenanceResult(
                    work_id=work.work_id,
                    kind=work.kind,
                    passed=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            finally:
                if workspace_reservation is not None:
                    self.tier_ledger.rollback(workspace_reservation)
            result = replace(
                result,
                duration_seconds=time.perf_counter() - started,
            )
            results.append(result)
            self._results.append(result)
            consumed += max(estimated, result.bytes_moved)
        return results

    def cold_key_for_tokens(
        self,
        *,
        token_ids: Sequence[int],
        request_scope: str,
        state_fingerprint: str,
    ) -> ColdObjectKey:
        return ColdObjectKey.from_tokens(
            hot_backend_fingerprint=self.hot_backend.spec.fingerprint,
            artifact_fingerprint=self.hot_backend.spec.artifact_fingerprint,
            hot_generation=int(getattr(self.hot_backend, "generation", 1)),
            hot_codec=self.hot_backend.spec.hot_codec_key,
            cold_codec=self.codec.name,
            token_ids=token_ids,
            request_scope=request_scope,
            state_fingerprint=state_fingerprint,
        )

    def observability_snapshot(self) -> dict[str, Any]:
        hot_snapshot = getattr(self.hot_backend, "observability_snapshot", None)
        return {
            "hot": hot_snapshot() if callable(hot_snapshot) else {},
            "tier": {
                "codec": self.codec.name,
                "pending_maintenance": len(self._queue),
                "results": [asdict_result(result) for result in self._results[-64:]],
                "store": self.store.snapshot(),
                "ledger": self.tier_ledger.snapshot(),
            },
        }

    def assert_conserved(self) -> None:
        self.store.assert_conserved()
        self.tier_ledger.assert_conserved()
        if set(self._cold_owner_by_key) != set(self.store.snapshot()["objects"]):
            raise AssertionError("cold object/ledger owner membership drift")

    def drain(self) -> None:
        while self._queue:
            self.maintenance({"bytes": 1 << 60})
        for fingerprint in tuple(self.store.snapshot()["objects"]):
            record = self.store._objects[fingerprint]
            self.store.delete(record.key)
            owner = self._cold_owner_by_key.pop(fingerprint, None)
            if owner is not None and self.tier_ledger.has_owner(owner):
                self.tier_ledger.release(owner, operation_id=f"tier-drain:{fingerprint}")
        self.assert_conserved()

    def _execute_offload(self, work: TierMaintenanceWork) -> TierMaintenanceResult:
        assert work.key is not None and work.lease is not None and work.payload is not None
        encoded = self.codec.encode(work.key, work.payload)
        if encoded.original_bytes > self.transfer_workspace_bytes:
            raise MemoryError("tier offload exceeds transfer workspace")
        tier = self.store.choose_tier(encoded.encoded_bytes, tenant_id=work.tenant_id)
        pool_id = _TIER_IDS[tier]
        claims = ResourceClaimSet(
            claim_id=f"tier-object:{work.key.fingerprint}",
            claims=(
                ResourceClaim(pool_id, encoded.encoded_bytes, ClaimLifetime.CACHE),
            ),
        )
        reservation = self.tier_ledger.reserve_provisional(claims)
        owner = f"cold:{work.key.fingerprint}"
        try:
            record = self.store.put(
                work.key,
                encoded.encoded,
                original_bytes=encoded.original_bytes,
                tenant_id=work.tenant_id,
                tier=tier,
            )
            self.tier_ledger.commit(reservation, owner_id=owner)
        except Exception:
            self.tier_ledger.rollback(reservation)
            if self.store.contains(work.key):
                self.store.delete(work.key)
            raise
        try:
            self.hot_backend.reclaim(work.lease)
        except Exception:
            self.store.delete(work.key)
            self.tier_ledger.release(owner, operation_id=f"offload-hot-rollback:{work.work_id}")
            raise
        self._cold_owner_by_key[work.key.fingerprint] = owner
        return TierMaintenanceResult(
            work_id=work.work_id,
            kind=work.kind,
            passed=True,
            bytes_moved=encoded.original_bytes,
            tier=record.tier,
        )

    def _execute_restore(self, work: TierMaintenanceWork) -> TierMaintenanceResult:
        assert work.key is not None and work.request is not None and work.restore_callback is not None
        record, encoded = self.store.get(work.key)
        if record.original_bytes > self.transfer_workspace_bytes:
            raise MemoryError("tier restore exceeds transfer workspace")
        payload = self.codec.decode(work.key, encoded)
        claims = self.hot_backend.estimate(work.request, None, {"kind": "admission"})
        lease = self.hot_backend.reserve(claims)
        try:
            work.restore_callback(lease, payload)
        except Exception:
            self.hot_backend.reclaim(lease)
            raise
        self.store.delete(work.key)
        owner = self._cold_owner_by_key.pop(work.key.fingerprint)
        self.tier_ledger.release(owner, operation_id=f"tier-restore:{work.work_id}")
        return TierMaintenanceResult(
            work_id=work.work_id,
            kind=work.kind,
            passed=True,
            bytes_moved=len(payload),
            tier=record.tier,
            lease=lease,
        )

    def _execute_evict(self, work: TierMaintenanceWork) -> TierMaintenanceResult:
        evicted = self.store.evict_lru(required_bytes=work.required_bytes)
        for record in evicted:
            owner = self._cold_owner_by_key.pop(record.key.fingerprint, None)
            if owner is not None:
                self.tier_ledger.release(
                    owner,
                    operation_id=f"tier-evict:{record.key.fingerprint}",
                )
        return TierMaintenanceResult(
            work_id=work.work_id,
            kind=work.kind,
            passed=True,
            bytes_moved=sum(record.encoded_bytes for record in evicted),
            evicted_keys=tuple(record.key.fingerprint for record in evicted),
        )

    def _validate_key(self, key: ColdObjectKey) -> None:
        if key.hot_backend_fingerprint != self.hot_backend.spec.fingerprint:
            raise ValueError("cold key hot backend fingerprint mismatch")
        if key.artifact_fingerprint != self.hot_backend.spec.artifact_fingerprint:
            raise ValueError("cold key artifact fingerprint mismatch")
        if key.hot_codec != self.hot_backend.spec.hot_codec_key:
            raise ValueError("cold key hot codec mismatch")
        if key.cold_codec != self.codec.name:
            raise ValueError("cold key cold codec mismatch")
        if key.hot_generation != int(getattr(self.hot_backend, "generation", 1)):
            raise ValueError("cold key hot generation mismatch")

    def _build_tier_plan(self) -> KVPoolPlan:
        return KVPoolPlan(
            backend_fingerprint=self.spec.fingerprint,
            generation=int(getattr(self.hot_backend, "generation", 1)),
            pools=(
                KVPoolSpec(
                    "tier.host_bytes",
                    self.store.host_capacity_bytes,
                    unit="bytes",
                    plane_role="cold_host",
                    lifetimes=(ClaimLifetime.CACHE,),
                ),
                KVPoolSpec(
                    "tier.nvme_bytes",
                    max(1, self.store.nvme_capacity_bytes),
                    unit="bytes",
                    plane_role="cold_nvme",
                    lifetimes=(ClaimLifetime.CACHE,),
                ),
                KVPoolSpec(
                    "tier.transfer_workspace_bytes",
                    self.transfer_workspace_bytes,
                    unit="bytes",
                    plane_role="transfer_workspace",
                    lifetimes=(ClaimLifetime.WORK_ITEM,),
                ),
            ),
        )


def asdict_result(result: TierMaintenanceResult) -> dict[str, Any]:
    return {
        "work_id": result.work_id,
        "kind": result.kind,
        "passed": result.passed,
        "bytes_moved": result.bytes_moved,
        "tier": result.tier,
        "lease_id": None if result.lease is None else result.lease.lease_id,
        "evicted_keys": list(result.evicted_keys),
        "error": result.error,
        "duration_seconds": result.duration_seconds,
    }


__all__ = [
    "ColdCodecResult",
    "ColdObjectKey",
    "ColdStoredObject",
    "ColdTierStore",
    "KVTCColdCodec",
    "RestoreEconomics",
    "TierMaintenanceResult",
    "TierMaintenanceWork",
    "TieredKVCacheBackend",
    "evaluate_restore_economics",
]
