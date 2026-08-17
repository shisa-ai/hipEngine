"""Device-backed adapter for the generation-checked global KV page pool."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from hipengine.kvcache.global_pool import GlobalKVPoolSet
from hipengine.kvcache.pool import (
    DeviceKVPoolAllocation,
    DeviceKVPoolStats,
    KVPoolChunk,
)


class GlobalDeviceKVPool:
    """Expose ``GlobalKVPoolSet`` through the resident device-pool lifecycle ABI.

    Storage and pointer tables are allocated once by the model/backend package.
    Request leases may select arbitrary free global page IDs; no request-local
    backing chunk or contiguous-run constraint is imposed. The compatibility
    allocation record remains only as the binding envelope used by resident
    model plugins while they migrate to ``KVBatchView`` directly.
    """

    generation2_compatible = True
    compatibility_reason = None

    def __init__(
        self,
        *,
        page_bytes: int,
        backend_fingerprint: str,
        generation: int,
        backing: Any,
        plane_page_pointers: Mapping[str, Sequence[int]],
        pointer_table_pointers: Mapping[str, int],
        metadata_descriptor_pointer: int,
        close_storage: Callable[[], None],
    ) -> None:
        if int(page_bytes) <= 0:
            raise ValueError("page_bytes must be positive")
        if not callable(close_storage):
            raise TypeError("close_storage must be callable")
        planes = {
            str(role): tuple(int(pointer) for pointer in pointers)
            for role, pointers in plane_page_pointers.items()
        }
        self.global_pool = GlobalKVPoolSet(
            backend_fingerprint=str(backend_fingerprint),
            generation=int(generation),
            plane_page_pointers=planes,
            pointer_table_pointers={
                str(role): int(pointer)
                for role, pointer in pointer_table_pointers.items()
            },
            metadata_descriptor_pointer=int(metadata_descriptor_pointer),
        )
        self.page_bytes = int(page_bytes)
        self.low_water_pages = self.global_pool.page_capacity
        self.high_water_pages = self.global_pool.page_capacity
        self.chunk_pages = self.global_pool.page_capacity
        self.idle_grace_seconds = 0.0
        self._backing = backing
        self._close_storage = close_storage
        self._primary_plane = sorted(planes)[0]
        self._request_allocations: dict[int, DeviceKVPoolAllocation] = {}
        self._pin_counts: dict[int, int] = {}
        self._last_active_seconds = 0.0
        self._high_water_observed_pages = 0
        self._prefix_reuse_events = 0
        self._prefix_reused_pages = 0
        self._cow_fork_events = 0
        self._cow_forked_pages = 0
        self._closed = False
        self._lock = threading.RLock()

    @property
    def chunks(self) -> tuple[KVPoolChunk, ...]:
        return (KVPoolChunk(start_block_id=0, pages=self.current_pages),)

    @property
    def current_pages(self) -> int:
        return self.global_pool.page_capacity

    @property
    def allocations(self) -> dict[int, DeviceKVPoolAllocation]:
        with self._lock:
            return dict(self._request_allocations)

    @property
    def stats(self) -> DeviceKVPoolStats:
        with self._lock:
            records = [
                self.global_pool.page(page_id)
                for page_id in range(self.global_pool.page_capacity)
            ]
            return DeviceKVPoolStats(
                current_pages=self.current_pages,
                current_bytes=self.current_pages * self.page_bytes,
                high_water_observed_pages=self._high_water_observed_pages,
                high_water_observed_bytes=(
                    self._high_water_observed_pages * self.page_bytes
                ),
                free_pages=self.global_pool.free_pages,
                refcounted_pages=sum(
                    bool(record.active_lease_ids or record.cache_references)
                    for record in records
                ),
                pinned_pages=sum(record.session_pins > 0 for record in records),
                grow_events=0,
                grow_failures=0,
                shrink_events=0,
                prefix_reuse_events=self._prefix_reuse_events,
                prefix_reused_pages=self._prefix_reused_pages,
                cow_fork_events=self._cow_fork_events,
                cow_forked_pages=self._cow_forked_pages,
            )

    def storage_view(self):
        return self.global_pool.storage_view()

    def pointer_for(self, block_id: int) -> int:
        return self.global_pool.page_pointer(self._primary_plane, int(block_id))

    def refcount(self, block_id: int) -> int:
        record = self.global_pool.page(int(block_id))
        return len(record.active_lease_ids) + int(record.cache_references)

    def pin_count(self, block_id: int) -> int:
        return int(self.global_pool.page(int(block_id)).session_pins)

    def allocate(
        self,
        request_id: int,
        pages: int,
        *,
        now_seconds: float = 0.0,
    ) -> DeviceKVPoolAllocation:
        rid = int(request_id)
        count = int(pages)
        if count <= 0:
            raise ValueError("pages must be positive")
        with self._lock:
            self._require_open()
            if rid in self._request_allocations:
                raise ValueError(f"request_id {rid} already has a device KV allocation")
            lease = self.global_pool.allocate(
                self._lease_id(rid),
                private_pages=count,
                growth_credit_pages=0,
            )
            allocation = self._allocation(rid, lease)
            self._request_allocations[rid] = allocation
            self._last_active_seconds = float(now_seconds)
            self._observe_high_water()
            return allocation

    def admit_with_shared_prefix(
        self,
        request_id: int,
        prefix_block_ids: Sequence[int],
        *,
        suffix_pages: int,
        now_seconds: float = 0.0,
    ) -> DeviceKVPoolAllocation:
        rid = int(request_id)
        shared = tuple(int(page_id) for page_id in prefix_block_ids)
        private = int(suffix_pages)
        if private < 0:
            raise ValueError("suffix_pages must be non-negative")
        if not shared and private <= 0:
            raise ValueError("admission must reuse or allocate at least one device KV page")
        with self._lock:
            self._require_open()
            if rid in self._request_allocations:
                raise ValueError(f"request_id {rid} already has a device KV allocation")
            lease = self.global_pool.allocate(
                self._lease_id(rid),
                private_pages=private,
                growth_credit_pages=0,
                shared_page_ids=shared,
            )
            allocation = self._allocation(rid, lease)
            self._request_allocations[rid] = allocation
            self._last_active_seconds = float(now_seconds)
            self._prefix_reuse_events += 1
            self._prefix_reused_pages += len(shared)
            self._observe_high_water()
            return allocation

    def fork_copy_on_write(
        self,
        request_id: int,
        prefix_block_ids: Sequence[int],
        *,
        suffix_pages: int,
        first_divergent_token: int,
        now_seconds: float = 0.0,
    ) -> DeviceKVPoolAllocation:
        divergent = int(first_divergent_token)
        if divergent < 0:
            raise ValueError("first_divergent_token must be non-negative")
        if int(suffix_pages) <= 0:
            raise ValueError("suffix_pages must be positive for copy-on-write")
        allocation = self.admit_with_shared_prefix(
            request_id,
            prefix_block_ids,
            suffix_pages=int(suffix_pages),
            now_seconds=now_seconds,
        )
        fork = DeviceKVPoolAllocation(
            request_id=allocation.request_id,
            block_ids=allocation.block_ids,
            pointers=allocation.pointers,
            chunk_start_block_id=0,
            backing=allocation.backing,
            reused_block_ids=allocation.reused_block_ids,
            allocated_block_ids=allocation.allocated_block_ids,
            first_divergent_token=divergent,
        )
        with self._lock:
            self._request_allocations[int(request_id)] = fork
            self._cow_fork_events += 1
            self._cow_forked_pages += len(fork.allocated_block_ids)
        return fork

    def retain_blocks(self, block_ids: Sequence[int]) -> None:
        self.global_pool.retain_cache(tuple(int(page_id) for page_id in block_ids))

    def release_blocks(self, block_ids: Sequence[int]) -> None:
        self.global_pool.release_cache(tuple(int(page_id) for page_id in block_ids))

    def pin(self, block_ids: Sequence[int]) -> None:
        pages = tuple(int(page_id) for page_id in block_ids)
        with self._lock:
            self._require_open()
            lease_id = self._active_lease_for(pages)
            self.global_pool.pin_session(lease_id, pages)
            for page_id in pages:
                self._pin_counts[page_id] = self._pin_counts.get(page_id, 0) + 1

    def unpin(self, block_ids: Sequence[int]) -> None:
        pages = tuple(int(page_id) for page_id in block_ids)
        with self._lock:
            for page_id in pages:
                if self._pin_counts.get(page_id, 0) <= 0:
                    raise ValueError("device KV page is not graph-pinned")
            self.global_pool.unpin_session(pages)
            for page_id in pages:
                self._pin_counts[page_id] -= 1

    def release(
        self,
        request_id: int,
        *,
        now_seconds: float = 0.0,
    ) -> DeviceKVPoolAllocation:
        rid = int(request_id)
        with self._lock:
            try:
                allocation = self._request_allocations.pop(rid)
            except KeyError as exc:
                raise KeyError(
                    f"request_id {rid} has no device KV allocation"
                ) from exc
            self.global_pool.release(self._lease_id(rid))
            self._last_active_seconds = float(now_seconds)
            return allocation

    def shrink_idle(self, *, now_seconds: float) -> int:
        del now_seconds
        return 0

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._request_allocations:
                raise RuntimeError(
                    "cannot close global device KV pool with live request allocations"
                )
            snapshot = self.global_pool.snapshot()
            if int(snapshot["free_pages"]) != int(snapshot["page_capacity"]):
                raise RuntimeError(
                    "cannot close global device KV pool with retained or pinned pages"
                )
            self.global_pool.assert_conserved()
            self._close_storage()
            self._closed = True

    @staticmethod
    def _lease_id(request_id: int) -> str:
        return f"request:{int(request_id)}"

    def _allocation(self, request_id: int, lease: Any) -> DeviceKVPoolAllocation:
        block_ids = tuple(int(page_id) for page_id in lease.logical_page_ids)
        return DeviceKVPoolAllocation(
            request_id=int(request_id),
            block_ids=block_ids,
            pointers=tuple(self.pointer_for(page_id) for page_id in block_ids),
            chunk_start_block_id=0,
            backing=self._backing,
            reused_block_ids=tuple(int(page_id) for page_id in lease.shared_page_ids),
            allocated_block_ids=tuple(int(page_id) for page_id in lease.private_page_ids),
        )

    def _active_lease_for(self, page_ids: tuple[int, ...]) -> str:
        requested = set(page_ids)
        candidates = [
            self._lease_id(request_id)
            for request_id, allocation in self._request_allocations.items()
            if requested.issubset(allocation.block_ids)
        ]
        if not candidates:
            raise ValueError("cannot graph-pin an unreferenced device KV page")
        return sorted(candidates)[0]

    def _observe_high_water(self) -> None:
        active = self.current_pages - self.global_pool.free_pages
        self._high_water_observed_pages = max(
            self._high_water_observed_pages,
            int(active),
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("global device KV pool is closed")


__all__ = ["GlobalDeviceKVPool"]
