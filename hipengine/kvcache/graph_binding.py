"""Graph bindings over stable global KV storage and changing page IDs."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from hipengine.kvcache.backend import KVBatchView, KVStorageView
from hipengine.kvcache.global_pool import GlobalKVPoolSet


@dataclass(frozen=True, slots=True)
class GraphStorageSignature:
    layout_key: str
    generation: int
    artifact_fingerprint: str
    planes: tuple[tuple[str, str, int, tuple[int, ...], tuple[int, ...]], ...]
    metadata_descriptor_ptr: int
    metadata_descriptor_bytes: int

    @classmethod
    def from_view(cls, view: KVStorageView) -> "GraphStorageSignature":
        return cls(
            layout_key=view.layout_key,
            generation=view.generation,
            artifact_fingerprint=view.artifact_fingerprint,
            planes=tuple(
                (plane.role, plane.dtype, plane.ptr, plane.shape, plane.strides)
                for plane in view.planes
            ),
            metadata_descriptor_ptr=view.metadata_descriptor_ptr,
            metadata_descriptor_bytes=view.metadata_descriptor_bytes,
        )


@dataclass(frozen=True, slots=True)
class GraphReplayBinding:
    graph_key: str
    replay_id: int
    epoch: int
    generation: int
    request_ids: tuple[int, ...]
    lease_ids: tuple[str, ...]
    slot_ids: tuple[int, ...]
    page_ids: tuple[int, ...]


class GraphReplayBindingRegistry:
    """Capture stable storage once and bind changing request metadata per replay."""

    def __init__(self, pool: GlobalKVPoolSet) -> None:
        self.pool = pool
        self._captures: dict[str, GraphStorageSignature] = {}
        self._in_flight: dict[int, GraphReplayBinding] = {}
        self._next_replay_id = 0
        self._next_epoch = 0
        self._capture_count = 0
        self._replay_count = 0
        self._rejection_count = 0
        self._slot_reuse_count = 0
        self._last_request_by_slot: dict[int, int] = {}
        self._lock = threading.RLock()

    def capture(self, graph_key: str, batch_view: KVBatchView) -> GraphStorageSignature:
        key = _key(graph_key)
        signature = GraphStorageSignature.from_view(batch_view.storage_view)
        self._validate_signature(signature)
        with self._lock:
            existing = self._captures.get(key)
            if existing is not None and existing != signature:
                raise ValueError("graph key is already captured for different KV storage")
            if existing is None:
                self._captures[key] = signature
                self._capture_count += 1
            return signature

    def bind_replay(
        self,
        graph_key: str,
        batch_view: KVBatchView,
        *,
        request_ids: tuple[int, ...],
        lease_ids: tuple[str, ...],
        slot_ids: tuple[int, ...],
    ) -> GraphReplayBinding:
        key = _key(graph_key)
        requests = tuple(int(request_id) for request_id in request_ids)
        leases = tuple(str(lease_id) for lease_id in lease_ids)
        slots = tuple(int(slot_id) for slot_id in slot_ids)
        if not requests or len(requests) != len(set(requests)):
            raise ValueError("graph replay request_ids must be non-empty and unique")
        if len(leases) != len(requests) or len(slots) != len(requests):
            raise ValueError("graph replay requests, leases, and slots must align")
        if len(leases) != len(set(leases)) or len(slots) != len(set(slots)):
            raise ValueError("graph replay leases and slots must be unique")
        if any(slot < 0 for slot in slots):
            raise ValueError("graph replay slots must be non-negative")
        signature = GraphStorageSignature.from_view(batch_view.storage_view)
        with self._lock:
            captured = self._captures.get(key)
            if captured is None:
                self._rejection_count += 1
                raise KeyError(f"graph key {key!r} is not captured")
            if signature != captured:
                self._rejection_count += 1
                raise ValueError("graph replay KV storage signature changed")
            self._validate_signature(signature)
            lease_pages = tuple(
                (lease_id, self.pool.lease(lease_id).logical_page_ids)
                for lease_id in leases
            )
            pages: set[int] = set()
            epoch = self._next_epoch
            try:
                for lease_id, active_pages in lease_pages:
                    if active_pages:
                        self.pool.mark_in_flight(lease_id, active_pages, epoch=epoch)
                        pages.update(active_pages)
            except Exception:
                # A physical group binds atomically even though the pool API is
                # lease-oriented. Retiring an absent epoch is intentionally safe.
                self.pool.retire_epoch(epoch)
                raise
            for request_id, slot_id in zip(requests, slots, strict=True):
                previous = self._last_request_by_slot.get(slot_id)
                if previous is not None and previous != request_id:
                    self._slot_reuse_count += 1
                self._last_request_by_slot[slot_id] = request_id
            binding = GraphReplayBinding(
                graph_key=key,
                replay_id=self._next_replay_id,
                epoch=epoch,
                generation=signature.generation,
                request_ids=requests,
                lease_ids=leases,
                slot_ids=slots,
                page_ids=tuple(sorted(pages)),
            )
            self._next_replay_id += 1
            self._next_epoch += 1
            self._in_flight[binding.replay_id] = binding
            self._replay_count += 1
            return binding

    def retire(self, binding: GraphReplayBinding) -> None:
        with self._lock:
            current = self._in_flight.get(binding.replay_id)
            if current != binding:
                raise KeyError(f"graph replay {binding.replay_id} is not in flight")
            self.pool.retire_epoch(binding.epoch)
            del self._in_flight[binding.replay_id]

    def invalidate_generation(self, generation: int) -> int:
        target = int(generation)
        with self._lock:
            if any(binding.generation == target for binding in self._in_flight.values()):
                raise RuntimeError("cannot invalidate a graph generation while replay is in flight")
            keys = [
                key
                for key, signature in self._captures.items()
                if signature.generation == target
            ]
            for key in keys:
                del self._captures[key]
            return len(keys)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "generation": self.pool.generation,
                "captures": len(self._captures),
                "capture_count": self._capture_count,
                "replay_count": self._replay_count,
                "rejection_count": self._rejection_count,
                "in_flight": len(self._in_flight),
                "slot_reuse_count": self._slot_reuse_count,
                "graph_keys": sorted(self._captures),
            }

    def assert_conserved(self) -> None:
        with self._lock:
            epochs = {binding.epoch for binding in self._in_flight.values()}
            for page_id in range(self.pool.page_capacity):
                page_epochs = set(self.pool.page(page_id).in_flight_epochs)
                if not page_epochs.issubset(epochs):
                    raise AssertionError(
                        f"KV page {page_id} references an unknown graph epoch"
                    )
            self.pool.assert_conserved()

    def _validate_signature(self, signature: GraphStorageSignature) -> None:
        if signature.generation != self.pool.generation:
            raise ValueError("graph storage generation does not match global KV pool")
        if signature.artifact_fingerprint != self.pool.backend_fingerprint:
            raise ValueError("graph storage artifact does not match global KV pool")


def _key(value: str) -> str:
    key = str(value)
    if not key or key != key.strip():
        raise ValueError("graph_key must be a non-empty trimmed string")
    return key


__all__ = [
    "GraphReplayBinding",
    "GraphReplayBindingRegistry",
    "GraphStorageSignature",
]
