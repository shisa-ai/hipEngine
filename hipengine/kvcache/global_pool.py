"""Stable global KV page/plane pools with arbitrary-page indirection."""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hipengine.core import Device
from hipengine.kvcache.backend import KVPlaneView, KVStorageView


class KVPageState(str, Enum):
    FREE = "free"
    ACTIVE_PRIVATE = "active_private"
    ACTIVE_SHARED = "active_shared"
    CACHED_EVICTABLE = "cached_evictable"
    PINNED_SESSION = "pinned_session"
    RESERVED_CREDIT = "reserved_credit"
    IN_FLIGHT = "in_flight"


@dataclass(frozen=True, slots=True)
class KVPageRecord:
    page_id: int
    state: KVPageState
    active_lease_ids: tuple[str, ...]
    private_owner_id: str | None
    credit_owner_id: str | None
    cache_references: int
    session_pins: int
    in_flight_epochs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GlobalPageLease:
    lease_id: str
    generation: int
    private_page_ids: tuple[int, ...]
    shared_page_ids: tuple[int, ...]
    growth_credit_page_ids: tuple[int, ...]

    @property
    def owned_page_ids(self) -> tuple[int, ...]:
        return (*self.private_page_ids, *self.shared_page_ids)

    @property
    def logical_page_ids(self) -> tuple[int, ...]:
        return (*self.shared_page_ids, *self.private_page_ids)


@dataclass(slots=True)
class _Page:
    page_id: int
    active_lease_ids: set[str] = field(default_factory=set)
    private_owner_id: str | None = None
    credit_owner_id: str | None = None
    cache_references: int = 0
    session_pins: int = 0
    in_flight_epochs: set[int] = field(default_factory=set)

    @property
    def state(self) -> KVPageState:
        if self.in_flight_epochs:
            return KVPageState.IN_FLIGHT
        if self.session_pins:
            return KVPageState.PINNED_SESSION
        if self.cache_references and not self.active_lease_ids:
            return KVPageState.CACHED_EVICTABLE
        if self.credit_owner_id is not None:
            return KVPageState.RESERVED_CREDIT
        if self.active_lease_ids and (
            len(self.active_lease_ids) > 1
            or self.cache_references > 0
            or self.private_owner_id not in self.active_lease_ids
        ):
            return KVPageState.ACTIVE_SHARED
        if self.active_lease_ids:
            return KVPageState.ACTIVE_PRIVATE
        return KVPageState.FREE


@dataclass(slots=True)
class _Lease:
    lease_id: str
    private_page_ids: list[int]
    shared_page_ids: list[int]
    growth_credit_page_ids: list[int]


class GlobalKVPoolSet:
    """One load-time pool set whose logical pages span arbitrary allocations.

    Device allocation is injected as stable per-page pointers. No operation can
    grow or replace the storage view, which makes graph pointer invariants
    mechanical while allowing leases to cross allocation-segment boundaries.
    """

    def __init__(
        self,
        *,
        backend_fingerprint: str,
        generation: int,
        plane_page_pointers: dict[str, tuple[int, ...]],
        pointer_table_pointers: dict[str, int],
        metadata_descriptor_pointer: int | None = None,
        metadata_descriptor_bytes: int = 256,
        device: Device | None = None,
    ) -> None:
        fingerprint = str(backend_fingerprint)
        if not fingerprint:
            raise ValueError("backend_fingerprint must be non-empty")
        if int(generation) < 0:
            raise ValueError("generation must be non-negative")
        if not plane_page_pointers:
            raise ValueError("global KV pool requires at least one storage plane")
        capacities = {len(pointers) for pointers in plane_page_pointers.values()}
        if len(capacities) != 1 or next(iter(capacities)) <= 0:
            raise ValueError("all storage planes must have the same positive page capacity")
        roles = set(plane_page_pointers)
        if roles != set(pointer_table_pointers):
            raise ValueError("pointer table roles must exactly match storage plane roles")
        if any(not role or role != role.strip() for role in roles):
            raise ValueError("storage plane roles must be non-empty trimmed strings")
        for role, pointers in plane_page_pointers.items():
            if any(int(pointer) <= 0 for pointer in pointers):
                raise ValueError(f"storage plane {role} contains a non-positive page pointer")
            if len(set(int(pointer) for pointer in pointers)) != len(pointers):
                raise ValueError(f"storage plane {role} contains duplicate page pointers")
        if any(int(pointer) <= 0 for pointer in pointer_table_pointers.values()):
            raise ValueError("pointer table pointers must be positive")
        descriptor_bytes = int(metadata_descriptor_bytes)
        if descriptor_bytes <= 0:
            raise ValueError("metadata_descriptor_bytes must be positive")
        descriptor_pointer = (
            max(int(pointer) for pointer in pointer_table_pointers.values()) + 0x1000
            if metadata_descriptor_pointer is None
            else int(metadata_descriptor_pointer)
        )
        if descriptor_pointer <= 0:
            raise ValueError("metadata_descriptor_pointer must be positive")

        self.backend_fingerprint = fingerprint
        self.generation = int(generation)
        self.device = Device("hip", 0) if device is None else device
        self._plane_page_pointers = {
            role: tuple(int(pointer) for pointer in pointers)
            for role, pointers in plane_page_pointers.items()
        }
        self._pointer_table_pointers = {
            role: int(pointer) for role, pointer in pointer_table_pointers.items()
        }
        self._pages = [_Page(page_id) for page_id in range(next(iter(capacities)))]
        self._free_page_ids = set(range(len(self._pages)))
        self._leases: dict[str, _Lease] = {}
        self._lock = threading.RLock()
        self._storage_view = KVStorageView(
            layout_key=f"global-arbitrary-pages:g{self.generation}",
            generation=self.generation,
            planes=tuple(
                KVPlaneView(
                    role=f"{role}.page_table",
                    dtype="int64",
                    ptr=self._pointer_table_pointers[role],
                    shape=(len(self._pages),),
                    strides=(1,),
                )
                for role in sorted(self._plane_page_pointers)
            ),
            artifact_fingerprint=self.backend_fingerprint,
            metadata_descriptor_ptr=descriptor_pointer,
            metadata_descriptor_bytes=descriptor_bytes,
        )

    @property
    def page_capacity(self) -> int:
        return len(self._pages)

    @property
    def free_pages(self) -> int:
        with self._lock:
            return len(self._free_page_ids)

    def storage_view(self) -> KVStorageView:
        return self._storage_view

    def page_pointer(self, plane_role: str, page_id: int) -> int:
        try:
            return self._plane_page_pointers[str(plane_role)][int(page_id)]
        except KeyError as exc:
            raise KeyError(f"unknown KV storage plane {plane_role!r}") from exc
        except IndexError as exc:
            raise IndexError(f"KV page_id {page_id} is outside the global pool") from exc

    def allocate(
        self,
        lease_id: str,
        *,
        private_pages: int,
        growth_credit_pages: int,
        shared_page_ids: tuple[int, ...] = (),
    ) -> GlobalPageLease:
        identifier = _identifier(lease_id, "lease_id")
        private_count = _non_negative(private_pages, "private_pages")
        credit_count = _non_negative(growth_credit_pages, "growth_credit_pages")
        shared = tuple(int(page_id) for page_id in shared_page_ids)
        if len(shared) != len(set(shared)):
            raise ValueError("shared_page_ids must be unique")
        with self._lock:
            if identifier in self._leases:
                raise ValueError(f"KV lease {identifier!r} already exists")
            for page_id in shared:
                page = self._get_page(page_id)
                if page.credit_owner_id is not None or page.state is KVPageState.FREE:
                    raise ValueError(f"KV page {page_id} is not shareable")
            acquired = self._take_free(private_count + credit_count)
            private = acquired[:private_count]
            credits = acquired[private_count:]
            lease = _Lease(identifier, list(private), list(shared), list(credits))
            self._leases[identifier] = lease
            for page_id in private:
                page = self._pages[page_id]
                page.private_owner_id = identifier
                page.active_lease_ids.add(identifier)
            for page_id in shared:
                self._pages[page_id].active_lease_ids.add(identifier)
            for page_id in credits:
                self._pages[page_id].credit_owner_id = identifier
            return self._snapshot_lease(lease)

    def lease(self, lease_id: str) -> GlobalPageLease:
        with self._lock:
            return self._snapshot_lease(self._get_lease(lease_id))

    def page(self, page_id: int) -> KVPageRecord:
        with self._lock:
            page = self._get_page(page_id)
            return KVPageRecord(
                page_id=page.page_id,
                state=page.state,
                active_lease_ids=tuple(sorted(page.active_lease_ids)),
                private_owner_id=page.private_owner_id,
                credit_owner_id=page.credit_owner_id,
                cache_references=page.cache_references,
                session_pins=page.session_pins,
                in_flight_epochs=tuple(sorted(page.in_flight_epochs)),
            )

    def consume_growth_credit(self, lease_id: str) -> int:
        with self._lock:
            lease = self._get_lease(lease_id)
            if not lease.growth_credit_page_ids:
                raise MemoryError(f"KV lease {lease.lease_id} has no growth credit")
            page_id = lease.growth_credit_page_ids.pop(0)
            page = self._pages[page_id]
            if page.credit_owner_id != lease.lease_id:
                raise AssertionError("growth-credit owner mismatch")
            page.credit_owner_id = None
            page.private_owner_id = lease.lease_id
            page.active_lease_ids.add(lease.lease_id)
            lease.private_page_ids.append(page_id)
            return page_id

    def add_growth_credit(self, lease_id: str, pages: int) -> tuple[int, ...]:
        count = _non_negative(pages, "pages")
        with self._lock:
            lease = self._get_lease(lease_id)
            acquired = self._take_free(count)
            for page_id in acquired:
                self._pages[page_id].credit_owner_id = lease.lease_id
            lease.growth_credit_page_ids.extend(acquired)
            return acquired

    def copy_on_write(self, lease_id: str, shared_page_id: int) -> int:
        with self._lock:
            lease = self._get_lease(lease_id)
            page_id = int(shared_page_id)
            if page_id not in lease.shared_page_ids:
                raise ValueError(f"KV page {page_id} is not shared by lease {lease.lease_id}")
            replacement = self.consume_growth_credit(lease.lease_id)
            lease = self._get_lease(lease_id)
            lease.shared_page_ids.remove(page_id)
            old_page = self._pages[page_id]
            old_page.active_lease_ids.remove(lease.lease_id)
            self._maybe_free(old_page)
            return replacement

    def mark_in_flight(
        self,
        lease_id: str,
        page_ids: tuple[int, ...],
        *,
        epoch: int,
    ) -> None:
        if int(epoch) < 0:
            raise ValueError("epoch must be non-negative")
        with self._lock:
            lease = self._get_lease(lease_id)
            owned = set(lease.private_page_ids) | set(lease.shared_page_ids)
            for page_id in page_ids:
                if int(page_id) not in owned:
                    raise ValueError(f"KV page {page_id} is not active in lease {lease.lease_id}")
            for page_id in page_ids:
                self._pages[int(page_id)].in_flight_epochs.add(int(epoch))

    def retire_epoch(self, epoch: int) -> None:
        target = int(epoch)
        with self._lock:
            for page in self._pages:
                if target in page.in_flight_epochs:
                    page.in_flight_epochs.remove(target)
                    self._maybe_free(page)

    def retain_cache(self, page_ids: tuple[int, ...]) -> None:
        pages = tuple(int(page_id) for page_id in page_ids)
        if not pages or len(pages) != len(set(pages)):
            raise ValueError("cache page_ids must be non-empty and unique")
        with self._lock:
            for page_id in pages:
                page = self._get_page(page_id)
                if page.credit_owner_id is not None or not page.active_lease_ids:
                    raise ValueError(f"KV page {page_id} is not active and cacheable")
            for page_id in pages:
                self._pages[page_id].cache_references += 1

    def release_cache(self, page_ids: tuple[int, ...]) -> None:
        pages = tuple(int(page_id) for page_id in page_ids)
        if not pages or len(pages) != len(set(pages)):
            raise ValueError("cache page_ids must be non-empty and unique")
        with self._lock:
            for page_id in pages:
                page = self._get_page(page_id)
                if page.cache_references <= 0:
                    raise ValueError(f"KV page {page_id} has no cache reference")
            for page_id in pages:
                page = self._pages[page_id]
                page.cache_references -= 1
                self._maybe_free(page)

    def pin_session(self, lease_id: str, page_ids: tuple[int, ...]) -> None:
        with self._lock:
            lease = self._get_lease(lease_id)
            owned = set(lease.private_page_ids) | set(lease.shared_page_ids)
            for page_id in page_ids:
                if int(page_id) not in owned:
                    raise ValueError(f"KV page {page_id} is not active in lease {lease.lease_id}")
            for page_id in page_ids:
                self._pages[int(page_id)].session_pins += 1

    def unpin_session(self, page_ids: tuple[int, ...]) -> None:
        with self._lock:
            for page_id in page_ids:
                page = self._get_page(page_id)
                if page.session_pins <= 0:
                    raise ValueError(f"KV page {page_id} is not session pinned")
            for page_id in page_ids:
                page = self._pages[int(page_id)]
                page.session_pins -= 1
                self._maybe_free(page)

    def release(self, lease_id: str) -> GlobalPageLease:
        with self._lock:
            identifier = str(lease_id)
            lease = self._get_lease(identifier)
            snapshot = self._snapshot_lease(lease)
            for page_id in (*lease.private_page_ids, *lease.shared_page_ids):
                page = self._pages[page_id]
                page.active_lease_ids.remove(identifier)
                if page.private_owner_id == identifier:
                    page.private_owner_id = None
                self._maybe_free(page)
            for page_id in lease.growth_credit_page_ids:
                page = self._pages[page_id]
                if page.credit_owner_id != identifier:
                    raise AssertionError("growth-credit owner mismatch during release")
                page.credit_owner_id = None
                self._maybe_free(page)
            del self._leases[identifier]
            return snapshot

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counts = Counter(page.state.value for page in self._pages)
            return {
                "backend_fingerprint": self.backend_fingerprint,
                "generation": self.generation,
                "page_capacity": self.page_capacity,
                "free_pages": len(self._free_page_ids),
                "counts": dict(sorted(counts.items())),
                "lease_count": len(self._leases),
                "leases": {
                    lease_id: {
                        "private_page_ids": list(lease.private_page_ids),
                        "shared_page_ids": list(lease.shared_page_ids),
                        "growth_credit_page_ids": list(lease.growth_credit_page_ids),
                    }
                    for lease_id, lease in sorted(self._leases.items())
                },
            }

    def assert_conserved(self) -> None:
        with self._lock:
            expected_free = {
                page.page_id for page in self._pages if page.state is KVPageState.FREE
            }
            if expected_free != self._free_page_ids:
                raise AssertionError(
                    f"global KV free-page mismatch: expected={expected_free}, actual={self._free_page_ids}"
                )
            for page in self._pages:
                if page.credit_owner_id is not None and (
                    page.active_lease_ids or page.private_owner_id is not None
                ):
                    raise AssertionError(f"KV page {page.page_id} is credit and active")
                if page.private_owner_id is not None and (
                    page.private_owner_id not in page.active_lease_ids
                ):
                    raise AssertionError(f"KV page {page.page_id} private owner is not active")
                for lease_id in page.active_lease_ids:
                    lease = self._leases.get(lease_id)
                    if lease is None or page.page_id not in (
                        *lease.private_page_ids,
                        *lease.shared_page_ids,
                    ):
                        raise AssertionError(
                            f"KV page {page.page_id} has stale active lease {lease_id}"
                        )
                if page.credit_owner_id is not None:
                    lease = self._leases.get(page.credit_owner_id)
                    if lease is None or page.page_id not in lease.growth_credit_page_ids:
                        raise AssertionError(
                            f"KV page {page.page_id} has stale credit owner"
                        )
            for lease in self._leases.values():
                all_ids = (
                    *lease.private_page_ids,
                    *lease.shared_page_ids,
                    *lease.growth_credit_page_ids,
                )
                if len(all_ids) != len(set(all_ids)):
                    raise AssertionError(f"KV lease {lease.lease_id} contains duplicate pages")

    def _take_free(self, count: int) -> tuple[int, ...]:
        if count > len(self._free_page_ids):
            raise MemoryError(
                f"global KV page pool cannot allocate {count} pages with "
                f"{len(self._free_page_ids)} free"
            )
        selected = tuple(sorted(self._free_page_ids)[:count])
        self._free_page_ids.difference_update(selected)
        return selected

    def _maybe_free(self, page: _Page) -> None:
        if page.state is KVPageState.FREE:
            self._free_page_ids.add(page.page_id)

    def _get_page(self, page_id: int) -> _Page:
        identifier = int(page_id)
        if identifier < 0 or identifier >= len(self._pages):
            raise IndexError(f"KV page_id {page_id} is outside the global pool")
        return self._pages[identifier]

    def _get_lease(self, lease_id: str) -> _Lease:
        identifier = str(lease_id)
        try:
            return self._leases[identifier]
        except KeyError as exc:
            raise KeyError(f"unknown KV lease {identifier!r}") from exc

    def _snapshot_lease(self, lease: _Lease) -> GlobalPageLease:
        return GlobalPageLease(
            lease_id=lease.lease_id,
            generation=self.generation,
            private_page_ids=tuple(lease.private_page_ids),
            shared_page_ids=tuple(lease.shared_page_ids),
            growth_credit_page_ids=tuple(lease.growth_credit_page_ids),
        )


def _identifier(value: str, field_name: str) -> str:
    result = str(value)
    if not result or result != result.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string")
    return result


def _non_negative(value: int, field_name: str) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return result


__all__ = [
    "GlobalKVPoolSet",
    "GlobalPageLease",
    "KVPageRecord",
    "KVPageState",
]
