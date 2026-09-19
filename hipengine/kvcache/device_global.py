"""Device-backed adapter for the generation-checked global KV page pool."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from hipengine.kvcache.global_pool import GlobalKVPoolSet
from hipengine.kvcache.pool import (
    DeviceKVContiguityError,
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
        grow_storage: Callable[
            [int, int],
            tuple[Mapping[str, Sequence[int]], Mapping[str, int]]
            | tuple[Mapping[str, Sequence[int]], Mapping[str, int], Any],
        ]
        | None = None,
        before_grow: Callable[[], None] | None = None,
        max_pages: int | None = None,
        growth_chunk_pages: int | None = None,
        on_pressure: Callable[[int], None] | None = None,
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
        self.high_water_pages = (
            self.global_pool.page_capacity
            if max_pages is None
            else int(max_pages)
        )
        self.chunk_pages = (
            self.global_pool.page_capacity
            if growth_chunk_pages is None
            else max(1, int(growth_chunk_pages))
        )
        self.idle_grace_seconds = 0.0
        self._backing = backing
        # Storage chunks in page-id order: (start_page, page_count, backing).
        # Growth appends a chunk with its own device buffers, and consumers
        # address a chunk from its own base, so an allocation must not span two.
        self._chunks: list[tuple[int, int, Any]] = [
            (0, int(self.global_pool.page_capacity), backing)
        ]
        self._close_storage = close_storage
        self._grow_storage = grow_storage
        self._before_grow = before_grow
        self._max_pages = None if max_pages is None else int(max_pages)
        if self._max_pages is not None and self._max_pages < self.global_pool.page_capacity:
            raise ValueError("max_pages cannot be below the initial pool capacity")
        self._growth_chunk_pages = (
            max(1, int(growth_chunk_pages))
            if growth_chunk_pages is not None
            else 1
        )
        self._on_pressure = on_pressure
        self._primary_plane = sorted(planes)[0]
        self._request_allocations: dict[int, DeviceKVPoolAllocation] = {}
        self._workspace_leases: dict[str, tuple[int, ...]] = {}
        self._private_workspace_reservations: dict[object, int] = {}
        self._pin_counts: dict[int, int] = {}
        self._last_active_seconds = 0.0
        self._high_water_observed_pages = 0
        self._prefix_reuse_events = 0
        self._prefix_reused_pages = 0
        self._cow_fork_events = 0
        self._cow_forked_pages = 0
        self._allocation_failures = 0
        self._grow_events = 0
        self._closed = False
        self._lock = threading.RLock()

    @property
    def chunks(self) -> tuple[KVPoolChunk, ...]:
        return (KVPoolChunk(start_block_id=0, pages=self.current_pages),)

    @property
    def current_pages(self) -> int:
        return self.global_pool.page_capacity

    @property
    def max_pages(self) -> int | None:
        return self._max_pages

    @property
    def budget_bytes(self) -> int | None:
        return None if self._max_pages is None else self._max_pages * self.page_bytes

    @property
    def private_workspace_bytes(self) -> int:
        with self._lock:
            return sum(self._private_workspace_reservations.values())

    @property
    def accounted_bytes(self) -> int:
        """Allocated arena plus reserved private KV payload, including idle pages."""
        with self._lock:
            return self.current_pages * self.page_bytes + self.private_workspace_bytes

    def reserve_private_workspace(self, nbytes: int) -> object:
        """Charge private KV before allocation; retain the charge until buffers free."""
        count = int(nbytes)
        if count <= 0:
            raise ValueError("private workspace bytes must be positive")
        with self._lock:
            self._require_open()
            self._check_byte_budget(count)
            token = object()
            self._private_workspace_reservations[token] = count
            return token

    def release_private_workspace(self, token: object) -> None:
        with self._lock:
            del self._private_workspace_reservations[token]

    def _check_byte_budget(self, additional_bytes: int) -> None:
        budget = self.budget_bytes
        if budget is not None and self.accounted_bytes + additional_bytes > budget:
            raise MemoryError(
                "arena and private workspace KV exceed the pool budget: "
                f"{self.accounted_bytes} + {additional_bytes} > {budget} bytes"
            )

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
                grow_events=self._grow_events,
                grow_failures=self._allocation_failures,
                shrink_events=0,
                prefix_reuse_events=self._prefix_reuse_events,
                prefix_reused_pages=self._prefix_reused_pages,
                cow_fork_events=self._cow_fork_events,
                cow_forked_pages=self._cow_forked_pages,
            )

    def storage_view(self):
        return self.global_pool.storage_view()

    @property
    def backing(self) -> Any:
        """Return the pool's plane backing (per-layer contiguous plane buffers).

        Workspace leases borrow these planes: a packed execution workspace
        addresses ``plane_ptr + page_id * plane_page_bytes`` exactly like the
        request path, so its KV payload lives inside the same arena.
        """

        return self._backing

    def pointer_for(self, block_id: int) -> int:
        return self.global_pool.page_pointer(self._primary_plane, int(block_id))

    def refcount(self, block_id: int) -> int:
        record = self.global_pool.page(int(block_id))
        return len(record.active_lease_ids) + int(record.cache_references)

    def pin_count(self, block_id: int) -> int:
        return int(self.global_pool.page(int(block_id)).session_pins)

    def lease_workspace(
        self,
        key: str,
        pages: int,
        *,
        now_seconds: float = 0.0,
    ) -> tuple[int, ...]:
        """Lease pinned non-request pages for a persistent execution workspace.

        Workspace pages are ledger-owned exactly like request pages: they are
        not free, they count as pinned, they are visible in stats, and they
        must be released before ``close()``. Unlike request leases they are
        keyed by a stable workspace name so load-time execution state (for
        example the packed-AR KV backing) can live inside the same global
        arena and ledger instead of a private hidden allocation.
        """

        name = str(key)
        if not name:
            raise ValueError("workspace key must be non-empty")
        count = int(pages)
        if count <= 0:
            raise ValueError("workspace pages must be positive")
        lease_id = self._workspace_lease_id(name)
        with self._lock:
            self._require_open()
            if lease_id in self._workspace_leases:
                raise ValueError(f"workspace lease {name!r} already exists")
            self._ensure_free_pages(count, now_seconds=now_seconds)
            try:
                lease = self.global_pool.allocate(
                    lease_id,
                    private_pages=count,
                    growth_credit_pages=0,
                )
            except MemoryError:
                self._allocation_failures += 1
                raise
            page_ids = tuple(int(page_id) for page_id in lease.private_page_ids)
            self.global_pool.pin_session(lease_id, page_ids)
            self._workspace_leases[lease_id] = page_ids
            self._last_active_seconds = float(now_seconds)
            self._observe_high_water()
            return page_ids

    def release_workspace(self, key: str) -> tuple[int, ...]:
        """Release a workspace lease previously created by ``lease_workspace``."""

        name = str(key)
        lease_id = self._workspace_lease_id(name)
        with self._lock:
            try:
                page_ids = self._workspace_leases.pop(lease_id)
            except KeyError:
                raise KeyError(f"workspace lease {name!r} does not exist") from None
            self.global_pool.unpin_session(page_ids)
            self.global_pool.release(lease_id)
            return page_ids

    def workspace_pages(self, key: str) -> tuple[int, ...] | None:
        """Return the leased page IDs for a workspace, or None when absent."""

        with self._lock:
            return self._workspace_leases.get(self._workspace_lease_id(str(key)))

    def allocate(
        self,
        request_id: int,
        pages: int,
        *,
        now_seconds: float = 0.0,
        require_contiguous: bool = False,
    ) -> DeviceKVPoolAllocation:
        """Lease request pages, optionally requiring one contiguous page-id run.

        The long-context packed prefill path can only reach a context at or above
        the AOTriton slot threshold through a slot-local contiguous KV view, so a
        caller that will need one asks for a run here. Cached prefix pages are
        reclaimable and can occupy the only free run, so a failed placement asks
        for them once before reporting the failure.
        """

        rid = int(request_id)
        count = int(pages)
        if count <= 0:
            raise ValueError("pages must be positive")
        with self._lock:
            self._require_open()
            if rid in self._request_allocations:
                raise ValueError(f"request_id {rid} already has a device KV allocation")
            lease_id = self._lease_id(rid)
            pressure_released = False
            while True:
                try:
                    lease = self._allocate_within_one_chunk(
                        lease_id,
                        private_pages=count,
                        require_contiguous=bool(require_contiguous),
                    )
                    break
                except DeviceKVContiguityError:
                    if pressure_released or not callable(self._on_pressure):
                        raise
                    pressure_released = True
                    self._on_pressure(count)
                except MemoryError:
                    if not pressure_released and callable(self._on_pressure):
                        pressure_released = True
                        free = self.global_pool.free_pages
                        self._on_pressure(count - free if free < count else count)
                        continue
                    # Request KV must fit within one backing chunk. Growing
                    # only the aggregate free-page deficit can leave every
                    # chunk too small and retry forever once total free >= count.
                    growth = max(self._growth_chunk_pages, count)
                    if self.budget_bytes is not None:
                        remaining = (
                            self.budget_bytes - self.accounted_bytes
                        ) // self.page_bytes
                        growth = min(growth, remaining)
                    if growth < count:
                        self._allocation_failures += 1
                        raise MemoryError(
                            f"cannot allocate {count} KV pages in one backing "
                            "chunk within the pool budget"
                        )
                    self.grow(growth, now_seconds=now_seconds)
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
        require_contiguous: bool = False,
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
            # Growth appends a NEW chunk, and a shared admission has to place its
            # suffix in the chunk that already holds the prefix, so growing can
            # never satisfy this path - it would only enlarge the pool for good.
            # Eviction can, because it frees pages inside existing chunks.
            if self.global_pool.free_pages < private and callable(self._on_pressure):
                self._on_pressure(private - self.global_pool.free_pages)
            try:
                lease = self._allocate_within_one_chunk(
                    self._lease_id(rid),
                    private_pages=private,
                    require_contiguous=bool(require_contiguous),
                    shared_page_ids=shared,
                )
            except MemoryError:
                self._allocation_failures += 1
                raise
            allocation = self._allocation(rid, lease)
            self._request_allocations[rid] = allocation
            self._last_active_seconds = float(now_seconds)
            self._prefix_reuse_events += 1
            self._prefix_reused_pages += len(shared)
            self._observe_high_water()
            return allocation

    def _ensure_free_pages(self, pages: int, *, now_seconds: float = 0.0) -> None:
        """Make enough unowned pages available within the configured budget."""

        needed = int(pages)
        if needed <= 0 or self.global_pool.free_pages >= needed:
            return
        if callable(self._on_pressure):
            self._on_pressure(needed - self.global_pool.free_pages)
        missing = needed - self.global_pool.free_pages
        if missing <= 0:
            return
        target = self.global_pool.page_capacity + max(
            self._growth_chunk_pages, missing
        )
        if self._max_pages is not None:
            target = min(
                target,
                (self.budget_bytes - self.private_workspace_bytes) // self.page_bytes,
            )
        if target <= self.global_pool.page_capacity:
            self._allocation_failures += 1
            raise MemoryError(
                f"cannot allocate {needed} KV pages within the pool budget"
            )
        self.grow(target - self.global_pool.page_capacity, now_seconds=now_seconds)

    def fork_copy_on_write(
        self,
        request_id: int,
        prefix_block_ids: Sequence[int],
        *,
        suffix_pages: int,
        first_divergent_token: int,
        now_seconds: float = 0.0,
        require_contiguous: bool = False,
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
            require_contiguous=bool(require_contiguous),
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

    def grow(
        self,
        pages: int,
        *,
        now_seconds: float = 0.0,
    ) -> int:
        """Append device-backed pages through the runtime growth callback."""

        count = int(pages)
        if count <= 0:
            raise ValueError("pages must be positive")
        grow_storage = getattr(self, "_grow_storage", None)
        if not callable(grow_storage):
            self._allocation_failures += 1
            raise MemoryError(
                "cannot allocate KV pages: global device KV pool has no growth provider"
            )
        with self._lock:
            self._require_open()
            try:
                self._check_byte_budget(count * self.page_bytes)
                if callable(self._before_grow):
                    self._before_grow()
                chunk_start = int(self.global_pool.page_capacity)
                appended = grow_storage(count, chunk_start)
                # A provider may return its new chunk's backing as a third
                # element; without it the pool cannot confine allocations to the
                # appended chunk and growth stays unusable for consumers that
                # address one chunk at a time.
                if len(appended) == 3:
                    plane_pointers, pointer_tables, chunk_backing = appended
                else:
                    plane_pointers, pointer_tables = appended
                    chunk_backing = None
                self.global_pool.append_pages(plane_pointers, pointer_tables)
                added = int(self.global_pool.page_capacity) - chunk_start
                if added > 0:
                    self._chunks.append(
                        (
                            chunk_start,
                            added,
                            self._backing if chunk_backing is None else chunk_backing,
                        )
                    )
            except MemoryError:
                self._allocation_failures += 1
                raise
            self._last_active_seconds = float(now_seconds)
            self._grow_events += 1
            return count

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._request_allocations:
                raise RuntimeError(
                    "cannot close global device KV pool with live request allocations"
                )
            if self._private_workspace_reservations:
                raise RuntimeError("cannot close global device KV pool with private workspace")
            snapshot = self.global_pool.snapshot()
            if int(snapshot["free_pages"]) != int(snapshot["page_capacity"]):
                raise RuntimeError(
                    "cannot close global device KV pool with retained or pinned pages"
                )
            self.global_pool.assert_conserved()
            self._close_storage()
            self._closed = True

    @staticmethod
    def _workspace_lease_id(key: str) -> str:
        return f"workspace:{key}"

    @staticmethod
    def _lease_id(request_id: int) -> str:
        return f"request:{int(request_id)}"

    def _allocate_within_one_chunk(
        self,
        lease_id: str,
        *,
        private_pages: int,
        require_contiguous: bool,
        shared_page_ids: tuple[int, ...] = (),
    ) -> Any:
        """Lease pages from a single storage chunk.

        Every consumer of this pool addresses KV as ``chunk_base + page *
        page_stride`` with the page id taken straight from the block table, so an
        allocation whose pages straddle two appended chunks indexes past the end
        of the first one. Before this confinement a grown pool produced exactly
        that: a GPU memory fault inside the paged-KV write kernel.

        With one chunk this is the previous behaviour. With several, the chunk
        holding any shared pages is required; otherwise chunks are tried in
        page-id order and the last failure is reported, so a caller still sees
        ``MemoryError`` (which admission turns into a clean rejection) or
        ``DeviceKVContiguityError`` rather than an unbindable placement.
        """

        if len(self._chunks) == 1:
            return self.global_pool.allocate(
                lease_id,
                private_pages=int(private_pages),
                growth_credit_pages=0,
                shared_page_ids=shared_page_ids,
                require_contiguous=bool(require_contiguous),
            )
        if shared_page_ids:
            owning = self._chunk_for_page(int(shared_page_ids[0]))
            if any(
                self._chunk_for_page(int(page_id)) is not owning
                for page_id in shared_page_ids
            ):
                raise MemoryError("shared KV prefix pages span two storage chunks")
            candidates = [owning]
        else:
            candidates = list(self._chunks)
        failure: BaseException | None = None
        for start, page_count, _chunk_backing in candidates:
            try:
                return self.global_pool.allocate(
                    lease_id,
                    private_pages=int(private_pages),
                    growth_credit_pages=0,
                    shared_page_ids=shared_page_ids,
                    require_contiguous=bool(require_contiguous),
                    within_page_range=(int(start), int(start) + int(page_count)),
                )
            except (MemoryError, DeviceKVContiguityError) as exc:
                failure = exc
        assert failure is not None
        raise failure

    def _chunk_for_page(self, page_id: int) -> tuple[int, int, Any]:
        for chunk in self._chunks:
            start, page_count, _ = chunk
            if start <= int(page_id) < start + int(page_count):
                return chunk
        raise ValueError(f"KV page {int(page_id)} is outside every storage chunk")

    def _allocation(self, request_id: int, lease: Any) -> DeviceKVPoolAllocation:
        block_ids = tuple(int(page_id) for page_id in lease.logical_page_ids)
        # Confinement guarantees one chunk owns every page, so the first page
        # names the chunk whose buffers the consumer must bind.
        chunk_start, _chunk_pages, chunk_backing = self._chunk_for_page(block_ids[0])
        return DeviceKVPoolAllocation(
            request_id=int(request_id),
            block_ids=block_ids,
            pointers=tuple(self.pointer_for(page_id) for page_id in block_ids),
            chunk_start_block_id=int(chunk_start),
            backing=chunk_backing,
            reused_block_ids=tuple(int(page_id) for page_id in lease.shared_page_ids),
            allocated_block_ids=tuple(int(page_id) for page_id in lease.private_page_ids),
            # Growth appends backing chunks while keeping page ids stable, so
            # the first chunk's page count stops being the right bound as soon
            # as the pool grows. The pointer tables cover every page.
            pool_page_capacity=int(self.global_pool.page_capacity),
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
