"""Chunked KV pool bookkeeping for continuous-batching admission tests.

The real runtime allocator still owns HIP device memory.  This module provides a
host-only, deterministic model of the C4 dynamic-pool contract: one startup
chunk, append-only growth, stable block ids/pointers, idle tail shrink, and
simple refcount/free-page accounting.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KVPoolAllocation:
    """Allocated fixed-page KV block ids."""

    block_ids: tuple[int, ...]
    pointers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class KVPoolStats:
    """Observable fake KV-pool counters used by scheduler/bench tests."""

    current_pages: int
    current_bytes: int
    high_water_observed_pages: int
    high_water_observed_bytes: int
    free_pages: int
    refcounted_pages: int
    grow_events: int
    grow_failures: int
    shrink_events: int


@dataclass(frozen=True, slots=True)
class KVPoolChunk:
    """One append-only chunk of KV page ids."""

    start_block_id: int
    pages: int

    @property
    def block_ids(self) -> range:
        return range(self.start_block_id, self.start_block_id + self.pages)


class ChunkedKVPool:
    """Host-only chunked page allocator with stable block ids.

    Growth appends fresh block ids past the historical high-water id.  Idle
    shrink removes only fully-free tail chunks and never rewrites the pointer of
    a block id that remains in the pool.
    """

    def __init__(
        self,
        *,
        page_bytes: int,
        initial_pages: int,
        low_water_pages: int | None = None,
        high_water_pages: int | None = None,
        chunk_pages: int | None = None,
        idle_grace_seconds: float = 0.0,
        base_pointer: int = 0x10000000,
    ) -> None:
        if page_bytes <= 0:
            raise ValueError("page_bytes must be positive")
        if initial_pages <= 0:
            raise ValueError("initial_pages must be positive")
        if low_water_pages is not None and low_water_pages <= 0:
            raise ValueError("low_water_pages must be positive")
        if high_water_pages is not None and high_water_pages < initial_pages:
            raise ValueError("high_water_pages cannot be below initial_pages")
        if chunk_pages is not None and chunk_pages <= 0:
            raise ValueError("chunk_pages must be positive")
        if idle_grace_seconds < 0:
            raise ValueError("idle_grace_seconds must be non-negative")
        if base_pointer < 0:
            raise ValueError("base_pointer must be non-negative")
        self.page_bytes = int(page_bytes)
        self.low_water_pages = int(low_water_pages if low_water_pages is not None else initial_pages)
        if self.low_water_pages > int(initial_pages):
            raise ValueError("low_water_pages cannot exceed initial_pages")
        self.high_water_pages = None if high_water_pages is None else int(high_water_pages)
        self.chunk_pages = int(chunk_pages if chunk_pages is not None else initial_pages)
        self.idle_grace_seconds = float(idle_grace_seconds)
        self.base_pointer = int(base_pointer)
        self._chunks: list[KVPoolChunk] = []
        self._free_block_ids: set[int] = set()
        self._refcounts: dict[int, int] = {}
        self._known_pointers: dict[int, int] = {}
        self._next_block_id = 0
        self._grow_events = 0
        self._grow_failures = 0
        self._shrink_events = 0
        self._last_active_seconds = 0.0
        self._high_water_observed_pages = 0
        self._append_chunk(int(initial_pages))

    @property
    def chunks(self) -> tuple[KVPoolChunk, ...]:
        return tuple(self._chunks)

    @property
    def stats(self) -> KVPoolStats:
        free = len(self._free_block_ids)
        refcounted = sum(1 for block_id in self._known_live_block_ids() if self._refcounts.get(block_id, 0) > 0)
        current = self.current_pages
        return KVPoolStats(
            current_pages=current,
            current_bytes=current * self.page_bytes,
            high_water_observed_pages=self._high_water_observed_pages,
            high_water_observed_bytes=self._high_water_observed_pages * self.page_bytes,
            free_pages=free,
            refcounted_pages=refcounted,
            grow_events=self._grow_events,
            grow_failures=self._grow_failures,
            shrink_events=self._shrink_events,
        )

    @property
    def current_pages(self) -> int:
        return sum(chunk.pages for chunk in self._chunks)

    def pointer_for(self, block_id: int) -> int:
        try:
            return self._known_pointers[int(block_id)]
        except KeyError as exc:
            raise KeyError(f"unknown block id {int(block_id)}") from exc

    def refcount(self, block_id: int) -> int:
        return int(self._refcounts.get(int(block_id), 0))

    def allocate(self, pages: int, *, now_seconds: float = 0.0) -> KVPoolAllocation:
        """Allocate fixed pages, growing by chunks when admission needs them."""

        if pages <= 0:
            raise ValueError("pages must be positive")
        self._last_active_seconds = float(now_seconds)
        if len(self._free_block_ids) < int(pages):
            self._grow_for_missing_pages(int(pages) - len(self._free_block_ids))
        if len(self._free_block_ids) < int(pages):
            self._grow_failures += 1
            raise MemoryError("KV pool cannot grow enough pages for admission")
        block_ids = tuple(sorted(self._free_block_ids)[: int(pages)])
        for block_id in block_ids:
            self._free_block_ids.remove(block_id)
            self._refcounts[block_id] = 1
        return KVPoolAllocation(
            block_ids=block_ids,
            pointers=tuple(self.pointer_for(block_id) for block_id in block_ids),
        )

    def incref(self, block_ids: tuple[int, ...] | list[int]) -> None:
        for raw_block_id in block_ids:
            block_id = int(raw_block_id)
            if block_id not in self._known_live_block_ids():
                raise KeyError(f"unknown live block id {block_id}")
            if self._refcounts.get(block_id, 0) <= 0:
                raise ValueError("cannot incref a free block")
            self._refcounts[block_id] += 1

    def release(self, block_ids: tuple[int, ...] | list[int], *, now_seconds: float = 0.0) -> None:
        """Drop one reference per block and return zero-refcount pages to free list."""

        self._last_active_seconds = float(now_seconds)
        for raw_block_id in block_ids:
            block_id = int(raw_block_id)
            if block_id not in self._known_live_block_ids():
                raise KeyError(f"unknown live block id {block_id}")
            count = self._refcounts.get(block_id, 0)
            if count <= 0:
                raise ValueError("block is already free")
            if count == 1:
                self._refcounts[block_id] = 0
                self._free_block_ids.add(block_id)
            else:
                self._refcounts[block_id] = count - 1

    def shrink_idle(self, *, now_seconds: float) -> int:
        """Free fully idle tail chunks after the configured grace period."""

        if float(now_seconds) - self._last_active_seconds < self.idle_grace_seconds:
            return 0
        freed_pages = 0
        while len(self._chunks) > 1 and self.current_pages > self.low_water_pages:
            tail = self._chunks[-1]
            tail_ids = set(tail.block_ids)
            if any(self._refcounts.get(block_id, 0) > 0 for block_id in tail_ids):
                break
            pages_to_free = min(tail.pages, self.current_pages - self.low_water_pages)
            if pages_to_free != tail.pages:
                break
            self._chunks.pop()
            for block_id in tail_ids:
                self._free_block_ids.discard(block_id)
                self._refcounts.pop(block_id, None)
            freed_pages += tail.pages
            self._shrink_events += 1
        return freed_pages

    def block_pointer_map(self, block_ids: tuple[int, ...] | list[int]) -> dict[int, int]:
        return {int(block_id): self.pointer_for(int(block_id)) for block_id in block_ids}

    def _grow_for_missing_pages(self, missing_pages: int) -> None:
        while missing_pages > 0:
            grow_pages = self.chunk_pages
            if self.high_water_pages is not None:
                remaining = self.high_water_pages - self.current_pages
                if remaining <= 0:
                    return
                grow_pages = min(grow_pages, remaining)
            self._append_chunk(grow_pages)
            self._grow_events += 1
            missing_pages -= grow_pages

    def _append_chunk(self, pages: int) -> None:
        if pages <= 0:
            return
        chunk = KVPoolChunk(start_block_id=self._next_block_id, pages=int(pages))
        self._chunks.append(chunk)
        for block_id in chunk.block_ids:
            self._known_pointers[block_id] = self.base_pointer + block_id * self.page_bytes
            self._refcounts[block_id] = 0
            self._free_block_ids.add(block_id)
        self._next_block_id += int(pages)
        self._high_water_observed_pages = max(self._high_water_observed_pages, self.current_pages)

    def _known_live_block_ids(self) -> set[int]:
        known: set[int] = set()
        for chunk in self._chunks:
            known.update(chunk.block_ids)
        return known


__all__ = ["ChunkedKVPool", "KVPoolAllocation", "KVPoolChunk", "KVPoolStats"]
