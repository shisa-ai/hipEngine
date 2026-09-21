"""Grow-then-shrink arena ownership: final reclaim leaves nothing live.

The readiness checklist requires that a pool grow/shrink cycle with MTP work
interleaved leaves zero outstanding device allocations after the final reclaim
and after engine close, with retained target snapshot arenas closed. This pins
that contract for ``_GGUFPrefixSnapshotArenaPool``, which had no coverage at
all.

The pool and ``DeviceMemoryArena`` reach the device only through
``hipengine.core.memory.malloc`` / ``free``, which delegate to
``runtime.malloc`` / ``runtime.free`` and record into the process-local
tracker. A fake runtime therefore exercises the real ownership path on CPU,
and ``memory_stats()`` supplies the "zero outstanding" assertion directly.

An arena retains its allocation while pooled; ``rewind`` keeps the same HIP
owner so a later acquire is allocation-free. Orphaned ownership would show up
as either a live tracked allocation after the final close, or an arena left
unclosed.
"""

from __future__ import annotations

import itertools

import pytest

from hipengine.core.memory import DeviceMemoryArena, memory_stats
from hipengine.runtime.qwen35_gguf_runner import _GGUFPrefixSnapshotArenaPool

# The tracker keys live allocations by pointer, so synthetic pointers must not
# collide with each other or with any allocation another test left live. Each
# fake runtime gets its own high, disjoint address range.
_RUNTIME_ORDINAL = itertools.count()
_RUNTIME_BASE = 0x7F00_0000_0000
_RUNTIME_STRIDE = 0x1000_0000


class _FakeRuntime:
    """Hand out unique synthetic pointers and record every release."""

    def __init__(self) -> None:
        self.next_ptr = _RUNTIME_BASE + next(_RUNTIME_ORDINAL) * _RUNTIME_STRIDE
        self.freed: list[int] = []

    def malloc(self, nbytes: int) -> int:
        ptr = self.next_ptr
        self.next_ptr += max(4096, int(nbytes))
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(int(ptr))


def _live_allocations() -> int:
    return int(memory_stats()["active_allocations"])


def test_prefix_arena_pool_reclaim_leaves_no_live_allocation() -> None:
    """Grow past the retention budget, shrink, then close: nothing stays live."""

    runtime = _FakeRuntime()
    baseline = _live_allocations()
    pool = _GGUFPrefixSnapshotArenaPool(runtime, max_retained=2)

    # Growth: more distinct arenas than the retention budget.
    arenas = [pool.acquire(8192) for _ in range(5)]
    assert pool.stats()["arena_allocations"] == 5
    assert _live_allocations() == baseline + 5

    # Shrink: releasing all five retains at most two and frees the rest.
    for arena in arenas:
        pool.release(arena)

    assert pool.stats()["retained_arenas"] == 2
    assert _live_allocations() == baseline + 2
    closed = [arena for arena in arenas if arena.closed]
    assert len(closed) == 3
    assert len(runtime.freed) == 3

    # Final reclaim: every retained arena is closed and its owner released.
    pool.close()

    assert pool.stats()["retained_arenas"] == 0
    assert all(arena.closed for arena in arenas)
    assert len(runtime.freed) == 5
    assert _live_allocations() == baseline


def test_prefix_arena_pool_retains_by_capacity_without_cross_bucketing() -> None:
    """A geometry change must not hand back an arena of the wrong size.

    Retention is keyed by capacity, so a shrink that changes the snapshot
    geometry allocates a correctly sized arena rather than reusing a stale one.
    """

    runtime = _FakeRuntime()
    baseline = _live_allocations()
    pool = _GGUFPrefixSnapshotArenaPool(runtime, max_retained=4)

    small = pool.acquire(4096)
    pool.release(small)

    large = pool.acquire(65536)

    assert large.capacity_bytes == 65536
    assert pool.stats()["arena_allocations"] == 2
    assert pool.stats()["retained_arenas"] == 1

    # Same capacity does reuse, and reuse allocates nothing new.
    pool.release(large)
    again = pool.acquire(65536)

    assert again is large
    assert pool.stats()["arena_allocations"] == 2
    assert _live_allocations() == baseline + 2

    # The reused arena is out of the pool while acquired, so it must be
    # returned before the final reclaim can free it.
    pool.release(again)
    pool.close()
    assert _live_allocations() == baseline


def test_prefix_arena_pool_release_after_close_frees_instead_of_retaining() -> None:
    """A late release must not resurrect ownership in a closed pool.

    This is the engine-close case: a snapshot that outlives the pool returns
    its arena, and the arena must be freed rather than retained by a pool that
    will never close it again.
    """

    runtime = _FakeRuntime()
    baseline = _live_allocations()
    pool = _GGUFPrefixSnapshotArenaPool(runtime, max_retained=4)
    arena = pool.acquire(8192)

    pool.close()
    assert arena.closed is False
    assert _live_allocations() == baseline + 1

    pool.release(arena)

    assert arena.closed is True
    assert pool.stats()["retained_arenas"] == 0
    assert runtime.freed == [arena.owner.ptr]
    assert _live_allocations() == baseline


def test_prefix_arena_pool_close_is_idempotent_and_refuses_late_acquire() -> None:
    """Closing twice frees nothing twice, and a closed pool refuses to allocate."""

    runtime = _FakeRuntime()
    baseline = _live_allocations()
    pool = _GGUFPrefixSnapshotArenaPool(runtime, max_retained=4)
    pool.release(pool.acquire(8192))

    pool.close()
    pool.close()

    assert len(runtime.freed) == 1
    assert _live_allocations() == baseline
    with pytest.raises(RuntimeError, match="closed"):
        pool.acquire(8192)


def test_prefix_arena_pool_zero_retention_frees_every_release() -> None:
    """``max_retained=0`` is a legal configuration and retains nothing."""

    runtime = _FakeRuntime()
    baseline = _live_allocations()
    pool = _GGUFPrefixSnapshotArenaPool(runtime, max_retained=0)

    arenas = [pool.acquire(4096) for _ in range(3)]
    for arena in arenas:
        pool.release(arena)

    assert pool.stats()["retained_arenas"] == 0
    assert all(arena.closed for arena in arenas)
    assert _live_allocations() == baseline

    pool.close()
    assert _live_allocations() == baseline


def test_device_arena_close_releases_its_owner_exactly_once() -> None:
    """The arena itself must not double-free or leak its owner on re-close."""

    runtime = _FakeRuntime()
    baseline = _live_allocations()
    arena = DeviceMemoryArena.create(8192, runtime=runtime)

    assert _live_allocations() == baseline + 1

    arena.close()
    arena.close()

    assert runtime.freed == [arena.owner.ptr]
    assert _live_allocations() == baseline
