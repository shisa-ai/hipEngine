from __future__ import annotations

import pytest

from hipengine.core.memory import DeviceMemoryArena, free, malloc, memory_stats, reset_memory_stats


class FakeRuntime:
    def __init__(self) -> None:
        self.next_ptr = 0x1000
        self.freed: list[int] = []

    def malloc(self, nbytes: int) -> int:
        self.next_ptr += 0x1000
        return self.next_ptr

    def free(self, ptr: int) -> None:
        self.freed.append(int(ptr))


def test_memory_stats_track_current_peak_and_reset_live_allocations() -> None:
    runtime = FakeRuntime()
    reset_memory_stats()

    first = malloc(4, runtime=runtime)  # type: ignore[arg-type]
    second = malloc(6, runtime=runtime)  # type: ignore[arg-type]

    stats = memory_stats()
    assert stats["current_allocated_bytes"] == 10
    assert stats["peak_allocated_bytes"] == 10
    assert stats["total_allocated_bytes"] == 10
    assert stats["active_allocations"] == 2

    free(first, runtime=runtime)  # type: ignore[arg-type]
    stats = memory_stats()
    assert stats["current_allocated_bytes"] == 6
    assert stats["peak_allocated_bytes"] == 10
    assert stats["total_freed_bytes"] == 4
    assert stats["active_allocations"] == 1

    reset_memory_stats()
    stats = memory_stats()
    assert stats["current_allocated_bytes"] == 6
    assert stats["peak_allocated_bytes"] == 6
    assert stats["total_allocated_bytes"] == 0
    assert stats["total_freed_bytes"] == 0
    assert stats["active_allocations"] == 1

    free(second, runtime=runtime)  # type: ignore[arg-type]
    # Double-free of an already-untracked pointer should not underflow counters.
    free(second, runtime=runtime)  # type: ignore[arg-type]
    stats = memory_stats()
    assert stats["current_allocated_bytes"] == 0
    assert stats["total_freed_bytes"] == 6
    assert stats["active_allocations"] == 0

    reset_memory_stats()


def test_device_arena_allocates_aligned_views_and_tracks_one_owner() -> None:
    runtime = FakeRuntime()
    reset_memory_stats()
    arena = DeviceMemoryArena.create(20_480, runtime=runtime, alignment=4096)  # type: ignore[arg-type]

    views = tuple(arena.allocate(nbytes) for nbytes in (1, 4097, 8192))

    assert tuple(view.ptr - arena.owner.ptr for view in views) == (0, 4096, 12_288)
    assert arena.requested_bytes == 12_290
    assert arena.used_bytes == 20_480
    assert arena.allocation_count == 3
    assert memory_stats()["current_allocated_bytes"] == 20_480
    assert memory_stats()["active_allocations"] == 1

    for view in views:
        arena.release(view)
    assert runtime.freed == []
    arena.close()
    assert runtime.freed == [arena.owner.ptr]
    assert memory_stats()["current_allocated_bytes"] == 0
    assert memory_stats()["active_allocations"] == 0


def test_device_arena_rejects_overflow_and_double_close_is_safe() -> None:
    runtime = FakeRuntime()
    reset_memory_stats()
    arena = DeviceMemoryArena.create(4096, runtime=runtime, alignment=4096)  # type: ignore[arg-type]

    arena.allocate(4096)
    with pytest.raises(MemoryError, match="arena capacity"):
        arena.allocate(1)

    arena.close()
    arena.close()
    assert runtime.freed == [arena.owner.ptr]
