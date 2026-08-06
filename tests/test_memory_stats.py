from __future__ import annotations

from hipengine.core.memory import free, malloc, memory_stats, reset_memory_stats


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
