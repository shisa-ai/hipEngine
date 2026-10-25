"""Prefix-state capture ownership for GGUF hybrid resident sessions.

``Qwen35GGUFPrefixStateSnapshot.capture`` copies one boundary's paired per-layer
Conv/GDN state into a single pooled arena. The transaction is all-or-nothing on
device memory:

1. A failed capture leaves nothing live: the arena goes back to the pool (or is
   closed by it), and no per-layer view is ever freed individually.
2. Copies are issued asynchronously, so a failure after a copy may have been
   submitted must retire that work before the arena can return to the pool.
3. If the cleanup synchronization itself fails, the arena is quarantined in the
   pool: it is never recycled and never freed until a later ``pool.close()``
   synchronizes the capture stream. A close whose synchronization fails leaves
   the quarantine and the pool retryable, with nothing freed.

Everything here runs on CPU through the real capture/close paths with a fake
runtime; no HIP, no GPU.
"""

from __future__ import annotations

import itertools

import pytest

from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer, memory_stats
from hipengine.runtime.qwen35_gguf_prefix_state import (
    Qwen35GGUFPrefixStateSnapshot,
    _GGUFPrefixSnapshotArenaPool,
)

# The process-local allocation tracker keys live buffers by pointer, so each
# fake runtime gets its own high, disjoint address range.
_RUNTIME_ORDINAL = itertools.count()
_RUNTIME_BASE = 0x7E00_0000_0000
_RUNTIME_STRIDE = 0x0100_0000

# Synthetic live-state pointers, disjoint from every fake runtime range.
_STATE_BASE = 0x7D00_0000_0000
_STATE_STRIDE = 0x0000_1000


def _live_allocations() -> int:
    return int(memory_stats()["active_allocations"])


class _FakeRuntime:
    """Synthetic device runtime that records every operation in order."""

    def __init__(self) -> None:
        self.next_ptr = _RUNTIME_BASE + next(_RUNTIME_ORDINAL) * _RUNTIME_STRIDE
        self.log: list[tuple[object, ...]] = []
        self.freed: list[int] = []
        self.memcpy_async_calls: list[tuple[int, int, int, object, int]] = []
        self.device_synchronize_calls = 0
        self.stream_synchronize_calls: list[int] = []
        self.fail_memcpy_on: int | None = None
        self.fail_synchronize = False
        self.fail_synchronize_streams: set[int] = set()

    def malloc(self, nbytes: int) -> int:
        ptr = self.next_ptr
        self.next_ptr += max(4096, int(nbytes))
        self.log.append(("malloc", ptr))
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(int(ptr))
        self.log.append(("free", int(ptr)))

    def memcpy_async(
        self,
        dst: int,
        src: int,
        nbytes: int,
        kind: object,
        stream: int,
    ) -> None:
        self.memcpy_async_calls.append(
            (int(dst), int(src), int(nbytes), kind, int(stream))
        )
        self.log.append(("memcpy", int(dst), int(src)))
        if self.fail_memcpy_on is not None and len(self.memcpy_async_calls) == int(
            self.fail_memcpy_on
        ):
            raise RuntimeError("injected memcpy failure")

    def device_synchronize(self) -> None:
        self.device_synchronize_calls += 1
        self.log.append(("sync", "device"))
        if self.fail_synchronize:
            raise RuntimeError("injected synchronize failure")

    def stream_synchronize(self, stream: int) -> None:
        self.stream_synchronize_calls.append(int(stream))
        self.log.append(("sync", "stream", int(stream)))
        if self.fail_synchronize or int(stream) in self.fail_synchronize_streams:
            raise RuntimeError("injected synchronize failure")


class _RecordingPool(_GGUFPrefixSnapshotArenaPool):
    """Pool that records acquired arenas and every release."""

    def __init__(self, runtime: _FakeRuntime, *, max_retained: int = 4) -> None:
        super().__init__(runtime, max_retained=max_retained)
        self.arenas = []
        self.releases = []

    def acquire(self, nbytes: int):
        arena = super().acquire(nbytes)
        self.arenas.append(arena)
        return arena

    def release(self, arena) -> None:
        self.releases.append(arena)
        self.runtime.log.append(("pool.release", int(arena.owner.ptr)))
        super().release(arena)


def _state(nbytes: int, index: int) -> DeviceBuffer:
    return DeviceBuffer(ptr=_STATE_BASE + index * _STATE_STRIDE, nbytes=int(nbytes))


def _capture(
    runtime: _FakeRuntime,
    pool: _GGUFPrefixSnapshotArenaPool,
    *,
    conv_states,
    recurrent_states,
    stream: int = 0,
) -> Qwen35GGUFPrefixStateSnapshot:
    return Qwen35GGUFPrefixStateSnapshot.capture(
        runtime=runtime,
        runner=object(),
        kv_storage_dtype=DType.BF16,
        kv_storage_layout="uniform",
        position=512,
        block_ids=(4, 9),
        backing=object(),
        layer_conv_states=conv_states,
        layer_recurrent_states=recurrent_states,
        arena_pool=pool,
        stream=stream,
    )


def _two_layer_state():
    return (
        (_state(1000, 0), _state(3000, 2)),
        (_state(2000, 1), _state(4000, 3)),
    )


def test_capture_copies_paired_state_into_one_pooled_arena() -> None:
    runtime = _FakeRuntime()
    pool = _RecordingPool(runtime)
    runner = object()
    backing = object()
    conv_states, recurrent_states = _two_layer_state()

    snapshot = Qwen35GGUFPrefixStateSnapshot.capture(
        runtime=runtime,
        runner=runner,
        kv_storage_dtype=DType.BF16,
        kv_storage_layout="uniform",
        position=512,
        block_ids=(4, 9),
        backing=backing,
        layer_conv_states=conv_states,
        layer_recurrent_states=recurrent_states,
        arena_pool=pool,
    )

    arena = pool.arenas[0]
    assert snapshot.arena is arena
    assert snapshot.arena_pool is pool
    assert pool.stats()["arena_allocations"] == 1
    assert arena.capacity_bytes == 16384
    assert snapshot.closed is False
    assert snapshot.runtime is runtime
    assert snapshot.runner is runner
    assert snapshot.backing is backing
    assert snapshot.kv_storage_dtype == DType.BF16
    assert snapshot.kv_storage_layout == "uniform"
    assert snapshot.position == 512
    assert snapshot.block_ids == (4, 9)
    assert snapshot.nbytes == 10000

    expected_calls = []
    for conv_state, recurrent_state, conv_view, recurrent_view in zip(
        conv_states,
        recurrent_states,
        snapshot.layer_conv_states,
        snapshot.layer_recurrent_states,
    ):
        assert conv_view is not None and recurrent_view is not None
        assert arena.owns(conv_view) and arena.owns(recurrent_view)
        expected_calls.append(
            (
                conv_view.ptr,
                conv_state.ptr,
                conv_state.nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                0,
            )
        )
        expected_calls.append(
            (
                recurrent_view.ptr,
                recurrent_state.ptr,
                recurrent_state.nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                0,
            )
        )
    assert runtime.memcpy_async_calls == expected_calls
    assert runtime.device_synchronize_calls == 1
    assert runtime.stream_synchronize_calls == []
    assert runtime.freed == []

    snapshot.close()

    assert snapshot.closed is True
    assert pool.releases == [arena]
    assert arena.closed is False
    assert pool.stats()["retained_arenas"] == 1
    assert runtime.freed == [], "an arena-backed snapshot must not free its views"

    snapshot.close()
    assert pool.releases == [arena], "closing a snapshot twice released its arena twice"

    pool.close()
    assert arena.closed is True
    assert runtime.freed == [arena.owner.ptr]


def test_capture_reuses_the_pooled_arena_after_close() -> None:
    runtime = _FakeRuntime()
    pool = _RecordingPool(runtime)
    conv_states, recurrent_states = _two_layer_state()

    first = _capture(runtime, pool, conv_states=conv_states, recurrent_states=recurrent_states)
    first.close()
    second = _capture(runtime, pool, conv_states=conv_states, recurrent_states=recurrent_states)

    assert second.arena is first.arena
    assert pool.stats() == {
        "acquires": 2,
        "arena_allocations": 1,
        "retained_arenas": 0,
        "quarantined_arenas": 0,
    }
    assert [entry for entry in runtime.log if entry[0] == "malloc"] == [
        ("malloc", first.arena.owner.ptr)
    ], "reusing a pooled arena must not allocate"

    second.close()
    pool.close()
    assert runtime.freed == [first.arena.owner.ptr]


def test_capture_partial_state_mismatch_synchronizes_before_releasing_arena() -> None:
    runtime = _FakeRuntime()
    pool = _RecordingPool(runtime)
    conv_states, recurrent_states = _two_layer_state()

    with pytest.raises(ValueError, match="partial linear state at layer 1"):
        _capture(
            runtime,
            pool,
            conv_states=conv_states,
            recurrent_states=(recurrent_states[0], None),
        )

    arena = pool.arenas[0]
    assert len(runtime.memcpy_async_calls) == 2, (
        "only the paired layer that preceded the mismatch may be copied"
    )
    assert runtime.device_synchronize_calls == 1, (
        "submitted copies must be retired before the arena is released"
    )
    assert runtime.log.index(("sync", "device")) < runtime.log.index(
        ("pool.release", arena.owner.ptr)
    )
    assert pool.releases == [arena]
    assert arena.closed is False
    assert pool.stats()["retained_arenas"] == 1
    assert runtime.freed == []

    pool.close()
    assert runtime.freed == [arena.owner.ptr]


def test_capture_partial_state_mismatch_before_any_copy_releases_arena_directly() -> None:
    runtime = _FakeRuntime()
    pool = _RecordingPool(runtime)
    conv_states, recurrent_states = _two_layer_state()

    with pytest.raises(ValueError, match="partial linear state at layer 0"):
        _capture(
            runtime,
            pool,
            conv_states=(conv_states[0], None),
            recurrent_states=(None, recurrent_states[1]),
        )

    arena = pool.arenas[0]
    assert runtime.memcpy_async_calls == []
    assert runtime.device_synchronize_calls == 0, (
        "an arena nothing was copied into needs no retirement sync"
    )
    assert pool.releases == [arena]
    assert pool.stats()["retained_arenas"] == 1
    assert runtime.freed == []

    pool.close()
    assert runtime.freed == [arena.owner.ptr]


def test_capture_memcpy_failure_synchronizes_before_releasing_arena() -> None:
    runtime = _FakeRuntime()
    pool = _RecordingPool(runtime)
    conv_states, recurrent_states = _two_layer_state()
    runtime.fail_memcpy_on = 3

    with pytest.raises(RuntimeError, match="injected memcpy failure"):
        _capture(
            runtime, pool, conv_states=conv_states, recurrent_states=recurrent_states
        )

    arena = pool.arenas[0]
    assert len(runtime.memcpy_async_calls) == 3
    assert runtime.device_synchronize_calls == 1
    assert runtime.log.index(("sync", "device")) < runtime.log.index(
        ("pool.release", arena.owner.ptr)
    )
    assert pool.releases == [arena]
    assert pool.stats()["retained_arenas"] == 1
    assert runtime.freed == []

    pool.close()
    assert runtime.freed == [arena.owner.ptr]


def test_capture_first_copy_failure_still_synchronizes_before_release() -> None:
    """A memcpy that raises on its first call may already have enqueued work."""

    runtime = _FakeRuntime()
    pool = _RecordingPool(runtime)
    conv_states, recurrent_states = _two_layer_state()
    runtime.fail_memcpy_on = 1

    with pytest.raises(RuntimeError, match="injected memcpy failure"):
        _capture(
            runtime, pool, conv_states=conv_states, recurrent_states=recurrent_states
        )

    arena = pool.arenas[0]
    assert len(runtime.memcpy_async_calls) == 1
    assert runtime.device_synchronize_calls == 1, (
        "a first-call copy failure must still retire the stream before release"
    )
    assert runtime.log.index(("sync", "device")) < runtime.log.index(
        ("pool.release", arena.owner.ptr)
    )
    assert pool.releases == [arena]
    assert pool.stats()["quarantined_arenas"] == 0
    assert pool.stats()["retained_arenas"] == 1
    assert runtime.freed == []

    pool.close()
    assert runtime.freed == [arena.owner.ptr]


def test_capture_sync_failure_quarantines_arena_until_pool_close_retires_it() -> None:
    """A failed retirement sync quarantines the arena; a later close frees it once."""

    runtime = _FakeRuntime()
    pool = _RecordingPool(runtime)
    conv_states, recurrent_states = _two_layer_state()
    runtime.fail_memcpy_on = 3
    runtime.fail_synchronize = True
    baseline = _live_allocations()

    with pytest.raises(RuntimeError, match="injected memcpy failure") as excinfo:
        _capture(
            runtime, pool, conv_states=conv_states, recurrent_states=recurrent_states
        )

    arena = pool.arenas[0]
    assert pool.releases == [], (
        "the arena must not return to the pool while queued copies may still read it"
    )
    assert pool.stats()["quarantined_arenas"] == 1
    assert pool.stats()["retained_arenas"] == 0
    assert arena.closed is False
    assert runtime.freed == [], "the arena owner must not be freed"
    assert _live_allocations() == baseline + 1, (
        "the quarantined arena must still own its allocation"
    )
    notes = getattr(excinfo.value, "__notes__", ())
    assert any("quarantined" in note for note in notes), (
        "the capture failure must record that the arena was quarantined"
    )
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "injected synchronize failure" in str(excinfo.value.__cause__)

    runtime.fail_synchronize = False
    pool.close()

    assert pool.stats()["quarantined_arenas"] == 0
    assert arena.closed is True
    assert runtime.freed == [arena.owner.ptr]
    assert _live_allocations() == baseline

    pool.close()
    assert runtime.freed == [arena.owner.ptr]
    assert _live_allocations() == baseline


def test_capture_first_copy_failure_with_sync_failure_quarantines_arena() -> None:
    """The first copy may enqueue work before raising, so it too is quarantined."""

    runtime = _FakeRuntime()
    pool = _RecordingPool(runtime)
    conv_states, recurrent_states = _two_layer_state()
    runtime.fail_memcpy_on = 1
    runtime.fail_synchronize = True
    baseline = _live_allocations()

    with pytest.raises(RuntimeError, match="injected memcpy failure") as excinfo:
        _capture(
            runtime, pool, conv_states=conv_states, recurrent_states=recurrent_states
        )

    arena = pool.arenas[0]
    assert runtime.device_synchronize_calls == 1, (
        "the first-call failure path must attempt a retirement sync"
    )
    assert pool.stats()["quarantined_arenas"] == 1
    assert pool.releases == []
    assert arena.closed is False
    assert runtime.freed == []
    assert _live_allocations() == baseline + 1
    assert any("quarantined" in note for note in getattr(excinfo.value, "__notes__", ()))

    runtime.fail_synchronize = False
    pool.close()

    assert arena.closed is True
    assert runtime.freed == [arena.owner.ptr]
    assert _live_allocations() == baseline


def test_pool_close_frees_no_quarantined_arena_when_any_stream_sync_fails() -> None:
    """Close is all-or-nothing: one unretired stream blocks every quarantine free."""

    runtime = _FakeRuntime()
    pool = _RecordingPool(runtime)
    conv_states, recurrent_states = _two_layer_state()
    runtime.fail_synchronize = True
    for stream in (3, 5):
        runtime.fail_memcpy_on = len(runtime.memcpy_async_calls) + 1
        with pytest.raises(RuntimeError, match="injected memcpy failure"):
            _capture(
                runtime,
                pool,
                conv_states=conv_states,
                recurrent_states=recurrent_states,
                stream=stream,
            )

    arenas = list(pool.arenas)
    assert len(arenas) == 2
    assert pool.stats()["arena_allocations"] == 2
    assert pool.stats()["quarantined_arenas"] == 2
    assert runtime.stream_synchronize_calls == [3, 5]
    live_before = _live_allocations()

    runtime.fail_synchronize = False
    runtime.fail_synchronize_streams = {5}
    with pytest.raises(RuntimeError, match="injected synchronize failure"):
        pool.close()

    assert runtime.stream_synchronize_calls == [3, 5, 3, 5]
    assert all(not arena.closed for arena in arenas), (
        "no quarantined arena may be freed before every capture stream retires"
    )
    assert runtime.freed == []
    assert pool.stats()["quarantined_arenas"] == 2
    assert pool.closed is False
    assert _live_allocations() == live_before

    runtime.fail_synchronize_streams = set()
    pool.close()

    assert runtime.stream_synchronize_calls == [3, 5, 3, 5, 3, 5]
    assert all(arena.closed for arena in arenas)
    assert runtime.freed == [arena.owner.ptr for arena in arenas]
    assert pool.stats()["quarantined_arenas"] == 0
    assert _live_allocations() == live_before - 2

    pool.close()
    assert runtime.freed == [arena.owner.ptr for arena in arenas]


def test_pool_close_retry_retires_quarantine_then_allocates_nothing() -> None:
    """A close whose sync fails frees nothing and leaves the pool retryable."""

    runtime = _FakeRuntime()
    pool = _RecordingPool(runtime)
    conv_states, recurrent_states = _two_layer_state()
    runtime.fail_memcpy_on = 1
    runtime.fail_synchronize = True

    with pytest.raises(RuntimeError, match="injected memcpy failure"):
        _capture(
            runtime,
            pool,
            conv_states=conv_states,
            recurrent_states=recurrent_states,
            stream=7,
        )

    arena = pool.arenas[0]
    live_before = _live_allocations()
    assert pool.stats()["quarantined_arenas"] == 1
    assert runtime.stream_synchronize_calls == [7]
    allocations_before = [entry for entry in runtime.log if entry[0] == "malloc"]

    with pytest.raises(RuntimeError, match="injected synchronize failure"):
        pool.close()

    assert pool.stats()["quarantined_arenas"] == 1, (
        "a failed close must keep the quarantine for a retry"
    )
    assert arena.closed is False
    assert runtime.freed == [], "a failed close must free nothing"
    assert pool.closed is False, "a failed close must leave the pool retryable"
    assert _live_allocations() == live_before
    assert runtime.stream_synchronize_calls == [7, 7], (
        "close must synchronize the quarantined capture stream"
    )

    pool.quarantine(arena, stream=7)
    assert pool.stats()["quarantined_arenas"] == 1
    with pytest.raises(RuntimeError, match="cannot release a quarantined"):
        pool.release(arena)
    assert not arena.closed
    runtime.fail_synchronize = False
    pool.close()

    assert runtime.stream_synchronize_calls == [7, 7, 7]
    assert pool.stats()["quarantined_arenas"] == 0
    assert arena.closed is True
    assert runtime.freed == [arena.owner.ptr]
    assert pool.closed is True
    assert _live_allocations() == live_before - 1
    assert [entry for entry in runtime.log if entry[0] == "malloc"] == allocations_before, (
        "retiring a quarantine must not allocate"
    )

    pool.close()
    assert runtime.freed == [arena.owner.ptr]
    assert runtime.stream_synchronize_calls == [7, 7, 7]
    assert _live_allocations() == live_before - 1


def test_pool_can_retire_late_quarantine_after_close():
    runtime = _FakeRuntime()
    pool = _RecordingPool(runtime)
    baseline = _live_allocations()
    arena = pool.acquire(4096)
    pool.close()
    pool.quarantine(arena, stream=3)
    pool.close()
    assert runtime.stream_synchronize_calls == [3]
    assert arena.closed
    assert _live_allocations() == baseline


def test_capture_allocation_failure_leaves_pool_reusable(monkeypatch):
    runtime = _FakeRuntime()
    pool = _RecordingPool(runtime)
    conv, recurrent = _two_layer_state()
    baseline = _live_allocations()
    allocate = runtime.malloc
    def fail(_nbytes):
        raise RuntimeError("injected allocation failure")
    monkeypatch.setattr(runtime, "malloc", fail)
    with pytest.raises(RuntimeError, match="injected allocation failure"):
        _capture(runtime, pool, conv_states=conv, recurrent_states=recurrent)
    assert _live_allocations() == baseline
    assert not pool.arenas
    assert not pool.closed
    monkeypatch.setattr(runtime, "malloc", allocate)
    snapshot = _capture(runtime, pool, conv_states=conv, recurrent_states=recurrent)
    snapshot.close()
    pool.close()
    assert _live_allocations() == baseline


def test_capture_synchronizes_the_requested_stream() -> None:
    runtime = _FakeRuntime()
    pool = _RecordingPool(runtime)
    conv_states, recurrent_states = _two_layer_state()

    snapshot = _capture(
        runtime,
        pool,
        conv_states=conv_states,
        recurrent_states=recurrent_states,
        stream=7,
    )

    assert runtime.stream_synchronize_calls == [7]
    assert runtime.device_synchronize_calls == 0
    assert [call[4] for call in runtime.memcpy_async_calls] == [7, 7, 7, 7]

    snapshot.close()
    pool.close()


def test_snapshot_outlives_its_pool_and_frees_the_arena_exactly_once() -> None:
    runtime = _FakeRuntime()
    pool = _RecordingPool(runtime)
    conv_states, recurrent_states = _two_layer_state()

    snapshot = _capture(
        runtime, pool, conv_states=conv_states, recurrent_states=recurrent_states
    )
    arena = snapshot.arena

    pool.close()

    assert arena.closed is False, "an outstanding arena is not in the pool's free list"
    assert runtime.freed == []
    with pytest.raises(RuntimeError, match="pool is closed"):
        pool.acquire(4096)

    snapshot.close()

    assert snapshot.closed is True
    assert arena.closed is True
    assert runtime.freed == [arena.owner.ptr]

    snapshot.close()
    assert runtime.freed == [arena.owner.ptr], "the arena owner was freed twice"


def test_unpooled_snapshot_close_frees_each_view_exactly_once() -> None:
    runtime = _FakeRuntime()
    conv = DeviceBuffer(ptr=_STATE_BASE + 0x8000, nbytes=64)
    recurrent = DeviceBuffer(ptr=_STATE_BASE + 0x9000, nbytes=128)
    snapshot = Qwen35GGUFPrefixStateSnapshot(
        runtime=runtime,
        runner=object(),
        kv_storage_dtype=DType.FP16,
        kv_storage_layout="uniform",
        position=256,
        block_ids=(1,),
        backing=object(),
        layer_conv_states=(conv, None),
        layer_recurrent_states=(recurrent, None),
    )

    assert snapshot.nbytes == 192
    snapshot.close()

    assert runtime.freed == [recurrent.ptr, conv.ptr]
    assert snapshot.closed is True

    snapshot.close()
    assert runtime.freed == [recurrent.ptr, conv.ptr]
