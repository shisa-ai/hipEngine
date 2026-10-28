"""CPU contracts for the Qwen GGUF stage timing owners.

The four owners moved out of ``qwen35_gguf_runner`` keep the runner's timing
numerics: aliased marks accumulate under every name, a negative event elapsed
time lands in the negative-accounting keys and charges zero, wall-clock ticks
divide by the device rate in kilohertz, and a disabled recorder touches no
runtime resource. The teardown paths are exercised with injected failures:
resolve-time errors must still destroy every HIP event, a pre-release event
destroy failure must destroy the other handles while keeping the failed handle
tracked for an explicit retry and clearing the interval state, and a pre-release
free failure must leave the wall-clock buffer owned so the close can be retried.

Everything here runs on CPU against recording stand-ins for the module's HIP
and memory primitives.
"""

from __future__ import annotations

import ast
import ctypes
import inspect

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.core.memory import copy_device_to_host as real_copy_device_to_host
from hipengine.core.memory import free as real_free
from hipengine.core.memory import malloc as real_malloc
from hipengine.kernels.hip_gfx1100.runtime import (
    wall_clock_mark_u64 as real_wall_clock_mark_u64,
    wall_clock_rate_khz as real_wall_clock_rate_khz,
)
from hipengine.runtime import qwen35_gguf_timing as gguf_timing
from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime.qwen35_gguf_timing import (
    _HipEventStageRecorder,
    _HipWallClockStageRecorder,
    _add_sync_stage_timing,
    _mark_sync_stage,
)


class _FakeEventRuntime:
    """Records HIP event traffic and can inject failures by call index."""

    def __init__(self) -> None:
        self.next_event = 100
        self.records: list[tuple[int, int]] = []
        self.synced: list[int] = []
        self.destroyed: list[int] = []
        self.synchronize_failure_at: int | None = None
        self.elapsed_failure_at: int | None = None
        self.destroy_failure_for: int | None = None

    def event_create(self) -> int:
        event = self.next_event
        self.next_event += 1
        return event

    def event_record(self, event: int, stream: int = 0) -> None:
        self.records.append((int(event), int(stream)))

    def event_synchronize(self, event: int) -> None:
        if self.synchronize_failure_at is not None:
            if len(self.synced) == self.synchronize_failure_at:
                raise RuntimeError("event synchronize failed")
        self.synced.append(int(event))

    def event_elapsed_time_ms(self, start: int, stop: int) -> float:
        if self.elapsed_failure_at is not None:
            if len(self.synced) - 1 == self.elapsed_failure_at:
                raise RuntimeError("event elapsed failed")
        return float(int(stop) - int(start))

    def event_destroy(self, event: int) -> None:
        if event == self.destroy_failure_for:
            raise RuntimeError(f"event destroy failed for {event}")
        self.destroyed.append(int(event))


class _FakeClock:
    def __init__(self, values: list[float]) -> None:
        self._values = list(values)
        self.calls = 0

    def perf_counter(self) -> float:
        self.calls += 1
        if not self._values:
            raise AssertionError("perf_counter called more often than scripted")
        return self._values.pop(0)


class _FakeSyncRuntime:
    def __init__(self) -> None:
        self.syncs = 0

    def device_synchronize(self) -> None:
        self.syncs += 1


# --- sync-stage helpers ----------------------------------------------------


def test_add_sync_stage_timing_accumulates_and_ignores_missing_timings() -> None:
    _add_sync_stage_timing(None, "ignored", -1.0)
    _add_sync_stage_timing(None, "ignored", 2.0)

    timings: dict[str, float] = {"existing": 1.0}
    _add_sync_stage_timing(timings, "existing", 2.5)
    _add_sync_stage_timing(timings, "fresh", np.float32(0.25))

    assert timings == {"existing": 3.5, "fresh": 0.25}
    assert isinstance(timings["fresh"], float)


def test_add_sync_stage_timing_rejects_negative_samples() -> None:
    with pytest.raises(RuntimeError, match="negative GGUF sync stage timing for kv_write: -0.5"):
        _add_sync_stage_timing({}, "kv_write", -0.5)


def test_mark_sync_stage_returns_start_without_synchronizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeSyncRuntime()
    clock = _FakeClock([99.0])
    monkeypatch.setattr(gguf_timing, "time", clock)

    assert _mark_sync_stage(runtime, {}, False, "stage", 3.0) == 3.0
    assert _mark_sync_stage(runtime, None, True, "stage", 4.0) == 4.0
    assert runtime.syncs == 0
    assert clock.calls == 0


def test_mark_sync_stage_synchronizes_and_charges_elapsed_milliseconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeSyncRuntime()
    clock = _FakeClock([1.5, 1.75])
    monkeypatch.setattr(gguf_timing, "time", clock)
    timings = {"stage": 10.0}

    next_start = _mark_sync_stage(runtime, timings, True, "stage", 1.0)

    assert runtime.syncs == 1
    assert timings == {"stage": 510.0}
    assert next_start == 1.75
    assert clock.calls == 2


# --- HIP-event stage recorder ---------------------------------------------


@pytest.mark.parametrize("value,enabled", [(None, False), ("0", False), ("1", True)])
def test_packed_verifier_event_timing_is_explicit_opt_in(monkeypatch, value, enabled):
    name = "HIPENGINE_GGUF_PACKED_VERIFY_GPU_STAGE_TIMINGS"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    assert gguf_runner._gguf_packed_verify_gpu_stage_timings_enabled() is enabled


def test_event_recorder_accumulates_aliases_in_mark_order_and_closes() -> None:
    runtime = _FakeEventRuntime()
    recorder = _HipEventStageRecorder(runtime, enabled=True, stream=7)
    recorder.start()
    recorder.mark("first", "total")
    recorder.mark("second")
    timings: dict[str, float] = {}

    recorder.resolve_into(timings)

    assert timings == {"first": 1.0, "total": 1.0, "second": 1.0}
    assert runtime.records == [(100, 7), (101, 7), (102, 7)]
    assert runtime.synced == [101, 102]
    assert sorted(runtime.destroyed) == [100, 101, 102]
    assert recorder._events == []
    assert recorder._intervals == []
    assert recorder._last_event is None


def test_event_recorder_books_negative_elapsed_into_negative_keys() -> None:
    class NegativeElapsedRuntime(_FakeEventRuntime):
        def event_elapsed_time_ms(self, start: int, stop: int) -> float:
            return -0.25

    runtime = NegativeElapsedRuntime()
    recorder = _HipEventStageRecorder(runtime, enabled=True, stream=3)
    recorder.start()
    recorder.mark("bad_interval")
    timings: dict[str, float] = {}

    recorder.resolve_into(timings)

    assert timings == {
        "bad_interval_negative_event_elapsed": 0.25,
        "packed_verify_gpu_negative_event_elapsed": 0.25,
        "bad_interval": 0.0,
    }
    assert sorted(runtime.destroyed) == [100, 101]


def test_event_recorder_negative_elapsed_without_timings_stays_silent() -> None:
    class NegativeElapsedRuntime(_FakeEventRuntime):
        def event_elapsed_time_ms(self, start: int, stop: int) -> float:
            return -2.0

    runtime = NegativeElapsedRuntime()
    recorder = _HipEventStageRecorder(runtime, enabled=True)
    recorder.start()
    recorder.mark("bad_interval")

    recorder.resolve_into(None)

    assert sorted(runtime.destroyed) == [100, 101]


def test_event_recorder_disabled_creates_no_events() -> None:
    runtime = _FakeEventRuntime()
    recorder = _HipEventStageRecorder(runtime, enabled=False, stream=5)

    recorder.start()
    recorder.mark("first", "total")
    timings: dict[str, float] = {}
    recorder.resolve_into(timings)
    recorder.close()

    assert timings == {}
    assert runtime.records == []
    assert runtime.synced == []
    assert runtime.destroyed == []
    assert recorder._events == []


def test_event_recorder_mark_without_start_begins_at_first_event() -> None:
    runtime = _FakeEventRuntime()
    recorder = _HipEventStageRecorder(runtime, enabled=True, stream=2)

    recorder.mark("only")

    assert recorder._intervals == [(("only",), 100, 101)]
    assert runtime.records == [(100, 2), (101, 2)]
    recorder.close()
    assert sorted(runtime.destroyed) == [100, 101]


def test_event_recorder_resolve_destroys_events_when_synchronize_fails() -> None:
    runtime = _FakeEventRuntime()
    runtime.synchronize_failure_at = 1
    recorder = _HipEventStageRecorder(runtime, enabled=True, stream=4)
    recorder.start()
    recorder.mark("first")
    recorder.mark("second")
    timings: dict[str, float] = {}

    with pytest.raises(RuntimeError, match="event synchronize failed"):
        recorder.resolve_into(timings)

    assert timings == {"first": 1.0}
    assert sorted(runtime.destroyed) == [100, 101, 102]
    assert recorder._events == []
    assert recorder._last_event is None


def test_event_recorder_resolve_destroys_events_when_elapsed_fails() -> None:
    runtime = _FakeEventRuntime()
    runtime.elapsed_failure_at = 0
    recorder = _HipEventStageRecorder(runtime, enabled=True, stream=4)
    recorder.start()
    recorder.mark("first")
    timings: dict[str, float] = {}

    with pytest.raises(RuntimeError, match="event elapsed failed"):
        recorder.resolve_into(timings)

    assert timings == {}
    assert sorted(runtime.destroyed) == [100, 101]
    assert recorder._events == []


def test_event_recorder_close_retains_failed_handle_and_clears_timing() -> None:
    runtime = _FakeEventRuntime()
    runtime.destroy_failure_for = 102
    recorder = _HipEventStageRecorder(runtime, enabled=True, stream=1)
    recorder.start()
    recorder.mark("first")
    recorder.mark("second")

    with pytest.raises(RuntimeError, match="event destroy failed for 102"):
        recorder.close()

    # Best-effort teardown released the other handles; the failed handle is
    # still tracked so an explicit retry can release it. Timing state is gone,
    # so a later resolve cannot report stale intervals.
    assert runtime.destroyed == [101, 100]
    assert recorder._events == [102]
    assert recorder._intervals == []
    assert recorder._last_event is None

    runtime.destroy_failure_for = None
    timings: dict[str, float] = {}
    recorder.resolve_into(timings)

    assert timings == {}
    assert runtime.destroyed == [101, 100, 102]
    assert recorder._events == []
    assert recorder._intervals == []
    assert recorder._last_event is None

    recorder.close()
    assert runtime.destroyed == [101, 100, 102]


def test_event_recorder_close_retains_every_failed_handle_in_creation_order() -> None:
    runtime = _FakeEventRuntime()
    recorder = _HipEventStageRecorder(runtime, enabled=True, stream=1)
    recorder.start()
    recorder.mark("first")
    recorder.mark("second")
    failing = {101, 102}

    def failing_destroy(event: int) -> None:
        if event in failing:
            raise RuntimeError(f"event destroy failed for {event}")
        runtime.destroyed.append(int(event))

    runtime.event_destroy = failing_destroy
    with pytest.raises(RuntimeError, match="event destroy failed for 102"):
        recorder.close()

    assert runtime.destroyed == [100]
    assert recorder._events == [101, 102]
    assert recorder._intervals == []
    assert recorder._last_event is None

    failing.clear()
    recorder.close()

    assert runtime.destroyed == [100, 102, 101]
    assert recorder._events == []


# --- wall-clock stage recorder --------------------------------------------


def _install_wall_clock_primitives(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ticks: np.ndarray,
    launches: list[tuple[int, int, int]],
    freed: list[DeviceBuffer],
    rate_khz: int = 2,
    capacity: int = 4,
    buffer: DeviceBuffer | None = None,
) -> DeviceBuffer:
    owner = (
        buffer
        if buffer is not None
        else DeviceBuffer(ptr=0x1000, nbytes=capacity * np.dtype(np.uint64).itemsize)
    )

    def fake_malloc(nbytes: int, **kwargs: object) -> DeviceBuffer:
        assert nbytes == owner.nbytes
        return owner

    def fake_mark(ptr: int, index: int, **kwargs: object) -> None:
        launches.append((int(ptr), int(index), int(kwargs["stream"])))

    def fake_copy(host_ptr: int, _buffer: DeviceBuffer, *, nbytes: int, **kwargs: object) -> None:
        assert _buffer is owner
        ctypes.memmove(host_ptr, ticks.ctypes.data, nbytes)

    monkeypatch.setattr(gguf_timing, "malloc", fake_malloc)
    monkeypatch.setattr(gguf_timing, "free", lambda item, **kwargs: freed.append(item))
    monkeypatch.setattr(gguf_timing, "wall_clock_mark_u64", fake_mark)
    monkeypatch.setattr(gguf_timing, "wall_clock_rate_khz", lambda **kwargs: rate_khz)
    monkeypatch.setattr(gguf_timing, "copy_device_to_host", fake_copy)
    return owner


def test_wall_clock_recorder_converts_ticks_and_frees_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object()
    launches: list[tuple[int, int, int]] = []
    freed: list[DeviceBuffer] = []
    ticks = np.asarray([100, 112, 150], dtype=np.uint64)
    owner = _install_wall_clock_primitives(
        monkeypatch, ticks=ticks, launches=launches, freed=freed
    )

    recorder = _HipWallClockStageRecorder(runtime, enabled=True, stream=9, capacity=4)
    recorder.start()
    recorder.mark("first", "total")
    recorder.mark("second")
    timings: dict[str, float] = {}
    recorder.resolve_into(timings)
    recorder.close()
    recorder.close()

    assert launches == [(0x1000, 0, 9), (0x1000, 1, 9), (0x1000, 2, 9)]
    assert timings == {"first": 6.0, "total": 6.0, "second": 19.0}
    assert freed == [owner]
    assert recorder._buffer is None


def test_wall_clock_recorder_rejects_small_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocations: list[int] = []
    monkeypatch.setattr(gguf_timing, "malloc", lambda nbytes, **_: allocations.append(nbytes))

    with pytest.raises(ValueError, match="capacity must be at least two"):
        _HipWallClockStageRecorder(object(), enabled=True, capacity=1)

    assert allocations == []


def test_wall_clock_recorder_disabled_is_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    allocations: list[int] = []
    frees: list[DeviceBuffer] = []
    launches: list[tuple[int, int, int]] = []
    monkeypatch.setattr(gguf_timing, "malloc", lambda nbytes, **_: allocations.append(nbytes))
    monkeypatch.setattr(gguf_timing, "free", lambda item, **_: frees.append(item))
    monkeypatch.setattr(
        gguf_timing,
        "wall_clock_mark_u64",
        lambda ptr, index, **kwargs: launches.append((int(ptr), int(index), int(kwargs["stream"]))),
    )
    recorder = _HipWallClockStageRecorder(object(), enabled=False, stream=3)

    recorder.start()
    recorder.mark("first")
    timings: dict[str, float] = {}
    recorder.resolve_into(timings)
    recorder.close()

    assert allocations == []
    assert frees == []
    assert launches == []
    assert timings == {}
    assert recorder._buffer is None


def test_wall_clock_recorder_requires_start_before_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launches: list[tuple[int, int, int]] = []
    freed: list[DeviceBuffer] = []
    owner = _install_wall_clock_primitives(
        monkeypatch,
        ticks=np.asarray([100, 112], dtype=np.uint64),
        launches=launches,
        freed=freed,
    )
    recorder = _HipWallClockStageRecorder(object(), enabled=True, stream=0, capacity=4)

    with pytest.raises(RuntimeError, match="must be started before marking"):
        recorder.mark("first")

    assert launches == []
    recorder.close()
    assert freed == [owner]


def test_wall_clock_recorder_rejects_boundary_past_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launches: list[tuple[int, int, int]] = []
    freed: list[DeviceBuffer] = []
    owner = _install_wall_clock_primitives(
        monkeypatch,
        ticks=np.asarray([100, 112], dtype=np.uint64),
        launches=launches,
        freed=freed,
        capacity=2,
    )
    recorder = _HipWallClockStageRecorder(object(), enabled=True, stream=0, capacity=2)
    recorder.start()
    recorder.mark("first")

    with pytest.raises(RuntimeError, match="exceeded capacity 2"):
        recorder.mark("second")

    assert launches == [(0x1000, 0, 0), (0x1000, 1, 0)]
    assert recorder._count == 2
    recorder.close()
    assert freed == [owner]


def test_wall_clock_recorder_keeps_ownership_when_resolve_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launches: list[tuple[int, int, int]] = []
    freed: list[DeviceBuffer] = []
    owner = _install_wall_clock_primitives(
        monkeypatch,
        ticks=np.asarray([100, 112], dtype=np.uint64),
        launches=launches,
        freed=freed,
    )
    recorder = _HipWallClockStageRecorder(object(), enabled=True, stream=0, capacity=4)
    recorder.start()
    recorder.mark("first")

    def failing_copy(*args: object, **kwargs: object) -> None:
        raise RuntimeError("device to host copy failed")

    monkeypatch.setattr(gguf_timing, "copy_device_to_host", failing_copy)
    with pytest.raises(RuntimeError, match="device to host copy failed"):
        recorder.resolve_into({})

    assert recorder._buffer is owner
    assert freed == []
    recorder.close()
    assert freed == [owner]


def test_wall_clock_recorder_keeps_ownership_when_rate_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launches: list[tuple[int, int, int]] = []
    freed: list[DeviceBuffer] = []
    owner = _install_wall_clock_primitives(
        monkeypatch,
        ticks=np.asarray([100, 112], dtype=np.uint64),
        launches=launches,
        freed=freed,
    )
    recorder = _HipWallClockStageRecorder(object(), enabled=True, stream=0, capacity=4)
    recorder.start()
    recorder.mark("first")

    def failing_rate(**kwargs: object) -> int:
        raise RuntimeError("wall clock rate unavailable")

    monkeypatch.setattr(gguf_timing, "wall_clock_rate_khz", failing_rate)
    with pytest.raises(RuntimeError, match="wall clock rate unavailable"):
        recorder.resolve_into({})

    assert recorder._buffer is owner
    assert freed == []
    recorder.close()
    assert freed == [owner]


def test_wall_clock_recorder_rejects_backwards_device_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launches: list[tuple[int, int, int]] = []
    freed: list[DeviceBuffer] = []
    owner = _install_wall_clock_primitives(
        monkeypatch,
        ticks=np.asarray([150, 100], dtype=np.uint64),
        launches=launches,
        freed=freed,
    )
    recorder = _HipWallClockStageRecorder(object(), enabled=True, stream=0, capacity=4)
    recorder.start()
    recorder.mark("first")

    with pytest.raises(RuntimeError, match="device wall clock moved backwards"):
        recorder.resolve_into({})

    assert freed == []
    recorder.close()
    assert freed == [owner]


def test_wall_clock_recorder_close_retries_after_pre_release_free_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launches: list[tuple[int, int, int]] = []
    freed: list[DeviceBuffer] = []
    owner = _install_wall_clock_primitives(
        monkeypatch,
        ticks=np.asarray([100, 112], dtype=np.uint64),
        launches=launches,
        freed=freed,
    )
    recorder = _HipWallClockStageRecorder(object(), enabled=True, stream=0, capacity=4)
    recorder.start()
    attempts: list[DeviceBuffer] = []
    released: list[DeviceBuffer] = []
    fail_first = [True]

    def free_or_fail(item: DeviceBuffer, **kwargs: object) -> None:
        attempts.append(item)
        if fail_first[0]:
            fail_first[0] = False
            raise RuntimeError("device free failed before releasing")
        released.append(item)

    monkeypatch.setattr(gguf_timing, "free", free_or_fail)
    with pytest.raises(RuntimeError, match="device free failed before releasing"):
        recorder.close()

    # A failed free did not release the buffer, so the recorder keeps owning it
    # and the close stays retryable.
    assert recorder._buffer is owner
    assert recorder._closed is False
    assert released == []

    recorder.close()
    recorder.close()

    assert attempts == [owner, owner]
    assert released == [owner]
    assert recorder._buffer is None
    assert recorder._closed is True
    assert recorder._intervals == []


# --- ownership boundary ---------------------------------------------------


def test_timing_module_uses_direct_primitives_and_no_runner_imports() -> None:
    tree = ast.parse(inspect.getsource(gguf_timing))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)

    assert not [name for name in imported if "qwen35_gguf_runner" in name]
    assert not [name for name in imported if "session" in name]
    assert gguf_timing.malloc is real_malloc
    assert gguf_timing.free is real_free
    assert gguf_timing.copy_device_to_host is real_copy_device_to_host
    assert gguf_timing.wall_clock_mark_u64 is real_wall_clock_mark_u64
    assert gguf_timing.wall_clock_rate_khz is real_wall_clock_rate_khz
    assert gguf_timing.__all__ == [
        "_HipEventStageRecorder",
        "_HipWallClockStageRecorder",
        "_add_sync_stage_timing",
        "_mark_sync_stage",
    ]
