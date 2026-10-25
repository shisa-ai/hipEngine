"""Stage timing for Qwen hybrid GGUF prefill, verify, and decode.

This module owns the two stage recorders and the two sync-stage helpers that
were previously defined next to the runner:

* :func:`_add_sync_stage_timing` - accumulate one named millisecond sample.
* :func:`_mark_sync_stage` - device-synchronize and charge elapsed host time.
* :class:`_HipEventStageRecorder` - HIP-event intervals for queued work, with
  no per-stage synchronize.
* :class:`_HipWallClockStageRecorder` - device steady wall-clock boundaries on
  one stream.

Both recorders keep their exact timing contract. The event recorder resolves
its intervals and then closes itself in a ``finally``; the wall-clock recorder
holds a device buffer until the caller closes it. Aliased marks accumulate the
same interval under every name, a negative event elapsed time is booked into
the negative-accounting keys and charged as zero, and wall-clock ticks convert
to milliseconds through the device rate in kilohertz.

Nothing here reads a runner, a session, or a module-global proxy. The HIP and
memory primitives are imported directly, so a ``monkeypatch.setattr`` on this
module intercepts the call the recorder issues. The runner re-exports these
same objects, so existing ``qwen35_gguf_runner.<name>`` references keep
resolving to the identical class or function.
"""

from __future__ import annotations

import time

import numpy as np

from hipengine.core.hip import HipRuntime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.runtime import (
    wall_clock_mark_u64,
    wall_clock_rate_khz,
)

__all__ = [
    "_HipEventStageRecorder",
    "_HipWallClockStageRecorder",
    "_add_sync_stage_timing",
    "_mark_sync_stage",
]


def _add_sync_stage_timing(
    timings: dict[str, float] | None,
    name: str,
    ms: float,
) -> None:
    if timings is None:
        return
    if ms < 0.0:
        raise RuntimeError(f"negative GGUF sync stage timing for {name}: {ms}")
    timings[name] = timings.get(name, 0.0) + float(ms)


def _mark_sync_stage(
    runtime: HipRuntime,
    timings: dict[str, float] | None,
    enabled: bool,
    name: str,
    t0: float,
) -> float:
    if not enabled or timings is None:
        return t0
    runtime.device_synchronize()
    _add_sync_stage_timing(timings, name, (time.perf_counter() - t0) * 1000)
    return time.perf_counter()


class _HipEventStageRecorder:
    """HIP-event stage timing for queued work without per-stage synchronizes."""

    def __init__(self, runtime: HipRuntime, *, enabled: bool, stream: int = 0) -> None:
        self.runtime = runtime
        self.enabled = bool(enabled)
        self.stream = int(stream)
        self._last_event: int | None = None
        self._events: list[int] = []
        self._intervals: list[tuple[tuple[str, ...], int, int]] = []

    def start(self) -> None:
        if not self.enabled:
            return
        self._last_event = self._new_event()
        self.runtime.event_record(self._last_event, self.stream)

    def mark(self, name: str, *aliases: str) -> None:
        if not self.enabled:
            return
        if self._last_event is None:
            self.start()
        assert self._last_event is not None
        event = self._new_event()
        self.runtime.event_record(event, self.stream)
        names = (str(name), *(str(alias) for alias in aliases))
        self._intervals.append((names, self._last_event, event))
        self._last_event = event

    def resolve_into(self, timings: dict[str, float] | None) -> None:
        if not self.enabled:
            return
        try:
            for names, start, stop in self._intervals:
                self.runtime.event_synchronize(stop)
                ms = self.runtime.event_elapsed_time_ms(start, stop)
                if ms < 0.0:
                    _add_sync_stage_timing(timings, f"{names[0]}_negative_event_elapsed", -ms)
                    _add_sync_stage_timing(timings, "packed_verify_gpu_negative_event_elapsed", -ms)
                    ms = 0.0
                for name in names:
                    _add_sync_stage_timing(timings, name, ms)
        finally:
            self.close()

    def close(self) -> None:
        # Best-effort teardown: one failing destroy must not strand the handles
        # after it. A handle whose destroy failed stays tracked so an explicit
        # retry can still release it, and the intervals are dropped either way
        # so a later resolve cannot report stale timing. The first failure
        # propagates.
        failure: BaseException | None = None
        retained: list[int] = []
        while self._events:
            event = self._events.pop()
            try:
                self.runtime.event_destroy(event)
            except BaseException as exc:
                if failure is None:
                    failure = exc
                retained.append(event)
        self._events.extend(reversed(retained))
        self._last_event = None
        self._intervals.clear()
        if failure is not None:
            raise failure

    def _new_event(self) -> int:
        event = self.runtime.event_create()
        self._events.append(event)
        return event


class _HipWallClockStageRecorder:
    """Record same-stream stage boundaries with the device steady wall clock."""

    def __init__(
        self,
        runtime: HipRuntime,
        *,
        enabled: bool,
        stream: int = 0,
        library: object | None = None,
        capacity: int = 4096,
    ) -> None:
        if capacity < 2:
            raise ValueError("wall-clock stage recorder capacity must be at least two")
        self.runtime = runtime
        self.enabled = bool(enabled)
        self.stream = int(stream)
        self.library = library
        self.capacity = int(capacity)
        self._buffer: DeviceBuffer | None = None
        self._intervals: list[tuple[tuple[str, ...], int, int]] = []
        self._count = 0
        self._closed = False
        if self.enabled:
            self._buffer = malloc(self.capacity * np.dtype(np.uint64).itemsize, runtime=runtime)

    def _record_boundary(self) -> int:
        if self._buffer is None:
            raise RuntimeError("wall-clock stage recorder buffer is unavailable")
        if self._count >= self.capacity:
            raise RuntimeError(f"wall-clock stage recorder exceeded capacity {self.capacity}")
        index = self._count
        wall_clock_mark_u64(
            self._buffer.ptr,
            index,
            stream=self.stream,
            library=self.library,
            runtime=self.runtime,
        )
        self._count += 1
        return index

    def start(self) -> None:
        if self.enabled:
            self._record_boundary()

    def mark(self, name: str, *aliases: str) -> None:
        if not self.enabled:
            return
        if self._count == 0:
            raise RuntimeError("stage recorder must be started before marking")
        start_index = self._count - 1
        stop_index = self._record_boundary()
        names = (str(name), *(str(alias) for alias in aliases))
        self._intervals.append((names, start_index, stop_index))

    def resolve_into(self, timings: dict[str, float]) -> None:
        if not self.enabled or self._buffer is None:
            return
        ticks = np.empty(self._count, dtype=np.uint64)
        copy_device_to_host(
            host_array_ptr(ticks),
            self._buffer,
            nbytes=int(ticks.nbytes),
            runtime=self.runtime,
        )
        rate_khz = wall_clock_rate_khz(library=self.library, runtime=self.runtime)
        for names, start_index, stop_index in self._intervals:
            elapsed_ticks = int(ticks[stop_index]) - int(ticks[start_index])
            if elapsed_ticks < 0:
                raise RuntimeError(
                    "device wall clock moved backwards between stage markers "
                    f"{start_index} and {stop_index}"
                )
            elapsed_ms = float(elapsed_ticks) / float(rate_khz)
            for name in names:
                timings[name] = timings.get(name, 0.0) + elapsed_ms

    def close(self) -> None:
        if self._closed:
            return
        if self._buffer is not None:
            free(self._buffer, runtime=self.runtime)
            self._buffer = None
        self._intervals.clear()
        self._closed = True
