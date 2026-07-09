#!/usr/bin/env python3
"""HIP event and host-wall timing for serial and independent microbenchmarks."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, NamedTuple


class HipTimingSamples(NamedTuple):
    gpu_sequence_us: list[float]
    host_sequence_us: list[float]


class HipSequenceTimer:
    """Measure one ordered stream or disjoint work spread over nonblocking streams."""

    def __init__(self, runtime: Any, timing_mode: str, independent_streams: int = 4):
        if timing_mode not in {"serial_latency", "independent_throughput"}:
            raise ValueError("invalid timing mode")
        if independent_streams <= 0:
            raise ValueError("independent_streams must be positive")
        self.runtime = runtime
        self.timing_mode = timing_mode
        self.coordinator = runtime.stream_create(nonblocking=True)
        if timing_mode == "serial_latency":
            self.workers = [self.coordinator]
        else:
            self.workers = [
                runtime.stream_create(nonblocking=True) for _ in range(independent_streams)
            ]
        self.start = runtime.event_create()
        self.stop = runtime.event_create()
        self.done = [runtime.event_create() for _ in self.workers]
        self._closed = False

    @property
    def stream_count(self) -> int:
        return len(self.workers)

    def __enter__(self) -> "HipSequenceTimer":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        for event in self.done:
            self.runtime.event_destroy(event)
        self.runtime.event_destroy(self.start)
        self.runtime.event_destroy(self.stop)
        for stream in self.workers:
            if stream != self.coordinator:
                self.runtime.stream_destroy(stream)
        self.runtime.stream_destroy(self.coordinator)
        self._closed = True

    def _pre_sample_sync(self) -> None:
        self.runtime.stream_synchronize(self.coordinator)
        for worker in self.workers:
            if worker != self.coordinator:
                self.runtime.stream_synchronize(worker)

    def measure(
        self,
        logical_iterations: int,
        samples: int,
        launch: Callable[[int, int], None],
    ) -> HipTimingSamples:
        if logical_iterations <= 0 or samples <= 0:
            raise ValueError("logical_iterations and samples must be positive")
        gpu_sequence_us: list[float] = []
        host_sequence_us: list[float] = []
        for _ in range(samples):
            self._pre_sample_sync()
            host_start = time.perf_counter_ns()
            self.runtime.event_record(self.start, self.coordinator)
            if self.timing_mode == "independent_throughput":
                for worker in self.workers:
                    self.runtime.stream_wait_event(worker, self.start)
            for rep in range(logical_iterations):
                launch(rep, self.workers[rep % len(self.workers)])
            if self.timing_mode == "independent_throughput":
                for worker, done in zip(self.workers, self.done, strict=True):
                    self.runtime.event_record(done, worker)
                    self.runtime.stream_wait_event(self.coordinator, done)
            self.runtime.event_record(self.stop, self.coordinator)
            self.runtime.event_synchronize(self.stop)
            host_stop = time.perf_counter_ns()
            gpu_sequence_us.append(
                self.runtime.event_elapsed_time_ms(self.start, self.stop) * 1000.0
            )
            host_sequence_us.append((host_stop - host_start) / 1000.0)
        return HipTimingSamples(gpu_sequence_us, host_sequence_us)

    def run_and_wait(
        self,
        logical_iterations: int,
        launch: Callable[[int, int], None],
    ) -> None:
        if logical_iterations <= 0:
            raise ValueError("logical_iterations must be positive")
        for rep in range(logical_iterations):
            launch(rep, self.workers[rep % len(self.workers)])
        for worker in self.workers:
            self.runtime.stream_synchronize(worker)
