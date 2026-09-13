from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "micro" / "hip_timing.py"
    spec = importlib.util.spec_from_file_location("micro_hip_timing", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeRuntime:
    def __init__(self):
        self.next_handle = 10
        self.calls: list[tuple] = []

    def _handle(self, kind: str) -> int:
        self.next_handle += 1
        self.calls.append((f"{kind}_create", self.next_handle))
        return self.next_handle

    def stream_create(self, *, nonblocking: bool = True) -> int:
        assert nonblocking
        return self._handle("stream")

    def stream_destroy(self, stream: int) -> None:
        self.calls.append(("stream_destroy", stream))

    def stream_synchronize(self, stream: int) -> None:
        self.calls.append(("stream_sync", stream))

    def stream_wait_event(self, stream: int, event: int) -> None:
        self.calls.append(("stream_wait", stream, event))

    def event_create(self) -> int:
        return self._handle("event")

    def event_destroy(self, event: int) -> None:
        self.calls.append(("event_destroy", event))

    def event_record(self, event: int, stream: int = 0) -> None:
        self.calls.append(("event_record", event, stream))

    def event_synchronize(self, event: int) -> None:
        self.calls.append(("event_sync", event))

    def event_elapsed_time_ms(self, start: int, stop: int) -> float:
        self.calls.append(("event_elapsed", start, stop))
        return 0.08


def test_serial_timer_uses_one_ordered_stream() -> None:
    module = _load_module()
    runtime = _FakeRuntime()
    launches: list[tuple[int, int]] = []

    with module.HipSequenceTimer(runtime, "serial_latency") as timer:
        result = timer.measure(4, 2, lambda rep, stream: launches.append((rep, stream)))
        assert timer.stream_count == 1

    assert launches == [(rep, 11) for _ in range(2) for rep in range(4)]
    assert result.gpu_sequence_us == [80.0, 80.0]
    assert not [call for call in runtime.calls if call[0] == "stream_wait"]


def test_independent_timer_fans_out_and_back_with_disjoint_rep_ids() -> None:
    module = _load_module()
    runtime = _FakeRuntime()
    launches: list[tuple[int, int]] = []

    with module.HipSequenceTimer(
        runtime, "independent_throughput", independent_streams=2
    ) as timer:
        timer.measure(4, 1, lambda rep, stream: launches.append((rep, stream)))

    assert launches == [(0, 12), (1, 13), (2, 12), (3, 13)]
    waits = [call for call in runtime.calls if call[0] == "stream_wait"]
    assert len(waits) == 4
    assert waits[0][1] == 12
    assert waits[1][1] == 13
    assert waits[2][1] == 11
    assert waits[3][1] == 11
