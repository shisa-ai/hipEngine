"""Real-HIP lifecycle coverage for opt-in packed-verifier stage events."""
from __future__ import annotations

from collections import Counter
import ctypes

import pytest

from hipengine.core.hip import HipRuntime
from hipengine.runtime.qwen35_gguf_timing import (
    _HipEventStageRecorder,
    _PackedVerifyGpuStageRecorderScope,
)


@pytest.fixture
def event_runtime():
    try:
        ctypes.CDLL("libamdhip64.so")
        runtime = HipRuntime.load()
        runtime.mem_get_info()
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"HIP runtime is not available: {exc}")
    yield runtime


@pytest.mark.parametrize("failure_stage", [None, "start", "body", "resolve"])
def test_packed_timing_scope_releases_real_events(event_runtime, monkeypatch, failure_stage):
    runtime = event_runtime
    created, destroyed = [], []
    create, destroy = runtime.event_create, runtime.event_destroy
    record, synchronize = runtime.event_record, runtime.event_synchronize
    stream = runtime.stream_create()

    def tracked_create(**kwargs):
        event = create(**kwargs)
        created.append(event)
        return event

    def tracked_destroy(event):
        destroy(event)
        destroyed.append(event)

    def fail(*args, **kwargs):
        raise RuntimeError(f"injected {failure_stage} failure")

    monkeypatch.setattr(runtime, "event_create", tracked_create)
    monkeypatch.setattr(runtime, "event_destroy", tracked_destroy)
    if failure_stage == "start":
        monkeypatch.setattr(runtime, "event_record", fail)
    elif failure_stage == "resolve":
        monkeypatch.setattr(runtime, "event_synchronize", fail)
    recorder = _HipEventStageRecorder(runtime, enabled=True, stream=stream)
    timings = {}

    def run():
        with _PackedVerifyGpuStageRecorderScope(recorder, timings):
            recorder.mark("packed_verify_gpu_real_hip", "stage_alias")
            if failure_stage == "body":
                fail()

    try:
        if failure_stage:
            with pytest.raises(RuntimeError, match=f"injected {failure_stage} failure"):
                run()
        else:
            run()
            assert timings["packed_verify_gpu_real_hip"] >= 0
            assert timings["stage_alias"] == timings["packed_verify_gpu_real_hip"]
        assert created
        assert Counter(created) == Counter(destroyed)
        assert not recorder._events
    finally:
        monkeypatch.setattr(runtime, "event_record", record)
        monkeypatch.setattr(runtime, "event_synchronize", synchronize)
        recorder.close()
        runtime.stream_synchronize(stream)
        runtime.stream_destroy(stream)
