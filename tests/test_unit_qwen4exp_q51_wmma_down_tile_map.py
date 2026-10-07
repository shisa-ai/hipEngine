"""Unit tests for the F16 Q5_1 grouped down tile-map prerequisite.

The F16 Q5_1 grouped down owner consumes a 16-row WMMA tile map and requires a
positive multiple-of-16 host row count. Only the non-exact gate/up paths build
that map for the current invocation, so a down-only configuration (production
exact gate/up with the F16 down owner) would otherwise hand the kernel a stale
zero, and a preceding MMQ32 map carries 32-row geometry the F16 owner cannot
consume. ``_run_qwen4_exp_q51_wmma_down`` rebuilds the map from the current
invocation's compact expert starts, reads the count back, validates it, and only
then calls the owner.

These tests mock the kernel launches on CPU and assert launch/argument behavior:
map before readback before owner, the count the owner receives, and the failure
paths that must not call the owner.
"""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest


def _runner():
    from hipengine.runtime import qwen4_exp_runner

    return qwen4_exp_runner


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.tile_map_calls: list[dict] = []
        self.owner_calls: list[dict] = []
        self.sync_calls: list[int] = []


def _install(monkeypatch, recorder: _Recorder, *, total_rows: int) -> SimpleNamespace:
    module = _runner()

    def fake_tile_map(expert_start, wmma_start, tile_expert, wmma_total,
                      num_experts, *, tile_capacity=0, stream=0, runtime=None):
        recorder.events.append(("tile_map", num_experts, tile_capacity, stream))
        recorder.tile_map_calls.append(
            {
                "expert_start": expert_start,
                "wmma_start": wmma_start,
                "tile_expert": tile_expert,
                "wmma_total": wmma_total,
                "num_experts": num_experts,
                "tile_capacity": tile_capacity,
                "stream": stream,
            }
        )

    def fake_owner(*args, **kwargs):
        recorder.events.append(("owner", args, kwargs))
        recorder.owner_calls.append({"args": args, "kwargs": kwargs})

    def fake_sync(stream):
        recorder.events.append(("sync", stream))
        recorder.sync_calls.append(stream)

    def fake_copy(host_ptr, buffer, nbytes, *, runtime=None):
        recorder.events.append(("copy", buffer, nbytes))
        ctypes.cast(host_ptr, ctypes.POINTER(ctypes.c_int64))[0] = int(total_rows)

    runtime = SimpleNamespace(stream_synchronize=fake_sync)
    monkeypatch.setattr(module, "qwen35_moe_wmma_tile_map", fake_tile_map)
    monkeypatch.setattr(module, "copy_device_to_host", fake_copy)
    monkeypatch.setattr(module, "host_array_ptr", lambda array: int(array.ctypes.data))
    monkeypatch.setattr(
        module, "qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out",
        fake_owner,
    )
    return runtime


def _call(module, runtime, *, total_buffer=None, stream=77, tile_capacity=4096):
    return module._run_qwen4_exp_q51_wmma_down(
        input_ptr=11,
        expert_start_ptr=12,
        wmma_expert_start_ptr=13,
        tile_expert_ptr=14,
        weights_ptr=15,
        output_ptr=16,
        wmma_total_buffer=total_buffer
        if total_buffer is not None
        else SimpleNamespace(ptr=0xDEAD, nbytes=8),
        compact_rows=64,
        num_experts=512,
        in_features=640,
        out_features=2560,
        tile_capacity=tile_capacity,
        stream=stream,
        runtime=runtime,
    )


def test_q51_wmma_down_builds_current_map_then_passes_count(monkeypatch) -> None:
    """Map -> stream sync -> readback -> owner, with a valid multiple-of-16 count."""

    module = _runner()
    recorder = _Recorder()
    runtime = _install(monkeypatch, recorder, total_rows=512)

    rows = _call(module, runtime, stream=77)

    assert rows == 512
    kinds = [event[0] for event in recorder.events]
    assert kinds == ["tile_map", "sync", "copy", "owner"]

    # The map is built from the caller's current compact expert starts, not from
    # any prior invocation's state.
    tile = recorder.tile_map_calls[0]
    assert tile["expert_start"] == 12
    assert tile["wmma_start"] == 13
    assert tile["tile_expert"] == 14
    assert tile["num_experts"] == 512
    assert tile["tile_capacity"] == 4096
    assert tile["stream"] == 77
    # Readback and owner use the same buffer the map wrote.
    assert tile["wmma_total"] == 0xDEAD
    assert recorder.sync_calls == [77]

    owner = recorder.owner_calls[0]
    # Positional ABI: (..., compact_rows, num_experts, in_features, out_features,
    # wmma_total_rows) at indices 6-10.
    assert owner["args"][6:11] == (64, 512, 640, 2560, 512)
    assert owner["kwargs"]["stream"] == 77


def test_q51_wmma_down_rejects_zero_count_without_calling_owner(monkeypatch) -> None:
    module = _runner()
    recorder = _Recorder()
    runtime = _install(monkeypatch, recorder, total_rows=0)

    with pytest.raises(RuntimeError):
        _call(module, runtime)

    assert [event[0] for event in recorder.events] == ["tile_map", "sync", "copy"]
    assert recorder.owner_calls == []


def test_q51_wmma_down_rejects_non_multiple_of_16_without_calling_owner(
    monkeypatch,
) -> None:
    module = _runner()
    recorder = _Recorder()
    runtime = _install(monkeypatch, recorder, total_rows=40)

    with pytest.raises(RuntimeError):
        _call(module, runtime)

    assert recorder.owner_calls == []


def test_q51_wmma_down_rejects_count_over_capacity_without_calling_owner(
    monkeypatch,
) -> None:
    module = _runner()
    recorder = _Recorder()
    runtime = _install(monkeypatch, recorder, total_rows=4096 * 16 + 16)

    with pytest.raises(RuntimeError):
        _call(module, runtime, tile_capacity=4096)

    assert recorder.owner_calls == []


def test_q51_wmma_down_skips_sync_without_a_stream(monkeypatch) -> None:
    """Stream ordering: no explicit sync on the default (0) stream."""

    module = _runner()
    recorder = _Recorder()
    runtime = _install(monkeypatch, recorder, total_rows=16)

    rows = _call(module, runtime, stream=0)

    assert rows == 16
    assert [event[0] for event in recorder.events] == ["tile_map", "copy", "owner"]
    assert recorder.sync_calls == []
