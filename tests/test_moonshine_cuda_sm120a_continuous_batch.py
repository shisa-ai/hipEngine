"""Continuous Moonshine CUDA admission/reclaim/graph-cache gates."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from hipengine.runtime.moonshine_cuda_continuous import (
    ContinuousBatchBackpressureError,
    MoonshineCudaContinuousBatchRuntime,
)

_FIXTURE_DIR = Path(
    os.environ.get(
        "HIPENGINE_MOONSHINE_SIX_FIXTURE_DIR",
        "/home/lhl/moonshine-prod-inference/results/raw/moonshine-fixtures-six",
    )
)
_SNAPSHOT = Path(
    os.environ.get(
        "HIPENGINE_MOONSHINE_SNAPSHOT",
        "/data/huggingface/hub/models--shisa-ai--shisa-realtime-asr-0.92b/"
        "snapshots/cb0b524b74f6e0bfe6a8780b8dc9854ffa429c7d",
    )
)
_FIXTURES = (
    "audio-hai-fp16",
    "audio-konichiwa-fp16",
    "audio-konichiwa.ogenkidesuka-fp16",
    "audio-kumbawa-fp16",
    "audio-sosososo-fp16",
    "audio-sumimasen-fp16",
)


def _cuda_gate_enabled() -> bool:
    import ctypes

    if os.environ.get("HIPENGINE_RUN_CUDA_SM120A") != "1":
        return False
    if os.environ.get("HIPENGINE_CUDA_ARCH") != "sm_120a":
        return False
    try:
        ctypes.CDLL("libcudart.so.13")
    except OSError:
        return False
    return _SNAPSHOT.is_dir() and all(
        (_FIXTURE_DIR / f"{name}.npz").is_file()
        and (_FIXTURE_DIR / f"{name}.json").is_file()
        for name in _FIXTURES
    )


@dataclass
class _Spec:
    decoder_layers: int = 1
    decoder_kv_heads: int = 1
    head_dim: int = 1
    self_cache_capacity: int = 8
    vocab_size: int = 256


class _FakeRuntime:
    def __init__(self) -> None:
        self.capturing = False
        self.capture_callback = None
        self.graphs: dict[int, object] = {}
        self.execs: dict[int, int] = {}
        self.next_handle = 1
        self.destroyed_graphs: list[int] = []
        self.destroyed_execs: list[int] = []

    def stream_synchronize(self, _stream: int) -> None:
        return None

    def stream_begin_capture(self, _stream: int) -> None:
        assert not self.capturing
        self.capturing = True
        self.capture_callback = None

    def stream_end_capture(self, _stream: int) -> int:
        assert self.capturing
        self.capturing = False
        graph = self.next_handle
        self.next_handle += 1
        self.graphs[graph] = self.capture_callback
        return graph

    def graph_instantiate(self, graph: int) -> int:
        graph_exec = self.next_handle
        self.next_handle += 1
        self.execs[graph_exec] = graph
        return graph_exec

    def graph_launch(self, graph_exec: int, _stream: int) -> None:
        callback = self.graphs[self.execs[graph_exec]]
        assert callable(callback)
        callback()

    def graph_exec_destroy(self, graph_exec: int) -> None:
        self.destroyed_execs.append(graph_exec)
        self.execs.pop(graph_exec)

    def graph_destroy(self, graph: int) -> None:
        self.destroyed_graphs.append(graph)
        self.graphs.pop(graph)


class _FakeDecoder:
    def __init__(self, max_batch: int = 2) -> None:
        self.max_batch = max_batch
        self.encoder_frames = 1
        self.spec = _Spec()
        self.runtime = _FakeRuntime()
        self.stream = 7
        self.closed = False
        self.decoder_libraries = object()
        self.program_lengths = [0] * max_batch
        self.tokens = np.zeros(max_batch, dtype=np.int64)
        self.positions = np.zeros(max_batch, dtype=np.int64)
        self.mixed_state_history: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self.moves: list[tuple[int, int]] = []

    def load_cross_cache_row(self, row, keys, values, *, mask) -> None:
        assert len(keys) == len(values) == 1
        assert mask.shape == (1,)
        self.program_lengths[row] = int(keys[0].reshape(-1)[0])

    def move_batch_row(self, source: int, destination: int) -> None:
        self.program_lengths[destination] = self.program_lengths[source]
        self.moves.append((source, destination))

    def set_mixed_batch_decode_state(self, *, tokens, positions, active_batch) -> None:
        assert len(tokens) == len(positions) == active_batch
        self.tokens[:active_batch] = tokens
        self.positions[:active_batch] = positions
        self.mixed_state_history.append((tuple(tokens), tuple(positions)))

    def _compute(self, active_batch: int) -> None:
        for row in range(active_batch):
            position = int(self.positions[row])
            if position + 1 >= self.program_lengths[row]:
                self.tokens[row] = 2
            else:
                self.tokens[row] = 100 + position

    def _enqueue_batch_token_step(
        self, *, route_position, threads, stream, active_batch
    ) -> None:
        assert route_position == 7
        assert threads == 256
        assert stream == self.stream
        if self.runtime.capturing:
            self.runtime.capture_callback = lambda: self._compute(active_batch)
        else:
            self._compute(active_batch)

    def read_tokens(self, *, active_batch=None) -> np.ndarray:
        count = self.max_batch if active_batch is None else active_batch
        return self.tokens[:count].copy()

    def close(self) -> None:
        self.closed = True


def _request_arrays(length: int):
    values = np.array([[[length]]], dtype=np.float16)
    return [values], [values.copy()], np.ones(1, dtype=np.int32)


def test_continuous_scheduler_fifo_mixed_positions_reclaim_and_lru() -> None:
    decoder = _FakeDecoder(max_batch=2)
    scheduler = MoonshineCudaContinuousBatchRuntime(
        decoder,
        owns_decoder=True,
        max_pending=3,
        max_graphs=1,
    )
    for request_id, length in (("a", 1), ("b", 4), ("c", 2)):
        keys, values, mask = _request_arrays(length)
        scheduler.submit(request_id, keys, values, mask=mask)

    first = scheduler.step()
    assert first.admitted == ("a", "b")
    assert first.tokens == {"a": 2, "b": 100}
    assert first.completed == ("a",)
    assert first.active == ("b",)
    assert first.pending == ("c",)
    assert decoder.moves == [(1, 0)]

    second = scheduler.step()
    assert second.admitted == ("c",)
    assert second.tokens == {"b": 101, "c": 100}
    assert second.active == ("b", "c")
    assert decoder.mixed_state_history[-1][1] == (1, 0)

    third = scheduler.step()
    assert third.completed == ("c",)
    assert third.active == ("b",)
    fourth = scheduler.step()
    assert fourth.completed == ("b",)
    assert scheduler.idle is True
    assert scheduler.take_completed("a").tokens == (2,)
    assert scheduler.take_completed("b").tokens == (100, 101, 102, 2)
    assert scheduler.take_completed("c").tokens == (100, 2)

    contract = scheduler.graph_cache_contract()
    assert contract["max_graphs"] == 1
    assert contract["size"] <= 1
    assert contract["captures"] >= 2
    assert contract["evictions"] >= 1
    assert contract["replays"] == 4
    assert scheduler.scheduler_contract()["compactions"] >= 1
    scheduler.close()
    assert decoder.closed is True
    assert decoder.runtime.destroyed_graphs
    assert decoder.runtime.destroyed_execs


def test_continuous_scheduler_backpressure_duplicates_cancel_and_bounds() -> None:
    decoder = _FakeDecoder(max_batch=2)
    scheduler = MoonshineCudaContinuousBatchRuntime(
        decoder,
        max_pending=2,
        max_graphs=2,
    )
    keys, values, mask = _request_arrays(3)
    scheduler.submit("first", keys, values, mask=mask)
    with pytest.raises(ValueError, match="already exists"):
        scheduler.submit("first", keys, values, mask=mask)
    scheduler.submit("second", keys, values, mask=mask)
    with pytest.raises(ContinuousBatchBackpressureError):
        scheduler.submit("third", keys, values, mask=mask)
    assert scheduler.cancel("second") is True
    assert scheduler.cancel("missing") is False
    cancelled = scheduler.take_completed("second")
    assert cancelled.reason == "cancelled_pending"
    assert cancelled.tokens == ()
    scheduler.close()


def _load_fixture(name: str, frames: int) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, list[int]]:
    manifest = json.loads((_FIXTURE_DIR / f"{name}.json").read_text())
    real_frames = int(manifest["input"]["encoder_frames"])
    reference = [int(value) for value in manifest["decoder"]["token_ids"]]
    keys: list[np.ndarray] = []
    values: list[np.ndarray] = []
    with np.load(_FIXTURE_DIR / f"{name}.npz") as fixture:
        for layer in range(8):
            for target, kind in ((keys, "key"), (values, "value")):
                source = np.asarray(
                    fixture[f"cross.layer_{layer}.{kind}"], dtype=np.float16
                )
                padded = np.zeros((8, frames, 52), dtype=np.float16)
                padded[:, :real_frames, :] = source[0]
                target.append(padded)
    mask = np.zeros(frames, dtype=np.int32)
    mask[:real_frames] = 1
    return keys, values, mask, reference


@pytest.mark.skipif(
    not _cuda_gate_enabled(),
    reason="CUDA sm_120a gate or Moonshine fixtures are not available",
)
def test_continuous_scheduler_real_fixtures_dynamic_arrival_exact_to_eos() -> None:
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.core.device import Device
    from hipengine.loading.moonshine import load_moonshine_model
    from hipengine.runtime.moonshine_cuda_batch import MoonshineCudaBatchRuntime

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    frames = max(
        int(
            json.loads((_FIXTURE_DIR / f"{name}.json").read_text())["input"][
                "encoder_frames"
            ]
        )
        for name in _FIXTURES
    )
    loaded = load_moonshine_model(
        _SNAPSHOT, device=Device("cuda", 0), runtime=runtime
    )
    decoder = MoonshineCudaBatchRuntime(
        max_batch=4,
        encoder_frames=frames,
        loaded_model=loaded,
        owns_weights=False,
    )
    decoder.prepare_decoder_kernels()
    scheduler = MoonshineCudaContinuousBatchRuntime(
        decoder,
        owns_decoder=True,
        max_pending=8,
        max_graphs=3,
    )
    expected: dict[str, list[int]] = {}
    try:
        for name in _FIXTURES[:2]:
            keys, values, mask, reference = _load_fixture(name, frames)
            scheduler.submit(name, keys, values, mask=mask, seed_token_id=reference[0])
            expected[name] = reference[1 : reference.index(2, 1) + 1]
        scheduler.step()
        scheduler.step()
        for name in _FIXTURES[2:]:
            keys, values, mask, reference = _load_fixture(name, frames)
            scheduler.submit(name, keys, values, mask=mask, seed_token_id=reference[0])
            expected[name] = reference[1 : reference.index(2, 1) + 1]
        while not scheduler.idle:
            scheduler.step()

        for name in _FIXTURES:
            result = scheduler.take_completed(name)
            assert result.reason == "eos"
            assert list(result.tokens) == expected[name]
        contract = scheduler.scheduler_contract()
        assert contract["submitted"] == 6
        assert contract["completed"] == 6
        assert contract["admissions"] == 6
        assert contract["compactions"] > 0
        assert contract["maximum_active"] == 4
        graph = scheduler.graph_cache_contract()
        assert graph["size"] <= 3
        assert graph["replays"] > 0
        assert graph["topology"] == "uniform_t256_rederived"
    finally:
        scheduler.close()
        assert decoder.teardown_returned_to_baseline is True
        loaded.weights.free(runtime=runtime)
