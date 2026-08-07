"""CPU regressions for the complete-ASR CUDA campaign driver."""

from __future__ import annotations

import ctypes
import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.benchmark_moonshine_cuda_complete import Route


class _FakeHidden:
    def __init__(self, ptr: int) -> None:
        self.ptr = ptr

    def data_ptr(self) -> int:
        return self.ptr


class _TraceRuntime:
    def __init__(self, trace: list[tuple[object, ...]]) -> None:
        self.trace = trace

    def stream_wait_event(self, stream: int, event: int) -> None:
        self.trace.append(("wait", stream, event))

    def stream_synchronize(self, stream: int) -> None:
        self.trace.append(("sync", stream))

    def event_destroy(self, event: int) -> None:
        self.trace.append(("event_destroy", event))


class _TraceDecoder:
    def __init__(self, trace: list[tuple[object, ...]]) -> None:
        self.trace = trace
        self.stream = 0xD00D

    def set_encoder_state_from_device(self, **kwargs) -> None:
        self.trace.append(("set", kwargs))

    def precompute_cross_kv(self, **kwargs) -> None:
        self.trace.append(("precompute", kwargs))

    def close(self) -> None:
        self.trace.append(("decoder_close",))


def _bare_route() -> Route:
    route = Route.__new__(Route)
    route._torch_hidden = None
    route._torch_hidden_ptr = 0
    return route


def test_torch_encoder_output_is_retained_across_external_stream_handoff() -> None:
    route = _bare_route()
    hidden = _FakeHidden(0xABC000)
    reference = weakref.ref(hidden)

    route._retain_torch_hidden(hidden)
    del hidden
    gc.collect()
    # Integer pointers do not own caching-allocator storage; the Route must.
    assert reference() is route._torch_hidden
    assert route._torch_hidden_ptr == 0xABC000

    # Replacement stays bounded to one producer output and releases the old one.
    replacement = _FakeHidden(0xDEF000)
    route._retain_torch_hidden(replacement)
    gc.collect()
    assert reference() is None
    assert route._torch_hidden is replacement
    assert route._torch_hidden_ptr == 0xDEF000


def test_torch_encoder_event_wait_precedes_external_d2d() -> None:
    trace: list[tuple[object, ...]] = []
    route = _bare_route()
    hidden = _FakeHidden(0xABC000)
    route._retain_torch_hidden(hidden)
    route.mode = "torch-encoder"
    route._producer_event = 0xE001
    route._torch_mask_ptr = 0xABC800
    route.cuda_runtime = _TraceRuntime(trace)
    route.dec = _TraceDecoder(trace)

    route._set_encoder_state(route.dec, source_frames=40, synchronize=False)

    assert trace[0] == ("wait", route.dec.stream, route._producer_event)
    assert trace[1] == (
        "set",
        {
            "hidden_fp16_ptr": hidden.data_ptr(),
            "attention_mask_int32_ptr": route._torch_mask_ptr,
            "source_frames": 40,
            "synchronize": False,
        },
    )
    assert trace[2] == (
        "precompute",
        {"synchronize": False, "reset": False},
    )
    assert route._torch_hidden is hidden


def test_teardown_drains_consumer_before_releasing_torch_output() -> None:
    trace: list[tuple[object, ...]] = []
    route = _bare_route()
    route._retain_torch_hidden(_FakeHidden(0xABC000))
    route.cuda_runtime = _TraceRuntime(trace)
    route.dec = _TraceDecoder(trace)
    route.enc = None
    route._enc_chain_exec = None
    route._enc_chain_graph = 0
    route._producer_event = 0xE001
    route._device_buffers = []

    route.close()

    assert trace == [
        ("sync", route.dec.stream),
        ("decoder_close",),
        ("event_destroy", 0xE001),
    ]
    assert route._torch_hidden is None
    assert route._torch_hidden_ptr == 0


def test_encoder_mask_readback_fails_immediately_on_mismatch(monkeypatch) -> None:
    route = _bare_route()
    route.fixture = SimpleNamespace(encoder_mask=np.ones(4, dtype=np.int32))
    route.cuda_runtime = object()
    route.dec = SimpleNamespace(
        workspace=SimpleNamespace(
            allocation=lambda _name: SimpleNamespace(buffer=object())
        )
    )

    def copy_mismatch(host_ptr, _buffer, nbytes, *, runtime) -> None:
        del runtime
        count = nbytes // ctypes.sizeof(ctypes.c_int32)
        host = (ctypes.c_int32 * count).from_address(host_ptr)
        for index in range(count):
            host[index] = 0

    monkeypatch.setattr("hipengine.core.memory.copy_device_to_host", copy_mismatch)

    with pytest.raises(RuntimeError, match="encoder-mask readback mismatch"):
        route._verify_installed_encoder_mask(source_frames=4)


def test_encoder_mask_readback_returns_matching_report(monkeypatch) -> None:
    expected = np.asarray([1, 0, 1, 1], dtype=np.int32)
    route = _bare_route()
    route.fixture = SimpleNamespace(encoder_mask=expected)
    route.cuda_runtime = object()
    route.dec = SimpleNamespace(
        workspace=SimpleNamespace(
            allocation=lambda _name: SimpleNamespace(buffer=object())
        )
    )

    def copy_match(host_ptr, _buffer, nbytes, *, runtime) -> None:
        del runtime
        count = nbytes // ctypes.sizeof(ctypes.c_int32)
        host = (ctypes.c_int32 * count).from_address(host_ptr)
        for index, value in enumerate(expected[:count]):
            host[index] = int(value)

    monkeypatch.setattr("hipengine.core.memory.copy_device_to_host", copy_match)

    assert route._verify_installed_encoder_mask(source_frames=4) == {
        "source_frames": 4,
        "readback_matches": True,
        "readback": [1, 0, 1, 1],
        "expected": [1, 0, 1, 1],
    }
