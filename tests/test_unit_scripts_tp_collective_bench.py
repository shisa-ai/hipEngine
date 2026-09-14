"""CPU-only tests for scripts/tp_collective_bench.py statistics and encoding."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "tp_collective_bench.py"


def _load():
    spec = importlib.util.spec_from_file_location("tp_collective_bench_mod", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def test_percentile_linear_interpolation(mod) -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert mod.percentile(values, 0.0) == 1.0
    assert mod.percentile(values, 1.0) == 4.0
    assert mod.percentile(values, 0.5) == pytest.approx(2.5)
    assert mod.percentile(values, 0.25) == pytest.approx(1.75)
    assert mod.percentile([5.0], 0.9) == 5.0
    with pytest.raises(ValueError):
        mod.percentile([], 0.5)
    with pytest.raises(ValueError):
        mod.percentile([1.0], 1.5)


def test_summarize_samples_reports_tails(mod) -> None:
    summary = mod.summarize_samples([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["count"] == 5
    assert summary["p50_ms"] == 3.0
    assert summary["p95_ms"] == pytest.approx(4.8)
    assert summary["p99_ms"] == pytest.approx(4.96)
    assert summary["min_ms"] == 1.0
    assert summary["max_ms"] == 5.0
    with pytest.raises(ValueError):
        mod.summarize_samples([])


def test_bandwidth_and_bus_bandwidth(mod) -> None:
    # 20 MiB in 1 ms = 20.97 GB/s algorithm bandwidth.
    assert mod.bandwidth_gbs(payload_bytes=20 * (1 << 20), latency_ms=1.0) == pytest.approx(20.971, rel=1e-3)
    assert mod.bandwidth_gbs(payload_bytes=100, latency_ms=0.0) == 0.0
    # Two-rank all-reduce bus bandwidth is algorithm bandwidth.
    assert mod.bus_bandwidth_gbs(payload_bytes=1 << 20, latency_ms=1.0, world_size=2) == pytest.approx(
        mod.bandwidth_gbs(payload_bytes=1 << 20, latency_ms=1.0)
    )
    # Four-rank all-reduce bus bandwidth scales by 2*(n-1)/n = 1.5.
    assert mod.bus_bandwidth_gbs(payload_bytes=1 << 20, latency_ms=1.0, world_size=4) == pytest.approx(
        1.5 * mod.bandwidth_gbs(payload_bytes=1 << 20, latency_ms=1.0)
    )


def test_build_cases_orders_decode_then_prefill(mod) -> None:
    cases = mod.build_cases(
        hidden_size=5120,
        dtypes=("fp32", "bf16"),
        rows=(1, 2),
        prefill_rows=(128,),
        ops=("all_reduce",),
    )
    assert [(case.rows, case.dtype) for case in cases] == [
        (1, "fp32"),
        (2, "fp32"),
        (128, "fp32"),
        (1, "bf16"),
        (2, "bf16"),
        (128, "bf16"),
    ]
    assert cases[0].count == 5120
    assert cases[0].payload_bytes == 5120 * 4
    assert cases[3].payload_bytes == 5120 * 2
    with pytest.raises(ValueError):
        mod.build_cases(hidden_size=8, dtypes=("fp32",), rows=(0,), prefill_rows=(), ops=("all_reduce",))
    with pytest.raises(ValueError):
        mod.build_cases(hidden_size=8, dtypes=("fp32",), rows=(1,), prefill_rows=(), ops=("reduce_scatter",))


def test_encode_decode_round_trip_fp32(mod) -> None:
    values = [1.0, 2.0, -3.5, 0.0]
    encoded = mod.encode_values(values, "fp32")
    assert encoded.dtype == np.float32
    np.testing.assert_array_equal(mod.decode_values(encoded, "fp32"), np.array(values, dtype=np.float32))


def test_encode_decode_round_trip_fp16(mod) -> None:
    values = [1.0, 2.0, -3.5]
    encoded = mod.encode_values(values, "fp16")
    assert encoded.dtype == np.float16
    np.testing.assert_array_equal(mod.decode_values(encoded, "fp16"), np.array(values, dtype=np.float32))


def test_encode_decode_round_trip_bf16_exact_values(mod) -> None:
    """Integers up to 2^8 are exactly representable in bf16."""

    values = [1.0, 2.0, 3.0, 128.0]
    encoded = mod.encode_values(values, "bf16")
    assert encoded.dtype == np.uint16
    np.testing.assert_array_equal(mod.decode_values(encoded, "bf16"), np.array(values, dtype=np.float32))


def test_bf16_rounding_matches_expected_bit_pattern(mod) -> None:
    # 1.0 -> 0x3F80, 2.0 -> 0x4000, 1.0078125 (1 + 2^-7) is exact in bf16.
    encoded = mod.encode_values([1.0, 2.0, 1.0078125], "bf16")
    assert [int(bits) for bits in encoded] == [0x3F80, 0x4000, 0x3F81]


def test_bf16_halfway_values_round_to_even(mod) -> None:
    # 1 + 2^-8 is exactly halfway between 0x3F80 and 0x3F81; RNE keeps 0x3F80.
    # 1 + 3*2^-8 is halfway between 0x3F81 and 0x3F82; RNE moves up to 0x3F82.
    encoded = mod.encode_values([1.00390625, 1.01171875], "bf16")
    assert [int(bits) for bits in encoded] == [0x3F80, 0x3F82]


def test_wire_itemsize(mod) -> None:
    assert mod.wire_itemsize("fp32") == 4
    assert mod.wire_itemsize("fp16") == 2
    assert mod.wire_itemsize("bf16") == 2


def test_encode_rejects_unknown_dtype(mod) -> None:
    with pytest.raises(ValueError):
        mod.encode_values([1.0], "int8")
    with pytest.raises(ValueError):
        mod.decode_values(np.zeros(1, dtype=np.uint8), "int8")


def test_link_sampler_records_transitions(mod, tmp_path: pathlib.Path) -> None:
    device = tmp_path / "device"
    device.mkdir()
    (device / "current_link_width").write_text("16\n", encoding="utf-8")
    (device / "current_link_speed").write_text("16.0 GT/s PCIe\n", encoding="utf-8")
    sampler = mod.LinkSampler({0: device}, interval_s=0.01)
    with sampler:
        import time

        time.sleep(0.03)
    payload = sampler.to_dict()
    assert payload["rank0"]["last_observed"] == {"current_width_lanes": 16, "current_speed": "16.0 GT/s PCIe"}


# -- R9: partial graph capture must release every handle it created -------------


class _FakeCaptureRuntime:
    """Runtime stand-in that records graph/executable lifecycle and can fail.

    Only the surface ``_capture_graph_probe`` uses is implemented, so a leak is
    visible as a created handle with no matching destroy call.
    """

    def __init__(self, *, fail_instantiate_at: int | None = None) -> None:
        self.fail_instantiate_at = fail_instantiate_at
        self.created_graphs: list[int] = []
        self.destroyed_graphs: list[int] = []
        self.created_execs: list[int] = []
        self.destroyed_execs: list[int] = []
        self.ended_streams: list[int] = []
        self.instantiate_calls = 0
        self._next = 0x100
        self.current = 0

    # device selection
    def get_device(self) -> int:
        return self.current

    def set_device(self, device: int) -> None:
        self.current = int(device)

    # enqueue path
    def memset_async(self, dst: int, value: int, nbytes: int, stream: int) -> None:
        return None

    # capture
    def stream_begin_capture(self, stream: int, mode: int | None = None) -> None:
        return None

    def stream_end_capture(self, stream: int) -> int:
        if stream in self.ended_streams:
            raise RuntimeError("stream is not capturing")
        self.ended_streams.append(stream)
        self._next += 1
        self.created_graphs.append(self._next)
        return self._next

    def graph_nodes(self, graph: int):
        return [graph]

    def graph_instantiate(self, graph: int) -> int:
        call = self.instantiate_calls
        self.instantiate_calls += 1
        if self.fail_instantiate_at is not None and call == self.fail_instantiate_at:
            raise RuntimeError("hipGraphInstantiate failed")
        self._next += 1
        self.created_execs.append(self._next)
        return self._next

    def graph_destroy(self, graph: int) -> None:
        self.destroyed_graphs.append(graph)

    def graph_exec_destroy(self, exec_: int) -> None:
        self.destroyed_execs.append(exec_)


def _capture_probe_kwargs(mod, runtime):
    """Minimal real arguments for ``_capture_graph_probe``."""

    from hipengine.core.device import Device

    class _Transport:
        """Enough transport for the enqueue path: a group, streams, a no-op sum."""

        world_size = 2
        devices = (Device("hip", 0), Device("hip", 1))

        def stream(self, rank: int) -> int:
            return 0x10 + rank

        def group_start(self) -> None:
            return None

        def group_end(self) -> None:
            return None

        def all_reduce_sum(self, rank, send_ptr, recv_ptr, *, count, dtype) -> None:
            return None

        def sync(self, *, timeout_s=None) -> None:
            return None

    class _Case:
        op = "all_reduce"
        payload_bytes = 64
        count = 16
        dtype = "fp32"

    class _Buffer:
        def __init__(self, ptr: int) -> None:
            self.ptr = ptr
            self.nbytes = 64

    buffers = [_Buffer(0x1000), _Buffer(0x2000)]
    return {
        "transport": _Transport(),
        "runtime": runtime,
        "case": _Case(),
        "send": buffers,
        "recv": buffers,
        "producer": buffers,
        "consumer": buffers,
        "iterations": 1,
        "warmup": 0,
        "chain_depth": 1,
        "timeout_s": 1.0,
    }


@pytest.fixture
def stub_snapshot(monkeypatch):
    """Keep the eager reference read CPU-only.

    ``_snapshot_recv`` copies from device buffers with the default HIP runtime;
    this probe test owns no device memory, so the read-back is stubbed.
    """

    from hipengine.core import memory as memory_module

    monkeypatch.setattr(
        memory_module,
        "copy_device_to_host",
        lambda host_ptr, buffer, nbytes=None, runtime=None: None,
    )
    monkeypatch.setattr(
        memory_module,
        "copy_host_array_to_device",
        lambda buffer, array, nbytes=None, runtime=None: None,
    )
    return None


def test_partial_capture_failure_destroys_graphs_and_executables(mod, stub_snapshot) -> None:
    """A failure while instantiating rank 1 must not leak rank 0's executable."""

    runtime = _FakeCaptureRuntime(fail_instantiate_at=1)
    kwargs = _capture_probe_kwargs(mod, runtime)
    # The enqueue path needs a transport it can issue on; the capture of rank 0
    # is what matters here, so a no-op group is enough.
    result = mod._capture_graph_probe(**kwargs)
    assert result["captured"] is False
    assert runtime.created_execs, "rank 0's executable must have been created"
    assert sorted(runtime.destroyed_execs) == sorted(runtime.created_execs), (
        "every created executable must be destroyed on the failure path"
    )
    assert sorted(runtime.destroyed_graphs) == sorted(runtime.created_graphs), (
        "every created graph must be destroyed on the failure path"
    )


def test_capture_failure_ends_only_streams_still_capturing(mod, stub_snapshot) -> None:
    """Cleanup must not re-end a stream whose capture already ended."""

    runtime = _FakeCaptureRuntime(fail_instantiate_at=1)
    kwargs = _capture_probe_kwargs(mod, runtime)
    mod._capture_graph_probe(**kwargs)
    assert len(runtime.ended_streams) == len(set(runtime.ended_streams)), (
        "cleanup ended an already-ended capture"
    )
