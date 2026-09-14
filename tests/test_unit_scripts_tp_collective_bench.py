"""CPU-only tests for scripts/tp_collective_bench.py statistics and encoding."""

from __future__ import annotations

import ctypes
import importlib.util
import pathlib
import sys
from dataclasses import dataclass

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


# -- R3: the dependent reduction chain ----------------------------------------


def test_dependent_chain_expected_value_grows_with_depth(mod) -> None:
    """Each step reduces the previous result, so the value scales per step."""

    assert mod.dependent_chain_expected(1.0, world_size=2, depth=1) == 2.0
    assert mod.dependent_chain_expected(1.0, world_size=2, depth=8) == 256.0
    assert mod.dependent_chain_expected(1.0, world_size=4, depth=3) == 64.0


def test_chain_marginal_divides_by_the_depth_difference(mod) -> None:
    """A fixed per-measurement overhead must cancel in the marginal."""

    report = mod.chain_marginal_ms([(1, 0.400), (16, 0.850), (64, 2.650)])
    segments = report["segments"]
    assert segments[0]["from_depth"] == 1 and segments[0]["to_depth"] == 16
    assert segments[0]["marginal_us_per_step"] == pytest.approx((0.850 - 0.400) * 1e3 / 15)
    assert report["overall_us_per_step"] == pytest.approx((2.650 - 0.400) * 1e3 / 63)
    assert mod.chain_marginal_ms([(4, 0.5)])["marginal_us_per_step"] is None


class _ShadowChainTransport:
    """A transport that reduces host-side shadow buffers instead of device ones.

    The device pointers the benchmark passes index into the runtime's per-rank
    work and scratch arrays, so the chain's arithmetic can be checked on CPU. The
    model follows the stream semantics the real transport has, because the point
    of these tests is to check that the protocol detects a broken dependency:

    * non-collective work enqueued inside a group runs *before* every collective
      in that group, so it becomes visible only to the next group's collectives;
    * collectives inside a group execute in call order on one stream, so each
      reads what the previous one wrote, and each runs after the work already
      queued on its ranks' streams;
    * a collective issued outside a group is an error, as on the real transport.
    """

    algorithm = "rccl"

    def __init__(self, *, world_size: int, runtime) -> None:
        from hipengine.core.device import Device

        self.world_size = int(world_size)
        self.devices = tuple(Device("hip", rank) for rank in range(self.world_size))
        self.runtime = runtime
        self.groups: list[tuple[int, int]] = []
        self._open = 0
        self._group_id = 0
        # When set, collectives never publish their result: a chain that cannot
        # consume its predecessor even though its buffers alternate.
        self.stale_snapshot = False
        # One collective is one operation across ranks: the per-rank host call is
        # bookkeeping, so a reduction is applied once per batch of world_size
        # consecutive calls, and every rank's row receives the same total.
        self._calls_this_group = 0

    def stream(self, rank: int) -> int:
        return 0x10 + int(rank)

    def group_start(self) -> None:
        self._open += 1
        self._group_id += 1
        self._calls_this_group = 0
        self.runtime.group_depth = self._open

    def group_end(self) -> None:
        self._open -= 1
        if self._open < 0:
            raise AssertionError("group_end without group_start")
        self.runtime.group_depth = self._open
        if self._open == 0:
            self.runtime.publish_staged()

    def all_reduce_sum(self, rank, send_ptr, recv_ptr, *, count, dtype) -> None:
        if self._open <= 0:
            raise AssertionError("collective issued outside a group")
        self.groups.append((self._group_id, int(rank)))
        self._calls_this_group += 1
        if self._calls_this_group % self.world_size:
            return
        # A collective executes after everything its ranks already enqueued on
        # their streams, so queued copies have run by the time it reads. Without
        # this the shadow would read a buffer the protocol had not yet filled.
        self.runtime.drain_all()
        kind = "scratch" if int(send_ptr) >= _ShadowChainRuntime.SCRATCH_BASE else "work"
        total = float(sum(row[0] for row in getattr(self.runtime, kind)))
        if self.stale_snapshot:
            return
        for row in self.runtime.target_for(int(recv_ptr)):
            row[0] = total

    def sync(self, *, timeout_s=None) -> None:
        # A transport-wide barrier completes every rank, so it drains each rank's
        # queued work. It is not a host wait on one stream, so it is not counted
        # as one.
        self.runtime.drain_all()
        return None


@dataclass(frozen=True)
class _QueuedOp:
    """One submitted copy, held until its stream is synchronized."""

    stream: int
    dst: int
    src: int
    nbytes: int
    kind: object


class _ShadowChainRuntime:
    """Runtime stand-in with per-rank shadow memory and no device calls."""

    device_kind = "hip"
    WORK_BASE = 0x2000
    SCRATCH_BASE = 0x4000
    STRIDE = 4096

    def __init__(self, *, world_size: int, count: int, consume_previous: bool = True) -> None:
        self.world_size = int(world_size)
        self.count = int(count)
        self.consume_previous = consume_previous
        self.work = [[0.0] * self.count for _ in range(self.world_size)]
        self.scratch = [[0.0] * self.count for _ in range(self.world_size)]
        self.seed_value = 1.0
        self.current = 0
        self.copies = 0
        self.events = 0
        # 0 outside a group. A non-collective write while inside a group is staged
        # and published at group_end, matching "enqueued work runs before the
        # group's collectives".
        self.group_depth = 0
        self._staged: list[tuple[list, int, float]] = []
        self.streams = 0
        self.created_streams: list[int] = []
        self.destroyed_streams: list[int] = []
        self.syncs = 0
        self.registered_host: set[int] = set()
        # Nothing in this queue executes until its stream is synchronized.
        self.queue: list[_QueuedOp] = []
        # When set, that many submissions succeed and the next one raises, so a
        # failure path can be exercised without a device.
        self.fail_after_ops: int | None = None
        self._submitted_ops = 0

    def work_ptr(self, rank: int) -> int:
        return self.WORK_BASE + self.STRIDE * int(rank)

    def scratch_ptr(self, rank: int) -> int:
        return self.SCRATCH_BASE + self.STRIDE * int(rank)

    def queue_device_copy(self, *, dst: int, src: int, stream: int) -> None:
        from hipengine.core.runtime import MemcpyKind

        self.memcpy_async(dst, src, self.count * 4, MemcpyKind.DEVICE_TO_DEVICE, stream)

    def slot(self, ptr: int) -> tuple[list, int]:
        if self.WORK_BASE <= ptr < self.SCRATCH_BASE:
            return self.work, (ptr - self.WORK_BASE) // self.STRIDE
        return self.scratch, (ptr - self.SCRATCH_BASE) // self.STRIDE

    def source_for(self, ptr: int) -> list:
        """The buffer a collective reads, for every rank."""

        return self.scratch if ptr >= self.SCRATCH_BASE else self.work

    def target_for(self, ptr: int) -> list:
        return self.scratch if ptr >= self.SCRATCH_BASE else self.work

    def publish_staged(self) -> None:
        for buffer, row, value in self._staged:
            buffer[row][0] = value
        self._staged = []

    def get_device(self) -> int:
        return self.current

    def set_device(self, device: int) -> None:
        self.current = int(device)

    def stream_create(self) -> int:
        self.streams += 1
        self.created_streams.append(self.streams)
        return self.streams

    def stream_destroy(self, stream) -> None:
        self.destroyed_streams.append(int(stream))

    def stream_synchronize(self, stream) -> None:
        # Counted so a test can check how many host waits a protocol performs, and
        # the point at which that stream's queued work actually executes.
        self.syncs += 1
        self.drain(int(stream))

    def drain(self, stream: int) -> None:
        """Execute one stream's queued operations in submission order.

        Nothing runs at submission time. That is what makes this an ownership
        oracle rather than a copy: a host-to-device read that is still queued sees
        whatever the host slot holds when the stream is finally drained, so a
        premature host write into a slot whose read is outstanding corrupts the
        device value instead of passing silently. Per-stream queues also give rank
        skew for free, because one rank's queue runs only when its own stream is
        synchronized.
        """

        stream = int(stream)
        remaining: list[_QueuedOp] = []
        for op in self.queue:
            if op.stream == stream:
                self._execute(op)
            else:
                remaining.append(op)
        self.queue = remaining

    def drain_all(self) -> None:
        for stream in sorted({op.stream for op in self.queue}):
            self.drain(stream)

    def host_register(self, ptr: int, nbytes: int) -> None:
        self.registered_host.add(int(ptr))

    def host_unregister(self, ptr: int) -> None:
        self.registered_host.discard(int(ptr))

    def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
        self._submitted_ops += 1
        if self.fail_after_ops is not None and self._submitted_ops > self.fail_after_ops:
            raise RuntimeError("injected copy failure")
        self.queue.append(
            _QueuedOp(
                stream=int(stream),
                dst=int(dst),
                src=int(src),
                nbytes=int(nbytes),
                kind=kind,
            )
        )

    def _execute(self, op: "_QueuedOp") -> None:
        from hipengine.core.runtime import MemcpyKind

        dst, src, nbytes, kind = op.dst, op.src, op.nbytes, op.kind
        if kind in (MemcpyKind.HOST_TO_DEVICE, MemcpyKind.DEVICE_TO_HOST):
            # Host staging is real memory: the staged path sums the pinned slot,
            # so the shadow has to move actual bytes through it, at drain time.
            if kind == MemcpyKind.DEVICE_TO_HOST:
                buffer, row = self.slot(int(src))
                payload = np.zeros(self.count, dtype=np.float32)
                payload[:] = np.asarray(buffer[row][: self.count], dtype=np.float32)
                ctypes.memmove(int(dst), payload.tobytes(), int(nbytes))
            else:
                payload = np.frombuffer(
                    ctypes.string_at(int(src), int(nbytes)), dtype=np.float32
                )
                buffer, row = self.slot(int(dst))
                buffer[row] = [float(value) for value in payload] + [0.0] * (
                    self.count - int(payload.size)
                )
            return
        dst, src, nbytes = int(dst), int(src), int(nbytes)
        self.copies += 1
        target, target_index = self.slot(dst)
        if not self.consume_previous:
            # The old ladder re-read an unchanged input: the consumer never
            # consumes the reduction result.
            value = self.seed_value
        else:
            _, source_index = self.slot(src)
            value = self.work[source_index][0]
        if self.group_depth:
            self._staged.append((target, target_index, value))
        else:
            target[target_index][0] = value

    def event_create(self) -> int:
        self.events += 1
        return self.events

    def event_record(self, event, stream) -> None:
        return None

    def event_elapsed_time_ms(self, start, end) -> float:
        return 0.5

    def event_destroy(self, event) -> None:
        return None


def _shadow_chain_case(mod):
    class _Case:
        op = "all_reduce"
        dtype = "fp32"
        rows = 1
        count = 4
        payload_bytes = 16

    return _Case()


def _shadow_chain_args(mod, *, world_size=2, consume_previous=True):
    from hipengine.core.device import Device

    case = _shadow_chain_case(mod)
    runtime = _ShadowChainRuntime(
        world_size=world_size, count=case.count, consume_previous=consume_previous
    )
    transport = _ShadowChainTransport(world_size=world_size, runtime=runtime)

    class _Buffer:
        def __init__(self, ptr: int) -> None:
            self.ptr = ptr
            self.nbytes = case.payload_bytes

    work = [_Buffer(0x2000 + 4096 * rank) for rank in range(world_size)]
    scratch = [_Buffer(0x4000 + 4096 * rank) for rank in range(world_size)]
    return {
        "transport": transport,
        "runtime": runtime,
        "case": case,
        "work": work,
        "scratch": scratch,
        "depths": (1, 4),
        "iterations": 2,
        "warmup": 0,
        "seed": 1.0,
        "timeout_s": 1.0,
    }


def _stub_chain_memory(mod, monkeypatch, kwargs) -> None:
    """Point the chain's device read/write helpers at the shadow arrays.

    ``_measure_dependent_chain`` uses the real host/device copy helpers; the
    shadow runtime owns no device memory, so they are redirected to the shadow
    buffers keyed by the same pointers.
    """

    runtime = kwargs["runtime"]

    # Width-aware: a scalar-only stub cannot tell a full vector from its first
    # element, which is what the rank-distinct vector check exists to test.
    def copy_host_array_to_device(buffer, array, nbytes=None, **call_kwargs):
        target, index = runtime.slot(int(buffer.ptr))
        values = np.asarray(array, dtype=np.float32).reshape(-1)
        row = target[index]
        row[: values.size] = [float(value) for value in values]
        for extra in range(values.size, len(row)):
            row[extra] = 0.0

    def copy_device_to_host(host_ptr, buffer, nbytes=None, **call_kwargs):
        source, index = runtime.slot(int(buffer.ptr))
        values = np.asarray(host_ptr, dtype=np.float32).reshape(-1)
        values[:] = np.asarray(source[index][: values.size], dtype=np.float32)

    monkeypatch.setattr(mod, "decode_values", lambda host, dtype: list(host))
    monkeypatch.setattr(mod, "encode_values", lambda values, dtype: values)
    from hipengine.core import memory as memory_module

    monkeypatch.setattr(memory_module, "copy_host_array_to_device", copy_host_array_to_device)
    monkeypatch.setattr(memory_module, "copy_device_to_host", copy_device_to_host)
    monkeypatch.setattr(memory_module, "host_array_ptr", lambda array: array)


def test_dependent_chain_runs_one_group_per_reduction(mod, monkeypatch) -> None:
    """Every reduction gets its own native group.

    One group for the whole ladder let RCCL aggregate the reductions, which is
    why the previous marginal latency could not be read as a per-layer cost.
    """

    kwargs = _shadow_chain_args(mod)
    monkeypatch.setattr(mod, "decode_values", lambda host, dtype: list(host.view("float32")))
    monkeypatch.setattr(mod, "encode_values", lambda values, dtype: values)
    _stub_chain_memory(mod, monkeypatch, kwargs)

    report = mod._measure_dependent_chain(**kwargs)
    world = kwargs["transport"].world_size
    iterations = int(kwargs["iterations"])
    depths = tuple(int(depth) for depth in kwargs["depths"])

    # Group shape per mode: (collectives in each group, how many such groups).
    per_step_segments = [
        (world, int(depth) * iterations) for depth in depths
    ]
    per_chain_segments = [
        (int(depth) * world, iterations) for depth in depths
    ]
    expected: list[tuple[int, int]] = []
    expected += per_step_segments  # per_step
    expected += per_chain_segments  # per_chain
    expected += per_step_segments  # per_step_alternating
    # The single-group mode is the contrast structure: one group per chain.
    single_group_segments = [(int(depth) * world, iterations) for depth in depths]
    expected += single_group_segments  # single_group_alternating
    # Captured replay is opt-in: the transport refuses capture, and driving that
    # path across depths hangs or faults the GPU, so a default run must not reach
    # it. The staged exchange uses no group at all, so it contributes no groups.
    assert "single_group_alternating_graph" not in report["modes"]
    assert "staged_exchange_host_sync" in report["modes"]

    counts: dict[int, int] = {}
    for group_id, _rank in kwargs["transport"].groups:
        counts[group_id] = counts.get(group_id, 0) + 1
    ordered = [counts[group_id] for group_id in sorted(counts)]
    flat_expected: list[int] = []
    for size, repeats in expected:
        flat_expected.extend([size] * repeats)
    assert ordered == flat_expected

    assert report["modes"]["per_step"]["group_boundary"] == "one native group per reduction"
    assert report["modes"]["per_step_alternating"]["group_boundary"] == (
        "one native group per reduction"
    )
    assert report["modes"]["per_step_alternating"]["device_copy_per_reduction"] is False
    assert report["modes"]["single_group_alternating"]["group_boundary"] == (
        "one native group for the whole chain"
    )
    assert report["modes"]["single_group_alternating"]["dependency_carried_by"] == (
        "stream order within one group"
    )


def test_alternating_chain_needs_no_device_copy(mod, monkeypatch) -> None:
    """The copy-free protocol must issue zero copies.

    Driven directly so the copy census covers only the alternating structure:
    the copy-bearing mode is the contrast case and is measured separately.
    """

    kwargs = _shadow_chain_args(mod)
    monkeypatch.setattr(mod, "decode_values", lambda host, dtype: list(host.view("float32")))
    monkeypatch.setattr(mod, "encode_values", lambda values, dtype: values)
    _stub_chain_memory(mod, monkeypatch, kwargs)
    runtime = kwargs["runtime"]

    report = mod._measure_alternating_chain(
        transport=kwargs["transport"],
        runtime=runtime,
        case=kwargs["case"],
        buffers=(kwargs["work"], kwargs["scratch"]),
        depths=kwargs["depths"],
        iterations=kwargs["iterations"],
        warmup=kwargs["warmup"],
        seed=kwargs["seed"],
        timeout_s=kwargs["timeout_s"],
        graph=False,
        group_boundary="per_step",
    )
    assert runtime.copies == 0, "the alternating chain must not copy between reductions"
    assert report["device_copy_per_reduction"] is False
    assert report["consumes_predecessor_via"] == "alternating buffers"
    assert report["depths"]["4"]["final_value_matches"] is True
    assert report["depends_on_every_step"] is True
    # One group per reduction, each holding one collective per rank.
    counts: dict[int, int] = {}
    for group_id, _rank in kwargs["transport"].groups:
        counts[group_id] = counts.get(group_id, 0) + 1
    assert sorted(counts.values()) == [kwargs["transport"].world_size] * 2 * (1 + 4)


def test_alternating_chain_value_check_accepts_a_dependent_chain(mod, monkeypatch) -> None:
    kwargs = _shadow_chain_args(mod)
    _stub_chain_memory(mod, monkeypatch, kwargs)
    report = mod._measure_dependent_chain(**kwargs)
    alternating = report["modes"]["per_step_alternating"]
    assert alternating["depths"]["4"]["final_value_matches"] is True
    assert alternating["depths"]["4"]["observed_final_value"] == [16.0, 16.0]
    assert alternating["depends_on_every_step"] is True
    # Step i writes the buffer step i+1 reads, so an even depth ends in buffer 0.
    assert alternating["depths"]["4"]["final_buffer"] == 0
    assert alternating["depths"]["1"]["final_buffer"] == 1


def test_alternating_chain_value_check_rejects_a_stale_input(mod, monkeypatch) -> None:
    """A chain that re-reads its original input must fail, copy or no copy."""

    kwargs = _shadow_chain_args(mod)
    kwargs["transport"].stale_snapshot = True
    _stub_chain_memory(mod, monkeypatch, kwargs)
    report = mod._measure_dependent_chain(**kwargs)
    alternating = report["modes"]["per_step_alternating"]
    assert alternating["depths"]["4"]["final_value_matches"] is False
    assert alternating["depends_on_every_step"] is False
    # A chain whose collectives never publish leaves the seed in buffer 0, so a
    # four-step chain ends holding seed instead of seed * world^4.
    assert alternating["depths"]["4"]["observed_final_value"] == [1.0, 1.0]
    assert alternating["depths"]["4"]["expected_final_value"] == 16.0


def test_single_group_alternating_chain_still_consumes_its_predecessor(mod, monkeypatch) -> None:
    """Stream order inside one group must carry the dependency on its own.

    This is the structure the transport can capture, so whether the dependency
    survives without a per-reduction group boundary is the question that decides
    whether a captured replay is a valid measurement at all.
    """

    kwargs = _shadow_chain_args(mod)
    _stub_chain_memory(mod, monkeypatch, kwargs)
    report = mod._measure_dependent_chain(**kwargs)
    single = report["modes"]["single_group_alternating"]
    assert single["depths"]["1"]["final_value_matches"] is True
    assert single["depths"]["4"]["final_value_matches"] is True
    assert single["depths"]["4"]["observed_final_value"] == [16.0, 16.0]
    assert single["depends_on_every_step"] is True
    # One group for the whole chain, holding one collective per rank per step.
    counts: dict[int, int] = {}
    for group_id, _rank in kwargs["transport"].groups:
        counts[group_id] = counts.get(group_id, 0) + 1
    assert max(counts.values()) == kwargs["transport"].world_size * 4


def test_alternating_chain_refuses_an_unknown_group_boundary(mod) -> None:
    kwargs = _shadow_chain_args(mod)
    with pytest.raises(ValueError, match="group_boundary"):
        mod._measure_alternating_chain(
            transport=kwargs["transport"],
            runtime=kwargs["runtime"],
            case=kwargs["case"],
            buffers=(kwargs["work"], kwargs["scratch"]),
            depths=(1,),
            iterations=1,
            warmup=0,
            seed=1.0,
            timeout_s=1.0,
            graph=False,
            group_boundary="batched",
        )


def test_chain_final_buffer_index_alternates() -> None:
    mod = _load()
    assert mod._chain_final_buffer_index(0) == 0
    assert mod._chain_final_buffer_index(1) == 1
    assert mod._chain_final_buffer_index(2) == 0
    assert mod._chain_final_buffer_index(7) == 1


def test_chain_attribution_decomposes_the_cost() -> None:
    mod = _load()
    modes = {
        "per_step": {"marginal": {"overall_us_per_step": 178.0}},
        "per_step_alternating": {"marginal": {"overall_us_per_step": 150.0}},
        "per_step_alternating_graph": {"marginal": {"overall_us_per_step": 142.0}},
        "single_group_alternating": {"marginal": {"overall_us_per_step": 90.0}},
        "single_group_alternating_graph": {"marginal": {"overall_us_per_step": 60.0}},
    }
    attribution = mod._chain_attribution(modes)
    assert attribution["device_copy_us"] == pytest.approx(28.0)
    # Host submission is measured by replaying the *same* device structure, so the
    # delta is host cost rather than a device-side structure change.
    assert attribution["host_submission_us"] == pytest.approx(8.0)
    assert attribution["per_step_over_single_group_us"] == pytest.approx(60.0)
    assert attribution["graph_launch_amortized_us"] == pytest.approx(30.0)
    assert attribution["collective_and_wait_us"] == pytest.approx(142.0)
    assert attribution["per_step_over_graph_replay"] == pytest.approx(150.0 / 142.0)
    # A missing mode leaves its term out rather than inventing a zero.
    partial = mod._chain_attribution({"per_step": {"marginal": {"overall_us_per_step": 178.0}}})
    assert "device_copy_us" not in partial
    assert "host_submission_us" not in partial
    assert partial["graph_replay_per_step_us"] is None


def test_dependent_chain_accepts_a_chain_that_consumes_its_predecessor(mod, monkeypatch) -> None:
    kwargs = _shadow_chain_args(mod)
    _stub_chain_memory(mod, monkeypatch, kwargs)
    report = mod._measure_dependent_chain(**kwargs)
    per_step = report["modes"]["per_step"]
    assert per_step["depths"]["1"]["final_value_matches"] is True
    assert per_step["depths"]["4"]["final_value_matches"] is True
    assert per_step["depths"]["4"]["observed_final_value"] == [16.0, 16.0]
    assert per_step["depends_on_every_step"] is True


def test_dependent_chain_rejects_a_chain_that_ignores_its_predecessor(mod, monkeypatch) -> None:
    """A ladder over unchanged inputs must fail the check, not pass quietly."""

    kwargs = _shadow_chain_args(mod, consume_previous=False)
    _stub_chain_memory(mod, monkeypatch, kwargs)
    report = mod._measure_dependent_chain(**kwargs)
    per_step = report["modes"]["per_step"]
    assert per_step["depths"]["4"]["final_value_matches"] is False
    assert per_step["depths"]["4"]["observed_final_value"] == [2.0, 2.0]
    assert per_step["depends_on_every_step"] is False


def test_dependent_chain_is_skipped_for_non_reductions(mod, monkeypatch) -> None:
    kwargs = _shadow_chain_args(mod)
    kwargs["case"].op = "broadcast"
    report = mod._measure_dependent_chain(**kwargs)
    assert "skipped" in report
def test_dependency_verdict_ignores_a_saturated_depth(mod, monkeypatch) -> None:
    """A depth whose closed form is ``inf`` cannot prove the chain compounded.

    At depth 128 the fp32 chain saturates, so the observed and expected values
    are both ``inf`` and the equality check passes trivially. The verdict has to
    come from the deepest depth with a finite expected value.
    """

    depths = {
        "16": {
            "observed_final_value": [65536.0, 65536.0],
            "final_value_matches": True,
            "value_check_informative": True,
        },
        "128": {
            "observed_final_value": [float("inf"), float("inf")],
            "final_value_matches": True,
            "value_check_informative": False,
        },
    }
    verdict, reason = mod._dependency_verdict(depths, single_step_value=2.0)
    assert verdict is True
    assert reason is None
    assert mod._deepest_informative_depth(depths) == "16"

    # A chain that never compounds is still rejected, and the rejection names the
    # depth the verdict came from rather than the saturated one.
    stalled = {
        "16": {
            "observed_final_value": [2.0, 2.0],
            "final_value_matches": False,
            "value_check_informative": True,
        },
        "128": {
            "observed_final_value": [float("inf"), float("inf")],
            "final_value_matches": True,
            "value_check_informative": False,
        },
    }
    verdict, reason = mod._dependency_verdict(stalled, single_step_value=2.0)
    assert verdict is False
    assert reason == "depth 16 did not reach the closed form"

    # With no informative depth there is no evidence at all.
    verdict, reason = mod._dependency_verdict(
        {
            "128": {
                "observed_final_value": [float("inf")],
                "final_value_matches": True,
                "value_check_informative": False,
            }
        },
        single_step_value=2.0,
    )
    assert verdict is False
    assert reason == "no depth has a finite expected value, so no check is informative"


def test_opt_in_modes_stay_out_of_a_default_run(mod) -> None:
    """The single-group capture faults the GPU, so it is not a default.

    Capturing one group for the whole chain faults at depth 32 on this host. The
    per-step capture keeps the dependency and is measured by default; the staged
    exchange replaces the transport and is also measured by default.
    """

    assert "single_group_alternating_graph" not in mod.DEFAULT_CHAIN_MODES
    assert "single_group_alternating_graph" in mod.OPT_IN_CHAIN_MODES
    assert "per_step_alternating_graph" in mod.DEFAULT_CHAIN_MODES
    assert "staged_exchange_host_sync" in mod.DEFAULT_CHAIN_MODES
def _shadow_runtime(mod, *, count: int = 4):
    return _ShadowChainRuntime(world_size=2, count=count, consume_previous=True)


def test_queued_host_reads_execute_at_sync_not_at_submission(mod) -> None:
    """The ownership oracle must be able to see a premature host-slot write.

    ``memcpy_async`` queues; nothing moves until the stream is synchronized. A
    host-to-device read that is still queued therefore observes whatever the host
    slot holds when the stream is finally drained. This test pins that property
    directly, because it is the only reason the protocol tests below can catch a
    slot being rewritten while a read of it is outstanding.
    """

    import ctypes

    from hipengine.core.memory import host_buffer_ptr
    from hipengine.core.runtime import MemcpyKind

    runtime = _shadow_runtime(mod)
    slot = ctypes.create_string_buffer(4 * 4)
    runtime.work = [[0.0, 0.0, 0.0, 0.0] for _ in range(2)]
    runtime.scratch = [[0.0, 0.0, 0.0, 0.0] for _ in range(2)]
    # A host-to-device read of the slot, then a host write into that slot before
    # the stream is drained.
    runtime.memcpy_async(
        runtime.work_ptr(0),
        host_buffer_ptr(slot),
        16,
        MemcpyKind.HOST_TO_DEVICE,
        stream=1,
    )
    assert runtime.queue, "the copy must be queued, not executed"
    np.frombuffer(slot, dtype=np.float32)[:] = np.float32(7.0)
    runtime.stream_synchronize(1)
    # The read ran at drain time, so it consumed the write that happened after
    # submission. That is exactly how a premature slot write becomes visible: a
    # protocol that rewrites a slot whose read is still queued lands here with
    # the new contents and produces the wrong value instead of passing silently.
    assert runtime.work[0][0] == 7.0
    assert not runtime.queue


def test_streams_queue_independently_so_rank_skew_is_modelled(mod) -> None:
    """One rank's queued work must not run when the other rank synchronizes."""

    runtime = _shadow_runtime(mod)
    runtime.work = [[float(rank + 1), 0.0, 0.0, 0.0] for rank in range(2)]
    runtime.scratch = [[0.0, 0.0, 0.0, 0.0] for _ in range(2)]
    runtime.queue_device_copy(
        dst=runtime.scratch_ptr(0), src=runtime.work_ptr(0), stream=0x10
    )
    runtime.queue_device_copy(
        dst=runtime.scratch_ptr(1), src=runtime.work_ptr(1), stream=0x11
    )
    runtime.stream_synchronize(0x10)
    assert runtime.scratch[0][0] == 1.0, "rank 0's queued copy must have run"
    assert runtime.scratch[1][0] == 0.0, "rank 1's copy must still be queued"
    assert [op.stream for op in runtime.queue] == [0x11]
    runtime.stream_synchronize(0x11)
    assert runtime.scratch[1][0] == 2.0
    assert not runtime.queue


@pytest.mark.parametrize("depth", [3, 4, 10])
def test_staged_chain_validates_at_odd_even_and_reused_slots(mod, monkeypatch, depth) -> None:
    """Odd and even depths, with more slot reuse than the ladder covers."""

    kwargs = _shadow_chain_args(mod)
    kwargs["depths"] = (depth,)
    kwargs["iterations"] = 2
    kwargs["warmup"] = 0
    kwargs["modes"] = ("staged_exchange_batched",)
    _stub_chain_memory(mod, monkeypatch, kwargs)
    report = mod._measure_dependent_chain(**kwargs)
    entry = report["modes"]["staged_exchange_batched"]
    assert entry["depths"][str(depth)]["final_value_matches"] is True
    assert entry["depends_on_every_step"] is True
    runtime = kwargs["runtime"]
    assert not runtime.queue, "the chain must drain every queued copy"
    assert not runtime.registered_host, "every host slot must be unregistered"


def test_staged_chain_releases_slots_and_streams_when_a_copy_fails(
    mod, monkeypatch
) -> None:
    """A mid-chain failure must not leak registrations or streams."""

    kwargs = _shadow_chain_args(mod)
    kwargs["depths"] = (4,)
    kwargs["iterations"] = 1
    kwargs["warmup"] = 0
    kwargs["modes"] = ("staged_exchange_batched",)
    _stub_chain_memory(mod, monkeypatch, kwargs)
    runtime = kwargs["runtime"]
    runtime.fail_after_ops = 3
    with pytest.raises(RuntimeError, match="injected copy failure"):
        mod._measure_dependent_chain(**kwargs)
    assert not runtime.registered_host, "a failed chain must unregister its slots"
    assert sorted(runtime.destroyed_streams) == sorted(runtime.created_streams), (
        "a failed chain must destroy every stream it created"
    )


def test_full_vector_check_is_rank_distinct_and_bounded(mod, monkeypatch) -> None:
    """The vector check must cover every element with distinct rank inputs."""

    kwargs = _shadow_chain_args(mod)
    kwargs["depths"] = (4,)
    kwargs["iterations"] = 1
    kwargs["warmup"] = 0
    kwargs["modes"] = ("staged_exchange_batched",)
    _stub_chain_memory(mod, monkeypatch, kwargs)
    report = mod._measure_dependent_chain(**kwargs)
    check = report["modes"]["staged_exchange_batched"]["vector_check"]
    assert check["elements"] == kwargs["case"].count > 1, "element zero is not a vector"
    assert check["rank_seeds_differ"] is True
    assert check["ranks_agree"] is True
    assert check["full_vector_matches"] is True
    assert check["max_abs_error"] == 0.0
    assert check["depth"] == 4


def test_batched_staging_drops_the_host_to_device_waits(mod, monkeypatch) -> None:
    """The batched protocol must submit both ranks before waiting, and skip the
    host-to-device waits entirely.

    Dropping a wait is only safe because the slot-reuse guard is the
    device-to-host wait the host already performs two steps later, so the count
    has to be exact: the batched protocol performs exactly the two
    device-to-host waits per step, plus one drain per chain, and nothing else.
    """

    depth = 4
    chains = 1  # warmup 0, one timed iteration
    counts: dict[str, int] = {}
    for mode in ("staged_exchange_host_sync", "staged_exchange_batched"):
        kwargs = _shadow_chain_args(mod)
        kwargs["depths"] = (depth,)
        kwargs["iterations"] = chains
        kwargs["warmup"] = 0
        kwargs["modes"] = (mode,)
        _stub_chain_memory(mod, monkeypatch, kwargs)
        report = mod._measure_dependent_chain(**kwargs)
        entry = report["modes"][mode]
        assert entry["depths"][str(depth)]["final_value_matches"] is True
        assert entry["depends_on_every_step"] is True
        if mode == "staged_exchange_batched":
            assert entry["protocol"] == "batched"
            assert entry["host_waits_per_reduction"] == 2
        else:
            assert entry["protocol"] == "serial"
            assert entry["host_waits_per_reduction"] == 4
        # The full-vector check runs through the same batched protocol and has its
        # own waits; they are a separate population from the ladder's, so they are
        # subtracted rather than left inside the protocol's count.
        counts[mode] = kwargs["runtime"].syncs - entry.get("vector_check", {}).get(
            "stream_syncs", 0
        )

    steps = depth * chains
    # Serial: four waits per step. Batched: two per step plus one drain per chain.
    # The event probe inside each run synchronizes the same number of times in
    # both, so it cancels out of the difference.
    assert counts["staged_exchange_host_sync"] - counts["staged_exchange_batched"] == (
        2 * steps - 2 * chains
    )
