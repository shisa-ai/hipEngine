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
    work and scratch arrays, so the chain's arithmetic can be checked on CPU.
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

    def stream(self, rank: int) -> int:
        return 0x10 + int(rank)

    def group_start(self) -> None:
        self._open += 1
        self._group_id += 1
        # A real reduction reads every rank's contribution simultaneously, so
        # snapshot the buffers at the group boundary instead of letting one
        # rank's result feed the next rank's sum.
        self._snapshots = {
            "work": [row[0] for row in self.runtime.work],
            "scratch": [row[0] for row in self.runtime.scratch],
        }

    def group_end(self) -> None:
        self._open -= 1
        if self._open < 0:
            raise AssertionError("group_end without group_start")

    def all_reduce_sum(self, rank, send_ptr, recv_ptr, *, count, dtype) -> None:
        if self._open <= 0:
            raise AssertionError("collective issued outside a group")
        self.groups.append((self._group_id, int(rank)))
        kind = "scratch" if int(send_ptr) >= _ShadowChainRuntime.SCRATCH_BASE else "work"
        total = float(sum(self._snapshots[kind]))
        self.runtime.target_for(int(recv_ptr))[int(rank)][0] = total

    def sync(self, *, timeout_s=None) -> None:
        return None


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

    def slot(self, ptr: int) -> tuple[list, int]:
        if self.WORK_BASE <= ptr < self.SCRATCH_BASE:
            return self.work, (ptr - self.WORK_BASE) // self.STRIDE
        return self.scratch, (ptr - self.SCRATCH_BASE) // self.STRIDE

    def source_for(self, ptr: int) -> list:
        """The buffer a collective reads, for every rank."""

        return self.scratch if ptr >= self.SCRATCH_BASE else self.work

    def target_for(self, ptr: int) -> list:
        return self.scratch if ptr >= self.SCRATCH_BASE else self.work

    def get_device(self) -> int:
        return self.current

    def set_device(self, device: int) -> None:
        self.current = int(device)

    def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
        self.copies += 1
        target, target_index = self.slot(int(dst))
        if not self.consume_previous:
            # The old ladder re-read an unchanged input: the consumer never
            # consumes the reduction result.
            target[target_index][0] = self.seed_value
            return
        _, source_index = self.slot(int(src))
        source = self.work
        target[target_index][0] = source[source_index][0]

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

    def copy_host_array_to_device(buffer, array, nbytes=None, **call_kwargs):
        target, index = runtime.slot(int(buffer.ptr))
        target[index][0] = float(array[0])

    def copy_device_to_host(host_ptr, buffer, nbytes=None, **call_kwargs):
        source, index = runtime.slot(int(buffer.ptr))
        host_ptr[0] = source[index][0]

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
    # depth 1 -> one group per rank; depth 4 -> four groups per rank, per
    # measurement iteration, in both modes; per_chain adds one enclosing group.
    assert report["modes"]["per_step"]["group_boundary"] == "one native group per reduction"
    world = kwargs["transport"].world_size
    iterations = int(kwargs["iterations"])
    assert len(kwargs["transport"].groups) == 2 * (1 + 4) * world * iterations
    counts: dict[int, int] = {}
    for group_id, _rank in kwargs["transport"].groups:
        counts[group_id] = counts.get(group_id, 0) + 1
    ordered = [counts[group_id] for group_id in sorted(counts)]
    # per_step runs first: one group per reduction, each holding exactly one
    # collective per rank - the boundary the previous ladder did not have.
    per_step_groups = (1 + 4) * iterations
    assert ordered[:per_step_groups] == [world] * per_step_groups
    # per_chain then batches each whole chain into a single group: the depth-1
    # chain holds one collective per rank, the depth-4 chain four.
    assert ordered[per_step_groups:] == [world] * iterations + [4 * world] * iterations


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
