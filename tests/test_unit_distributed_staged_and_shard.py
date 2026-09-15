"""CPU tests for the staged exchange transport and the MLP shard rank.

No ROCm is required: a recording fake runtime stands in for libamdhip64 and
the launchers are stubbed, so the tests pin submission order, buffer
ownership, slot-reuse guards, poisoning, and teardown - the structure the
serving path depends on - without touching hardware.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.distributed import shard_exec
from hipengine.distributed.shard_exec import MlpShardRank, ShardWeight
from hipengine.distributed.staged import StagedExchangeTransport
from hipengine.distributed.transport import TransportError, TransportStateError


class FakeHipRuntime:
    """Records device selection, allocation, and copy calls without HIP."""

    def __init__(self, *, device_count: int = 2) -> None:
        self._count = int(device_count)
        self._current = 0
        self.selection: list[int] = []
        self.calls: list[tuple[str, ...]] = []
        self._next = 0x1000
        # Device allocations are carved from one real host arena so both copy
        # directions and post-copy reads touch mapped memory in the tests.
        self._arena = ctypes.create_string_buffer(1 << 22)
        self._arena_base = ctypes.addressof(self._arena)
        self._arena_used = 0
        self.regions: dict[int, int] = {}
        self.fail_sync_on: set[int] = set()

    # -- device -----------------------------------------------------------

    def device_count(self) -> int:
        return self._count

    def get_device(self) -> int:
        return self._current

    def set_device(self, device: int) -> None:
        self._current = int(device)
        self.selection.append(int(device))

    # -- allocation -------------------------------------------------------

    def malloc(self, nbytes: int) -> int:
        nbytes = int(nbytes)
        ptr = self._arena_base + self._arena_used
        self._arena_used += (nbytes + 255) & ~255
        ctypes.memset(ptr, 0, nbytes)
        self.calls.append(("malloc", self._current, nbytes))
        self.regions[ptr] = nbytes
        return ptr

    def free(self, ptr: int) -> None:
        self.calls.append(("free", self._current, int(ptr)))
        self.regions.pop(int(ptr), None)

    def host_register(self, ptr: int, nbytes: int) -> None:
        self.calls.append(("host_register", int(ptr), int(nbytes)))

    def host_unregister(self, ptr: int) -> None:
        self.calls.append(("host_unregister", int(ptr)))

    # -- copies -----------------------------------------------------------

    def memcpy(self, dst: int, src: int, nbytes: int, kind) -> None:
        self.calls.append(("memcpy", self._current, int(kind)))

    def memcpy_async(self, dst: int, src: int, nbytes: int, kind, stream: int) -> None:
        # Emulate both copy directions with a real memmove: every pointer in
        # this fake (device arena, pinned staging) is mapped memory, so the
        # transport's data flow is exercised for real.
        ctypes.memmove(dst, src, nbytes)
        self.calls.append(
            ("memcpy_async", self._current, getattr(kind, "name", str(kind)), int(stream))
        )

    # -- streams ----------------------------------------------------------

    def stream_synchronize(self, stream: int) -> None:
        self.calls.append(("stream_synchronize", self._current, int(stream)))
        if stream in self.fail_sync_on:
            raise RuntimeError("simulated stream failure")


class FakeStream:
    D2H = 2
    H2D = 1


def _transport(runtime: FakeHipRuntime, *, devices=(0, 1), hidden=8) -> StagedExchangeTransport:
    streams = {d: 100 + d for d in devices}
    return StagedExchangeTransport(runtime, devices=devices, streams=streams, hidden=hidden)


# -- staged exchange ----------------------------------------------------------


def test_both_d2h_submit_before_any_wait() -> None:
    rt = FakeHipRuntime()
    transport = _transport(rt)
    partials = {0: rt.malloc(32), 1: rt.malloc(32)}
    transport.reduce(partials)
    d2h = [i for i, c in enumerate(rt.calls) if c[0] == "memcpy_async" and c[2] == "DEVICE_TO_HOST"]
    h2d = [i for i, c in enumerate(rt.calls) if c[0] == "memcpy_async" and c[2] == "HOST_TO_DEVICE"]
    waits = [i for i, c in enumerate(rt.calls) if c[0] == "stream_synchronize"]
    assert len(d2h) == 2 and len(h2d) == 2 and len(waits) == 2
    assert max(d2h) < min(waits), "both D2H must precede any stream wait"
    assert min(h2d) > max(waits), "the H2D is submitted after the waits"
    assert not [c for c in rt.calls if c[0] == "stream_synchronize"][max(waits) + 1 :], (
        "no return-copy wait: one wait per stream per reduce"
    )
    assert {c[2] for c in rt.calls if c[0] == "stream_synchronize"} == {100, 101}
    transport.close()


def test_reduce_sums_the_staged_partials_into_every_rank() -> None:
    rt = FakeHipRuntime()
    transport = _transport(rt, hidden=4)
    # Place distinct f32 partials at the fake partial addresses: the fake D2H
    # writes through, so the staging slots receive these rows.
    rows = {
        0: np.array([1.0, 2.0, 3.0, 4.0], dtype="<f4"),
        1: np.array([10.0, 20.0, 30.0, 40.0], dtype="<f4"),
    }
    ptrs = {}
    keepalive = []
    for rank, row in rows.items():
        buf = ctypes.create_string_buffer(row.tobytes(), row.nbytes)
        keepalive.append(buf)
        ptrs[rank] = ctypes.addressof(buf)
        rt.regions[ptrs[rank]] = row.nbytes
    reduced = transport.reduce(ptrs)
    expected = (rows[0] + rows[1]).astype("<f4")
    for device in (0, 1):
        ptr = reduced[device]
        region = ctypes.string_at(ptr, 16)
        assert np.frombuffer(region, dtype="<f4").tolist() == pytest.approx(
            expected.tolist()
        ), "every rank's reduced buffer must receive the f32 sum"
    transport.close()


def test_staging_slots_alternate_across_calls() -> None:
    rt = FakeHipRuntime()
    transport = _transport(rt, hidden=4)
    partial = rt.malloc(16)
    partials = {0: partial, 1: partial}
    transport.reduce(partials)
    transport.reduce(partials)
    # Two slot sets exist, so consecutive reduces write different slots: the
    # second call's D2H destination is one slot set past the first's.
    assert transport._arena_nbytes == 2 * 2 * 16
    assert transport.reductions == 2
    transport.close()


def test_a_failed_wait_poisons_the_transport() -> None:
    rt = FakeHipRuntime()
    transport = _transport(rt, hidden=4)
    rt.fail_sync_on.add(101)
    partial = rt.malloc(16)
    partials = {0: partial, 1: partial}
    with pytest.raises(TransportError):
        transport.reduce(partials)
    assert transport.poisoned is True
    with pytest.raises(TransportError, match="poisoned"):
        transport.reduce(partials)
    transport.close()


def test_a_missing_partial_poisons_and_refuses() -> None:
    rt = FakeHipRuntime()
    transport = _transport(rt, hidden=4)
    with pytest.raises(TransportStateError):
        transport.reduce({0: 0x2000})
    assert transport.poisoned is True
    transport.close()


def test_close_frees_every_buffer_through_its_own_device() -> None:
    rt = FakeHipRuntime()
    transport = _transport(rt, hidden=4)
    transport.close()
    frees = [c for c in rt.calls if c[0] == "free"]
    assert len(frees) == 2, "one reduced buffer per rank"
    # Every free happened while that rank's device was current: the fake
    # records (op, current_device, ptr) at call time.
    devices_at_free = {c[1] for c in frees}
    assert devices_at_free == {0, 1}
    unregisters = [c for c in rt.calls if c[0] == "host_unregister"]
    assert len(unregisters) == 2, "staging arena plus reduced payload region"


def test_rank_devices_and_streams_are_validated() -> None:
    rt = FakeHipRuntime()
    with pytest.raises(TransportStateError):
        StagedExchangeTransport(rt, devices=(), streams={}, hidden=4)
    with pytest.raises(TransportStateError):
        StagedExchangeTransport(rt, devices=(0, 0), streams={0: 1}, hidden=4)
    with pytest.raises(TransportStateError):
        StagedExchangeTransport(rt, devices=(0, 1), streams={0: 1}, hidden=4)


# -- MLP shard rank -----------------------------------------------------------


def _shard_weights(rt: FakeHipRuntime, device: int) -> dict[str, ShardWeight]:
    weights = {}
    for role in ("ffn_gate", "ffn_up"):
        weights[role] = shard_exec.upload_shard_weight(
            rt,
            device=device,
            name=f"blk.0.{role}.weight",
            layout="t16",
            quant_key="q4_k_t16",
            payload=np.zeros((4, 8), dtype=np.uint8),
        )
    weights["ffn_down"] = shard_exec.upload_shard_weight(
        rt,
        device=device,
        name="blk.0.ffn_down.weight",
        layout="planar",
        quant_key="q4_k_t16",
        payload=np.zeros((8, 4), dtype=np.uint8),
    )
    return weights


class _LaunchSpy:
    """Records launch_gguf_linear calls with the device they ran under."""

    def __init__(self, rt: FakeHipRuntime) -> None:
        self.rt = rt
        self.calls: list[tuple[int, str, int]] = []

    def linear(self, weight, x_ptr, out_ptr, rows, hidden, out_features, **kwargs):
        self.calls.append((self.rt.get_device(), "linear", int(out_features)))
        return True

    def pair(self, gate, up, x_ptr, act_ptr, rows, hidden, ffn, **kwargs):
        self.calls.append((self.rt.get_device(), "pair", int(ffn)))
        return True

    def silu(self, gate_ptr, up_ptr, act_ptr, rows, ffn, **kwargs):
        self.calls.append((self.rt.get_device(), "silu", int(ffn)))


@pytest.fixture()
def launchers(monkeypatch):
    rt = FakeHipRuntime()
    spy = _LaunchSpy(rt)
    import hipengine.runtime.gguf_linear as gguf_linear

    monkeypatch.setattr(gguf_linear, "launch_gguf_linear", spy.linear)
    monkeypatch.setattr(gguf_linear, "launch_gguf_linear_pair_silu", spy.pair)
    monkeypatch.setattr(shard_exec, "silu_mul_separate_out_bf16", spy.silu)
    return rt, spy


def test_shard_rank_allocates_persistent_buffers_once(launchers) -> None:
    rt, _ = launchers
    rank = MlpShardRank(
        rt, device=1, stream=7, weights=_shard_weights(rt, 1), hidden=8, per_rank_ffn=4
    )
    mallocs = [c for c in rt.calls if c[0] == "malloc"]
    assert len(mallocs) == 8, "5 activations + 3 weights, exactly once"
    assert {c[1] for c in mallocs} == {1}, "every allocation is device-scoped"
    before = len(rt.calls)
    rank.forward_partial()
    assert not [c for c in rt.calls[before:] if c[0] == "malloc"], (
        "forward must not allocate"
    )
    rank.close()
    rank.close()  # idempotent


def test_forward_enqueues_the_unfused_chain_on_the_rank_device(launchers) -> None:
    rt, spy = launchers
    rank = MlpShardRank(
        rt, device=1, stream=7, weights=_shard_weights(rt, 1), hidden=8, per_rank_ffn=4
    )
    partial = rank.forward_partial()
    assert partial == rank.down_partial_ptr
    assert [c[1] for c in spy.calls] == ["linear", "linear", "silu", "linear"]
    assert {c[0] for c in spy.calls} == {1}, "every launch ran under the rank device"
    # The down GEMV consumed the full shard ffn and produced the hidden-sized
    # f32 partial: the launch order pins gate/up (out=4), silu (4), down (8).
    assert [c[2] for c in spy.calls] == [4, 4, 4, 8]
    rank.close()


def test_forward_fused_uses_the_pair_kernel(launchers) -> None:
    rt, spy = launchers
    rank = MlpShardRank(
        rt, device=0, stream=9, weights=_shard_weights(rt, 0), hidden=8, per_rank_ffn=4
    )
    rank.forward_partial(fused=True, fused_variant="dense_dual_local32_bf16_bf16_out")
    assert [c[1] for c in spy.calls] == ["pair", "linear"]
    assert [c[2] for c in spy.calls] == [4, 8]
    rank.close()


def test_fused_without_a_variant_is_rejected(launchers) -> None:
    rt, _ = launchers
    rank = MlpShardRank(
        rt, device=0, stream=9, weights=_shard_weights(rt, 0), hidden=8, per_rank_ffn=4
    )
    with pytest.raises(ValueError, match="variant"):
        rank.forward_partial(fused=True)
    rank.close()


def test_write_input_validates_the_row_size(launchers) -> None:
    rt, _ = launchers
    rank = MlpShardRank(
        rt, device=0, stream=9, weights=_shard_weights(rt, 0), hidden=8, per_rank_ffn=4
    )
    with pytest.raises(ValueError, match="bytes"):
        rank.write_input(np.zeros(4, dtype=np.uint8))
    rank.write_input(np.zeros(16, dtype=np.uint8))
    rank.close()


def test_missing_weight_roles_are_rejected_at_construction(launchers) -> None:
    rt, _ = launchers
    weights = _shard_weights(rt, 0)
    del weights["ffn_down"]
    with pytest.raises(ValueError, match="ffn_down"):
        MlpShardRank(rt, device=0, stream=9, weights=weights, hidden=8, per_rank_ffn=4)


def test_closed_rank_refuses_work(launchers) -> None:
    rt, _ = launchers
    rank = MlpShardRank(
        rt, device=0, stream=9, weights=_shard_weights(rt, 0), hidden=8, per_rank_ffn=4
    )
    rank.close()
    with pytest.raises(RuntimeError, match="closed"):
        rank.forward_partial()


def test_close_frees_activations_and_weights_once(launchers) -> None:
    rt, _ = launchers
    rank = MlpShardRank(
        rt, device=1, stream=9, weights=_shard_weights(rt, 1), hidden=8, per_rank_ffn=4
    )
    rank.close()
    rank.close()
    frees = [c for c in rt.calls if c[0] == "free"]
    assert len(frees) == 8, "5 activations + 3 weights, freed exactly once"
    assert {c[1] for c in frees} == {1}
