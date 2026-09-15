"""CPU tests for the MLP shard group over the staged exchange.

No ROCm is required: the same recording fake runtime the staged/shard tests
use stands in for HIP, and the GEMV/SiLU/cast launchers are stubbed, so the
tests pin the group's protocol - both chains enqueued before either is
awaited, the bf16 boundary cast per rank, no per-call allocation, poison
propagation, and exactly-once teardown.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.distributed import shard_exec, shard_group
from hipengine.distributed.shard_exec import MlpShardRank
from hipengine.distributed.shard_group import MlpShardGroup, ShardGroupError
from hipengine.distributed.transport import TransportError
from tests.test_unit_distributed_staged_and_shard import FakeHipRuntime


class _KernelSpy:
    """Records launcher calls with the device they ran under.

    Launches are appended to the runtime's call log too, so ordering between
    launches and runtime copies/waits is assertable from one list.
    """

    def __init__(self, rt: FakeHipRuntime) -> None:
        self.rt = rt
        self.calls: list[tuple[int, str, int]] = []

    def _record(self, name: str, feature: int) -> None:
        self.calls.append((self.rt.get_device(), name, int(feature)))
        self.rt.calls.append(("launch", self.rt.get_device(), name))

    def linear(self, weight, x_ptr, out_ptr, rows, hidden, out_features, **kwargs):
        self._record("linear", out_features)
        return True

    def silu(self, gate_ptr, up_ptr, act_ptr, rows, ffn, **kwargs):
        self._record("silu", ffn)

    def cast(self, x_ptr, out_ptr, count, **kwargs):
        self._record("f32_to_bf16", count)


@pytest.fixture()
def group_env(monkeypatch):
    rt = FakeHipRuntime()
    spy = _KernelSpy(rt)
    import hipengine.kernels.hip_gfx1100.convert as convert
    import hipengine.kernels.hip_gfx1100.fused.paro_silu as paro_silu
    import hipengine.runtime.gguf_linear as gguf_linear

    monkeypatch.setattr(gguf_linear, "launch_gguf_linear", spy.linear)
    monkeypatch.setattr(paro_silu, "silu_mul_separate_out_bf16", spy.silu)
    monkeypatch.setattr(shard_exec, "silu_mul_separate_out_bf16", spy.silu)
    monkeypatch.setattr(convert, "f32_to_bf16", spy.cast)
    return rt, spy


def _weights(rt: FakeHipRuntime, device: int, hidden: int, per_rank_ffn: int):
    from hipengine.distributed.shard_exec import upload_shard_weight

    weights = {}
    for role in ("ffn_gate", "ffn_up"):
        weights[role] = upload_shard_weight(
            rt,
            device=device,
            name=f"blk.0.{role}.rank{device}",
            layout="dense_bf16",
            quant_key="dense_bf16",
            payload=np.zeros((per_rank_ffn, hidden), dtype=np.uint8),
        )
    weights["ffn_down"] = upload_shard_weight(
        rt,
        device=device,
        name=f"blk.0.ffn_down.rank{device}",
        layout="dense_bf16",
        quant_key="dense_bf16",
        payload=np.zeros((hidden, per_rank_ffn), dtype=np.uint8),
    )
    return weights


def _inputs(rt: FakeHipRuntime, *, hidden: int) -> dict[int, int]:
    """Two mapped input rows: the fake's copies move real bytes."""

    return {device: rt.malloc(hidden * 2) for device in (0, 1)}


def _group(rt: FakeHipRuntime, *, layers=(0, 1), hidden=8, per_rank_ffn=4):
    weights = {
        layer: {
            device: _weights(rt, device, hidden, per_rank_ffn)
            for device in (0, 1)
        }
        for layer in layers
    }
    return MlpShardGroup(
        rt,
        devices=(0, 1),
        streams={0: 0, 1: 0},
        hidden=hidden,
        per_rank_ffn=per_rank_ffn,
        weights=weights,
        # These tests pin the Python route's protocol (H2D return copies,
        # per-rank reduced buffers, transport-level sync failures); the
        # compiled driver has its own file.
        driver="python",
    ), weights


def test_forward_enqueues_both_chains_before_the_exchange(group_env) -> None:
    rt, spy = _group_env = group_env
    group, _weights = _group(rt, layers=(0,))
    inputs = _inputs(rt, hidden=8)
    group.forward(0, inputs)
    calls = rt.calls
    # Every rank's D2D input copy and its whole shard chain (4 launches each)
    # precede any wait; the waits are the exchange's one-per-stream sync.
    first_wait = next(i for i, c in enumerate(calls) if c[0] == "stream_synchronize")
    pre_wait = calls[:first_wait]
    assert len([c for c in pre_wait if c[0] == "launch"]) == 8
    d2d = [c for c in pre_wait if c[0] == "memcpy_async" and c[2] == "DEVICE_TO_DEVICE"]
    assert len(d2d) == 2, "each rank stages its input by a same-device D2D"
    d2h = [c for c in pre_wait if c[0] == "memcpy_async" and c[2] == "DEVICE_TO_HOST"]
    assert len(d2h) == 2, "both partials stage before either rank is awaited"
    # Both ranks' chains ran under their own device.
    devices_in_chain = {c[0] for c in spy.calls if c[1] in {"linear", "silu"}}
    assert devices_in_chain == {0, 1}
    # One bf16 cast per rank after the reduction.
    casts = [c for c in spy.calls if c[1] == "f32_to_bf16"]
    assert {c[0] for c in casts} == {0, 1}
    assert all(c[2] == 8 for c in casts), "the cast covers one hidden row"
    group.close()


def test_forward_returns_a_persistent_bf16_output_pointer_per_rank(group_env) -> None:
    rt, _ = group_env
    group, _w = _group(rt, layers=(0,))
    inputs = _inputs(rt, hidden=8)
    first = group.forward(0, inputs)
    second = group.forward(0, inputs)
    assert first == second, "the boundary buffers are persistent per rank"
    before = len([c for c in rt.calls if c[0] == "malloc"])
    group.forward(0, inputs)
    assert len([c for c in rt.calls if c[0] == "malloc"]) == before, (
        "forward must not allocate"
    )
    group.close()


def test_the_exchange_walls_record_one_entry_per_forward(group_env) -> None:
    rt, _ = group_env
    group, _w = _group(rt, layers=(0,))
    inputs = _inputs(rt, hidden=8)
    group.forward(0, inputs)
    group.forward(0, inputs)
    assert len(group.exchange_walls_s) == 2
    assert group.reductions == 2
    group.close()


def test_a_missing_input_refuses_without_poisoning_the_transport(group_env) -> None:
    rt, _ = group_env
    group, _w = _group(rt, layers=(0,))
    with pytest.raises(ShardGroupError, match="input"):
        group.forward(0, {0: 0x3000})
    assert group.poisoned is False, "a caller mistake is not a rank failure"
    group.close()


def test_an_unknown_layer_is_refused(group_env) -> None:
    rt, _ = group_env
    group, _w = _group(rt, layers=(0,))
    with pytest.raises(ShardGroupError, match="layer"):
        group.forward(7, {0: 0x3000, 1: 0x3100})
    group.close()


def test_a_rank_chain_failure_raises_a_group_error(group_env, monkeypatch) -> None:
    rt, _ = group_env
    group, _w = _group(rt, layers=(0,))

    def broken_forward_partial(self, **kwargs):
        raise RuntimeError("simulated chain failure")

    monkeypatch.setattr(MlpShardRank, "forward_partial", broken_forward_partial)
    with pytest.raises(ShardGroupError, match="simulated chain failure"):
        group.forward(0, _inputs(rt, hidden=8))
    group.close()


def test_a_transport_failure_poisons_the_group(group_env, monkeypatch) -> None:
    rt, _ = group_env
    group, _w = _group(rt, layers=(0,))
    rt.fail_sync_on.add(0)

    with pytest.raises(TransportError):
        group.forward(0, _inputs(rt, hidden=8))
    assert group.poisoned is True
    with pytest.raises(TransportError, match="poisoned"):
        group.forward(0, _inputs(rt, hidden=8))
    group.close()
    assert group._closed is True


def test_close_frees_everything_exactly_once(group_env) -> None:
    rt, _ = group_env
    group, _w = _group(rt, layers=(0, 1))
    group.close()
    group.close()
    frees = [c for c in rt.calls if c[0] == "free"]
    # 2 bf16 boundary buffers + 2 ranks x 2 layers x (5 activations + 3
    # weights) + 2 transport reduced buffers.
    assert len(frees) == 2 + 2 * 2 * 8 + 2
    devices = {c[1] for c in frees}
    assert devices == {0, 1}, "every free ran through its own device"


def test_a_group_needs_weights_for_every_rank(group_env) -> None:
    rt, _ = group_env
    weights = {
        0: {
            0: _weights(rt, 0, 8, 4),
        }
    }
    with pytest.raises(ShardGroupError, match="no shard weights"):
        MlpShardGroup(
            rt,
            devices=(0, 1),
            streams={0: 0, 1: 0},
            hidden=8,
            per_rank_ffn=4,
            weights=weights,
        )


def test_closed_group_refuses_work(group_env) -> None:
    rt, _ = group_env
    group, _w = _group(rt, layers=(0,))
    group.close()
    with pytest.raises(ShardGroupError, match="closed"):
        group.forward(0, {0: 0x3000, 1: 0x3100})
