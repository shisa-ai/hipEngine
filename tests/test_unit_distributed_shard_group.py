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


# ---------------------------------------------------------------------------
# Per-rank shard widths (the uneven split)
# ---------------------------------------------------------------------------


def _uneven_group(rt: FakeHipRuntime, *, widths, layers=(0,), hidden=8, variants=None,
                  weight_widths=None):
    """A group whose ranks hold different shard widths.

    ``weight_widths`` lets a test supply weights for every rank while the
    group's own ``per_rank_ffn`` mapping is incomplete, so the refusal under
    test is the group's and not the helper's.
    """

    held = dict(weight_widths or widths)
    weights = {
        layer: {device: _weights(rt, device, hidden, held[device]) for device in (0, 1)}
        for layer in layers
    }
    return MlpShardGroup(
        rt,
        devices=(0, 1),
        streams={0: 0, 1: 0},
        hidden=hidden,
        per_rank_ffn=widths,
        weights=weights,
        driver="python",
        mlp_decode_variant=variants,
    )


def test_uneven_group_gives_each_rank_its_own_width(group_env) -> None:
    rt, _ = group_env
    group = _uneven_group(rt, widths={0: 2, 1: 6})
    assert group.per_rank_ffn == {0: 2, 1: 6}
    assert group._ranks[(0, 0)].per_rank_ffn == 2
    assert group._ranks[(0, 1)].per_rank_ffn == 6


def test_scalar_width_still_applies_to_every_rank(group_env) -> None:
    rt, _ = group_env
    group, _w = _group(rt, layers=(0,), per_rank_ffn=4)
    assert group.per_rank_ffn == {0: 4, 1: 4}
    assert group._ranks[(0, 0)].per_rank_ffn == 4
    assert group._ranks[(0, 1)].per_rank_ffn == 4


def test_uneven_group_rejects_a_missing_rank_width(group_env) -> None:
    rt, _ = group_env
    with pytest.raises(ShardGroupError, match="no width for ranks"):
        _uneven_group(rt, widths={0: 4}, weight_widths={0: 4, 1: 4})


def test_uneven_group_rejects_a_non_positive_width(group_env) -> None:
    rt, _ = group_env
    with pytest.raises(ShardGroupError, match="must be positive"):
        _uneven_group(rt, widths={0: 0, 1: 8})


def test_uniform_variants_report_a_scalar(group_env) -> None:
    rt, _ = group_env
    group = _uneven_group(
        rt,
        widths={0: 4, 1: 4},
        variants={0: "dense_dual_local32_bf16_bf16_out", 1: "dense_dual_local32_bf16_bf16_out"},
    )
    assert group.mlp_decode_variant == "dense_dual_local32_bf16_bf16_out"
    assert group._ranks[(0, 0)].mlp_decode_variant == "dense_dual_local32_bf16_bf16_out"


def test_divergent_variants_report_per_rank(group_env) -> None:
    # An uneven split changes each rank's shard shape, so the shape-qualified
    # policy can admit the fused route on one rank and not the other. Reporting
    # a single scalar then would be a lie.
    rt, _ = group_env
    group = _uneven_group(
        rt,
        widths={0: 2, 1: 6},
        variants={0: "dense_dual_local32_bf16_bf16_out", 1: None},
    )
    assert group.mlp_decode_variant == {0: "dense_dual_local32_bf16_bf16_out", 1: None}
    assert group._ranks[(0, 0)].mlp_decode_variant == "dense_dual_local32_bf16_bf16_out"
    assert group._ranks[(0, 1)].mlp_decode_variant is None


def test_uneven_group_forward_launches_at_each_rank_own_width(group_env) -> None:
    rt, spy = group_env
    group = _uneven_group(rt, widths={0: 2, 1: 6})
    group.forward(0, _inputs(rt, hidden=8))
    # The spy records each launch's out_features, so the widths the kernels are
    # asked for are directly visible: two projections at this rank's shard
    # width (gate, up) and one back to hidden (down).
    per_device: dict[int, list[int]] = {0: [], 1: []}
    for device, name, feature in spy.calls:
        if name == "linear":
            per_device[device].append(feature)
    assert sorted(per_device[0]) == [2, 2, 8]
    assert sorted(per_device[1]) == [6, 6, 8]
    silu_widths = {device: feature for device, name, feature in spy.calls if name == "silu"}
    assert silu_widths == {0: 2, 1: 6}
    group.close()


class _FakeDeviceExchange:
    """Records the device reduction's submission order without HIP."""

    def __init__(self, runtime, **kwargs):
        self.runtime = runtime
        self.kwargs = dict(kwargs)
        self.devices = tuple(kwargs["devices"])
        self.hidden = int(kwargs["hidden"])
        self.rows = int(kwargs["rows"])
        self.num_layers = int(kwargs["num_layers"])
        self.bumps = 0
        self.resets = 0
        self.waits = 0
        self.enqueues: list[tuple[int, int, int, int]] = []
        self.closed = 0

    def bump(self) -> None:
        self.bumps += 1

    def reset_timeouts(self) -> None:
        self.resets += 1

    def wait(self) -> None:
        self.waits += 1

    def enqueue_rank(self, rank, partial, slot, out) -> int:
        self.enqueues.append((int(rank), int(partial), int(slot), int(out)))
        return int(out)

    def close(self) -> None:
        self.closed += 1


def _device_group(
    rt: FakeHipRuntime, monkeypatch, *, rows: int = 8, hidden: int = 16, per_rank_ffn: int = 8
):
    """A group in device reduce mode over a recording fake exchange."""

    created: list[_FakeDeviceExchange] = []

    class _Exchange(_FakeDeviceExchange):
        def __init__(self, runtime, **kwargs):
            super().__init__(runtime, **kwargs)
            created.append(self)

    monkeypatch.setattr(shard_group, "CompiledDeviceExchange", _Exchange)
    weights = {
        0: {
            device: _weights(rt, device, hidden, per_rank_ffn)
            for device in (0, 1)
        }
    }
    group = MlpShardGroup(
        rt,
        devices=(0, 1),
        streams={0: 0, 1: 0},
        hidden=hidden,
        per_rank_ffn=per_rank_ffn,
        weights=weights,
        staging_dtype="bf16",
        rows=rows,
        reduce_mode="device",
    )
    return group, created[0]


def test_device_reduce_is_rejected_outside_bf16_and_two_ranks(group_env) -> None:
    """The spin-add kernel is bf16 in and out and serves exactly two ranks, so
    the other shapes must fail at construction rather than at the first layer."""

    rt, _spy = group_env
    weights = {0: {device: _weights(rt, device, 16, 8) for device in (0, 1)}}
    with pytest.raises(ShardGroupError, match="exactly two ranks"):
        MlpShardGroup(
            rt,
            devices=(0,),
            streams={0: 0},
            hidden=16,
            per_rank_ffn=8,
            weights={0: {0: _weights(rt, 0, 16, 8)}},
            staging_dtype="bf16",
            rows=4,
            reduce_mode="device",
        )
    with pytest.raises(ShardGroupError, match="bf16 in and out"):
        MlpShardGroup(
            rt,
            devices=(0, 1),
            streams={0: 0, 1: 0},
            hidden=16,
            per_rank_ffn=8,
            weights=weights,
            staging_dtype="f32",
            rows=4,
            reduce_mode="device",
        )
    with pytest.raises(ShardGroupError, match="unknown reduce_mode"):
        MlpShardGroup(
            rt,
            devices=(0, 1),
            streams={0: 0, 1: 0},
            hidden=16,
            per_rank_ffn=8,
            weights=weights,
            staging_dtype="bf16",
            rows=4,
            reduce_mode="allreduce",
        )


def test_device_reduce_bumps_once_per_layer_and_alternates_two_slots(group_env, monkeypatch) -> None:
    rt, _spy = group_env

    """Two reused slots need a counter bump per layer.

    Without it the peer's published flag already satisfies the spin's
    comparison, so layer 2 would sum layer 0's staging. This pins the bump and
    the alternation, and that the reduction writes the boundary buffer the
    residual add already consumes (so no cast is enqueued and nothing is read
    back over PCIe).
    """

    group, exchange = _device_group(rt, monkeypatch, rows=8)
    assert exchange.num_layers == 2, "two alternating slots, not one slot per layer"
    assert exchange.hidden == 16 and exchange.rows == 8

    group.begin_device_group()
    for layer_id in range(3):
        group._forward_device_reduce(
            layer_id, {0: 0xA0 + layer_id, 1: 0xB0 + layer_id}, rows=8
        )

    assert exchange.bumps == 3, "one bump per layer"
    assert [entry[2] for entry in exchange.enqueues] == [0, 0, 1, 1, 0, 0]
    assert [entry[0] for entry in exchange.enqueues] == [0, 1, 0, 1, 0, 1]
    assert [entry[3] for entry in exchange.enqueues] == [
        group._out_ptrs[0], group._out_ptrs[1],
        group._out_ptrs[0], group._out_ptrs[1],
        group._out_ptrs[0], group._out_ptrs[1],
    ]
    # The staged route's cast is skipped: the kernel already wrote bf16.
    assert not [call for call in rt.calls if call[0] == "launch" and "cast" in call[2]]


def test_device_reduce_refuses_more_rows_than_capacity(group_env, monkeypatch) -> None:
    """The exchange stages a fixed rows x hidden block, so a call wider than the
    group's capacity would reduce a region that was never allocated.

    A *shorter* call is allowed: the block is staged whole and the tail is
    allocated memory the consumer never reads, which is what lets a session
    whose workspace grew from a longer prompt prefill a shorter one.
    """

    rt, _spy = group_env
    group, exchange = _device_group(rt, monkeypatch, rows=8)
    with pytest.raises(ShardGroupError, match="cannot exceed the group capacity"):
        group._forward_device_reduce(0, {0: 1, 1: 2}, rows=16)

    # Shorter is fine, and still bumps and enqueues both ranks.
    group._forward_device_reduce(0, {0: 1, 1: 2}, rows=4)
    assert exchange.bumps == 1
    assert len(exchange.enqueues) == 2


def test_device_group_brackets_one_reset_and_one_wait(group_env, monkeypatch) -> None:
    rt, _spy = group_env

    """The timeout flags are sticky by design: reset once per group and wait
    once at its end, so a timeout in any layer still surfaces instead of being
    cleared by the next layer's bump."""

    group, exchange = _device_group(rt, monkeypatch)
    group.begin_device_group()
    group.finish_device_group()
    assert (exchange.resets, exchange.waits) == (1, 1)


def test_staged_mode_does_not_build_a_device_exchange(group_env, monkeypatch) -> None:
    rt, _spy = group_env

    """The staged transport stays the registered fallback; its bracket calls are
    no-ops rather than errors."""

    group, _exchange = _device_group(rt, monkeypatch)
    assert group._device_exchange is not None

    host = MlpShardGroup(
        rt,
        devices=(0, 1),
        streams={0: 0, 1: 0},
        hidden=16,
        per_rank_ffn=8,
        weights={0: {device: _weights(rt, device, 16, 8) for device in (0, 1)}},
        staging_dtype="bf16",
        rows=8,
        reduce_mode="host",
    )
    assert host._device_exchange is None
    host.begin_device_group()
    host.finish_device_group()
