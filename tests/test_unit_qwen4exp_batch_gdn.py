"""Dispatch-shape and admission tests for the N-row GDN token mixer.

Every device call in the mixer is a module-level name, so these tests replace
them with recorders: the mixer's own control flow (row count per dispatch,
dispatch order, which row-to-slot vector reaches which kernel, and that a
rejected call dispatches nothing at all) is asserted without a GPU.

The state owner is exercised here too, because its admission rules are what make
the mixer's row/slot contract checkable: unique slots, permuted and shrinking
active sets, and the closing/closed lifecycle.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import hipengine.runtime.qwen4_exp_batch_gdn as batch_gdn
from hipengine.core.memory import DeviceBuffer
from hipengine.runtime.qwen4_exp_batch_gdn import (
    Qwen4ExpBatchGdnState,
    run_qwen4_exp_batch_gdn_token_mixer,
)
from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGDNScratch

SLOTS = 4
HIDDEN = 8
K_HEADS = 2
V_HEADS = 4
HEAD_DIM = 128
KERNEL = 4
QKV_WIDTH = 2 * K_HEADS * HEAD_DIM + V_HEADS * HEAD_DIM
CORE_WIDTH = V_HEADS * HEAD_DIM


class FakeRuntime:
    def __init__(self) -> None:
        self.next_ptr = 0x4000
        self.allocations: dict[int, int] = {}
        self.freed: list[int] = []
        self.memsets: list[tuple[int, int]] = []
        self.copies: list[tuple[int, int]] = []

    def malloc(self, nbytes: int) -> int:
        ptr = self.next_ptr
        self.next_ptr += max(int(nbytes), 1) + 0x100
        self.allocations[ptr] = int(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(ptr)
        self.allocations.pop(ptr, None)

    def memset(self, dst: int, value: int, nbytes: int) -> None:
        assert dst in self.allocations
        self.memsets.append((dst, int(nbytes)))

    def memcpy(self, dst: int, src: int, nbytes: int, kind) -> None:
        assert dst in self.allocations
        self.copies.append((dst, int(nbytes)))


class Recorder:
    """Records one dispatch stage in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def stage(self, name: str):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return record

    def names(self) -> list[str]:
        return [name for name, _, _ in self.calls]

    def one(self, name: str) -> tuple[tuple, dict]:
        matches = [(args, kwargs) for label, args, kwargs in self.calls if label == name]
        assert len(matches) == 1, f"expected exactly one {name} dispatch, got {len(matches)}"
        return matches[0]


@pytest.fixture
def recorder(monkeypatch) -> Recorder:
    record = Recorder()
    monkeypatch.setattr(batch_gdn, "launch_gguf_linear", record.stage("projection"))
    monkeypatch.setattr(
        batch_gdn, "qwen4_exp_linear_attn_conv_decode_indexed_f32",
        record.stage("conv"),
    )
    monkeypatch.setattr(
        batch_gdn, "qwen4_exp_gdn_decode_rowsplit_indexed_f32",
        record.stage("recurrence"),
    )
    return record


class Fixture:
    def __init__(self, *, scratch_rows: int = SLOTS) -> None:
        self.runtime = FakeRuntime()
        self.state = Qwen4ExpBatchGdnState.allocate(
            state_slots=SLOTS,
            num_k_heads=K_HEADS,
            num_v_heads=V_HEADS,
            head_dim=HEAD_DIM,
            conv_kernel=KERNEL,
            runtime=self.runtime,
        )
        self.scratch = Qwen4ExpGDNScratch.allocate(
            rows=scratch_rows,
            qkv_width=QKV_WIDTH,
            core_width=CORE_WIDTH,
            scalar_width=V_HEADS,
            hidden=HIDDEN,
            runtime=self.runtime,
        )
        self.weights = {name: f"{name}-weight" for name in batch_gdn._REQUIRED_PROJECTIONS}
        self.mixed = DeviceBuffer(ptr=0x1000, nbytes=SLOTS * HIDDEN * 4)

    def close(self) -> None:
        self.scratch.close()
        self.state.close()


@pytest.fixture
def fixture() -> Fixture:
    built = Fixture()
    try:
        yield built
    finally:
        built.close()


def _run(fixture: Fixture, recorder: Recorder, **overrides) -> DeviceBuffer:
    kwargs = dict(
        state=fixture.state,
        scratch=fixture.scratch,
        rows=2,
        hidden=HIDDEN,
        num_k_heads=K_HEADS,
        num_v_heads=V_HEADS,
        head_dim=HEAD_DIM,
        conv_kernel=KERNEL,
        conv_weight_ptr=0x910,
        dt_bias_ptr=0x920,
        a_ptr=0x930,
        norm_weight_ptr=0x940,
        runtime=fixture.runtime,
    )
    kwargs.update(overrides)
    weights = kwargs.pop("weights", fixture.weights)
    return run_qwen4_exp_batch_gdn_token_mixer(fixture.mixed.ptr, weights, **kwargs)


def test_owner_allocates_zeroed_per_request_slabs(fixture: Fixture) -> None:
    state = fixture.state
    assert state.state_slots == SLOTS
    assert state.qkv_width == QKV_WIDTH
    assert state.core_width == CORE_WIDTH
    assert state.recurrent_state.nbytes == SLOTS * V_HEADS * HEAD_DIM * HEAD_DIM * 4
    assert state.conv_state.nbytes == SLOTS * QKV_WIDTH * KERNEL * 4
    assert state.recurrent_bytes_per_slot == V_HEADS * HEAD_DIM * HEAD_DIM * 4
    assert state.conv_bytes_per_slot == QKV_WIDTH * KERNEL * 4
    # Every slot is a request, so the slabs start as fresh zero state.
    assert {ptr for ptr, _ in fixture.runtime.memsets} == {
        state.recurrent_state.ptr,
        state.conv_state.ptr,
    }
    # Slot addresses are disjoint and stride by exactly one slot's bytes.
    pointers = [state.recurrent_slot_ptr(slot) for slot in range(SLOTS)]
    assert pointers == [
        state.recurrent_state.ptr + slot * state.recurrent_bytes_per_slot
        for slot in range(SLOTS)
    ]
    assert [state.conv_slot_ptr(slot) for slot in range(SLOTS)] == [
        state.conv_state.ptr + slot * state.conv_bytes_per_slot for slot in range(SLOTS)
    ]


def test_prepare_publishes_the_row_to_slot_mapping(fixture: Fixture) -> None:
    active = fixture.state.prepare([2, 0])
    assert active.rows == 2
    assert active.slot_ids == (2, 0)
    assert active.indices_host.tolist() == [2, 0]
    assert active.indices_host.dtype == np.int64
    assert active.indices.shape == (2,)
    assert active.indices.ptr == fixture.state.indices.ptr
    assert fixture.state.indices_host[:2].tolist() == [2, 0]
    # The vector is uploaded into the leading rows only.
    assert fixture.state.indices_host[2:].tolist() == [0, 0]
    assert fixture.state.active_rows == 2
    # The upload copies exactly one index per row: the buffer is slot-width, and
    # a whole-buffer copy would read past a shorter host vector.
    assert fixture.runtime.copies[-1] == (fixture.state.indices.ptr, 2 * 8)


def test_one_dispatch_per_stage_for_the_prepared_rows(fixture, recorder) -> None:
    active = fixture.state.prepare([2, 0])
    output = _run(fixture, recorder)

    assert recorder.names() == [
        "projection",
        "projection",
        "projection",
        "projection",
        "conv",
        "recurrence",
        "projection",
    ]
    assert output is fixture.scratch.output
    assert output.nbytes == fixture.scratch.output.nbytes

    projections = [args for name, args, _ in recorder.calls if name == "projection"]
    expected = [
        ("attn_qkv-weight", fixture.mixed.ptr, fixture.scratch.qkv.ptr, HIDDEN, QKV_WIDTH),
        ("attn_gate-weight", fixture.mixed.ptr, fixture.scratch.gate.ptr, HIDDEN, CORE_WIDTH),
        ("ssm_alpha-weight", fixture.mixed.ptr, fixture.scratch.alpha.ptr, HIDDEN, V_HEADS),
        ("ssm_beta-weight", fixture.mixed.ptr, fixture.scratch.beta.ptr, HIDDEN, V_HEADS),
        ("ssm_out-weight", fixture.scratch.core.ptr, fixture.scratch.output.ptr, CORE_WIDTH, HIDDEN),
    ]
    assert len(projections) == 5
    for args, (weight, in_ptr, out_ptr, in_features, out_features) in zip(
        projections, expected
    ):
        assert args[0] == weight
        assert args[1] == in_ptr
        assert args[2] == out_ptr
        assert args[3] == 2
        assert args[4] == in_features
        assert args[5] == out_features

    (conv_args, conv_kwargs) = recorder.one("conv")
    assert conv_args[0] == fixture.scratch.qkv.ptr
    assert conv_args[1] == fixture.state.conv_state.ptr
    assert conv_args[2] == 0x910
    assert conv_args[3] == fixture.scratch.conv.ptr
    assert conv_args[4] == active.indices.ptr
    assert conv_kwargs["rows"] == 2
    assert conv_kwargs["state_slots"] == SLOTS
    assert conv_kwargs["channels"] == QKV_WIDTH
    assert conv_kwargs["kernel_size"] == KERNEL

    (gdn_args, gdn_kwargs) = recorder.one("recurrence")
    assert gdn_args[0] == fixture.scratch.conv.ptr
    assert gdn_args[1] == fixture.scratch.gate.ptr
    assert gdn_args[2] == fixture.scratch.alpha.ptr
    assert gdn_args[3] == fixture.scratch.beta.ptr
    assert gdn_args[4] == 0x920
    assert gdn_args[5] == 0x930
    assert gdn_args[6] == 0x940
    assert gdn_args[7] == fixture.state.recurrent_state.ptr
    assert gdn_args[8] == fixture.scratch.core.ptr
    assert gdn_args[9] == active.indices.ptr
    assert gdn_kwargs["rows"] == 2
    assert gdn_kwargs["state_slots"] == SLOTS
    assert gdn_kwargs["num_k_heads"] == K_HEADS
    assert gdn_kwargs["num_v_heads"] == V_HEADS
    assert gdn_kwargs["head_k_dim"] == HEAD_DIM
    assert gdn_kwargs["head_v_dim"] == HEAD_DIM


def test_rows_one_matches_the_c1_dispatch_shape(fixture, recorder) -> None:
    active = fixture.state.prepare([3])
    _run(fixture, recorder, rows=1)
    assert recorder.names().count("projection") == 5
    (conv_args, conv_kwargs) = recorder.one("conv")
    (gdn_args, gdn_kwargs) = recorder.one("recurrence")
    assert conv_kwargs["rows"] == 1 and gdn_kwargs["rows"] == 1
    assert conv_args[4] == active.indices.ptr and gdn_args[9] == active.indices.ptr


def test_permuted_and_changing_active_sets_reuse_the_same_slabs(fixture, recorder) -> None:
    for slots in ([0, 1], [3, 0], [2], [1, 2, 3, 0]):
        active = fixture.state.prepare(slots)
        _run(fixture, recorder, rows=len(slots))
        # One dispatch per stage regardless of the row count: the mixer is not a
        # per-request step loop.
        assert recorder.names() == [
            "projection",
            "projection",
            "projection",
            "projection",
            "conv",
            "recurrence",
            "projection",
        ]
        (conv_args, conv_kwargs) = recorder.one("conv")
        assert conv_args[4] == active.indices.ptr
        assert conv_kwargs["rows"] == len(slots)
        assert active.indices_host.tolist() == slots
        recorder.calls.clear()
    # The slabs and the index buffer were allocated once, at owner construction.
    assert len(fixture.runtime.allocations) == 3 + 7


def test_rejected_calls_dispatch_nothing(fixture, recorder) -> None:
    fixture.state.prepare([0, 1])
    small_scratch = Qwen4ExpGDNScratch.allocate(
        rows=1,
        qkv_width=QKV_WIDTH,
        core_width=CORE_WIDTH,
        scalar_width=V_HEADS,
        hidden=HIDDEN,
        runtime=fixture.runtime,
    )
    other_runtime = FakeRuntime()
    cases = [
        (dict(rows=0), ValueError, "rows"),
        (dict(rows=3), ValueError, "does not match"),
        (dict(rows=SLOTS + 1), ValueError, "does not match"),
        (dict(head_dim=64), ValueError, "head_dim 128"),
        (dict(head_dim=HEAD_DIM, num_k_heads=1), ValueError, "match the state owner"),
        (dict(num_v_heads=6), ValueError, "match the state owner"),
        (dict(conv_kernel=8), ValueError, "match the state owner"),
        (dict(num_k_heads=3, num_v_heads=4), ValueError, "divisible"),
        (dict(hidden=0), ValueError, "hidden"),
        (dict(conv_weight_ptr=0), ValueError, "conv_weight_ptr"),
        (dict(norm_weight_ptr=0), ValueError, "norm_weight_ptr"),
        (dict(scratch=small_scratch), ValueError, "scratch"),
        (dict(runtime=other_runtime), ValueError, "runtime"),
        (dict(weights={}), ValueError, "missing"),
        (
            dict(
                weights={
                    name: SimpleNamespace(
                        spec=SimpleNamespace(
                            source=SimpleNamespace(shape=(1, 1))
                        )
                    )
                    for name in batch_gdn._REQUIRED_PROJECTIONS
                }
            ),
            ValueError,
            "attn_qkv weight shape",
        ),
    ]
    try:
        for overrides, error, match in cases:
            with pytest.raises(error, match=match):
                _run(fixture, recorder, **overrides)
            assert recorder.calls == [], overrides
    finally:
        small_scratch.close()


def test_prepare_rejects_duplicate_and_out_of_range_slots(fixture, recorder) -> None:
    before = fixture.state.indices_host.copy()
    cases = [
        ([], ValueError, "at least one"),
        ([0, 0], ValueError, "duplicate slot 0"),
        ([SLOTS], ValueError, "outside the declared state slab"),
        ([-1], ValueError, "non-negative"),
        ([0, 1, 2, 3, 0], ValueError, "at most"),
        (["a"], TypeError, "integer"),
        ((0, 1.5), TypeError, "integer"),
    ]
    for slots, error, match in cases:
        with pytest.raises(error, match=match):
            fixture.state.prepare(slots)
        assert fixture.state.active is None
        assert fixture.state.indices_host.tolist() == before.tolist()
    assert recorder.calls == []


def test_unprepared_closed_and_closing_states_are_rejected(fixture, recorder) -> None:
    with pytest.raises(RuntimeError, match="no active rows"):
        _run(fixture, recorder)
    assert recorder.calls == []

    fixture.state.prepare([0, 1])
    fixture.scratch.close()
    with pytest.raises(RuntimeError, match="closed"):
        _run(fixture, recorder)
    assert recorder.calls == []


def test_closing_state_is_rejected_before_dispatch(fixture, recorder) -> None:
    fixture.state.prepare([0, 1])
    fixture.state.closing = True
    with pytest.raises(RuntimeError, match="closing"):
        _run(fixture, recorder)
    assert recorder.calls == []


def test_tampered_index_vector_is_re_admitted_before_dispatch(fixture, recorder) -> None:
    fixture.state.prepare([0, 1])
    # A caller that rewrote the published rows would otherwise hand the kernels a
    # duplicate slot, which both primitives reject only as a whole batch.
    fixture.state.active = type(fixture.state.active)(
        rows=2,
        slot_ids=(0, 1),
        indices=fixture.state.active.indices,
        indices_host=np.asarray([0, 0], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="duplicate slot 0"):
        _run(fixture, recorder)
    assert recorder.calls == []


def test_close_frees_every_buffer_once_and_retries_a_partial_failure() -> None:
    runtime = FakeRuntime()
    state = Qwen4ExpBatchGdnState.allocate(
        state_slots=2,
        num_k_heads=K_HEADS,
        num_v_heads=V_HEADS,
        head_dim=HEAD_DIM,
        conv_kernel=KERNEL,
        runtime=runtime,
    )
    owned = [state.recurrent_state.ptr, state.conv_state.ptr, state.indices.ptr]
    assert sorted(runtime.allocations) == sorted(owned)

    # Fail the first free, as a driver-level release failure would.
    original_free = runtime.free
    failures = [1]

    def flaky_free(ptr: int) -> None:
        if failures:
            failures.pop()
            raise RuntimeError("device free failed")
        original_free(ptr)

    runtime.free = flaky_free
    with pytest.raises(RuntimeError, match="device free failed"):
        state.close()
    assert state.closed is False
    assert state.closing is True
    # Uploads are rejected once cleanup has begun, even though nothing was freed.
    with pytest.raises(RuntimeError, match="closing"):
        state.prepare([0])
    assert sorted(runtime.allocations) == sorted(owned)

    state.close()
    assert state.closed is True
    assert sorted(runtime.freed) == sorted(owned)
    assert runtime.allocations == {}
    # A second close is a no-op, not a double free.
    state.close()
    assert sorted(runtime.freed) == sorted(owned)


def test_owner_rejects_geometry_the_primitives_cannot_address() -> None:
    runtime = FakeRuntime()
    cases = [
        (dict(state_slots=0), ValueError, "state_slots"),
        (dict(state_slots=QWEN4_EXP_GDN_INDEXED_MAX_STATE_SLOTS + 1), ValueError, "slab width"),
        (dict(num_v_heads=3, num_k_heads=2), ValueError, "divisible"),
        (dict(conv_kernel=QWEN4_EXP_CONV_INDEXED_MAX_KERNEL_SIZE + 1), ValueError, "kernel bound"),
        (dict(head_dim=0), ValueError, "head_dim"),
    ]
    for overrides, error, match in cases:
        kwargs = dict(
            state_slots=2,
            num_k_heads=K_HEADS,
            num_v_heads=V_HEADS,
            head_dim=HEAD_DIM,
            conv_kernel=KERNEL,
            runtime=runtime,
        )
        kwargs.update(overrides)
        with pytest.raises(error, match=match):
            Qwen4ExpBatchGdnState.allocate(**kwargs)
    # Nothing was allocated by any rejected call.
    assert runtime.allocations == {}


def test_slot_pointers_reject_a_slot_outside_the_slab(fixture: Fixture) -> None:
    with pytest.raises(ValueError, match="outside"):
        fixture.state.recurrent_slot_ptr(SLOTS)
    with pytest.raises(ValueError, match="outside"):
        fixture.state.conv_slot_ptr(-1)
    with pytest.raises(TypeError, match="integer"):
        fixture.state.conv_slot_ptr(True)


# Imported late so the module-level constant names above stay readable.
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (  # noqa: E402
    QWEN4_EXP_CONV_INDEXED_MAX_KERNEL_SIZE,
)
from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (  # noqa: E402
    QWEN4_EXP_GDN_INDEXED_MAX_STATE_SLOTS,
)
