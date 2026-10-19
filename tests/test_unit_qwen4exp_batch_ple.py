"""Dispatch-shape and admission tests for the N-row PLE stage.

Every device call in the stage is a module-level name (or the resolved Conv
route), so these tests replace them with recorders: the stage's own control flow
(row count per dispatch, dispatch order, which row-to-slot vector reaches the
indexed Conv, and that a rejected call dispatches nothing at all) is asserted
without a GPU.

The state owner is exercised here too, because its admission rules are what make
the stage's row/slot contract checkable: unique slots, permuted and shrinking
active sets, and the closing/closed lifecycle.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import hipengine.runtime.qwen4_exp_batch_ple as batch_ple
from hipengine.core.memory import DeviceBuffer
from hipengine.kernels.registry import KernelKey
from hipengine.runtime.qwen4_exp_batch_ple import (
    Qwen4ExpBatchPleState,
    run_qwen4_exp_batch_ple_stage,
)
from hipengine.runtime.qwen4_exp_runner import Qwen4ExpPLEScratch

SLOTS = 4
BRANCHES = 2
HIDDEN = 4
CHANNELS = BRANCHES * HIDDEN
KERNEL = 4
DILATION = 3
HISTORY_ROWS = (KERNEL - 1) * DILATION


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
    monkeypatch.setattr(batch_ple, "launch_gguf_linear", record.stage("projection"))
    for name in (
        "qwen4_exp_grouped_rmsnorm_f32",
        "qwen4_exp_grouped_rmsnorm_bf16_f32",
        "qwen4_exp_ple_signed_sqrt_gate_f32",
        "qwen4_exp_ple_repeat_gated_value_f32",
        "qwen4_exp_scaled_silu_f32",
        "qwen4_exp_ple_add_delta_bf16_f32",
    ):
        monkeypatch.setattr(batch_ple, name, record.stage(name.split("qwen4_exp_")[-1]))
    # The Conv route is resolved per backend; the recorder stands in for it.
    monkeypatch.setattr(
        batch_ple, "_resolve_indexed_conv", lambda weights: record.stage("conv")
    )
    return record


class Fixture:
    def __init__(self, *, scratch_rows: int = SLOTS) -> None:
        self.runtime = FakeRuntime()
        self.state = Qwen4ExpBatchPleState.allocate(
            state_slots=SLOTS,
            branches=BRANCHES,
            hidden=HIDDEN,
            conv_kernel=KERNEL,
            dilation=DILATION,
            runtime=self.runtime,
        )
        self.scratch = Qwen4ExpPLEScratch.allocate(
            rows=scratch_rows,
            branches=BRANCHES,
            hidden=HIDDEN,
            runtime=self.runtime,
        )
        self.weights = {name: f"{name}-weight" for name in batch_ple._REQUIRED_PROJECTIONS}
        self.residual = DeviceBuffer(ptr=0x1000, nbytes=SLOTS * CHANNELS * 2)
        self.embedding = DeviceBuffer(ptr=0x2000, nbytes=SLOTS * HIDDEN * 4)

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


def test_split_stages_keep_history_updates_out_of_shared_projections(fixture, recorder):
    fixture.state.prepare((3, 1))
    expected_output = _run(fixture, recorder)
    original = list(recorder.calls)
    recorder.calls.clear()
    plan = batch_ple.validate_qwen4_exp_batch_ple_stage(
        fixture.weights, state=fixture.state, scratch=fixture.scratch,
        residual_ptr=fixture.residual.ptr, embedding_ptr=fixture.embedding.ptr,
        norm_key_ptr=0x910, norm_query_ptr=0x920, norm_conv_ptr=0x930,
        conv_weight_ptr=0x940, rows=2, branches=BRANCHES, hidden=HIDDEN,
        conv_kernel=KERNEL, dilation=DILATION, runtime=fixture.runtime,
    )
    assert recorder.calls == []
    batch_ple.run_qwen4_exp_batch_ple_prepare(
        plan, fixture.residual.ptr, fixture.embedding.ptr, fixture.weights,
        scratch=fixture.scratch, norm_key_ptr=0x910, norm_query_ptr=0x920,
        norm_conv_ptr=0x930,
    )
    assert 'conv' not in recorder.names()
    batch_ple.run_qwen4_exp_batch_ple_convolution(
        plan, state=fixture.state, scratch=fixture.scratch, conv_weight_ptr=0x940,
    )
    actual = batch_ple.run_qwen4_exp_batch_ple_finish(
        plan, fixture.residual.ptr, scratch=fixture.scratch,
    )
    assert actual is expected_output
    assert recorder.calls == original
    assert recorder.names().count('conv') == 1


def _run(fixture: Fixture, recorder: Recorder, **overrides) -> DeviceBuffer:
    kwargs = dict(
        state=fixture.state,
        scratch=fixture.scratch,
        norm_key_ptr=0x910,
        norm_query_ptr=0x920,
        norm_conv_ptr=0x930,
        conv_weight_ptr=0x940,
        rows=2,
        branches=BRANCHES,
        hidden=HIDDEN,
        conv_kernel=KERNEL,
        dilation=DILATION,
        runtime=fixture.runtime,
    )
    kwargs.update(overrides)
    weights = kwargs.pop("weights", fixture.weights)
    residual_ptr = kwargs.pop("residual_ptr", fixture.residual.ptr)
    embedding_ptr = kwargs.pop("embedding_ptr", fixture.embedding.ptr)
    return run_qwen4_exp_batch_ple_stage(
        residual_ptr, embedding_ptr, weights, **kwargs
    )


def test_owner_allocates_one_zeroed_history_slab(fixture: Fixture) -> None:
    state = fixture.state
    assert state.channels == CHANNELS
    assert state.history_rows == HISTORY_ROWS
    assert state.history.nbytes == SLOTS * HISTORY_ROWS * CHANNELS * 4
    assert state.history_bytes_per_slot == HISTORY_ROWS * CHANNELS * 4
    assert fixture.runtime.memsets == [(state.history.ptr, state.history.nbytes)]
    assert [state.history_slot_ptr(slot) for slot in range(SLOTS)] == [
        state.history.ptr + slot * state.history_bytes_per_slot for slot in range(SLOTS)
    ]


def test_prepare_publishes_the_row_to_slot_mapping(fixture: Fixture) -> None:
    active = fixture.state.prepare([3, 1])
    assert active.rows == 2
    assert active.slot_ids == (3, 1)
    assert active.indices_host.tolist() == [3, 1]
    assert active.indices.shape == (2,)
    assert active.indices.ptr == fixture.state.indices.ptr
    assert fixture.state.indices_host[:2].tolist() == [3, 1]
    assert fixture.state.indices_host[2:].tolist() == [0, 0]
    # The upload copies exactly one index per row: the buffer is slot-width, and
    # a whole-buffer copy would read past a shorter host vector.
    assert fixture.runtime.copies[-1] == (fixture.state.indices.ptr, 2 * 8)


def test_one_dispatch_per_stage_for_the_prepared_rows(fixture, recorder) -> None:
    active = fixture.state.prepare([3, 1])
    output = _run(fixture, recorder)

    assert recorder.names() == [
        "projection",
        "projection",
        "grouped_rmsnorm_f32",
        "grouped_rmsnorm_bf16_f32",
        "ple_signed_sqrt_gate_f32",
        "ple_repeat_gated_value_f32",
        "grouped_rmsnorm_f32",
        "conv",
        "scaled_silu_f32",
        "ple_add_delta_bf16_f32",
    ]
    assert output is fixture.scratch.output
    assert output.nbytes == fixture.scratch.output.nbytes

    projections = [args for name, args, _ in recorder.calls if name == "projection"]
    expected = [
        ("ple_key-weight", fixture.embedding.ptr, fixture.scratch.key.ptr, HIDDEN, CHANNELS),
        ("ple_value-weight", fixture.embedding.ptr, fixture.scratch.value.ptr, HIDDEN, HIDDEN),
    ]
    assert len(projections) == 2
    for args, (weight, in_ptr, out_ptr, in_features, out_features) in zip(
        projections, expected
    ):
        assert args[0] == weight
        assert args[1] == in_ptr
        assert args[2] == out_ptr
        assert args[3] == 2
        assert args[4] == in_features
        assert args[5] == out_features

    # The Conv route is the only stage that touches request-owned state.
    (conv_args, conv_kwargs) = recorder.one("conv")
    assert conv_args[0] == fixture.scratch.normalized.ptr
    assert conv_args[1] == 0x940
    assert conv_args[2] == fixture.state.history.ptr
    assert conv_args[3] == fixture.scratch.conv.ptr
    assert conv_args[4] == active.indices.ptr
    assert conv_kwargs["rows"] == 2
    assert conv_kwargs["state_slots"] == SLOTS
    assert conv_kwargs["channels"] == CHANNELS
    assert conv_kwargs["kernel_size"] == KERNEL
    assert conv_kwargs["dilation"] == DILATION

    # Elementwise stages carry the packed element count, not a per-row loop.
    (silu_args, _) = recorder.one("scaled_silu_f32")
    assert silu_args[0] == silu_args[1] == fixture.scratch.conv.ptr
    assert silu_args[2] == 2 * CHANNELS
    (delta_args, _) = recorder.one("ple_add_delta_bf16_f32")
    assert delta_args[0] == fixture.residual.ptr
    assert delta_args[1] == fixture.scratch.gated_value.ptr
    assert delta_args[2] == fixture.scratch.conv.ptr
    assert delta_args[3] == fixture.scratch.output.ptr
    assert delta_args[4] == 2 * CHANNELS


def test_rows_one_matches_the_c1_dispatch_shape(fixture, recorder) -> None:
    active = fixture.state.prepare([2])
    _run(fixture, recorder, rows=1)
    assert recorder.names().count("projection") == 2
    (conv_args, conv_kwargs) = recorder.one("conv")
    assert conv_kwargs["rows"] == 1
    assert conv_args[4] == active.indices.ptr


def test_permuted_and_changing_active_sets_reuse_the_same_slab(fixture, recorder) -> None:
    for slots in ([0, 1], [3, 0], [2], [1, 2, 3, 0]):
        active = fixture.state.prepare(slots)
        _run(fixture, recorder, rows=len(slots))
        # One dispatch per stage regardless of the row count: the stage is not a
        # per-request step loop.
        assert len(recorder.calls) == 10
        (conv_args, conv_kwargs) = recorder.one("conv")
        assert conv_args[4] == active.indices.ptr
        assert conv_kwargs["rows"] == len(slots)
        assert active.indices_host.tolist() == slots
        recorder.calls.clear()
    # The slab and the index buffer were allocated once, at owner construction.
    assert len(fixture.runtime.allocations) == 2 + 8


def test_rejected_calls_dispatch_nothing(fixture, recorder) -> None:
    fixture.state.prepare([0, 1])
    small_scratch = Qwen4ExpPLEScratch.allocate(
        rows=1, branches=BRANCHES, hidden=HIDDEN, runtime=fixture.runtime
    )
    other_runtime = FakeRuntime()
    cases = [
        (dict(rows=0), ValueError, "rows"),
        (dict(rows=3), ValueError, "does not match"),
        (dict(rows=SLOTS + 1), ValueError, "does not match"),
        (dict(branches=3), ValueError, "match the state owner"),
        (dict(hidden=HIDDEN + 1), ValueError, "match the state owner"),
        (dict(conv_kernel=KERNEL + 1), ValueError, "match the state owner"),
        (dict(dilation=DILATION + 1), ValueError, "match the state owner"),
        (dict(eps=0.0), ValueError, "eps"),
        (dict(eps=float("nan")), ValueError, "eps"),
        (dict(residual_ptr=0), ValueError, "residual_ptr"),
        (dict(embedding_ptr=0), ValueError, "embedding_ptr"),
        (dict(conv_weight_ptr=0), ValueError, "conv_weight_ptr"),
        (dict(scratch=small_scratch), ValueError, "scratch"),
        (dict(runtime=other_runtime), ValueError, "runtime"),
        (dict(weights={"ple_key": "k"}), ValueError, "missing"),
        (
            dict(
                weights={
                    "ple_key": SimpleNamespace(
                        spec=SimpleNamespace(source=SimpleNamespace(shape=(1, 1)))
                    ),
                    "ple_value": SimpleNamespace(
                        spec=SimpleNamespace(source=SimpleNamespace(shape=(HIDDEN, HIDDEN)))
                    ),
                }
            ),
            ValueError,
            "ple_key weight shape",
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
        ([1, 1], ValueError, "duplicate slot 1"),
        ([SLOTS], ValueError, "outside the declared state slab"),
        ([-1], ValueError, "non-negative"),
        ([0, 1, 2, 3, 0], ValueError, "at most"),
        (["a"], TypeError, "integer"),
    ]
    for slots, error, match in cases:
        with pytest.raises(error, match=match):
            fixture.state.prepare(slots)
        assert fixture.state.active is None
        assert fixture.state.indices_host.tolist() == before.tolist()
    assert recorder.calls == []


def test_tampered_index_vector_is_re_admitted_before_dispatch(fixture, recorder) -> None:
    fixture.state.prepare([0, 1])
    # A caller that rewrote the published rows would otherwise hand the Conv a
    # duplicate slot, which the primitive rejects only as a whole batch.
    fixture.state.active = type(fixture.state.active)(
        rows=2,
        slot_ids=(0, 1),
        indices=fixture.state.active.indices,
        indices_host=np.asarray([0, 0], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="duplicate slot 0"):
        _run(fixture, recorder)
    assert recorder.calls == []


def test_unprepared_closed_and_closing_states_are_rejected(fixture, recorder) -> None:
    with pytest.raises(RuntimeError, match="no active rows"):
        _run(fixture, recorder)
    assert recorder.calls == []

    fixture.state.prepare([0, 1])
    fixture.state.closing = True
    with pytest.raises(RuntimeError, match="closing"):
        _run(fixture, recorder)
    assert recorder.calls == []


def test_conv_route_is_resolved_exactly_and_reports_a_missing_variant(
    monkeypatch,
) -> None:
    weights = {"ple_key": SimpleNamespace(backend="hip_test_backend")}
    seen: list[str] = []
    monkeypatch.setattr(
        batch_ple, "load_backend_kernel_package", lambda backend: seen.append(backend)
    )
    monkeypatch.setattr(batch_ple, "is_registered", lambda key: False)
    with pytest.raises(RuntimeError, match="is not registered for backend"):
        batch_ple._resolve_indexed_conv(weights)
    assert seen == ["hip_test_backend"]

    # With the exact variant registered the route is the resolved kernel.
    monkeypatch.setattr(batch_ple, "is_registered", lambda key: True)
    monkeypatch.setattr(batch_ple, "resolve", lambda **kwargs: ("resolved", kwargs))
    assert batch_ple._resolve_indexed_conv(weights) == (
        "resolved",
        {
            "backend": "hip_test_backend",
            "layer": "ple_dilated_depthwise_conv",
            "quant": "f32",
            "variant": "qwen4exp_ple_indexed_decode",
        },
    )


def test_close_frees_every_buffer_once_and_retries_a_partial_failure() -> None:
    runtime = FakeRuntime()
    state = Qwen4ExpBatchPleState.allocate(
        state_slots=2,
        branches=BRANCHES,
        hidden=HIDDEN,
        conv_kernel=KERNEL,
        dilation=DILATION,
        runtime=runtime,
    )
    owned = [state.history.ptr, state.indices.ptr]
    assert sorted(runtime.allocations) == sorted(owned)

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
    with pytest.raises(RuntimeError, match="closing"):
        state.prepare([0])
    assert sorted(runtime.allocations) == sorted(owned)

    state.close()
    assert state.closed is True
    assert sorted(runtime.freed) == sorted(owned)
    state.close()
    assert sorted(runtime.freed) == sorted(owned)


def test_owner_rejects_geometry_the_indexed_decode_cannot_address() -> None:
    runtime = FakeRuntime()
    cases = [
        (dict(state_slots=0), ValueError, "state_slots"),
        (
            dict(state_slots=batch_ple.QWEN4_EXP_PLE_INDEXED_MAX_STATE_SLOTS + 1),
            ValueError,
            "slab width",
        ),
        (dict(hidden=0), ValueError, "hidden"),
        (
            dict(conv_kernel=batch_ple.QWEN4_EXP_PLE_INDEXED_MAX_KERNEL_SIZE + 1),
            ValueError,
            "kernel bound",
        ),
        (
            dict(dilation=batch_ple.QWEN4_EXP_PLE_INDEXED_MAX_DILATION + 1),
            ValueError,
            "dilation bound",
        ),
    ]
    for overrides, error, match in cases:
        kwargs = dict(
            state_slots=2,
            branches=BRANCHES,
            hidden=HIDDEN,
            conv_kernel=KERNEL,
            dilation=DILATION,
            runtime=runtime,
        )
        kwargs.update(overrides)
        with pytest.raises(error, match=match):
            Qwen4ExpBatchPleState.allocate(**kwargs)
    assert runtime.allocations == {}


def test_history_slot_pointer_rejects_a_slot_outside_the_slab(fixture: Fixture) -> None:
    with pytest.raises(ValueError, match="outside"):
        fixture.state.history_slot_ptr(SLOTS)
    with pytest.raises(ValueError, match="outside"):
        fixture.state.history_slot_ptr(-1)
    with pytest.raises(TypeError, match="integer"):
        fixture.state.history_slot_ptr(True)
