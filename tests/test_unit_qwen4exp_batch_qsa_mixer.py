"""Dispatch-shape and admission tests for the N-row dense QSA token mixer.

Every device call in the mixer is a module-level name, so these tests replace
them with recorders: the mixer's *own* control flow (row count per dispatch,
dispatch order, which metadata vector reaches which kernel, and that a rejected
call dispatches nothing at all) is asserted without a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.runtime.qwen4_exp_batch_qsa as batch_qsa
from hipengine.core.memory import DeviceBuffer
from hipengine.runtime.qwen4_exp_batch_qsa import (
    run_qwen4_exp_batch_dense_qsa_token_mixer,
)
from hipengine.runtime.qwen4_exp_batch_state import Qwen4ExpBatchAttentionState
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpQSAMixerDeviceWeights,
    Qwen4ExpQSAScratch,
)

SLOTS = 4
MAX_POSITIONS = 1024
BLOCK_SIZE = 256
HIDDEN = 8
QUERY_HEADS = 2
KV_HEADS = 1
HEAD_DIM = 4
ROTARY_DIM = 4
THETA = 100.0
Q_WIDTH = QUERY_HEADS * HEAD_DIM
KV_WIDTH = KV_HEADS * HEAD_DIM


class FakeRuntime:
    def __init__(self) -> None:
        self.next_ptr = 0x4000
        self.allocations: dict[int, int] = {}
        self.freed: list[int] = []

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

    def memcpy(self, dst: int, src: int, nbytes: int, kind) -> None:
        assert dst in self.allocations


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
    monkeypatch.setattr(batch_qsa, "launch_gguf_linear", record.stage("projection"))
    monkeypatch.setattr(
        batch_qsa, "qwen4_exp_qsa_split_norm_rope_rows_f32", record.stage("split_rope")
    )
    monkeypatch.setattr(
        batch_qsa,
        "qwen4_exp_qsa_split_norm_mrope_rows_f32",
        record.stage("split_mrope"),
    )
    monkeypatch.setattr(
        batch_qsa, "qwen35_write_paged_kv_f32_batch_spans", record.stage("kv_write")
    )
    monkeypatch.setattr(
        batch_qsa,
        "qwen35_paged_full_attn_decode_context_bf16_batch_spans",
        record.stage("attention"),
    )
    monkeypatch.setattr(
        batch_qsa, "qwen4_exp_qsa_gate_context_f32", record.stage("gate")
    )
    return record


class Fixture:
    def __init__(self, *, scratch_rows: int = SLOTS) -> None:
        self.runtime = FakeRuntime()
        self.state = Qwen4ExpBatchAttentionState.allocate(
            state_slots=SLOTS,
            max_positions=MAX_POSITIONS,
            block_size=BLOCK_SIZE,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            runtime=self.runtime,
        )
        self.scratch = Qwen4ExpQSAScratch.allocate(
            rows=scratch_rows,
            hidden=HIDDEN,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            runtime=self.runtime,
        )
        self.weights = Qwen4ExpQSAMixerDeviceWeights(
            projections={name: f"{name}-weight" for name in batch_qsa._REQUIRED_PROJECTIONS},
            q_norm_weight_ptr=0x900,
            k_norm_weight_ptr=0x904,
        )
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
        attention_state=fixture.state,
        scratch=fixture.scratch,
        rows=2,
        hidden=HIDDEN,
        query_heads=QUERY_HEADS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        theta=THETA,
        dense_equivalent_limit=MAX_POSITIONS,
        runtime=fixture.runtime,
    )
    kwargs.update(overrides)
    weights = kwargs.pop("weights", fixture.weights)
    return run_qwen4_exp_batch_dense_qsa_token_mixer(
        fixture.mixed.ptr, weights, **kwargs
    )


def _shaped_weight(shape: tuple[int, int]) -> SimpleNamespace:
    """A weight stand-in that exposes the stored GGUF tensor shape."""

    return SimpleNamespace(spec=SimpleNamespace(source=SimpleNamespace(shape=shape)))


def _shaped_weights(**shapes: tuple[int, int]) -> Qwen4ExpQSAMixerDeviceWeights:
    geometry = {
        "attn_q": (Q_WIDTH * 2, HIDDEN),
        "attn_k": (KV_WIDTH, HIDDEN),
        "attn_v": (KV_WIDTH, HIDDEN),
        "attn_output": (HIDDEN, Q_WIDTH),
    }
    geometry.update(shapes)
    return Qwen4ExpQSAMixerDeviceWeights(
        projections={name: _shaped_weight(shape) for name, shape in geometry.items()},
        q_norm_weight_ptr=0x900,
        k_norm_weight_ptr=0x904,
    )


def test_one_dispatch_per_stage_for_the_prepared_rows(fixture, recorder) -> None:
    active = fixture.state.prepare([2, 0], [255, 256])
    output = _run(fixture, recorder)

    # Projections, split+RoPE, KV write, attention, gate, output projection:
    # four projection launches for three input slots plus attn_output.
    assert recorder.names() == [
        "projection",
        "projection",
        "projection",
        "split_rope",
        "kv_write",
        "attention",
        "gate",
        "projection",
    ]
    assert output is fixture.scratch.output
    assert output.nbytes == fixture.scratch.output.nbytes

    projections = [args for name, args, _ in recorder.calls if name == "projection"]
    expected_slots = [
        ("attn_q-weight", fixture.mixed.ptr, fixture.scratch.q_projected.ptr, HIDDEN, Q_WIDTH * 2),
        ("attn_k-weight", fixture.mixed.ptr, fixture.scratch.key_projected.ptr, HIDDEN, KV_WIDTH),
        ("attn_v-weight", fixture.mixed.ptr, fixture.scratch.value_projected.ptr, HIDDEN, KV_WIDTH),
        ("attn_output-weight", fixture.scratch.gated.ptr, fixture.scratch.output.ptr, Q_WIDTH, HIDDEN),
    ]
    assert len(projections) == 4
    for args, (weight, in_ptr, out_ptr, in_features, out_features) in zip(
        projections, expected_slots
    ):
        assert args[0] == weight
        assert args[1] == in_ptr
        assert args[2] == out_ptr
        assert args[3] == 2  # rows
        assert args[4] == in_features
        assert args[5] == out_features
    for name, _, kwargs in recorder.calls:
        if name != "projection":
            continue
        assert kwargs["activation_dtype"] == batch_qsa.GGUF_ACTIVATION_F32
        assert kwargs["output_dtype"] == batch_qsa.GGUF_OUTPUT_F32
        assert kwargs["runtime"] is fixture.runtime

    (split_args, split_kwargs) = recorder.one("split_rope")
    assert split_args[4] == active.position.ptr  # positions uploaded by prepare
    assert split_args[5] == fixture.scratch.query.ptr
    assert split_args[6] == fixture.scratch.key.ptr
    assert split_args[7] == fixture.scratch.gate.ptr
    assert split_args[8] == 2  # rows
    assert split_args[9:13] == (QUERY_HEADS, KV_HEADS, HEAD_DIM, ROTARY_DIM)

    (write_args, _) = recorder.one("kv_write")
    assert write_args[2] == fixture.state.key_cache.ptr
    assert write_args[3] == fixture.state.value_cache.ptr
    assert write_args[4] is active.append_spans
    assert write_args[5] == 2  # rows
    assert write_args[6] == BLOCK_SIZE

    (attn_args, _) = recorder.one("attention")
    assert attn_args[3] == fixture.scratch.context.ptr
    assert attn_args[4] is active.decode_spans
    assert attn_args[5] == 2  # rows
    assert attn_args[6] == active.max_context_len == 257
    assert attn_args[7] == BLOCK_SIZE
    assert attn_args[8:11] == (QUERY_HEADS, KV_HEADS, HEAD_DIM)
    assert attn_args[11] == pytest.approx(HEAD_DIM**-0.5)

    (gate_args, _) = recorder.one("gate")
    assert gate_args[0] == fixture.scratch.context.ptr
    assert gate_args[1] == fixture.scratch.gate.ptr
    assert gate_args[2] == fixture.scratch.gated.ptr
    assert gate_args[3] == 2 * Q_WIDTH

    # The mixer never re-uploads metadata: the prepared rows are exactly what it
    # consumed, in the order prepare published.
    assert fixture.state.position_host[:2].tolist() == [255, 256]
    assert fixture.state.block_table_host[:2].tolist() == [
        fixture.state.block_host[2].tolist(),
        fixture.state.block_host[0].tolist(),
    ]
    assert fixture.state.active_rows == 2


def test_rows_one_matches_the_c1_dispatch_shape(fixture, recorder) -> None:
    active = fixture.state.prepare([1], [0])
    _run(fixture, recorder, rows=1)
    assert recorder.names().count("projection") == 4
    for name, args, _ in recorder.calls:
        if name in {"kv_write", "attention"}:
            assert args[5 if name == "kv_write" else 5] == 1
    (split_args, _) = recorder.one("split_rope")
    assert split_args[8] == 1
    assert split_args[4] == active.position.ptr


def test_mrope_positions_select_the_mrope_rows_kernel(fixture, recorder) -> None:
    active = fixture.state.prepare([0, 1], [7, 9])
    rope_positions = DeviceBuffer(ptr=0x2000, nbytes=3 * 2 * 8)
    _run(fixture, recorder, rope_positions_ptr=rope_positions.ptr)
    assert "split_mrope" in recorder.names()
    assert "split_rope" not in recorder.names()
    (args, _) = recorder.one("split_mrope")
    assert args[4] == rope_positions.ptr
    assert args[8] == 2
    # The plain path is untouched by the override.
    assert active.position.ptr != rope_positions.ptr


def test_context_limit_override_only_widens_the_launch_bound(fixture, recorder) -> None:
    active = fixture.state.prepare([0, 1], [10, 20])
    _run(fixture, recorder, attention_context_limit=MAX_POSITIONS)
    (attn_args, _) = recorder.one("attention")
    assert attn_args[6] == MAX_POSITIONS
    assert attn_args[4] is active.decode_spans


def test_rejected_calls_dispatch_nothing(fixture, recorder) -> None:
    fixture.state.prepare([0, 1], [10, 20])
    small_scratch = Qwen4ExpQSAScratch.allocate(
        rows=1,
        hidden=HIDDEN,
        query_heads=QUERY_HEADS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        runtime=fixture.runtime,
    )
    other_runtime = FakeRuntime()
    cases = [
        (dict(rows=0), ValueError, "rows"),
        (dict(rows=3), ValueError, "does not match"),
        (dict(rows=SLOTS + 1), ValueError, "does not match"),
        (dict(kv_heads=2), ValueError, "head geometry"),
        (dict(head_dim=8), ValueError, "head geometry"),
        (dict(query_heads=3, kv_heads=2), ValueError, "divisible"),
        (dict(rotary_dim=3), ValueError, "rotary_dim"),
        (dict(rotary_dim=HEAD_DIM + 2), ValueError, "rotary_dim"),
        (dict(theta=0.0), ValueError, "theta"),
        (dict(theta=-1.0), ValueError, "theta"),
        (dict(theta=float("nan")), ValueError, "theta"),
        (dict(theta=float("inf")), ValueError, "theta"),
        (dict(eps=0.0), ValueError, "eps"),
        (dict(eps=float("nan")), ValueError, "eps"),
        (dict(dense_equivalent_limit=0), ValueError, "dense_equivalent_limit"),
        (dict(dense_equivalent_limit=20), ValueError, "dense-equivalent"),
        (dict(rope_positions_ptr=0), ValueError, "rope_positions_ptr"),
        (dict(rope_positions_ptr=0x4004), ValueError, "aligned"),
        (dict(weights=_shaped_weights(attn_k=(KV_WIDTH, HIDDEN + 1))), ValueError, "attn_k"),
        (
            dict(weights=_shaped_weights(attn_output=(HIDDEN, Q_WIDTH + 1))),
            ValueError,
            "attn_output",
        ),
        (dict(hidden=0), ValueError, "hidden"),
        (dict(scratch=small_scratch), ValueError, "scratch"),
        (dict(attention_context_limit=10), ValueError, "does not cover"),
        (dict(attention_context_limit=MAX_POSITIONS + 1), ValueError, "capacity"),
        (
            dict(weights=Qwen4ExpQSAMixerDeviceWeights(projections={}, q_norm_weight_ptr=1, k_norm_weight_ptr=1)),
            ValueError,
            "missing",
        ),
        (
            dict(
                weights=Qwen4ExpQSAMixerDeviceWeights(
                    projections=fixture.weights.projections,
                    q_norm_weight_ptr=0,
                    k_norm_weight_ptr=1,
                )
            ),
            ValueError,
            "norm weights",
        ),
        (dict(runtime=other_runtime), ValueError, "runtime"),
    ]
    try:
        for overrides, error, match in cases:
            with pytest.raises(error, match=match):
                _run(fixture, recorder, **overrides)
            assert recorder.calls == [], overrides
    finally:
        small_scratch.close()


def test_dense_equivalent_limit_bounds_every_prepared_row(fixture, recorder) -> None:
    # Contexts are 11 and 21; a limit of 21 admits both, 20 rejects row 1.
    fixture.state.prepare([0, 1], [10, 20])
    _run(fixture, recorder, dense_equivalent_limit=21)
    assert recorder.names() == [
        "projection",
        "projection",
        "projection",
        "split_rope",
        "kv_write",
        "attention",
        "gate",
        "projection",
    ]
    recorder.calls.clear()
    with pytest.raises(ValueError, match=r"row 1 context 21 exceeds"):
        _run(fixture, recorder, dense_equivalent_limit=20)
    assert recorder.calls == []


def test_projection_geometry_preflight_admits_matching_shapes(fixture, recorder) -> None:
    fixture.state.prepare([0, 1], [10, 20])
    output = _run(fixture, recorder, weights=_shaped_weights())
    assert output is fixture.scratch.output
    assert recorder.names().count("projection") == 4


def test_head_dim_beyond_the_dense_decode_bound_is_rejected_before_the_kv_write(
    recorder,
) -> None:
    # The decode wrapper rejects head_dim > 256 only after the KV write, so the
    # mixer must reject a fully matching 512-wide state/scratch up front.
    runtime = FakeRuntime()
    state = Qwen4ExpBatchAttentionState.allocate(
        state_slots=2,
        max_positions=512,
        block_size=BLOCK_SIZE,
        kv_heads=KV_HEADS,
        head_dim=512,
        runtime=runtime,
    )
    scratch = Qwen4ExpQSAScratch.allocate(
        rows=2,
        hidden=HIDDEN,
        query_heads=QUERY_HEADS,
        kv_heads=KV_HEADS,
        head_dim=512,
        runtime=runtime,
    )
    mixed = DeviceBuffer(ptr=0x1000, nbytes=2 * HIDDEN * 4)
    try:
        state.prepare([0, 1], [0, 1])
        with pytest.raises(ValueError, match="head_dim 512"):
            run_qwen4_exp_batch_dense_qsa_token_mixer(
                mixed.ptr,
                _shaped_weights(
                    attn_q=(QUERY_HEADS * 512 * 2, HIDDEN),
                    attn_k=(KV_HEADS * 512, HIDDEN),
                    attn_v=(KV_HEADS * 512, HIDDEN),
                    attn_output=(HIDDEN, QUERY_HEADS * 512),
                ),
                attention_state=state,
                scratch=scratch,
                rows=2,
                hidden=HIDDEN,
                query_heads=QUERY_HEADS,
                kv_heads=KV_HEADS,
                head_dim=512,
                rotary_dim=ROTARY_DIM,
                theta=THETA,
                dense_equivalent_limit=MAX_POSITIONS,
                runtime=runtime,
            )
        assert recorder.calls == []
    finally:
        scratch.close()
        state.close()


def test_unprepared_closed_and_closing_states_are_rejected(fixture, recorder) -> None:
    with pytest.raises(RuntimeError, match="no active rows"):
        _run(fixture, recorder)
    assert recorder.calls == []

    fixture.state.prepare([0, 1], [3, 4])
    fixture.scratch.close()
    with pytest.raises(RuntimeError, match="closed"):
        _run(fixture, recorder)
    assert recorder.calls == []


def test_block_size_other_than_the_dense_decode_abi_is_rejected(recorder) -> None:
    runtime = FakeRuntime()
    state = Qwen4ExpBatchAttentionState.allocate(
        state_slots=2,
        max_positions=512,
        block_size=128,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        runtime=runtime,
    )
    scratch = Qwen4ExpQSAScratch.allocate(
        rows=2,
        hidden=HIDDEN,
        query_heads=QUERY_HEADS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        runtime=runtime,
    )
    weights = Qwen4ExpQSAMixerDeviceWeights(
        projections={name: name for name in batch_qsa._REQUIRED_PROJECTIONS},
        q_norm_weight_ptr=1,
        k_norm_weight_ptr=1,
    )
    mixed = DeviceBuffer(ptr=0x1000, nbytes=2 * HIDDEN * 4)
    try:
        state.prepare([0, 1], [5, 6])
        with pytest.raises(ValueError, match="block_size"):
            run_qwen4_exp_batch_dense_qsa_token_mixer(
                mixed.ptr,
                weights,
                attention_state=state,
                scratch=scratch,
                rows=2,
                hidden=HIDDEN,
                query_heads=QUERY_HEADS,
                kv_heads=KV_HEADS,
                head_dim=HEAD_DIM,
                rotary_dim=ROTARY_DIM,
                theta=THETA,
                dense_equivalent_limit=MAX_POSITIONS,
                runtime=runtime,
            )
        assert recorder.calls == []
    finally:
        scratch.close()
        state.close()


def test_closing_state_is_rejected_before_dispatch(fixture, recorder) -> None:
    fixture.state.prepare([0, 1], [3, 4])
    fixture.state.closing = True
    with pytest.raises(RuntimeError, match="closing"):
        _run(fixture, recorder)
    assert recorder.calls == []
