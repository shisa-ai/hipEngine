"""Dispatch-shape and admission tests for the mixed dense/sparse QSA mixer.

Every device call in the mixer is a module-level name, so these tests replace
them with recorders. The mixer's own control flow is what is asserted: one N-row
dispatch per projection, one index-key append per row *on every step including
dense steps*, the sparse group's selection published for the sparse rows only
and staged into the batch-wide selection buffer at each row's own index, and a
rejected call dispatching nothing at all.

The attention, KV-write, and index owners are used for real against a fake
runtime: their contracts (per-row page tables, host cursor mirrors) define what
the mixer may pass them, so faking them would test nothing.
"""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

import hipengine.runtime.qwen4_exp_batch_qsa as batch_qsa
import hipengine.runtime.qwen4_exp_batch_qsa_index as batch_index
from hipengine.core.memory import DeviceBuffer
from hipengine.runtime.qwen4_exp_batch_qsa import (
    Qwen4ExpBatchQsaSelectionScratch,
    run_qwen4_exp_batch_dense_qsa_token_mixer,
    run_qwen4_exp_batch_qsa_mixed_token_mixer,
)
from hipengine.runtime.qwen4_exp_batch_qsa_index import Qwen4ExpBatchQsaIndexState
from hipengine.runtime.qwen4_exp_batch_state import Qwen4ExpBatchAttentionState
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpQSAMixerDeviceWeights,
    Qwen4ExpQSAScratch,
)

SLOTS = 4
MAX_POSITIONS = 16
BLOCK_SIZE = 256
HIDDEN = 8
QUERY_HEADS = 2
KV_HEADS = 1
HEAD_DIM = 4
ROTARY_DIM = 4
THETA = 100.0
Q_WIDTH = QUERY_HEADS * HEAD_DIM
KV_WIDTH = KV_HEADS * HEAD_DIM
INDEX_HEADS = 2
INDEX_DIM = 4
INDEX_ROTARY_DIM = 4
RATIO = 2
BUDGET = 3
OUTPUT_STRIDE = BUDGET * RATIO + RATIO - 1
# ``blocks = (cursor + 1) // ratio``; dense while ``blocks <= budget``.
DENSE_POSITION = 3  # context 4 -> 2 blocks
SPARSE_POSITION = 8  # context 9 -> 4 blocks


class FakeRuntime:
    """Allocation bookkeeping only: no device work, no real pointers."""

    def __init__(self) -> None:
        self.next_ptr = 0x4000
        self.allocations: dict[int, int] = {}
        self.freed: list[int] = []
        self.copies: list[tuple[int, int, object]] = []
        self.free_calls = 0
        self.fail_free_on_call: int | None = None

    def malloc(self, nbytes: int) -> int:
        ptr = self.next_ptr
        self.next_ptr += max(int(nbytes), 1) + 0x100
        self.allocations[ptr] = int(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.free_calls += 1
        if self.fail_free_on_call == self.free_calls:
            raise RuntimeError("injected free failure")
        self.freed.append(ptr)
        self.allocations.pop(ptr, None)

    def memset(self, dst: int, value: int, nbytes: int) -> None:
        assert dst in self.allocations

    def memcpy(self, dst: int, src: int, nbytes: int, kind) -> None:
        # The mixer passes interior offsets (a row inside a slab), so check that
        # the pointer lands inside an owned allocation rather than on its base.
        def owned(pointer: int) -> bool:
            return any(
                base <= pointer < base + size
                for base, size in self.allocations.items()
            )

        assert owned(dst) or owned(src), (dst, src)
        self.copies.append((dst, int(nbytes), kind))


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def stage(self, name: str):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return record

    def names(self) -> list[str]:
        return [name for name, _, _ in self.calls]

    def all(self, name: str) -> list[tuple[tuple, dict]]:
        return [(args, kwargs) for label, args, kwargs in self.calls if label == name]

    def one(self, name: str) -> tuple[tuple, dict]:
        matches = self.all(name)
        assert len(matches) == 1, f"expected one {name} dispatch, got {len(matches)}"
        return matches[0]


@pytest.fixture
def recorder(monkeypatch) -> Recorder:
    record = Recorder()
    for name in (
        "qwen4_exp_qsa_split_norm_rope_rows_f32",
        "qwen4_exp_qsa_split_norm_mrope_rows_f32",
        "qwen4_exp_qsa_norm_rope_rows_f32",
        "qwen4_exp_qsa_norm_mrope_rows_f32",
        "qwen35_write_paged_kv_f32_batch_spans",
        "qwen35_paged_full_attn_decode_context_bf16_batch_spans",
        "qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32",
        "qwen4_exp_qsa_gate_context_f32",
    ):
        monkeypatch.setattr(batch_qsa, name, record.stage(name))
    monkeypatch.setattr(batch_qsa, "launch_gguf_linear", record.stage("projection"))
    for name in (
        "qwen4_exp_qsa_scatter_index_key_device_position_f32",
        "qwen4_exp_qsa_pool_norm_rope_f32",
        "qwen4_exp_qsa_score_f32",
        "qwen4_exp_qsa_topk_expand_rows_f32_i64",
    ):
        monkeypatch.setattr(batch_index, name, record.stage(name))
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
        self.index = Qwen4ExpBatchQsaIndexState.allocate(
            attention_state=self.state,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
            compression_ratio=RATIO,
            block_budget=BUDGET,
            runtime=self.runtime,
        )
        self.selection = Qwen4ExpBatchQsaSelectionScratch.allocate(
            rows=SLOTS,
            stride=OUTPUT_STRIDE,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
            runtime=self.runtime,
        )
        self.scratch = Qwen4ExpQSAScratch.allocate(
            rows=scratch_rows,
            hidden=HIDDEN,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            runtime=self.runtime,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
        )
        self.weights = Qwen4ExpQSAMixerDeviceWeights(
            projections={
                name: f"{name}-weight"
                for name in (*batch_qsa._REQUIRED_PROJECTIONS, *batch_qsa._INDEX_PROJECTIONS)
            },
            q_norm_weight_ptr=0x900,
            k_norm_weight_ptr=0x904,
            index_q_norm_weight_ptr=0x908,
            index_k_norm_weight_ptr=0x90C,
        )
        self.mixed = DeviceBuffer(ptr=0x1000, nbytes=SLOTS * HIDDEN * 4)
        self.rows = 0

    def admit(self, positions: list[int], slots: list[int] | None = None) -> None:
        """Admit the same row set on both owners at those positions.

        The index cursor of a row at position ``p`` is ``p``: the caller appended
        one key per token ``0 .. p-1`` in earlier steps, and this call appends the
        key for token ``p``.
        """

        row_slots = list(range(len(positions))) if slots is None else list(slots)
        for slot, position in zip(row_slots, positions):
            self.index.cursor_host[slot] = position
            self.index.pooled_count_host[slot] = position // RATIO
        self.state.prepare(row_slots, positions)
        self.index.prepare(row_slots)
        self.rows = len(row_slots)

    def close(self) -> None:
        self.scratch.close()
        self.selection.close()
        self.index.close()
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
        index_state=fixture.index,
        selection=fixture.selection,
        scratch=fixture.scratch,
        rows=fixture.rows,
        hidden=HIDDEN,
        query_heads=QUERY_HEADS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        theta=THETA,
        index_heads=INDEX_HEADS,
        index_dim=INDEX_DIM,
        index_rotary_dim=INDEX_ROTARY_DIM,
        runtime=fixture.runtime,
    )
    kwargs.update(overrides)
    weights = kwargs.pop("weights", fixture.weights)
    return run_qwen4_exp_batch_qsa_mixed_token_mixer(fixture.mixed.ptr, weights, **kwargs)


def _fake_readback(monkeypatch, *, counts: np.ndarray, positions: np.ndarray) -> None:
    """Replace the index owner's device read-back with known host contents."""

    rows = counts.size

    def fake_copy(dst_ptr, src, nbytes, *, runtime=None):
        assert src is not None
        if nbytes == rows * 4:
            _write(dst_ptr, counts.astype(np.int32))
        elif nbytes == positions.nbytes:
            _write(dst_ptr, positions)
        else:  # pragma: no cover - guards an unexpected read-back size
            raise AssertionError(f"unexpected read-back of {nbytes} bytes")

    monkeypatch.setattr(batch_index, "copy_device_to_host", fake_copy)


def _write(dst_ptr: int, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    ctypes.memmove(dst_ptr, contiguous.ctypes.data, contiguous.nbytes)


def _cursor(fixture: Fixture, slot: int) -> int:
    return int(fixture.index.cursor_host[slot])


def test_dense_only_batch_still_appends_one_index_key_per_row(
    fixture: Fixture, recorder: Recorder, monkeypatch
) -> None:
    """A dense step must capture index keys, or a later sparse step loses history."""

    fixture.admit([DENSE_POSITION, DENSE_POSITION + 1])
    _fake_readback(
        monkeypatch,
        counts=np.array([0], dtype=np.int32),
        positions=np.zeros((1, OUTPUT_STRIDE), dtype=np.int64),
    )
    _run(fixture, recorder)

    # index_k is projected for every row even though no row selects.
    projections = recorder.all("projection")
    assert len(projections) == 5, "attn_q/attn_k/attn_v/index_k/attn_output"
    assert recorder.all("qwen4_exp_qsa_scatter_index_key_device_position_f32")
    assert len(recorder.all("qwen4_exp_qsa_scatter_index_key_device_position_f32")) == 2
    # No selection work at all, and the batch still attends every row.
    assert recorder.all("qwen4_exp_qsa_pool_norm_rope_f32") == []
    assert recorder.all("qwen4_exp_qsa_score_f32") == []
    assert recorder.all("qwen4_exp_qsa_topk_expand_rows_f32_i64") == []
    (attention_args, attention_kwargs) = recorder.one(
        "qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32"
    )
    assert attention_kwargs["rows"] == 2
    assert attention_kwargs["selected_stride"] == OUTPUT_STRIDE
    # Dense rows carry every visible position, so the count is the context.
    assert fixture.selection.counts_host[:2].tolist() == [DENSE_POSITION + 1, DENSE_POSITION + 2]
    assert fixture.selection.positions_host[0, : DENSE_POSITION + 1].tolist() == list(
        range(DENSE_POSITION + 1)
    )
    assert fixture.selection.positions_host[1, : DENSE_POSITION + 2].tolist() == list(
        range(DENSE_POSITION + 2)
    )
    # The cursors advanced by exactly one per row.
    assert _cursor(fixture, 0) == DENSE_POSITION + 1
    assert _cursor(fixture, 1) == DENSE_POSITION + 2


def test_sparse_rows_are_staged_at_their_own_row_index(
    fixture: Fixture, recorder: Recorder, monkeypatch
) -> None:
    """The sparse group is selected as a subset and scattered back by row."""

    fixture.admit([DENSE_POSITION, SPARSE_POSITION, SPARSE_POSITION + 1])
    subset_positions = np.array(
        [[8, 7, 6, 5, 4, 3, 2], [9, 8, 7, 6, 5, 4, 3]], dtype=np.int64
    )
    _fake_readback(
        monkeypatch,
        counts=np.array([5, 3], dtype=np.int32),
        positions=subset_positions,
    )
    _run(fixture, recorder)

    # Selection covers the sparse rows only, in one batched selection dispatch.
    (topk_args, topk_kwargs) = recorder.one("qwen4_exp_qsa_topk_expand_rows_f32_i64")
    assert topk_args[4] == 2, "selection runs over the sparse group"
    assert len(recorder.all("qwen4_exp_qsa_score_f32")) == 2
    # Row 0 is dense: all visible positions. Rows 1 and 2 carry the selection.
    assert fixture.selection.counts_host[:3].tolist() == [DENSE_POSITION + 1, 5, 3]
    assert fixture.selection.positions_host[0, :4].tolist() == [0, 1, 2, 3]
    assert fixture.selection.positions_host[1, :5].tolist() == [8, 7, 6, 5, 4]
    assert fixture.selection.positions_host[2, :3].tolist() == [9, 8, 7]
    # index_q is projected once for the whole batch, not once per sparse row.
    assert len(recorder.all("projection")) == 6, "attn_q/k/v, index_k, index_q, attn_output"
    assert len(recorder.all("qwen4_exp_qsa_norm_rope_rows_f32")) == 1
    # The index owner is left holding the full row set again.
    assert fixture.index.active.slot_ids == fixture.state.active.slot_ids


def test_stage_order_puts_the_index_append_before_the_kv_write(
    fixture: Fixture, recorder: Recorder, monkeypatch
) -> None:
    """The key and the KV token must land at the same position in one step."""

    fixture.admit([SPARSE_POSITION])
    _fake_readback(
        monkeypatch,
        counts=np.array([4], dtype=np.int32),
        positions=np.arange(OUTPUT_STRIDE, dtype=np.int64)[None, :],
    )
    _run(fixture, recorder)
    names = recorder.names()
    assert names.index("qwen4_exp_qsa_scatter_index_key_device_position_f32") < names.index(
        "qwen35_write_paged_kv_f32_batch_spans"
    )
    assert names.index("qwen35_write_paged_kv_f32_batch_spans") < names.index(
        "qwen4_exp_qsa_topk_expand_rows_f32_i64"
    )
    assert names.index("qwen4_exp_qsa_topk_expand_rows_f32_i64") < names.index(
        "qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32"
    )
    assert names[-1] == "projection"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"index_heads": INDEX_HEADS + 1}, "index geometry"),
        ({"index_dim": INDEX_DIM + 1}, "index geometry"),
        ({"index_rotary_dim": INDEX_DIM + 1}, "index_rotary_dim"),
        ({"index_rotary_dim": 3}, "index_rotary_dim"),
        ({"rows": SLOTS + 1}, "does not match"),
    ],
)
def test_invalid_metadata_dispatches_nothing_and_keeps_the_cursors(
    fixture: Fixture, recorder: Recorder, overrides, message
) -> None:
    fixture.admit([DENSE_POSITION, SPARSE_POSITION])
    before = fixture.index.cursor_host.copy()
    with pytest.raises((ValueError, TypeError), match=message):
        _run(fixture, recorder, **overrides)
    assert recorder.calls == []
    assert np.array_equal(fixture.index.cursor_host, before)
    assert fixture.selection.counts_host.tolist() == [0] * SLOTS


def test_a_dense_only_caller_that_skipped_index_keys_is_rejected(
    fixture: Fixture, recorder: Recorder
) -> None:
    """Cursor/position disagreement is the signature of a skipped index append."""

    fixture.admit([DENSE_POSITION, SPARSE_POSITION])
    fixture.index.cursor_host[0] = 1  # row 0 never appended its earlier keys
    with pytest.raises(ValueError, match="index cursor"):
        _run(fixture, recorder)
    assert recorder.calls == []


def test_index_state_prepared_for_a_different_row_set_is_rejected(
    fixture: Fixture, recorder: Recorder
) -> None:
    fixture.state.prepare([0, 1], [DENSE_POSITION, DENSE_POSITION])
    fixture.index.prepare([1, 0])  # same slots, different row order
    fixture.index.cursor_host[0] = DENSE_POSITION
    fixture.index.cursor_host[1] = DENSE_POSITION
    fixture.rows = 2
    with pytest.raises(ValueError, match="active slots do not match"):
        _run(fixture, recorder)
    assert recorder.calls == []


def test_index_state_mirroring_another_attention_owner_is_rejected(
    fixture: Fixture, recorder: Recorder
) -> None:
    other = Qwen4ExpBatchAttentionState.allocate(
        state_slots=SLOTS,
        max_positions=MAX_POSITIONS,
        block_size=BLOCK_SIZE,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        runtime=fixture.runtime,
    )
    try:
        other.prepare([0, 1], [DENSE_POSITION, DENSE_POSITION])
        fixture.rows = 2
        with pytest.raises(ValueError, match="must mirror the batch attention state"):
            _run(fixture, recorder, attention_state=other)
        assert recorder.calls == []
    finally:
        other.close()


def test_selection_scratch_must_cover_the_rows_and_the_index_stride(
    fixture: Fixture, recorder: Recorder
) -> None:
    fixture.admit([DENSE_POSITION])
    narrow = Qwen4ExpBatchQsaSelectionScratch.allocate(
        rows=SLOTS,
        stride=OUTPUT_STRIDE - 1,
        index_heads=INDEX_HEADS,
        index_dim=INDEX_DIM,
        runtime=fixture.runtime,
    )
    fixture.rows = 1
    try:
        with pytest.raises(ValueError, match="output stride"):
            _run(fixture, recorder, selection=narrow)
    finally:
        narrow.close()
    short = Qwen4ExpBatchQsaSelectionScratch.allocate(
        rows=1,
        stride=OUTPUT_STRIDE,
        index_heads=INDEX_HEADS,
        index_dim=INDEX_DIM,
        runtime=fixture.runtime,
    )
    try:
        fixture.admit([DENSE_POSITION, DENSE_POSITION])
        with pytest.raises(ValueError, match="selection scratch holds"):
            _run(fixture, recorder, selection=short)
    finally:
        short.close()
    assert recorder.calls == []


def test_missing_index_weights_are_rejected_before_the_append(
    fixture: Fixture, recorder: Recorder
) -> None:
    fixture.admit([DENSE_POSITION])
    weights = Qwen4ExpQSAMixerDeviceWeights(
        projections={
            name: f"{name}-weight" for name in batch_qsa._REQUIRED_PROJECTIONS
        },
        q_norm_weight_ptr=0x900,
        k_norm_weight_ptr=0x904,
        index_q_norm_weight_ptr=0x908,
        index_k_norm_weight_ptr=0x90C,
    )
    with pytest.raises(ValueError, match="missing Qwen4Exp QSA index weights"):
        _run(fixture, recorder, weights=weights)
    assert recorder.calls == []
    unnamed = Qwen4ExpQSAMixerDeviceWeights(
        projections={
            name: f"{name}-weight"
            for name in (*batch_qsa._REQUIRED_PROJECTIONS, *batch_qsa._INDEX_PROJECTIONS)
        },
        q_norm_weight_ptr=0x900,
        k_norm_weight_ptr=0x904,
    )
    with pytest.raises(ValueError, match="index Q/K norm weights are required"):
        _run(fixture, recorder, weights=unnamed)
    assert recorder.calls == []


def test_a_scratch_too_small_for_the_rows_is_rejected(recorder: Recorder) -> None:
    """One row of scratch cannot carry two rows of any stage, index included."""

    built = Fixture(scratch_rows=1)
    try:
        built.admit([DENSE_POSITION, DENSE_POSITION])
        with pytest.raises(ValueError, match="scratch"):
            _run(built, recorder)
        assert recorder.calls == []
        assert np.array_equal(built.index.cursor_host, np.array([DENSE_POSITION, DENSE_POSITION, 0, 0]))
    finally:
        built.close()


def test_a_closed_selection_owner_is_rejected(fixture: Fixture, recorder: Recorder) -> None:
    fixture.admit([DENSE_POSITION])
    selection = Qwen4ExpBatchQsaSelectionScratch.allocate(
        rows=SLOTS,
        stride=OUTPUT_STRIDE,
        index_heads=INDEX_HEADS,
        index_dim=INDEX_DIM,
        runtime=fixture.runtime,
    )
    selection.close()
    with pytest.raises(RuntimeError, match="closed"):
        _run(fixture, recorder, selection=selection)
    assert recorder.calls == []


def test_mrope_positions_select_the_mrope_rows_kernels(
    fixture: Fixture, recorder: Recorder, monkeypatch
) -> None:
    fixture.admit([SPARSE_POSITION])
    _fake_readback(
        monkeypatch,
        counts=np.array([2], dtype=np.int32),
        positions=np.zeros((1, OUTPUT_STRIDE), dtype=np.int64),
    )
    positions = np.zeros(3 * 1, dtype=np.int64)
    _run(fixture, recorder, rope_positions_ptr=0x5000)
    assert len(recorder.all("qwen4_exp_qsa_split_norm_mrope_rows_f32")) == 1
    assert len(recorder.all("qwen4_exp_qsa_norm_mrope_rows_f32")) == 1
    assert recorder.all("qwen4_exp_qsa_split_norm_rope_rows_f32") == []
    assert positions.nbytes == 24


def test_dense_entry_point_is_unchanged_by_the_mixed_path(
    fixture: Fixture, recorder: Recorder
) -> None:
    """The dense mixer still refuses a row past its bound and dispatches nothing."""

    fixture.admit([SPARSE_POSITION])
    with pytest.raises(ValueError, match="dense-equivalent limit"):
        run_qwen4_exp_batch_dense_qsa_token_mixer(
            fixture.mixed.ptr,
            fixture.weights,
            attention_state=fixture.state,
            scratch=fixture.scratch,
            rows=1,
            hidden=HIDDEN,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
            theta=THETA,
            dense_equivalent_limit=OUTPUT_STRIDE,
            runtime=fixture.runtime,
        )
    assert recorder.calls == []


def test_shaped_index_weights_are_checked_against_the_index_geometry(
    fixture: Fixture, recorder: Recorder
) -> None:
    fixture.admit([DENSE_POSITION])

    def shaped(shape):
        return SimpleNamespace(
            spec=SimpleNamespace(source=SimpleNamespace(shape=shape))
        )

    geometry = {
        "attn_q": (Q_WIDTH * 2, HIDDEN),
        "attn_k": (KV_WIDTH, HIDDEN),
        "attn_v": (KV_WIDTH, HIDDEN),
        "attn_output": (HIDDEN, Q_WIDTH),
        "index_q": (INDEX_HEADS * INDEX_DIM, HIDDEN),
        "index_k": (INDEX_DIM, HIDDEN),
    }
    weights = Qwen4ExpQSAMixerDeviceWeights(
        projections={name: shaped(shape) for name, shape in geometry.items()},
        q_norm_weight_ptr=0x900,
        k_norm_weight_ptr=0x904,
        index_q_norm_weight_ptr=0x908,
        index_k_norm_weight_ptr=0x90C,
    )
    _run(fixture, recorder, weights=weights)
    assert recorder.calls, "matching shapes must be admitted"
    recorder.calls.clear()

    geometry["index_k"] = (INDEX_DIM + 1, HIDDEN)
    wrong = Qwen4ExpQSAMixerDeviceWeights(
        projections={name: shaped(shape) for name, shape in geometry.items()},
        q_norm_weight_ptr=0x900,
        k_norm_weight_ptr=0x904,
        index_q_norm_weight_ptr=0x908,
        index_k_norm_weight_ptr=0x90C,
    )
    fixture.admit([DENSE_POSITION])
    with pytest.raises(ValueError, match="index_k weight shape"):
        _run(fixture, recorder, weights=wrong)
    assert recorder.calls == []


# -- selection-scratch lifecycle -------------------------------------------


def _scratch(runtime: FakeRuntime, **overrides) -> Qwen4ExpBatchQsaSelectionScratch:
    kwargs = dict(
        rows=SLOTS,
        stride=OUTPUT_STRIDE,
        index_heads=INDEX_HEADS,
        index_dim=INDEX_DIM,
        runtime=runtime,
    )
    kwargs.update(overrides)
    return Qwen4ExpBatchQsaSelectionScratch.allocate(**kwargs)


def test_host_mirror_failure_allocates_no_device_buffer(monkeypatch) -> None:
    """A host allocation failure must not leave device slabs behind."""

    runtime = FakeRuntime()

    class HostFailure:
        def __getattr__(self, name: str) -> object:
            return getattr(np, name)

        def zeros(self, *args: object, **kwargs: object) -> np.ndarray:
            raise MemoryError("injected host allocation failure")

    monkeypatch.setattr(batch_qsa, "np", HostFailure())
    with pytest.raises(MemoryError, match="injected host allocation failure"):
        _scratch(runtime)
    assert runtime.allocations == {}
    assert runtime.freed == []


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"stride": 2**61}, "positions byte size"),
        ({"rows": 2**61}, "positions byte size"),
        ({"index_dim": 2**60}, "queries byte size"),
    ],
)
def test_unaddressable_slabs_are_rejected_before_allocating(
    overrides: dict, message: str
) -> None:
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match=message):
        _scratch(runtime, **overrides)
    assert runtime.allocations == {}
    assert runtime.freed == []


def test_partial_close_failure_is_retried_exactly_once() -> None:
    """A free that fails part-way must not lose the remaining slabs."""

    runtime = FakeRuntime()
    scratch = _scratch(runtime)
    slabs = [scratch.positions.ptr, scratch.counts.ptr, scratch.queries.ptr]
    # Newest-first: the queries slab is released, then the counts free fails.
    runtime.fail_free_on_call = 2
    with pytest.raises(RuntimeError, match="injected free failure"):
        scratch.close()
    assert scratch.closing is True
    assert scratch.closed is False
    assert runtime.freed == [slabs[2]]
    # Poisoned: a half-released owner hands out no address at all.
    with pytest.raises(RuntimeError, match="closing"):
        scratch.query_row_ptr(0)
    with pytest.raises(RuntimeError, match="closing"):
        with scratch:
            pass
    runtime.fail_free_on_call = None
    scratch.close()
    assert scratch.closed is True
    assert runtime.freed == [slabs[2], slabs[1], slabs[0]]
    assert runtime.allocations == {}
    # Nothing is freed twice by a further call.
    scratch.close()
    assert runtime.freed == [slabs[2], slabs[1], slabs[0]]


def test_access_is_refused_after_close_and_close_is_idempotent() -> None:
    runtime = FakeRuntime()
    scratch = _scratch(runtime)
    assert scratch.query_row_ptr(0) == scratch.queries.ptr
    assert scratch.query_row_ptr(SLOTS - 1) == (
        scratch.queries.ptr + (SLOTS - 1) * INDEX_HEADS * INDEX_DIM * 4
    )
    with pytest.raises(IndexError, match="outside"):
        scratch.query_row_ptr(SLOTS)
    with scratch:
        assert scratch.closed is False
    assert scratch.closed is True
    assert runtime.allocations == {}
    with pytest.raises(RuntimeError, match="closed"):
        scratch.query_row_ptr(0)
    with pytest.raises(RuntimeError, match="closed"):
        with scratch:
            pass
    scratch.close()
    assert runtime.allocations == {}


def test_close_releases_a_hand_built_owner_without_a_pending_list() -> None:
    """``close`` must never free nothing just because ``allocate`` was bypassed."""

    runtime = FakeRuntime()
    slabs = [
        DeviceBuffer(ptr=runtime.malloc(64), nbytes=64) for _ in range(3)
    ]
    scratch = Qwen4ExpBatchQsaSelectionScratch(
        positions=slabs[0],
        counts=slabs[1],
        queries=slabs[2],
        positions_host=np.zeros((1, 1), dtype=np.int64),
        counts_host=np.zeros(1, dtype=np.int32),
        rows=1,
        stride=1,
        index_heads=1,
        index_dim=1,
        runtime=runtime,
    )
    scratch.close()
    assert runtime.freed == [slabs[2].ptr, slabs[1].ptr, slabs[0].ptr]
    assert runtime.allocations == {}
