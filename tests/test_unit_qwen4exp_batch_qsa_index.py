"""Dispatch-shape and admission tests for the N-row sparse QSA index state.

The committed index primitives are module-level names in the batch index module,
so these tests replace them with recorders. That is the only way to assert what
this module actually does at the dispatch level, which is the point of the unit:
append, pool, and score are one dispatch per row/slot, and selection is one
dispatch for all rows. A test that only compared positions would pass for a
per-row loop too.

The batch attention owner is used for real (with a fake runtime): the index
state's ownership contract is defined in terms of that owner's page tables and
per-row positions, so faking the owner would test nothing.
"""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

import hipengine.runtime.qwen4_exp_batch_qsa_index as batch_index
from hipengine.runtime.qwen4_exp_batch_qsa_index import (
    Qwen4ExpBatchQsaIndexRollback,
    Qwen4ExpBatchQsaIndexRowPlan,
    Qwen4ExpBatchQsaIndexState,
)
from hipengine.runtime.qwen4_exp_batch_state import (
    Qwen4ExpBatchAttentionState,
    Qwen4ExpBatchStepMetadata,
)

SLOTS = 4
MAX_POSITIONS = 16
BLOCK_SIZE = 8
RATIO = 2
BUDGET = 3
INDEX_HEADS = 2
INDEX_DIM = 4
COMPLETE_BLOCKS = MAX_POSITIONS // RATIO
OUTPUT_STRIDE = BUDGET * RATIO + RATIO - 1
BLOCKS_PER_REQUEST = (MAX_POSITIONS + BLOCK_SIZE - 1) // BLOCK_SIZE
SLOT_CAPACITY = BLOCKS_PER_REQUEST * BLOCK_SIZE


class FakeRuntime:
    """Allocation bookkeeping only: no device work, no real pointers."""

    def __init__(self) -> None:
        self.next_ptr = 0x4000
        self.allocations: dict[int, int] = {}
        self.freed: list[int] = []
        self.memsets: list[tuple[int, int]] = []
        self.copies: list[tuple[int, int, object]] = []

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
        # One side is a host pointer; the device side must be an owned buffer.
        # A destination may be an offset view into a larger owned slab.
        owned = any(
            base <= dst < base + size for base, size in self.allocations.items()
        )
        assert owned or src in self.allocations, (dst, src)
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
        "qwen4_exp_qsa_scatter_index_key_device_position_f32",
        "qwen4_exp_qsa_pool_norm_rope_f32",
        "qwen4_exp_qsa_score_f32",
        "qwen4_exp_qsa_topk_expand_rows_f32_i64",
    ):
        monkeypatch.setattr(batch_index, name, record.stage(name))
    return record


class Fixture:
    def __init__(self) -> None:
        self.runtime = FakeRuntime()
        self.attention = Qwen4ExpBatchAttentionState.allocate(
            state_slots=SLOTS,
            max_positions=MAX_POSITIONS,
            block_size=BLOCK_SIZE,
            kv_heads=1,
            head_dim=4,
            runtime=self.runtime,
        )
        self.state = Qwen4ExpBatchQsaIndexState.allocate(
            attention_state=self.attention,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
            compression_ratio=RATIO,
            block_budget=BUDGET,
            runtime=self.runtime,
        )

    def admit(self, slots, positions) -> None:
        """Admit the same row set on the attention owner and the index owner."""

        self.attention.prepare(slots, positions)
        self.state.prepare(slots)

    def close(self) -> None:
        self.state.close()
        self.attention.close()


@pytest.fixture
def fixture() -> Fixture:
    built = Fixture()
    try:
        yield built
    finally:
        built.close()


def _readback(monkeypatch, *, counts: int, positions: np.ndarray) -> None:
    """Replace the device read-back with known host contents."""

    def fake_copy(dst_ptr, src, nbytes, *, runtime=None):
        assert src is not None
        if nbytes == counts * 4:
            _write(dst_ptr, np.full(counts, 3, dtype=np.int32))
        elif nbytes == positions.nbytes:
            _write(dst_ptr, positions)
        else:  # pragma: no cover - guards an unexpected read-back size
            raise AssertionError(f"unexpected read-back of {nbytes} bytes")

    monkeypatch.setattr(batch_index, "copy_device_to_host", fake_copy)


def _write(dst_ptr: int, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    ctypes.memmove(dst_ptr, contiguous.ctypes.data, contiguous.nbytes)


def _fill(
    state: Qwen4ExpBatchQsaIndexState, slot: int, cursor: int, *, pooled: int | None = None
) -> None:
    """Set a slot's counters; ``pooled=None`` means every complete block is pooled."""

    state.cursor_host[slot] = cursor
    state.pooled_count_host[slot] = cursor // RATIO if pooled is None else int(pooled)


def test_owner_builds_disjoint_per_slot_member_maps(fixture: Fixture) -> None:
    state = fixture.state
    assert state.capacity == MAX_POSITIONS
    assert state.complete_blocks == COMPLETE_BLOCKS
    assert state.slot_capacity_tokens == SLOT_CAPACITY
    assert state.output_stride == OUTPUT_STRIDE
    assert state.dense_equivalent_limit == OUTPUT_STRIDE
    assert state.member_host.shape == (SLOTS, COMPLETE_BLOCKS, RATIO)
    assert state.member_host.dtype == np.int32
    # Slot s's members are the global physical positions of its own page blocks.
    for slot in range(SLOTS):
        expected = np.arange(SLOT_CAPACITY, dtype=np.int64) + slot * SLOT_CAPACITY
        assert state.member_host[slot].reshape(-1).tolist() == (
            expected[: COMPLETE_BLOCKS * RATIO].tolist()
        )
    assert np.unique(state.member_host).size == state.member_host.size
    # Block starts are logical and therefore identical in every slot's segment.
    assert state.block_starts_host.tolist() == (
        np.tile(np.arange(COMPLETE_BLOCKS) * RATIO, SLOTS).tolist()
    )
    assert state.raw_keys.nbytes == SLOTS * SLOT_CAPACITY * INDEX_DIM * 4
    assert state.pooled_keys.nbytes == SLOTS * COMPLETE_BLOCKS * INDEX_DIM * 4
    assert state.scores.nbytes == SLOTS * COMPLETE_BLOCKS * 4
    assert state.selected_positions.nbytes == SLOTS * OUTPUT_STRIDE * 8
    # Raw and pooled slabs are zeroed so an unappended key is a checkable value.
    assert [
        entry
        for entry in fixture.runtime.memsets
        if entry[0] in (state.raw_keys.ptr, state.pooled_keys.ptr)
    ] == [
        (state.raw_keys.ptr, state.raw_keys.nbytes),
        (state.pooled_keys.ptr, state.pooled_keys.nbytes),
    ]


def test_allocate_rejects_geometry_the_selection_cannot_address() -> None:
    runtime = FakeRuntime()
    attention = Qwen4ExpBatchAttentionState.allocate(
        state_slots=2,
        max_positions=MAX_POSITIONS,
        block_size=BLOCK_SIZE,
        kv_heads=1,
        head_dim=4,
        runtime=runtime,
    )
    cases = [
        (dict(compression_ratio=MAX_POSITIONS * 2), ValueError, "no complete"),
        (dict(block_budget=COMPLETE_BLOCKS + 1), ValueError, "fewer than the selection budget"),
        (dict(index_heads=0), ValueError, "index_heads"),
        (dict(index_dim=-1), ValueError, "index_dim"),
        (dict(compression_ratio=0), ValueError, "compression_ratio"),
        (dict(block_budget=0), ValueError, "block_budget"),
        (dict(index_heads=True), TypeError, "integer"),
    ]
    for overrides, error, match in cases:
        kwargs = dict(
            attention_state=attention,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
            compression_ratio=RATIO,
            block_budget=BUDGET,
            runtime=runtime,
        )
        kwargs.update(overrides)
        with pytest.raises(error, match=match):
            Qwen4ExpBatchQsaIndexState.allocate(**kwargs)
    with pytest.raises(ValueError, match="runtime must match"):
        Qwen4ExpBatchQsaIndexState.allocate(
            attention_state=attention,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
            compression_ratio=RATIO,
            block_budget=BUDGET,
            runtime=FakeRuntime(),
        )
    attention.close()
    with pytest.raises(RuntimeError, match="open attention state"):
        Qwen4ExpBatchQsaIndexState.allocate(
            attention_state=attention,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
            compression_ratio=RATIO,
            block_budget=BUDGET,
            runtime=runtime,
        )


def test_prepare_publishes_query_positions_and_blocks(fixture: Fixture) -> None:
    _fill(fixture.state, 3, 5)
    _fill(fixture.state, 0, 4)
    fixture.attention.prepare([3, 0], [4, 4])
    active = fixture.state.prepare([3, 0])
    assert active.rows == 2
    assert active.slot_ids == (3, 0)
    assert active.slots.tolist() == [3, 0]
    # The query position is the last appended key; blocks are complete only.
    assert active.positions.tolist() == [4, 3]
    assert active.blocks.tolist() == [2, 2]
    assert active.row_of(0) == 1
    # One upload, exactly one position per row.
    assert fixture.runtime.copies[-1][1] == 2 * 8


def test_prepare_rejects_duplicate_and_out_of_range_slots(fixture) -> None:
    _fill(fixture.state, 1, 3)
    cases = [
        ([], ValueError, "at least one"),
        ([1, 1], ValueError, "slot 1 appears twice"),
        ([SLOTS], ValueError, "outside the declared state slab|outside"),
        (["a"], TypeError, "integer"),
        ([0, 1, 2, 3, 0], ValueError, "at most"),
    ]
    for slots, error, match in cases:
        with pytest.raises(error, match=match):
            fixture.state.prepare(slots)
        assert fixture.state.active is None


def test_prepare_admits_an_unkeyed_slot_so_its_first_append_has_a_row(
    fixture: Fixture, recorder: Recorder
) -> None:
    # The first key of a request can only be appended through an active row, so
    # prepare must admit a zero cursor and selection must refuse it until a key
    # exists.
    fixture.admit([2], [0])
    active = fixture.state.prepare([2])
    assert active.positions.tolist() == [-1]
    assert active.blocks.tolist() == [0]
    fixture.state.append_active(0x8000)
    assert fixture.state.cursor_host.tolist() == [0, 0, 1, 0]
    assert len(recorder.all("qwen4_exp_qsa_scatter_index_key_device_position_f32")) == 1

    fixture.state.active = None
    fixture.attention.prepare([3], [0])
    fixture.state.prepare([3])
    with pytest.raises(ValueError, match="no appended index key"):
        fixture.state.select_active(
            0x9000, key_norm_weight_ptr=0xA000, rotary_dim=INDEX_DIM, theta=100.0
        )
    assert recorder.names() == [
        "qwen4_exp_qsa_scatter_index_key_device_position_f32"
    ]


def test_append_active_scatters_one_key_per_row_at_its_own_position(
    fixture: Fixture, recorder: Recorder
) -> None:
    _fill(fixture.state, 3, 2)
    _fill(fixture.state, 1, 2)
    fixture.admit([3, 1], [2, 2])
    fixture.state.append_active(0x8000)

    assert recorder.names() == [
        "qwen4_exp_qsa_scatter_index_key_device_position_f32"
    ] * 2
    dim_bytes = INDEX_DIM * 4
    table_stride = BLOCKS_PER_REQUEST * 4
    table = fixture.attention.active.block_table.ptr
    position = fixture.attention.active.position.ptr
    for row, args in enumerate(recorder.all("qwen4_exp_qsa_scatter_index_key_device_position_f32")):
        args, kwargs = args
        assert args[0] == 0x8000 + row * dim_bytes
        assert args[1] == fixture.state.raw_keys.ptr
        assert args[2] == table + row * table_stride
        assert args[3] == position + row * 8
        assert args[4] == BLOCK_SIZE
        assert args[5] == BLOCKS_PER_REQUEST
        assert args[6] == INDEX_DIM
        assert kwargs["runtime"] is fixture.runtime
    assert fixture.state.cursor_host.tolist() == [0, 3, 0, 3]


def test_append_active_rejects_a_step_that_disagrees_with_the_kv_write(
    fixture: Fixture, recorder: Recorder
) -> None:
    _fill(fixture.state, 0, 2)
    fixture.admit([0], [3])
    with pytest.raises(ValueError, match="does not match the attention position"):
        fixture.state.append_active(0x8000)
    assert recorder.calls == []

    fixture.admit([0], [2])
    _fill(fixture.state, 0, MAX_POSITIONS)
    fixture.attention.position_host[0] = MAX_POSITIONS
    with pytest.raises(ValueError, match="reached the per-request capacity"):
        fixture.state.append_active(0x8000)
    assert recorder.calls == []

    with pytest.raises(ValueError, match="raw_keys_ptr"):
        fixture.state.append_active(0)
    fixture.state.active = None
    with pytest.raises(RuntimeError, match="no active rows"):
        fixture.state.append_active(0x8000)
    assert recorder.calls == []


def test_select_active_is_one_score_per_row_and_one_batched_selection(
    fixture: Fixture, recorder: Recorder, monkeypatch
) -> None:
    _fill(fixture.state, 2, 10, pooled=0)
    _fill(fixture.state, 0, 8, pooled=0)
    _fill(fixture.state, 3, 9, pooled=0)
    fixture.admit([2, 0, 3], [9, 7, 8])
    expected = np.arange(3 * OUTPUT_STRIDE, dtype=np.int64).reshape(3, OUTPUT_STRIDE)
    _readback(monkeypatch, counts=3, positions=expected)
    counts, positions = fixture.state.select_active(
        0x9000,
        key_norm_weight_ptr=0xA000,
        rotary_dim=INDEX_DIM,
        theta=100.0,
    )

    # Per slot with new complete blocks: one pool each, in slot order.
    pools = recorder.all("qwen4_exp_qsa_pool_norm_rope_f32")
    assert len(pools) == 3
    pooled_stride = COMPLETE_BLOCKS * INDEX_DIM * 4
    member_stride = COMPLETE_BLOCKS * RATIO * 4
    starts_stride = COMPLETE_BLOCKS * 8
    blocks_per_slot = {2: 5, 0: 4, 3: 4}
    for index, slot in enumerate([2, 0, 3]):
        args, kwargs = pools[index]
        assert args[0] == fixture.state.raw_keys.ptr
        assert args[1] == fixture.state.member_indices.ptr + slot * member_stride
        assert args[2] == fixture.state.block_starts.ptr + slot * starts_stride
        assert args[3] == 0xA000
        assert args[4] == fixture.state.pooled_keys.ptr + slot * pooled_stride
        assert args[5] == blocks_per_slot[slot]
        assert args[6] == RATIO
        assert args[7] == INDEX_DIM
        assert args[8] == INDEX_DIM
        assert args[9] == 100.0
        assert args[10] == 1e-6
        assert kwargs["runtime"] is fixture.runtime

    # One score dispatch per row, each against that row's own slot keys.
    scores = recorder.all("qwen4_exp_qsa_score_f32")
    assert len(scores) == 3
    query_stride = INDEX_HEADS * INDEX_DIM * 4
    score_stride = COMPLETE_BLOCKS * 4
    for row, (args, kwargs) in enumerate(scores):
        slot = [2, 0, 3][row]
        assert args[0] == 0x9000 + row * query_stride
        assert args[1] == fixture.state.pooled_keys.ptr + slot * pooled_stride
        assert args[2] == fixture.state.scores.ptr + row * score_stride
        assert args[3] == 1
        assert args[4] == blocks_per_slot[slot]
        assert args[5] == INDEX_HEADS
        assert args[6] == INDEX_DIM
        assert "parent" not in kwargs

    # One selection dispatch for every row.
    (args, kwargs) = recorder.one("qwen4_exp_qsa_topk_expand_rows_f32_i64")
    assert args[0] == fixture.state.scores.ptr
    assert args[1] == fixture.state.positions.ptr
    assert args[2] == fixture.state.selected_positions.ptr
    assert args[3] == fixture.state.selected_counts.ptr
    assert args[4] == 3
    assert args[5] == COMPLETE_BLOCKS
    assert args[6] == OUTPUT_STRIDE
    assert args[7] == RATIO
    assert args[8] == BUDGET

    assert counts.tolist() == [3, 3, 3]
    assert positions.tolist() == expected[:, :3].tolist()
    assert fixture.state.pooled_count_host.tolist() == [4, 0, 5, 4]
    assert recorder.names().count("qwen4_exp_qsa_pool_norm_rope_f32") == 3
    assert recorder.names()[-1] == "qwen4_exp_qsa_topk_expand_rows_f32_i64"


def test_select_active_pools_only_new_blocks(
    fixture: Fixture, recorder: Recorder, monkeypatch
) -> None:
    _fill(fixture.state, 0, 8, pooled=2)
    fixture.admit([0], [7])
    _readback(
        monkeypatch,
        counts=1,
        positions=np.arange(OUTPUT_STRIDE, dtype=np.int64).reshape(1, OUTPUT_STRIDE),
    )
    fixture.state.select_active(
        0x9000, key_norm_weight_ptr=0xA000, rotary_dim=INDEX_DIM, theta=100.0
    )
    (pool_args, _) = recorder.one("qwen4_exp_qsa_pool_norm_rope_f32")
    # Only the two newly completed blocks are pooled, at their own offset.
    assert pool_args[5] == 2
    assert pool_args[1] == fixture.state.member_indices.ptr + 2 * RATIO * 4
    assert pool_args[4] == fixture.state.pooled_keys.ptr + 2 * INDEX_DIM * 4
    assert fixture.state.pooled_count_host.tolist() == [4, 0, 0, 0]

    # No new complete block: the second call dispatches no pool at all.
    recorder.calls.clear()
    _fill(fixture.state, 0, 9)
    fixture.admit([0], [8])
    fixture.state.select_active(
        0x9000, key_norm_weight_ptr=0xA000, rotary_dim=INDEX_DIM, theta=100.0
    )
    assert recorder.names() == [
        "qwen4_exp_qsa_score_f32",
        "qwen4_exp_qsa_topk_expand_rows_f32_i64",
    ]


def test_select_active_rejects_dense_rows_and_malformed_calls(
    fixture: Fixture, recorder: Recorder
) -> None:
    _fill(fixture.state, 0, 6)  # 3 blocks, exactly the budget
    _fill(fixture.state, 1, 7)  # 3 blocks, exactly the budget
    fixture.admit([0, 1], [5, 6])
    cases = [
        (dict(prepared_queries_ptr=0), ValueError, "prepared_queries_ptr"),
        (dict(key_norm_weight_ptr=0), ValueError, "key_norm_weight_ptr"),
        (dict(rotary_dim=INDEX_DIM + 2), ValueError, "exceeds index_dim"),
        (dict(rotary_dim=3), ValueError, "must be even"),
        (dict(theta=0.0), ValueError, "theta"),
        (dict(theta=float("nan")), ValueError, "theta"),
        (dict(eps=-1.0), ValueError, "eps"),
    ]
    for overrides, error, match in cases:
        kwargs = dict(
            prepared_queries_ptr=0x9000,
            key_norm_weight_ptr=0xA000,
            rotary_dim=INDEX_DIM,
            theta=100.0,
        )
        kwargs.update(overrides)
        with pytest.raises(error, match=match):
            fixture.state.select_active(**kwargs)
        assert recorder.calls == [], overrides

    # A row inside the dense equivalence bound is refused before any dispatch.
    with pytest.raises(ValueError, match="not more than the budget 3"):
        fixture.state.select_active(
            0x9000, key_norm_weight_ptr=0xA000, rotary_dim=INDEX_DIM, theta=100.0
        )
    assert recorder.calls == []

    # Stale published metadata is re-derived and rejected.
    _fill(fixture.state, 0, 9)
    fixture.attention.prepare([0], [8])
    fixture.state.prepare([0])
    fixture.state.active = SimpleNamespace(
        rows=1,
        slot_ids=(0,),
        slots=np.asarray([0], dtype=np.int64),
        positions=np.asarray([7], dtype=np.int64),
        blocks=np.asarray([4], dtype=np.int64),
        row_of=lambda slot: 0,
    )
    with pytest.raises(ValueError, match="published row metadata"):
        fixture.state.select_active(
            0x9000, key_norm_weight_ptr=0xA000, rotary_dim=INDEX_DIM, theta=100.0
        )
    assert recorder.calls == []


def test_row_order_follows_the_slot(fixture: Fixture, recorder: Recorder, monkeypatch) -> None:
    _fill(fixture.state, 0, 8)
    _fill(fixture.state, 2, 10)
    fixture.admit([2, 0], [9, 7])
    _readback(
        monkeypatch, counts=2, positions=np.zeros((2, OUTPUT_STRIDE), np.int64)
    )
    fixture.state.select_active(
        0x9000, key_norm_weight_ptr=0xA000, rotary_dim=INDEX_DIM, theta=100.0
    )
    pooled_stride = COMPLETE_BLOCKS * INDEX_DIM * 4
    scores = recorder.all("qwen4_exp_qsa_score_f32")
    assert [args[1] for args, _ in scores] == [
        fixture.state.pooled_keys.ptr + 2 * pooled_stride,
        fixture.state.pooled_keys.ptr + 0 * pooled_stride,
    ]
    assert [args[4] for args, _ in scores] == [5, 4]

    # The same slots in the opposite row order swap the per-row bases.
    recorder.calls.clear()
    fixture.attention.prepare([0, 2], [7, 9])
    fixture.state.prepare([0, 2])
    fixture.state.select_active(
        0x9000, key_norm_weight_ptr=0xA000, rotary_dim=INDEX_DIM, theta=100.0
    )
    scores = recorder.all("qwen4_exp_qsa_score_f32")
    assert [args[1] for args, _ in scores] == [
        fixture.state.pooled_keys.ptr + 0 * pooled_stride,
        fixture.state.pooled_keys.ptr + 2 * pooled_stride,
    ]
    assert [args[2] for args, _ in scores] == [
        fixture.state.scores.ptr,
        fixture.state.scores.ptr + COMPLETE_BLOCKS * 4,
    ]


def test_partition_rows_splits_dense_and_sparse(fixture: Fixture) -> None:
    _fill(fixture.state, 0, 6)   # 3 blocks == budget -> dense
    _fill(fixture.state, 1, 8)   # 4 blocks -> sparse
    _fill(fixture.state, 2, 12)  # 6 blocks -> sparse
    fixture.admit([0, 1, 2], [5, 7, 11])
    assert fixture.state.partition_rows() == ((0,), (1, 2))


def _counters(state: Qwen4ExpBatchQsaIndexState) -> tuple:
    return (
        state.cursor_host.tolist(),
        state.pooled_count_host.tolist(),
        state.reset_generation.tolist(),
        state.pending_rollback,
        state.active,
    )


def test_begin_rollback_issues_an_immutable_single_use_token(fixture: Fixture) -> None:
    _fill(fixture.state, 0, 4)
    _fill(fixture.state, 1, 6)
    token = fixture.state.begin_rollback()
    assert isinstance(token, Qwen4ExpBatchQsaIndexRollback)
    # Counters are immutable tuples, so the token cannot be mutated into false
    # provenance after it was issued.
    assert token.cursors == (4, 6, 0, 0)
    assert token.pooled_counts == (2, 3, 0, 0)
    assert token.generation == (0, 0, 0, 0)
    assert token.owner == fixture.state.identity
    with pytest.raises(TypeError):
        token.cursors[0] = 99  # type: ignore[index]
    with pytest.raises(AttributeError):
        token.cursors = (0, 0, 0, 0)  # type: ignore[misc]

    fixture.state.cursor_host[:] = [5, 7, 0, 0]
    fixture.state.pooled_count_host[:] = [2, 3, 0, 0]
    fixture.admit([0], [4])
    fixture.state.rollback(token)
    assert fixture.state.cursor_host.tolist() == [4, 6, 0, 0]
    assert fixture.state.pooled_count_host.tolist() == [2, 3, 0, 0]
    assert fixture.state.active is None
    assert fixture.state.pending_rollback is None
    with pytest.raises(RuntimeError, match="no active rows"):
        fixture.state.select_active(
            0x9000, key_norm_weight_ptr=0xA000, rotary_dim=INDEX_DIM, theta=100.0
        )


def test_begin_rollback_rejects_a_second_live_token(fixture: Fixture) -> None:
    _fill(fixture.state, 0, 4)
    first = fixture.state.begin_rollback()
    with pytest.raises(RuntimeError, match="already pending"):
        fixture.state.begin_rollback()
    assert fixture.state.pending_rollback is first
    # Discarding frees the interface without rewinding anything.
    assert fixture.state.discard_rollback() is True
    assert fixture.state.pending_rollback is None
    assert fixture.state.cursor_host.tolist() == [4, 0, 0, 0]
    assert fixture.state.discard_rollback() is False
    fixture.state.begin_rollback()


def test_rollback_token_is_single_use_across_a_rewind_and_overwrite(
    fixture: Fixture,
) -> None:
    _fill(fixture.state, 0, 4)
    token = fixture.state.begin_rollback()
    fixture.state.cursor_host[0] = 5
    fixture.state.rollback(token)
    assert fixture.state.cursor_host.tolist() == [4, 0, 0, 0]

    # Overwrite the rewound position with a different key, then try to re-apply
    # the consumed token: it must be refused and must change nothing.
    fixture.state.cursor_host[0] = 6
    before = _counters(fixture.state)
    with pytest.raises(RuntimeError, match="already used"):
        fixture.state.rollback(token)
    assert _counters(fixture.state) == before
    # A fresh token is the only way forward, and it captures the new state.
    fresh = fixture.state.begin_rollback()
    assert fresh.cursors == (6, 0, 0, 0)
    assert fresh is not token


def test_rollback_rejects_a_token_from_another_owner() -> None:
    runtime = FakeRuntime()
    attention = Qwen4ExpBatchAttentionState.allocate(
        state_slots=SLOTS,
        max_positions=MAX_POSITIONS,
        block_size=BLOCK_SIZE,
        kv_heads=1,
        head_dim=4,
        runtime=runtime,
    )
    first = Qwen4ExpBatchQsaIndexState.allocate(
        attention_state=attention,
        index_heads=INDEX_HEADS,
        index_dim=INDEX_DIM,
        compression_ratio=RATIO,
        block_budget=BUDGET,
        runtime=runtime,
    )
    second = Qwen4ExpBatchQsaIndexState.allocate(
        attention_state=attention,
        index_heads=INDEX_HEADS,
        index_dim=INDEX_DIM,
        compression_ratio=RATIO,
        block_budget=BUDGET,
        runtime=runtime,
    )
    try:
        assert first.identity != second.identity
        _fill(first, 0, 4)
        token = first.begin_rollback()
        _fill(second, 0, 9)
        before = _counters(second)
        # Same geometry, same counters shape: only owner identity separates them.
        with pytest.raises(ValueError, match="different index state"):
            second.rollback(token)
        assert _counters(second) == before
        # The token is still valid on the owner that issued it.
        first.cursor_host[0] = 5
        first.rollback(token)
        assert first.cursor_host.tolist() == [4, 0, 0, 0]
    finally:
        second.close()
        first.close()
        attention.close()


def test_rollback_rejects_a_token_from_before_a_slot_reset(fixture: Fixture) -> None:
    _fill(fixture.state, 0, 4)
    _fill(fixture.state, 1, 6)
    token = fixture.state.begin_rollback()
    fixture.state.reset_slot(1)
    assert fixture.state.reset_generation.tolist() == [0, 1, 0, 0]
    _fill(fixture.state, 1, 3)
    before = _counters(fixture.state)
    with pytest.raises(ValueError, match=r"slot\(s\) \[1\] were reset"):
        fixture.state.rollback(token)
    # A stale token must not re-expose the previous request's counters.
    assert _counters(fixture.state) == before
    assert fixture.state.pending_rollback is token
    # The caller is told to make a decision rather than being silently freed.
    with pytest.raises(RuntimeError, match="already pending"):
        fixture.state.begin_rollback()
    fixture.state.discard_rollback()
    fixture.state.begin_rollback()


def test_rollback_rejects_moving_a_slot_forward(fixture: Fixture) -> None:
    _fill(fixture.state, 0, 4)
    _fill(fixture.state, 2, 5)
    token = fixture.state.begin_rollback()
    forged = Qwen4ExpBatchQsaIndexRollback(
        owner=token.owner,
        generation=token.generation,
        cursors=(9, 0, 0, 0),
        pooled_counts=(4, 0, 0, 0),
    )
    fixture.state.pending_rollback = forged
    before = _counters(fixture.state)
    with pytest.raises(ValueError, match=r"move slot\(s\) \[0\] forward"):
        fixture.state.rollback(forged)
    assert _counters(fixture.state) == before


def test_invalid_rollbacks_leave_every_counter_untouched(fixture: Fixture) -> None:
    _fill(fixture.state, 0, 4)
    fixture.admit([0], [3])
    token = fixture.state.begin_rollback()
    cases = [
        (SimpleNamespace(cursors=(0, 0, 0, 0)), TypeError, "Qwen4ExpBatchQsaIndexRollback"),
        (
            Qwen4ExpBatchQsaIndexRollback(
                owner=token.owner + 1,
                generation=token.generation,
                cursors=token.cursors,
                pooled_counts=token.pooled_counts,
            ),
            ValueError,
            "different index state",
        ),
        (
            Qwen4ExpBatchQsaIndexRollback(
                owner=token.owner,
                generation=(0, 0, 0, 5),
                cursors=token.cursors,
                pooled_counts=token.pooled_counts,
            ),
            ValueError,
            "were reset",
        ),
        (
            Qwen4ExpBatchQsaIndexRollback(
                owner=token.owner,
                generation=token.generation,
                cursors=token.cursors + (0,),
                pooled_counts=token.pooled_counts,
            ),
            ValueError,
            "must carry 4 cursors",
        ),
        # A short pooled tuple must be refused before the cursor write, or the
        # rejection would itself leave a half-applied rollback behind.
        (
            Qwen4ExpBatchQsaIndexRollback(
                owner=token.owner,
                generation=token.generation,
                cursors=(2, 0, 0, 0),
                pooled_counts=(1,),
            ),
            ValueError,
            "must carry 4 cursors",
        ),
        (
            Qwen4ExpBatchQsaIndexRollback(
                owner=token.owner,
                generation=token.generation,
                cursors=(1.5, 0, 0, 0),
                pooled_counts=token.pooled_counts,
            ),
            ValueError,
            "must be integer tuples",
        ),
    ]
    before = _counters(fixture.state)
    for bad, error, match in cases:
        with pytest.raises(error, match=match):
            fixture.state.rollback(bad)
        assert _counters(fixture.state) == before
    # The pending token is still usable after every refusal.
    fixture.state.rollback(token)
    assert fixture.state.cursor_host.tolist() == [4, 0, 0, 0]


def test_reset_slot_releases_one_slot_for_reuse(fixture: Fixture) -> None:
    _fill(fixture.state, 0, 4)
    _fill(fixture.state, 1, 6)
    fixture.admit([0, 1], [3, 5])
    fixture.state.reset_slot(1)
    assert fixture.state.cursor_host.tolist() == [4, 0, 0, 0]
    assert fixture.state.pooled_count_host.tolist() == [2, 0, 0, 0]
    assert fixture.state.reset_generation.tolist() == [0, 1, 0, 0]
    assert fixture.state.active is None
    with pytest.raises(ValueError, match="outside"):
        fixture.state.reset_slot(SLOTS)
    with pytest.raises(TypeError, match="integer"):
        fixture.state.reset_slot(True)


def test_pointer_helpers_validate_their_arguments(fixture: Fixture) -> None:
    state = fixture.state
    # Slot 1's logical position 3 lives at its own global physical position.
    assert state.raw_position_ptr(1, 3) == state.raw_keys.ptr + (
        SLOT_CAPACITY + 3
    ) * INDEX_DIM * 4
    assert state.pooled_slot_ptr(2) == state.pooled_keys.ptr + (
        2 * COMPLETE_BLOCKS * INDEX_DIM * 4
    )
    with pytest.raises(ValueError, match="outside"):
        state.raw_position_ptr(0, SLOT_CAPACITY)
    with pytest.raises(ValueError, match="outside"):
        state.raw_position_ptr(SLOTS, 0)
    with pytest.raises(TypeError, match="integer"):
        state.pooled_slot_ptr(0.5)
    _fill(state, 0, 8)
    fixture.admit([0], [7])
    assert state.selected_positions_row_ptr(0) == state.selected_positions.ptr
    assert state.selected_counts_row_ptr(0) == state.selected_counts.ptr
    with pytest.raises(ValueError, match="outside"):
        state.selected_positions_row_ptr(1)


def test_close_frees_every_buffer_once_and_retries_a_partial_failure() -> None:
    runtime = FakeRuntime()
    attention = Qwen4ExpBatchAttentionState.allocate(
        state_slots=2,
        max_positions=MAX_POSITIONS,
        block_size=BLOCK_SIZE,
        kv_heads=1,
        head_dim=4,
        runtime=runtime,
    )
    state = Qwen4ExpBatchQsaIndexState.allocate(
        attention_state=attention,
        index_heads=INDEX_HEADS,
        index_dim=INDEX_DIM,
        compression_ratio=RATIO,
        block_budget=BUDGET,
        runtime=runtime,
    )
    owned = sorted(
        [
            state.raw_keys.ptr,
            state.member_indices.ptr,
            state.block_starts.ptr,
            state.pooled_keys.ptr,
            state.scores.ptr,
            state.positions.ptr,
            state.selected_positions.ptr,
            state.selected_counts.ptr,
        ]
    )
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
    assert all(ptr in runtime.allocations for ptr in owned)

    state.close()
    assert state.closed is True
    assert sorted(runtime.freed) == owned
    state.close()
    assert sorted(runtime.freed) == owned
    assert state.pending_rollback is None
    # Closing the mirrored attention owner poisons the index owner too.
    attention.close()
    with pytest.raises(RuntimeError, match="closed"):
        state.begin_rollback()


# -- packed per-step metadata ------------------------------------------------


def _packed_fixture() -> tuple[
    Qwen4ExpBatchStepMetadata, Qwen4ExpBatchAttentionState, Qwen4ExpBatchQsaIndexState
]:
    """An attention/index pair publishing through one packed metadata buffer."""

    runtime = FakeRuntime()
    metadata = Qwen4ExpBatchStepMetadata.allocate(
        layers=1, state_slots=SLOTS, runtime=runtime
    )
    views = metadata.views(0)
    attention = Qwen4ExpBatchAttentionState.allocate(
        state_slots=SLOTS,
        max_positions=MAX_POSITIONS,
        block_size=BLOCK_SIZE,
        kv_heads=1,
        head_dim=4,
        runtime=runtime,
        position_view=views.position,
        context_view=views.context,
    )
    state = Qwen4ExpBatchQsaIndexState.allocate(
        attention_state=attention,
        index_heads=INDEX_HEADS,
        index_dim=INDEX_DIM,
        compression_ratio=RATIO,
        block_budget=BUDGET,
        runtime=runtime,
        position_view=views.index_position,
    )
    return metadata, attention, state


def test_publish_through_packed_views_reads_the_packed_query_positions() -> None:
    metadata, attention, state = _packed_fixture()
    try:
        views = metadata.views(0)
        _fill(state, 3, 5)
        _fill(state, 0, 4)
        metadata.stage(
            0,
            positions=[4, 4],
            contexts=[5, 5],
            index_positions=[4, 3],
        )
        metadata.upload()
        attention.prepare([3, 0], [4, 4], packed=views)
        copies_before = len(state.runtime.copies)
        active = state.prepare([3, 0], packed=views)
        assert active.rows == 2
        assert active.slot_ids == (3, 0)
        assert active.positions.tolist() == [4, 3]
        assert active.blocks.tolist() == [2, 2]
        # The published query positions are the packed view, and publishing them
        # cost no upload of its own.
        assert active.positions_device.ptr == views.index_position.ptr
        assert len(state.runtime.copies) == copies_before
        # Selection reads that same device vector, not the owner's own buffer.
        assert state.positions.ptr != views.index_position.ptr
    finally:
        state.close()
        attention.close()
        metadata.close()


def test_select_active_dispatches_the_packed_query_positions(
    recorder: Recorder, monkeypatch
) -> None:
    metadata, attention, state = _packed_fixture()
    try:
        views = metadata.views(0)
        _fill(state, 2, 8)
        metadata.stage(0, positions=[7], contexts=[8], index_positions=[7])
        metadata.upload()
        attention.prepare([2], [7], packed=views)
        state.prepare([2], packed=views)
        _readback(monkeypatch, counts=1, positions=np.zeros((1, OUTPUT_STRIDE), dtype=np.int64))
        state.select_active(
            0x9000, key_norm_weight_ptr=0xA000, rotary_dim=INDEX_DIM, theta=100.0
        )
        (args, _) = recorder.one("qwen4_exp_qsa_topk_expand_rows_f32_i64")
        assert args[1] == views.index_position.ptr
        assert args[1] != state.positions.ptr
    finally:
        state.close()
        attention.close()
        metadata.close()


def test_publish_rejects_a_malformed_index_plan_before_uploading() -> None:
    metadata, attention, state = _packed_fixture()
    try:
        plan = state.validate_rows([1, 0])
        copies_before = len(state.runtime.copies)
        bad_plans = [
            Qwen4ExpBatchQsaIndexRowPlan(
                slot_ids=(1, 0),
                slots=plan.slots,
                positions=plan.positions[:1],
                blocks=plan.blocks,
            ),
            Qwen4ExpBatchQsaIndexRowPlan(
                slot_ids=(1, 1),
                slots=plan.slots,
                positions=plan.positions,
                blocks=plan.blocks,
            ),
            Qwen4ExpBatchQsaIndexRowPlan(
                slot_ids=(1, SLOTS),
                slots=np.asarray([1, SLOTS], dtype=np.int64),
                positions=plan.positions,
                blocks=plan.blocks,
            ),
            Qwen4ExpBatchQsaIndexRowPlan(
                slot_ids=(1, 0),
                slots=np.asarray([0, 1], dtype=np.int64),
                positions=plan.positions,
                blocks=plan.blocks,
            ),
            Qwen4ExpBatchQsaIndexRowPlan(
                slot_ids=(),
                slots=np.zeros(0, dtype=np.int64),
                positions=np.zeros(0, dtype=np.int64),
                blocks=np.zeros(0, dtype=np.int64),
            ),
        ]
        for bad in bad_plans:
            with pytest.raises(ValueError):
                state.publish(bad)
            assert len(state.runtime.copies) == copies_before
            assert state.active is None
        with pytest.raises(TypeError, match="validate_rows"):
            state.publish({"slot_ids": (1,)})  # type: ignore[arg-type]
        # A packed view of the wrong width is refused before the publish too.
        other = Qwen4ExpBatchStepMetadata.allocate(
            layers=1, state_slots=SLOTS + 1, runtime=state.runtime
        )
        try:
            with pytest.raises(ValueError, match="one int64 per state slot"):
                state.publish(plan, packed=other.views(0))
            assert state.active is None
            assert len(state.runtime.copies) == copies_before
        finally:
            other.close()
        # The same plan publishes once it is well formed.
        assert state.publish(plan).slot_ids == (1, 0)
    finally:
        state.close()
        attention.close()
        metadata.close()


def test_index_allocate_rejects_a_packed_view_of_the_wrong_width() -> None:
    runtime = FakeRuntime()
    metadata = Qwen4ExpBatchStepMetadata.allocate(
        layers=1, state_slots=SLOTS + 1, runtime=runtime
    )
    attention = Qwen4ExpBatchAttentionState.allocate(
        state_slots=SLOTS,
        max_positions=MAX_POSITIONS,
        block_size=BLOCK_SIZE,
        kv_heads=1,
        head_dim=4,
        runtime=runtime,
    )
    try:
        with pytest.raises(ValueError, match="one int64 per state slot"):
            Qwen4ExpBatchQsaIndexState.allocate(
                attention_state=attention,
                index_heads=INDEX_HEADS,
                index_dim=INDEX_DIM,
                compression_ratio=RATIO,
                block_budget=BUDGET,
                runtime=runtime,
                position_view=metadata.views(0).index_position,
            )
    finally:
        attention.close()
        metadata.close()


# -- host-source ownership ---------------------------------------------------


def test_index_allocate_routes_metadata_uploads_through_the_owning_array_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, np.ndarray, int | None]] = []
    real_upload = batch_index.copy_host_array_to_device

    def recording_upload(buffer, array, nbytes=None, *, runtime=None):
        calls.append((buffer, array, nbytes))
        real_upload(buffer, array, nbytes, runtime=runtime)

    monkeypatch.setattr(
        batch_index, "copy_host_array_to_device", recording_upload
    )
    fixture = Fixture()
    try:
        assert len(calls) == 3
        for _, array, nbytes in calls:
            assert isinstance(array, np.ndarray)
            assert array.flags.c_contiguous
            assert nbytes is None or nbytes == array.nbytes
    finally:
        fixture.close()


def test_publish_contiguizes_a_strided_positions_plan_before_upload(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fill(fixture.state, 0, 4)
    _fill(fixture.state, 1, 5)
    plan = fixture.state.validate_rows([0, 1])
    # Same values, non-contiguous stride: the old inline
    # ``np.ascontiguousarray`` was the only owner of the upload source.
    strided = Qwen4ExpBatchQsaIndexRowPlan(
        slot_ids=plan.slot_ids,
        slots=plan.slots,
        positions=np.repeat(plan.positions, 2)[::2],
        blocks=plan.blocks,
    )
    calls: list[tuple[object, np.ndarray, int | None]] = []
    real_upload = batch_index.copy_host_array_to_device

    def recording_upload(buffer, array, nbytes=None, *, runtime=None):
        calls.append((buffer, array, nbytes))
        real_upload(buffer, array, nbytes, runtime=runtime)

    monkeypatch.setattr(
        batch_index, "copy_host_array_to_device", recording_upload
    )
    active = fixture.state.publish(strided)
    assert active.positions.tolist() == plan.positions.tolist()
    assert len(calls) == 1
    uploaded = calls[0][1]
    assert uploaded.flags.c_contiguous
    assert uploaded.nbytes == plan.positions.nbytes
    assert uploaded.tolist() == plan.positions.tolist()
    assert calls[0][2] == plan.positions.nbytes


def test_map_slot_upload_preserves_its_device_offset_and_uses_an_owner(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[object, np.ndarray, int | None]] = []
    real_upload = batch_index.copy_host_array_to_device

    def recording_upload(buffer, array, nbytes=None, *, runtime=None):
        calls.append((buffer, array, nbytes))
        real_upload(buffer, array, nbytes, runtime=runtime)

    monkeypatch.setattr(
        batch_index, "copy_host_array_to_device", recording_upload
    )
    fixture.state.map_slot(1, 2)
    assert len(calls) == 1
    destination, uploaded, nbytes = calls[0]
    complete = fixture.state.complete_blocks
    row_bytes = fixture.state.compression_ratio * 4
    expected_offset = 1 * complete * row_bytes
    assert destination.ptr == fixture.state.member_indices.ptr + expected_offset
    assert destination.nbytes == complete * row_bytes
    assert nbytes == complete * row_bytes
    assert uploaded.flags.c_contiguous
    assert uploaded.nbytes == complete * row_bytes
    assert uploaded is not fixture.state.member_host[1, :complete]
    assert uploaded.tolist() == fixture.state.member_host[1, :complete].tolist()
