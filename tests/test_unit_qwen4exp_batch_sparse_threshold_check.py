"""Unit regression for the sparse-threshold diagnostic's reporting and indexing.

The diagnostic in ``scripts/qwen4exp_batch_sparse_threshold_check.py`` produced
four wrong reports before it produced a right one, and each was a pure
indexing/keying mistake that a CPU test can pin:

1. the c1 reference arm ran with graph capture left on, so the comparison mixed
   two execution profiles;
2. the sparse score tap read *mixer* rows while ``select_active`` writes
   *sparse-subset* rows, which fabricated 12 "policy mismatch" entries in a mixed
   dense/sparse step;
3. the idle-slot digest read the shared KV slab from block 0 instead of the
   slot's own window, so a neighbour's writes looked like corruption;
4. the composition baseline was keyed by slot, but every single-row record
   carries slot 0, so the second prompt compared against the first prompt's
   selections;
5. the selection tap always read the batch-wide host staging mirror, which an
   all-sparse step no longer fills: that step hands the index owner's own device
   buffers to the attention kernel, so the tap recorded count 0 and an empty
   selection for every row.

These tests are CPU-only: no HIP runtime, no GPU, no model.
"""

import numpy as np
import pytest

import scripts.qwen4exp_batch_sparse_threshold_check as threshold_check
from scripts.qwen4exp_batch_sparse_threshold_check import (
    EAGER_MOE_GRAPH_ENABLED,
    EAGER_PROFILE_ENV,
    Taps,
    _apply_eager_profile,
    _baseline_for,
    _block_margin,
    _compare_selection,
    _physical_positions,
    _selection_baseline_key,
    _slot_window_offset,
    _sparse_subset_row,
)

RATIO = 4
BUDGET = 2


def _record(*, scores, selected, position=11, count=12, layer=7, slot=0, blocks=3):
    """One synthetic sparse comparison record with the fields the checker reads."""

    dim = 2
    heads = 1
    return {
        "side": "synthetic",
        "layer": layer,
        "slot": slot,
        "row": slot,
        "position": position,
        "count": count,
        "blocks": blocks,
        "sparse": True,
        "selected_count": int(len(selected)),
        "query": np.zeros(heads * dim, dtype=np.float32),
        "raw_prefix": np.zeros((count, dim), dtype=np.float32),
        "pooled": np.zeros((blocks, dim), dtype=np.float32),
        "starts": (np.arange(blocks, dtype=np.int64) * RATIO),
        "scores": np.asarray(scores, dtype=np.float32),
        "selected": np.asarray(selected, dtype=np.int64),
    }


def test_sparse_score_rows_are_indexed_by_subset_position():
    """A mixed step scores only its sparse rows, in that group's own order."""

    # Rows (0, 2) are dense; the sparse group is (1, 3) and owns score rows 0, 1.
    sparse_group = (1, 3)
    assert _sparse_subset_row(sparse_group, 1) == 0
    assert _sparse_subset_row(sparse_group, 3) == 1
    # The pre-repair behaviour: the mixer row index is not the subset index.
    assert _sparse_subset_row(sparse_group, 3) != 3
    # A fully sparse step keeps the mapping at identity, which is why a
    # single-row run could not expose the bug.
    assert _sparse_subset_row((0, 1), 1) == 1


def test_sparse_subset_row_rejects_a_dense_row():
    """Asking for a dense row is a caller error, not a silent score row."""

    try:
        _sparse_subset_row((1, 3), 2)
    except ValueError:
        return
    raise AssertionError("a dense row must not map to a sparse score row")


def test_composition_baseline_is_keyed_by_prompt_not_slot():
    """Every single-row record carries slot 0, so the key must name the prompt."""

    first = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    second = np.array([0, 1, 2, 3, 8, 9, 10, 11], dtype=np.int64)
    baselines = {
        _selection_baseline_key(0, layer=3, position=2052): {"batch": first},
        _selection_baseline_key(1, layer=3, position=2052): {"batch": second},
    }
    assert _baseline_for(baselines, slot=1, layer=3, position=2052)["batch"] is second
    assert _baseline_for(baselines, slot=0, layer=3, position=2052)["batch"] is first
    # A slot-keyed baseline would hand slot 1 the first prompt's selection.
    assert _baseline_for(baselines, slot=1, layer=3, position=2052)["batch"] is not first
    assert _baseline_for(baselines, slot=1, layer=4, position=2052) is None


def test_slot_window_offset_keeps_slots_disjoint():
    """Each slot's KV window starts at its own block range, not at block 0."""

    block_bytes = 256 * 2 * 2 * 2  # block_size x kv_heads x head_dim x 2 bytes
    blocks_per_request = 9
    assert _slot_window_offset(0, blocks_per_request, block_bytes) == 0
    assert _slot_window_offset(1, blocks_per_request, block_bytes) == 9 * block_bytes
    assert _slot_window_offset(2, blocks_per_request, block_bytes) == 18 * block_bytes
    used_bytes = 6 * block_bytes
    assert (
        _slot_window_offset(0, blocks_per_request, block_bytes) + used_bytes
        <= _slot_window_offset(1, blocks_per_request, block_bytes)
    )


def test_eager_profile_pins_moe_graphs_per_runner_and_layer_graphs_by_env():
    """The reference arm must not keep c1's single-request graph capture on."""

    environ = {"HIPENGINE_UNRELATED": "keep"}
    applied = _apply_eager_profile(environ)
    assert applied == EAGER_PROFILE_ENV
    assert EAGER_PROFILE_ENV == {"HIPENGINE_QWEN4_EXP_LAYER_GRAPHS": "0"}
    # The MoE section has no env switch any more; the harness pins it per runner.
    assert EAGER_MOE_GRAPH_ENABLED is False
    assert "HIPENGINE_QWEN4_EXP_MOE_GRAPH" not in environ
    assert environ["HIPENGINE_QWEN4_EXP_LAYER_GRAPHS"] == "0"
    assert environ["HIPENGINE_UNRELATED"] == "keep"


def test_physical_positions_gathers_through_the_page_table():
    """Logical positions map through the block table, never as identity."""

    table = np.array([5, 2], dtype=np.int64)
    physical = _physical_positions(table, block_size=4, count=6)
    assert physical.tolist() == [20, 21, 22, 23, 8, 9]
    # Two slots with disjoint block ranges never address the same row.
    other = _physical_positions(np.array([9, 6], dtype=np.int64), 4, 6)
    assert not set(physical.tolist()) & set(other.tolist())


def test_block_margin_is_the_cut_gap():
    assert _block_margin(np.array([5.0, 4.0, 1.0], dtype=np.float32), 2) == 3.0
    # Fewer blocks than the budget means nothing was cut.
    assert _block_margin(np.array([1.0, 2.0], dtype=np.float32), 2) == float("inf")


def test_identical_score_bytes_select_identically_and_are_not_a_near_tie():
    """The baseline case: same bytes, same rule, same selection."""

    c1 = _record(scores=[5.0, 4.0, 1.0], selected=[0, 1, 2, 3, 4, 5, 6, 7])
    batch = _record(scores=[5.0, 4.0, 1.0], selected=[0, 1, 2, 3, 4, 5, 6, 7])
    result = _compare_selection(c1, batch, ratio=RATIO, budget=BUDGET)
    assert result["mode"] == "sparse"
    assert result["routing_agrees"]
    assert result["selected_equal"]
    assert result["scores_equal"]
    assert result["policy_c1_matches_host"]
    assert result["policy_batch_matches_host"]
    assert result["cut_is_near_tie"] is False


def test_a_selection_difference_is_classified_against_the_cut_margin():
    """A flip at a cut the drift can reach is reported as a near tie."""

    c1 = _record(scores=[10.0, 9.0, 1.0], selected=[0, 1, 2, 3, 4, 5, 6, 7])
    # Block 2 crosses block 1, so the selection moves to blocks {0, 2}.
    batch = _record(scores=[10.0, 2.0, 9.5], selected=[0, 1, 2, 3, 8, 9, 10, 11])
    result = _compare_selection(c1, batch, ratio=RATIO, budget=BUDGET)
    assert not result["selected_equal"]
    assert result["selected_differing_positions"] == 8
    # Both sides still follow the documented rule on their own score bytes.
    assert result["policy_c1_matches_host"]
    assert result["policy_batch_matches_host"]
    assert result["score_max_abs_delta"] == 8.5
    assert result["cut_is_near_tie"] is True


def test_the_near_tie_test_is_conservative_not_decisive():
    """A large unrelated delta marks a comparison near-tie, by construction.

    ``cut_is_near_tie`` compares each side's cut margin with the *largest* score
    delta on that layer, so one large delta anywhere makes the comparison look
    like a near tie. That is why the diagnostic reports the count as supporting
    evidence and rests the policy claim on the host-rule and cross-feed checks.
    """

    c1 = _record(scores=[10.0, 9.0, 1.0], selected=[0, 1, 2, 3, 4, 5, 6, 7])
    # Block 0 moves by 90 and the flip happens on a wide cut (margin 8 vs 0.4),
    # yet the largest delta is large enough to satisfy the near-tie test.
    batch = _record(scores=[100.0, 9.4, 9.5], selected=[0, 1, 2, 3, 8, 9, 10, 11])
    result = _compare_selection(c1, batch, ratio=RATIO, budget=BUDGET)
    assert not result["selected_equal"]
    assert result["block_margin_c1"] == 8.0
    assert result["score_max_abs_delta"] == 90.0
    assert result["cut_is_near_tie"] is True


# -- the selection tap must read the source the step actually consumed -------

TAP_RATIO = 4
TAP_BUDGET = 2
TAP_STRIDE = 7  # index owner's output_stride for these fake rows
TAP_DIM = 2
TAP_HEADS = 1
DEVICE_COUNTS = 0xD000
DEVICE_POSITIONS = 0xD100
RAW_KEYS = 0xE000
POOLED_KEYS = 0xE100
BLOCK_STARTS = 0xE200
SCORES = 0xE300
INDEX_QUERY = 0xE400
STAGED_MARKER = -5


class _Active:
    def __init__(self, slots: tuple[int, ...]) -> None:
        self.slots = slots
        self.rows = len(slots)


class _Buffer:
    def __init__(self, ptr: int) -> None:
        self.ptr = ptr


class _FakeIndexState:
    """The fields ``Taps._batch_records`` reads, with no device behind them."""

    def __init__(self, *, slots, cursors, state_slots=8, slot_capacity_tokens=64) -> None:
        self.active = _Active(tuple(slots))
        self.cursor_host = dict(cursors)
        self.state_slots = state_slots
        self.slot_capacity_tokens = slot_capacity_tokens
        self.compression_ratio = TAP_RATIO
        self.block_budget = TAP_BUDGET
        self.complete_blocks = 8
        self.index_dim = TAP_DIM
        self.index_heads = TAP_HEADS
        self.block_size = 4
        self.output_stride = TAP_STRIDE
        self.raw_keys = _Buffer(RAW_KEYS)
        self.pooled_keys = _Buffer(POOLED_KEYS)
        self.block_starts = _Buffer(BLOCK_STARTS)
        self.scores = _Buffer(SCORES)
        self.attention = type(
            "Attention",
            (),
            {
                # One page table per slot, distinct so ownership stays visible.
                # Blocks map into a slab of ``state_slots * slot_capacity_tokens``
                # positions, which is what the raw-key read reshapes to.
                "block_host": [
                    np.arange(16, dtype=np.int64) + 16 * slot for slot in range(state_slots)
                ]
            },
        )()

    def partition_rows(self):
        blocks = np.array([self.cursor_host[slot] for slot in self.active.slots]) // TAP_RATIO
        sparse = tuple(int(row) for row in np.nonzero(blocks > TAP_BUDGET)[0])
        dense = tuple(int(row) for row in np.nonzero(blocks <= TAP_BUDGET)[0])
        return dense, sparse

    def selected_counts_row_ptr(self, row: int) -> int:
        return DEVICE_COUNTS + int(row) * np.dtype(np.int32).itemsize

    def selected_positions_row_ptr(self, row: int) -> int:
        return DEVICE_POSITIONS + int(row) * TAP_STRIDE * np.dtype(np.int64).itemsize


class _Selection:
    """The batch-wide staging mirror, holding deliberately different bytes."""

    def __init__(self, counts, positions) -> None:
        self.counts_host = np.asarray(counts, dtype=np.int32)
        self.positions_host = np.asarray(positions, dtype=np.int64)


def _device_slabs(rows: int, counts, positions) -> dict[int, np.ndarray]:
    return {
        DEVICE_COUNTS: np.asarray(counts, dtype=np.int32),
        DEVICE_POSITIONS: np.asarray(positions, dtype=np.int64).reshape(rows, TAP_STRIDE),
    }


def _install_reader(monkeypatch, slabs: dict[int, np.ndarray]) -> list[tuple[int, int]]:
    """Serve ``_read`` from fake slabs and record every (ptr, count) it served."""

    reads: list[tuple[int, int]] = []

    def fake_read(runtime, ptr, dtype, count):
        reads.append((int(ptr), int(count)))
        itemsize = np.dtype(dtype).itemsize
        for base, values in slabs.items():
            if base <= ptr < base + values.nbytes:
                offset = (ptr - base) // itemsize
                flat = values.reshape(-1)
                return np.asarray(flat[offset : offset + int(count)], dtype=dtype)
        return np.zeros(int(count), dtype=dtype)

    monkeypatch.setattr(threshold_check, "_read", fake_read)
    return reads


def _tap_records(index_state, selection, rows: int) -> list[dict]:
    taps = Taps()
    taps.runtime = None
    qsa = type("Qsa", (), {"index_query": _Buffer(INDEX_QUERY)})()
    kwargs = {
        "scratch": type("Scratch", (), {"qsa": qsa})(),
        "index_state": index_state,
        "selection": selection,
        "rows": rows,
    }
    weights = type("Weights", (), {"layer_id": 3, "qsa_state_index": 1})()
    return taps._batch_records(weights, kwargs)


def _staged(rows: int, counts) -> _Selection:
    return _Selection(
        counts,
        np.full((rows, TAP_STRIDE), STAGED_MARKER, dtype=np.int64),
    )


def test_all_sparse_records_read_the_index_owners_device_selection(monkeypatch) -> None:
    """Every row sparse: the kernel consumed the device buffers, so read those."""

    # Permuted slots, so the row order is not the slot order.
    slots = (2, 0, 3)
    index_state = _FakeIndexState(slots=slots, cursors={2: 40, 0: 44, 3: 48})
    device_counts = [6, 5, 4]
    device_positions = [
        [100 * (row + 1) + column for column in range(TAP_STRIDE)] for row in range(3)
    ]
    reads = _install_reader(monkeypatch, _device_slabs(3, device_counts, device_positions))
    # Staging holds nothing for this step; reading it would report count 0.
    selection = _staged(3, [0, 0, 0])

    records = _tap_records(index_state, selection, 3)

    assert [record["slot"] for record in records] == [2, 0, 3]
    assert [record["sparse"] for record in records] == [True, True, True]
    assert [record["selected_count"] for record in records] == device_counts
    for row, record in enumerate(records):
        assert record["selected"].tolist() == device_positions[row][: device_counts[row]]
        assert (DEVICE_COUNTS + row * 4, 1) in reads
        assert (DEVICE_POSITIONS + row * TAP_STRIDE * 8, device_counts[row]) in reads


def test_mixed_records_keep_reading_the_staged_host_metadata(monkeypatch) -> None:
    """A mixed step stages every row, so the tap must not bypass the mirror."""

    # Rows 1 and 3 are sparse; the sparse group is (1, 3), not the row set.
    slots = (0, 2, 1, 3)
    index_state = _FakeIndexState(slots=slots, cursors={0: 8, 2: 40, 1: 4, 3: 52})
    assert index_state.partition_rows()[1] == (1, 3)
    # Device buffers hold markers, so any device read is visible in the record.
    reads = _install_reader(monkeypatch, _device_slabs(4, [0] * 4, [[-9] * TAP_STRIDE] * 4))
    staged_counts = [8, 5, 4, 6]
    selection = _staged(4, staged_counts)
    for row in range(4):
        selection.positions_host[row, : staged_counts[row]] = 100 * (row + 1)

    records = _tap_records(index_state, selection, 4)

    assert [record["sparse"] for record in records] == [False, True, False, True]
    sparse = [record for record in records if record["sparse"]]
    assert [record["selected_count"] for record in sparse] == [5, 6]
    assert sparse[0]["selected"].tolist() == [200] * 5
    assert sparse[1]["selected"].tolist() == [400] * 6
    # No device selection read at all: the staging mirror is the consumed source.
    assert not [
        pointer
        for pointer, _ in reads
        if DEVICE_COUNTS <= pointer < DEVICE_COUNTS + 64
        or DEVICE_POSITIONS <= pointer < DEVICE_POSITIONS + 4 * TAP_STRIDE * 8
    ]
    # The score slab still uses the sparse-subset row, which is not the mixer row:
    # mixer row 1 owns subset row 0, mixer row 3 owns subset row 1.
    assert (SCORES + 0 * 8 * 4, 10) in reads
    assert (SCORES + 1 * 8 * 4, 13) in reads
