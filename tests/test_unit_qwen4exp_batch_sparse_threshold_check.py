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
   selections.

These tests are CPU-only: no HIP runtime, no GPU, no model.
"""

import numpy as np

from scripts.qwen4exp_batch_sparse_threshold_check import (
    EAGER_PROFILE_ENV,
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


def test_eager_profile_forces_both_graph_switches():
    """The reference arm must not keep c1's single-request graph capture on."""

    environ = {"HIPENGINE_UNRELATED": "keep"}
    applied = _apply_eager_profile(environ)
    assert applied == EAGER_PROFILE_ENV
    assert environ["HIPENGINE_QWEN4_EXP_MOE_GRAPH"] == "0"
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
