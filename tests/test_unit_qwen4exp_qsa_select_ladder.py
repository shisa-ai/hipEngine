"""R0162 selector coverage: the exact top-k context ladder and its index state.

The scalable exact device selector (``qsa_topk_expand_f32_i64``, a one-CTA
radix threshold with stable wave-prefix compaction) is the promoted route. This
module owns the CPU half of its coverage so the GPU ladder and the CPU oracle
share one definition of every context the R0162 gate names.

``query_position`` is zero-based: a row whose visible prefix is
``blocks * ratio + tail`` tokens sits at ``blocks * ratio + tail - 1``, and its
incomplete tail is ``(query_position + 1) % ratio``.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

from hipengine.kernels.cpu_reference.qwen4_exp import qsa_select_positions

RATIO = 4
BUDGET = 512

#: Context lengths named by the R0162 gate. ``blocks = ctx // RATIO``.
CONTEXT_LADDER = (2_052, 4_096, 16_384, 65_536, 262_144)

#: ``(blocks, tail)`` rows: every named context complete, plus incomplete tails
#: so the expansion tail arithmetic is exercised off a block boundary.
TIE_CASES = (
    tuple((ctx // RATIO, 0) for ctx in CONTEXT_LADDER)
    + ((513, 3), (1_023, 3), (65_535, 3))
)


def query_position(blocks: int, tail: int) -> int:
    """Zero-based position of the row that has just seen ``blocks*RATIO+tail``."""

    if blocks <= 0 or not 0 <= tail < RATIO:
        raise ValueError("blocks must be positive and tail in 0..ratio-1")
    return blocks * RATIO + tail - 1


def tie_scores(blocks: int) -> np.ndarray:
    """The synthetic tie fixture: a distinct high tail plus exact-zero ties."""

    high = min(124, blocks)
    scores = np.zeros(blocks, dtype=np.float32)
    scores[-high:] = np.arange(1, high + 1, dtype=np.float32)
    return scores


def tie_selected_blocks(blocks: int, budget: int = BUDGET) -> np.ndarray:
    """Lowest block indices fill the budget remainder, then the distinct highs."""

    high = min(124, blocks)
    ties = np.arange(budget - high, dtype=np.int64)
    highs = np.arange(blocks - high, blocks, dtype=np.int64)
    return np.sort(np.concatenate((ties, highs)))


def expansion(blocks: int, tail: int, *, budget: int = BUDGET) -> np.ndarray:
    """Expected sorted token positions: whole blocks, then the incomplete tail."""

    selected = tie_selected_blocks(blocks, budget)
    whole = (selected[:, None] * RATIO + np.arange(RATIO, dtype=np.int64)).reshape(-1)
    if tail:
        start = blocks * RATIO
        return np.concatenate(
            (whole, np.arange(start, start + tail, dtype=np.int64))
        )
    return whole


def test_the_ladder_names_every_context_and_lands_on_its_last_token() -> None:
    for ctx in CONTEXT_LADDER:
        assert ctx % RATIO == 0
        assert query_position(ctx // RATIO, 0) == ctx - 1


def test_every_tie_row_fits_inside_the_named_context() -> None:
    """A tail row past 262,144 would be an off-by-one, not a ladder rung."""

    widest = max(blocks * RATIO + tail for blocks, tail in TIE_CASES)
    assert widest <= CONTEXT_LADDER[-1]


def test_every_tie_row_is_zero_based_and_its_tail_is_off_boundary() -> None:
    for blocks, tail in TIE_CASES:
        position = query_position(blocks, tail)
        assert position >= 0
        assert (position + 1) // RATIO == blocks
        assert (position + 1) % RATIO == tail


def test_the_tie_fixture_always_leaves_more_ties_than_slots() -> None:
    for blocks, _ in TIE_CASES:
        ties = int(np.count_nonzero(tie_scores(blocks) == 0.0))
        assert ties > BUDGET - min(124, blocks)


def test_the_expectation_takes_the_lowest_indices_at_the_threshold() -> None:
    for blocks, _ in TIE_CASES:
        selected = tie_selected_blocks(blocks)
        assert selected.size == BUDGET
        assert np.array_equal(selected[: BUDGET - 124], np.arange(BUDGET - 124))
        assert selected[-1] == blocks - 1


def test_the_expectation_matches_the_cpu_reference() -> None:
    for blocks, tail in TIE_CASES:
        position = query_position(blocks, tail)
        selection = qsa_select_positions(
            tie_scores(blocks)[None],
            np.arange(blocks, dtype=np.int64) * RATIO,
            query_positions=np.asarray([position], dtype=np.int64),
            available_positions=np.arange(blocks * RATIO + tail),
            compression_ratio=RATIO,
            block_budget=BUDGET,
        )
        np.testing.assert_array_equal(
            selection.selected_positions[0], expansion(blocks, tail)
        )


def test_the_device_index_state_keeps_no_host_route_mirror() -> None:
    """Only ``select_positions_host`` read these; its removal takes them with it."""

    from hipengine.runtime.qwen4_exp_runner import Qwen4ExpQSAIndexDeviceState

    assert not hasattr(Qwen4ExpQSAIndexDeviceState, "select_positions_host")
    names = {field.name for field in dataclasses.fields(Qwen4ExpQSAIndexDeviceState)}
    for removed in (
        "block_starts_host",
        "selected_starts_host",
        "selected_count_host",
        "query_position_host",
        "selected_positions_host",
    ):
        assert removed not in names


def test_the_remaining_readers_keep_their_buffers() -> None:
    """``scores_host`` is the retrieval gate's D2H target and the stateful graph
    probe snapshots ``selected_starts``/``query_position``; both stay."""

    from hipengine.runtime.qwen4_exp_runner import Qwen4ExpQSAIndexDeviceState

    names = {field.name for field in dataclasses.fields(Qwen4ExpQSAIndexDeviceState)}
    assert {"scores_host", "member_host", "selected_starts", "query_position"} <= names

    root = pathlib.Path(__file__).resolve().parents[1]
    gate = (root / "scripts" / "qwen4_exp_qsa_retrieval_gate.py").read_text()
    assert "state.scores_host" in gate
    probe = (root / "scripts" / "qwen4exp_stateful_layer_graph_probe.py").read_text()
    assert "'selected_starts'" in probe
    assert "'query_position'" in probe
