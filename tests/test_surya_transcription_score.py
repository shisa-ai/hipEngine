"""Unit tests for the Surya transcription scorer.

Pure: builds ``FullPageBlock`` objects directly, so no model or GPU is needed.
The scorer is what the acceptance test asserts on, so these tests pin the
semantics that make its numbers trustworthy — especially that it cannot be
satisfied by reproducing a model's own output, and that repeated expected lines
consume repeated candidates.
"""

from __future__ import annotations

import pytest

from hipengine.generation.surya_protocol import FullPageBlock, parse_full_page_html
from scripts.surya_transcription_score import (
    TranscriptionThresholds,
    evaluate,
    levenshtein,
    normalize,
    score_transcription,
    similarity,
)


def _block(label: str, text: str, bbox=(0.0, 0.0, 1.0, 1.0)) -> FullPageBlock:
    return FullPageBlock(label=label, bbox=bbox, html=f"<p>{text}</p>", text=text)


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("", "", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("abc", "abc", 0),
        ("abc", "abd", 1),
        ("kitten", "sitting", 3),
        ("line.", "literature", 7),
    ],
)
def test_levenshtein(a: str, b: str, expected: int) -> None:
    assert levenshtein(a, b) == expected


def test_similarity_is_one_for_whitespace_only_differences() -> None:
    assert similarity("  Item   Spec ", "Item Spec") == 1.0


def test_similarity_penalizes_a_wrong_word() -> None:
    assert similarity("... in this line.", "... in this literature") < 0.8


def test_normalize_collapses_whitespace() -> None:
    assert normalize(" a \n b\t c ") == "a b c"


# ---------------------------------------------------------------------------
# line recall, omissions, CER
# ---------------------------------------------------------------------------


def test_perfect_transcription_scores_clean() -> None:
    expected = ("first line", "second line")
    blocks = [_block("Text", "first line"), _block("Text", "second line")]

    score = score_transcription(
        page="p", blocks=blocks, expected_lines=expected, finish_reason="eos"
    )

    assert score.lines_found == 2
    assert score.line_recall == 1.0
    assert score.lines_exact == 2
    assert score.line_exact_rate == 1.0
    assert score.cer == 0.0
    assert score.omissions == []
    assert score.reading_order_violations == 0
    assert not score.truncated


def test_dropped_line_is_an_omission_and_costs_recall() -> None:
    expected = ("first line", "second line")
    blocks = [_block("Text", "first line")]

    score = score_transcription(
        page="p", blocks=blocks, expected_lines=expected, finish_reason="eos"
    )

    assert score.lines_found == 1
    assert score.line_recall == 0.5
    assert score.omissions == ["second line"]
    assert score.line_exact_rate == 0.5
    assert score.cer > 0.0


def test_corrupted_line_is_found_but_not_exact_and_costs_cer() -> None:
    """A read-with-a-wrong-word line is not an omission, but it is not exact.

    This is the real mixed-page failure mode: the model transcribed the region
    and substituted a word.
    """

    expected = ("Summary: the deposition step dominates yield loss in this line.",)
    blocks = [
        _block("Text", "Summary: the deposition step dominates yield loss in this literature")
    ]

    score = score_transcription(
        page="p", blocks=blocks, expected_lines=expected, finish_reason="eos",
    )

    assert score.lines_found == 1
    assert score.line_recall == 1.0
    assert score.lines_exact == 0
    assert score.line_exact_rate == 0.0
    assert score.cer > 0.0
    assert score.omissions == []


def test_garbage_candidate_is_an_omission_not_a_match() -> None:
    expected = ("Summary: the deposition step dominates yield loss in this line.",)
    blocks = [_block("Text", "totally unrelated content")]

    score = score_transcription(
        page="p", blocks=blocks, expected_lines=expected, finish_reason="eos",
    )

    assert score.lines_found == 0
    assert score.line_recall == 0.0
    assert score.omissions == list(expected)


def test_a_loose_presence_threshold_can_recover_a_corrupted_line() -> None:
    expected = ("Conclusion: uptime and yield improve together, not separately.",)
    blocks = [
        _block("Text", "Conclusion: uptime and yield improve together, not separate")
    ]

    score = score_transcription(
        page="p", blocks=blocks, expected_lines=expected, finish_reason="eos",
        presence_threshold=0.95,
    )

    assert score.lines_found == 1
    assert score.lines_exact == 0
    assert score.cer > 0.0


def test_repeated_expected_lines_consume_repeated_candidates() -> None:
    expected = ("same line", "same line", "same line")
    blocks = [_block("Text", "same line")]

    score = score_transcription(
        page="p", blocks=blocks, expected_lines=expected, finish_reason="eos"
    )

    assert score.lines_found == 1
    assert score.omissions == ["same line", "same line"]


def test_one_block_holding_many_lines_still_matches_each() -> None:
    expected = ("alpha", "beta", "gamma")
    blocks = [_block("Text", "alpha\nbeta\ngamma")]

    score = score_transcription(
        page="p", blocks=blocks, expected_lines=expected, finish_reason="eos"
    )

    assert score.lines_found == 3
    assert score.reading_order_violations == 0


def test_empty_expected_text_is_recall_one_by_convention() -> None:
    score = score_transcription(
        page="blank", blocks=[_block("Image", "")], expected_lines=(),
        finish_reason="eos",
    )

    assert score.lines_expected == 0
    assert score.line_recall == 1.0
    assert score.cer == 0.0
    assert score.skipped_non_text_blocks == 1


# ---------------------------------------------------------------------------
# reading order
# ---------------------------------------------------------------------------


def test_swapped_regions_are_reading_order_violations() -> None:
    expected = ("top line", "bottom line")
    blocks = [_block("Text", "bottom line"), _block("Text", "top line")]

    score = score_transcription(
        page="p", blocks=blocks, expected_lines=expected, finish_reason="eos"
    )

    assert score.lines_found == 2
    assert score.reading_order_violations >= 1


def test_same_block_lines_are_not_order_violations() -> None:
    expected = ("a", "b")
    blocks = [_block("Text", "b\na")]

    score = score_transcription(
        page="p", blocks=blocks, expected_lines=expected, finish_reason="eos"
    )

    # Both lines live in block 0, so the block index never goes backwards even
    # though the intra-block line order is reversed.
    assert score.reading_order_violations == 0


# ---------------------------------------------------------------------------
# truncation
# ---------------------------------------------------------------------------


def test_length_finish_is_truncation_and_not_eos() -> None:
    score = score_transcription(
        page="p", blocks=[_block("Text", "partial")], expected_lines=("partial",),
        finish_reason="length",
    )

    assert score.truncated
    assert score.as_dict()["truncated"] is True


def test_eos_finish_is_not_truncation() -> None:
    score = score_transcription(
        page="p", blocks=[_block("Text", "partial")], expected_lines=("partial",),
        finish_reason="eos",
    )

    assert not score.truncated


# ---------------------------------------------------------------------------
# table structure
# ---------------------------------------------------------------------------


_TABLE_EXPECTED = {
    "header": ("Item", "Spec", "Measured", "Result"),
    "rows": (
        ("Thickness", "12.0 nm", "12.3 nm", "Pass"),
        ("Roughness", "< 0.40 nm", "0.52 nm", "Fail"),
    ),
}


def _table_block(rows: list[list[str]], label: str = "Table") -> FullPageBlock:
    html = "<table>" + "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    ) + "</table>"
    return FullPageBlock(label=label, bbox=(0.0, 0.0, 1.0, 1.0), html=html,
                         text="", tables=[rows])


def test_exact_table_scores_perfectly() -> None:
    grid = [
        ["Item", "Spec", "Measured", "Result"],
        ["Thickness", "12.0 nm", "12.3 nm", "Pass"],
        ["Roughness", "< 0.40 nm", "0.52 nm", "Fail"],
    ]
    score = score_transcription(
        page="table", blocks=[_table_block(grid)], expected_lines=(),
        finish_reason="eos", expected_table=_TABLE_EXPECTED,
    )

    assert score.table_found
    assert score.table_shape_match
    assert score.table_header_match
    assert score.table_cell_accuracy == 1.0
    assert score.table_cells_matched == score.table_cells_expected


def test_wrong_cell_lowers_cell_accuracy() -> None:
    grid = [
        ["Item", "Spec", "Measured", "Result"],
        ["Thickness", "12.0 nm", "12.3 nm", "Pass"],
        ["Roughness", "< 0.40 nm", "0.52 nm", "Pass"],
    ]
    score = score_transcription(
        page="table", blocks=[_table_block(grid)], expected_lines=(),
        finish_reason="eos", expected_table=_TABLE_EXPECTED,
    )

    assert score.table_shape_match
    assert score.table_cell_accuracy < 1.0
    assert score.table_cell_accuracy > 0.8


def test_missing_table_is_reported() -> None:
    score = score_transcription(
        page="table", blocks=[_block("Text", "no table")], expected_lines=(),
        finish_reason="eos", expected_table=_TABLE_EXPECTED,
    )

    assert score.table_expected
    assert not score.table_found
    assert score.table_cell_accuracy == 0.0


def test_table_shape_mismatch_is_reported() -> None:
    grid = [
        ["Item", "Spec", "Measured", "Result"],
        ["Thickness", "12.0 nm", "12.3 nm", "Pass"],
    ]
    score = score_transcription(
        page="table", blocks=[_table_block(grid)], expected_lines=(),
        finish_reason="eos", expected_table=_TABLE_EXPECTED,
    )

    assert score.table_found
    assert not score.table_shape_match
    assert score.table_cell_accuracy == pytest.approx(8 / 12)


def test_a_spurious_small_table_does_not_hide_the_real_one() -> None:
    real = [
        ["Item", "Spec", "Measured", "Result"],
        ["Thickness", "12.0 nm", "12.3 nm", "Pass"],
        ["Roughness", "< 0.40 nm", "0.52 nm", "Fail"],
    ]
    spurious = [["x"]]
    score = score_transcription(
        page="table", blocks=[_table_block(spurious), _table_block(real)],
        expected_lines=(), finish_reason="eos", expected_table=_TABLE_EXPECTED,
    )

    assert score.table_cell_accuracy == 1.0


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------


def test_evaluate_names_every_failure() -> None:
    score = score_transcription(
        page="p",
        blocks=[_block("Text", "wrong")],
        expected_lines=("first line", "second line"),
        finish_reason="length",
    )

    passed, failures = evaluate(
        score, TranscriptionThresholds(min_line_exact_rate=0.5)
    )

    assert not passed
    assert any("line_recall" in f for f in failures)
    assert any("line_exact_rate" in f for f in failures)
    assert any("cer" in f for f in failures)
    assert any("truncated" in f for f in failures)


def test_evaluate_passes_a_clean_score() -> None:
    score = score_transcription(
        page="p", blocks=[_block("Text", "first line")],
        expected_lines=("first line",), finish_reason="eos",
    )

    passed, failures = evaluate(score, TranscriptionThresholds())

    assert passed, failures


def test_evaluate_can_tolerate_declared_truncation() -> None:
    score = score_transcription(
        page="p", blocks=[_block("Text", "first line")],
        expected_lines=("first line",), finish_reason="length",
    )

    passed, failures = evaluate(score, TranscriptionThresholds(require_eos=False))

    assert passed, failures


# ---------------------------------------------------------------------------
# end-to-end through the protocol parser
# ---------------------------------------------------------------------------


def test_scores_parsed_full_page_output() -> None:
    text = (
        '<div data-bbox="0 0 1 1" data-label="Section-Header">'
        "<h1>Inspection summary</h1></div>"
        '<div data-bbox="0 2 1 3" data-label="Table"><table>'
        "<tr><th>Item</th><th>Spec</th><th>Measured</th><th>Result</th></tr>"
        "<tr><td>Thickness</td><td>12.0 nm</td><td>12.3 nm</td><td>Pass</td></tr>"
        "<tr><td>Roughness</td><td>&lt; 0.40 nm</td><td>0.52 nm</td><td>Fail</td></tr>"
        "</table></div>"
    )
    blocks = parse_full_page_html(text)

    score = score_transcription(
        page="table", blocks=blocks,
        expected_lines=("Inspection summary",), finish_reason="eos",
        expected_table=_TABLE_EXPECTED,
    )

    assert score.line_recall == 1.0
    assert score.table_cell_accuracy == 1.0
    assert score.table_header_match
    assert score.reading_order_violations == 0
