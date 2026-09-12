"""Fixture-integrity tests for the Surya transcription-acceptance pages.

The bench pages silently draw text past the canvas edge on four of seven pages,
which makes their intended text an invalid ground truth (the model is scored
against characters that are not in the image). The acceptance fixtures assert
fit at generation time; these tests keep that assertion live and keep the
committed PNGs reproducible.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")

from scripts.surya_bench_pages import (  # noqa: E402
    FIT_PAGES,
    GROUND_TRUTH,
    acceptance_page,
    expected_lines,
    write_fit_pages,
)

FIXTURES = Path("tests/fixtures/surya")


def test_every_page_has_ground_truth() -> None:
    assert set(GROUND_TRUTH) == {"ja", "mixed", "dense", "table", "blank", "scan", "long"}


def test_acceptance_page_uses_fit_variant_only_where_needed() -> None:
    assert acceptance_page("ja") == "page_ja_fit.png"
    assert acceptance_page("mixed") == "page_mixed_fit.png"
    assert acceptance_page("scan") == "page_scan_fit.png"
    assert acceptance_page("long") == "page_long_fit.png"
    # These bench pages already fit their canvas.
    assert acceptance_page("dense") == "page_dense.png"
    assert acceptance_page("table") == "page_table.png"
    assert acceptance_page("blank") == "page_blank.png"


def test_acceptance_page_rejects_an_unknown_name() -> None:
    with pytest.raises(KeyError):
        acceptance_page("nope")


def test_fit_pages_regenerate_byte_identically(tmp_path: Path) -> None:
    written = write_fit_pages(tmp_path)

    assert len(written) == len(FIT_PAGES)
    for path in written:
        committed = FIXTURES / path.name
        assert committed.exists(), f"{committed} is not committed"
        assert path.read_bytes() == committed.read_bytes(), (
            f"{path.name} is not reproducible; regenerate the fixtures"
        )


def test_fit_assertion_rejects_the_clipping_bench_geometry(tmp_path: Path) -> None:
    """The check must actually fire, or the fixtures can silently clip again."""

    from scripts.surya_bench_pages import make_page_ja_fit, make_page_long_fit

    # 512x512 is the bench canvas, where the Japanese body lines are 576-663 px
    # wide against 480 px of usable width.
    with pytest.raises(ValueError, match="runs off the page"):
        make_page_ja_fit(tmp_path / "ja512.png", size=512)

    # 1024 px tall is the bench canvas, where blocks 5 and 6 fall off the bottom.
    with pytest.raises(ValueError, match="runs off the page"):
        make_page_long_fit(tmp_path / "long1024.png", height=1024)


@pytest.mark.parametrize("page", ["ja", "mixed", "dense", "scan", "long"])
def test_text_pages_have_expected_lines(page: str) -> None:
    assert len(expected_lines(page)) >= 6


def test_table_page_ground_truth_matches_the_drawn_grid() -> None:
    table = GROUND_TRUTH["table"]["table"]

    assert table["header"] == ("Item", "Spec", "Measured", "Result")
    assert len(table["rows"]) == 7
    assert all(len(row) == 4 for row in table["rows"])


def test_blank_page_has_no_expected_lines() -> None:
    assert expected_lines("blank") == ()
