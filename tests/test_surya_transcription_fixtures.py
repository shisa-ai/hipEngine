"""Fixture-integrity tests for the Surya transcription-acceptance pages.

The bench pages silently draw text past the canvas edge on four of seven pages,
which makes their intended text an invalid ground truth (the model is scored
against characters that are not in the image). The acceptance fixtures assert
fit at generation time; these tests keep that assertion live and keep the
committed PNGs reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
MODEL_ID = "datalab-to/surya-ocr-2"


def test_every_page_has_ground_truth() -> None:
    assert set(GROUND_TRUTH) == {
        "ja", "mixed", "dense", "table", "blank", "scan", "long", "a4"
    }


def test_acceptance_page_uses_fit_variant_only_where_needed() -> None:
    assert acceptance_page("ja") == "page_ja_fit.png"
    assert acceptance_page("mixed") == "page_mixed_fit.png"
    assert acceptance_page("scan") == "page_scan_fit.png"
    assert acceptance_page("long") == "page_long_fit.png"
    # These bench pages already fit their canvas.
    assert acceptance_page("dense") == "page_dense.png"
    assert acceptance_page("table") == "page_table.png"
    assert acceptance_page("blank") == "page_blank.png"
    # The A4 page is generated with the same fit assertion, so it needs no
    # variant; it is its own page-scale fixture.
    assert acceptance_page("a4") == "page_a4.png"


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


def test_a4_page_is_the_page_scale_grid_the_memory_plan_describes() -> None:
    """The A4 fixture must be the grid the page-scale evidence talks about.

    The 220x156 / 34320-patch / 8580-image-token numbers in the memory plan are
    only meaningful if a fixture actually produces that grid, so this pins the
    fixture to them rather than leaving the plan untested arithmetic.
    """

    from PIL import Image

    from hipengine.loading.surya import smart_resize_surya

    page = FIXTURES / "page_a4.png"
    assert page.exists(), "page_a4.png is not committed"
    with Image.open(page) as image:
        width, height = image.size
    # A4 at 300 DPI.
    assert (width, height) == (2480, 3508)
    resized_h, resized_w = smart_resize_surya(height, width)
    assert (resized_h // 16, resized_w // 16) == (220, 156)
    patches = (resized_h // 16) * (resized_w // 16)
    assert patches == 34320
    assert (resized_h // 32) * (resized_w // 32) == 8580


def test_a4_page_regenerates_byte_identically(tmp_path: Path) -> None:
    from scripts.surya_bench_pages import make_page_a4

    written = tmp_path / "page_a4.png"
    make_page_a4(written)
    assert written.read_bytes() == (FIXTURES / "page_a4.png").read_bytes(), (
        "page_a4.png is not reproducible; regenerate the fixture"
    )


def test_a4_ground_truth_is_paragraph_level_not_drawn_line_level() -> None:
    """The A4 body is wrapped prose, so its text unit is the paragraph.

    Scoring the drawn lines as separate units reads a correct paragraph-level
    transcription as 22 omissions and CER 0.89. This pins the unit, and pins
    that the unit is the drawn prose joined at the wrap points rather than a
    retyped copy that could drift from the page.
    """

    from scripts.surya_bench_pages import (
        _A4_ABSTRACT,
        _A4_META,
        _A4_SECTIONS,
        _A4_TABLE_CAPTION,
        _A4_TITLE,
    )

    units = expected_lines("a4")

    assert len(units) == 12, "title, meta, 4 headings, 4 paragraphs, caption"
    assert _A4_TITLE in units
    assert _A4_META in units
    assert _A4_TABLE_CAPTION in units
    # The wrapped lines are NOT units ...
    for line in _A4_ABSTRACT:
        assert line not in units
    # ... the paragraph they form is.
    assert " ".join(_A4_ABSTRACT) in units
    for heading_text, lines in _A4_SECTIONS:
        assert heading_text in units
        assert " ".join(lines) in units
        for line in lines:
            assert line not in units, "a wrapped line must not be its own unit"

    # Reading order: each paragraph follows its heading.
    for heading_text, lines in _A4_SECTIONS:
        assert units.index(" ".join(lines)) == units.index(heading_text) + 1


def test_a4_page_prompt_is_the_8580_image_token_prefill() -> None:
    """Close the loop from the A4 image to the prefill the page actually runs.

    The 8580-image-token number is used as evidence for the page-scale memory
    plan, so it must be what the model is fed and not only what the grid
    arithmetic predicts. `render_chat_prompt` is the same call the generator
    makes, and its output is the exact `input_ids` handed to the prefill.
    """

    from PIL import Image

    from hipengine.generation.surya_protocol import FULL_PAGE_HTML_PROMPT
    from hipengine.loading.surya import preprocess_image_surya, render_chat_prompt

    page = FIXTURES / "page_a4.png"
    assert page.exists(), "page_a4.png is not committed"

    # preprocess_image_surya takes a path or an array; the generator passes the
    # decoded page, so decode it here too.
    with Image.open(page) as image:
        array = np.asarray(image.convert("RGB"))
    _pixel_rows, grid = preprocess_image_surya(array)
    n_image_tokens = (grid[1] // 2) * (grid[2] // 2)
    assert grid == (1, 220, 156)
    assert n_image_tokens == 8580

    from hipengine.loading.surya import SuryaTokenizer, resolve_surya_path

    try:
        tokenizer = SuryaTokenizer(resolve_surya_path(MODEL_ID))
    except (FileNotFoundError, ImportError) as exc:
        pytest.skip(f"cannot build the Surya tokenizer locally: {exc}")

    input_ids, mm = render_chat_prompt(tokenizer, FULL_PAGE_HTML_PROMPT, n_image_tokens)

    # The image pads are the prefill's vision slots, and nothing else is one.
    assert sum(mm) == 8580
    assert len(input_ids) == 8701, "121 text tokens plus 8580 image tokens"
    assert len(input_ids) == len(mm)
    # They are contiguous, so the vision features land in one span.
    first = mm.index(1)
    assert mm[first:first + 8580] == [1] * 8580
    assert 1 not in mm[:first] and 1 not in mm[first + 8580:]
