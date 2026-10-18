"""Fixture-integrity tests for the Surya benchmark pages.

Four of the seven text pages used to draw past the canvas edge, which made
their intended text an invalid ground truth (the model is scored against
characters that are not in the image). The generators now assert fit at
drawing time and the pages were re-cut onto canvases that hold them; these
tests keep that assertion live and keep the committed PNGs reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")

from scripts.surya_bench_pages import (  # noqa: E402
    BENCH_LONG_HEIGHT,
    BENCH_PAGE_SIZE,
    GROUND_TRUTH,
    PAGES,
    expected_lines,
    page_filename,
)

FIXTURES = Path("tests/fixtures/surya")
MODEL_ID = "datalab-to/surya-ocr-2"


def test_every_page_has_ground_truth() -> None:
    assert set(GROUND_TRUTH) == {
        "ja", "mixed", "dense", "table", "blank", "scan", "long", "a4"
    }


def test_page_filename_is_the_bench_page_for_every_name() -> None:
    for name in GROUND_TRUTH:
        assert page_filename(name) == f"page_{name}.png"


def test_page_filename_rejects_an_unknown_name() -> None:
    with pytest.raises(KeyError):
        page_filename("nope")


def test_pages_regenerate_byte_identically(tmp_path: Path) -> None:
    """Pin decoded pixels, independent of PNG encoder/compression versions."""

    from PIL import Image

    for name, generator in PAGES.items():
        if name == "a4":
            continue
        written = tmp_path / page_filename(name)
        generator(written)
        committed = FIXTURES / page_filename(name)
        assert committed.exists(), f"{committed} is not committed"
        with Image.open(written) as actual, Image.open(committed) as expected:
            assert actual.mode == expected.mode and actual.size == expected.size
            assert actual.tobytes() == expected.tobytes(), (
                f"{page_filename(name)} pixels differ; review fixture rendering"
            )


def test_fit_assertion_rejects_the_old_clipping_geometry(tmp_path: Path) -> None:
    """The check must actually fire, or the pages can silently clip again.

    512x512 is the canvas the Japanese and mixed pages used to be drawn on,
    where their lines are 576-663 px and 503/506 px wide against 480 px of
    usable width; 1024 px tall is where the long page's blocks 5 and 6 fell off
    the bottom.
    """

    from scripts.surya_bench_pages import (
        make_page_ja,
        make_page_long,
        make_page_mixed,
        make_page_scan,
    )

    with pytest.raises(ValueError, match="runs off the page"):
        make_page_ja(tmp_path / "ja512.png", size=512)
    with pytest.raises(ValueError, match="runs off the page"):
        make_page_mixed(tmp_path / "mixed512.png", size=512)
    with pytest.raises(ValueError, match="runs off the page"):
        make_page_scan(tmp_path / "scan512.png", size=512)
    with pytest.raises(ValueError, match="runs off the page"):
        make_page_long(tmp_path / "long1024.png", height=1024)


def test_text_pages_fit_the_canvas_they_are_drawn_on() -> None:
    """Every text page is on a canvas the fit assertion accepts.

    Pins the canvas sizes: the four re-cut pages need the larger ones, and a
    silent shrink back to the old geometry would fail the generators above.
    """

    from PIL import Image

    expected = {
        "ja": (BENCH_PAGE_SIZE, BENCH_PAGE_SIZE),
        "mixed": (BENCH_PAGE_SIZE, BENCH_PAGE_SIZE),
        "scan": (BENCH_PAGE_SIZE, BENCH_PAGE_SIZE),
        "long": (BENCH_PAGE_SIZE, BENCH_LONG_HEIGHT),
        "dense": (512, 512),
        "table": (512, 512),
        "blank": (512, 512),
    }
    for name, size in expected.items():
        with Image.open(FIXTURES / page_filename(name)) as image:
            assert image.size == size, f"{name} is {image.size}, expected {size}"


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
    from PIL import Image
    from scripts.surya_bench_pages import make_page_a4

    written = tmp_path / "page_a4.png"
    make_page_a4(written)
    with Image.open(written) as actual, Image.open(FIXTURES / "page_a4.png") as expected:
        assert actual.mode == expected.mode and actual.size == expected.size
        assert actual.tobytes() == expected.tobytes(), "page_a4.png pixels differ"


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
