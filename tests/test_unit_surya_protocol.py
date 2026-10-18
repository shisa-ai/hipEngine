"""Unit tests for the torch-free Surya full-page HTML protocol.

``hipengine.generation.surya_protocol`` holds the model's training-time prompt
strings and the parser for the full-page HTML output. Everything here is pure:
no model checkpoint, GPU, torch, or third-party HTML library.

The parser is the contract the transcription acceptance test scores against, so
these tests pin the properties that test depends on: top-level block discovery,
attribute stripping, block-separated text extraction, entity handling, and
table structure.
"""

from __future__ import annotations

import pytest

from hipengine.generation.surya_protocol import (
    BLOCK_HTML_PROMPT,
    FULL_PAGE_HTML_PROMPT,
    LAYOUT_JSON_PROMPT,
    FullPageBlock,
    extract_tables,
    extract_text,
    parse_full_page_html,
)


# ---------------------------------------------------------------------------
# prompt contract
# ---------------------------------------------------------------------------


def test_prompt_strings_are_the_training_time_contract() -> None:
    """The wording is the checkpoint's contract; paraphrasing changes the task."""

    assert FULL_PAGE_HTML_PROMPT == (
        "OCR this image to HTML. Each block is a div with data-label and "
        "data-bbox (x0 y0 x1 y1, normalized 0-1000)."
    )
    assert LAYOUT_JSON_PROMPT.startswith("Output the layout of this image as JSON.")
    assert BLOCK_HTML_PROMPT == "OCR this block image to HTML."


# ---------------------------------------------------------------------------
# top-level block discovery
# ---------------------------------------------------------------------------


def test_parses_one_block_per_top_level_div() -> None:
    text = (
        '<div data-bbox="61 53 464 96" data-label="Section-Header">'
        "<h1>Inspection summary</h1></div>"
    )

    blocks = parse_full_page_html(text)

    assert len(blocks) == 1
    block = blocks[0]
    assert isinstance(block, FullPageBlock)
    assert block.label == "Section-Header"
    assert block.bbox == (61.0, 53.0, 464.0, 96.0)
    assert block.html == "<h1>Inspection summary</h1>"
    assert block.text == "Inspection summary"


def test_nested_divs_do_not_become_top_level_blocks() -> None:
    text = (
        '<div data-bbox="0 0 10 10" data-label="Text">'
        '<div data-bbox="1 1 2 2" data-label="Text">inner</div>'
        "</div>"
    )

    blocks = parse_full_page_html(text)

    assert len(blocks) == 1
    # The nested debug attributes are stripped from the inner HTML.
    assert "data-bbox" not in blocks[0].html
    assert "data-label" not in blocks[0].html
    assert blocks[0].text == "inner"


def test_blocks_keep_document_order_and_reading_order_index() -> None:
    text = (
        '<div data-bbox="0 0 1 1" data-label="Page-Header">top</div>'
        '<div data-bbox="0 2 1 3" data-label="Text">middle</div>'
        '<div data-bbox="0 4 1 5" data-label="Page-Footer">bottom</div>'
    )

    blocks = parse_full_page_html(text)

    assert [b.label for b in blocks] == ["Page-Header", "Text", "Page-Footer"]
    assert [b.reading_order for b in blocks] == [0, 1, 2]


def test_blank_and_fenced_and_empty_inputs() -> None:
    assert parse_full_page_html("") == []
    assert parse_full_page_html("   \n ") == []
    assert parse_full_page_html("```html\n```") == []


def test_strips_code_fences_around_output() -> None:
    text = (
        "```html\n"
        '<div data-bbox="0 0 1 1" data-label="Text">hi</div>\n'
        "```"
    )

    blocks = parse_full_page_html(text)

    assert len(blocks) == 1
    assert blocks[0].text == "hi"


def test_skips_divs_missing_label_or_bbox() -> None:
    text = (
        '<div data-bbox="0 0 1 1">no label</div>'
        '<div data-label="Text">no bbox</div>'
        '<div data-bbox="0 0 1 1" data-label="Text">kept</div>'
    )

    blocks = parse_full_page_html(text)

    assert [b.text for b in blocks] == ["kept"]


def test_rejects_malformed_bbox_without_dropping_the_block() -> None:
    text = (
        '<div data-bbox="0 0 1" data-label="Text">short</div>'
        '<div data-bbox="a b c d" data-label="Text">nonnumeric</div>'
        '<div data-bbox="0 0 1 1" data-label="Text">ok</div>'
    )

    blocks = parse_full_page_html(text)

    assert [b.text for b in blocks] == ["short", "nonnumeric", "ok"]
    assert blocks[0].bbox is None
    assert blocks[1].bbox is None
    assert blocks[2].bbox == (0.0, 0.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# text extraction
# ---------------------------------------------------------------------------


def test_extract_text_separates_block_elements_with_newlines() -> None:
    html = "<p>first line</p><p>second line</p>"

    assert extract_text(html) == "first line\nsecond line"


def test_extract_text_unescapes_entities() -> None:
    html = "<p>&lt; 2.0 %</p><p>&gt; 4.0 MPa</p>"

    assert extract_text(html) == "< 2.0 %\n> 4.0 MPa"


def test_extract_text_collapses_inline_whitespace_only() -> None:
    html = "<p>  Item   Spec  </p>"

    assert extract_text(html) == "Item Spec"


def test_extract_text_handles_br_and_void_tags() -> None:
    html = "<p>a<br>b</p><hr/><img src='x'/><p>c</p>"

    assert extract_text(html) == "a\nb\nc"


# ---------------------------------------------------------------------------
# table structure
# ---------------------------------------------------------------------------


def test_extract_tables_reads_header_and_body_cells() -> None:
    html = (
        "<table border='1'><thead>"
        "<tr><th>Item</th><th>Spec</th></tr></thead>"
        "<tbody><tr><td>Thickness</td><td>12.0 nm</td></tr>"
        "<tr><td>Uniformity</td><td>1.6 %</td></tr></tbody></table>"
    )

    tables = extract_tables(html)

    assert tables == [
        [
            ["Item", "Spec"],
            ["Thickness", "12.0 nm"],
            ["Uniformity", "1.6 %"],
        ]
    ]


def test_extract_tables_returns_empty_list_without_a_table() -> None:
    assert extract_tables("<p>no table here</p>") == []


def test_block_exposes_its_tables() -> None:
    text = (
        '<div data-bbox="0 0 1 1" data-label="Table">'
        "<table><tr><td>a</td><td>b</td></tr></table></div>"
    )

    block = parse_full_page_html(text)[0]

    assert block.tables == [[["a", "b"]]]
    assert block.text == "a\nb"


@pytest.mark.parametrize("label", ["Text", "Section-Header", "Table", "Caption"])
def test_common_labels_round_trip(label: str) -> None:
    text = f'<div data-bbox="0 0 1 1" data-label="{label}">x</div>'

    assert parse_full_page_html(text)[0].label == label
