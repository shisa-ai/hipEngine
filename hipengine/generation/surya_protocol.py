"""Surya OCR 2 prompt and output protocol.

Surya is a single checkpoint driven by different *prompts*, and each prompt is a
different task with a different output contract. The wording is the model's
training-time contract: paraphrasing it changes the task, so the strings here
are pinned rather than derived.

Three prompts are defined:

``FULL_PAGE_HTML_PROMPT``
    Full-page transcription. The model emits a flat sequence of top-level
    ``<div data-label="..." data-bbox="x0 y0 x1 y1">inner HTML</div>`` blocks,
    one per detected region, in reading order. This is the protocol a caller
    wants for "transcribe this page".

``LAYOUT_JSON_PROMPT``
    Layout detection only. The model emits a JSON array of
    ``{"label", "bbox", "count"}`` entries. It carries no transcription, so it
    cannot qualify OCR quality on its own.

``BLOCK_HTML_PROMPT``
    Per-block transcription of a cropped region, used by the block-mode
    pipeline after layout detection.

:func:`parse_full_page_html` turns full-page output into typed blocks, and
:func:`extract_text` / :func:`extract_tables` read the inner HTML. Everything
here is torch-free and stdlib-only so it can run on the hot path and in tests
without a third-party HTML library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

# ---------------------------------------------------------------------------
# prompt contract
# ---------------------------------------------------------------------------

FULL_PAGE_HTML_PROMPT = (
    "OCR this image to HTML. Each block is a div with data-label and "
    "data-bbox (x0 y0 x1 y1, normalized 0-1000)."
)

LAYOUT_JSON_PROMPT = (
    "Output the layout of this image as JSON. Each entry is a dict with "
    '"label", "bbox", and "count" fields. Bbox is x0 y0 x1 y1, normalized 0-1000.'
)

BLOCK_HTML_PROMPT = "OCR this block image to HTML."

# HTML void elements never receive an end tag, so the block extractor must not
# count them as open when matching a block's closing </div>.
_VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
     "meta", "param", "source", "track", "wbr"}
)

# Elements whose boundaries become line breaks in extracted text. A `<p>` per
# line and a `<td>` per cell are both how the model separates regions, so both
# have to produce a separator for line-oriented scoring to line up.
_TEXT_BREAK_TAGS = frozenset(
    {"p", "div", "br", "hr", "tr", "td", "th", "li", "ul", "ol", "table",
     "thead", "tbody", "caption", "pre", "section", "article", "blockquote",
     "h1", "h2", "h3", "h4", "h5", "h6"}
)

_WS_RE = re.compile(r"[ \t\r\f\v]+")
_FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z0-9_-]*[ \t]*\n?")
_FENCE_CLOSE_RE = re.compile(r"\n?```[ \t]*$")


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _FENCE_OPEN_RE.sub("", cleaned)
        cleaned = _FENCE_CLOSE_RE.sub("", cleaned)
    return cleaned.strip()


def _parse_bbox(value: str | None) -> tuple[float, float, float, float] | None:
    """Parse a ``"x0 y0 x1 y1"`` string. ``None`` when it is not four numbers."""

    if not value:
        return None
    parts = value.replace(",", " ").split()
    if len(parts) != 4:
        return None
    try:
        numbers = tuple(float(part) for part in parts)
    except ValueError:
        return None
    return numbers  # type: ignore[return-value]


@dataclass
class FullPageBlock:
    """One top-level region of a full-page transcription.

    ``bbox`` is ``None`` when the model emitted a malformed box. The block is
    kept anyway: its transcription is still useful, and silently dropping it
    would turn a coordinate bug into a false omission.
    """

    label: str
    bbox: tuple[float, float, float, float] | None
    html: str
    text: str = ""
    tables: list[list[list[str]]] = field(default_factory=list)
    reading_order: int = 0


# ---------------------------------------------------------------------------
# full-page block extraction
# ---------------------------------------------------------------------------


def _serialize_start(tag: str, attrs: list[tuple[str, str | None]]) -> str:
    """Rebuild an inner tag, dropping the model's nested debug attributes."""

    kept = [
        (name, value)
        for name, value in attrs
        if name not in ("data-bbox", "data-label")
    ]
    if not kept:
        return f"<{tag}>"
    rendered = "".join(
        f' {name}="{value}"' if value is not None else f" {name}"
        for name, value in kept
    )
    return f"<{tag}{rendered}>"


class _BlockExtractor(HTMLParser):
    """Collect top-level ``<div data-label data-bbox>`` blocks and their inner HTML.

    Only divs that are not already inside a block qualify, so the nested
    ``data-bbox`` debug divs the model sometimes emits stay inside their parent.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[dict] = []
        self._cur: dict | None = None
        self._open = 0

    def _begin(self, label: str, bbox: tuple | None) -> None:
        self._cur = {"label": label, "bbox": bbox, "parts": []}
        self._open = 1

    def handle_starttag(self, tag: str, attrs) -> None:
        if self._cur is None:
            if tag == "div":
                values = dict(attrs)
                if "data-label" in values and "data-bbox" in values:
                    self._begin(
                        str(values["data-label"]),
                        _parse_bbox(values["data-bbox"]),
                    )
            return
        if tag in _VOID_TAGS:
            self._cur["parts"].append(_serialize_start(tag, attrs))
            return
        self._open += 1
        self._cur["parts"].append(_serialize_start(tag, attrs))

    def handle_startendtag(self, tag: str, attrs) -> None:
        if self._cur is not None:
            self._cur["parts"].append(_serialize_start(tag, attrs))

    def handle_endtag(self, tag: str) -> None:
        if self._cur is None or tag in _VOID_TAGS:
            return
        self._open -= 1
        if self._open <= 0:
            self.blocks.append(self._cur)
            self._cur = None
            self._open = 0
            return
        self._cur["parts"].append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if self._cur is not None:
            self._cur["parts"].append(data)

    def close(self) -> None:  # noqa: A003 - HTMLParser's name
        super().close()
        if self._cur is not None:
            self.blocks.append(self._cur)
            self._cur = None
            self._open = 0


class _TextExtractor(HTMLParser):
    """Flatten inner HTML to block-separated text lines."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _TEXT_BREAK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs) -> None:
        if tag in _TEXT_BREAK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _TEXT_BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def extract_text(html: str) -> str:
    """Flatten HTML to newline-separated lines with entities decoded.

    Line breaks land on block boundaries, and inline whitespace collapses to a
    single space, so a line extracted here is directly comparable to a line of
    ground-truth text.
    """

    if not html:
        return ""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    lines: list[str] = []
    for raw_line in "".join(parser.parts).split("\n"):
        collapsed = _WS_RE.sub(" ", raw_line).strip()
        if collapsed:
            lines.append(collapsed)
    return "\n".join(lines)


class _TableExtractor(HTMLParser):
    """Read tables as ``tables -> rows -> cells``."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table_depth = 0
        self._rows: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._rows = []
            return
        if self._table_depth != 1:
            return
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_startendtag(self, tag: str, attrs) -> None:
        # No void element carries table content, but a self-closed <td/> should
        # still produce an empty cell rather than being ignored.
        if self._table_depth == 1 and tag in ("td", "th") and self._cell is None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            if self._table_depth == 1 and self._rows is not None:
                self.tables.append(self._rows)
                self._rows = None
            self._table_depth = max(0, self._table_depth - 1)
            return
        if self._table_depth != 1:
            return
        if tag in ("td", "th"):
            if self._cell is not None and self._row is not None:
                self._row.append(_WS_RE.sub(" ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr":
            if self._row is not None and self._rows is not None:
                self._rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def extract_tables(html: str) -> list[list[list[str]]]:
    """Return ``[table][row][cell]`` text for every table in ``html``."""

    if not html:
        return []
    parser = _TableExtractor()
    parser.feed(html)
    parser.close()
    if parser._table_depth == 1 and parser._rows is not None:
        # Unterminated trailing table: keep the rows rather than lose them.
        parser.tables.append(parser._rows)
    return parser.tables


def parse_full_page_html(text: str) -> list[FullPageBlock]:
    """Parse ``FULL_PAGE_HTML_PROMPT`` output into typed blocks.

    Tolerates code fences, stray text outside blocks, malformed bboxes, and
    unterminated final tags. Blocks keep document order, which is the model's
    reading order.
    """

    cleaned = _strip_fences(text)
    if not cleaned:
        return []
    parser = _BlockExtractor()
    parser.feed(cleaned)
    parser.close()
    blocks: list[FullPageBlock] = []
    for index, raw in enumerate(parser.blocks):
        inner = "".join(raw["parts"]).strip()
        blocks.append(
            FullPageBlock(
                label=raw["label"],
                bbox=raw["bbox"],
                html=inner,
                text=extract_text(inner),
                tables=extract_tables(inner),
                reading_order=index,
            )
        )
    return blocks
