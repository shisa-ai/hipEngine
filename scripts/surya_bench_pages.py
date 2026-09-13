#!/usr/bin/env python3
"""Deterministic page fixtures for the Surya OCR benchmark suite.

The original fixtures are bar patterns and simple prose pages. Those are enough
to gate numerical parity but not to judge transcription quality or to tune
against, so this module adds the document types that actually move the answer:
Japanese and mixed-script pages, dense small text, ruled tables, a blank page,
a noisy/rotated scan, and a block-heavy page whose output runs long.

Every text page is drawn through :func:`_draw_text_fit`, which raises rather
than let a line run off the canvas, so the ground truth in :data:`GROUND_TRUTH`
is exactly the text in the image. Four pages were previously drawn past the
edge; the canvas sizes below are the ones that fit them.

Everything is seeded and geometry-aligned to a multiple of 32 so the smart
resize is a no-op and the vision grid is clean, matching the existing fixtures.
Regenerating produces byte-identical PNGs.

Usage:
    python3 scripts/surya_bench_pages.py --out-dir tests/fixtures/surya
    python3 scripts/surya_bench_pages.py --only ja,scan
"""

from __future__ import annotations

import argparse
from pathlib import Path

CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-DemiLight.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)
LATIN_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
)

# Seeded noise for the scan fixture: fixed so regenerating is byte-identical.
_SCAN_SEED = 20260911

_JA_BODY = (
    "本報告書は、半導体製造装置の歩留まり改善に関する中間結果をまとめたものである。",
    "第一に、成膜工程における膜厚のばらつきが主要な不良要因であることを確認した。",
    "第二に、検査装置の校正間隔を短縮することで、検出遅れが四割減少した。",
    "第三に、歩留まりの改善効果は、装置の稼働率にも有意に寄与することが分かった。",
    "今後は、異常検知モデルの精度検証を継続し、四半期ごとに結果を報告する。",
)
_JA_HEADING = "製造工程 歩留まり改善 中間報告"
_LONG_TITLE = "Annual process qualification report"
_MIXED_HEADING = "Mixed script page / 混在文書"
_SCAN_HEADING = "Scanned inspection record"

_MIXED_LINES = (
    "Summary: the deposition step dominates yield loss in this line.",
    "第一に、膜厚のばらつきが主要因であることを確認した。",
    "Measurement: thickness sigma fell from 3.1 nm to 2.4 nm.",
    "第二に、検査間隔の短縮により検出遅れが減少した。",
    "Conclusion: uptime and yield improve together, not separately.",
)

_DENSE_LINES = (
    "1.1 Scope and definitions of the measured quantity under test",
    "1.2 Instrument calibration interval and traceability record",
    "1.3 Sampling plan, stratification, and rejection criteria",
    "1.4 Environmental controls: temperature, humidity, vibration",
    "2.1 Data acquisition rate, filtering, and smoothing window",
    "2.2 Outlier policy and the handling of missing observations",
    "2.3 Uncertainty budget with correlated contribution terms",
    "3.1 Regression model, residuals, and goodness of fit",
    "3.2 Confidence intervals and the treatment of bias",
    "3.3 Sensitivity analysis over the operating envelope",
    "4.1 Acceptance thresholds and the decision rule applied",
    "4.2 Nonconformance reporting, escalation, and disposition",
    "4.3 Retest policy and the retention period for raw data",
    "5.1 Summary of findings against the original objectives",
    "5.2 Residual risk, open items, and recommended next steps",
)

_TABLE_HEADER = ("Item", "Spec", "Measured", "Result")
_TABLE_ROWS = (
    ("Thickness", "12.0 nm", "12.3 nm", "Pass"),
    ("Uniformity", "< 2.0 %", "1.6 %", "Pass"),
    ("Roughness", "< 0.40 nm", "0.52 nm", "Fail"),
    ("Resistivity", "18-22 uOhm", "20.1 uOhm", "Pass"),
    ("Defect density", "< 0.10 /cm2", "0.07 /cm2", "Pass"),
    ("Adhesion", "> 4.0 MPa", "4.4 MPa", "Pass"),
    ("Refractive index", "1.46 +/- 0.01", "1.47", "Marginal"),
)

# Canvas sizes. The text pages are drawn large enough that every line fits with
# the margin the fit assertion enforces; ``long`` needs a taller page because it
# is deliberately block-heavy.
BENCH_PAGE_SIZE = 1024
BENCH_LONG_HEIGHT = 1600


def _font(paths: tuple[str, ...], size: int):
    from PIL import ImageFont

    for candidate in paths:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    raise RuntimeError(f"no usable font found among {paths}")


def _draw_text_fit(
    draw, xy, text: str, font, *, max_x: int, max_y: int, fill
) -> None:
    """Draw one line, refusing to let it run off the canvas.

    The text pages that carry their own ground truth go through this, so the
    text the suite claims to measure is provably the text in the image. Four of
    them previously drew past the edge (``ja``/``mixed``/``scan`` horizontally,
    ``long`` vertically); a transcription test scored against the *intended*
    text then reads correct model output as an error, which is how an earlier
    ``line.`` -> ``literature`` "hallucination" arose from a clipped word.
    The ruled table is the exception: its cell text is bounded by the grid, not
    the canvas, so it draws directly.
    """

    x, y = xy
    width = draw.textlength(text, font=font)
    if x + width > max_x:
        raise ValueError(
            f"line runs off the page (x={x}, width={width:.1f}, max_x={max_x}): "
            f"{text!r}"
        )
    if y + font.size > max_y:
        raise ValueError(
            f"line runs off the page (y={y}, size={font.size}, max_y={max_y}): "
            f"{text!r}"
        )
    draw.text((x, y), text, fill=fill, font=font)


def make_page_ja(path: Path, size: int = BENCH_PAGE_SIZE) -> None:
    """Japanese-only dense page: heading plus five body paragraphs."""

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    heading = _font(CJK_FONT_CANDIDATES, 26)
    body = _font(CJK_FONT_CANDIDATES, 17)

    _draw_text_fit(draw, (32, 30), _JA_HEADING, heading, max_x=size - 32,
                   max_y=size, fill=(10, 10, 10))
    y = 84
    for line in _JA_BODY:
        _draw_text_fit(draw, (32, y), line, body, max_x=size - 32, max_y=size,
                       fill=(30, 30, 30))
        y += 52
    draw.rectangle([32, y + 8, size - 32, y + 11], fill=(150, 150, 150))
    img.save(path)


def make_page_mixed(path: Path, size: int = BENCH_PAGE_SIZE) -> None:
    """Japanese and English interleaved, so both scripts share one page."""

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    heading = _font(CJK_FONT_CANDIDATES, 22)
    cjk = _font(CJK_FONT_CANDIDATES, 16)
    latin = _font(LATIN_FONT_CANDIDATES, 16)

    _draw_text_fit(draw, (32, 28), _MIXED_HEADING, heading, max_x=size - 32,
                   max_y=size, fill=(10, 10, 10))
    y = 80
    for index, line in enumerate(_MIXED_LINES):
        font = latin if index % 2 == 0 else cjk
        _draw_text_fit(draw, (32, y), line, font, max_x=size - 32, max_y=size,
                       fill=(30, 30, 30))
        y += 46
    img.save(path)


def make_page_dense(path: Path, size: int = 512) -> None:
    """Dense small text: fifteen tightly spaced lines, no headings."""

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    small = _font(LATIN_FONT_CANDIDATES, 12)
    y = 20
    for line in _DENSE_LINES:
        draw.text((20, y), line, fill=(35, 35, 35), font=small)
        y += 32
    img.save(path)


def make_page_table(path: Path, size: int = 512) -> None:
    """A ruled table with a header row and seven data rows."""

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cell = _font(LATIN_FONT_CANDIDATES, 15)
    heading = _font(LATIN_FONT_CANDIDATES, 20)

    draw.text((32, 26), "Inspection summary", fill=(10, 10, 10), font=heading)
    left, top, right = 32, 76, size - 32
    row_h = 44
    columns = [left, left + 150, left + 268, left + 396, right]

    for index in range(len(_TABLE_ROWS) + 2):
        y = top + index * row_h
        draw.line([left, y, right, y], fill=(90, 90, 90), width=2)
    for x in columns:
        draw.line([x, top, x, top + row_h * (len(_TABLE_ROWS) + 1)],
                  fill=(90, 90, 90), width=2)

    for column, text in zip(range(4), _TABLE_HEADER):
        draw.text((columns[column] + 8, top + 12), text, fill=(10, 10, 10), font=cell)
    for row, values in enumerate(_TABLE_ROWS):
        for column, text in enumerate(values):
            draw.text((columns[column] + 8, top + row_h * (row + 1) + 12),
                      text, fill=(35, 35, 35), font=cell)
    img.save(path)


_A4_TITLE = "Process qualification report"
_A4_META = "Document PQ-2026-014  |  Revision C  |  Issued 2026-09-12"
_A4_ABSTRACT = (
    "This report records the qualification of the deposition and inspection",
    "line against the acceptance thresholds agreed at the start of the",
    "campaign. Every measured quantity is traceable to a calibrated",
    "instrument and to the raw data retained with this document.",
)
_A4_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "1. Scope and method",
        (
            "The qualification covers film thickness, uniformity, roughness and",
            "resistivity across the full operating envelope of the tool. Sampling",
            "followed a stratified plan: three wafers per lot, five sites per",
            "wafer, and one repeat measurement per site to bound repeatability.",
            "Instruments were calibrated within seven days of every measurement.",
        ),
    ),
    (
        "2. Results",
        (
            "Thickness and uniformity met their thresholds with margin. Roughness",
            "was the single nonconformance: the measured 0.52 nm exceeds the",
            "0.40 nm limit, and the excursion is correlated with the chamber that",
            "was serviced during the campaign. Resistivity and defect density both",
            "remained inside their windows for every lot measured.",
        ),
    ),
    (
        "3. Conclusion",
        (
            "The line is qualified for production with one open nonconformance",
            "and a narrowed process window on the serviced chamber. Yield and",
            "uptime improved together over the campaign, which is consistent with",
            "the reduced detection delay reported in the previous interval.",
        ),
    ),
    (
        "4. Open items",
        (
            "Repeat the roughness measurement on the serviced chamber after its",
            "next preventive maintenance. Extend the sampling plan to the second",
            "deposition module, and re-derive the uncertainty budget once the",
            "additional data is available.",
        ),
    ),
)
_A4_TABLE_CAPTION = "Table 1. Process measurement summary"


# A4 at 300 DPI. smart_resize rounds 2480x3508 to 2496x3520, i.e. exactly the
# 220x156 patch grid and 8580 merged image tokens the page-scale memory plan is
# written against, so this fixture is the grid that plan describes.
A4_300DPI_SIZE = (2480, 3508)


def make_page_a4(path: Path) -> None:
    """A real 300-DPI A4 page: the page-scale transcription case.

    Every line is drawn through :func:`_draw_text_fit`, so the ground truth is
    exactly the text on the page. The table is ruled, so the table scorer has a
    real grid to read.
    """

    from PIL import Image, ImageDraw

    width, height = A4_300DPI_SIZE
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    title = _font(LATIN_FONT_CANDIDATES, 96)
    meta = _font(LATIN_FONT_CANDIDATES, 44)
    heading = _font(LATIN_FONT_CANDIDATES, 52)
    body = _font(LATIN_FONT_CANDIDATES, 44)
    cell = _font(LATIN_FONT_CANDIDATES, 40)

    left, right = 300, width - 300
    _draw_text_fit(draw, (left, 300), _A4_TITLE, title,
                   max_x=right, max_y=height, fill=(10, 10, 10))
    _draw_text_fit(draw, (left, 430), _A4_META, meta,
                   max_x=right, max_y=height, fill=(60, 60, 60))
    draw.rectangle([left, 500, right, 504], fill=(70, 70, 70))

    y = 560
    for line in _A4_ABSTRACT:
        _draw_text_fit(draw, (left, y), line, body,
                       max_x=right, max_y=height, fill=(35, 35, 35))
        y += 60

    for heading_text, lines in _A4_SECTIONS:
        y += 60
        _draw_text_fit(draw, (left, y), heading_text, heading,
                       max_x=right, max_y=height, fill=(10, 10, 10))
        y += 70
        for line in lines:
            _draw_text_fit(draw, (left, y), line, body,
                           max_x=right, max_y=height, fill=(35, 35, 35))
            y += 60
        if heading_text.startswith("2."):
            y += 40
            _draw_text_fit(draw, (left, y), _A4_TABLE_CAPTION, meta,
                           max_x=right, max_y=height, fill=(45, 45, 45))
            y += 60
            row_h = 58
            columns = [left, left + 620, left + 1020, left + 1400, right]
            rows = len(_TABLE_ROWS) + 1
            for index in range(rows + 1):
                line_y = y + index * row_h
                draw.line([left, line_y, right, line_y], fill=(90, 90, 90), width=2)
            for column_x in columns:
                draw.line([column_x, y, column_x, y + row_h * rows],
                          fill=(90, 90, 90), width=2)
            for column, text in enumerate(_TABLE_HEADER):
                _draw_text_fit(draw, (columns[column] + 12, y + 10), text, cell,
                               max_x=right, max_y=height, fill=(10, 10, 10))
            for row, values in enumerate(_TABLE_ROWS):
                for column, text in enumerate(values):
                    _draw_text_fit(
                        draw, (columns[column] + 12, y + row_h * (row + 1) + 10),
                        text, cell, max_x=right, max_y=height, fill=(35, 35, 35),
                    )
            y += row_h * rows

    if y > height - 300:
        raise ValueError(f"A4 page overflows its bottom margin (y={y})")
    img.save(path)


def make_page_blank(path: Path, size: int = 512) -> None:
    """A blank page with a faint border: the empty-output edge case."""

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, size - 9, size - 9], outline=(225, 225, 225), width=2)
    img.save(path)


def make_page_scan(path: Path, size: int = BENCH_PAGE_SIZE) -> None:
    """A degraded scan: real text, slight rotation, speckle, and blur.

    Deterministic: the noise field comes from a fixed-seed generator, so the
    fixture is reproducible byte for byte. The right margin is deliberately
    wide: the rotation moves every glyph, so a line that only just fits before
    rotation would still be cut afterwards.
    """

    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    img = Image.new("RGB", (size, size), (252, 250, 245))
    draw = ImageDraw.Draw(img)
    body = _font(LATIN_FONT_CANDIDATES, 17)
    heading = _font(LATIN_FONT_CANDIDATES, 24)
    max_x = size - 200

    _draw_text_fit(draw, (36, 40), _SCAN_HEADING, heading, max_x=max_x,
                   max_y=size, fill=(25, 25, 25))
    y = 96
    for line in _DENSE_LINES[:8]:
        _draw_text_fit(draw, (36, y), line, body, max_x=max_x, max_y=size,
                       fill=(45, 45, 45))
        y += 40

    img = img.rotate(-1.4, resample=Image.BILINEAR, fillcolor=(252, 250, 245))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))

    rng = np.random.default_rng(_SCAN_SEED)
    pixels = np.asarray(img).astype(np.int16)
    speckle = rng.normal(0.0, 7.0, pixels.shape)
    pixels = np.clip(pixels + speckle, 0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def _long_blocks() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """``(heading, body lines)`` per qualification block, in drawing order.

    Shared by :func:`make_page_long` and :data:`GROUND_TRUTH` so the drawn page
    and its expected text cannot drift apart.
    """

    return tuple(
        (
            f"{block + 1}. Section heading for qualification block",
            tuple(_DENSE_LINES[block % len(_DENSE_LINES)] for _ in range(3)),
        )
        for block in range(6)
    )


def make_page_long(path: Path, width: int = BENCH_PAGE_SIZE,
                   height: int = BENCH_LONG_HEIGHT) -> None:
    """A block-heavy full page: many separated regions, so the output runs long."""

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    heading = _font(LATIN_FONT_CANDIDATES, 36)
    body = _font(LATIN_FONT_CANDIDATES, 20)
    caption = _font(LATIN_FONT_CANDIDATES, 18)

    _draw_text_fit(draw, (64, 56), _LONG_TITLE, heading, max_x=width - 64,
                   max_y=height, fill=(10, 10, 10))
    draw.rectangle([64, 118, width - 64, 122], fill=(70, 70, 70))

    y = 168
    for block_heading, body_lines in _long_blocks():
        _draw_text_fit(draw, (64, y), block_heading, body, max_x=width - 64,
                       max_y=height, fill=(20, 20, 20))
        y += 36
        for line in body_lines:
            _draw_text_fit(draw, (80, y), line, caption, max_x=width - 64,
                           max_y=height, fill=(45, 45, 45))
            y += 28
        if y + 70 > height:
            raise ValueError(
                f"image block runs off the page (y={y + 70}, height={height})"
            )
        draw.rectangle([80, y + 6, 300, y + 70], fill=(215, 215, 215))
        y += 104
    img.save(path)


PAGES = {
    "ja": make_page_ja,
    "mixed": make_page_mixed,
    "dense": make_page_dense,
    "table": make_page_table,
    "blank": make_page_blank,
    "scan": make_page_scan,
    "long": make_page_long,
    "a4": make_page_a4,
}


# ---------------------------------------------------------------------------
# independent ground truth
# ---------------------------------------------------------------------------
#
# The text actually drawn on each fixture page. This is the reference a
# transcription acceptance test scores against: it comes from the drawing
# source, never from a model run, so reproducing a model's output (including
# its mistakes) cannot satisfy it. ``lines`` is in drawn reading order.
# ``table`` is the ruled grid for pages that have one.

def _a4_paragraphs() -> tuple[str, ...]:
    """The A4 page's text in reading order, as logical paragraphs.

    Every other fixture draws one logical unit per physical line, so its ground
    truth is the drawn lines. The A4 body is *wrapped prose*: a paragraph of
    4-5 physical lines is one sentence-flow, and a transcription that returns
    it as one block is correct. Scoring the wrapped lines as separate units
    reads that correct output as 22 omissions and CER 0.89, so the ground truth
    is the paragraph — the same text, joined at the wrap points.

    Shared by :func:`make_page_a4`'s source constants and :data:`GROUND_TRUTH`
    so the drawn page and its expected text cannot drift apart.
    """

    paragraphs = [_A4_TITLE, _A4_META, " ".join(_A4_ABSTRACT)]
    for heading_text, lines in _A4_SECTIONS:
        paragraphs.append(heading_text)
        paragraphs.append(" ".join(lines))
        if heading_text.startswith("2."):
            paragraphs.append(_A4_TABLE_CAPTION)
    return tuple(paragraphs)


GROUND_TRUTH: dict[str, dict[str, object]] = {
    "ja": {"lines": (_JA_HEADING, *_JA_BODY)},
    "mixed": {"lines": (_MIXED_HEADING, *_MIXED_LINES)},
    "dense": {"lines": _DENSE_LINES},
    "table": {
        "lines": ("Inspection summary",),
        "table": {"header": _TABLE_HEADER, "rows": _TABLE_ROWS},
    },
    "blank": {"lines": ()},
    "scan": {"lines": (_SCAN_HEADING, *_DENSE_LINES[:8])},
    "long": {
        "lines": (
            _LONG_TITLE,
            *tuple(
                line
                for block_heading, body_lines in _long_blocks()
                for line in (block_heading, *body_lines)
            ),
        )
    },
    "a4": {
        # Paragraph-level, not drawn-line-level: the body is wrapped prose.
        "lines": _a4_paragraphs(),
        "table": {"header": _TABLE_HEADER, "rows": _TABLE_ROWS},
    },
}


def expected_lines(page: str) -> tuple[str, ...]:
    """Reading-order text units for a fixture page name.

    One unit per drawn line for a page whose lines are one logical unit each,
    and one unit per paragraph for a page whose body is wrapped prose.
    """

    return tuple(GROUND_TRUTH[page]["lines"])  # type: ignore[arg-type]


def page_filename(name: str) -> str:
    """Fixture filename for ``name``, validated against the ground truth."""

    if name not in GROUND_TRUTH:
        raise KeyError(f"unknown page {name!r}; known: {sorted(GROUND_TRUTH)}")
    return f"page_{name}.png"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("tests/fixtures/surya"))
    parser.add_argument("--only", default=None,
                        help="comma-separated page names (default: all)")
    args = parser.parse_args()

    only = (
        [name.strip() for name in args.only.split(",") if name.strip()]
        if args.only else None
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    wanted = only if only is not None else list(PAGES)
    unknown = [name for name in wanted if name not in PAGES]
    if unknown:
        raise SystemExit(f"unknown page(s) {unknown}; known: {sorted(PAGES)}")

    for name in wanted:
        path = args.out_dir / page_filename(name)
        PAGES[name](path)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
