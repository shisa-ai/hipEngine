#!/usr/bin/env python3
"""Deterministic page fixtures for the Surya OCR benchmark suite.

The original fixtures are bar patterns and simple prose pages. Those are enough
to gate numerical parity but not to judge transcription quality or to tune
against, so this module adds the document types that actually move the answer:
Japanese and mixed-script pages, dense small text, ruled tables, a blank page,
a noisy/rotated scan, and a block-heavy page whose output runs long.

Everything is seeded and geometry-aligned to a multiple of 32 so the smart
resize is a no-op and the vision grid is clean, matching the existing fixtures.
Regenerating produces byte-identical PNGs.

Usage:
    python3 scripts/surya_bench_pages.py --out-dir tests/fixtures/surya
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


def _font(paths: tuple[str, ...], size: int):
    from PIL import ImageFont

    for candidate in paths:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    raise RuntimeError(f"no usable font found among {paths}")


def make_page_ja(path: Path, size: int = 512) -> None:
    """Japanese-only dense page: heading plus five body paragraphs."""

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    heading = _font(CJK_FONT_CANDIDATES, 26)
    body = _font(CJK_FONT_CANDIDATES, 17)

    draw.text((32, 30), _JA_HEADING, fill=(10, 10, 10), font=heading)
    y = 84
    for line in _JA_BODY:
        draw.text((32, y), line, fill=(30, 30, 30), font=body)
        y += 52
    draw.rectangle([32, y + 8, size - 32, y + 11], fill=(150, 150, 150))
    img.save(path)


def make_page_mixed(path: Path, size: int = 512) -> None:
    """Japanese and English interleaved, so both scripts share one page."""

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    heading = _font(CJK_FONT_CANDIDATES, 22)
    cjk = _font(CJK_FONT_CANDIDATES, 16)
    latin = _font(LATIN_FONT_CANDIDATES, 16)

    draw.text((32, 28), "Mixed script page / 混在文書", fill=(10, 10, 10), font=heading)
    y = 80
    for index, line in enumerate(_MIXED_LINES):
        font = latin if index % 2 == 0 else cjk
        draw.text((32, y), line, fill=(30, 30, 30), font=font)
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


def make_page_blank(path: Path, size: int = 512) -> None:
    """A blank page with a faint border: the empty-output edge case."""

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, size - 9, size - 9], outline=(225, 225, 225), width=2)
    img.save(path)


def make_page_scan(path: Path, size: int = 512) -> None:
    """A degraded scan: real text, slight rotation, speckle, and blur.

    Deterministic: the noise field comes from a fixed-seed generator, so the
    fixture is reproducible byte for byte.
    """

    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    img = Image.new("RGB", (size, size), (252, 250, 245))
    draw = ImageDraw.Draw(img)
    body = _font(LATIN_FONT_CANDIDATES, 17)
    heading = _font(LATIN_FONT_CANDIDATES, 24)

    draw.text((36, 40), "Scanned inspection record", fill=(25, 25, 25), font=heading)
    y = 96
    for line in _DENSE_LINES[:8]:
        draw.text((36, y), line, fill=(45, 45, 45), font=body)
        y += 40

    img = img.rotate(-1.4, resample=Image.BILINEAR, fillcolor=(252, 250, 245))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))

    rng = np.random.default_rng(_SCAN_SEED)
    pixels = np.asarray(img).astype(np.int16)
    speckle = rng.normal(0.0, 7.0, pixels.shape)
    pixels = np.clip(pixels + speckle, 0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def make_page_long(path: Path, size: int = 1024) -> None:
    """A block-heavy full page: many separated regions, so the output runs long."""

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    heading = _font(LATIN_FONT_CANDIDATES, 36)
    body = _font(LATIN_FONT_CANDIDATES, 20)
    caption = _font(LATIN_FONT_CANDIDATES, 18)

    draw.text((64, 56), "Annual process qualification report", fill=(10, 10, 10),
              font=heading)
    draw.rectangle([64, 118, size - 64, 122], fill=(70, 70, 70))

    y = 168
    for block in range(6):
        draw.text((64, y), f"{block + 1}. Section heading for qualification block",
                  fill=(20, 20, 20), font=body)
        y += 36
        for _ in range(3):
            draw.text((80, y), _DENSE_LINES[block % len(_DENSE_LINES)],
                      fill=(45, 45, 45), font=caption)
            y += 28
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
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("tests/fixtures/surya"))
    parser.add_argument("--only", default=None,
                        help="comma-separated page names (default: all)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wanted = (
        [name.strip() for name in args.only.split(",") if name.strip()]
        if args.only else list(PAGES)
    )
    unknown = [name for name in wanted if name not in PAGES]
    if unknown:
        raise SystemExit(f"unknown page(s) {unknown}; known: {sorted(PAGES)}")

    for name in wanted:
        path = args.out_dir / f"page_{name}.png"
        PAGES[name](path)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
