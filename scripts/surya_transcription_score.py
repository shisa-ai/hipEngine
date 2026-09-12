"""Score a Surya full-page transcription against independent ground truth.

The lane's older gate was "reproduce the oracle's greedy token ids". That is
regression coverage: it can pass while the page is transcribed as garbage, and
it cannot say *how much* of the page was read. This module measures the things
a transcription claim actually needs:

* **reading order** — do the regions the model emitted follow the drawn order?
* **line recall / omissions** — was each drawn line transcribed at all?
* **character error rate** — how much of the drawn text was wrong?
* **table structure** — were the rows, columns, header, and cell values read?
* **truncation** — did the output stop because it ran out of token budget?

Ground truth comes from ``scripts/surya_bench_pages.py`` (the text drawn on the
fixture), never from a model run, so reproducing a model's own mistakes cannot
satisfy it.

Pure module: no torch, no GPU, no model. Importable by tests and by the
qualification harness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from hipengine.generation.surya_protocol import FullPageBlock

_WS_RE = re.compile(r"\s+")

# Labels whose content is an image or a rule, not text. A region the model
# marks as one of these is expected to carry no transcription.
NON_TEXT_LABELS = frozenset(
    {"Figure", "Image", "Diagram", "Blank-Page", "Chemical-Block"}
)


def normalize(text: str) -> str:
    """Whitespace-normalized comparison form."""

    return _WS_RE.sub(" ", text).strip()


def levenshtein(a: str, b: str) -> int:
    """Character edit distance (insert/delete/substitute), all costs 1."""

    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,          # deletion
                    current[j - 1] + 1,       # insertion
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def similarity(a: str, b: str) -> float:
    """Normalized similarity in ``[0, 1]``; 1.0 means identical."""

    a = normalize(a)
    b = normalize(b)
    if not a and not b:
        return 1.0
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - levenshtein(a, b) / longest


@dataclass
class CandidateLine:
    """One text line the model produced, and the block it came from."""

    text: str
    block_index: int


@dataclass
class LineMatch:
    expected: str
    matched: str | None
    block_index: int | None
    similarity: float
    distance: int

    @property
    def present(self) -> bool:
        return self.matched is not None

    @property
    def exact(self) -> bool:
        """The line was transcribed exactly (after whitespace normalization)."""

        return self.matched is not None and self.similarity >= 1.0


@dataclass
class TranscriptionScore:
    """Measured transcription quality for one page."""

    page: str
    finish_reason: str
    n_blocks: int
    n_candidate_lines: int

    lines_expected: int = 0
    lines_found: int = 0
    line_recall: float = 1.0
    lines_exact: int = 0
    line_exact_rate: float = 1.0
    cer: float = 0.0
    omissions: list[str] = field(default_factory=list)
    reading_order_violations: int = 0

    table_expected: bool = False
    table_found: bool = False
    table_shape_match: bool = False
    table_header_match: bool = False
    table_cells_expected: int = 0
    table_cells_matched: int = 0
    table_cell_accuracy: float = 1.0

    skipped_non_text_blocks: int = 0

    @property
    def truncated(self) -> bool:
        """The decode stopped at the token budget instead of at EOS."""

        return self.finish_reason != "eos"

    def as_dict(self) -> dict:
        payload = {
            "page": self.page,
            "finish_reason": self.finish_reason,
            "truncated": self.truncated,
            "n_blocks": self.n_blocks,
            "n_candidate_lines": self.n_candidate_lines,
            "lines_expected": self.lines_expected,
            "lines_found": self.lines_found,
            "line_recall": self.line_recall,
            "lines_exact": self.lines_exact,
            "line_exact_rate": self.line_exact_rate,
            "cer": self.cer,
            "omissions": list(self.omissions),
            "reading_order_violations": self.reading_order_violations,
            "skipped_non_text_blocks": self.skipped_non_text_blocks,
            "table_expected": self.table_expected,
            "table_found": self.table_found,
            "table_shape_match": self.table_shape_match,
            "table_header_match": self.table_header_match,
            "table_cells_expected": self.table_cells_expected,
            "table_cells_matched": self.table_cells_matched,
            "table_cell_accuracy": self.table_cell_accuracy,
        }
        return payload


def candidate_lines(blocks: list[FullPageBlock]) -> list[CandidateLine]:
    """Every non-empty line of every block, tagged with its block index.

    A block that holds several ``<p>`` lines contributes several candidates, so
    a model that groups a paragraph into one region is not penalized.
    """

    out: list[CandidateLine] = []
    for index, block in enumerate(blocks):
        for line in block.text.split("\n"):
            line = normalize(line)
            if line:
                out.append(CandidateLine(text=line, block_index=index))
    return out


def _assign(
    expected: tuple[str, ...], candidates: list[CandidateLine], threshold: float
) -> list[LineMatch]:
    """One-to-one greedy best-similarity assignment of expected lines.

    Greedy over the whole matrix rather than per-expected-line: an expected
    line that appears three times on the page must consume three candidates, so
    a page that drops two of them reports two omissions instead of scoring a
    perfect recall off a single emission.

    ``threshold`` is the *presence* bar: how close a candidate has to be before
    the region counts as transcribed at all. Correctness is reported separately
    through exact matches and the character error rate, so a line the model read
    with a wrong word is found-and-wrong rather than silently missing.
    """

    pairs: list[tuple[float, int, int]] = []
    for i, want in enumerate(expected):
        for j, cand in enumerate(candidates):
            score = similarity(want, cand.text)
            if score >= threshold:
                pairs.append((score, i, j))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    matches: list[LineMatch | None] = [None] * len(expected)
    used_candidate: set[int] = set()
    for score, i, j in pairs:
        if matches[i] is not None or j in used_candidate:
            continue
        cand = candidates[j]
        matches[i] = LineMatch(
            expected=expected[i],
            matched=cand.text,
            block_index=cand.block_index,
            similarity=score,
            distance=levenshtein(normalize(expected[i]), cand.text),
        )
        used_candidate.add(j)

    out: list[LineMatch] = []
    for i, want in enumerate(expected):
        if matches[i] is not None:
            out.append(matches[i])  # type: ignore[arg-type]
        else:
            best = max(
                (similarity(want, c.text) for c in candidates), default=0.0
            )
            out.append(
                LineMatch(
                    expected=want,
                    matched=None,
                    block_index=None,
                    similarity=best,
                    distance=len(normalize(want)),
                )
            )
    return out


def _score_table(
    score: TranscriptionScore, blocks: list[FullPageBlock], expected_table: dict
) -> None:
    header = tuple(normalize(cell) for cell in expected_table.get("header", ()))
    rows = tuple(
        tuple(normalize(cell) for cell in row) for row in expected_table.get("rows", ())
    )
    expected_grid = (header, *rows) if header else rows
    score.table_expected = True
    score.table_cells_expected = sum(len(row) for row in expected_grid)

    found_tables = [table for block in blocks for table in block.tables]
    if not found_tables:
        score.table_cell_accuracy = 0.0
        return

    # Pick the found table with the most matching cells at aligned positions;
    # a page may also emit a small spurious table, and that should not hide the
    # real one.
    def matched_cells(table: list[list[str]]) -> int:
        total = 0
        for r, expected_row in enumerate(expected_grid):
            if r >= len(table):
                break
            found_row = table[r]
            for c, cell in enumerate(expected_row):
                if c < len(found_row) and normalize(found_row[c]) == cell:
                    total += 1
        return total

    best = max(found_tables, key=matched_cells)
    score.table_found = True
    score.table_cells_matched = matched_cells(best)
    score.table_cell_accuracy = (
        score.table_cells_matched / score.table_cells_expected
        if score.table_cells_expected
        else 1.0
    )
    score.table_shape_match = len(best) == len(expected_grid) and all(
        len(best[r]) == len(expected_grid[r]) for r in range(len(expected_grid))
    )
    score.table_header_match = bool(header) and len(best) > 0 and all(
        c < len(best[0]) and normalize(best[0][c]) == header[c]
        for c in range(len(header))
    )


def score_transcription(
    *,
    page: str,
    blocks: list[FullPageBlock],
    expected_lines: tuple[str, ...],
    finish_reason: str,
    expected_table: dict | None = None,
    presence_threshold: float = 0.60,
) -> TranscriptionScore:
    """Measure one page's transcription against independent ground truth."""

    candidates = candidate_lines(blocks)
    score = TranscriptionScore(
        page=page,
        finish_reason=finish_reason,
        n_blocks=len(blocks),
        n_candidate_lines=len(candidates),
        skipped_non_text_blocks=sum(
            1 for block in blocks if block.label in NON_TEXT_LABELS and not block.text
        ),
    )

    matches = _assign(expected_lines, candidates, presence_threshold)
    score.lines_expected = len(expected_lines)
    score.lines_found = sum(1 for m in matches if m.present)
    score.line_recall = (
        score.lines_found / score.lines_expected if score.lines_expected else 1.0
    )
    score.lines_exact = sum(1 for m in matches if m.exact)
    score.line_exact_rate = (
        score.lines_exact / score.lines_expected if score.lines_expected else 1.0
    )
    total_chars = sum(len(normalize(line)) for line in expected_lines)
    total_distance = sum(m.distance for m in matches)
    score.cer = total_distance / total_chars if total_chars else 0.0
    score.omissions = [m.expected for m in matches if not m.present]

    # Reading order: the block index of each expected line, walking the drawn
    # order, must not go backwards.
    last = -1
    for match in matches:
        if match.block_index is None:
            continue
        if match.block_index < last:
            score.reading_order_violations += 1
        last = max(last, match.block_index)
    if expected_table is not None:
        _score_table(score, blocks, expected_table)

    return score


@dataclass(frozen=True)
class TranscriptionThresholds:
    """Acceptance bars for one page.

    Defaults are calibrated on the fp32 baseline and recorded with the
    measurement artifact; they are not tuning targets.
    """

    min_line_recall: float = 0.99
    min_line_exact_rate: float = 0.0
    max_cer: float = 0.02
    max_reading_order_violations: int = 0
    min_table_cell_accuracy: float = 0.99
    require_table_shape: bool = True
    require_eos: bool = True


def evaluate(
    score: TranscriptionScore, thresholds: TranscriptionThresholds
) -> tuple[bool, list[str]]:
    """Apply thresholds, returning ``(passed, failures)`` with named failures."""

    failures: list[str] = []
    if score.line_recall < thresholds.min_line_recall:
        failures.append(
            f"line_recall {score.line_recall:.4f} < {thresholds.min_line_recall:.4f}"
        )
    if score.line_exact_rate < thresholds.min_line_exact_rate:
        failures.append(
            f"line_exact_rate {score.line_exact_rate:.4f} < "
            f"{thresholds.min_line_exact_rate:.4f}"
        )
    if score.cer > thresholds.max_cer:
        failures.append(f"cer {score.cer:.4f} > {thresholds.max_cer:.4f}")
    if score.reading_order_violations > thresholds.max_reading_order_violations:
        failures.append(
            f"reading_order_violations {score.reading_order_violations} > "
            f"{thresholds.max_reading_order_violations}"
        )
    if thresholds.require_eos and score.truncated:
        failures.append(f"truncated (finish_reason={score.finish_reason})")
    if score.table_expected:
        if not score.table_found:
            failures.append("table not found")
        else:
            if score.table_cell_accuracy < thresholds.min_table_cell_accuracy:
                failures.append(
                    f"table_cell_accuracy {score.table_cell_accuracy:.4f} < "
                    f"{thresholds.min_table_cell_accuracy:.4f}"
                )
            if thresholds.require_table_shape and not score.table_shape_match:
                failures.append("table shape mismatch")
    return (not failures, failures)
