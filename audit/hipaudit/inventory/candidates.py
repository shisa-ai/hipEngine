"""Lost optimizations: campaign candidates that carry a measured win.

`docs/campaigns/` holds 1,605 "rejected", 97 "deferred", and 31 "parked"
mentions. Some of those are correct calls backed by a measured loss. Others are
a win that was parked behind a gate, a blocker that has since been fixed, or a
candidate removed for failing an exactness bar it was never required to meet —
which `AGENTS.md` and `docs/OPTIMIZATION.md` §4.1 now explicitly reject as a
reason.

This extractor finds candidate rows that were **not promoted but recorded a
number**, because that pairing is where retrievable performance hides. It does
not decide whether the call was right.
"""

from __future__ import annotations

import re

from ..core import REPO_ROOT, Row, digest, slug
from . import doc_corpus, register, tokens

NOT_PROMOTED = re.compile(
    r"\b(rejected|deferred|parked|withdrawn|not promoted|blocked|gated off|default[- ]off|removed)\b", re.I)
#  A speedup ratio, a rate, a percentage, or a millisecond delta.
MEASURED = re.compile(r"\b\d+(?:\.\d+)?\s*(?:x\b|%|tok/s|ms\b|us\b|µs\b|GiB\b)", re.I)
#  Language that names an exactness bar rather than a production gate.
EXACTNESS = re.compile(r"\b(bit[- ]exact|byte[- ]exact|exact(?:ness)? (?:match|parity|gate)|strict parity|flip[- ]free)\b", re.I)
RESOLVED_HINT = re.compile(r"\b(promoted|landed|retained|default[- ]on|now the default)\b", re.I)


@register("candidates")
def extract() -> tuple[list[Row], dict]:
    docs = doc_corpus()
    rows: list[Row] = []
    scanned = 0

    for path, text in sorted(docs.items()):
        if not path.startswith("docs/campaigns/") or path.endswith("README.md"):
            continue
        scanned += 1
        for number, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            #  Table rows and bullets are where candidates are recorded.
            if not (stripped.startswith("|") or stripped.startswith(("- ", "* "))):
                continue
            if len(stripped) < 60:
                continue
            if not NOT_PROMOTED.search(stripped):
                continue
            measurements = MEASURED.findall(stripped)
            if not measurements:
                continue

            signals: list[str] = []
            verdict = NOT_PROMOTED.search(stripped).group(1).lower()
            signals.append(f"recorded as {verdict!r} while citing a measurement")
            if EXACTNESS.search(stripped):
                signals.append(
                    "cites an exactness bar — OPTIMIZATION.md 4.1 says that alone cannot reject "
                    "a production-correct candidate")
            if RESOLVED_HINT.search(stripped):
                signals.append("same row also mentions promotion — may already be resolved")
            if re.search(r"\bblocked\b", stripped, re.I):
                signals.append("recorded as blocked — check whether the blocker still holds")

            #  First cell of a table row is usually the candidate id.
            cells = [c.strip() for c in stripped.strip("|").split("|")] if stripped.startswith("|") else []
            label = cells[0] if cells and len(cells[0]) < 40 else stripped[:80]

            rows.append(Row(
                kind="candidate",
                key=f"{slug(path.rsplit('/', 1)[-1][:-3], 40)}-{slug(label, 24)}-{digest(path, stripped)}",
                title=f"{label} — {path.rsplit('/', 1)[-1]}",
                location=f"{path}:{number}",
                evidence={
                    "verdict_word": verdict,
                    "measurements": measurements[:6],
                    "excerpt": stripped[:400],
                },
                signals=signals,
                hints={
                    "anchor": path,
                    "refs": sorted(set(measurements)),
                    "tokens": tokens(label + " " + stripped[:200]),
                },
            ))

    return rows, {"campaign_docs_scanned": scanned, "candidates": len(rows)}
