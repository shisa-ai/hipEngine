#!/usr/bin/env python3
"""Resolve git conflict markers in append-only WORKLOG.md.

WORKLOG.md is append-only. When two branches both add new dated sections,
the correct merge is the common prefix plus both sides' new content. Git's
built-in ``merge=union`` driver (configured in ``.gitattributes``) handles
this automatically going forward, but this script recovers a WORKLOG that
already has ``<<<<<<<`` / ``=======`` / ``>>>>>>>`` markers in it (e.g. a
conflict left over from a rebase performed before the union driver was set
up, or a hand-edited stash).

Behavior
--------
For each conflict block in the file:

* Concatenate HEAD lines (between ``<<<<<<<`` and ``=======``) followed by
  the incoming lines (between ``=======`` and ``>>>>>>>``).
* If ``--sort-by-date`` is passed, top-level ``## YYYY-MM-DD`` sections inside
  the resolved block are re-ordered by date (stable within the same date).
* Identical adjacent sections are deduplicated (defensive; should not happen
  with a strict append-only discipline).

Usage
-----
    python3 scripts/resolve_worklog_conflict.py [WORKLOG.md] [--check] [--sort-by-date]

``--check`` exits non-zero if conflict markers remain after resolution
(useful as a pre-commit gate). Without ``--check`` the file is rewritten
in-place.

This script is intentionally conservative: it does not touch any file
that has no conflict markers, and it never reorders content outside of a
conflict block.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

CONFLICT_START = "<<<<<<<"
CONFLICT_MID = "======="
CONFLICT_END = ">>>>>>>"

SECTION_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2})\b")


def _split_sections(lines: List[str]) -> List[Tuple[str, List[str]]]:
    """Split a list of lines into (date-or-empty, section-lines) chunks.

    The first chunk may have date == "" if the block does not start with a
    dated header; subsequent chunks each start at a ``## YYYY-MM-DD`` line.
    """
    chunks: List[Tuple[str, List[str]]] = []
    current_date = ""
    current: List[str] = []
    for line in lines:
        m = SECTION_RE.match(line)
        if m is not None:
            if current:
                chunks.append((current_date, current))
            current_date = m.group(1)
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append((current_date, current))
    return chunks


def _resolve_block(ours: List[str], theirs: List[str], *, sort_by_date: bool) -> List[str]:
    combined = ours + theirs
    if not sort_by_date:
        return combined

    chunks = _split_sections(combined)
    # Stable sort: undated leading chunk (date == "") keeps its position by
    # being treated as the empty string, which sorts before any date.
    chunks.sort(key=lambda kv: kv[0])

    # Deduplicate adjacent identical chunks (defensive).
    deduped: List[Tuple[str, List[str]]] = []
    for chunk in chunks:
        if deduped and deduped[-1] == chunk:
            continue
        deduped.append(chunk)

    out: List[str] = []
    for _, section_lines in deduped:
        out.extend(section_lines)
    return out


def resolve(text: str, *, sort_by_date: bool = False) -> Tuple[str, int]:
    """Return (resolved_text, num_blocks_resolved)."""
    lines = text.splitlines(keepends=True)
    out: List[str] = []
    i = 0
    n = len(lines)
    blocks = 0
    while i < n:
        line = lines[i]
        if line.startswith(CONFLICT_START):
            # Find the corresponding ======= and >>>>>>> markers.
            mid = -1
            end = -1
            for j in range(i + 1, n):
                stripped = lines[j].rstrip("\n")
                if mid == -1 and stripped == CONFLICT_MID:
                    mid = j
                elif lines[j].startswith(CONFLICT_END):
                    end = j
                    break
            if mid == -1 or end == -1:
                raise ValueError(
                    f"Unterminated conflict block starting at line {i + 1}: "
                    "missing ======= or >>>>>>>"
                )
            ours = lines[i + 1 : mid]
            theirs = lines[mid + 1 : end]
            resolved = _resolve_block(ours, theirs, sort_by_date=sort_by_date)
            out.extend(resolved)
            blocks += 1
            i = end + 1
            continue
        out.append(line)
        i += 1
    return "".join(out), blocks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="WORKLOG.md", help="Path to WORKLOG.md (default: WORKLOG.md)")
    ap.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if conflict markers are present; do not rewrite the file.",
    )
    ap.add_argument(
        "--sort-by-date",
        action="store_true",
        help="Sort resolved sections by their ## YYYY-MM-DD header (stable).",
    )
    ap.add_argument(
        "--stdout",
        action="store_true",
        help="Print resolved content to stdout instead of rewriting the file.",
    )
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 2
    text = path.read_text()

    has_markers = any(
        m in text for m in (f"\n{CONFLICT_START}", f"\n{CONFLICT_MID}\n", f"\n{CONFLICT_END}")
    ) or text.startswith(CONFLICT_START)

    if args.check:
        if has_markers:
            print(f"{path}: conflict markers present", file=sys.stderr)
            return 1
        return 0

    if not has_markers:
        print(f"{path}: no conflict markers, nothing to do")
        return 0

    resolved, n = resolve(text, sort_by_date=args.sort_by_date)

    if args.stdout:
        sys.stdout.write(resolved)
        return 0

    path.write_text(resolved)
    print(f"{path}: resolved {n} conflict block(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
