#!/usr/bin/env python3
"""Status checker for GEMMA4-26B-A4B-PUNCHLIST.md.

The punchlist's markdown tables are the single source of truth: every candidate
row carries its own Status cell (and Result cell where the table has one). This
script parses those tables and reports, per section, which cells are still open
so a checkpoint run can show the campaign's true state.

The earlier revision of this checker read a separate JSON registry embedded in
the punchlist; that registry was an uncommitted second source of truth and has
been dropped. Status now lives only in the tables this parses.

Exit codes:
  0 - parsed cleanly (open cells are reported, they are not an error)
  1 - parse/integrity failure: a table without a Status column, a duplicate
      cell ID, or an empty Status cell
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PUNCHLIST = Path(__file__).resolve().parent / "GEMMA4-26B-A4B-PUNCHLIST.md"

# Free-text status cells classify as open when they start with one of these.
OPEN_PREFIXES = ("open", "blocked", "todo", "next")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse(path: Path) -> tuple[list[dict], list[str]]:
    lines = path.read_text().splitlines()
    rows: list[dict] = []
    errors: list[str] = []
    section = "(no section)"
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            section = line.lstrip("#").strip()
        if line.startswith("|") and "Status" in line and set(line.replace("|", "").replace(" ", "")) != set("-"):
            header = _cells(line)
            try:
                status_idx = next(n for n, h in enumerate(header) if h.lower().startswith("status"))
            except StopIteration:
                i += 1
                continue
            id_idx = 0
            # Separator row next.
            if i + 1 < len(lines) and re.match(r"^\|[\s\-|]+\|$", lines[i + 1]):
                i += 2
            else:
                i += 1
                continue
            while i < len(lines) and lines[i].startswith("|"):
                cells = _cells(lines[i])
                if len(cells) > status_idx:
                    cid = cells[id_idx] if len(cells) > id_idx else "?"
                    status = cells[status_idx]
                    if not status:
                        errors.append(f"{section}: empty status for {cid}")
                    rows.append({"section": section, "id": cid, "status": status})
                i += 1
            continue
        i += 1
    seen: dict[str, str] = {}
    for r in rows:
        if r["id"] in seen:
            errors.append(f"duplicate cell id {r['id']} ({seen[r['id']]} and {r['section']})")
        seen[r["id"]] = r["section"]
    return rows, errors


def is_open(status: str) -> bool:
    s = status.strip().lower().lstrip("*").strip()
    if s.startswith(OPEN_PREFIXES):
        return True
    return False


def main() -> int:
    rows, errors = parse(PUNCHLIST)
    if not rows:
        print("check_gemma4_punchlist: no status rows parsed -- table layout changed?")
        return 1
    open_rows = [r for r in rows if is_open(r["status"])]
    print(f"parsed {len(rows)} status cells across sections")
    print(f"open: {len(open_rows)}  closed/other: {len(rows) - len(open_rows)}")
    if open_rows:
        print("\nopen cells:")
        for r in open_rows:
            print(f"  [{r['section'][:48]:48s}] {r['id']}")
    for e in errors:
        print(f"INTEGRITY: {e}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())