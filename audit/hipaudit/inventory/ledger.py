"""docs/REFACTOR.md — the cleanup ledger.

396 `##` headings, no ids, almost none marked resolved. Each heading names code:
paths in backticks, `HIPENGINE_*` flags, symbols. This extractor checks whether
those referents still exist, because an entry about a file that is gone is
either already done or was never real, and either way it is not debt.

It reports what it found. Whether an entry is stale is a triage decision.
"""

from __future__ import annotations

import datetime as dt
import re

from ..core import REPO_ROOT, Row, digest, slug
from . import corpus, grep, register

LEDGER = REPO_ROOT / "docs" / "REFACTOR.md"

HEADING = re.compile(r"^##\s+(.*)$")   # entries are `##`; deeper levels are parts of one
DATE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
RESOLVED = re.compile(r"\b(RESOLVED|DONE|COMPLETE[D]?|LANDED|REMOVED)\b", re.I)
FLAG = re.compile(r"\bHIPENGINE_[A-Z0-9_]+\b")
#  A backticked token that looks like a repo path.
PATHISH = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|hip|md|json|toml|h))`")
#  Language that states when the entry should disappear.
CONDITION = re.compile(r"\b(remove (?:it |this )?(?:when|after|once)|delete when|drop when|until|unblock)\b", re.I)


def sections(text: str) -> list[tuple[int, str, str]]:
    """`(line number, heading, body)` for each `##` block."""
    lines = text.splitlines()
    marks = [(i, m.group(1).strip()) for i, line in enumerate(lines) if (m := HEADING.match(line))]
    out = []
    for index, (line_no, heading) in enumerate(marks):
        end = marks[index + 1][0] if index + 1 < len(marks) else len(lines)
        out.append((line_no + 1, heading, "\n".join(lines[line_no + 1:end])))
    return out


@register("ledger")
def extract() -> tuple[list[Row], dict]:
    if not LEDGER.exists():
        return [], {"source": "docs/REFACTOR.md", "present": False}
    text = LEDGER.read_text(encoding="utf-8", errors="replace")
    code = corpus()
    today = dt.date.today()
    rows: list[Row] = []

    for line_no, heading, body in sections(text):
        block = f"{heading}\n{body}"
        signals: list[str] = []

        paths = sorted(set(PATHISH.findall(block)))
        missing = [p for p in paths if not (REPO_ROOT / p).exists()]
        flags = sorted(set(FLAG.findall(block)))
        unread = [f for f in flags if not grep(f, code)]

        if RESOLVED.search(heading):
            signals.append("heading marked resolved")
        date_match = DATE.search(block)
        age_days = None
        if date_match:
            try:
                found = dt.date(*(int(g) for g in date_match.groups()))
                age_days = (today - found).days
                if age_days > 120:
                    signals.append(f"dated {found.isoformat()}, {age_days} days old")
            except ValueError:
                pass
        else:
            signals.append("no date in entry")

        if paths and missing:
            signals.append(f"names {len(paths)} path(s), {len(missing)} no longer exist: {', '.join(missing[:3])}")
        if flags and unread:
            signals.append(f"names flag(s) no code reads: {', '.join(unread[:3])}")
        if not paths and not flags:
            signals.append("names no path or flag — cannot be checked mechanically")
        if not CONDITION.search(block):
            signals.append("states no removal condition")
        if len(body.strip()) < 80:
            signals.append("body under 80 characters")

        rows.append(Row(
            kind="ledger",
            key=f"{slug(heading, 50)}-{digest(heading)}",
            title=heading[:160],
            location=f"docs/REFACTOR.md:{line_no}",
            evidence={
                "paths_named": paths[:12], "paths_missing": missing[:12],
                "flags_named": flags[:12], "flags_unread": unread[:12],
                "age_days": age_days, "body_chars": len(body.strip()),
            },
            signals=signals,
        ))

    return rows, {"source": "docs/REFACTOR.md", "headings": len(rows)}
