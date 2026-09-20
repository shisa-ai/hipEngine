"""worklog/entries/ — work the journal itself declared unfinished.

Every Worklog2 entry is immutable and carries a `status`, so the journal is
already a ledger of what was claimed. Most of it is history: a `completed`
entry's `## Next` is the journal's forward reference to the *following* unit,
not a claim that this one is open. Reading every entry as debt would bury the
few that matter — about a tenth of the current entries, and 109 of the 7,232
ported ones, yield a row.

This extractor therefore emits a row only where an entry declares unfinished
business, by one of three mechanical markers:

- **status** — `blocked`, `handoff`, or `checkpoint`: the entry's own field says
  the unit is blocked, passed to another worker, or was not complete.
- **dependency wording** — a `completed`/`decision` entry whose `## Next` names a
  blocker, gate, or approval it did not have. That is the entry saying the next
  step could not proceed, which is why the wording is kept narrow: a bare mention
  of the word "blocker" is usually a record being preserved, not an open one.
- **legacy marker** — a ported pre-cutoff entry whose verbatim body carries an
  explicit `### Next` subsection. The pre-Worklog2 journal used that heading
  only when something was left over (109 of 7,232 entries), whereas the current
  schema requires a `## Next` section on every entry.

Whether the work was finished later is a judgement, so it is left to triage. A
later entry by the same worker on the same topic prefix is reported as an
observation and never closes a row on its own — the topic prefix is a hint, and
a worker who renames a topic would otherwise make debt disappear.
"""

from __future__ import annotations

import datetime as dt
import functools
import pathlib
import re
from collections import defaultdict

from ..core import REPO_ROOT, Row
from . import corpus, register, tokens

ENTRY_DIR = REPO_ROOT / "worklog" / "entries"

OPEN_STATUSES = ("blocked", "handoff", "checkpoint")
STATUS_MEANING = {
    "blocked": "the entry says the unit is blocked",
    "handoff": "the entry says the work was handed to another worker",
    "checkpoint": "the entry says the unit was not complete at this commit",
}

#  The entry saying the next step could not proceed: a blocker, a gate, or an
#  approval it did not have.
DEPENDENCY = re.compile(
    r"(?:requires?|needs?|awaiting|pending)\s+(?:explicit\s+|human\s+|lead\s+|user\s+|separate\s+)?"
    r"(?:approval|sign[- ]?off|decision|authoriz)"
    r"|\bblocked (?:on|by|until|pending|behind)\b"
    r"|cannot (?:proceed|start|land|be)|can't (?:proceed|start)"
    r"|waiting (?:on|for)\b"
    r"|gated (?:on|behind|by)\b"
    r"|\bnot yet (?:landed|available|implemented|registered|ported|exists?)\b"
    r"|\bno (?:route|path|mechanism) (?:yet|exists)",
    re.I)

#  The port fills `## Next` with this when the old entry had nothing left over.
BOILERPLATE = re.compile(
    r"^(?:[-*]\s*)?(?:none\b|no follow[- ]?up|nothing\b|n/?a\b|\(none\)|historical record)", re.I)

FLAG = re.compile(r"\bHIPENGINE_[A-Z0-9_]+\b")
#  A repo-relative path: the slash matters, because a bare `foo.py` in prose is a
#  filename the entry is talking about, not a location in this tree.
PATHISH = re.compile(r"`([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+\.(?:py|hip|md|json|toml|h|sh))`")
#  A fence is indented at most three spaces and opens a run of backticks or
#  tildes; it closes on the same character, at least as long, with no info
#  string. Ported bodies whose text contains its own headings are wrapped in a
#  longer fence, so headings inside one must not be read as sections.
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def split_entry(text: str) -> tuple[dict[str, str], dict[str, int], str, int]:
    """Front-matter fields, their line numbers, the body, and its first line."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing opening front-matter delimiter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError("missing closing front-matter delimiter") from exc
    fields: dict[str, str] = {}
    where: dict[str, int] = {}
    for number, line in enumerate(lines[1:end], 2):
        key, separator, value = line.partition(":")
        if not separator or not key.strip():
            continue
        fields[key.strip()] = value.strip()
        where[key.strip()] = number
    return fields, where, "\n".join(lines[end + 1:]), end + 2


def parse_body(body: str, offset: int) -> tuple[str, dict[str, tuple[int, str]]]:
    """The level-one title and `## Heading` sections, ignoring fenced code.

    Returns `{name: (line number, text)}`, with line numbers absolute in the
    entry file so a row can point at the line that carries its evidence.
    """
    title = ""
    sections: dict[str, tuple[int, str]] = {}
    collected: dict[str, list[str]] = {}
    current = ""
    fence: tuple[str, int] | None = None
    for number, line in enumerate(body.splitlines(), offset):
        match = FENCE.match(line)
        if match:
            run, info = match.group(1), match.group(2).strip()
            if fence is None:
                fence = (run[0], len(run))
            elif run[0] == fence[0] and len(run) >= fence[1] and not info:
                fence = None
        elif fence is None:
            if line.startswith("# ") and not title:
                title = line[2:].strip()
                continue
            if line.startswith("## "):
                current = line[3:].strip()
                sections[current] = (number, "")
                collected[current] = []
                continue
        if current:
            collected[current].append(line)
    #  Section text keeps its own blank lines so a line number computed inside
    #  it stays exact; callers strip it when they only want the words.
    for name, lines in collected.items():
        sections[name] = (sections[name][0], "\n".join(lines))
    return title, sections


def family(topic: str) -> str:
    """The campaign a topic belongs to, as its first two words."""
    return "-".join(topic.split("-")[:2])


def _unread_flags(text: str) -> list[str]:
    """`HIPENGINE_*` names in `text` that no code file mentions.

    The whole code corpus is one haystack: a flag is read if it appears
    anywhere, and the per-file `path:line` that `grep` would return is not worth
    a full corpus scan per row here.
    """
    named = sorted(set(FLAG.findall(text)))
    if not named:
        return []
    haystack = _code_haystack()
    return [name for name in named if name not in haystack]


@functools.lru_cache(maxsize=1)
def _code_haystack() -> str:
    return "\n".join(corpus().values())


def _relative(path: pathlib.Path) -> str:
    """Repo-relative where possible, so a row's anchor is stable in a checkout."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


@register("worklog")
def extract() -> tuple[list[Row], dict]:
    paths = sorted(ENTRY_DIR.glob("*.md")) if ENTRY_DIR.exists() else []
    parsed = []
    unparsable = 0
    for path in paths:
        try:
            fields, where, body, offset = split_entry(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            unparsable += 1
            continue
        title, sections = parse_body(body, offset)
        parsed.append((path, fields, where, title, sections))

    #  What a continuation looks like from outside: a later entry, same worker,
    #  same topic prefix. Never used to suppress a row, only to rank it.
    timeline: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for _, fields, _, _, _ in parsed:
        if fields.get("worker") != "legacy":
            timeline[fields.get("worker", "")].append(
                (fields.get("timestamp", ""), family(fields.get("topic", ""))))

    rows: list[Row] = []
    by_rule: dict[str, int] = defaultdict(int)
    today = dt.date.today()

    for path, fields, where, title, sections in parsed:
        status = fields.get("status", "")
        worker = fields.get("worker", "")
        legacy = worker == "legacy"
        next_line, next_raw = sections.get("Next", (0, ""))
        changes_line, changes_raw = sections.get("Changes", (0, ""))
        next_text = next_raw.strip()
        legacy_next = re.search(r"^### Next\b", changes_raw, re.M) if legacy else None

        dependency = None
        if next_text and not BOILERPLATE.match(next_text):
            dependency = DEPENDENCY.search(next_text)

        if legacy and legacy_next:
            open_by, open_text = "legacy-next", changes_raw.strip()
            location_line = changes_line + changes_raw[:legacy_next.start()].count("\n") + 1
        elif status in OPEN_STATUSES:
            open_by, open_text = "status", next_text
            location_line = where.get("status", where.get("topic", 1))
        elif dependency and status in ("completed", "decision"):
            open_by, open_text = "dependency", next_text
            location_line = next_line
        else:
            continue

        written = fields.get("timestamp", "")[:10]
        age_days = None
        try:
            age_days = (today - dt.date.fromisoformat(written)).days
        except ValueError:
            pass

        later = 0
        if not legacy:
            mine = (fields.get("timestamp", ""), family(fields.get("topic", "")))
            later = sum(1 for stamp, group in timeline.get(worker, []) if stamp > mine[0] and group == mine[1])

        named_paths = sorted(set(PATHISH.findall(open_text)))
        missing = [name for name in named_paths if not (REPO_ROOT / name).exists()]
        unread = _unread_flags(open_text)

        signals: list[str] = []
        if open_by == "status":
            signals.append(f"recorded as {status} — {STATUS_MEANING[status]}")
            if dependency:
                signals.append(f"Next also names a dependency: {dependency.group(0)!r}")
        elif open_by == "dependency":
            signals.append(f"Next names a dependency: {dependency.group(0)!r}")
        else:
            signals.append("carries an explicit `### Next` subsection — the pre-Worklog2 "
                           f"journal used it only when work was left over; this entry describes "
                           f"the tree as of {written or 'the cutoff'}")
        if not legacy:
            if later:
                signals.append(f"{later} later entr{'y' if later == 1 else 'ies'} by the same worker "
                               f"share this topic prefix")
            else:
                signals.append("no later entry by the same worker shares this topic prefix")
        if age_days is not None and age_days > 90 and not legacy:
            signals.append(f"written {age_days} days ago")
        if missing:
            signals.append(f"names {len(missing)} path(s) that are not in the tree: {', '.join(missing[:3])}")
        if unread:
            signals.append(f"names flag(s) no code reads: {', '.join(unread[:3])}")

        by_rule[open_by] += 1
        rows.append(Row(
            kind="worklog",
            key=path.stem,
            title=(title or fields.get("topic", path.stem))[:160],
            location=f"{_relative(path)}:{location_line}",
            evidence={
                "status": status,
                "worker": worker,
                "topic": fields.get("topic", ""),
                "written": written,
                "age_days": age_days,
                "open_by": open_by,
                "continuation_entries": later,
                "open_text": open_text[:600],
                "paths_missing": missing[:8],
                "flags_unread": unread[:8],
            },
            signals=signals,
            hints={
                "anchor": _relative(path),
                "refs": sorted(set(named_paths) | set(FLAG.findall(open_text))),
                "tokens": tokens(title or fields.get("topic", "")),
            },
        ))

    meta = {
        "source": "worklog/entries",
        "entries": len(paths),
        "current": sum(1 for _, f, _, _, _ in parsed if f.get("worker") != "legacy"),
        "legacy": sum(1 for _, f, _, _, _ in parsed if f.get("worker") == "legacy"),
        "unparsable": unparsable,
        "rows": len(rows),
        "by_rule": dict(sorted(by_rule.items())),
    }
    return rows, meta
