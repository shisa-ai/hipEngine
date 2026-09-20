#!/usr/bin/env python3
"""Port the frozen pre-Worklog2 journal into immutable Worklog2 entries.

The legacy append-only journal (``WORKLOG-LEGACY.md``, frozen at the Worklog2
cutoff) is split fence-aware into its historical entries.  Each entry becomes
one immutable file under ``worklog/entries/`` whose timestamp, base commit, and
filename come from the first Git commit that appended it to the journal,
recovered from the full patch history of ``WORKLOG.md``/``WORKLOG-LEGACY.md``.
Entry bodies are preserved verbatim; ``verify`` re-derives every file from the
frozen journal and proves byte-exact reconstruction plus Worklog2 schema
validity.

Commands:
  report     build the commit map and print statistics and anomalies.
  generate   write the entry files and ``worklog/legacy-port-manifest.json``.
  verify     prove on-disk entries match the frozen journal byte-for-byte.

Standard library only; reuses ``scripts/worklog.py`` for schema validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import worklog  # noqa: E402  (scripts/worklog.py is the schema authority)

LEGACY_PATH = ROOT / "WORKLOG-LEGACY.md"
LEGACY_MANIFEST_PATH = ROOT / "worklog" / "legacy-manifest.json"
ENTRY_DIR = ROOT / "worklog" / "entries"
PORT_MANIFEST_PATH = ROOT / "worklog" / "legacy-port-manifest.json"
JOURNAL_PATHS = ("WORKLOG.md", "WORKLOG-LEGACY.md")

WORKER = "legacy"
BRANCH = "pre-cutoff"
WORKTREE = "pre-cutoff"
STATUS = "completed"
SCHEMA = 1

SECTION_NAMES = ("## Summary", "## Changes", "## Validation", "## Next")
FALLBACK_HOUR = 12  # window fallback: heading date at 12:00:00Z


class PortError(ValueError):
    """The legacy port could not be completed consistently."""


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise PortError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


# ---------------------------------------------------------------------------
# Legacy journal splitting
# ---------------------------------------------------------------------------

def fence_inside_map(lines: list[str]) -> set[int]:
    """Indices of lines rendered inside fenced code, per the Worklog2 scanner."""
    inside: set[int] = set()
    fence: tuple[str, int] | None = None
    for index, line in enumerate(lines):
        fence = worklog._fence_transition(line, fence)
        if fence is not None:
            inside.add(index)
    return inside


def split_legacy(text: str) -> tuple[list[str], list[int], list[str], set[int]]:
    """Split the journal into (lines, heading line indices, headings, fenced set)."""
    lines = text.split("\n")
    inside = fence_inside_map(lines)
    heading_indices = [
        index
        for index, line in enumerate(lines)
        if index not in inside and line.startswith("## ")
    ]
    headings = [lines[index][3:] for index in heading_indices]
    return lines, heading_indices, headings, inside


def entry_span(heading_indices: list[int], which: int, total_lines: int) -> tuple[int, int]:
    """One-based inclusive source line range of entry ``which`` (heading through separator)."""
    start = heading_indices[which] + 1
    end = heading_indices[which + 1] if which + 1 < len(heading_indices) else total_lines
    return start, end


def entry_body_lines(
    lines: list[str], heading_indices: list[int], which: int, total_lines: int
) -> list[str]:
    start, end = entry_span(heading_indices, which, total_lines)
    return lines[start:end]  # start/end are one-based; body starts after the heading


def body_needs_wrap(body_lines: list[str], inside: set[int], heading_index: int) -> bool:
    """A body must be fenced when verbatim lines would violate the entry schema."""
    offset = heading_index + 1
    if not any(line.strip() for line in body_lines):
        return True  # an empty section would fail Worklog2's non-empty check
    for position, line in enumerate(body_lines):
        if offset + position in inside:
            continue
        if line.startswith("# ") or line in SECTION_NAMES:
            return True
    return False


# ---------------------------------------------------------------------------
# Commit -> entry mapping
# ---------------------------------------------------------------------------

def build_commit_map(final_counts: Counter[str]) -> dict[str, Any]:
    """Attribute each heading to the first commit that appended it to the journal."""
    stream = run_git(
        "log", "--topo-order", "--reverse", "-p", "-M", "--full-history",
        "--diff-merges=first-parent", "--format=%x01%H%x02%aI%x02%s",
        "--", *JOURNAL_PATHS,
    )
    seen: Counter[str] = Counter()
    events: dict[str, list[dict[str, Any]]] = {}
    anomalies: dict[str, Any] = {"deletion_commits": [], "readd_skipped": 0, "records": 0}
    for record in stream.split("\x01")[1:]:
        newline = record.find("\n")
        header = record[:newline] if newline >= 0 else record
        patch = record[newline + 1 :] if newline >= 0 else ""
        parts = header.split("\x02")
        if len(parts) != 3:
            continue  # boundary or synthetic record without an attributable patch
        anomalies["records"] += 1
        sha, adate, subject = parts
        added: list[str] = []
        deleted = 0
        for line in patch.split("\n"):
            if line.startswith("+++"):
                continue
            if line.startswith("---"):
                if line.startswith("--- a/") or line.startswith("--- /dev/null"):
                    continue
                deleted += 1
                continue
            if line.startswith("+"):
                if line[1:].startswith("## "):
                    added.append(line[4:])
            elif line.startswith("-"):
                deleted += 1
        if deleted:
            anomalies["deletion_commits"].append(
                {
                    "commit": sha,
                    "author_date": adate,
                    "subject": subject,
                    "deleted_lines": deleted,
                    "added_headings": len(added),
                }
            )
        fresh: list[str] = []
        for heading in added:
            if seen[heading] >= final_counts.get(heading, 0):
                anomalies["readd_skipped"] += 1  # union-merge or rewrite replay
                continue
            seen[heading] += 1
            fresh.append(heading)
        for order, heading in enumerate(fresh):
            events.setdefault(heading, []).append(
                {"commit": sha, "author_date": adate, "subject": subject, "order": order}
            )
    missing = {
        heading: final_counts[heading] - seen[heading]
        for heading in final_counts
        if final_counts[heading] > seen[heading]
    }
    return {"events": events, "anomalies": anomalies, "missing": missing}


def parse_author_date(adate: str) -> datetime:
    return datetime.fromisoformat(adate).astimezone(timezone.utc)


def heading_window_date(heading: str) -> datetime:
    """Fallback timestamp: the date written in the heading, at 12:00:00Z."""
    date_text = heading.split(" ", 1)[0].split("/", 1)[0]
    return datetime.strptime(date_text, "%Y-%m-%d").replace(
        hour=FALLBACK_HOUR, tzinfo=timezone.utc
    )


def assign_occurrences(
    headings: list[str], mapping: dict[str, Any], cutoff_commit: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bind each file occurrence of a heading to one attributed append event."""
    assignments: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    occurrence_counts: Counter[str] = Counter()
    fallback_order: Counter[str] = Counter()
    for heading in headings:
        events = list(mapping["events"].get(heading, []))
        occurrence = occurrence_counts[heading]
        occurrence_counts[heading] += 1
        if len(events) > 1:
            if occurrence == 0:  # one manifest record per repeated heading text
                duplicates.append(
                    {
                        "heading": heading,
                        "occurrences": len(events),
                        "assigned_commits": [event["commit"] for event in events],
                    }
                )
        if occurrence < len(events):
            # Union merges append in roughly chronological order, so repeated
            # headings bind their file occurrences to append events by date.
            events.sort(
                key=lambda event: (parse_author_date(event["author_date"]), event["order"])
            )
            event = events[occurrence]
            source = "commit"
        else:
            event = {
                "commit": cutoff_commit,
                "author_date": None,
                "subject": "<not attributed; heading-date window fallback>",
                "order": fallback_order[heading],
            }
            fallback_order[heading] += 1
            source = "window"
        assignments.append(
            {"heading": heading, "event": event, "timestamp_source": source}
        )
    return assignments, duplicates


def entry_timestamp(assignment: dict[str, Any], heading: str) -> tuple[datetime, str]:
    """(UTC datetime, timestamp text); intra-commit order lives in the microseconds."""
    event = assignment["event"]
    if assignment["timestamp_source"] == "commit":
        moment = parse_author_date(event["author_date"])
    else:
        moment = heading_window_date(heading)
    micros = min(int(event["order"]), 999_999)
    text = moment.strftime("%Y-%m-%dT%H:%M:%S") + f".{micros:06d}Z"
    return moment, text


# ---------------------------------------------------------------------------
# Entry rendering
# ---------------------------------------------------------------------------

def stamp_of(timestamp_text: str) -> str:
    """`2026-05-13T02:04:59.000000Z` -> `20260513T020459.000000Z`."""
    return (
        timestamp_text[0:4]
        + timestamp_text[5:7]
        + timestamp_text[8:10]
        + "T"
        + timestamp_text[11:13]
        + timestamp_text[14:16]
        + timestamp_text[17:]
    )


def wrapper_fence_run(body_lines: list[str]) -> int:
    """Backtick run for the wrapper fence: one longer than any body fence line.

    A wrapped body can itself contain fenced code; a wrapper exactly one run
    longer than the longest body fence line can never be closed from inside,
    so the entry's structural sections stay visible to the Worklog2 scanner.
    """
    longest = 3
    for line in body_lines:
        match = worklog._FENCE_RE.match(line)
        if match is not None and match.group(1)[0] == "`":
            longest = max(longest, len(match.group(1)))
    return longest + 1


def render_entry(
    *,
    heading: str,
    body_lines: list[str],
    span: tuple[int, int],
    timestamp_text: str,
    commit: str,
    subject: str,
    topic: str,
    wrapped: bool,
) -> tuple[str, str]:
    """(filename, entry text) for one ported legacy entry."""
    suffix = hashlib.sha256(
        f"{timestamp_text}|{commit}|{span[0]}".encode("utf-8")
    ).hexdigest()[:6]
    filename = f"{stamp_of(timestamp_text)}-legacy-{topic}-{suffix}.md"
    source_note = f"source lines {span[0]}-{span[1]}"
    lines = [
        "---",
        f"schema: {SCHEMA}",
        f"timestamp: {timestamp_text}",
        f"worker: {WORKER}",
        f"branch: {BRANCH}",
        f"worktree: {WORKTREE}",
        f"base_commit: {commit}",
        f"status: {STATUS}",
        f"topic: {topic}",
        "---",
        "",
        f"# {heading}",
        "",
        "## Summary",
        "",
        "- Historical pre-Worklog2 journal entry, ported verbatim from"
        f" `WORKLOG-LEGACY.md` ({source_note}).",
        f"- First appended to the journal in commit `{commit[:8]}` (subject: {subject}).",
        "",
        "## Changes",
        "",
    ]
    if wrapped:
        run = wrapper_fence_run(body_lines)
        lines.append("`" * run + "markdown")
    lines.extend(body_lines)
    if wrapped:
        run = wrapper_fence_run(body_lines)
        lines.append("`" * run)
    validation = (
        "- Ported by `scripts/worklog_port_legacy.py`; `verify` confirms this body"
        f" matches `WORKLOG-LEGACY.md` {source_note} byte-for-byte."
    )
    if wrapped:
        validation += (
            " The body is wrapped in a fenced block because verbatim lines"
            " would otherwise violate the Worklog2 one-title rule."
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            validation,
            "",
            "## Next",
            "",
            "- None. Historical record; the frozen legacy journal remains the"
            " canonical pre-cutoff reference.",
            "",
        ]
    )
    return filename, "\n".join(lines)


def validate_in_memory(filename: str, text: str) -> None:
    """Run the real Worklog2 schema checks on generated content before writing."""
    if not worklog.FILENAME_RE.fullmatch(filename):
        raise PortError(f"generated filename is invalid: {filename}")
    relpath = f"worklog/entries/{filename}"
    try:
        fields, _body = worklog.parse_entry_text(text)
    except worklog.WorklogError as exc:
        raise PortError(f"{relpath}: {exc}") from exc
    problems = worklog.filename_errors(relpath, fields)
    if problems:
        raise PortError(f"{relpath}: {problems}")


# ---------------------------------------------------------------------------
# Legacy integrity
# ---------------------------------------------------------------------------

def load_legacy() -> tuple[str, dict[str, Any]]:
    text = LEGACY_PATH.read_text(encoding="utf-8")
    legacy_manifest = json.loads(LEGACY_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload = LEGACY_PATH.read_bytes()
    actual = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "lines": len(payload.splitlines()),
    }
    for field, value in actual.items():
        if legacy_manifest[field] != value:
            raise PortError(
                f"frozen legacy journal {field} mismatch:"
                f" manifest={legacy_manifest[field]!r} actual={value!r}"
            )
    return text, legacy_manifest


# ---------------------------------------------------------------------------
# Plan assembly
# ---------------------------------------------------------------------------

def build_plan() -> dict[str, Any]:
    text, legacy_manifest = load_legacy()
    lines, heading_indices, headings, inside = split_legacy(text)
    if not heading_indices or heading_indices[0] == 0:
        raise PortError("legacy journal has no preamble; splitter assumptions broken")
    final_counts = Counter(headings)
    mapping = build_commit_map(final_counts)
    assignments, duplicates = assign_occurrences(headings, mapping, legacy_manifest["cutoff_commit"])

    records: list[dict[str, Any]] = []
    wrapped_count = 0
    inversions = 0
    previous: datetime | None = None
    per_month: Counter[str] = Counter()
    for which, heading in enumerate(headings):
        assignment = assignments[which]
        span = entry_span(heading_indices, which, len(lines))
        body_lines = entry_body_lines(lines, heading_indices, which, len(lines))
        wrapped = body_needs_wrap(body_lines, inside, heading_indices[which])
        wrapped_count += 1 if wrapped else 0
        moment, timestamp_text = entry_timestamp(assignment, heading)
        if previous is not None and moment < previous:
            inversions += 1
        previous = moment
        per_month[moment.strftime("%Y-%m")] += 1
        topic = worklog.slugify(heading, fallback="entry")
        filename, entry_text = render_entry(
            heading=heading,
            body_lines=body_lines,
            span=span,
            timestamp_text=timestamp_text,
            commit=assignment["event"]["commit"],
            subject=assignment["event"]["subject"],
            topic=topic,
            wrapped=wrapped,
        )
        validate_in_memory(filename, entry_text)
        records.append(
            {
                "file": f"worklog/entries/{filename}",
                "source_line": span[0],
                "source_end_line": span[1],
                "heading": heading,
                "commit": assignment["event"]["commit"],
                "commit_subject": assignment["event"]["subject"],
                "author_date": assignment["event"]["author_date"],
                "timestamp": timestamp_text,
                "timestamp_source": assignment["timestamp_source"],
                "topic": topic,
                "wrapped": wrapped,
            }
        )
    return {
        "lines": lines,
        "heading_indices": heading_indices,
        "headings": headings,
        "records": records,
        "legacy_manifest": legacy_manifest,
        "stats": {
            "entries": len(headings),
            "preamble_lines": heading_indices[0],
            "wrapped": wrapped_count,
            "duplicates": duplicates,
            "timestamp_inversions": inversions,
            "per_month": dict(sorted(per_month.items())),
            "unattributed": sum(
                1 for record in records if record["timestamp_source"] == "window"
            ),
            "mapping": mapping["anomalies"],
        },
        "missing": mapping["missing"],
    }


def print_report(plan: dict[str, Any]) -> None:
    stats = plan["stats"]
    print(
        f"legacy journal: {stats['entries']} entries"
        f" (preamble {stats['preamble_lines']} lines)"
    )
    print(f"wrapped bodies: {stats['wrapped']}")
    print(f"window-fallback timestamps: {stats['unattributed']}")
    print(f"timestamp inversions in file order: {stats['timestamp_inversions']}")
    print(f"duplicate heading groups: {len(stats['duplicates'])}")
    for group in stats["duplicates"]:
        print(f"  x{group['occurrences']}: {group['heading'][:80]}")
    anomalies = stats["mapping"]
    print(
        f"commit records: {anomalies['records']},"
        f" re-adds skipped: {anomalies['readd_skipped']},"
        f" deletion commits: {len(anomalies['deletion_commits'])}"
    )
    print("entries per month:")
    for month, count in stats["per_month"].items():
        print(f"  {month}: {count}")
    if plan["missing"]:
        print(f"UNATTRIBUTED HEADINGS ({len(plan['missing'])}):")
        for heading, count in plan["missing"].items():
            print(f"  x{count}: {heading[:90]}")
    else:
        print("every entry attributed to a first-append commit")


def write_manifest(plan: dict[str, Any], path: Path) -> None:
    manifest = {
        "schema": SCHEMA,
        "source_path": "WORKLOG-LEGACY.md",
        "source_sha256": plan["legacy_manifest"]["sha256"],
        "source_bytes": plan["legacy_manifest"]["bytes"],
        "source_lines": plan["legacy_manifest"]["lines"],
        "cutoff_commit": plan["legacy_manifest"]["cutoff_commit"],
        "generated_by": "scripts/worklog_port_legacy.py",
        "worker": WORKER,
        "branch": BRANCH,
        "worktree": WORKTREE,
        "status": STATUS,
        "stats": {
            "entries": plan["stats"]["entries"],
            "preamble_lines": plan["stats"]["preamble_lines"],
            "wrapped": plan["stats"]["wrapped"],
            "timestamp_inversions": plan["stats"]["timestamp_inversions"],
            "per_month": plan["stats"]["per_month"],
            "unattributed": plan["stats"]["unattributed"],
            "commit_records": plan["stats"]["mapping"]["records"],
            "readd_skipped": plan["stats"]["mapping"]["readd_skipped"],
            "deletion_commits": plan["stats"]["mapping"]["deletion_commits"],
        },
        "duplicates": plan["stats"]["duplicates"],
        "entries": plan["records"],
    }
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_report(_args: argparse.Namespace) -> int:
    print_report(build_plan())
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    plan = build_plan()
    print_report(plan)
    if plan["missing"]:
        raise PortError(
            "unattributed entries would need window fallback; review before generating"
        )
    ENTRY_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    for which, record in enumerate(plan["records"]):
        body_lines = entry_body_lines(plan["lines"], plan["heading_indices"], which, len(plan["lines"]))
        filename, text = render_entry(
            heading=record["heading"],
            body_lines=body_lines,
            span=(record["source_line"], record["source_end_line"]),
            timestamp_text=record["timestamp"],
            commit=record["commit"],
            subject=record["commit_subject"],
            topic=record["topic"],
            wrapped=record["wrapped"],
        )
        target = ROOT / record["file"]
        if target.exists():
            if target.read_text(encoding="utf-8") == text:
                skipped += 1
                continue
            if not args.force:
                raise PortError(
                    f"{record['file']}: exists with different content (use --force)"
                )
        target.write_text(text, encoding="utf-8")
        written += 1
    write_manifest(plan, PORT_MANIFEST_PATH)
    print(
        f"worklog-port: wrote {written} entries, skipped {skipped} identical;"
        f" manifest at {PORT_MANIFEST_PATH.relative_to(ROOT)}"
    )
    return 0


def cmd_verify(_args: argparse.Namespace) -> int:
    text, _legacy_manifest = load_legacy()
    lines, heading_indices, headings, inside = split_legacy(text)
    manifest = json.loads(PORT_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["source_sha256"] != hashlib.sha256(LEGACY_PATH.read_bytes()).hexdigest():
        raise PortError("port manifest does not match the frozen journal hash")
    records = manifest["entries"]
    if len(records) != len(headings):
        raise PortError(
            f"manifest has {len(records)} records, journal splits into {len(headings)}"
        )
    checked = 0
    for which, record in enumerate(records):
        if record["heading"] != headings[which]:
            raise PortError(f"manifest record {which} heading out of sync with the split")
        span = entry_span(heading_indices, which, len(lines))
        if (record["source_line"], record["source_end_line"]) != span:
            raise PortError(f"manifest record {which} source lines out of sync with the split")
        body_lines = entry_body_lines(lines, heading_indices, which, len(lines))
        expected_wrapped = body_needs_wrap(body_lines, inside, heading_indices[which])
        if record["wrapped"] != expected_wrapped:
            raise PortError(f"manifest record {which} wrapped flag out of sync")
        filename, expected = render_entry(
            heading=record["heading"],
            body_lines=body_lines,
            span=span,
            timestamp_text=record["timestamp"],
            commit=record["commit"],
            subject=record["commit_subject"],
            topic=record["topic"],
            wrapped=record["wrapped"],
        )
        if filename != Path(record["file"]).name:
            raise PortError(f"{record['file']}: filename changed under regeneration")
        disk = ROOT / record["file"]
        if not disk.exists():
            raise PortError(f"{record['file']}: missing")
        actual = disk.read_text(encoding="utf-8")
        if actual != expected:
            raise PortError(f"{record['file']}: content differs from byte-exact regeneration")
        validate_in_memory(filename, actual)
        checked += 1
    print(f"worklog-port: verified {checked} entries byte-for-byte against the frozen journal")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    report = subparsers.add_parser("report", help="show mapping statistics and anomalies")
    report.set_defaults(func=cmd_report)

    generate = subparsers.add_parser("generate", help="write entries and the port manifest")
    generate.add_argument(
        "--force",
        action="store_true",
        help="overwrite entry files that differ from regenerated content",
    )
    generate.set_defaults(func=cmd_generate)

    verify = subparsers.add_parser("verify", help="prove on-disk entries match the journal")
    verify.set_defaults(func=cmd_verify)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except PortError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
