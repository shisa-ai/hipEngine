#!/usr/bin/env python3
"""Create, validate, and render contention-free hipEngine worklog entries.

Tracked current history lives in immutable Markdown files under
``worklog/entries``. The pre-Worklog2 journal was frozen at
``WORKLOG-LEGACY.md``, ported verbatim into entries, and then retired from
the tree; its exact bytes stay pinned by the history ref recorded in
``worklog/legacy-port-manifest.json``. ``WORKLOG.md`` remains a tracked
navigation page; generated local views go under the ignored ``.worklog``
directory.

A historical entry that never satisfied the current schema is repaired
without editing its bytes by one immutable, hash-bound record under
``worklog/corrections``. The record names the exact entry path, the sha256 of
its commit-tree bytes, the specific defect it is allowed to have, and either
the resolved full ``base_commit`` or the new immutable entry that supersedes
it. New entries never take that route: they are validated strictly, and an
entry with no matching record is still rejected for the same defects.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENTRY_DIR = ROOT / "worklog" / "entries"
CORRECTIONS_DIR = ROOT / "worklog" / "corrections"
LEGACY_PATH = ROOT / "WORKLOG-LEGACY.md"
LEGACY_MANIFEST_PATH = ROOT / "worklog" / "legacy-manifest.json"
TRACKED_NAVIGATION_PATH = ROOT / "WORKLOG.md"
DEFAULT_OUTPUT = ROOT / ".worklog" / "WORKLOG.md"
ENTRY_DIR_REL = str(ENTRY_DIR.relative_to(ROOT))
CORRECTIONS_DIR_REL = str(CORRECTIONS_DIR.relative_to(ROOT))
LEGACY_REL = str(LEGACY_PATH.relative_to(ROOT))
LEGACY_MANIFEST_REL = str(LEGACY_MANIFEST_PATH.relative_to(ROOT))
# Every path whose tracked content is immutable and whose index state is gated.
WORKLOG_PATHS = (ENTRY_DIR_REL, CORRECTIONS_DIR_REL, LEGACY_REL, LEGACY_MANIFEST_REL)
SCHEMA_VERSION = "1"
REQUIRED_FIELDS = (
    "schema",
    "timestamp",
    "worker",
    "branch",
    "worktree",
    "base_commit",
    "status",
    "topic",
)
REQUIRED_SECTIONS = ("## Summary", "## Changes", "## Validation", "## Next")
ALLOWED_STATUSES = {"completed", "checkpoint", "decision", "blocked", "handoff"}
PLACEHOLDERS = ("<required>", "<none-or-required-follow-up>")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
FILENAME_RE = re.compile(
    r"^\d{8}T\d{6}\.\d{6}Z-[a-z0-9][a-z0-9-]*-[a-z0-9][a-z0-9-]*-[0-9a-f]{6}\.md$"
)
LEGACY_MANIFEST_FIELDS = (
    "schema",
    "path",
    "cutoff_commit",
    "sha256",
    "bytes",
    "lines",
    "first_heading",
    "last_heading",
)
CORRECTION_FIELDS = (
    "schema",
    "path",
    "sha256",
    "defects",
    "reason",
    "base_commit",
    "superseded_by",
)
# The only schema rules a hash-bound correction may relax. Each defect name has
# exactly one companion field that carries the repair.
CORRECTION_DEFECTS = ("abbreviated_base_commit", "missing_required_sections")
SUPERSEDING_STATUSES = frozenset({"decision", "checkpoint"})
MANAGED_HOOK = """#!/bin/sh
# hipEngine Worklog2 managed pre-commit hook.
root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
if test -f "$root/scripts/worklog.py"; then
    python3 "$root/scripts/worklog.py" check || exit 1
fi
exit 0
"""


class WorklogError(ValueError):
    """A worklog entry or repository invariant is malformed."""


def run_git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise WorklogError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def run_git_raw(args: list[str], *, stdin: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise WorklogError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def index_listing(*, unmerged_only: bool = False) -> list[str]:
    """Worklog paths the Git index tracks (stage 0 unless only unmerged paths are asked for)."""
    args = ["ls-files", "-z"]
    if unmerged_only:
        args.append("-u")
    args.extend(["--", *WORKLOG_PATHS])
    paths: list[str] = []
    for record in run_git_raw(args).decode("utf-8").split("\0"):
        if not record:
            continue
        # `-u`/`-s` records are "<mode> <sha> <stage>\t<path>"; plain listings are bare paths.
        paths.append(record.split("\t", 1)[1] if "\t" in record else record)
    return paths


def unstaged_entry_paths() -> list[str]:
    """Entry files that exist only in the working tree (someone's unfinished work)."""
    payload = run_git_raw(
        ["ls-files", "--others", "--exclude-standard", "-z", "--", ENTRY_DIR_REL]
    )
    return sorted(
        path for path in payload.decode("utf-8").split("\0") if path.endswith(".md")
    )


def read_index_blobs(paths: list[str]) -> dict[str, bytes]:
    """Read staged content for `paths` through one `git cat-file --batch` process."""
    if not paths:
        return {}
    payload = "".join(f":{path}\n" for path in paths).encode("utf-8")
    output = run_git_raw(["cat-file", "--batch"], stdin=payload)
    blobs: dict[str, bytes] = {}
    cursor = 0
    for path in paths:
        newline = output.find(b"\n", cursor)
        if newline < 0:
            raise WorklogError("git cat-file --batch returned truncated output")
        header = output[cursor:newline].decode("utf-8", "replace").split()
        cursor = newline + 1
        if len(header) == 2 and header[1] == "missing":
            raise WorklogError(f"{path}: path is not present in the Git index")
        if len(header) != 3 or header[1] != "blob":
            raise WorklogError(f"{path}: unexpected git cat-file --batch header: {header!r}")
        size = int(header[2])
        blobs[path] = output[cursor : cursor + size]
        cursor += size + 1
    return blobs


def _path_in_head(relpath: str) -> bool:
    """Whether `relpath` is already committed in HEAD, i.e. is historical."""
    if run_git("rev-parse", "--verify", "HEAD", check=False) == "":
        return False
    return bool(run_git("ls-tree", "--name-only", "HEAD", "--", relpath, check=False))


def slugify(value: str, *, fallback: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    # Strip again after truncation: cutting at 48 characters can leave a
    # trailing dash, which would fail the slug self-check in parse_entry_text.
    return value[:48].rstrip("-") or fallback


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def git_context() -> tuple[str, str, str]:
    branch = run_git("branch", "--show-current") or (
        f"detached-{run_git('rev-parse', '--short', 'HEAD')}"
    )
    base_commit = run_git("rev-parse", "HEAD")
    worktree = ROOT.name
    return branch, worktree, base_commit


def default_worker() -> str:
    return (
        os.environ.get("WORKLOG_WORKER", "").strip()
        or run_git("config", "--get", "user.name", check=False)
        or os.environ.get("USER", "").strip()
        or "worker"
    )


def entry_template(
    *,
    timestamp: datetime,
    worker: str,
    branch: str,
    worktree: str,
    base_commit: str,
    status: str,
    topic: str,
    title: str,
) -> str:
    iso_timestamp = timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return f"""---
schema: 1
timestamp: {iso_timestamp}
worker: {worker}
branch: {branch}
worktree: {worktree}
base_commit: {base_commit}
status: {status}
topic: {topic}
---

# {title.strip()}

## Summary

<required>

## Changes

- <required>

## Validation

- <required>

## Next

- <none-or-required-follow-up>
"""


def create_entry(args: argparse.Namespace) -> int:
    if args.status not in ALLOWED_STATUSES:
        raise WorklogError(f"unsupported status: {args.status}")

    title = args.title.strip()
    worker = args.worker.strip() if args.worker else default_worker()
    if not title:
        raise WorklogError("title must not be empty")
    if not worker:
        raise WorklogError("worker must not be empty")

    now = utc_now()
    worker_slug = slugify(worker, fallback="worker")
    topic = slugify(args.topic or title, fallback="work")
    branch, worktree, base_commit = git_context()
    stamp = now.strftime("%Y%m%dT%H%M%S.%fZ")
    ENTRY_DIR.mkdir(parents=True, exist_ok=True)

    for _ in range(10):
        suffix = secrets.token_hex(3)
        path = ENTRY_DIR / f"{stamp}-{worker_slug}-{topic}-{suffix}.md"
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        content = entry_template(
            timestamp=now,
            worker=worker,
            branch=branch,
            worktree=worktree,
            base_commit=base_commit,
            status=args.status,
            topic=topic,
            title=title,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        print(path.relative_to(ROOT))
        return 0

    raise WorklogError("could not allocate a unique worklog entry filename")


_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def _fence_transition(line: str, fence: tuple[str, int] | None) -> tuple[str, int] | None:
    """Apply one body line to the fenced-code state machine.

    A fence line is indented at most three spaces and opens with a run of
    backticks or tildes; it closes an open fence of the same character and at
    least the same length when it carries no info string. Indented fence-like
    lines inside a fence are content, not closers, matching CommonMark.
    """
    match = _FENCE_RE.match(line)
    if match is None:
        return fence
    chars, info = match.group(1), match.group(2).strip()
    if fence is None:
        return chars[0], len(chars)
    char, run = fence
    if chars[0] == char and len(chars) >= run and info == "":
        return None
    return fence


def _markdown_heading_positions(body: str) -> tuple[list[int], dict[str, list[int]]]:
    """Return title and required-section line positions outside fenced code."""
    title_positions: list[int] = []
    section_positions = {section: [] for section in REQUIRED_SECTIONS}
    fence: tuple[str, int] | None = None
    for index, line in enumerate(body.splitlines()):
        fence = _fence_transition(line, fence)
        if fence is not None:
            continue
        if line.startswith("# "):
            title_positions.append(index)
        if line in section_positions:
            section_positions[line].append(index)
    return title_positions, section_positions


def _line_conflict_marker(body: str) -> str | None:
    """A git conflict marker starting a line outside fenced code, if any.

    Markers inside fenced code blocks or mid-line (for example documented
    examples of merge output) are legitimate content, so only column-zero
    markers outside fences are treated as unresolved conflicts.
    """
    fence: tuple[str, int] | None = None
    for line in body.splitlines():
        fence = _fence_transition(line, fence)
        if fence is not None:
            continue
        if line.startswith("<<<<<<<") or line.startswith(">>>>>>>") or line == "=======":
            return line[:80]
    return None


def parse_entry_text(
    text: str,
    *,
    base_commit_override: str | None = None,
    allow_missing_sections: bool = False,
) -> tuple[dict[str, str], str]:
    """Parse one entry, optionally under a hash-bound correction record.

    The two keyword arguments are only ever supplied from a bound
    `EntryCorrection`; every other caller parses strictly. `base_commit_override`
    must extend the recorded abbreviation, and `allow_missing_sections` relaxes
    exactly the required-heading presence and non-empty-body rules while keeping
    heading order, the title, frontmatter, placeholder, and conflict-marker rules.
    """
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise WorklogError("missing opening frontmatter delimiter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise WorklogError("missing closing frontmatter delimiter") from exc

    fields: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" not in line:
            raise WorklogError(f"malformed frontmatter line: {line!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise WorklogError(f"empty frontmatter key/value: {line!r}")
        if key in fields:
            raise WorklogError(f"duplicate frontmatter field: {key}")
        fields[key] = value

    missing = [field for field in REQUIRED_FIELDS if field not in fields]
    extra = [field for field in fields if field not in REQUIRED_FIELDS]
    if missing:
        raise WorklogError(f"missing frontmatter fields: {', '.join(missing)}")
    if extra:
        raise WorklogError(f"unknown frontmatter fields: {', '.join(extra)}")
    if tuple(fields) != REQUIRED_FIELDS:
        raise WorklogError("frontmatter fields are out of order")
    if fields["schema"] != SCHEMA_VERSION:
        raise WorklogError(f"unsupported schema: {fields['schema']}")

    timestamp_text = fields["timestamp"]
    if not TIMESTAMP_RE.fullmatch(timestamp_text):
        raise WorklogError("timestamp must use YYYY-MM-DDTHH:MM:SS.ffffffZ format")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WorklogError(f"invalid timestamp: {timestamp_text}") from exc
    if (
        parsed_timestamp.tzinfo is None
        or parsed_timestamp.utcoffset() != timezone.utc.utcoffset(None)
    ):
        raise WorklogError("timestamp must include UTC timezone")
    if fields["status"] not in ALLOWED_STATUSES:
        raise WorklogError(f"unsupported status: {fields['status']}")
    if not re.fullmatch(r"[0-9a-f]{40}", fields["base_commit"]):
        recorded_commit = fields["base_commit"]
        if base_commit_override is None:
            raise WorklogError(
                "base_commit must be a full 40-character lowercase commit hash"
            )
        if not re.fullmatch(r"[0-9a-f]{4,39}", recorded_commit):
            raise WorklogError(
                "recorded base_commit is not an abbreviated lowercase commit prefix: "
                f"{recorded_commit!r}"
            )
        if not re.fullmatch(r"[0-9a-f]{40}", base_commit_override):
            raise WorklogError(
                "corrected base_commit must be a full 40-character lowercase commit hash"
            )
        if not base_commit_override.startswith(recorded_commit):
            raise WorklogError(
                f"corrected base_commit {base_commit_override!r} does not extend the "
                f"recorded value {recorded_commit!r}"
            )
        fields["base_commit"] = base_commit_override
    if slugify(fields["topic"], fallback="") != fields["topic"]:
        raise WorklogError(
            "topic must be a lowercase kebab-case slug of at most 48 characters "
            f"(got {len(fields['topic'])}: {fields['topic']!r})"
        )

    body = "\n".join(lines[end + 1 :]).strip() + "\n"
    body_lines = body.splitlines()
    title_positions, section_map = _markdown_heading_positions(body)
    if title_positions != [0]:
        raise WorklogError(
            "body must contain exactly one leading level-one title, "
            f"found {len(title_positions)}"
        )
    positions: list[int] = []
    for section in REQUIRED_SECTIONS:
        section_lines = section_map[section]
        if len(section_lines) != 1:
            if allow_missing_sections and section != "## Summary" and not section_lines:
                continue
            raise WorklogError(
                f"expected exactly one {section!r} heading, found {len(section_lines)}"
            )
        positions.append(section_lines[0])
    if positions != sorted(positions):
        raise WorklogError("required sections are out of order")
    if not allow_missing_sections:
        for index, section in enumerate(REQUIRED_SECTIONS):
            content_start = positions[index] + 1
            content_end = (
                positions[index + 1] if index + 1 < len(positions) else len(body_lines)
            )
            if not "\n".join(body_lines[content_start:content_end]).strip():
                raise WorklogError(f"{section!r} section is empty")
    for placeholder in PLACEHOLDERS:
        if placeholder in body:
            raise WorklogError(f"unfinished placeholder remains: {placeholder}")
    conflict_line = _line_conflict_marker(body)
    if conflict_line is not None:
        raise WorklogError(f"git conflict marker present: {conflict_line!r}")

    return fields, text.rstrip() + "\n"


def parse_worktree_entry(relpath: str) -> tuple[dict[str, str], str]:
    return parse_entry_text((ROOT / relpath).read_text(encoding="utf-8"))


def filename_errors(relpath: str, fields: dict[str, str]) -> list[str]:
    """Reasons the entry filename does not match the entry's own metadata."""
    parsed_timestamp = datetime.fromisoformat(fields["timestamp"].replace("Z", "+00:00"))
    expected_stamp = parsed_timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
    expected_worker = slugify(fields["worker"], fallback="worker")
    expected_prefix = f"{expected_stamp}-{expected_worker}-{fields['topic']}-"
    if not Path(relpath).name.startswith(expected_prefix):
        return ["filename does not match entry metadata"]
    return []


@dataclasses.dataclass(frozen=True)
class EntryCorrection:
    """One hash-bound compatibility record for a historical invalid entry."""

    source: str
    path: str
    sha256: str
    defects: tuple[str, ...]
    reason: str
    base_commit: str | None
    superseded_by: str | None


def entry_path_error(relpath: str) -> str | None:
    """Why `relpath` is not a plain entry file directly under worklog/entries."""
    prefix = f"{ENTRY_DIR_REL}/"
    if not relpath.startswith(prefix):
        return f"must name a path under {ENTRY_DIR_REL}/"
    name = relpath[len(prefix) :]
    if "/" in name:
        return f"must name a file directly under {ENTRY_DIR_REL}/"
    if not FILENAME_RE.fullmatch(name):
        return "must name a worklog entry filename"
    return None


def parse_correction_text(relpath: str, text: str) -> EntryCorrection:
    """Strictly parse one immutable correction record.

    A record is accepted only when every field is present, ordered, and typed as
    documented, when the defect it names has its companion repair, and when its
    own filename is the corrected entry's filename with a `.json` extension. The
    sha256 is checked against the commit-tree bytes by `bind_corrections`.
    """
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WorklogError(f"{relpath}: correction record is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise WorklogError(f"{relpath}: correction record must be a JSON object")
    if tuple(value) != CORRECTION_FIELDS:
        missing = [field for field in CORRECTION_FIELDS if field not in value]
        extra = [field for field in value if field not in CORRECTION_FIELDS]
        detail: list[str] = []
        if missing:
            detail.append(f"missing {', '.join(missing)}")
        if extra:
            detail.append(f"unknown {', '.join(extra)}")
        if not detail:
            detail.append("fields are out of order")
        raise WorklogError(f"{relpath}: correction fields invalid: {'; '.join(detail)}")

    schema = value["schema"]
    if isinstance(schema, bool) or schema != 1:
        raise WorklogError(f"{relpath}: correction schema must be 1")

    path = value["path"]
    if not isinstance(path, str):
        raise WorklogError(f"{relpath}: path must be a string")
    path_error = entry_path_error(path)
    if path_error:
        raise WorklogError(f"{relpath}: path {path_error}")
    expected_name = f"{Path(path).stem}.json"
    if Path(relpath).name != expected_name:
        raise WorklogError(
            f"{relpath}: correction filename must be {expected_name!r} "
            "(the corrected entry filename with a .json extension)"
        )

    digest = value["sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise WorklogError(
            f"{relpath}: sha256 must be the 64-character lowercase hash of the entry's "
            "commit-tree bytes"
        )

    defects = value["defects"]
    if (
        not isinstance(defects, list)
        or not defects
        or not all(isinstance(item, str) for item in defects)
    ):
        raise WorklogError(f"{relpath}: defects must be a non-empty list of defect names")
    if sorted(set(defects)) != defects:
        raise WorklogError(f"{relpath}: defects must be sorted and free of duplicates")
    unknown = [item for item in defects if item not in CORRECTION_DEFECTS]
    if unknown:
        raise WorklogError(
            f"{relpath}: unknown defect(s) {', '.join(unknown)}; "
            f"known defects are {', '.join(CORRECTION_DEFECTS)}"
        )

    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip() or "\n" in reason:
        raise WorklogError(f"{relpath}: reason must be one non-empty line")

    base_commit = value["base_commit"]
    if base_commit is not None and (
        not isinstance(base_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", base_commit)
    ):
        raise WorklogError(
            f"{relpath}: base_commit amendment must be a full 40-character lowercase "
            "commit hash or null"
        )
    if ("abbreviated_base_commit" in defects) != (base_commit is not None):
        raise WorklogError(
            f"{relpath}: base_commit must carry the resolved full hash exactly when "
            "defects lists abbreviated_base_commit, and be null otherwise"
        )

    superseded_by = value["superseded_by"]
    if superseded_by is not None:
        if not isinstance(superseded_by, str):
            raise WorklogError(f"{relpath}: superseded_by must be a string or null")
        superseded_error = entry_path_error(superseded_by)
        if superseded_error:
            raise WorklogError(f"{relpath}: superseded_by {superseded_error}")
        if superseded_by == path:
            raise WorklogError(f"{relpath}: superseded_by must name a different entry")
    if ("missing_required_sections" in defects) != (superseded_by is not None):
        raise WorklogError(
            f"{relpath}: superseded_by must name the new correcting entry exactly when "
            "defects lists missing_required_sections, and be null otherwise"
        )

    return EntryCorrection(
        source=relpath,
        path=path,
        sha256=digest,
        defects=tuple(defects),
        reason=reason,
        base_commit=base_commit,
        superseded_by=superseded_by,
    )


def load_corrections(blobs: dict[str, bytes]) -> tuple[dict[str, EntryCorrection], list[str]]:
    """Parse every staged correction record, keyed by the entry it corrects.

    The filename rule (`<entry stem>.json`) makes the directory hold at most one
    record per corrected entry, so a second record for the same entry path cannot
    be staged without colliding on the path itself.
    """
    corrections: dict[str, EntryCorrection] = {}
    errors: list[str] = []
    for relpath in sorted(blobs):
        try:
            text = blobs[relpath].decode("utf-8")
        except UnicodeError as exc:
            errors.append(f"{relpath}: {exc}")
            continue
        try:
            correction = parse_correction_text(relpath, text)
        except WorklogError as exc:
            errors.append(str(exc))
            continue
        corrections[correction.path] = correction
    return corrections, errors


def bind_corrections(
    corrections: dict[str, EntryCorrection],
    entry_blobs: dict[str, bytes],
) -> tuple[dict[str, EntryCorrection], list[str]]:
    """Keep only corrections whose entry bytes and HEAD membership match exactly.

    A record that names an untracked entry, an entry that is not already
    committed, or bytes whose sha256 differs is an error and is not applied, so
    the entry it names still faces the full strict schema.
    """
    bound: dict[str, EntryCorrection] = {}
    errors: list[str] = []
    for path, correction in corrections.items():
        if path not in entry_blobs:
            errors.append(f"{correction.source}: corrected entry is not tracked: {path}")
            continue
        actual = hashlib.sha256(entry_blobs[path]).hexdigest()
        if actual != correction.sha256:
            errors.append(
                f"{correction.source}: sha256 mismatch for {path}: "
                f"recorded {correction.sha256}, commit tree has {actual}"
            )
            continue
        if not _path_in_head(path):
            errors.append(
                f"{correction.source}: corrected entry must already be committed in HEAD, "
                f"so a correction cannot exempt a new entry: {path}"
            )
            continue
        bound[path] = correction
    return bound, errors


def supersession_errors(
    corrections: dict[str, EntryCorrection],
    parsed: list[tuple[Path, dict[str, str], str]],
) -> list[str]:
    """A structural correction must be carried by a valid, linked new entry."""
    entries = {str(path.relative_to(ROOT)): (fields, text) for path, fields, text in parsed}
    errors: list[str] = []
    for path, correction in corrections.items():
        target = correction.superseded_by
        if target is None:
            continue
        if target in corrections:
            errors.append(
                f"{correction.source}: superseded_by must pass the strict schema "
                f"without a correction: {target}"
            )
            continue
        record = entries.get(target)
        if record is None:
            errors.append(
                f"{correction.source}: superseded_by is not a valid entry in the commit tree: "
                f"{target}"
            )
            continue
        fields, text = record
        if fields["status"] not in SUPERSEDING_STATUSES:
            allowed = " or ".join(sorted(SUPERSEDING_STATUSES))
            errors.append(
                f"{correction.source}: superseded_by entry {target} must have status "
                f"{allowed}, got {fields['status']!r}"
            )
        if path not in text:
            errors.append(
                f"{correction.source}: superseded_by entry {target} must name the "
                f"corrected path {path}"
            )
    return errors


def validate_records(
    records: list[tuple[str, str]],
    *,
    corrections: dict[str, EntryCorrection] | None = None,
) -> tuple[list[tuple[Path, dict[str, str], str]], list[str]]:
    """Validate (path, text) pairs; return the parsed entries plus every error found.

    `corrections` carries only records already bound to an exact sha256 of these
    bytes. An entry with no bound record faces the full strict schema, and a bound
    record that is not needed (the entry already parses strictly) is itself an
    error so a correction cannot quietly broaden the schema.
    """
    parsed: list[tuple[Path, dict[str, str], str]] = []
    errors: list[str] = []
    bound = corrections or {}
    for relpath, text in records:
        if not FILENAME_RE.fullmatch(Path(relpath).name):
            errors.append(f"{relpath}: invalid filename")
            continue
        correction = bound.get(relpath)
        if correction is None:
            try:
                fields, body = parse_entry_text(text)
            except (UnicodeError, WorklogError) as exc:
                errors.append(f"{relpath}: {exc}")
                continue
        else:
            try:
                parse_entry_text(text)
            except (UnicodeError, WorklogError):
                pass
            else:
                errors.append(
                    f"{relpath}: correction record {correction.source} is unnecessary; "
                    "the entry already passes strict validation"
                )
                continue
            try:
                fields, body = parse_entry_text(
                    text,
                    base_commit_override=correction.base_commit,
                    allow_missing_sections="missing_required_sections" in correction.defects,
                )
            except (UnicodeError, WorklogError) as exc:
                errors.append(
                    f"{relpath}: correction record {correction.source} does not make this "
                    f"entry valid: {exc}"
                )
                continue
        mismatch = filename_errors(relpath, fields)
        if mismatch:
            errors.extend(f"{relpath}: {reason}" for reason in mismatch)
            continue
        parsed.append((ROOT / relpath, fields, body))
    return parsed, errors


def entry_directory_errors(index_paths: list[str]) -> list[str]:
    """Tracked paths under `worklog/entries` must be plain `*.md` files directly inside it."""
    errors: list[str] = []
    for path in index_paths:
        if "/" in path[len(ENTRY_DIR_REL) + 1 :] or not path.endswith(".md"):
            errors.append(f"unexpected path under worklog entries: {path}")
    return errors


def correction_directory_errors(index_paths: list[str]) -> list[str]:
    """Tracked paths under `worklog/corrections` must be plain `*.json` files there."""
    errors: list[str] = []
    for path in index_paths:
        if "/" in path[len(CORRECTIONS_DIR_REL) + 1 :] or not path.endswith(".json"):
            errors.append(f"unexpected path under worklog corrections: {path}")
    return errors


def immutability_errors(unmerged_paths: set[str]) -> list[str]:
    """Reject tracked worklog changes in the index and in-place edits of tracked files.

    Untracked files are deliberately out of scope: in a shared working tree an
    unfinished entry belonging to another worker is not part of this commit and
    must not block it. Unmerged paths are reported by their own error.
    """
    errors: list[str] = []
    staged_additions: set[str] = set()
    if run_git("rev-parse", "--verify", "HEAD", check=False) == "":
        staged_additions.update(index_listing())
    else:
        output = run_git("diff", "--cached", "--name-status", "HEAD", "--", *WORKLOG_PATHS)
        for line in output.splitlines():
            if not line:
                continue
            status, *paths = line.split("\t")
            if any(path in unmerged_paths for path in paths):
                continue
            if status == "A":
                staged_additions.update(paths)
                continue
            if status == "D" and all(
                path in (LEGACY_REL, LEGACY_MANIFEST_REL) for path in paths
            ):
                # Sanctioned retirement of the frozen journal pair: the port
                # first verified every entry byte-for-byte, and the exact
                # journal bytes stay pinned by the history ref recorded in
                # worklog/legacy-port-manifest.json. Entry deletions stay
                # rejected; working-tree deletions below stay rejected too.
                continue
            errors.append(
                f"tracked worklog content is immutable: {status} {' -> '.join(paths)}"
            )

    for line in run_git("diff", "--name-status", "--", *WORKLOG_PATHS).splitlines():
        if not line:
            continue
        status, *paths = line.split("\t")
        if any(path in unmerged_paths for path in paths):
            continue
        if len(paths) == 1 and paths[0] in staged_additions:
            errors.append(
                f"{paths[0]}: working-tree copy differs from its staged content; "
                "re-run `git add` to commit the final content"
            )
        else:
            errors.append(
                f"tracked worklog content is immutable: {status} {' -> '.join(paths)} "
                "(restore the working-tree file; correct a committed entry with a new entry)"
            )
    return errors


def _load_legacy_manifest() -> dict[str, Any]:
    try:
        value = json.loads(LEGACY_MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorklogError(f"legacy manifest is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise WorklogError("legacy manifest must be a JSON object")
    if tuple(value) != LEGACY_MANIFEST_FIELDS:
        missing = [field for field in LEGACY_MANIFEST_FIELDS if field not in value]
        extra = [field for field in value if field not in LEGACY_MANIFEST_FIELDS]
        detail: list[str] = []
        if missing:
            detail.append(f"missing {', '.join(missing)}")
        if extra:
            detail.append(f"unknown {', '.join(extra)}")
        if not detail:
            detail.append("fields are out of order")
        raise WorklogError(f"legacy manifest fields invalid: {'; '.join(detail)}")
    return value


def legacy_errors() -> list[str]:
    if not LEGACY_PATH.exists() and not LEGACY_MANIFEST_PATH.exists():
        return []
    if LEGACY_PATH.exists() and not LEGACY_MANIFEST_PATH.exists():
        return ["legacy manifest is missing for WORKLOG-LEGACY.md"]
    if LEGACY_MANIFEST_PATH.exists() and not LEGACY_PATH.exists():
        return ["legacy manifest exists but WORKLOG-LEGACY.md is missing"]

    try:
        manifest = _load_legacy_manifest()
    except (OSError, UnicodeError, WorklogError) as exc:
        return [str(exc)]

    errors: list[str] = []
    if manifest["schema"] != 1:
        errors.append(f"legacy manifest schema must be 1, got {manifest['schema']!r}")
    if manifest["path"] != "WORKLOG-LEGACY.md":
        errors.append(f"legacy manifest path must be WORKLOG-LEGACY.md, got {manifest['path']!r}")
    if not isinstance(manifest["cutoff_commit"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", manifest["cutoff_commit"]
    ):
        errors.append("legacy manifest cutoff_commit must be a full lowercase SHA-1")
    if not isinstance(manifest["sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", manifest["sha256"]
    ):
        errors.append("legacy manifest sha256 must be 64 lowercase hex characters")
    if (
        not isinstance(manifest["bytes"], int)
        or isinstance(manifest["bytes"], bool)
        or manifest["bytes"] < 0
    ):
        errors.append("legacy manifest bytes must be a non-negative integer")
    if (
        not isinstance(manifest["lines"], int)
        or isinstance(manifest["lines"], bool)
        or manifest["lines"] < 0
    ):
        errors.append("legacy manifest lines must be a non-negative integer")
    for field in ("first_heading", "last_heading"):
        if not isinstance(manifest[field], str) or not manifest[field].startswith("## "):
            errors.append(f"legacy manifest {field} must be a level-two heading")
    if errors:
        return errors

    try:
        payload = LEGACY_PATH.read_bytes()
        text = payload.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        return [f"could not read legacy worklog: {exc}"]
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    actual = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "lines": len(payload.splitlines()),
        "first_heading": headings[0] if headings else "",
        "last_heading": headings[-1] if headings else "",
    }
    for field, actual_value in actual.items():
        if manifest[field] != actual_value:
            errors.append(
                f"legacy worklog {field} mismatch: "
                f"manifest={manifest[field]!r} actual={actual_value!r}"
            )
    return errors


def validate_entries(*, enforce_append_only: bool = True) -> list[tuple[Path, dict[str, str], str]]:
    """Validate the worklog content of the commit tree: HEAD plus the Git index.

    Only tracked and staged content is validated. An untracked working-tree file
    is somebody's unfinished entry rather than part of this commit, so it never
    fails the gate; `check_entries` and `render` report those files separately.

    Correction records come from the same commit tree, so an unstaged correction
    cannot exempt a tracked entry either.
    """
    errors: list[str] = []
    unmerged_paths = set(index_listing(unmerged_only=True))
    for path in sorted(unmerged_paths):
        errors.append(f"{path}: unresolved merge conflict in the Git index")

    entry_index_paths = [
        path
        for path in index_listing()
        if path.startswith(f"{ENTRY_DIR_REL}/") and path not in unmerged_paths
    ]
    correction_index_paths = [
        path
        for path in index_listing()
        if path.startswith(f"{CORRECTIONS_DIR_REL}/") and path not in unmerged_paths
    ]
    errors.extend(entry_directory_errors(entry_index_paths))
    errors.extend(correction_directory_errors(correction_index_paths))

    corrections, correction_errors = load_corrections(
        read_index_blobs(correction_index_paths)
    )
    errors.extend(correction_errors)

    entry_blobs = read_index_blobs(entry_index_paths)
    records: list[tuple[str, str]] = []
    for path, payload in entry_blobs.items():
        try:
            records.append((path, payload.decode("utf-8")))
        except UnicodeError as exc:
            errors.append(f"{path}: {exc}")
    bound, binding_errors = bind_corrections(corrections, entry_blobs)
    errors.extend(binding_errors)
    parsed, record_errors = validate_records(records, corrections=bound)
    errors.extend(record_errors)
    errors.extend(supersession_errors(bound, parsed))

    errors.extend(legacy_errors())
    if enforce_append_only:
        errors.extend(immutability_errors(unmerged_paths))
    if errors:
        raise WorklogError("\n".join(errors))

    parsed.sort(key=lambda item: (item[1]["timestamp"], item[0].name))
    return parsed


def validate_unstaged_entries() -> list[tuple[Path, dict[str, str], str]]:
    """Strictly validate entry files that exist only in the working tree."""
    records: list[tuple[str, str]] = []
    errors: list[str] = []
    for path in unstaged_entry_paths():
        try:
            records.append((path, (ROOT / path).read_text(encoding="utf-8")))
        except (OSError, UnicodeError) as exc:
            errors.append(f"{path}: {exc}")
    parsed, record_errors = validate_records(records)
    errors.extend(record_errors)
    if errors:
        raise WorklogError("\n".join(errors))
    return parsed


def report_unstaged_entries() -> None:
    """Name the working-tree entries this commit does not include, without failing."""
    for path in unstaged_entry_paths():
        try:
            fields, _body = parse_worktree_entry(path)
            mismatch = filename_errors(path, fields)
            if mismatch:
                raise WorklogError("; ".join(mismatch))
        except (OSError, UnicodeError, WorklogError) as exc:
            print(
                "note: unstaged entry is not part of this commit and is not valid yet: "
                f"{path}: {exc}",
                file=sys.stderr,
            )
        else:
            print(f"note: unstaged entry is not part of this commit: {path}", file=sys.stderr)


def check_entries(args: argparse.Namespace) -> int:
    parsed = validate_entries(enforce_append_only=not args.allow_modified)
    if args.include_unstaged:
        parsed = sorted(
            [*parsed, *validate_unstaged_entries()],
            key=lambda item: (item[1]["timestamp"], item[0].name),
        )
    print(f"worklog: {len(parsed)} valid entr{'y' if len(parsed) == 1 else 'ies'}")
    if not args.include_unstaged:
        report_unstaged_entries()
    return 0


def _retired_legacy_ref() -> str:
    """The `git show <commit>:<path>` ref for the retired journal, if recorded."""
    manifest = ROOT / "worklog" / "legacy-port-manifest.json"
    if not manifest.exists():
        return ""
    try:
        history = json.loads(manifest.read_text(encoding="utf-8")).get("source_history") or {}
    except (OSError, json.JSONDecodeError):
        return ""
    commit, path = history.get("commit"), history.get("path")
    return f"{commit}:{path}" if commit and path else ""


def rendered_worklog(
    parsed: list[tuple[Path, dict[str, str], str]],
    *,
    include_legacy: bool,
) -> str:
    lines = [
        "# WORKLOG (local generated view)",
        "",
        "> Generated from immutable tracked entries in `worklog/entries/`.",
        "> Do not edit this file; run `python3 scripts/worklog.py render`.",
        "> Pre-Worklog2 history is frozen in `WORKLOG-LEGACY.md`.",
        "",
    ]
    if include_legacy and LEGACY_PATH.exists():
        lines.extend(
            [
                "## Frozen legacy journal",
                "",
                LEGACY_PATH.read_text(encoding="utf-8").rstrip(),
            ]
        )
        lines.extend(["", "---", "", "## Worklog2 entries", ""])
    if not parsed:
        lines.extend(["No Worklog2 entries yet.", ""])
        return "\n".join(lines)

    for index, (path, _fields, text) in enumerate(parsed):
        if index:
            lines.extend(["", "---", ""])
        lines.extend([f"<!-- source: {path.relative_to(ROOT)} -->", "", text.rstrip()])
    lines.append("")
    return "\n".join(lines)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    try:
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def unstaged_entry_records() -> list[tuple[Path, dict[str, str], str]]:
    """Parse unstaged working-tree entries for the local rendered view, skipping invalid ones."""
    records: list[tuple[Path, dict[str, str], str]] = []
    for path in unstaged_entry_paths():
        try:
            text = (ROOT / path).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            print(f"note: unstaged entry not rendered: {path}: {exc}", file=sys.stderr)
            continue
        parsed, errors = validate_records([(path, text)])
        if errors:
            for error in errors:
                print(f"note: unstaged entry not rendered: {error}", file=sys.stderr)
            continue
        records.extend(parsed)
    return records


def render(args: argparse.Namespace) -> int:
    parsed = sorted(
        [*validate_entries(enforce_append_only=False), *unstaged_entry_records()],
        key=lambda item: (item[1]["timestamp"], item[0].name),
    )
    if args.include_legacy and not LEGACY_PATH.exists():
        note = "note: the legacy journal is retired from the tree"
        ref = _retired_legacy_ref()
        if ref:
            note += f"; original bytes: git show {ref}"
        print(note, file=sys.stderr)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    if output.resolve() == TRACKED_NAVIGATION_PATH.resolve():
        raise WorklogError("refusing to overwrite tracked WORKLOG.md navigation page")
    atomic_write(output, rendered_worklog(parsed, include_legacy=bool(args.include_legacy)))
    try:
        display_path = output.relative_to(ROOT)
    except ValueError:
        display_path = output
    print(
        f"worklog: rendered {len(parsed)} "
        f"entr{'y' if len(parsed) == 1 else 'ies'} to {display_path}"
    )
    return 0


def common_hooks_dir() -> Path:
    common = run_git("rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(common) / "hooks"


def install_hook(_args: argparse.Namespace) -> int:
    configured = run_git("config", "--get", "core.hooksPath", check=False)
    if configured:
        raise WorklogError(
            f"core.hooksPath is already configured as {configured!r}; refusing to install outside "
            "the standard common hooks directory"
        )
    hooks_dir = common_hooks_dir()
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hooks_dir / "pre-commit"
    managed = MANAGED_HOOK.encode("utf-8")
    if hook_path.exists():
        if hook_path.read_bytes() == managed:
            print(f"worklog: managed pre-commit hook already installed at {hook_path}")
            return 0
        raise WorklogError(f"refusing to overwrite existing unrelated pre-commit hook: {hook_path}")

    try:
        fd = os.open(hook_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o755)
    except FileExistsError as exc:
        raise WorklogError(f"pre-commit hook appeared concurrently: {hook_path}") from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(managed)
    hook_path.chmod(0o755)
    print(f"worklog: installed managed pre-commit hook at {hook_path}")
    print("worklog: hook executes checked-out repository code; use only in trusted worktrees")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    new_parser = subparsers.add_parser("new", help="create a unique worklog entry template")
    new_parser.add_argument("--title", required=True, help="human-readable entry title")
    new_parser.add_argument("--topic", help="lowercase topic slug; defaults to title")
    new_parser.add_argument(
        "--worker",
        help="worker/lane ID; defaults to WORKLOG_WORKER or git user.name",
    )
    new_parser.add_argument("--status", choices=sorted(ALLOWED_STATUSES), default="completed")
    new_parser.set_defaults(func=create_entry)

    check_parser = subparsers.add_parser(
        "check", help="validate the staged commit tree and immutable history"
    )
    check_parser.add_argument(
        "--allow-modified",
        action="store_true",
        help="validate entry/legacy format without rejecting tracked modifications",
    )
    check_parser.add_argument(
        "--include-unstaged",
        action="store_true",
        help="also validate entries that exist only in the working tree",
    )
    check_parser.set_defaults(func=check_entries)

    render_parser = subparsers.add_parser("render", help="render the ignored local worklog view")
    render_parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="output path")
    render_parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="prepend the frozen legacy journal to the local generated view",
    )
    render_parser.set_defaults(func=render)

    hook_parser = subparsers.add_parser(
        "install-hook",
        help="install the non-destructive pre-commit-only worklog checker",
    )
    hook_parser.set_defaults(func=install_hook)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except WorklogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
