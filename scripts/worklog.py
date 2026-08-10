#!/usr/bin/env python3
"""Create, validate, and render contention-free hipEngine worklog entries.

Tracked current history lives in immutable Markdown files under
``worklog/entries``. The pre-Worklog2 journal is frozen at
``WORKLOG-LEGACY.md`` after activation. ``WORKLOG.md`` remains a tracked
navigation page; generated local views go under the ignored ``.worklog``
directory.
"""

from __future__ import annotations

import argparse
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
LEGACY_PATH = ROOT / "WORKLOG-LEGACY.md"
LEGACY_MANIFEST_PATH = ROOT / "worklog" / "legacy-manifest.json"
TRACKED_NAVIGATION_PATH = ROOT / "WORKLOG.md"
DEFAULT_OUTPUT = ROOT / ".worklog" / "WORKLOG.md"
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


def slugify(value: str, *, fallback: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value[:48] or fallback


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


def _markdown_heading_positions(body: str) -> tuple[list[int], dict[str, list[int]]]:
    """Return title and required-section line positions outside fenced code."""
    title_positions: list[int] = []
    section_positions = {section: [] for section in REQUIRED_SECTIONS}
    fence: str | None = None
    for index, line in enumerate(body.splitlines()):
        stripped = line.lstrip()
        marker = (
            "```"
            if stripped.startswith("```")
            else "~~~"
            if stripped.startswith("~~~")
            else None
        )
        if marker is not None:
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is not None:
            continue
        if line.startswith("# "):
            title_positions.append(index)
        if line in section_positions:
            section_positions[line].append(index)
    return title_positions, section_positions


def parse_entry(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
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
        raise WorklogError("base_commit must be a full 40-character lowercase commit hash")
    if slugify(fields["topic"], fallback="") != fields["topic"]:
        raise WorklogError("topic must be a lowercase kebab-case slug")

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
            raise WorklogError(
                f"expected exactly one {section!r} heading, found {len(section_lines)}"
            )
        positions.append(section_lines[0])
    if positions != sorted(positions):
        raise WorklogError("required sections are out of order")
    for index, section in enumerate(REQUIRED_SECTIONS):
        content_start = positions[index] + 1
        content_end = positions[index + 1] if index + 1 < len(positions) else len(body_lines)
        if not "\n".join(body_lines[content_start:content_end]).strip():
            raise WorklogError(f"{section!r} section is empty")
    for placeholder in PLACEHOLDERS:
        if placeholder in body:
            raise WorklogError(f"unfinished placeholder remains: {placeholder}")
    if any(marker in text for marker in ("<<<<<<<", "=======", ">>>>>>>")):
        raise WorklogError("git conflict marker present")

    return fields, text.rstrip() + "\n"


def entry_paths() -> list[Path]:
    if not ENTRY_DIR.exists():
        return []
    return sorted(path for path in ENTRY_DIR.glob("*.md") if path.is_file())


def entry_directory_errors() -> list[str]:
    if not ENTRY_DIR.exists():
        return []
    errors: list[str] = []
    for path in sorted(ENTRY_DIR.iterdir()):
        if not path.is_file() or path.suffix != ".md":
            errors.append(f"unexpected path under worklog entries: {path.relative_to(ROOT)}")
    return errors


def append_only_errors() -> list[str]:
    if run_git("rev-parse", "--verify", "HEAD", check=False) == "":
        return []

    errors: list[str] = []
    entry_dir = str(ENTRY_DIR.relative_to(ROOT))
    output = run_git("diff", "--name-status", "HEAD", "--", entry_dir)
    for line in output.splitlines():
        if not line:
            continue
        status, *paths = line.split("\t")
        if status != "A":
            rendered_paths = " -> ".join(paths)
            errors.append(f"tracked worklog entries are immutable: {status} {rendered_paths}")

    for path in run_git("diff", "--name-only", "--", entry_dir).splitlines():
        if path:
            errors.append(f"worklog entry differs from its staged content: {path}")

    frozen_paths = [
        str(LEGACY_PATH.relative_to(ROOT)),
        str(LEGACY_MANIFEST_PATH.relative_to(ROOT)),
    ]
    frozen_output = run_git("diff", "--name-status", "HEAD", "--", *frozen_paths)
    for line in frozen_output.splitlines():
        if not line:
            continue
        status, *paths = line.split("\t")
        if status != "A":
            errors.append(
                f"frozen legacy worklog state is immutable: {status} {' -> '.join(paths)}"
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
    errors = entry_directory_errors()
    parsed: list[tuple[Path, dict[str, str], str]] = []

    for path in entry_paths():
        if not FILENAME_RE.fullmatch(path.name):
            errors.append(f"{path.relative_to(ROOT)}: invalid filename")
            continue
        try:
            fields, text = parse_entry(path)
        except (OSError, UnicodeError, WorklogError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
            continue
        parsed_timestamp = datetime.fromisoformat(
            fields["timestamp"].replace("Z", "+00:00")
        )
        expected_stamp = parsed_timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
        expected_worker = slugify(fields["worker"], fallback="worker")
        expected_prefix = f"{expected_stamp}-{expected_worker}-{fields['topic']}-"
        if not path.name.startswith(expected_prefix):
            errors.append(f"{path.relative_to(ROOT)}: filename does not match entry metadata")
            continue
        parsed.append((path, fields, text))

    errors.extend(legacy_errors())
    if enforce_append_only:
        errors.extend(append_only_errors())
    if errors:
        raise WorklogError("\n".join(errors))

    parsed.sort(key=lambda item: (item[1]["timestamp"], item[0].name))
    return parsed


def check_entries(args: argparse.Namespace) -> int:
    parsed = validate_entries(enforce_append_only=not args.allow_modified)
    print(f"worklog: {len(parsed)} valid entr{'y' if len(parsed) == 1 else 'ies'}")
    return 0


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


def render(args: argparse.Namespace) -> int:
    parsed = validate_entries(enforce_append_only=False)
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

    check_parser = subparsers.add_parser("check", help="validate entries and immutable history")
    check_parser.add_argument(
        "--allow-modified",
        action="store_true",
        help="validate entry/legacy format without rejecting tracked modifications",
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
