from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_TOOL = _ROOT / "scripts" / "worklog.py"


def _run(
    repo: Path,
    *args: str,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [*args],
        cwd=repo,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"command failed ({result.returncode}): {' '.join(args)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(repo, "git", *args, check=check)


def _tool(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(repo, sys.executable, "scripts/worklog.py", *args, check=check)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(_SOURCE_TOOL, repo / "scripts" / "worklog.py")
    (repo / "WORKLOG.md").write_text("# Worklog navigation\n", encoding="utf-8")
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Worklog Test")
    _git(repo, "config", "user.email", "worklog@example.invalid")
    _git(repo, "add", "WORKLOG.md", "scripts/worklog.py")
    _git(repo, "commit", "-m", "test: initialize worklog repository")
    return repo


def _new_entry(
    repo: Path,
    *,
    title: str = "Record test outcome",
    topic: str = "test-outcome",
    worker: str = "test-lane",
    status: str = "completed",
) -> Path:
    result = _tool(
        repo,
        "new",
        "--title",
        title,
        "--topic",
        topic,
        "--worker",
        worker,
        "--status",
        status,
    )
    path = repo / result.stdout.strip()
    text = path.read_text(encoding="utf-8")
    text = text.replace("<required>", "Recorded exact test evidence.")
    text = text.replace("<none-or-required-follow-up>", "No follow-up.")
    path.write_text(text, encoding="utf-8")
    return path


def _unfinished_entry(
    repo: Path,
    *,
    title: str = "Work in progress",
    topic: str = "work-in-progress",
    worker: str = "other-lane",
) -> Path:
    """Mint an entry template whose placeholders are still unfilled (working-tree WIP)."""
    result = _tool(repo, "new", "--title", title, "--topic", topic, "--worker", worker)
    return repo / result.stdout.strip()


def _commit_entry(repo: Path, entry: Path, message: str = "docs: add worklog entry") -> None:
    _git(repo, "add", str(entry.relative_to(repo)))
    _git(repo, "commit", "-m", message)


def _legacy_manifest(repo: Path, *, cutoff_commit: str | None = None) -> Path:
    legacy = repo / "WORKLOG-LEGACY.md"
    payload = legacy.read_bytes()
    headings = [
        line
        for line in payload.decode("utf-8").splitlines()
        if line.startswith("## ")
    ]
    manifest = {
        "schema": 1,
        "path": "WORKLOG-LEGACY.md",
        "cutoff_commit": cutoff_commit or _git(repo, "rev-parse", "HEAD").stdout.strip(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "lines": len(payload.splitlines()),
        "first_heading": headings[0],
        "last_heading": headings[-1],
    }
    path = repo / "worklog" / "legacy-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def test_new_check_and_render_keep_root_navigation_tracked(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    entry = _new_entry(repo)

    assert entry.parent == repo / "worklog" / "entries"
    _git(repo, "add", str(entry.relative_to(repo)))
    assert _tool(repo, "check").stdout.strip() == "worklog: 1 valid entry"

    root_before = (repo / "WORKLOG.md").read_bytes()
    result = _tool(repo, "render")
    output = repo / ".worklog" / "WORKLOG.md"
    assert result.stdout.strip() == "worklog: rendered 1 entry to .worklog/WORKLOG.md"
    assert output.is_file()
    assert "Generated from immutable tracked entries" in output.read_text(encoding="utf-8")
    assert str(entry.relative_to(repo)) in output.read_text(encoding="utf-8")
    assert (repo / "WORKLOG.md").read_bytes() == root_before


def test_rapid_new_calls_allocate_unique_paths(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    first = _new_entry(repo, title="First", topic="first")
    second = _new_entry(repo, title="Second", topic="second")

    assert first != second
    assert first.is_file()
    assert second.is_file()
    _git(repo, "add", str(first.relative_to(repo)), str(second.relative_to(repo)))
    assert _tool(repo, "check").stdout.strip() == "worklog: 2 valid entries"


def test_fenced_shell_comments_and_heading_text_are_not_markdown_headings(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    entry = _new_entry(repo)
    text = entry.read_text(encoding="utf-8").replace(
        "Recorded exact test evidence.",
        """Recorded exact test evidence.

```bash
# This is a shell comment, not a second title.
printf '## Summary\\n'
```
""",
        1,
    )
    entry.write_text(text, encoding="utf-8")
    _git(repo, "add", str(entry.relative_to(repo)))

    assert _tool(repo, "check").returncode == 0


def test_render_refuses_to_overwrite_tracked_navigation(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _new_entry(repo)

    result = _tool(repo, "render", "--output", "WORKLOG.md", check=False)
    assert result.returncode == 1
    assert "refusing to overwrite tracked WORKLOG.md" in result.stderr
    assert (repo / "WORKLOG.md").read_text(encoding="utf-8") == "# Worklog navigation\n"


def test_render_orders_equal_timestamps_by_filename(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    first = _new_entry(repo, title="First", topic="first", worker="lane-a")
    second = _new_entry(repo, title="Second", topic="second", worker="lane-b")

    first_text = first.read_text(encoding="utf-8")
    second_text = second.read_text(encoding="utf-8")
    first_stamp = next(
        line.removeprefix("timestamp: ")
        for line in first_text.splitlines()
        if line.startswith("timestamp: ")
    )
    second_stamp = next(
        line.removeprefix("timestamp: ")
        for line in second_text.splitlines()
        if line.startswith("timestamp: ")
    )
    second.write_text(second_text.replace(second_stamp, first_stamp), encoding="utf-8")
    second_stamp_name = second.name.split("-", 1)[0]
    first_stamp_name = first.name.split("-", 1)[0]
    renamed = second.rename(
        second.with_name(second.name.replace(second_stamp_name, first_stamp_name, 1))
    )
    _git(repo, "add", str(first.relative_to(repo)), str(renamed.relative_to(repo)))

    _tool(repo, "check")
    _tool(repo, "render")
    rendered = (repo / ".worklog" / "WORKLOG.md").read_text(encoding="utf-8")
    ordered_names = sorted(path.name for path in (repo / "worklog" / "entries").glob("*.md"))
    assert rendered.index(ordered_names[0]) < rendered.index(ordered_names[1])


def test_default_render_excludes_legacy_and_full_render_includes_it(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _new_entry(repo)
    legacy_text = "# Legacy journal\n\n## 2026-08-10 - Legacy sentinel\n\n- Exact old text.\n"
    (repo / "WORKLOG-LEGACY.md").write_text(legacy_text, encoding="utf-8")
    _legacy_manifest(repo)

    _tool(repo, "check")
    _tool(repo, "render")
    default = (repo / ".worklog" / "WORKLOG.md").read_text(encoding="utf-8")
    assert "Legacy sentinel" not in default
    assert "WORKLOG-LEGACY.md" in default

    _tool(repo, "render", "--include-legacy")
    complete = (repo / ".worklog" / "WORKLOG.md").read_text(encoding="utf-8")
    assert legacy_text in complete
    assert complete.index("Legacy sentinel") < complete.index("Record test outcome")


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda text: text.replace("schema: 1\n", ""), "missing frontmatter"),
        (lambda text: text.replace("schema: 1", "schema: 1\nextra: no"), "unknown frontmatter"),
        (
            lambda text: text.replace(
                "schema: 1\ntimestamp:", "timestamp:", 1
            ).replace("worker:", "schema: 1\nworker:", 1),
            "out of order",
        ),
        (lambda text: text.replace("schema: 1", "schema: 2"), "unsupported schema"),
        (
            lambda text: text.replace("2026", "not-a-year", 1),
            "timestamp",
        ),
        (lambda text: text.replace("base_commit: ", "base_commit: bad", 1), "base_commit"),
        (lambda text: text.replace("status: completed", "status: running"), "unsupported status"),
        (lambda text: text.replace("topic: test-outcome", "topic: Test Outcome"), "topic"),
        (lambda text: text.replace("## Validation", "## Summary"), "## Summary"),
        (
            lambda text: text.replace(
                "## Validation\n\n- Recorded exact test evidence.", ""
            ),
            "## Validation",
        ),
        (
            lambda text: text.replace(
                "## Changes\n\n- Recorded exact test evidence.", "## Changes"
            ),
            "empty",
        ),
        (
            lambda text: text.replace("## Changes", "## TEMP", 1)
            .replace("## Validation", "## Changes", 1)
            .replace("## TEMP", "## Validation", 1),
            "out of order",
        ),
        (
            lambda text: text.replace(
                "Recorded exact test evidence.", "<required>", 1
            ),
            "placeholder",
        ),
        (lambda text: text + "<<<<<<< conflict\n", "conflict marker"),
    ],
)
def test_check_rejects_malformed_staged_entries(tmp_path: Path, mutation, expected: str) -> None:
    repo = _init_repo(tmp_path)
    entry = _new_entry(repo)
    entry.write_text(mutation(entry.read_text(encoding="utf-8")), encoding="utf-8")
    _git(repo, "add", str(entry.relative_to(repo)))

    result = _tool(repo, "check", check=False)
    assert result.returncode == 1
    assert expected in result.stderr


@pytest.mark.parametrize("change", ["modify", "delete", "rename"])
def test_check_rejects_changes_to_committed_entries(tmp_path: Path, change: str) -> None:
    repo = _init_repo(tmp_path)
    entry = _new_entry(repo)
    _commit_entry(repo, entry)

    if change == "modify":
        entry.write_text(entry.read_text(encoding="utf-8") + "\nChanged.\n", encoding="utf-8")
    elif change == "delete":
        entry.unlink()
    else:
        entry.rename(entry.with_name(entry.name.replace("test-outcome", "renamed", 1)))

    result = _tool(repo, "check", check=False)
    assert result.returncode == 1
    assert "immutable" in result.stderr


def test_check_rejects_staged_worktree_divergence(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    entry = _new_entry(repo)
    _git(repo, "add", str(entry.relative_to(repo)))
    entry.write_text(
        entry.read_text(encoding="utf-8") + "\nChanged after staging.\n",
        encoding="utf-8",
    )

    result = _tool(repo, "check", check=False)
    assert result.returncode == 1
    assert "differs from its staged content" in result.stderr


def test_check_reports_unmerged_entry_conflict(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    entry = _new_entry(repo)
    _commit_entry(repo, entry)
    _git(repo, "switch", "-c", "lane")
    entry.write_text(entry.read_text(encoding="utf-8") + "\nLane text.\n", encoding="utf-8")
    _git(repo, "commit", "-am", "docs: lane edit")
    _git(repo, "switch", "main")
    entry.write_text(entry.read_text(encoding="utf-8") + "\nMain text.\n", encoding="utf-8")
    _git(repo, "commit", "-am", "docs: main edit")
    merge = _git(repo, "merge", "--no-edit", "lane", check=False)
    assert merge.returncode != 0

    result = _tool(repo, "check", check=False)
    assert result.returncode == 1
    assert "unresolved merge conflict in the Git index" in result.stderr
    assert "immutable" not in result.stderr


def test_check_ignores_unfinished_unstaged_entries(tmp_path: Path) -> None:
    """Another worker's unfinished entry is not part of this commit and cannot block it."""
    repo = _init_repo(tmp_path)
    mine = _new_entry(repo, title="Mine", topic="mine")
    _git(repo, "add", str(mine.relative_to(repo)))
    wip = _unfinished_entry(repo)
    (repo / "worklog" / "entries" / "scratch.txt").write_text("wip\n", encoding="utf-8")

    result = _tool(repo, "check")
    assert result.stdout.strip() == "worklog: 1 valid entry"
    assert "not part of this commit and is not valid yet" in result.stderr
    assert wip.name in result.stderr

    strict = _tool(repo, "check", "--include-unstaged", check=False)
    assert strict.returncode == 1
    assert "placeholder" in strict.stderr


def test_check_rejects_staged_unfinished_entry(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    wip = _unfinished_entry(repo)
    _git(repo, "add", str(wip.relative_to(repo)))

    result = _tool(repo, "check", check=False)
    assert result.returncode == 1
    assert "placeholder" in result.stderr


def test_render_includes_unstaged_entries_and_reports_invalid_ones(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    committed = _new_entry(repo, title="Committed outcome", topic="committed-outcome")
    _commit_entry(repo, committed)
    unstaged = _new_entry(repo, title="Unstaged outcome", topic="unstaged-outcome")
    _unfinished_entry(repo, title="Unfinished outcome", topic="unfinished-outcome")

    result = _tool(repo, "render")
    rendered = (repo / ".worklog" / "WORKLOG.md").read_text(encoding="utf-8")
    assert "Committed outcome" in rendered
    assert "Unstaged outcome" in rendered
    assert unstaged.name in rendered
    assert "Unfinished outcome" not in rendered
    assert "not rendered" in result.stderr


def test_independent_branch_entries_merge_without_conflict(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _git(repo, "switch", "-c", "lane-a")
    lane_a = _new_entry(repo, title="Lane A result", topic="lane-a", worker="lane-a")
    _commit_entry(repo, lane_a, "docs: record lane A")

    _git(repo, "switch", "main")
    lane_b = _new_entry(repo, title="Lane B result", topic="lane-b", worker="lane-b")
    _commit_entry(repo, lane_b, "docs: record lane B")

    merge = _git(repo, "merge", "--no-edit", "lane-a", check=False)
    assert merge.returncode == 0, merge.stderr
    assert len(list((repo / "worklog" / "entries").glob("*.md"))) == 2
    assert _tool(repo, "check").returncode == 0


def test_legacy_manifest_rejects_mutation_and_missing_manifest(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    legacy = repo / "WORKLOG-LEGACY.md"
    legacy.write_text("# Legacy\n\n## 2026-08-10 - Frozen\n\n- Evidence.\n", encoding="utf-8")
    manifest = _legacy_manifest(repo)
    _git(repo, "add", "WORKLOG-LEGACY.md", "worklog/legacy-manifest.json")
    _git(repo, "commit", "-m", "docs: freeze legacy worklog")
    _tool(repo, "check")

    frozen_payload = legacy.read_bytes()
    legacy.write_text(legacy.read_text(encoding="utf-8") + "mutation\n", encoding="utf-8")
    mismatch = _tool(repo, "check", check=False)
    assert mismatch.returncode == 1
    assert "legacy" in mismatch.stderr.lower()

    legacy.write_bytes(frozen_payload)
    manifest.unlink()
    missing = _tool(repo, "check", check=False)
    assert missing.returncode == 1
    assert "legacy manifest" in missing.stderr.lower()


def test_check_allows_sanctioned_retirement_of_the_frozen_journal(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    entry = _new_entry(repo)
    _commit_entry(repo, entry)
    legacy = repo / "WORKLOG-LEGACY.md"
    legacy.write_text("# Legacy\n\n## 2026-08-10 - Frozen\n\n- Evidence.\n", encoding="utf-8")
    _legacy_manifest(repo)
    _git(repo, "add", "WORKLOG-LEGACY.md", "worklog/legacy-manifest.json")
    _git(repo, "commit", "-m", "docs: freeze legacy worklog")

    _git(repo, "rm", "-q", "WORKLOG-LEGACY.md", "worklog/legacy-manifest.json")
    retired = _tool(repo, "check")
    assert retired.returncode == 0

    tracked_entry = next((repo / "worklog" / "entries").glob("*.md"))
    _git(repo, "rm", "-q", str(tracked_entry.relative_to(repo)))
    still_immutable = _tool(repo, "check", check=False)
    assert still_immutable.returncode == 1
    assert "immutable" in still_immutable.stderr.lower()


def test_install_hook_preserves_lfs_hooks_and_refuses_unrelated_precommit(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    hooks = repo / ".git" / "hooks"
    lfs_names = ("post-checkout", "post-commit", "post-merge", "pre-push")
    before: dict[str, bytes] = {}
    for name in lfs_names:
        payload = f"#!/bin/sh\n# fake LFS {name}\n".encode()
        path = hooks / name
        path.write_bytes(payload)
        path.chmod(0o755)
        before[name] = payload

    installed = _tool(repo, "install-hook")
    pre_commit = hooks / "pre-commit"
    assert "installed" in installed.stdout
    assert pre_commit.is_file()
    assert pre_commit.stat().st_mode & 0o111
    assert _git(repo, "config", "--get", "core.hooksPath", check=False).stdout == ""
    assert {name: (hooks / name).read_bytes() for name in lfs_names} == before

    assert "already installed" in _tool(repo, "install-hook").stdout
    pre_commit.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
    refused = _tool(repo, "install-hook", check=False)
    assert refused.returncode == 1
    assert "refusing to overwrite" in refused.stderr
    assert {name: (hooks / name).read_bytes() for name in lfs_names} == before


def test_managed_hook_validates_staged_entries_and_tolerates_branch_without_tool(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    _tool(repo, "install-hook")
    hook = repo / ".git" / "hooks" / "pre-commit"

    _unfinished_entry(repo)
    assert _run(repo, str(hook), check=False).returncode == 0

    entry = _new_entry(repo)
    entry.write_text(entry.read_text(encoding="utf-8") + "<<<<<<< bad\n", encoding="utf-8")
    _git(repo, "add", str(entry.relative_to(repo)))
    blocked = _run(repo, str(hook), check=False)
    assert blocked.returncode == 1
    assert "conflict marker" in blocked.stderr

    tool = repo / "scripts" / "worklog.py"
    hidden = repo / "scripts" / "worklog.py.hidden"
    tool.rename(hidden)
    try:
        result = _run(repo, str(hook), check=False)
    finally:
        hidden.rename(tool)
    assert result.returncode == 0


def test_commit_succeeds_with_unfinished_unstaged_entry_present(tmp_path: Path) -> None:
    """The installed hook must not force `git commit --no-verify` on a shared worktree."""
    repo = _init_repo(tmp_path)
    _tool(repo, "install-hook")
    wip = _unfinished_entry(repo)
    entry = _new_entry(repo, title="Mine", topic="mine")
    _git(repo, "add", str(entry.relative_to(repo)))

    commit = _git(repo, "commit", "-m", "docs: record my unit", check=False)
    assert commit.returncode == 0, commit.stderr
    assert wip.is_file()
    assert _git(repo, "ls-files", "--others", "--exclude-standard").stdout.strip() == str(
        wip.relative_to(repo)
    )


def _set_base_commit(entry: Path, value: str) -> None:
    text = entry.read_text(encoding="utf-8")
    entry.write_text(
        re.sub(r"(?m)^base_commit: .*$", f"base_commit: {value}", text, count=1),
        encoding="utf-8",
    )


def _resolved_base_commit(repo: Path, entry: Path) -> str:
    """Resolve an entry's recorded base_commit, possibly abbreviated, to its full hash."""
    for line in entry.read_text(encoding="utf-8").splitlines():
        if line.startswith("base_commit: "):
            recorded = line.removeprefix("base_commit: ")
            return _git(repo, "rev-parse", f"{recorded}^{{commit}}").stdout.strip()
    raise AssertionError(f"{entry} has no base_commit field")


def _historical_abbreviated_entry(repo: Path) -> Path:
    """A committed entry whose base_commit is a real abbreviation, as history left it."""
    entry = _new_entry(repo, title="Historical outcome", topic="historical-outcome")
    _set_base_commit(entry, _git(repo, "rev-parse", "--short=9", "HEAD").stdout.strip())
    _commit_entry(repo, entry, "docs: record historical unit")
    return entry


def _historical_entry_without_next(repo: Path) -> Path:
    entry = _new_entry(repo, title="Historical evidence", topic="historical-evidence")
    text = entry.read_text(encoding="utf-8").replace(
        "## Next", "## Remaining observations", 1
    )
    entry.write_text(text, encoding="utf-8")
    _commit_entry(repo, entry, "docs: record historical evidence")
    return entry


def _superseding_entry(repo: Path, corrected: Path) -> Path:
    """A new decision entry that names the historical entry it corrects."""
    entry = _new_entry(
        repo,
        title="Correct historical evidence headings",
        topic="correct-historical-evidence",
        worker="worklog-compat",
        status="decision",
    )
    text = entry.read_text(encoding="utf-8").replace(
        "## Summary\n\nRecorded exact test evidence.",
        f"## Summary\n\nSupersedes `{corrected.relative_to(repo)}`.",
        1,
    )
    entry.write_text(text, encoding="utf-8")
    return entry


def _correction_payload(
    repo: Path,
    entry: Path,
    *,
    defects: list[str],
    base_commit: str | None = None,
    superseded_by: str | None = None,
    sha256: str | None = None,
) -> dict[str, object]:
    return {
        "schema": 1,
        "path": str(entry.relative_to(repo)),
        "sha256": sha256 or hashlib.sha256(entry.read_bytes()).hexdigest(),
        "defects": list(defects),
        "reason": "historical entry predates the current schema rule",
        "base_commit": base_commit,
        "superseded_by": superseded_by,
    }


def _write_correction(
    repo: Path,
    entry: Path,
    payload: dict[str, object],
    *,
    name: str | None = None,
    raw: str | None = None,
) -> Path:
    path = repo / "worklog" / "corrections" / (name or f"{entry.stem}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _stage(repo: Path, *paths: Path) -> None:
    _git(repo, "add", *[str(path.relative_to(repo)) for path in paths])


def test_hash_bound_correction_amends_a_historical_base_commit(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    entry = _historical_abbreviated_entry(repo)
    rejected = _tool(repo, "check", check=False)
    assert rejected.returncode == 1
    assert "base_commit must be a full 40-character" in rejected.stderr

    full_commit = _resolved_base_commit(repo, entry)
    correction = _write_correction(
        repo,
        entry,
        _correction_payload(
            repo, entry, defects=["abbreviated_base_commit"], base_commit=full_commit
        ),
    )
    _stage(repo, correction)

    assert _tool(repo, "check").stdout.strip() == "worklog: 1 valid entry"
    _tool(repo, "render")
    rendered = (repo / ".worklog" / "WORKLOG.md").read_text(encoding="utf-8")
    assert "Historical outcome" in rendered
    assert str(entry.relative_to(repo)) in rendered


def test_unstaged_correction_does_not_exempt_a_tracked_entry(tmp_path: Path) -> None:
    """Corrections are read from the commit tree, not from an untracked working file."""
    repo = _init_repo(tmp_path)
    entry = _historical_abbreviated_entry(repo)
    _write_correction(
        repo,
        entry,
        _correction_payload(
            repo,
            entry,
            defects=["abbreviated_base_commit"],
            base_commit=_resolved_base_commit(repo, entry),
        ),
    )

    result = _tool(repo, "check", check=False)
    assert result.returncode == 1
    assert "base_commit must be a full 40-character" in result.stderr


def test_structural_correction_is_carried_by_a_linked_new_entry(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    entry = _historical_entry_without_next(repo)
    superseding = _superseding_entry(repo, entry)
    correction = _write_correction(
        repo,
        entry,
        _correction_payload(
            repo,
            entry,
            defects=["missing_required_sections"],
            superseded_by=str(superseding.relative_to(repo)),
        ),
    )
    _stage(repo, correction, superseding)

    assert _tool(repo, "check").stdout.strip() == "worklog: 2 valid entries"
    _tool(repo, "render")
    rendered = (repo / ".worklog" / "WORKLOG.md").read_text(encoding="utf-8")
    assert "Historical evidence" in rendered
    assert "Correct historical evidence headings" in rendered


def test_structural_correction_requires_a_valid_superseding_entry(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    entry = _historical_entry_without_next(repo)

    missing = _write_correction(
        repo,
        entry,
        _correction_payload(
            repo,
            entry,
            defects=["missing_required_sections"],
            superseded_by="worklog/entries/20990101T000000.000000Z-nobody-absent-topic-abcdef.md",
        ),
    )
    _stage(repo, missing)
    absent = _tool(repo, "check", check=False)
    assert absent.returncode == 1
    assert "superseded_by is not a valid entry in the commit tree" in absent.stderr

    unlinked = _new_entry(
        repo,
        title="Unlinked correction",
        topic="unlinked-correction",
        status="decision",
    )
    _write_correction(
        repo,
        entry,
        _correction_payload(
            repo,
            entry,
            defects=["missing_required_sections"],
            superseded_by=str(unlinked.relative_to(repo)),
        ),
    )
    _stage(repo, unlinked, missing)
    unnamed = _tool(repo, "check", check=False)
    assert unnamed.returncode == 1
    assert "must name the corrected path" in unnamed.stderr

    wrong_status = _new_entry(
        repo,
        title="Completed correction",
        topic="completed-correction",
        status="completed",
    )
    wrong_status.write_text(
        wrong_status.read_text(encoding="utf-8").replace(
            "## Summary\n\nRecorded exact test evidence.",
            f"## Summary\n\nSupersedes `{entry.relative_to(repo)}`.",
            1,
        ),
        encoding="utf-8",
    )
    _write_correction(
        repo,
        entry,
        _correction_payload(
            repo,
            entry,
            defects=["missing_required_sections"],
            superseded_by=str(wrong_status.relative_to(repo)),
        ),
    )
    _stage(repo, wrong_status, missing)
    status_error = _tool(repo, "check", check=False)
    assert status_error.returncode == 1
    assert "must have status checkpoint or decision" in status_error.stderr


@pytest.mark.parametrize("change", ["modify", "delete", "rename"])
def test_committed_correction_records_are_immutable(tmp_path: Path, change: str) -> None:
    repo = _init_repo(tmp_path)
    entry = _historical_abbreviated_entry(repo)
    correction = _write_correction(
        repo,
        entry,
        _correction_payload(
            repo,
            entry,
            defects=["abbreviated_base_commit"],
            base_commit=_resolved_base_commit(repo, entry),
        ),
    )
    _stage(repo, correction)
    _git(repo, "commit", "-m", "docs: bind historical entry correction")
    assert _tool(repo, "check").returncode == 0

    if change == "modify":
        correction.write_text(
            correction.read_text(encoding="utf-8").replace("predates", "antedates"),
            encoding="utf-8",
        )
    elif change == "delete":
        correction.unlink()
    else:
        correction.rename(correction.with_name("renamed.json"))

    result = _tool(repo, "check", check=False)
    assert result.returncode == 1
    assert "immutable" in result.stderr


def test_correction_does_not_weaken_new_entry_schema(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    historical = _historical_abbreviated_entry(repo)
    _stage(
        repo,
        _write_correction(
            repo,
            historical,
            _correction_payload(
                repo,
                historical,
                defects=["abbreviated_base_commit"],
                base_commit=_resolved_base_commit(repo, historical),
            ),
        ),
    )
    assert _tool(repo, "check").returncode == 0

    new_entry = _new_entry(repo, title="New outcome", topic="new-outcome")
    _set_base_commit(new_entry, "0123456")
    _stage(repo, new_entry)
    abbreviated = _tool(repo, "check", check=False)
    assert abbreviated.returncode == 1
    assert f"{new_entry.relative_to(repo)}: base_commit must be a full 40-character" in (
        abbreviated.stderr
    )

    _set_base_commit(new_entry, _git(repo, "rev-parse", "HEAD").stdout.strip())
    new_entry.write_text(
        new_entry.read_text(encoding="utf-8").replace(
            "## Next", "## Remaining observations", 1
        ),
        encoding="utf-8",
    )
    _stage(repo, new_entry)
    headings = _tool(repo, "check", check=False)
    assert headings.returncode == 1
    assert "expected exactly one '## Next' heading" in headings.stderr


def test_correction_cannot_exempt_a_new_entry(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    new_entry = _new_entry(repo, title="New outcome", topic="new-outcome")
    _set_base_commit(new_entry, "0123456")
    _stage(repo, new_entry)
    correction = _write_correction(
        repo,
        new_entry,
        _correction_payload(
            repo,
            new_entry,
            defects=["abbreviated_base_commit"],
            base_commit=_git(repo, "rev-parse", "HEAD").stdout.strip(),
        ),
    )
    _stage(repo, correction)

    result = _tool(repo, "check", check=False)
    assert result.returncode == 1
    assert "must already be committed in HEAD" in result.stderr


def test_correction_rejects_a_mismatched_sha256(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    entry = _historical_abbreviated_entry(repo)
    correction = _write_correction(
        repo,
        entry,
        _correction_payload(
            repo,
            entry,
            defects=["abbreviated_base_commit"],
            base_commit=_resolved_base_commit(repo, entry),
            sha256="0" * 64,
        ),
    )
    _stage(repo, correction)

    result = _tool(repo, "check", check=False)
    assert result.returncode == 1
    assert "sha256 mismatch" in result.stderr
    # A mismatched record is not applied, so the entry keeps failing on its own.
    assert "base_commit must be a full 40-character" in result.stderr


def test_correction_must_extend_the_recorded_abbreviation(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    entry = _new_entry(repo, title="Historical outcome", topic="historical-outcome")
    _set_base_commit(entry, "abcdef123")
    _commit_entry(repo, entry, "docs: record historical unit")
    correction = _write_correction(
        repo,
        entry,
        _correction_payload(
            repo, entry, defects=["abbreviated_base_commit"], base_commit="0" * 40
        ),
    )
    _stage(repo, correction)

    result = _tool(repo, "check", check=False)
    assert result.returncode == 1
    assert "does not extend the recorded value" in result.stderr


def test_correction_for_an_untracked_entry_is_rejected(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    absent = repo / "worklog" / "entries" / (
        "20990101T000000.000000Z-nobody-absent-topic-abcdef.md"
    )
    correction = _write_correction(
        repo,
        absent,
        _correction_payload(
            repo,
            absent,
            defects=["abbreviated_base_commit"],
            base_commit="0" * 40,
            sha256="1" * 64,
        ),
    )
    _stage(repo, correction)

    result = _tool(repo, "check", check=False)
    assert result.returncode == 1
    assert "corrected entry is not tracked" in result.stderr


def test_correction_for_an_already_valid_entry_is_rejected(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    entry = _new_entry(repo)
    _commit_entry(repo, entry)
    correction = _write_correction(
        repo,
        entry,
        _correction_payload(
            repo,
            entry,
            defects=["abbreviated_base_commit"],
            base_commit=_resolved_base_commit(repo, entry),
        ),
    )
    _stage(repo, correction)

    result = _tool(repo, "check", check=False)
    assert result.returncode == 1
    assert "is unnecessary" in result.stderr


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda p: {**p, "extra": 1}, "correction fields invalid"),
        (
            lambda p: {
                "path": p["path"],
                "schema": p["schema"],
                "sha256": p["sha256"],
                "defects": p["defects"],
                "reason": p["reason"],
                "base_commit": p["base_commit"],
                "superseded_by": p["superseded_by"],
            },
            "fields are out of order",
        ),
        (lambda p: {**p, "schema": 2}, "correction schema must be 1"),
        (lambda p: {**p, "defects": ["unknown_defect"]}, "unknown defect"),
        (
            lambda p: {**p, "defects": ["missing_required_sections", "abbreviated_base_commit"]},
            "sorted",
        ),
        (lambda p: {**p, "base_commit": None}, "base_commit must carry the resolved full hash"),
        (
            lambda p: {**p, "defects": ["missing_required_sections"]},
            "base_commit must carry the resolved full hash",
        ),
        (
            lambda p: {**p, "defects": ["missing_required_sections"], "base_commit": None},
            "superseded_by must name the new correcting entry",
        ),
        (lambda p: {**p, "sha256": "abc"}, "sha256 must be"),
        (lambda p: {**p, "path": "docs/elsewhere.md"}, "path must name a path under"),
        (lambda p: {**p, "path": "worklog/entries/nested/entry.md"}, "must name a file directly"),
        (lambda p: {**p, "reason": ""}, "reason must be one non-empty line"),
    ],
)
def test_correction_record_schema_is_strict(tmp_path: Path, mutation, expected: str) -> None:
    repo = _init_repo(tmp_path)
    entry = _historical_abbreviated_entry(repo)
    payload = _correction_payload(
        repo,
        entry,
        defects=["abbreviated_base_commit"],
        base_commit=_resolved_base_commit(repo, entry),
    )
    correction = _write_correction(repo, entry, mutation(payload))
    _stage(repo, correction)

    result = _tool(repo, "check", check=False)
    assert result.returncode == 1
    assert expected in result.stderr


def test_correction_record_rejects_bad_json_and_misnamed_files(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    entry = _historical_abbreviated_entry(repo)
    payload = _correction_payload(
        repo,
        entry,
        defects=["abbreviated_base_commit"],
        base_commit=_resolved_base_commit(repo, entry),
    )

    broken = _write_correction(repo, entry, payload, raw="{not json")
    _stage(repo, broken)
    invalid = _tool(repo, "check", check=False)
    assert invalid.returncode == 1
    assert "not valid JSON" in invalid.stderr

    misnamed = _write_correction(repo, entry, payload, name="unrelated-record.json")
    _stage(repo, misnamed)
    named = _tool(repo, "check", check=False)
    assert named.returncode == 1
    assert "correction filename must be" in named.stderr

    nested = _write_correction(repo, entry, payload, name="nested/record.json")
    _stage(repo, nested)
    placement = _tool(repo, "check", check=False)
    assert placement.returncode == 1
    assert "unexpected path under worklog corrections" in placement.stderr


def test_superseding_entry_must_pass_without_its_own_correction(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    entry = _historical_entry_without_next(repo)
    superseding = _superseding_entry(repo, entry)
    full_commit = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _set_base_commit(superseding, full_commit[:7])
    _stage(repo, superseding)
    _git(repo, "commit", "-m", "docs: reproduce malformed superseding entry")
    correction = _write_correction(
        repo, entry,
        _correction_payload(
            repo, entry, defects=["missing_required_sections"],
            superseded_by=str(superseding.relative_to(repo)),
        ),
    )
    target_correction = _write_correction(
        repo, superseding,
        _correction_payload(
            repo, superseding, defects=["abbreviated_base_commit"],
            base_commit=full_commit,
        ),
    )
    _stage(repo, correction, target_correction)
    result = _tool(repo, "check", check=False)
    assert result.returncode == 1
    assert "superseded_by must pass the strict schema without a correction" in result.stderr
