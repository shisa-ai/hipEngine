from __future__ import annotations

import hashlib
import json
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


# -- base_commit provenance ---------------------------------------------------


def test_check_reports_a_base_commit_that_is_not_a_commit(tmp_path: Path) -> None:
    """A hash-shaped value is not proof of provenance.

    Three entries in this repository recorded a correct short prefix with an
    invented tail. Format validation accepted them, so the defect survived until
    a reviewer resolved every hash against the object database. The check now
    counts and names them without failing, because a committed entry is
    immutable and there is no in-tree remedy for a historical one.
    """

    repo = _init_repo(tmp_path)
    entry = _new_entry(repo)
    text = entry.read_text(encoding="utf-8")
    real = _git(repo, "rev-parse", "HEAD").stdout.strip()
    invented = real[:8] + "0" * 32
    entry.write_text(
        text.replace(f"base_commit: {real}", f"base_commit: {invented}"), encoding="utf-8"
    )
    _commit_entry(repo, entry)

    result = _tool(repo, "check", check=False)
    assert result.returncode == 0, result.stderr
    assert "1 with a base_commit not in this repository" in result.stdout

    verbose = _tool(repo, "check", "--provenance", check=False)
    assert verbose.returncode == 0
    assert invented in verbose.stderr
    assert entry.name in verbose.stderr


def test_check_does_not_report_a_real_base_commit(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    entry = _new_entry(repo)
    _commit_entry(repo, entry)

    result = _tool(repo, "check", check=False)
    assert result.returncode == 0, result.stderr
    assert "not in this repository" not in result.stdout
