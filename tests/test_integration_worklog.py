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
    second.rename(second.with_name(second.name.replace(second_stamp_name, first_stamp_name, 1)))

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
def test_check_rejects_malformed_entries(tmp_path: Path, mutation, expected: str) -> None:
    repo = _init_repo(tmp_path)
    entry = _new_entry(repo)
    entry.write_text(mutation(entry.read_text(encoding="utf-8")), encoding="utf-8")

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


def test_managed_hook_validates_entries_and_tolerates_branch_without_tool(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _tool(repo, "install-hook")
    hook = repo / ".git" / "hooks" / "pre-commit"
    entry = _new_entry(repo)
    entry.write_text(entry.read_text(encoding="utf-8") + "<<<<<<< bad\n", encoding="utf-8")
    assert _run(repo, str(hook), check=False).returncode == 1

    tool = repo / "scripts" / "worklog.py"
    hidden = repo / "scripts" / "worklog.py.hidden"
    tool.rename(hidden)
    try:
        result = _run(repo, str(hook), check=False)
    finally:
        hidden.rename(tool)
    assert result.returncode == 0
