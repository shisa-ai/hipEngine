from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_lineage.py"


def test_check_lineage_reports_commits_and_worklog_hits(tmp_path: Path) -> None:
    source_repo = tmp_path / "source"
    source_repo.mkdir()
    run(["git", "init"], cwd=source_repo)
    run(["git", "config", "user.email", "test@example.invalid"], cwd=source_repo)
    run(["git", "config", "user.name", "Test User"], cwd=source_repo)

    kernel = source_repo / "kernels" / "foo.hip"
    kernel.parent.mkdir()
    kernel.write_text("extern \"C\" __global__ void foo() {}\n")
    run(["git", "add", "kernels/foo.hip"], cwd=source_repo)
    run(["git", "commit", "-m", "baseline foo kernel"], cwd=source_repo)
    baseline = git_output(source_repo, "rev-parse", "HEAD")

    kernel.write_text("extern \"C\" __global__ void foo_v2() {}\n")
    run(["git", "add", "kernels/foo.hip"], cwd=source_repo)
    run(["git", "commit", "-m", "perf: update foo kernel"], cwd=source_repo)
    head_short = git_output(source_repo, "rev-parse", "--short", "HEAD")

    worklog = tmp_path / "WORKLOG.md"
    worklog.write_text(f"## Entry\nCommitted child repo `{head_short}` for foo kernel.\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "evidence_paths": [str(worklog)],
                "repositories": {
                    "source": {
                        "path": str(source_repo),
                        "baseline_ref": baseline,
                    }
                },
                "files": [
                    {
                        "repo": "source",
                        "path": "kernels/foo.hip",
                        "kind": "kernel",
                        "family": "foo test kernel",
                    }
                ],
            }
        )
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--json"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads(result.stdout)
    source_report = report["sources"][0]

    assert source_report["changed"] is True
    assert source_report["commits_since_baseline"][0]["sha"] == head_short
    assert "kernels/foo.hip" in source_report["diffstat"]
    assert source_report["evidence_hits"][0]["line"] == 2
    assert head_short in source_report["evidence_hits"][0]["text"]


def test_check_lineage_fail_on_drift_exits_nonzero(tmp_path: Path) -> None:
    source_repo = tmp_path / "source"
    source_repo.mkdir()
    run(["git", "init"], cwd=source_repo)
    run(["git", "config", "user.email", "test@example.invalid"], cwd=source_repo)
    run(["git", "config", "user.name", "Test User"], cwd=source_repo)

    source_file = source_repo / "dispatch.py"
    source_file.write_text("VERSION = 1\n")
    run(["git", "add", "dispatch.py"], cwd=source_repo)
    run(["git", "commit", "-m", "baseline dispatch"], cwd=source_repo)
    baseline = git_output(source_repo, "rev-parse", "HEAD")

    source_file.write_text("VERSION = 2\n")
    run(["git", "add", "dispatch.py"], cwd=source_repo)
    run(["git", "commit", "-m", "feat: update dispatch"], cwd=source_repo)

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "repositories": {
                    "source": {
                        "path": str(source_repo),
                        "baseline_ref": baseline,
                    }
                },
                "files": [
                    {
                        "repo": "source",
                        "path": "dispatch.py",
                        "kind": "dispatch",
                        "family": "test dispatch",
                    }
                ],
            }
        )
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--fail-on-drift"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "[DRIFT] dispatch dispatch.py" in result.stdout


def run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=True)


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def test_check_lineage_missing_repository_reports_instead_of_crashing(
    tmp_path: Path,
) -> None:
    """A repository absent on this host must not discard every other report.

    ``build_report`` shells out to git against the repository path. When the
    manifest still names a repository that no longer exists on this machine,
    git raises and the whole run dies, so the check returns no reports at all
    and kernel work has no lineage gate to clear. The gap is reported by name on
    stderr instead, and the sources whose repository is present are still
    checked -- losing that coverage silently would be worse than the crash.
    """

    source_repo = tmp_path / "source"
    source_repo.mkdir()
    run(["git", "init"], cwd=source_repo)
    run(["git", "config", "user.email", "test@example.invalid"], cwd=source_repo)
    run(["git", "config", "user.name", "Test User"], cwd=source_repo)

    kernel = source_repo / "kernels" / "foo.hip"
    kernel.parent.mkdir()
    kernel.write_text("extern \"C\" __global__ void foo() {}\n")
    run(["git", "add", "kernels/foo.hip"], cwd=source_repo)
    run(["git", "commit", "-m", "baseline foo kernel"], cwd=source_repo)
    baseline = git_output(source_repo, "rev-parse", "HEAD")

    missing = tmp_path / "retired-repo"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "repositories": {
                    "source": {"path": str(source_repo), "baseline_ref": baseline},
                    "retired": {"path": str(missing), "baseline_ref": baseline},
                },
                "files": [
                    {
                        "repo": "source",
                        "path": "kernels/foo.hip",
                        "kind": "kernel",
                        "family": "foo test kernel",
                    },
                    {
                        "repo": "retired",
                        "path": "kernels/gone.hip",
                        "kind": "kernel",
                        "family": "retired test kernel",
                    },
                ],
            }
        )
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--json"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    # The run completes rather than raising out of git on the missing path.
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    # The present repository is still checked, so coverage is not lost.
    assert [s["repo"] for s in report["sources"]] == ["source"]
    # And the gap is named rather than silently skipped.
    assert "retired" in result.stderr
    assert str(missing) in result.stderr
