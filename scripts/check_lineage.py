#!/usr/bin/env python3
"""Report drift in external source-lineage files before porting kernels.

The script is read-only. It compares the tracked files in docs/source_lineage.json
against the recorded source baseline, reports commits/diffs since that baseline,
flags dirty parent-workspace files, and searches parent WORKLOG/docs for matching
commit or file references.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "source_lineage.json"


@dataclass(frozen=True)
class RepoSpec:
    name: str
    path: Path
    baseline_ref: str
    baseline_note: str = ""


@dataclass(frozen=True)
class SourceSpec:
    repo: str
    path: str
    kind: str
    family: str
    baseline_ref: str | None = None


@dataclass(frozen=True)
class CommitInfo:
    sha: str
    date: str
    subject: str


@dataclass(frozen=True)
class EvidenceHit:
    path: str
    line: int
    text: str


@dataclass(frozen=True)
class SourceReport:
    source: SourceSpec
    repo: RepoSpec
    head: str
    branch: str
    baseline_ref: str
    baseline_exists: bool
    dirty_status: str
    last_commit: CommitInfo | None
    commits_since_baseline: tuple[CommitInfo, ...]
    diffstat: str
    evidence_hits: tuple[EvidenceHit, ...]

    @property
    def changed(self) -> bool:
        return bool((not self.baseline_exists) or self.dirty_status or self.commits_since_baseline)


@dataclass(frozen=True)
class Manifest:
    repos: dict[str, RepoSpec]
    sources: tuple[SourceSpec, ...]
    evidence_paths: tuple[Path, ...]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Source-lineage manifest JSON. Defaults to docs/source_lineage.json.",
    )
    parser.add_argument(
        "--baseline-ref",
        help="Override manifest baseline ref for every source file.",
    )
    parser.add_argument(
        "--kind",
        action="append",
        help="Only include sources with this kind. May be repeated.",
    )
    parser.add_argument(
        "--file",
        action="append",
        dest="file_patterns",
        help="Only include source paths/families matching this fnmatch pattern.",
    )
    parser.add_argument(
        "--diff",
        choices=("none", "stat", "patch"),
        default="stat",
        help="Diff detail to print for changed files. Default: stat.",
    )
    parser.add_argument(
        "--evidence-limit",
        type=int,
        default=8,
        help="Maximum WORKLOG/doc evidence hits per source. Default: 8.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text.",
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit 1 when any tracked source has commits or dirty changes.",
    )
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    selected = select_sources(manifest.sources, args.kind, args.file_patterns)
    reports = tuple(
        build_report(
            manifest,
            source,
            baseline_override=args.baseline_ref,
            evidence_limit=max(args.evidence_limit, 0),
        )
        for source in selected
    )

    if args.json:
        print(json.dumps(reports_to_json(reports), indent=2))
    else:
        print_text_report(reports, manifest_path=args.manifest, diff_mode=args.diff)

    if args.fail_on_drift and any(report.changed for report in reports):
        return 1
    return 0


def load_manifest(path: Path) -> Manifest:
    data = json.loads(path.read_text())
    if data.get("schema") != 1:
        raise ValueError(f"unsupported source-lineage manifest schema: {data.get('schema')!r}")

    repos = {
        name: RepoSpec(
            name=name,
            path=_resolve_manifest_path(repo_data["path"], path),
            baseline_ref=repo_data["baseline_ref"],
            baseline_note=repo_data.get("baseline_note", ""),
        )
        for name, repo_data in data["repositories"].items()
    }
    sources = tuple(
        SourceSpec(
            repo=item["repo"],
            path=item["path"],
            kind=item["kind"],
            family=item.get("family", item["path"]),
            baseline_ref=item.get("baseline_ref"),
        )
        for item in data["files"]
    )
    evidence_paths = tuple(
        _resolve_manifest_path(value, path) for value in data.get("evidence_paths", [])
    )
    return Manifest(repos=repos, sources=sources, evidence_paths=evidence_paths)


def select_sources(
    sources: tuple[SourceSpec, ...],
    kinds: list[str] | None,
    patterns: list[str] | None,
) -> tuple[SourceSpec, ...]:
    selected = []
    for source in sources:
        if kinds and source.kind not in set(kinds):
            continue
        if patterns and not any(_source_matches(source, pattern) for pattern in patterns):
            continue
        selected.append(source)
    return tuple(selected)


def build_report(
    manifest: Manifest,
    source: SourceSpec,
    *,
    baseline_override: str | None,
    evidence_limit: int,
) -> SourceReport:
    repo = manifest.repos[source.repo]
    baseline_ref = baseline_override or source.baseline_ref or repo.baseline_ref
    baseline_exists = git_ok(repo.path, ["cat-file", "-e", f"{baseline_ref}^{{commit}}"])
    head = git(repo.path, ["rev-parse", "--short", "HEAD"])
    branch = git(repo.path, ["branch", "--show-current"], check=False) or "(detached)"
    dirty_status = git(repo.path, ["status", "--porcelain", "--", source.path], check=False)
    last_commit = parse_one_commit(
        git(
            repo.path,
            ["log", "-1", "--format=%h%x09%ad%x09%s", "--date=short", "--", source.path],
            check=False,
        )
    )
    commits = ()
    diffstat = ""
    if baseline_exists:
        commits = parse_commits(
            git(
                repo.path,
                [
                    "log",
                    "--format=%h%x09%ad%x09%s",
                    "--date=short",
                    f"{baseline_ref}..HEAD",
                    "--",
                    source.path,
                ],
                check=False,
            )
        )
        diffstat = git(
            repo.path,
            ["diff", "--stat", f"{baseline_ref}..HEAD", "--", source.path],
            check=False,
        )
    evidence_hits = find_evidence_hits(
        manifest.evidence_paths,
        source,
        commits,
        last_commit,
        limit=evidence_limit,
    )
    return SourceReport(
        source=source,
        repo=repo,
        head=head,
        branch=branch,
        baseline_ref=baseline_ref,
        baseline_exists=baseline_exists,
        dirty_status=dirty_status,
        last_commit=last_commit,
        commits_since_baseline=commits,
        diffstat=diffstat,
        evidence_hits=evidence_hits,
    )


def print_text_report(
    reports: tuple[SourceReport, ...],
    *,
    manifest_path: Path,
    diff_mode: str,
) -> None:
    changed = sum(1 for report in reports if report.changed)
    print("Source-lineage drift report")
    print(f"manifest: {manifest_path}")
    print(f"tracked_sources: {len(reports)} changed_or_dirty: {changed}")
    if not reports:
        print("no sources selected")
        return

    repo_keys = []
    for report in reports:
        key = (report.repo.name, str(report.repo.path), report.head, report.branch)
        if key not in repo_keys:
            repo_keys.append(key)
    for name, path, head, branch in repo_keys:
        print(f"repo {name}: {path} branch={branch} head={head}")
    print()

    for report in reports:
        status = "DRIFT" if report.changed else "clean"
        print(f"[{status}] {report.source.kind} {report.source.path}")
        print(f"  family: {report.source.family}")
        print(f"  baseline: {report.baseline_ref}")
        if report.repo.baseline_note:
            print(f"  baseline_note: {report.repo.baseline_note}")
        if not report.baseline_exists:
            print("  baseline_missing: true")
        if report.dirty_status:
            print("  dirty_status:")
            for line in report.dirty_status.splitlines():
                print(f"    {line}")
        if report.last_commit:
            print(
                "  last_commit: "
                f"{report.last_commit.sha} {report.last_commit.date} "
                f"{report.last_commit.subject}"
            )
        print(f"  commits_since_baseline: {len(report.commits_since_baseline)}")
        for commit in report.commits_since_baseline:
            print(f"    {commit.sha} {commit.date} {commit.subject}")
        if diff_mode != "none" and report.diffstat:
            print("  diffstat:")
            for line in report.diffstat.splitlines():
                print(f"    {line}")
        if diff_mode == "patch" and report.baseline_exists and report.changed:
            patch = git(
                report.repo.path,
                ["diff", f"{report.baseline_ref}..HEAD", "--", report.source.path],
                check=False,
            )
            if patch:
                print("  patch:")
                for line in patch.splitlines():
                    print(f"    {line}")
        if report.evidence_hits:
            print("  evidence_hits:")
            for hit in report.evidence_hits:
                print(f"    {hit.path}:{hit.line}: {hit.text}")
        else:
            print("  evidence_hits: none")
        print()


def reports_to_json(reports: tuple[SourceReport, ...]) -> dict[str, Any]:
    return {
        "schema": 1,
        "sources": [
            {
                "repo": report.repo.name,
                "repo_path": str(report.repo.path),
                "head": report.head,
                "branch": report.branch,
                "path": report.source.path,
                "kind": report.source.kind,
                "family": report.source.family,
                "baseline_ref": report.baseline_ref,
                "baseline_exists": report.baseline_exists,
                "changed": report.changed,
                "dirty_status": report.dirty_status.splitlines(),
                "last_commit": commit_to_json(report.last_commit),
                "commits_since_baseline": [
                    commit_to_json(commit) for commit in report.commits_since_baseline
                ],
                "diffstat": report.diffstat,
                "evidence_hits": [
                    {"path": hit.path, "line": hit.line, "text": hit.text}
                    for hit in report.evidence_hits
                ],
            }
            for report in reports
        ],
    }


def commit_to_json(commit: CommitInfo | None) -> dict[str, str] | None:
    if commit is None:
        return None
    return {"sha": commit.sha, "date": commit.date, "subject": commit.subject}


def find_evidence_hits(
    evidence_paths: tuple[Path, ...],
    source: SourceSpec,
    commits: tuple[CommitInfo, ...],
    last_commit: CommitInfo | None,
    *,
    limit: int,
) -> tuple[EvidenceHit, ...]:
    if limit <= 0:
        return ()

    # Prefer exact commit references. File-name searches are a fallback because broad
    # terms such as "activation" or "WMMA" produce stale/noisy hits in the parent log.
    commit_tokens = evidence_commit_tokens(commits, last_commit)
    hits = _find_token_hits(evidence_paths, commit_tokens, limit=limit)
    if hits:
        return hits
    return _find_token_hits(evidence_paths, evidence_path_tokens(source), limit=limit)


def evidence_commit_tokens(
    commits: tuple[CommitInfo, ...],
    last_commit: CommitInfo | None,
) -> tuple[str, ...]:
    tokens = {commit.sha.lower() for commit in commits}
    if last_commit is not None and not tokens:
        tokens.add(last_commit.sha.lower())
    return tuple(token for token in tokens if len(token) >= 4)


def evidence_path_tokens(source: SourceSpec) -> tuple[str, ...]:
    tokens = {source.path.lower(), Path(source.path).name.lower(), source.family.lower()}
    return tuple(token for token in tokens if len(token) >= 4)


def _find_token_hits(
    evidence_paths: tuple[Path, ...],
    tokens: tuple[str, ...],
    *,
    limit: int,
) -> tuple[EvidenceHit, ...]:
    if not tokens:
        return ()
    hits: list[EvidenceHit] = []
    for path in evidence_paths:
        if not path.exists() or not path.is_file():
            continue
        for line_number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            lower = line.lower()
            if any(token in lower for token in tokens):
                hits.append(EvidenceHit(str(path), line_number, line.strip()))
                if len(hits) >= limit:
                    return tuple(hits)
    return tuple(hits)


def parse_commits(text: str) -> tuple[CommitInfo, ...]:
    commits = []
    for line in text.splitlines():
        parsed = parse_one_commit(line)
        if parsed is not None:
            commits.append(parsed)
    return tuple(commits)


def parse_one_commit(line: str) -> CommitInfo | None:
    if not line.strip():
        return None
    parts = line.split("\t", 2)
    if len(parts) != 3:
        return CommitInfo(parts[0], "", " ".join(parts[1:]))
    return CommitInfo(parts[0], parts[1], parts[2])


def git(repo: Path, args: list[str], *, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        command = " ".join(["git", "-C", str(repo), *args])
        raise RuntimeError(f"{command} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def git_ok(repo: Path, args: list[str]) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _source_matches(source: SourceSpec, pattern: str) -> bool:
    return any(
        fnmatch.fnmatch(value, pattern)
        for value in (source.path, Path(source.path).name, source.kind, source.family)
    )


def _resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    expanded = Path(value).expanduser()
    if expanded.is_absolute():
        return expanded
    return (manifest_path.parent / expanded).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
