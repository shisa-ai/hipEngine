#!/usr/bin/env python3
"""Fail when a published benchmark artifact's recorded command no longer matches its script.

Every retained artifact records the command that produced it. That record is the whole value of
a published row: a rate nobody can re-run is a story. Refactors break it silently, and it broke
twice in one session here - a rollup rewrite deleted `--prior-config-changed` from a tool whose
artifact cited it, and `--require-mtp` disappeared from `scripts/gguf_mtp_c1c8_server_bench.py`
while the headline grouped-prefill promotion artifact still recorded it in `command`.

Scope is deliberate. Auditing all 1579 recorded `scripts/*.py` commands in `benchmarks/results`
finds 125 distinct historical drifts (flags removed months after the run, tools that no longer
exist), which is not a fixable backlog. What must stay executable is the set the README
publishes, so this checks the artifacts cited by `benchmarks/README.md`. Pre-existing problems
that belong to another lane are listed in EXCEPTIONS with a date and reason - reported, counted,
and never silently dropped - rather than rewritten, because a published row's provenance belongs
to its author.

Check is parse-level and side-effect free: the script's declared flags are read with `ast`, so
nothing is imported and no GPU is touched. A tool that declares flags dynamically (a non-literal
`add_argument(*names)`) is treated as un-inspectable and skipped rather than guessed at.

Usage:
    .venv/bin/python scripts/check_published_command_drift.py [--repo .] [--json out.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

COMMAND_KEYS = ("command", "source_command")
CITATION = re.compile(r"([A-Za-z0-9._-]+\.json)(?!l)")

# Pre-existing drift owned by another lane: key -> dated reason. Removing an entry here means the
# artifact was fixed (or the row was re-measured); leaving one that no longer matches fails the
# gate, which is intentional, so stale exemptions get pruned.
EXCEPTIONS: dict[str, str] = {
    "2026-08-09-cuda-sm120a-maple-splitk-global-decode-retained.json::SCRIPT-NOT-IN-REPO::"
    "/tmp/hipengine-maple-splitk-clean/scripts/maple_c1_bench.py": (
        "2026-08-09 cuda-sm120a lane: the recorded command invoked a script that lived only "
        "under /tmp, so the row was never reproducible from the repo. The owner should "
        "re-measure with a committed tool or relabel the row as a one-off probe. Not rewritten "
        "here because a published row's provenance belongs to its author."
    ),
    "2026-08-16-qwen36-35b-gfx1151-rocmfpx-opp3-silu-rotate-retained.json::SCRIPT-NOT-IN-REPO::"
    "/tmp/hipengine-rocmfpx-transfer-campaign/opp3_leaf.py": (
        "2026-08-16 gfx1151 lane: same shape - the command references a /tmp script from the "
        "ROCMFPX transfer campaign that was never committed."
    ),
    "2026-08-08-gfx1151-maple-d0-selector-snapshot-retained.json::UNKNOWN-FLAG::--comparison": (
        "2026-08-08 gfx1151 lane: maple_c1_bench.py dropped --comparison after the run; the row "
        "is a selector snapshot on hardware not present here. Owner to re-measure or annotate."
    ),
}


def exception_key(artifact: str, problem: str, detail: str) -> str:
    return f"{artifact}::{problem}::{detail}"


@lru_cache(maxsize=None)
def _declared_flags(script: Path) -> frozenset[str] | None:
    """Flags the script declares, or None when it cannot be inspected statically."""
    try:
        tree = ast.parse(script.read_text())
    except (OSError, SyntaxError):
        return None
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        # Only positional args are option strings. Keywords are argparse options (type=, default=,
        # action=) whose values say nothing about the CLI. Scanning them made every script with a
        # typed argument look un-inspectable, which silently disabled the gate for those tools.
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.startswith("-"):
                    flags.add(arg.value)
            else:
                # Dynamically constructed option names: we cannot claim to know the CLI.
                return None
    return frozenset(flags)


def _commands(payload: Any) -> list[str]:
    found: list[str] = []
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key in COMMAND_KEYS and isinstance(value, str):
                    found.append(value)
                else:
                    stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return found


def _violations_for_command(
    artifact: str, command: str, repo: Path
) -> list[dict[str, str]]:
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return [{"artifact": artifact, "problem": "COMMAND-UNPARSEABLE", "detail": str(exc)}]
    indexes = [index for index, token in enumerate(tokens) if token.endswith(".py")]
    if not indexes:
        return []  # not a python invocation (a pytest -k string, a shell pipeline, etc.)
    raw = tokens[indexes[0]]
    script = Path(raw)
    if script.is_absolute():
        try:
            script = script.relative_to(repo)
        except ValueError:
            return [
                {
                    "artifact": artifact,
                    "problem": "SCRIPT-NOT-IN-REPO",
                    "detail": raw,
                    "command": command,
                }
            ]
    path = repo / script
    if not path.is_file():
        return [
            {
                "artifact": artifact,
                "problem": "SCRIPT-MISSING",
                "detail": str(script),
                "command": command,
            }
        ]
    declared = _declared_flags(path)
    if declared is None:
        return []
    violations = []
    for token in tokens[indexes[0] + 1:]:
        if not token.startswith("--"):
            continue
        name = token.split("=", 1)[0]
        if name not in declared:
            violations.append(
                {
                    "artifact": artifact,
                    "problem": "UNKNOWN-FLAG",
                    "detail": name,
                    "script": str(script),
                    "command": command,
                }
            )
    return violations


def check_repo(repo: Path, exceptions: dict[str, str] | None = None) -> dict[str, Any]:
    """Audit the commands of every artifact cited by benchmarks/README.md."""
    repo = Path(repo)
    allow = EXCEPTIONS if exceptions is None else exceptions
    readme = repo / "benchmarks" / "README.md"
    if not readme.is_file():
        raise FileNotFoundError(f"no benchmarks/README.md under {repo}")
    cited = sorted(set(CITATION.findall(readme.read_text())))
    violations: list[dict[str, str]] = []
    for name in cited:
        path = repo / "benchmarks" / "results" / name
        if not path.is_file():
            violations.append(
                {"artifact": name, "problem": "MISSING-ARTIFACT", "detail": name}
            )
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            violations.append(
                {"artifact": name, "problem": "ARTIFACT-UNREADABLE", "detail": str(exc)}
            )
            continue
        for command in _commands(payload):
            violations.extend(_violations_for_command(name, command, repo))

    matched: list[str] = []
    kept: list[dict[str, str]] = []
    for violation in violations:
        key = exception_key(violation["artifact"], violation["problem"], violation["detail"])
        if key in allow:
            matched.append(key)
        else:
            kept.append(violation)
    matched_set = set(matched)
    return {
        "schema": "published_command_drift.v1",
        "repo": str(repo),
        "artifacts_cited": len(cited),
        "artifacts_checked": len(cited),
        "violations": kept,
        "exceptions_matched": sorted(matched_set),
        "exceptions_unmatched": sorted(set(allow) - matched_set),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    report = check_repo(args.repo)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not args.quiet:
        print(
            f"cited artifacts: {report['artifacts_cited']}  violations: "
            f"{len(report['violations'])}  exceptions matched: "
            f"{len(report['exceptions_matched'])}"
        )
        for violation in report["violations"]:
            print(
                f"  {violation['artifact'][:60]:<60} {violation['problem']} "
                f"{violation['detail']}"
            )
        for key in report["exceptions_unmatched"]:
            print(f"  STALE EXCEPTION (remove it): {key}")
    return 1 if report["violations"] or report["exceptions_unmatched"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
