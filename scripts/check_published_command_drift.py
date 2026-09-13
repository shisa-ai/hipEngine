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
nothing is imported and no GPU is touched. The gate reads the *composed* parser, because that is
what a reader copying the command actually runs:

  * `suite.build_parser()` - the script's parser starts from another module's builder
  * `add_kv_policy_args(parser, ...)` - shared flags added by a repo-local helper
  * `action=argparse.BooleanOptionalAction` - argparse also accepts the `--no-<flag>` form

Resolution stays static and bounded: a helper must be a repo-local module function reached
through a literal import, and a spread option tuple (`*legacy_storage_flags`) must be literal at
the call site. A tool that declares flags dynamically (a non-literal `add_argument(*names)`) is
treated as un-inspectable, skipped rather than guessed at, and named in `scripts_skipped` so a
disabled script is visible instead of silently ignored.

A recorded test path may name a file the explicit-tier migration renamed. Rewriting the artifact
would falsify the provenance of a measured row and a per-file exception would hide the drift, so
the path is resolved through the migration's own record and reported in `renamed_targets`.

Usage:
    .venv/bin/python scripts/check_published_command_drift.py [--repo .] [--json out.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
import subprocess
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

# The explicit-tier migration renamed every test module. Published artifacts recorded the paths
# that existed when they ran, so those commands name files that no longer exist. The migration's
# own record is the rename map: resolving through it keeps a historical command checkable without
# editing the artifact or hiding the redirect behind an exception.
TIER_RENAME_RECORD = "docs/testing/test-tier-migration-2026-09-12.json"
UD_TIER_RENAME_RECORD = "docs/testing/test-tier-migration-ud-2026-09-13.json"


def exception_key(artifact: str, problem: str, detail: str) -> str:
    return f"{artifact}::{problem}::{detail}"


@lru_cache(maxsize=None)
def _parsed(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return None


def _literal_strings(node: ast.AST) -> tuple[str, ...] | None:
    """A literal tuple/list/set of strings, or None when it is not statically known."""
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return None
    values: list[str] = []
    for element in node.elts:
        if not (isinstance(element, ast.Constant) and isinstance(element.value, str)):
            return None
        values.append(element.value)
    return tuple(values)


def _is_boolean_optional(call: ast.Call) -> bool:
    """True when `action=` makes argparse synthesize the `--no-<flag>` negation."""
    for keyword in call.keywords:
        if keyword.arg != "action":
            continue
        value = keyword.value
        name = value.attr if isinstance(value, ast.Attribute) else getattr(value, "id", "")
        return name == "BooleanOptionalAction"
    return False


def _add_argument_flags(
    call: ast.Call, bindings: dict[str, tuple[str, ...]]
) -> tuple[set[str], bool]:
    """Option strings of one `add_argument` call; False when the call cannot be read.

    Only positional args are option strings. Keywords are argparse options (type=, default=,
    action=) whose values say nothing about the CLI. Scanning them made every script with a
    typed argument look un-inspectable, which silently disabled the gate for those tools.
    """
    flags: set[str] = set()
    for argument in call.args:
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            if argument.value.startswith("-"):
                flags.add(argument.value)
        elif isinstance(argument, ast.Starred) and isinstance(argument.value, ast.Name):
            # `*legacy_storage_flags`: known only if the call site passes a literal tuple.
            spread = bindings.get(argument.value.id)
            if spread is None:
                return set(), False
            flags.update(value for value in spread if value.startswith("-"))
        else:
            # Dynamically constructed option names: we cannot claim to know the CLI.
            return set(), False
    if _is_boolean_optional(call):
        negations = {f"--no-{flag[2:]}" for flag in flags if flag.startswith("--")}
        flags |= negations
    return flags, True


def _module_path(repo: Path, module: str) -> Path | None:
    """The repo-local file for a dotted module name, or None for third-party imports."""
    parts = [part for part in module.split(".") if part]
    if not parts:
        return None
    for base in (repo, repo / "scripts"):
        candidate = base.joinpath(*parts)
        for path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
            if path.is_file() and path.is_relative_to(repo):
                return path
    return None


def _imported_modules(tree: ast.AST, repo: Path) -> dict[str, Path]:
    """Local name -> repo-local module file, for `import x as y` and `from x import y`."""
    found: dict[str, Path] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                path = _module_path(repo, alias.name)
                if path is not None:
                    found[alias.asname or alias.name.split(".")[0]] = path
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            module = f"{'.' * node.level}{node.module}" if node.level else node.module
            for alias in node.names:
                # `from scripts import suite as s` names a module; `from m import helper` a function.
                path = _module_path(repo, f"{module}.{alias.name}") or _module_path(repo, module)
                if path is not None:
                    found[alias.asname or alias.name] = path
    return found


def _named_function(tree: ast.Module | None, name: str) -> ast.FunctionDef | None:
    if tree is None:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _helper_target(call: ast.Call, *, imports: dict[str, Path]) -> tuple[Path, str] | None:
    """(module file, function name) when the call targets a repo-local module."""
    func = call.func
    if isinstance(func, ast.Name):
        path = imports.get(func.id)
        return (path, func.id) if path is not None else None
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        path = imports.get(func.value.id)
        return (path, func.attr) if path is not None else None
    return None


def _call_bindings(target: ast.FunctionDef, call: ast.Call) -> dict[str, tuple[str, ...]]:
    """Literal string tuples bound to `target`'s parameters at this call site."""
    positional = [argument.arg for argument in target.args.args]
    bindings: dict[str, tuple[str, ...]] = {}
    for index, argument in enumerate(call.args):
        if isinstance(argument, ast.Starred) or index >= len(positional):
            continue
        values = _literal_strings(argument)
        if values is not None:
            bindings[positional[index]] = values
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        values = _literal_strings(keyword.value)
        if values is not None:
            bindings[keyword.arg] = values
    return bindings


def _collect_flags(
    node: ast.AST,
    *,
    module: Path,
    repo: Path,
    bindings: dict[str, tuple[str, ...]],
    chain: frozenset[tuple[Path, str]],
) -> frozenset[str] | None:
    """Flags declared under `node`, including the repo-local helpers it calls."""
    imports = _imported_modules(_parsed(module) or ast.Module(body=[], type_ignores=[]), repo)
    flags: set[str] = set()
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "add_argument":
            found, readable = _add_argument_flags(call, bindings)
            if not readable:
                return None
            flags |= found
            continue
        helper = _helper_target(call, imports=imports)
        if helper is None:
            continue
        path, name = helper
        if (path, name) in chain:
            continue
        target = _named_function(_parsed(path), name)
        if target is None:
            continue
        nested = _collect_flags(
            target,
            module=path,
            repo=repo,
            bindings=_call_bindings(target, call),
            chain=chain | {(path, name)},
        )
        if nested is None:
            return None
        flags |= nested
    return frozenset(flags)


@lru_cache(maxsize=None)
def _declared_flags(script: Path, repo: Path) -> frozenset[str] | None:
    """Flags the script accepts, or None when its CLI cannot be inspected statically."""
    tree = _parsed(script)
    if tree is None:
        return None
    return _collect_flags(tree, module=script, repo=repo, bindings={}, chain=frozenset())


@lru_cache(maxsize=None)
def _tier_renames(repo: Path) -> dict[str, str]:
    """Recorded old -> new test paths from the explicit-tier migration."""
    modules = []
    for record in (TIER_RENAME_RECORD, UD_TIER_RENAME_RECORD):
        try:
            payload = json.loads((repo / record).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload.get("modules"), list):
            modules.extend(payload["modules"])
    return {
        str(module["old"]): str(module["new"])
        for module in modules
        if isinstance(module, dict) and module.get("old") and module.get("new")
    }


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


def _worktree_roots(repo: Path) -> tuple[Path, ...]:
    """Recognize only linked checkouts registered in this repository's Git metadata."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "worktree", "list", "--porcelain", "-z"],
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return (repo,)
    roots = tuple(Path(field.removeprefix("worktree ")).resolve()
                  for field in result.stdout.split("\0") if field.startswith("worktree "))
    return tuple(dict.fromkeys((repo, *roots)))


def _resolve_target(
    raw: str, repo: Path, worktrees: tuple[Path, ...]
) -> tuple[str, Path | None, str | None]:
    """(repo-relative name, path, problem) for one `.py` target in a recorded command."""
    script = Path(raw)
    if script.is_absolute():
        for root in (repo, *worktrees):
            if script.is_relative_to(root):
                script = script.relative_to(root)
                break
        else:
            return raw, None, "SCRIPT-NOT-IN-REPO"
    path = (repo / script).resolve()
    if not path.is_relative_to(repo):
        return raw, None, "SCRIPT-NOT-IN-REPO"
    return str(script), path, None


def _violations_for_command(
    artifact: str, command: str, repo: Path, *, worktrees: tuple[Path, ...] = ()
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    """(violations, renamed targets, un-inspectable scripts) for one recorded command."""
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return [{"artifact": artifact, "problem": "COMMAND-UNPARSEABLE", "detail": str(exc)}], [], []
    targets = [token for token in tokens if token.endswith(".py")]
    if not targets:
        return [], [], []  # not a python invocation (a pytest -k string, a shell pipeline, etc.)
    violations: list[dict[str, str]] = []
    renamed: list[dict[str, str]] = []
    scripts: list[tuple[str, Path]] = []
    # Every `.py` token is a target: `python -m pytest tests/a.py tests/b.py` has two, and a
    # renamed or deleted second one used to pass unnoticed because only the first was checked.
    for raw in targets:
        relative, path, problem = _resolve_target(raw, repo, worktrees)
        if problem is not None or path is None:
            violations.append(
                {"artifact": artifact, "problem": problem or "SCRIPT-NOT-IN-REPO",
                 "detail": raw, "command": command}
            )
            continue
        if not path.is_file():
            current = _tier_renames(repo).get(relative)
            if current is not None and (repo / current).is_file():
                renamed.append({"artifact": artifact, "recorded": relative, "current": current})
                continue
            violations.append(
                {"artifact": artifact, "problem": "SCRIPT-MISSING", "detail": relative,
                 "command": command}
            )
            continue
        scripts.append((relative, path))
    # A pytest invocation runs test modules, not an argparse CLI, so only the targets matter.
    if "pytest" not in tokens and scripts:
        relative, path = scripts[0]
        declared = _declared_flags(path, repo)
        if declared is None:
            return violations, renamed, [relative]
        for token in tokens[tokens.index(targets[0]) + 1:]:
            if not token.startswith("--"):
                continue
            name = token.split("=", 1)[0]
            if name not in declared:
                violations.append(
                    {
                        "artifact": artifact,
                        "problem": "UNKNOWN-FLAG",
                        "detail": name,
                        "script": relative,
                        "command": command,
                    }
                )
    return violations, renamed, []


def check_repo(repo: Path, exceptions: dict[str, str] | None = None) -> dict[str, Any]:
    """Audit the commands of every artifact cited by benchmarks/README.md."""
    repo = Path(repo).resolve()
    worktrees = _worktree_roots(repo)
    allow = EXCEPTIONS if exceptions is None else exceptions
    readme = repo / "benchmarks" / "README.md"
    if not readme.is_file():
        raise FileNotFoundError(f"no benchmarks/README.md under {repo}")
    cited = sorted(set(CITATION.findall(readme.read_text())))
    violations: list[dict[str, str]] = []
    renamed: list[dict[str, str]] = []
    skipped: set[str] = set()
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
            command_violations, command_renamed, command_skipped = _violations_for_command(
                name, command, repo, worktrees=worktrees
            )
            violations.extend(command_violations)
            renamed.extend(command_renamed)
            skipped.update(command_skipped)

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
        # Historical paths redirected through the tier-migration record. Reported, not hidden:
        # a reader needs the current path, and the row's own text stays as measured.
        "renamed_targets": sorted(
            {f"{entry['artifact']}::{entry['recorded']}->{entry['current']}" for entry in renamed}
        ),
        "scripts_skipped": sorted(skipped),
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
        for target in report["renamed_targets"]:
            print(f"  renamed by the tier migration: {target}")
        for script in report["scripts_skipped"]:
            print(f"  SKIPPED (CLI not statically readable): {script}")
    return 1 if report["violations"] or report["exceptions_unmatched"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
