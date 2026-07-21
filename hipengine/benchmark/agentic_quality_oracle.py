"""Deterministic external tool-result oracles for broad A6 quality tasks.

The committed oracle fixture is separate from model prompts.  This module
executes selected tool arguments against that fixture, so result, patch, and
test success are not inferred from exact-argument equality.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import operator
from fractions import Fraction
from typing import Any, Mapping

from hipengine.benchmark.agentic import AgenticBenchmarkError, AgenticWorkloadSuite


_TOOL_BY_KIND = {
    "read": "read",
    "grep": "grep",
    "lookup": "lookup",
    "calculate": "calculate",
    "patch": "apply_patch",
    "test": "run_tests",
}
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nonempty_argument(arguments: Mapping[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _fraction_value(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Expression):
        return _fraction_value(node.body)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, int) or isinstance(node.value, bool):
            raise ValueError("calculator accepts integer constants only")
        return Fraction(int(node.value), 1)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _fraction_value(node.left)
        right = _fraction_value(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("division by zero")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_fraction_value(node.operand))
    raise ValueError("calculator expression contains unsupported syntax")


def _calculate(expression: str) -> int | str:
    if len(expression) > 128:
        raise ValueError("calculator expression is too long")
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("calculator expression is invalid") from exc
    value = _fraction_value(parsed)
    if value.denominator == 1:
        return int(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _apply_patch(
    files: dict[str, str],
    patches: Mapping[str, Any],
    patch_id: str,
) -> tuple[str, str]:
    raw_patch = patches.get(patch_id)
    if not isinstance(raw_patch, Mapping):
        raise ValueError(f"unknown patch_id {patch_id!r}")
    path = str(raw_patch["path"])
    old = str(raw_patch["old"])
    new = str(raw_patch["new"])
    source = files[path]
    if source.count(old) != 1:
        raise ValueError(f"patch {patch_id!r} no longer has one exact source region")
    files[path] = source.replace(old, new, 1)
    return path, str(raw_patch["test_suite"])


def _run_test_suite(
    files: Mapping[str, str],
    test_suites: Mapping[str, Any],
    suite_id: str,
) -> tuple[bool, int]:
    raw_suite = test_suites.get(suite_id)
    if not isinstance(raw_suite, Mapping):
        raise ValueError(f"unknown test suite {suite_id!r}")
    expected = raw_suite.get("required_file_sha256")
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError(f"test suite {suite_id!r} has no checks")
    passed = all(
        hashlib.sha256(files[str(path)].encode("utf-8")).hexdigest() == expected_hash
        for path, expected_hash in expected.items()
    )
    return bool(passed), len(expected)


def _execute_case(
    oracle: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    selected_tool: str,
    arguments: Mapping[str, Any],
) -> tuple[dict[str, Any], bool | None, bool | None]:
    kind = str(case["kind"])
    expected_tool = _TOOL_BY_KIND[kind]
    if selected_tool != expected_tool:
        raise ValueError(f"selected tool {selected_tool!r} is not {expected_tool!r}")
    files = copy.deepcopy(dict(oracle["files"]))
    patches = oracle["patches"]
    test_suites = oracle["test_suites"]
    for patch_id in case.get("setup_patches", ()):
        _apply_patch(files, patches, str(patch_id))

    patch_applied: bool | None = None
    tests_passed: bool | None = None
    if kind == "read":
        path = _nonempty_argument(arguments, "path")
        mode = _nonempty_argument(arguments, "mode")
        if path not in files:
            raise ValueError(f"unknown file {path!r}")
        if mode == "raw":
            content = files[path]
        elif mode == "summary":
            summaries = oracle["summaries"]
            if path not in summaries:
                raise ValueError(f"file {path!r} has no summary")
            content = str(summaries[path])
        else:
            raise ValueError(f"unsupported read mode {mode!r}")
        result = {"path": path, "mode": mode, "content": content}
    elif kind == "grep":
        pattern = _nonempty_argument(arguments, "pattern")
        path_prefix = _nonempty_argument(arguments, "path")
        matches: list[dict[str, Any]] = []
        for path in sorted(files):
            if path != path_prefix and not path.startswith(path_prefix.rstrip("/") + "/"):
                continue
            for line_number, line in enumerate(files[path].splitlines(), start=1):
                if pattern in line:
                    matches.append({"path": path, "line": line_number, "text": line})
        result = {"pattern": pattern, "path": path_prefix, "matches": matches}
    elif kind == "lookup":
        key = _nonempty_argument(arguments, "key")
        lookups = oracle["lookups"]
        if key not in lookups:
            raise ValueError(f"unknown lookup key {key!r}")
        result = {"key": key, "value": str(lookups[key])}
    elif kind == "calculate":
        expression = _nonempty_argument(arguments, "expression")
        result = {"value": _calculate(expression)}
    elif kind == "patch":
        patch_id = _nonempty_argument(arguments, "patch_id")
        path, test_suite = _apply_patch(files, patches, patch_id)
        patch_applied = True
        tests_passed, checks = _run_test_suite(files, test_suites, test_suite)
        result = {
            "patch_id": patch_id,
            "path": path,
            "applied": True,
            "result_sha256": hashlib.sha256(files[path].encode("utf-8")).hexdigest(),
            "test_suite": test_suite,
            "tests_passed": tests_passed,
            "checks": checks,
        }
    else:
        suite_id = _nonempty_argument(arguments, "suite")
        tests_passed, checks = _run_test_suite(files, test_suites, suite_id)
        result = {"suite": suite_id, "tests_passed": tests_passed, "checks": checks}
    return result, patch_applied, tests_passed


def evaluate_quality_oracle(
    suite: AgenticWorkloadSuite,
    *,
    case_id: str,
    selected_tool: str | None,
    arguments: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Execute one model-selected tool call against the suite's external oracle."""

    oracle = suite.quality_oracle
    if oracle is None or suite.quality_oracle_file_sha256 is None:
        raise AgenticBenchmarkError("workload suite has no external quality oracle")
    cases = oracle["cases"]
    raw_case = cases.get(case_id)
    if not isinstance(raw_case, Mapping):
        raise AgenticBenchmarkError(f"unknown external quality oracle case {case_id!r}")
    expected_hash = str(raw_case["expected_result_sha256"])
    result_hash: str | None = None
    patch_applied: bool | None = None
    tests_passed: bool | None = None
    error: str | None = None
    try:
        if selected_tool is None:
            raise ValueError("no tool was selected")
        if arguments is None:
            raise ValueError("tool arguments were not a JSON object")
        result, patch_applied, tests_passed = _execute_case(
            oracle,
            case=raw_case,
            selected_tool=selected_tool,
            arguments=arguments,
        )
        result_hash = _canonical_sha256(result)
    except (KeyError, TypeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "case_id": str(case_id),
        "kind": str(raw_case["kind"]),
        "oracle_file_sha256": suite.quality_oracle_file_sha256,
        "evaluated": True,
        "passed": result_hash == expected_hash,
        "result_sha256": result_hash,
        "expected_result_sha256": expected_hash,
        "patch_applied": patch_applied,
        "tests_passed": tests_passed,
        "error": error,
    }


__all__ = ["evaluate_quality_oracle"]
