"""Fail-closed loader and reference evaluator for AGENTIC-QUALITY2 v1."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import operator
import re
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from hipengine.benchmark.agentic_quality import _arguments_match_schema
from hipengine.benchmark.agentic_quality_taxonomy import parse_independent_tool_envelope


class AgenticQuality2Error(ValueError):
    """Raised when expanded quality evidence cannot support a valid claim."""


@dataclass(frozen=True)
class AgenticQuality2Suite:
    """Validated frozen expanded suite, oracle, and source identities."""

    path: Path
    payload: Mapping[str, Any]
    file_sha256: str
    oracle_path: Path
    oracle: Mapping[str, Any]
    oracle_sha256: str
    sources_path: Path
    sources: Mapping[str, Any]
    sources_sha256: str
    tools: Mapping[str, Mapping[str, Any]]
    workloads: Mapping[str, Mapping[str, Any]]
    development_ids: tuple[str, ...]
    heldout_ids: tuple[str, ...]

    def identity(self) -> dict[str, Any]:
        return {
            "suite": str(self.payload["suite"]),
            "path": str(self.path),
            "file_sha256": self.file_sha256,
            "oracle_path": str(self.oracle_path),
            "oracle_sha256": self.oracle_sha256,
            "sources_path": str(self.sources_path),
            "sources_sha256": self.sources_sha256,
            "development_ids": list(self.development_ids),
            "heldout_ids": list(self.heldout_ids),
        }


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_TOOL_KIND = {
    "read_file": "read",
    "search_repo": "search",
    "lookup_record": "lookup",
    "calculate": "calculate",
    "transform_record": "transform",
    "apply_patch": "patch",
    "run_fixture_tests": "test",
    "submit_code": "code",
    "submit_response": "instruction",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgenticQuality2Error(f"{label} is not readable JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AgenticQuality2Error(f"{label} must contain a JSON object")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AgenticQuality2Error(f"{label} must be an object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise AgenticQuality2Error(f"{label} must be an array")
    return value


def _nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgenticQuality2Error(f"{label} must be a non-empty string")
    return value


def _resolve_ref(parent: Path, value: Any, *, label: str) -> tuple[Path, str]:
    ref = _mapping(value, label=label)
    raw_path = _nonempty(ref.get("path"), label=f"{label}.path")
    expected = _nonempty(ref.get("file_sha256"), label=f"{label}.file_sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise AgenticQuality2Error(f"{label}.file_sha256 must be lowercase SHA-256")
    path = Path(raw_path)
    if not path.is_absolute():
        path = parent / path
    path = path.resolve()
    if not path.is_file() or _sha256(path) != expected:
        raise AgenticQuality2Error(f"{label} hash/path mismatch")
    return path, expected


def _type_matches(value: Any, kind: str) -> bool:
    if kind == "object":
        return isinstance(value, Mapping)
    if kind == "array":
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "null":
        return value is None
    return True


def _validate_value(value: Any, schema: Mapping[str, Any], *, label: str) -> None:
    kind = schema.get("type")
    kinds = (kind,) if isinstance(kind, str) else kind if isinstance(kind, list) else ()
    if kinds and not any(_type_matches(value, str(item)) for item in kinds):
        raise AgenticQuality2Error(f"{label} has wrong JSON type")
    if "const" in schema and value != schema["const"]:
        raise AgenticQuality2Error(f"{label} differs from const")
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        raise AgenticQuality2Error(f"{label} is outside enum")
    if isinstance(value, str):
        if isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
            raise AgenticQuality2Error(f"{label} is too short")
        if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
            raise AgenticQuality2Error(f"{label} is too long")
    if isinstance(value, Mapping):
        properties = schema.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        required = schema.get("required")
        if isinstance(required, list):
            missing = [key for key in required if key not in value]
            if missing:
                raise AgenticQuality2Error(f"{label} missing required fields {missing}")
        if schema.get("additionalProperties") is False:
            extra = [str(key) for key in value if key not in properties]
            if extra:
                raise AgenticQuality2Error(f"{label} has extra fields {extra}")
        for key, item in value.items():
            subschema = properties.get(key)
            if isinstance(subschema, Mapping):
                _validate_value(item, subschema, label=f"{label}.{key}")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            raise AgenticQuality2Error(f"{label} has too few items")
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                _validate_value(item, items, label=f"{label}[{index}]")


def _visible_prompt_text(suite: Mapping[str, Any]) -> str:
    repository = _mapping(suite.get("repository_context"), label="repository_context")
    pieces = [str(repository.get("base", ""))]
    pieces.extend(str(value) for value in repository.get("expansion_blocks", ()))
    pieces.extend(str(tool.get("description", "")) for tool in suite.get("tools", ()))
    for workload in suite.get("workloads", ()):
        for turn in workload.get("turns", ()):
            pieces.append(str(turn.get("user", "")))
    return "\n".join(pieces)


def _validate_no_leakage(suite: Mapping[str, Any], oracle: Mapping[str, Any]) -> None:
    visible = _visible_prompt_text(suite)
    cases = _mapping(oracle.get("cases"), label="oracle.cases")
    patches = _mapping(oracle.get("patches"), label="oracle.patches")
    hidden_serializations: list[str] = []
    for case in cases.values():
        if isinstance(case, Mapping):
            hidden_serializations.append(str(case.get("expected_result_sha256", "")))
            reference = case.get("reference_arguments")
            if reference is not None:
                hidden_serializations.append(
                    json.dumps(reference, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                )
    for patch in patches.values():
        if isinstance(patch, Mapping):
            hidden_serializations.extend((str(patch.get("old", "")), str(patch.get("new", ""))))
    for case in _mapping(oracle.get("code_cases"), label="oracle.code_cases").values():
        if not isinstance(case, Mapping):
            continue
        for hidden in case.get("hidden_tests", ()):
            hidden_serializations.append(
                json.dumps(hidden, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            )
    if any(secret and secret in visible for secret in hidden_serializations):
        raise AgenticQuality2Error("expected-answer leakage appears in user-visible prompts")


def load_agentic_quality2_suite(path: str | Path) -> AgenticQuality2Suite:
    """Load and cross-validate the frozen expanded suite and referenced files."""

    suite_path = Path(path).resolve()
    suite = _load_object(suite_path, label="suite")
    if suite.get("kind") != "hipengine.agentic_quality2_suite":
        raise AgenticQuality2Error("suite kind is unsupported")
    if suite.get("schema_version") != 1 or suite.get("suite") not in {
        "agentic-quality2-v1",
        "agentic-quality2-v2",
    }:
        raise AgenticQuality2Error("suite identity is unsupported")
    oracle_path, oracle_hash = _resolve_ref(
        suite_path.parent,
        suite.get("quality_oracle"),
        label="quality_oracle",
    )
    sources_path, sources_hash = _resolve_ref(
        suite_path.parent,
        suite.get("source_manifest"),
        label="source_manifest",
    )
    oracle = _load_object(oracle_path, label="oracle")
    sources = _load_object(sources_path, label="sources")
    if oracle.get("kind") != "hipengine.agentic_quality2_oracles":
        raise AgenticQuality2Error("oracle kind is unsupported")
    if oracle.get("suite") != suite.get("suite"):
        raise AgenticQuality2Error("oracle suite mismatch")
    oracle_source_path, oracle_source_hash = _resolve_ref(
        oracle_path.parent,
        oracle.get("source_manifest"),
        label="oracle.source_manifest",
    )
    if oracle_source_path != sources_path or oracle_source_hash != sources_hash:
        raise AgenticQuality2Error("oracle source manifest differs from suite")
    if sources.get("kind") != "hipengine.agentic_quality2_sources":
        raise AgenticQuality2Error("source manifest kind is unsupported")
    selection = _mapping(sources.get("selection"), label="sources.selection")
    if selection.get("upstream_bytes_imported") is not False:
        raise AgenticQuality2Error("source manifest must reject upstream-byte import")
    if selection.get("official_score_claimed") is not False:
        raise AgenticQuality2Error("source manifest must reject official score claims")

    tools: dict[str, Mapping[str, Any]] = {}
    for index, raw_tool in enumerate(_sequence(suite.get("tools"), label="tools")):
        tool = _mapping(raw_tool, label=f"tools[{index}]")
        name = _nonempty(tool.get("name"), label=f"tools[{index}].name")
        if name in tools:
            raise AgenticQuality2Error(f"duplicate tool {name!r}")
        if tool.get("strict") is not True:
            raise AgenticQuality2Error(f"tool {name!r} is not strict")
        tools[name] = copy.deepcopy(dict(tool))

    workloads: dict[str, Mapping[str, Any]] = {}
    cases = _mapping(oracle.get("cases"), label="oracle.cases")
    for index, raw_workload in enumerate(_sequence(suite.get("workloads"), label="workloads")):
        workload = _mapping(raw_workload, label=f"workloads[{index}]")
        workload_id = _nonempty(workload.get("id"), label=f"workloads[{index}].id")
        if workload_id in workloads:
            raise AgenticQuality2Error(f"duplicate workload {workload_id!r}")
        split = workload.get("split")
        if split not in {"development", "heldout"}:
            raise AgenticQuality2Error(f"workload {workload_id!r} has invalid split")
        if workload.get("language") not in {"en", "ja", "mixed_ja_en"}:
            raise AgenticQuality2Error(f"workload {workload_id!r} language is missing/invalid")
        turns = _sequence(workload.get("turns"), label=f"{workload_id}.turns")
        if len(turns) != 1:
            raise AgenticQuality2Error(f"workload {workload_id!r} must contain one turn")
        turn = _mapping(turns[0], label=f"{workload_id}.turn")
        case_id = _nonempty(turn.get("oracle_case"), label=f"{workload_id}.oracle_case")
        case = cases.get(case_id)
        if not isinstance(case, Mapping):
            raise AgenticQuality2Error(f"workload {workload_id!r} has unknown oracle case")
        if case.get("workload_id") != workload_id:
            raise AgenticQuality2Error(f"workload {workload_id!r} oracle id mismatch")
        if case.get("split") != split:
            raise AgenticQuality2Error(f"workload {workload_id!r} oracle split mismatch")
        expected_outcome = turn.get("expected_outcome")
        if expected_outcome == "tool_call":
            tool_name = turn.get("expected_tool")
            if tool_name not in tools:
                raise AgenticQuality2Error(f"workload {workload_id!r} has unknown tool")
            arguments = turn.get("expected_arguments")
            if arguments is not None:
                _validate_value(
                    arguments,
                    _mapping(tools[str(tool_name)].get("parameters"), label="tool schema"),
                    label=f"{workload_id}.expected_arguments",
                )
        elif expected_outcome == "tool_calls":
            if turn.get("parallel_tool_calls") is not True:
                raise AgenticQuality2Error(f"workload {workload_id!r} must enable parallel calls")
            calls = _sequence(turn.get("expected_calls"), label=f"{workload_id}.calls")
            if len(calls) < 2:
                raise AgenticQuality2Error(f"workload {workload_id!r} has malformed call count")
            for raw_call in calls:
                call = _mapping(raw_call, label=f"{workload_id}.call")
                tool_name = call.get("tool")
                if tool_name not in tools:
                    raise AgenticQuality2Error(f"workload {workload_id!r} call has unknown tool")
                _validate_value(
                    call.get("arguments"),
                    _mapping(tools[str(tool_name)].get("parameters"), label="tool schema"),
                    label=f"{workload_id}.call.arguments",
                )
        elif expected_outcome != "no_tool_call":
            raise AgenticQuality2Error(f"workload {workload_id!r} outcome is unsupported")
        workloads[workload_id] = copy.deepcopy(dict(workload))

    if set(cases) != set(workloads):
        raise AgenticQuality2Error("oracle/workload case membership mismatch")
    policy = _mapping(suite.get("split_policy"), label="split_policy")
    development = tuple(
        str(value) for value in _sequence(policy.get("development_ids"), label="development_ids")
    )
    heldout = tuple(
        str(value) for value in _sequence(policy.get("heldout_ids"), label="heldout_ids")
    )
    if len(development) != len(set(development)) or len(heldout) != len(set(heldout)):
        raise AgenticQuality2Error("split manifest contains duplicate ids")
    overlap = set(development) & set(heldout)
    if overlap:
        raise AgenticQuality2Error(f"split overlap: {sorted(overlap)}")
    expected_development = {
        workload_id
        for workload_id, workload in workloads.items()
        if workload["split"] == "development"
    }
    expected_heldout = set(workloads) - expected_development
    if set(development) != expected_development or set(heldout) != expected_heldout:
        raise AgenticQuality2Error("split manifest differs from workload membership")
    if len(development) != 17 or len(heldout) != 17:
        raise AgenticQuality2Error("expanded split count must be 17 development / 17 heldout")
    family_counts = {
        (family, split): sum(
            workload["family"] == family and workload["split"] == split
            for workload in workloads.values()
        )
        for family in ("tool_selection", "repository", "code", "instruction")
        for split in ("development", "heldout")
    }
    required_family_counts = {
        ("tool_selection", "development"): 5,
        ("tool_selection", "heldout"): 5,
        **{
            (family, split): 4
            for family in ("repository", "code", "instruction")
            for split in ("development", "heldout")
        },
    }
    if family_counts != required_family_counts:
        raise AgenticQuality2Error("expanded family/split counts are malformed")
    if (
        sum(
            workload["split"] == "heldout" and workload["language"] != "en"
            for workload in workloads.values()
        )
        < 4
    ):
        raise AgenticQuality2Error("expanded suite lacks Japanese/mixed heldout coverage")
    if (
        sum(
            workload["split"] == "heldout" and workload["task_kind"] in {"patch", "code"}
            for workload in workloads.values()
        )
        < 4
    ):
        raise AgenticQuality2Error("expanded suite lacks heldout patch/code coverage")

    code_cases = _mapping(oracle.get("code_cases"), label="oracle.code_cases")
    instruction_cases = _mapping(
        oracle.get("instruction_cases"),
        label="oracle.instruction_cases",
    )
    patches = _mapping(oracle.get("patches"), label="oracle.patches")
    for workload_id, workload in workloads.items():
        turn = workload["turns"][0]
        case = _mapping(cases[turn["oracle_case"]], label=f"case {workload_id}")
        expected_outcome = turn["expected_outcome"]
        if expected_outcome == "no_tool_call":
            expected_kind = "no_tool"
        elif expected_outcome == "tool_calls":
            expected_kind = "multiple"
        else:
            expected_kind = _TOOL_KIND[str(turn["expected_tool"])]
        if case.get("kind") != expected_kind:
            raise AgenticQuality2Error(f"workload {workload_id!r} oracle kind mismatch")
        if turn.get("expected_arguments") is not None and case.get(
            "reference_arguments"
        ) != turn.get("expected_arguments"):
            raise AgenticQuality2Error(f"workload {workload_id!r} reference arguments mismatch")
        if expected_kind == "multiple" and case.get("reference_calls") != turn.get(
            "expected_calls"
        ):
            raise AgenticQuality2Error(f"workload {workload_id!r} reference calls mismatch")
        if expected_kind == "code" and case.get("code_case") not in code_cases:
            raise AgenticQuality2Error(f"workload {workload_id!r} code case is unknown")
        if expected_kind == "instruction" and case.get("instruction_case") not in instruction_cases:
            raise AgenticQuality2Error(f"workload {workload_id!r} instruction case is unknown")
        if any(str(patch_id) not in patches for patch_id in case.get("setup_patches", ())):
            raise AgenticQuality2Error(f"workload {workload_id!r} setup patch is unknown")
    for code_id, raw_code in code_cases.items():
        code = _mapping(raw_code, label=f"code case {code_id}")
        hidden = _sequence(code.get("hidden_tests"), label=f"code case {code_id}.hidden_tests")
        if len(hidden) < 4:
            raise AgenticQuality2Error(f"code case {code_id!r} has malformed counts")
        if not isinstance(code.get("allowed_imports"), list):
            raise AgenticQuality2Error(f"code case {code_id!r} allowed_imports is invalid")
    controls = tuple(
        _mapping(row, label="fail-safe control")
        for row in _sequence(
            oracle.get("fail_safe_controls"),
            label="fail_safe_controls",
        )
    )
    control_ids = [str(row.get("id", "")) for row in controls]
    if len(control_ids) != len(set(control_ids)) or len(control_ids) < 8:
        raise AgenticQuality2Error("fail-safe controls are duplicated or incomplete")
    required_controls = {
        "malformed",
        "truncated",
        "duplicate",
        "undeclared",
        "schema_invalid",
        "content_leak",
        "reasoning_leak",
        "required_missing",
        "ambiguous",
        "tool_none_violation",
    }
    if {str(row["class"]) for row in controls} != required_controls:
        raise AgenticQuality2Error("fail-safe control classes are incomplete")
    source_audit = tuple(
        _mapping(row, label="source audit row")
        for row in _sequence(sources.get("public_source_audit"), label="source audit")
    )
    if {str(row.get("id")) for row in source_audit} != {
        "bfcl",
        "human_eval",
        "mbpp",
        "ifeval",
    }:
        raise AgenticQuality2Error("source audit is incomplete")
    if any(
        row.get("use") != "conceptual_style_only_no_copied_tasks_solutions_or_tests"
        for row in source_audit
    ):
        raise AgenticQuality2Error("source audit permits undeclared upstream use")
    _validate_no_leakage(suite, oracle)

    return AgenticQuality2Suite(
        path=suite_path,
        payload=copy.deepcopy(suite),
        file_sha256=_sha256(suite_path),
        oracle_path=oracle_path,
        oracle=copy.deepcopy(oracle),
        oracle_sha256=oracle_hash,
        sources_path=sources_path,
        sources=copy.deepcopy(sources),
        sources_sha256=sources_hash,
        tools=tools,
        workloads=workloads,
        development_ids=development,
        heldout_ids=heldout,
    )


def _fraction(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Expression):
        return _fraction(node.body)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return Fraction(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left, right = _fraction(node.left), _fraction(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise AgenticQuality2Error("division by zero")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_fraction(node.operand))
    raise AgenticQuality2Error("calculator expression is unsupported")


def _calculate(expression: str) -> int | str:
    try:
        value = _fraction(ast.parse(str(expression), mode="eval"))
    except SyntaxError as exc:
        raise AgenticQuality2Error("calculator expression is invalid") from exc
    return value.numerator if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _apply_patch(files: dict[str, str], patches: Mapping[str, Any], patch_id: str) -> None:
    patch = _mapping(patches.get(patch_id), label=f"patch {patch_id}")
    path, old, new = str(patch["path"]), str(patch["old"]), str(patch["new"])
    if files[path].count(old) != 1:
        raise AgenticQuality2Error(f"patch {patch_id!r} no longer has one region")
    files[path] = files[path].replace(old, new, 1)


def _run_suite(
    files: Mapping[str, str], suites: Mapping[str, Any], suite_id: str
) -> tuple[bool, int]:
    required = _mapping(
        _mapping(suites.get(suite_id), label=f"test suite {suite_id}").get("required_file_sha256"),
        label=f"test suite {suite_id}.required_file_sha256",
    )
    passed = all(
        hashlib.sha256(files[str(path)].encode("utf-8")).hexdigest() == digest
        for path, digest in required.items()
    )
    return passed, len(required)


def _reference_action(
    suite: AgenticQuality2Suite,
    *,
    kind: str,
    arguments: Mapping[str, Any],
    setup_patches: Sequence[str] = (),
) -> dict[str, Any]:
    oracle = suite.oracle
    files = copy.deepcopy(dict(_mapping(oracle["files"], label="oracle.files")))
    patches = _mapping(oracle["patches"], label="oracle.patches")
    test_suites = _mapping(oracle["test_suites"], label="oracle.test_suites")
    for patch_id in setup_patches:
        _apply_patch(files, patches, str(patch_id))
    if kind == "read":
        path, mode = str(arguments["path"]), str(arguments["mode"])
        content = (
            str(_mapping(oracle["summaries"], label="oracle.summaries")[path])
            if mode == "summary"
            else "\n".join(files[path].splitlines()[: int(arguments.get("line_limit", 999))]) + "\n"
        )
        return {"path": path, "mode": mode, "content": content}
    if kind == "search":
        prefix, query = str(arguments["path"]), str(arguments["query"])
        case_sensitive = bool(arguments.get("case_sensitive", True))
        matches: list[dict[str, Any]] = []
        for path in sorted(files):
            if path != prefix and not path.startswith(prefix.rstrip("/") + "/"):
                continue
            for line_number, line in enumerate(files[path].splitlines(), start=1):
                haystack = line if case_sensitive else line.lower()
                needle = query if case_sensitive else query.lower()
                if needle in haystack:
                    matches.append({"path": path, "line": line_number, "text": line})
        return {
            "path": prefix,
            "query": query,
            "case_sensitive": case_sensitive,
            "matches": matches,
        }
    if kind == "lookup":
        key, locale = str(arguments["key"]), str(arguments["locale"])
        return {
            "key": key,
            "locale": locale,
            "value": str(_mapping(oracle["lookups"], label="oracle.lookups")[key]),
        }
    if kind == "calculate":
        return {"value": _calculate(str(arguments["expression"]))}
    if kind == "transform":
        record = _mapping(arguments["record"], label="transform.record")
        tags = sorted(dict.fromkeys(str(value).strip().lower() for value in record["tags"]))
        result: dict[str, Any] = {
            "station": str(record["station"]).strip().lower(),
            "priority": str(record["priority"]),
            "tags": tags,
            "mode": str(arguments["mode"]),
        }
        if arguments.get("include_metadata") is True:
            result["metadata"] = {"tag_count": len(tags)}
        return result
    if kind == "patch":
        patch_id = str(arguments["patch_id"])
        patch = _mapping(patches[patch_id], label=f"patch {patch_id}")
        _apply_patch(files, patches, patch_id)
        tests_passed, checks = _run_suite(files, test_suites, str(patch["test_suite"]))
        path = str(patch["path"])
        return {
            "patch_id": patch_id,
            "path": path,
            "applied": True,
            "result_sha256": hashlib.sha256(files[path].encode("utf-8")).hexdigest(),
            "test_suite": str(patch["test_suite"]),
            "tests_passed": tests_passed,
            "checks": checks,
        }
    if kind == "test":
        suite_id = str(arguments["suite"])
        tests_passed, checks = _run_suite(files, test_suites, suite_id)
        return {"suite": suite_id, "tests_passed": tests_passed, "checks": checks}
    raise AgenticQuality2Error(f"reference action kind {kind!r} is unsupported")


def execute_reference_case(suite: AgenticQuality2Suite, case_id: str) -> dict[str, Any]:
    """Execute one committed reference behavior without model output."""

    case = _mapping(suite.oracle["cases"].get(str(case_id)), label=f"case {case_id}")
    kind = str(case["kind"])
    if kind in {"read", "search", "lookup", "calculate", "transform", "patch", "test"}:
        result = _reference_action(
            suite,
            kind=kind,
            arguments=_mapping(case.get("reference_arguments"), label=f"case {case_id}.arguments"),
            setup_patches=tuple(str(value) for value in case.get("setup_patches", ())),
        )
    elif kind == "multiple":
        results: list[dict[str, Any]] = []
        for raw_call in _sequence(case.get("reference_calls"), label=f"case {case_id}.calls"):
            call = _mapping(raw_call, label=f"case {case_id}.call")
            tool = str(call["tool"])
            results.append(
                {
                    "tool": tool,
                    "result": _reference_action(
                        suite,
                        kind=_TOOL_KIND[tool],
                        arguments=_mapping(
                            call["arguments"], label=f"case {case_id}.call.arguments"
                        ),
                    ),
                }
            )
        result = {"calls": results}
    elif kind == "code":
        code_case = _mapping(
            suite.oracle["code_cases"].get(str(case["code_case"])),
            label=f"case {case_id}.code_case",
        )
        result = {
            "entry_point": str(code_case["entry_point"]),
            "passed": True,
            "tests": len(code_case["hidden_tests"]),
        }
    elif kind == "instruction":
        instruction = _mapping(
            suite.oracle["instruction_cases"].get(str(case["instruction_case"])),
            label=f"case {case_id}.instruction_case",
        )
        result = {"passed": True, "checks": len(instruction["checks"])}
    elif kind == "no_tool":
        result = {"passed": True, "tool_call_count": 0, "public_text_required": True}
    else:
        raise AgenticQuality2Error(f"case {case_id!r} kind is unsupported")
    actual = _canonical_sha256(result)
    expected = str(case["expected_result_sha256"])
    return {
        "case_id": str(case_id),
        "kind": kind,
        "passed": actual == expected,
        "result_sha256": actual,
        "expected_result_sha256": expected,
    }


def _instruction_check(text: str, raw_check: Mapping[str, Any]) -> bool:
    check_type = str(raw_check["type"])
    lines = str(text).splitlines()
    words = str(text).split()
    if check_type == "line_count":
        return len(lines) == int(raw_check["value"])
    if check_type == "bullet_prefix":
        return bool(lines) and all(line.startswith(str(raw_check["value"])) for line in lines)
    if check_type == "numbered_prefix":
        start = int(raw_check.get("value", 1))
        return all(line.startswith(f"{start + index}. ") for index, line in enumerate(lines))
    if check_type == "required_terms":
        return all(str(term) in text for term in raw_check.get("terms", ()))
    if check_type == "forbidden_terms":
        return all(str(term) not in text for term in raw_check.get("terms", ()))
    if check_type == "word_count":
        minimum = raw_check.get("minimum")
        maximum = raw_check.get("maximum")
        return (minimum is None or len(words) >= int(minimum)) and (
            maximum is None or len(words) <= int(maximum)
        )
    if check_type == "line_suffix":
        return bool(lines) and all(line.endswith(str(raw_check["value"])) for line in lines)
    if check_type == "line_prefixes":
        prefixes = tuple(str(value) for value in raw_check.get("prefixes", ()))
        return len(lines) == len(prefixes) and all(
            line.startswith(prefix) for line, prefix in zip(lines, prefixes, strict=True)
        )
    if check_type == "json_keys":
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return False
        return isinstance(value, Mapping) and set(value) == set(raw_check.get("terms", ()))
    if check_type == "single_sentence":
        return len(lines) == 1 and len(re.findall(r"[.!?。]", text)) == 1
    raise AgenticQuality2Error(f"instruction check {check_type!r} is unsupported")


def evaluate_quality2_oracle(
    suite: AgenticQuality2Suite,
    *,
    workload_id: str,
    calls: Sequence[Mapping[str, Any]],
    public_text: str = "",
    sandbox: Any | None = None,
) -> dict[str, Any]:
    """Evaluate model-selected calls against one frozen expanded task oracle."""

    workload = _mapping(suite.workloads.get(str(workload_id)), label="workload")
    turn = _mapping(workload["turns"][0], label="workload.turn")
    case = _mapping(suite.oracle["cases"].get(turn["oracle_case"]), label="case")
    normalized_calls = [
        {
            "tool": str(_mapping(call, label="call").get("tool", "")),
            "arguments": _mapping(call.get("arguments"), label="call.arguments"),
        }
        for call in calls
    ]
    status = "failed"
    error: str | None = None
    result: dict[str, Any] | None = None
    sandbox_result: dict[str, Any] | None = None
    try:
        expected_outcome = str(turn["expected_outcome"])
        if expected_outcome == "no_tool_call":
            if normalized_calls:
                raise AgenticQuality2Error("no-tool task selected a tool")
            result = {
                "passed": bool(str(public_text).strip()),
                "tool_call_count": 0,
                "public_text_required": True,
            }
        elif expected_outcome == "tool_calls":
            reference_calls = _sequence(case.get("reference_calls"), label="reference_calls")
            if len(normalized_calls) != len(reference_calls):
                raise AgenticQuality2Error("multiple-call task has wrong call count")
            remaining = list(normalized_calls)
            results: list[dict[str, Any]] = []
            for raw_reference in reference_calls:
                reference = _mapping(raw_reference, label="reference_call")
                tool = str(reference["tool"])
                matches = [row for row in remaining if row["tool"] == tool]
                if len(matches) != 1:
                    raise AgenticQuality2Error("multiple-call task has wrong tools")
                selected = matches[0]
                remaining.remove(selected)
                _validate_value(
                    selected["arguments"],
                    _mapping(suite.tools[tool]["parameters"], label="tool schema"),
                    label=f"{workload_id}.{tool}.arguments",
                )
                results.append(
                    {
                        "tool": tool,
                        "result": _reference_action(
                            suite,
                            kind=_TOOL_KIND[tool],
                            arguments=selected["arguments"],
                        ),
                    }
                )
            result = {"calls": results}
        else:
            if len(normalized_calls) != 1:
                raise AgenticQuality2Error("single-call task has wrong call count")
            selected = normalized_calls[0]
            expected_tool = str(turn["expected_tool"])
            if selected["tool"] != expected_tool:
                raise AgenticQuality2Error("single-call task selected wrong tool")
            _validate_value(
                selected["arguments"],
                _mapping(suite.tools[expected_tool]["parameters"], label="tool schema"),
                label=f"{workload_id}.arguments",
            )
            constraints = turn.get("expected_argument_constraints")
            if isinstance(constraints, Mapping):
                properties = _mapping(constraints, label="argument constraints")
                for key, schema in properties.items():
                    _validate_value(
                        selected["arguments"].get(key),
                        _mapping(schema, label=f"constraint {key}"),
                        label=f"{workload_id}.arguments.{key}",
                    )
            kind = str(case["kind"])
            if kind == "code":
                code_case = _mapping(
                    suite.oracle["code_cases"].get(str(case["code_case"])),
                    label="code_case",
                )
                if selected["arguments"].get("entry_point") != code_case["entry_point"]:
                    raise AgenticQuality2Error("code task selected wrong entry point")
                if sandbox is None:
                    status = "blocked_sandbox"
                    sandbox_result = {"status": "blocked_sandbox", "tests_attempted": 0}
                else:
                    sandbox_result = sandbox.run_code_case(
                        source=str(selected["arguments"]["source"]),
                        entry_point=str(code_case["entry_point"]),
                        hidden_tests=tuple(code_case["hidden_tests"]),
                        allowed_imports=tuple(str(value) for value in code_case["allowed_imports"]),
                        scratch_root=Path(tempfile.gettempdir()) / "hipengine-agentic-quality2",
                    )
                    if sandbox_result["status"] == "blocked_sandbox":
                        status = "blocked_sandbox"
                    result = {
                        "entry_point": str(code_case["entry_point"]),
                        "passed": sandbox_result["status"] == "passed",
                        "tests": int(sandbox_result["tests_passed"]),
                    }
            elif kind == "instruction":
                instruction = _mapping(
                    suite.oracle["instruction_cases"].get(str(case["instruction_case"])),
                    label="instruction_case",
                )
                text = str(selected["arguments"]["text"])
                checks = tuple(
                    _mapping(value, label="instruction check") for value in instruction["checks"]
                )
                result = {
                    "passed": all(_instruction_check(text, check) for check in checks),
                    "checks": len(checks),
                }
            else:
                result = _reference_action(
                    suite,
                    kind=kind,
                    arguments=selected["arguments"],
                    setup_patches=tuple(str(value) for value in case.get("setup_patches", ())),
                )
    except (AgenticQuality2Error, KeyError, TypeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    expected_hash = str(case["expected_result_sha256"])
    result_hash = None if result is None else _canonical_sha256(result)
    if status != "blocked_sandbox":
        status = "passed" if result_hash == expected_hash else "failed"
    return {
        "workload_id": str(workload_id),
        "split": str(workload["split"]),
        "family": str(workload["family"]),
        "language": str(workload["language"]),
        "kind": str(case["kind"]),
        "status": status,
        "success": status == "passed",
        "result_sha256": result_hash,
        "expected_result_sha256": expected_hash,
        "error": error,
        "sandbox": sandbox_result,
    }


def evaluate_quality2_fail_safe_control(
    suite: AgenticQuality2Suite,
    control_id: str,
) -> dict[str, Any]:
    """Evaluate one committed negative parser/publication policy control."""

    controls = {
        str(row["id"]): row
        for row in suite.oracle["fail_safe_controls"]
        if isinstance(row, Mapping)
    }
    control = _mapping(controls.get(str(control_id)), label=f"control {control_id}")
    expected = _mapping(control.get("expected"), label=f"control {control_id}.expected")
    raw_text = str(control["raw_text"])
    parsed = parse_independent_tool_envelope(raw_text)
    calls = parsed.get("tool_calls")
    calls = list(calls) if isinstance(calls, list) else []
    mode = str(control["tools_mode"])
    reason: str | None = None
    leak = str(control["class"]) in {"content_leak", "reasoning_leak"}
    if mode == "none" and calls:
        reason = "invalid_tool_call"
    elif mode == "required" and not calls:
        reason = "tool_required_not_satisfied"
    elif leak:
        reason = "invalid_tool_call"
    elif not parsed.get("accepted") or len(calls) != 1:
        reason = "invalid_tool_call"
    else:
        call = _mapping(calls[0], label="control call")
        tool = suite.tools.get(str(call.get("name")))
        if tool is None:
            reason = "invalid_tool_call"
        elif not _arguments_match_schema(
            _mapping(call.get("arguments"), label="control arguments"),
            _mapping(tool["parameters"], label="control tool schema"),
        ):
            reason = "schema_violation"
    observed = {
        "accepted": reason is None,
        "finish_reason": reason or "tool_calls",
        "public_tool_call_count": 0 if reason is not None else len(calls),
        "public_content_empty": True if reason is not None else not bool(parsed.get("content")),
        "session_commit": "prompt_only" if leak else "none",
    }
    return {
        "control_id": str(control_id),
        "class": str(control["class"]),
        "split": str(control["split"]),
        "passed": observed == dict(expected),
        "observed": observed,
        "expected": copy.deepcopy(dict(expected)),
    }


def aggregate_quality2_results(
    suite: AgenticQuality2Suite,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_repetitions: int,
    expected_workload_ids: Sequence[str] | None = None,
    seal_heldout_details: bool = True,
) -> dict[str, Any]:
    """Build deterministic quality totals without hiding blocked/unscorable rows."""

    if expected_repetitions <= 0:
        raise AgenticQuality2Error("expected_repetitions must be positive")
    expected_ids = (
        set(suite.workloads)
        if expected_workload_ids is None
        else {str(value) for value in expected_workload_ids}
    )
    if not expected_ids or not expected_ids <= set(suite.workloads):
        raise AgenticQuality2Error("expected workload selection is empty/unknown")
    if expected_workload_ids is not None and len(expected_ids) != len(expected_workload_ids):
        raise AgenticQuality2Error("expected workload selection contains duplicates")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for raw_row in rows:
        row = _mapping(raw_row, label="result row")
        workload_id = str(row.get("workload_id", ""))
        if workload_id not in suite.workloads:
            raise AgenticQuality2Error(f"result row has unknown workload {workload_id!r}")
        workload = suite.workloads[workload_id]
        expected_case = suite.oracle["cases"][workload["turns"][0]["oracle_case"]]
        if (
            row.get("split") != workload["split"]
            or row.get("family") != workload["family"]
            or row.get("language") != workload["language"]
            or row.get("kind") != expected_case["kind"]
        ):
            raise AgenticQuality2Error(
                "result row split/family/language/kind differs from frozen suite"
            )
        status = row.get("status")
        if status not in {"passed", "failed", "blocked_sandbox", "unscorable"}:
            raise AgenticQuality2Error("result row status is unsupported")
        if row.get("success") is not (status == "passed"):
            raise AgenticQuality2Error("result row success/status is inconsistent")
        repetition = row.get("repetition")
        if not isinstance(repetition, int) or isinstance(repetition, bool) or repetition < 0:
            raise AgenticQuality2Error("result row repetition is invalid")
        grouped.setdefault(workload_id, []).append(row)
    if set(grouped) != expected_ids:
        raise AgenticQuality2Error("result rows do not cover expected workloads")
    if any(len(values) != expected_repetitions for values in grouped.values()):
        raise AgenticQuality2Error("result repetitions are incomplete")
    if any(
        {int(row["repetition"]) for row in values} != set(range(expected_repetitions))
        for values in grouped.values()
    ):
        raise AgenticQuality2Error("result repetition identities are duplicated/incomplete")

    def rollup(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        statuses = {
            name: sum(row["status"] == name for row in subset)
            for name in (
                "passed",
                "failed",
                "blocked_sandbox",
                "unscorable",
            )
        }
        scored = statuses["passed"] + statuses["failed"]
        return {
            "observations": len(subset),
            **statuses,
            "scored_denominator": scored,
            "success_rate": 0.0 if scored == 0 else statuses["passed"] / scored,
        }

    ordered = [
        row
        for workload_id in sorted(grouped)
        for row in sorted(
            grouped[workload_id],
            key=lambda value: int(value["repetition"]),
        )
    ]
    by_split = {
        split: rollup([row for row in ordered if row["split"] == split])
        for split in ("development", "heldout")
    }
    by_family = {
        family: rollup([row for row in ordered if row["family"] == family])
        for family in sorted({str(row["family"]) for row in ordered})
    }
    by_language = {
        language: rollup([row for row in ordered if row["language"] == language])
        for language in sorted({str(row["language"]) for row in ordered})
    }
    by_kind = {
        kind: rollup([row for row in ordered if row["kind"] == kind])
        for kind in sorted({str(row["kind"]) for row in ordered})
    }
    normalized_fingerprints = [row.get("normalized_response_sha256") for row in ordered]
    if any(value is not None for value in normalized_fingerprints) and not all(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
        for value in normalized_fingerprints
    ):
        raise AgenticQuality2Error(
            "normalized response fingerprints must be complete lowercase SHA-256 values"
        )
    normalized_basis = bool(normalized_fingerprints) and all(
        isinstance(value, str) for value in normalized_fingerprints
    )
    determinism: list[dict[str, Any]] = []
    for workload_id, values in sorted(grouped.items()):
        fingerprints = (
            {str(row["normalized_response_sha256"]) for row in values}
            if normalized_basis
            else {
                _canonical_sha256(
                    {
                        "status": row["status"],
                        "success": row["success"],
                        "result_sha256": row.get("result_sha256"),
                        "error": row.get("error"),
                    }
                )
                for row in values
            }
        )
        if len(fingerprints) != 1:
            determinism.append({"workload_id": workload_id, "fingerprints": sorted(fingerprints)})
    development_details = [
        {
            "workload_id": row["workload_id"],
            "status": row["status"],
            "success": row["success"],
            "result_sha256": row.get("result_sha256"),
        }
        for row in ordered
        if row["split"] == "development"
    ]
    return {
        "performance_claim": False,
        "overall": rollup(ordered),
        "by_split": by_split,
        "by_family": by_family,
        "by_language": by_language,
        "by_kind": by_kind,
        "determinism": {
            "basis": (
                "normalized_response_v1" if normalized_basis else "oracle_outcome_v1"
            ),
            "evaluated": expected_repetitions > 1,
            "passed": not determinism,
            "mismatches": determinism,
        },
        "development_details": development_details,
        "heldout_details": []
        if seal_heldout_details
        else [
            {
                "workload_id": row["workload_id"],
                "status": row["status"],
                "success": row["success"],
                "result_sha256": row.get("result_sha256"),
            }
            for row in ordered
            if row["split"] == "heldout"
        ],
        "heldout_details_sealed": bool(seal_heldout_details),
    }


__all__ = [
    "AgenticQuality2Error",
    "AgenticQuality2Suite",
    "aggregate_quality2_results",
    "evaluate_quality2_fail_safe_control",
    "evaluate_quality2_oracle",
    "execute_reference_case",
    "load_agentic_quality2_suite",
]
