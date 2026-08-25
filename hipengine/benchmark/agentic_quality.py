"""Fail-closed natural tool-quality records for the coding-agent benchmark.

This module is intentionally separate from :mod:`hipengine.benchmark.agentic`.
Natural model failures are valid quality observations here, but they can never
enter the deterministic latency/goodput denominator or make a performance
claim.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from hipengine.benchmark.agentic import (
    AGENTIC_SCHEMA_VERSION,
    AgenticBenchmarkError,
    AgenticWorkloadSuite,
)
from hipengine.benchmark.agentic_quality_oracle import evaluate_quality_oracle
from hipengine.benchmark.provenance import validate_artifact_provenance
from hipengine.tokenization.identity import token_ids_sha256

AGENTIC_QUALITY_RECORDS_KIND = "hipengine_agentic_coding_quality_records"
AGENTIC_QUALITY_ARTIFACT_KIND = "hipengine_agentic_coding_quality_benchmark"
_RAW_MARKERS = ("<think", "</think", "<tool_call", "</tool_call")
_ZERO_OWNERSHIP_FIELDS = (
    "pending_requests",
    "active_requests",
    "stream_producers",
    "model_active_requests",
    "session_count",
    "kv_refcounted_pages",
    "kv_pinned_pages",
    "graph_owners",
    "workspace_owners",
)
_QUALITY_OUTCOMES = frozenset(
    {
        "passed",
        "invalid_tool_call",
        "no_tool_call",
        "multiple_tool_calls",
        "invalid_arguments",
        "schema_violation",
        "wrong_tool",
        "wrong_arguments",
        "content_alongside_tool_call",
        "raw_markup_leak",
        "finish_mismatch",
        "oracle_failed",
    }
)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AgenticBenchmarkError(f"{label} must be an object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not _is_sequence(value):
        raise AgenticBenchmarkError(f"{label} must be an array")
    return value


def _nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgenticBenchmarkError(f"{label} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AgenticBenchmarkError(f"{label} must be a non-negative integer")
    return int(value)


def _positive_int(value: Any, *, label: str) -> int:
    result = _nonnegative_int(value, label=label)
    if result == 0:
        raise AgenticBenchmarkError(f"{label} must be a positive integer")
    return result


def _token_row(value: Any, *, label: str) -> tuple[int, ...]:
    raw = _sequence(value, label=label)
    tokens: list[int] = []
    for index, token in enumerate(raw):
        if not isinstance(token, int) or isinstance(token, bool) or token < 0:
            raise AgenticBenchmarkError(f"{label}[{index}] must be a non-negative integer")
        tokens.append(int(token))
    if not tokens:
        raise AgenticBenchmarkError(f"{label} must not be empty")
    return tuple(tokens)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise AgenticBenchmarkError(f"{label} must be a lowercase SHA-256 string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise AgenticBenchmarkError(f"{label} must be a lowercase SHA-256 string") from exc
    return value


def _schema_type_matches(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return _is_sequence(value)
    if expected == "null":
        return value is None
    return False


def _arguments_match_schema(value: Any, schema: Mapping[str, Any]) -> bool:
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _schema_type_matches(value, expected_type):
        return False
    enum = schema.get("enum")
    if _is_sequence(enum) and value not in enum:
        return False
    if isinstance(value, str):
        minimum = schema.get("minLength")
        if isinstance(minimum, int) and len(value) < minimum:
            return False
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        properties = properties if isinstance(properties, Mapping) else {}
        required = schema.get("required", ())
        if _is_sequence(required) and any(key not in value for key in required):
            return False
        if schema.get("additionalProperties") is False and any(
            key not in properties for key in value
        ):
            return False
        for key, item in value.items():
            subschema = properties.get(key)
            if isinstance(subschema, Mapping) and not _arguments_match_schema(item, subschema):
                return False
    return True


def _quality_score(
    suite: AgenticWorkloadSuite,
    *,
    workload_id: str,
    turn_index: int,
    finish_reason: str,
    detail_reason: str | None,
    public_content: str,
    raw_markup_leaked: bool,
    tool_call_count: int,
    call_id: str | None,
    selected_tool: str | None,
    arguments: Mapping[str, Any] | None,
    arguments_json_valid: bool,
    repair_count: int,
    external_oracle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected = suite.workloads[workload_id]["turns"][turn_index]
    expected_tool = str(expected["expected_tool"])
    correct_tool = selected_tool == expected_tool
    schema_valid = bool(
        arguments_json_valid
        and selected_tool in suite.tools
        and arguments is not None
        and _arguments_match_schema(arguments, suite.tools[str(selected_tool)]["parameters"])
    )
    exact_arguments = bool(correct_tool and arguments == expected["expected_arguments"])
    valid_call = bool(
        tool_call_count == 1
        and call_id
        and selected_tool
        and arguments_json_valid
        and schema_valid
    )

    if raw_markup_leaked:
        outcome = "raw_markup_leak"
    elif public_content.strip() and tool_call_count:
        outcome = "content_alongside_tool_call"
    elif tool_call_count == 0 and detail_reason == "invalid_tool_call":
        outcome = "invalid_tool_call"
    elif tool_call_count == 0:
        outcome = "no_tool_call"
    elif tool_call_count != 1:
        outcome = "multiple_tool_calls"
    elif not arguments_json_valid:
        outcome = "invalid_arguments"
    elif not schema_valid:
        outcome = "schema_violation"
    elif not correct_tool:
        outcome = "wrong_tool"
    elif external_oracle is not None and external_oracle.get("passed") is not True:
        outcome = "oracle_failed"
    elif external_oracle is None and not exact_arguments:
        # Legacy deterministic fixtures use exact arguments as their oracle.
        # Broad quality suites execute arguments against a separate oracle, so
        # equivalent arguments are valid task successes and exactness remains
        # diagnostic only.
        outcome = "wrong_arguments"
    elif finish_reason != "tool_calls":
        outcome = "finish_mismatch"
    else:
        outcome = "passed"
    success = outcome == "passed"
    score = {
        "expected_tool": expected_tool,
        "selected_tool": selected_tool,
        "tool_call_count": int(tool_call_count),
        "call_id": call_id,
        "arguments": None if arguments is None else copy.deepcopy(dict(arguments)),
        "arguments_json_valid": bool(arguments_json_valid),
        "schema_valid": schema_valid,
        "correct_tool": correct_tool,
        "exact_arguments": exact_arguments,
        "valid_call": valid_call,
        "success": success,
        "outcome": outcome,
        "repair_count": int(repair_count),
    }
    if external_oracle is not None:
        score["external_oracle"] = copy.deepcopy(dict(external_oracle))
    return score


def normalize_chat_quality_turn(
    suite: AgenticWorkloadSuite,
    *,
    workload_id: str,
    turn_index: int,
    run_id: str,
    agent_id: str,
    session_id: str,
    request_id: str,
    prompt_token_ids: Sequence[int],
    payload: Mapping[str, Any],
    repair_count: int = 0,
) -> dict[str, Any]:
    """Normalize one natural blocking response without rejecting model-quality failures."""

    if workload_id not in suite.workloads:
        raise AgenticBenchmarkError(f"unknown workload_id {workload_id!r}")
    turns = suite.workloads[workload_id]["turns"]
    if turn_index < 0 or turn_index >= len(turns):
        raise AgenticBenchmarkError("turn_index is out of range")
    prompt_ids = _token_row(prompt_token_ids, label="prompt_token_ids")
    repairs = _nonnegative_int(repair_count, label="repair_count")
    choices = _sequence(payload.get("choices"), label="quality response choices")
    if len(choices) != 1:
        raise AgenticBenchmarkError("quality response must contain exactly one choice")
    choice = _mapping(choices[0], label="quality response choice")
    finish_reason = _nonempty(choice.get("finish_reason"), label="quality finish_reason")
    finish_details = choice.get("finish_details")
    finish_details = finish_details if isinstance(finish_details, Mapping) else {}
    raw_detail_reason = finish_details.get("reason")
    detail_reason = None if raw_detail_reason is None else str(raw_detail_reason)
    message = _mapping(choice.get("message"), label="quality response message")
    content_value = message.get("content")
    if content_value is not None and not isinstance(content_value, str):
        raise AgenticBenchmarkError("quality response message.content must be a string or null")
    public_content = str(content_value or "")
    raw_markup_leaked = any(marker in public_content for marker in _RAW_MARKERS)

    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        calls: Sequence[Any] = ()
    else:
        calls = _sequence(raw_calls, label="quality response tool_calls")
    call_id: str | None = None
    selected_tool: str | None = None
    arguments: Mapping[str, Any] | None = None
    arguments_json_valid = False
    if len(calls) == 1:
        call = _mapping(calls[0], label="quality response tool_call")
        if isinstance(call.get("id"), str) and str(call["id"]).strip():
            call_id = str(call["id"])
        function = call.get("function")
        if isinstance(function, Mapping):
            if isinstance(function.get("name"), str) and str(function["name"]).strip():
                selected_tool = str(function["name"])
            raw_arguments = function.get("arguments")
            if isinstance(raw_arguments, str):
                try:
                    parsed_arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(parsed_arguments, Mapping):
                        arguments = dict(parsed_arguments)
                        arguments_json_valid = True

    hipengine = _mapping(choice.get("hipengine"), label="quality response hipengine metadata")
    generated = _token_row(
        hipengine.get("generated_token_ids"),
        label="quality response generated_token_ids",
    )
    if hipengine.get("generated_tokens") not in {None, len(generated)}:
        raise AgenticBenchmarkError("quality response generated token accounting is inexact")
    usage = _mapping(payload.get("usage"), label="quality response usage")
    if usage.get("completion_tokens") != len(generated):
        raise AgenticBenchmarkError("quality response completion token accounting is inexact")

    expected_turn = suite.workloads[workload_id]["turns"][turn_index]
    oracle_case = expected_turn.get("oracle_case")
    external_oracle = None
    if oracle_case is not None:
        external_oracle = evaluate_quality_oracle(
            suite,
            case_id=str(oracle_case),
            selected_tool=selected_tool,
            arguments=arguments,
        )
    quality = _quality_score(
        suite,
        workload_id=workload_id,
        turn_index=turn_index,
        finish_reason=finish_reason,
        detail_reason=detail_reason,
        public_content=public_content,
        raw_markup_leaked=raw_markup_leaked,
        tool_call_count=len(calls),
        call_id=call_id,
        selected_tool=selected_tool,
        arguments=arguments,
        arguments_json_valid=arguments_json_valid,
        repair_count=repairs,
        external_oracle=external_oracle,
    )
    return {
        "workload_id": str(workload_id),
        "workload_sha256": suite.workload_sha256(workload_id),
        "run_id": _nonempty(run_id, label="run_id"),
        "agent_id": _nonempty(agent_id, label="agent_id"),
        "session_id": _nonempty(session_id, label="session_id"),
        "turn_index": int(turn_index),
        "request_id": _nonempty(request_id, label="request_id"),
        "prompt": {
            "token_count": len(prompt_ids),
            "token_ids_sha256": token_ids_sha256(prompt_ids),
        },
        "output": {
            "generated_token_ids": list(generated),
            "generated_token_ids_sha256": token_ids_sha256(generated),
            "generated_token_ids_source": "response",
            "public_content": public_content,
            "raw_markup_leaked": raw_markup_leaked,
        },
        "quality": quality,
        "finish": {"reason": finish_reason, "detail_reason": detail_reason},
    }


def _validate_final_ownership(value: Any) -> dict[str, int]:
    ownership = _mapping(value, label="final_ownership")
    normalized: dict[str, int] = {}
    for field in _ZERO_OWNERSHIP_FIELDS:
        count = _nonnegative_int(ownership.get(field), label=f"final_ownership.{field}")
        if count != 0:
            raise AgenticBenchmarkError(f"final_ownership.{field} must be zero")
        normalized[field] = count
    resident = _nonnegative_int(
        ownership.get("cache_resident_bytes"), label="final_ownership.cache_resident_bytes"
    )
    allowed = _nonnegative_int(
        ownership.get("allowed_cache_bytes"), label="final_ownership.allowed_cache_bytes"
    )
    if resident > allowed:
        raise AgenticBenchmarkError(
            "final_ownership.cache_resident_bytes exceeds allowed_cache_bytes"
        )
    normalized["cache_resident_bytes"] = resident
    normalized["allowed_cache_bytes"] = allowed
    return normalized


def _validate_quality_records(
    suite: AgenticWorkloadSuite,
    records_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    if records_payload.get("kind") != AGENTIC_QUALITY_RECORDS_KIND:
        raise AgenticBenchmarkError("quality records kind is unsupported")
    if records_payload.get("schema_version") != AGENTIC_SCHEMA_VERSION:
        raise AgenticBenchmarkError("quality records schema_version is unsupported")
    configuration = dict(_mapping(records_payload.get("configuration"), label="configuration"))
    _nonempty(configuration.get("id"), label="configuration.id")
    if configuration.get("lane") != "auto_tool":
        raise AgenticBenchmarkError("quality configuration.lane must be auto_tool")
    concurrency = _positive_int(
        configuration.get("concurrency"), label="configuration.concurrency"
    )
    if configuration.get("cache_mode") not in {"off", "radix"}:
        raise AgenticBenchmarkError("quality configuration.cache_mode must be off or radix")
    _nonempty(configuration.get("backend"), label="configuration.backend")
    _nonempty(configuration.get("model"), label="configuration.model")
    if configuration.get("tool_choice") != "auto":
        raise AgenticBenchmarkError("quality configuration.tool_choice must be auto")
    if configuration.get("performance_claim", False) is not False:
        raise AgenticBenchmarkError("quality artifacts cannot make performance claims")
    if "repetitions" in configuration:
        _positive_int(configuration.get("repetitions"), label="configuration.repetitions")
    if "max_tokens" in configuration:
        _positive_int(configuration.get("max_tokens"), label="configuration.max_tokens")
    capabilities = configuration.get("server_capabilities")
    capabilities_hash = configuration.get("server_capabilities_sha256")
    if (capabilities is None) != (capabilities_hash is None):
        raise AgenticBenchmarkError(
            "quality server capabilities and SHA-256 must be recorded together"
        )
    if capabilities is not None:
        capabilities = _mapping(
            capabilities,
            label="configuration.server_capabilities",
        )
        expected_hash = _sha256(
            capabilities_hash,
            label="configuration.server_capabilities_sha256",
        )
        if _canonical_sha256(capabilities) != expected_hash:
            raise AgenticBenchmarkError("quality server capabilities hash mismatch")
    if "persistent_ownership_baseline" in configuration:
        baseline = _mapping(
            configuration.get("persistent_ownership_baseline"),
            label="configuration.persistent_ownership_baseline",
        )
        for field in ("kv_refcounted_pages", "kv_pinned_pages"):
            _nonnegative_int(
                baseline.get(field),
                label=f"configuration.persistent_ownership_baseline.{field}",
            )
    require_complete = configuration.get("require_complete_workloads")
    if not isinstance(require_complete, bool):
        raise AgenticBenchmarkError("quality require_complete_workloads must be boolean")

    raw_records = records_payload.get("turn_records")
    if not _is_sequence(raw_records) or not raw_records:
        raise AgenticBenchmarkError("quality turn_records must be a non-empty array")
    normalized: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    agents_by_run: dict[str, set[str]] = {}
    agent_sessions: dict[tuple[str, str], str] = {}
    agent_workloads: dict[tuple[str, str], str] = {}
    agent_turns: dict[tuple[str, str], list[int]] = {}

    for index, raw_record in enumerate(raw_records):
        record = _mapping(raw_record, label=f"quality record[{index}]")
        workload_id = _nonempty(
            record.get("workload_id"), label=f"quality record[{index}].workload_id"
        )
        if workload_id not in suite.workloads:
            raise AgenticBenchmarkError(f"quality record[{index}] has unknown workload_id")
        if record.get("workload_sha256") != suite.workload_sha256(workload_id):
            raise AgenticBenchmarkError(f"quality record[{index}] workload hash mismatch")
        run_id = _nonempty(record.get("run_id"), label=f"quality record[{index}].run_id")
        agent_id = _nonempty(record.get("agent_id"), label=f"quality record[{index}].agent_id")
        session_id = _nonempty(
            record.get("session_id"), label=f"quality record[{index}].session_id"
        )
        turn_index = _nonnegative_int(
            record.get("turn_index"), label=f"quality record[{index}].turn_index"
        )
        if turn_index >= len(suite.workloads[workload_id]["turns"]):
            raise AgenticBenchmarkError(f"quality record[{index}].turn_index is out of range")
        request_id = _nonempty(
            record.get("request_id"), label=f"quality record[{index}].request_id"
        )
        if request_id in request_ids:
            raise AgenticBenchmarkError(f"duplicate quality request_id {request_id!r}")
        request_ids.add(request_id)
        key = (run_id, agent_id)
        agents_by_run.setdefault(run_id, set()).add(agent_id)
        if key in agent_sessions and agent_sessions[key] != session_id:
            raise AgenticBenchmarkError(f"quality agent {run_id}/{agent_id} changed session_id")
        if key in agent_workloads and agent_workloads[key] != workload_id:
            raise AgenticBenchmarkError(f"quality agent {run_id}/{agent_id} changed workload_id")
        agent_sessions[key] = session_id
        agent_workloads[key] = workload_id
        agent_turns.setdefault(key, []).append(turn_index)

        prompt = _mapping(record.get("prompt"), label=f"quality record[{index}].prompt")
        _positive_int(
            prompt.get("token_count"), label=f"quality record[{index}].prompt.token_count"
        )
        _sha256(
            prompt.get("token_ids_sha256"),
            label=f"quality record[{index}].prompt.token_ids_sha256",
        )
        output = _mapping(record.get("output"), label=f"quality record[{index}].output")
        generated = _token_row(
            output.get("generated_token_ids"),
            label=f"quality record[{index}].output.generated_token_ids",
        )
        if output.get("generated_token_ids_source") != "response":
            raise AgenticBenchmarkError(
                f"quality record[{index}].output.generated_token_ids_source must be response"
            )
        generated_hash = _sha256(
            output.get("generated_token_ids_sha256"),
            label=f"quality record[{index}].output.generated_token_ids_sha256",
        )
        if generated_hash != token_ids_sha256(generated):
            raise AgenticBenchmarkError(f"quality record[{index}] generated token hash mismatch")
        public_content = output.get("public_content")
        if not isinstance(public_content, str):
            raise AgenticBenchmarkError(
                f"quality record[{index}].output.public_content must be a string"
            )
        raw_markup_leaked = output.get("raw_markup_leaked")
        if not isinstance(raw_markup_leaked, bool):
            raise AgenticBenchmarkError(
                f"quality record[{index}].output.raw_markup_leaked must be boolean"
            )
        if raw_markup_leaked != any(marker in public_content for marker in _RAW_MARKERS):
            raise AgenticBenchmarkError(f"quality record[{index}] raw markup flag is inconsistent")

        quality = _mapping(record.get("quality"), label=f"quality record[{index}].quality")
        selected_tool = quality.get("selected_tool")
        if selected_tool is not None and not isinstance(selected_tool, str):
            raise AgenticBenchmarkError(
                f"quality record[{index}].quality.selected_tool must be string or null"
            )
        call_id = quality.get("call_id")
        if call_id is not None and not isinstance(call_id, str):
            raise AgenticBenchmarkError(
                f"quality record[{index}].quality.call_id must be string or null"
            )
        arguments = quality.get("arguments")
        if arguments is not None and not isinstance(arguments, Mapping):
            raise AgenticBenchmarkError(
                f"quality record[{index}].quality.arguments must be object or null"
            )
        arguments_json_valid = quality.get("arguments_json_valid")
        if not isinstance(arguments_json_valid, bool):
            raise AgenticBenchmarkError(
                f"quality record[{index}].quality.arguments_json_valid must be boolean"
            )
        tool_call_count = _nonnegative_int(
            quality.get("tool_call_count"),
            label=f"quality record[{index}].quality.tool_call_count",
        )
        repairs = _nonnegative_int(
            quality.get("repair_count"),
            label=f"quality record[{index}].quality.repair_count",
        )
        finish = _mapping(record.get("finish"), label=f"quality record[{index}].finish")
        finish_reason = _nonempty(
            finish.get("reason"), label=f"quality record[{index}].finish.reason"
        )
        detail_reason = finish.get("detail_reason")
        if detail_reason is not None and not isinstance(detail_reason, str):
            raise AgenticBenchmarkError(
                f"quality record[{index}].finish.detail_reason must be string or null"
            )
        expected_turn = suite.workloads[workload_id]["turns"][turn_index]
        oracle_case = expected_turn.get("oracle_case")
        external_oracle = None
        if oracle_case is not None:
            external_oracle = evaluate_quality_oracle(
                suite,
                case_id=str(oracle_case),
                selected_tool=selected_tool,
                arguments=arguments,
            )
        expected_quality = _quality_score(
            suite,
            workload_id=workload_id,
            turn_index=turn_index,
            finish_reason=finish_reason,
            detail_reason=detail_reason,
            public_content=public_content,
            raw_markup_leaked=raw_markup_leaked,
            tool_call_count=tool_call_count,
            call_id=call_id,
            selected_tool=selected_tool,
            arguments=arguments,
            arguments_json_valid=arguments_json_valid,
            repair_count=repairs,
            external_oracle=external_oracle,
        )
        if dict(quality) != expected_quality:
            raise AgenticBenchmarkError(f"quality record[{index}] derived quality score is invalid")
        if quality.get("outcome") not in _QUALITY_OUTCOMES:
            raise AgenticBenchmarkError(f"quality record[{index}] outcome is unsupported")
        normalized.append(copy.deepcopy(dict(record)))

    invalid_runs = {
        run_id: len(agents)
        for run_id, agents in agents_by_run.items()
        if len(agents) != concurrency
    }
    if invalid_runs:
        raise AgenticBenchmarkError(
            f"quality configuration.concurrency={concurrency} but per-run agent counts are "
            f"{invalid_runs}"
        )
    if require_complete:
        for key, indexes in agent_turns.items():
            workload_id = agent_workloads[key]
            expected_indexes = list(range(len(suite.workloads[workload_id]["turns"])))
            if sorted(indexes) != expected_indexes:
                raise AgenticBenchmarkError(
                    f"quality agent {key[0]}/{key[1]} does not contain every workload turn"
                )
    ownership = _validate_final_ownership(records_payload.get("final_ownership"))
    return configuration, normalized, ownership


def _quality_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attempts = len(records)
    successes = 0
    valid_calls = 0
    correct_tools = 0
    exact_arguments = 0
    repairs = 0
    outcomes: dict[str, int] = {}
    for record in records:
        quality = _mapping(record["quality"], label="quality")
        successes += int(quality["success"])
        valid_calls += int(quality["valid_call"])
        correct_tools += int(quality["correct_tool"])
        exact_arguments += int(quality["exact_arguments"])
        repairs += int(quality["repair_count"])
        outcome = str(quality["outcome"])
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    return {
        "attempts": attempts,
        "valid_calls": valid_calls,
        "correct_tools": correct_tools,
        "exact_arguments": exact_arguments,
        "successes": successes,
        "valid_call_rate": valid_calls / attempts,
        "correct_tool_rate": correct_tools / attempts,
        "exact_arguments_rate": exact_arguments / attempts,
        "success_rate": successes / attempts,
        "repair_attempts": repairs,
        "outcomes": dict(sorted(outcomes.items())),
    }


def _external_oracle_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    external = [
        _mapping(record["quality"].get("external_oracle"), label="external_oracle")
        for record in records
        if isinstance(record["quality"].get("external_oracle"), Mapping)
    ]
    attempts = len(external)
    passes = sum(int(row["passed"]) for row in external)
    patches = [row for row in external if row["kind"] == "patch"]
    tests = [row for row in external if row["kind"] == "test"]
    patch_successes = sum(
        int(row["passed"] and row["patch_applied"] and row["tests_passed"])
        for row in patches
    )
    test_successes = sum(
        int(row["passed"] and row["tests_passed"])
        for row in tests
    )
    return {
        "attempts": attempts,
        "passes": passes,
        "pass_rate": passes / attempts if attempts else 0.0,
        "patch_attempts": len(patches),
        "patch_successes": patch_successes,
        "patch_success_rate": patch_successes / len(patches) if patches else 0.0,
        "test_attempts": len(tests),
        "test_successes": test_successes,
        "test_success_rate": test_successes / len(tests) if tests else 0.0,
    }


def _repeat_determinism(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_repetitions: int,
) -> dict[str, Any] | None:
    if expected_repetitions < 2:
        return None
    rows: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (
            str(record["workload_id"]),
            str(record["agent_id"]),
            int(record["turn_index"]),
        )
        rows.setdefault(key, []).append(record)
    mismatches: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for (workload_id, agent_id, turn_index), repeats in sorted(rows.items()):
        if len(repeats) != expected_repetitions:
            incomplete.append(
                {
                    "workload_id": workload_id,
                    "agent_id": agent_id,
                    "turn_index": turn_index,
                    "observed_repetitions": len(repeats),
                    "expected_repetitions": expected_repetitions,
                }
            )
            continue
        normalized_hashes: list[str] = []
        for record in repeats:
            quality = copy.deepcopy(dict(record["quality"]))
            quality.pop("call_id", None)
            normalized_hashes.append(
                _canonical_sha256(
                    {
                        "prompt": record["prompt"],
                        "output": record["output"],
                        "quality": quality,
                        "finish": record["finish"],
                    }
                )
            )
        if len(set(normalized_hashes)) != 1:
            mismatches.append(
                {
                    "workload_id": workload_id,
                    "agent_id": agent_id,
                    "turn_index": turn_index,
                    "normalized_sha256": normalized_hashes,
                }
            )
    return {
        "evaluated": True,
        "expected_repetitions": expected_repetitions,
        "task_rows": len(rows),
        "passed": not mismatches and not incomplete,
        "mismatches": mismatches,
        "incomplete_rows": incomplete,
    }


def _quality_rollup(
    suite: AgenticWorkloadSuite,
    records: Sequence[Mapping[str, Any]],
    *,
    expected_repetitions: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    attempts = len(records)
    generated_tokens = 0
    workloads: set[str] = set()
    agents: set[tuple[str, str]] = set()
    runs: set[str] = set()
    agents_by_run: dict[str, set[str]] = {}
    family_records: dict[str, list[Mapping[str, Any]]] = {}
    has_external_oracle = False
    for record in records:
        output = _mapping(record["output"], label="output")
        generated_tokens += len(output["generated_token_ids"])
        workload_id = str(record["workload_id"])
        run_id = str(record["run_id"])
        agent_id = str(record["agent_id"])
        workloads.add(workload_id)
        runs.add(run_id)
        agents.add((run_id, agent_id))
        agents_by_run.setdefault(run_id, set()).add(agent_id)
        family = suite.workloads[workload_id].get("family")
        if family is not None:
            family_records.setdefault(str(family), []).append(record)
        has_external_oracle |= isinstance(
            record["quality"].get("external_oracle"), Mapping
        )
    coverage = {
        "workloads": sorted(workloads),
        "runs": len(runs),
        "concurrency": max((len(items) for items in agents_by_run.values()), default=0),
        "agents": len(agents),
        "turns": attempts,
        "generated_tokens": generated_tokens,
    }
    quality_rollup = _quality_counts(records)
    determinism = _repeat_determinism(
        records,
        expected_repetitions=expected_repetitions,
    )
    if determinism is not None:
        quality_rollup["determinism"] = determinism
    if has_external_oracle:
        coverage["families"] = sorted(family_records)
        quality_rollup["external_oracle"] = _external_oracle_counts(records)
        quality_rollup["families"] = {
            family: {
                **_quality_counts(rows),
                "external_oracle": _external_oracle_counts(rows),
            }
            for family, rows in sorted(family_records.items())
        }
    return coverage, quality_rollup


def validate_agentic_quality_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a natural quality artifact envelope and bound turn-record hash."""

    root = _mapping(payload, label="agentic quality artifact")
    if root.get("kind") != AGENTIC_QUALITY_ARTIFACT_KIND:
        raise AgenticBenchmarkError("agentic quality artifact kind is invalid")
    if root.get("schema_version") != AGENTIC_SCHEMA_VERSION:
        raise AgenticBenchmarkError("agentic quality artifact schema_version is invalid")
    if root.get("performance_claim") is not False:
        raise AgenticBenchmarkError("agentic quality artifacts cannot make performance claims")
    validation = _mapping(root.get("validation"), label="quality validation")
    if validation.get("passed") is not True or validation.get("failure_reasons") != []:
        raise AgenticBenchmarkError("agentic quality artifact validation did not pass")
    records = root.get("turn_records")
    if not _is_sequence(records) or not records:
        raise AgenticBenchmarkError("agentic quality artifact turn_records must be non-empty")
    record_hash = _sha256(root.get("turn_records_sha256"), label="turn_records_sha256")
    if record_hash != _canonical_sha256(records):
        raise AgenticBenchmarkError("turn_records_sha256 does not match turn_records")
    suite_payload = _mapping(root.get("workload_suite"), label="workload_suite")
    if not isinstance(root.get("configuration"), Mapping):
        raise AgenticBenchmarkError("agentic quality artifact configuration is invalid")
    if not isinstance(root.get("final_ownership"), Mapping):
        raise AgenticBenchmarkError("agentic quality artifact final_ownership is invalid")
    if not isinstance(suite_payload.get("file_sha256"), str):
        raise AgenticBenchmarkError("agentic quality artifact workload suite is invalid")
    _mapping(root.get("coverage"), label="coverage")
    _mapping(root.get("quality"), label="quality")
    provenance = root.get("hipengine_artifact_provenance")
    if provenance is not None:
        validate_artifact_provenance(
            _mapping(provenance, label="hipengine_artifact_provenance"),
            require_model=True,
        )
    return {"passed": True, "failure_reasons": []}


def build_agentic_quality_artifact(
    suite: AgenticWorkloadSuite,
    records_payload: Mapping[str, Any],
    *,
    created_at: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate natural quality records and build a non-performance artifact."""

    configuration, records, ownership = _validate_quality_records(suite, records_payload)
    coverage, quality = _quality_rollup(
        suite,
        records,
        expected_repetitions=int(configuration.get("repetitions", 1)),
    )
    artifact = {
        "kind": AGENTIC_QUALITY_ARTIFACT_KIND,
        "schema_version": AGENTIC_SCHEMA_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "workload_suite": suite.identity(),
        "configuration": copy.deepcopy(configuration),
        "coverage": coverage,
        "validation": {"passed": True, "failure_reasons": []},
        "quality": quality,
        "final_ownership": ownership,
        "turn_records_sha256": _canonical_sha256(records),
        "turn_records": records,
    }
    if provenance is not None:
        artifact["hipengine_artifact_provenance"] = validate_artifact_provenance(
            provenance,
            require_model=True,
        )
    validate_agentic_quality_artifact(artifact)
    return artifact


__all__ = [
    "AGENTIC_QUALITY_ARTIFACT_KIND",
    "AGENTIC_QUALITY_RECORDS_KIND",
    "build_agentic_quality_artifact",
    "normalize_chat_quality_turn",
    "validate_agentic_quality_artifact",
]
