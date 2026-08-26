#!/usr/bin/env python3
"""Collect the frozen expanded AGENTIC-QUALITY2 quality suite from a live server."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.agentic import AgenticBenchmarkError  # noqa: E402
from hipengine.benchmark.agentic_live import (  # noqa: E402
    build_canonical_turn_messages,
    build_openai_tools,
    render_workload_prefix,
)
from hipengine.benchmark.agentic_quality2 import (  # noqa: E402
    AgenticQuality2Error,
    AgenticQuality2Suite,
    _arguments_match_schema,
    aggregate_quality2_results,
    evaluate_quality2_fail_safe_control,
    evaluate_quality2_oracle,
    load_agentic_quality2_suite,
)
from hipengine.benchmark.agentic_quality2_sandbox import (  # noqa: E402
    AgenticQuality2Sandbox,
)
from hipengine.benchmark.provenance import collect_artifact_provenance  # noqa: E402
from hipengine.tokenization.identity import token_ids_sha256  # noqa: E402
from scripts.agentic_coding_live import LiveHTTPTransport  # noqa: E402
from scripts.agentic_coding_quality import (  # noqa: E402
    _atomic_write_json,
    _canonical_sha256,
    _idle_persistent_ownership_baseline,
    _quality_build_profile,
    _validate_live_capabilities,
    _wait_for_final_ownership,
)

_SYSTEM_POLICY = (
    "You are measuring automatic tool and task quality against a synthetic repository. "
    "Choose declared tools only when they are appropriate. Complete every requested "
    "action; when tools are used, return only the tool call or calls. A direct response "
    "is allowed when the user explicitly does not need a tool. Do not expose reasoning "
    "or raw tool markers.\n\n"
)
_RAW_MARKERS = ("<think", "</think", "<tool_call", "</tool_call")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AgenticQuality2Error(f"{label} must be an object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise AgenticQuality2Error(f"{label} must be an array")
    return value


def _response_generated_ids(payload: Mapping[str, Any]) -> tuple[int, ...]:
    choices = _sequence(payload.get("choices"), label="response.choices")
    if len(choices) != 1:
        raise AgenticQuality2Error("expanded response must contain one choice")
    choice = _mapping(choices[0], label="response.choices[0]")
    hipengine = choice.get("hipengine")
    token_ids = hipengine.get("generated_token_ids") if isinstance(hipengine, Mapping) else None
    if not isinstance(token_ids, list):
        response_hipengine = payload.get("hipengine")
        accounting = (
            response_hipengine.get("token_accounting")
            if isinstance(response_hipengine, Mapping)
            else None
        )
        rows = (
            accounting.get("choice_generated_token_ids")
            if isinstance(accounting, Mapping)
            else None
        )
        token_ids = rows[0] if isinstance(rows, list) and len(rows) == 1 else None
    if not isinstance(token_ids, list) or not token_ids:
        raise AgenticQuality2Error("expanded response omits response-owned generated IDs")
    if not all(
        isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in token_ids
    ):
        raise AgenticQuality2Error("expanded response generated IDs are invalid")
    return tuple(int(token) for token in token_ids)


def _public_calls(choice: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    message = _mapping(choice.get("message"), label="response.message")
    raw_calls = message.get("tool_calls", ())
    if raw_calls is None:
        raw_calls = ()
    calls: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, raw_call in enumerate(_sequence(raw_calls, label="response.tool_calls")):
        call = _mapping(raw_call, label=f"response.tool_calls[{index}]")
        function = _mapping(call.get("function"), label=f"response.tool_calls[{index}].function")
        name = function.get("name")
        raw_arguments = function.get("arguments")
        if not isinstance(name, str) or not name:
            errors.append(f"call {index} has no function name")
            continue
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else None
        except json.JSONDecodeError as exc:
            errors.append(f"call {index} arguments JSON failed at {exc.pos}")
            continue
        if not isinstance(arguments, Mapping):
            errors.append(f"call {index} arguments are not an object")
            continue
        calls.append({"tool": name, "arguments": copy.deepcopy(dict(arguments))})
    return calls, errors


def _quality_outcome(
    workload: Mapping[str, Any],
    *,
    calls: Sequence[Mapping[str, Any]],
    public_content: str,
    raw_markup_leaked: bool,
    finish_reason: str,
    evaluation: Mapping[str, Any],
    parse_errors: Sequence[str],
) -> str:
    expected = workload["turns"][0]
    if evaluation["status"] == "blocked_sandbox":
        return "blocked_sandbox"
    if raw_markup_leaked:
        return "raw_markup_leak"
    if public_content.strip() and calls:
        return "content_alongside_tool_call"
    if parse_errors:
        return "invalid_arguments"
    expected_outcome = expected["expected_outcome"]
    if expected_outcome == "no_tool_call":
        if calls:
            return "unexpected_tool_call"
        if not public_content.strip():
            return "missing_direct_response"
        return "passed" if evaluation["success"] else "oracle_failed"
    if not calls:
        return "no_tool_call"
    if expected_outcome == "tool_calls" and len(calls) != len(expected["expected_calls"]):
        return "wrong_call_count"
    if expected_outcome == "tool_call" and len(calls) != 1:
        return "wrong_call_count"
    expected_tools = (
        [str(expected["expected_tool"])]
        if expected_outcome == "tool_call"
        else [str(call["tool"]) for call in expected["expected_calls"]]
    )
    if sorted(str(call["tool"]) for call in calls) != sorted(expected_tools):
        return "wrong_tool"
    if evaluation["success"]:
        required_finish = "tool_calls"
        return "passed" if finish_reason == required_finish else "finish_mismatch"
    return "oracle_failed"


def _normalized_response_sha256(record: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {
            "calls": record["calls"],
            "call_parse_errors": record["call_parse_errors"],
            "finish": record["finish"],
            "generated_token_ids": record["output"]["generated_token_ids"],
            "public_content": record["output"]["public_content"],
        }
    )


def _expected_calls(workload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    turn = _mapping(workload["turns"][0], label="workload.turn")
    expected_outcome = str(turn["expected_outcome"])
    if expected_outcome == "no_tool_call":
        return ()
    if expected_outcome == "tool_calls":
        return tuple(
            {
                "tool": str(_mapping(call, label="expected call")["tool"]),
                "arguments": copy.deepcopy(dict(call["arguments"])),
            }
            for call in _sequence(turn["expected_calls"], label="expected calls")
        )
    expected_arguments = turn.get("expected_arguments")
    return (
        {
            "tool": str(turn["expected_tool"]),
            "arguments": (
                copy.deepcopy(dict(expected_arguments))
                if isinstance(expected_arguments, Mapping)
                else None
            ),
        },
    )


def _call_fingerprint(call: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {
            "tool": str(call["tool"]),
            "arguments": copy.deepcopy(dict(call["arguments"])),
        }
    )


def _quality_metrics(
    suite: AgenticQuality2Suite,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def ratio(passed: int, denominator: int) -> float:
        return 0.0 if denominator == 0 else passed / denominator

    public_call_blocks = 0
    json_valid_calls = 0
    declared_calls = 0
    schema_valid_calls = 0
    tool_expected = 0
    valid_call_observations = 0
    correct_tool = 0
    no_tool_expected = 0
    correct_no_tool = 0
    argument_denominator = 0
    exact_arguments = 0
    terminal = 0
    oracle_statuses: Counter[str] = Counter()
    by_kind: dict[str, Counter[str]] = {}
    outcomes: Counter[str] = Counter()
    raw_markup_leaks = 0
    content_alongside_calls = 0
    malformed_or_invalid = 0
    truncated = 0

    for raw_record in records:
        record = _mapping(raw_record, label="quality record")
        workload = _mapping(
            suite.workloads.get(str(record["workload_id"])),
            label="quality workload",
        )
        turn = _mapping(workload["turns"][0], label="quality workload turn")
        calls = tuple(
            _mapping(call, label="quality call")
            for call in _sequence(record["calls"], label="quality calls")
        )
        parse_errors = tuple(
            str(value)
            for value in _sequence(
                record["call_parse_errors"],
                label="quality call parse errors",
            )
        )
        public_call_blocks += len(calls) + len(parse_errors)
        json_valid_calls += len(calls)
        calls_declared = [str(call["tool"]) in suite.tools for call in calls]
        calls_schema_valid = [
            declared
            and _arguments_match_schema(
                _mapping(call["arguments"], label="quality call arguments"),
                _mapping(suite.tools[str(call["tool"])]["parameters"], label="tool schema"),
            )
            for call, declared in zip(calls, calls_declared, strict=True)
        ]
        declared_calls += sum(calls_declared)
        schema_valid_calls += sum(calls_schema_valid)
        expected = _expected_calls(workload)
        expected_tools = sorted(str(call["tool"]) for call in expected)
        actual_tools = sorted(str(call["tool"]) for call in calls)
        expected_outcome = str(turn["expected_outcome"])
        if expected_outcome == "no_tool_call":
            no_tool_expected += 1
            no_calls = not calls and not parse_errors
            correct_no_tool += int(no_calls)
            correct_tool += int(no_calls)
        else:
            tool_expected += 1
            valid_call_observations += int(
                bool(calls)
                and not parse_errors
                and all(calls_declared)
                and all(calls_schema_valid)
            )
            tools_correct = actual_tools == expected_tools and not parse_errors
            correct_tool += int(tools_correct)
            arguments_are_explicit = all(
                isinstance(call.get("arguments"), Mapping) for call in expected
            )
            if (
                str(turn.get("argument_scoring")) != "not_applicable"
                and arguments_are_explicit
            ):
                argument_denominator += 1
                exact_arguments += int(
                    tools_correct
                    and sorted(_call_fingerprint(call) for call in calls)
                    == sorted(_call_fingerprint(call) for call in expected)
                )
        finish = _mapping(record["finish"], label="quality finish")
        terminal += int(bool(str(finish.get("reason") or "")))
        truncated += int(str(finish.get("reason") or "") == "length")
        quality = _mapping(record["quality"], label="quality result")
        status = str(quality["status"])
        oracle_statuses[status] += 1
        kind = str(record["oracle_kind"])
        by_kind.setdefault(kind, Counter())[status] += 1
        outcome = str(quality["outcome"])
        outcomes[outcome] += 1
        raw_markup_leaks += int(bool(record["output"]["raw_markup_leaked"]))
        content_alongside_calls += int(outcome == "content_alongside_tool_call")
        malformed_or_invalid += int(bool(parse_errors))

    observations = len(records)
    external_scored = oracle_statuses["passed"] + oracle_statuses["failed"]
    return {
        "observations": {"attempted": observations, "terminal": terminal},
        "call_blocks": {
            "public": public_call_blocks,
            "json_valid": json_valid_calls,
            "declared": declared_calls,
            "strict_schema_valid": schema_valid_calls,
        },
        "valid_call": {
            "denominator": tool_expected,
            "passed": valid_call_observations,
            "rate": ratio(valid_call_observations, tool_expected),
        },
        "tool_selection": {
            "denominator": observations,
            "correct": correct_tool,
            "rate": ratio(correct_tool, observations),
            "no_tool_expected": no_tool_expected,
            "correct_no_tool": correct_no_tool,
        },
        "exact_arguments": {
            "diagnostic_only": True,
            "denominator": argument_denominator,
            "passed": exact_arguments,
            "rate": ratio(exact_arguments, argument_denominator),
        },
        "external_oracle": {
            "passed": oracle_statuses["passed"],
            "failed": oracle_statuses["failed"],
            "blocked_sandbox": oracle_statuses["blocked_sandbox"],
            "unscorable": oracle_statuses["unscorable"],
            "scored_denominator": external_scored,
            "success_rate": ratio(oracle_statuses["passed"], external_scored),
        },
        "by_oracle_kind": {
            kind: {
                "passed": counts["passed"],
                "failed": counts["failed"],
                "blocked_sandbox": counts["blocked_sandbox"],
                "unscorable": counts["unscorable"],
            }
            for kind, counts in sorted(by_kind.items())
        },
        "outcomes": dict(sorted(outcomes.items())),
        "safety": {
            "raw_markup_leak": raw_markup_leaks,
            "content_alongside_tool_call": content_alongside_calls,
            "malformed_or_invalid_arguments": malformed_or_invalid,
            "truncated": truncated,
            "runtime_error": 0,
        },
        "repair": {
            "telemetry_available": False,
            "attempted": None,
            "succeeded": None,
            "failed": None,
            "extra_generated_tokens": None,
        },
        "generated_response_owned_tokens": sum(
            int(record["output"]["generated_token_count"]) for record in records
        ),
    }


def _fail_safe_rollup(suite: AgenticQuality2Suite) -> dict[str, Any]:
    controls = [
        evaluate_quality2_fail_safe_control(suite, str(row["id"]))
        for row in suite.oracle["fail_safe_controls"]
    ]
    passed_count = sum(row["passed"] is True for row in controls)
    return {
        "mode": "independent_parser_publication_policy",
        "endpoint_exercised": False,
        "passed": passed_count == len(controls),
        "passed_count": passed_count,
        "total": len(controls),
        "controls": [
            {
                "control_id": str(row["control_id"]),
                "class": str(row["class"]),
                "split": str(row["split"]),
                "passed": bool(row["passed"]),
            }
            for row in controls
        ],
    }


def _normalize_turn(
    suite: AgenticQuality2Suite,
    *,
    workload_id: str,
    repetition: int,
    prompt_ids: Sequence[int],
    payload: Mapping[str, Any],
    sandbox: AgenticQuality2Sandbox,
) -> tuple[dict[str, Any], dict[str, Any]]:
    workload = suite.workloads[workload_id]
    choices = _sequence(payload.get("choices"), label="response.choices")
    if len(choices) != 1:
        raise AgenticQuality2Error("expanded response must contain one choice")
    choice = _mapping(choices[0], label="response.choices[0]")
    message = _mapping(choice.get("message"), label="response.message")
    content = str(message.get("content") or "")
    calls, parse_errors = _public_calls(choice)
    generated_ids = _response_generated_ids(payload)
    finish_reason = str(choice.get("finish_reason") or "")
    finish_details = choice.get("finish_details")
    detail_reason = (
        str(finish_details.get("reason"))
        if isinstance(finish_details, Mapping) and finish_details.get("reason") is not None
        else None
    )
    evaluation = evaluate_quality2_oracle(
        suite,
        workload_id=workload_id,
        calls=calls,
        public_text=content,
        sandbox=sandbox,
    )
    raw_markup_leaked = any(marker in content for marker in _RAW_MARKERS)
    outcome = _quality_outcome(
        workload,
        calls=calls,
        public_content=content,
        raw_markup_leaked=raw_markup_leaked,
        finish_reason=finish_reason,
        evaluation=evaluation,
        parse_errors=parse_errors,
    )
    if outcome != "passed" and evaluation["status"] == "passed":
        evaluation = {**evaluation, "status": "failed", "success": False}
    request_id = f"run-{repetition}-{workload_id}"
    record = {
        "workload_id": workload_id,
        "split": str(workload["split"]),
        "family": str(workload["family"]),
        "language": str(workload["language"]),
        "task_kind": str(workload["task_kind"]),
        "oracle_kind": str(evaluation["kind"]),
        "repetition": int(repetition),
        "request_id": request_id,
        "prompt": {
            "token_count": len(prompt_ids),
            "token_ids_sha256": token_ids_sha256(prompt_ids),
        },
        "output": {
            "generated_token_count": len(generated_ids),
            "generated_token_ids": list(generated_ids),
            "generated_token_ids_sha256": token_ids_sha256(generated_ids),
            "generated_token_ids_source": "response",
            "public_content": content,
            "public_content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "raw_markup_leaked": raw_markup_leaked,
        },
        "calls": calls,
        "call_parse_errors": parse_errors,
        "finish": {
            "reason": finish_reason,
            "detail_reason": detail_reason,
        },
        "quality": {
            "status": str(evaluation["status"]),
            "success": bool(evaluation["success"]),
            "outcome": outcome,
            "result_sha256": evaluation.get("result_sha256"),
            "expected_result_sha256": evaluation.get("expected_result_sha256"),
            "error": evaluation.get("error"),
            "sandbox": evaluation.get("sandbox"),
        },
    }
    record["normalized_response_sha256"] = _normalized_response_sha256(record)
    aggregate_row = {
        "workload_id": workload_id,
        "split": str(workload["split"]),
        "family": str(workload["family"]),
        "language": str(workload["language"]),
        "kind": str(evaluation["kind"]),
        "repetition": int(repetition),
        "status": str(evaluation["status"]),
        "success": bool(evaluation["success"]),
        "result_sha256": evaluation.get("result_sha256"),
        "error": evaluation.get("error"),
        "normalized_response_sha256": record["normalized_response_sha256"],
    }
    return record, aggregate_row


def collect_live_quality2_records(
    transport: LiveHTTPTransport,
    *,
    suite_path: str | Path,
    workload_ids: Sequence[str] | None,
    model: str,
    backend: str,
    repetitions: int,
    max_tokens: int,
    cache_mode: str,
    idle_timeout_s: float,
    sandbox: AgenticQuality2Sandbox,
    checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[AgenticQuality2Suite, dict[str, Any], dict[str, Any]]:
    """Collect expanded development then heldout rows with sealed aggregation."""

    if repetitions <= 0 or max_tokens <= 0:
        raise AgenticQuality2Error("repetitions and max_tokens must be positive")
    if cache_mode != "off":
        raise AgenticQuality2Error("expanded quality collector currently requires cache off")
    suite = load_agentic_quality2_suite(suite_path)
    frozen_policy = suite.payload["collector_policy"]
    if int(frozen_policy["repetitions"]) != int(repetitions):
        raise AgenticQuality2Error("collector repetitions differ from frozen suite")
    if int(frozen_policy["max_tokens"]) != int(max_tokens):
        raise AgenticQuality2Error("collector max_tokens differ from frozen suite")
    selected = (
        tuple(suite.development_ids) + tuple(suite.heldout_ids)
        if workload_ids is None
        else tuple(str(value) for value in workload_ids)
    )
    if len(selected) != len(set(selected)):
        raise AgenticQuality2Error("expanded workload selection contains duplicates")
    unknown = [value for value in selected if value not in suite.workloads]
    if unknown:
        raise AgenticQuality2Error(f"expanded workload selection is unknown: {unknown}")
    if workload_ids is None and selected != tuple(suite.development_ids) + tuple(suite.heldout_ids):
        raise AgenticQuality2Error("expanded full selection must run development then heldout")
    capabilities = transport.capabilities()
    if not isinstance(capabilities, Mapping):
        raise AgenticQuality2Error("server capabilities must be an object")
    _validate_live_capabilities(
        capabilities,
        model=model,
        backend=backend,
        cache_mode=cache_mode,
    )
    capabilities_payload = json.loads(json.dumps(capabilities, sort_keys=True, ensure_ascii=False))
    persistent_ownership = _idle_persistent_ownership_baseline(
        transport,
        cache_mode=cache_mode,
    )
    tools = build_openai_tools(suite)  # type: ignore[arg-type]
    records: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    raw_turns: list[dict[str, Any]] = []
    fail_safe_controls: dict[str, Any] | None = None
    total = repetitions * len(selected)

    def checkpoint(status: str, final_ownership: Mapping[str, Any] | None = None) -> None:
        if checkpoint_callback is None:
            return
        payload: dict[str, Any] = {
            "kind": "hipengine_agentic_quality2_checkpoint",
            "schema_version": 1,
            "status": status,
            "performance_claim": False,
            "suite": suite.identity(),
            "configuration": {
                "model": model,
                "backend": backend,
                "cache_mode": cache_mode,
                "repetitions": repetitions,
                "max_tokens": max_tokens,
                "server_capabilities_sha256": _canonical_sha256(capabilities_payload),
                "persistent_ownership_baseline": dict(persistent_ownership),
            },
            "progress": {"completed": len(records), "total": total},
            "records": records,
            "raw_turns": raw_turns,
        }
        if fail_safe_controls is not None:
            payload["fail_safe_controls"] = copy.deepcopy(fail_safe_controls)
        if final_ownership is not None:
            payload["final_ownership"] = dict(final_ownership)
        checkpoint_callback(payload)

    for repetition in range(repetitions):
        for workload_id in selected:
            workload = suite.workloads[workload_id]
            prefix = render_workload_prefix(  # type: ignore[arg-type]
                suite,
                workload_id,
                tokenize=transport.tokenize,
                detokenize=transport.detokenize,
            )
            messages = build_canonical_turn_messages(  # type: ignore[arg-type]
                suite,
                workload_id,
                turn_index=0,
                agent_id="agent-0",
                prefix_text=prefix.text,
                system_policy=_SYSTEM_POLICY,
            )
            prompt_ids = transport.rendered_prompt_ids(
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            parallel = workload["turns"][0]["expected_outcome"] == "tool_calls"
            request = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "parallel_tool_calls": bool(parallel),
                "temperature": 0.0,
                "max_tokens": int(max_tokens),
                "enable_thinking": False,
                "stream": False,
            }
            response = transport.chat_json(request)
            record, aggregate_row = _normalize_turn(
                suite,
                workload_id=workload_id,
                repetition=repetition,
                prompt_ids=prompt_ids,
                payload=response,
                sandbox=sandbox,
            )
            records.append(record)
            aggregate_rows.append(aggregate_row)
            raw_turns.append(
                {
                    "workload_id": workload_id,
                    "split": workload["split"],
                    "repetition": repetition,
                    "request_id": record["request_id"],
                    "prompt_token_ids": list(prompt_ids),
                    "response": response,
                }
            )
            checkpoint("in_progress")
            visible_outcome = (
                record["quality"]["outcome"]
                if workload["split"] == "development"
                else "heldout_sealed"
            )
            print(
                f"quality2 progress {len(records)}/{total}: split={workload['split']} "
                f"workload={workload_id} repetition={repetition} outcome={visible_outcome}",
                flush=True,
            )

    fail_safe_controls = _fail_safe_rollup(suite)
    if not fail_safe_controls["passed"]:
        raise AgenticQuality2Error("expanded fail-safe controls did not pass")
    checkpoint("controls_complete")
    ownership = _wait_for_final_ownership(
        transport,
        cache_mode=cache_mode,
        timeout_s=idle_timeout_s,
        persistent_ownership=persistent_ownership,
    )
    checkpoint("complete", ownership)
    aggregation = aggregate_quality2_results(
        suite,
        aggregate_rows,
        expected_repetitions=repetitions,
        expected_workload_ids=selected,
        seal_heldout_details=True,
    )
    outcomes = sorted({record["quality"]["outcome"] for record in records})
    aggregation["outcomes"] = {
        outcome: sum(record["quality"]["outcome"] == outcome for record in records)
        for outcome in outcomes
    }
    records_payload = {
        "kind": "hipengine_agentic_quality2_records",
        "schema_version": 1,
        "performance_claim": False,
        "suite": suite.identity(),
        "configuration": {
            "model": model,
            "backend": backend,
            "cache_mode": cache_mode,
            "repetitions": repetitions,
            "max_tokens": max_tokens,
            "server_capabilities": capabilities_payload,
            "server_capabilities_sha256": _canonical_sha256(capabilities_payload),
            "persistent_ownership_baseline": dict(persistent_ownership),
        },
        "records": records,
        "fail_safe_controls": copy.deepcopy(fail_safe_controls),
        "final_ownership": ownership,
    }
    summary = {
        "kind": "hipengine_agentic_quality2_baseline",
        "schema_version": 1,
        "performance_claim": False,
        "suite": suite.identity(),
        "configuration": copy.deepcopy(records_payload["configuration"]),
        "coverage": {
            "workloads": len(selected),
            "observations": len(records),
            "development": sum(
                suite.workloads[value]["split"] == "development" for value in selected
            ),
            "heldout": sum(suite.workloads[value]["split"] == "heldout" for value in selected),
            "generated_response_owned_tokens": sum(
                record["output"]["generated_token_count"] for record in records
            ),
        },
        "aggregation": aggregation,
        "quality_metrics": _quality_metrics(suite, records),
        "fail_safe_controls": copy.deepcopy(fail_safe_controls),
        "records_sha256": _canonical_sha256(records),
        "final_ownership": ownership,
        "validation": {
            "complete": len(records) == total,
            "response_owned_ids": all(
                record["output"]["generated_token_ids_source"] == "response" for record in records
            ),
            "heldout_details_sealed": aggregation["heldout_details_sealed"],
            "determinism_basis": aggregation["determinism"]["basis"],
            "determinism_passed": aggregation["determinism"]["passed"],
            "fail_safe_controls_passed": fail_safe_controls["passed"],
        },
    }
    return suite, records_payload, summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--workload", action="append")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--target-arch", required=True)
    parser.add_argument("--device-name", required=True)
    parser.add_argument("--quant", required=True)
    parser.add_argument("--kv-dtype", required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--require-clean-provenance", action="store_true")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--cache-mode", choices=("off",), default="off")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--idle-timeout-s", type=float, default=60.0)
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--checkpoint-json", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        transport = LiveHTTPTransport(
            args.base_url,
            api_key=args.api_key,
            timeout_s=args.timeout_s,
        )
        _suite, records, summary = collect_live_quality2_records(
            transport,
            suite_path=args.suite,
            workload_ids=args.workload,
            model=args.model,
            backend=args.backend,
            repetitions=args.repetitions,
            max_tokens=args.max_tokens,
            cache_mode=args.cache_mode,
            idle_timeout_s=args.idle_timeout_s,
            sandbox=AgenticQuality2Sandbox(),
            checkpoint_callback=lambda payload: _atomic_write_json(
                args.checkpoint_json,
                payload,
            ),
        )
        hipcc_version = args.compiler_version_file.read_text(encoding="utf-8")
        provenance = collect_artifact_provenance(
            repo_root=REPO_ROOT,
            configured_backend=args.backend,
            resolved_backend=args.backend,
            detected_arches=(args.target_arch,),
            target_arch=args.target_arch,
            device_name=args.device_name,
            model_path=args.model_path,
            quant=args.quant,
            kv_dtype=args.kv_dtype,
            command=(sys.executable, *sys.argv),
            environment={
                key: os.environ.get(key)
                for key in (
                    "HIPENGINE_HIP_ARCH",
                    "HIPENGINE_COMPILER_VERSION_FILE",
                    "HIPENGINE_QWEN35_NATIVE_SAMPLER",
                    "HIP_VISIBLE_DEVICES",
                    "ROCR_VISIBLE_DEVICES",
                    "GPU_MAX_HW_QUEUES",
                    "HSA_ENABLE_SDMA",
                    "HSA_USE_SVM",
                )
            },
            build_profile=_quality_build_profile(args.backend),
            timing_protocol=(
                "real localhost blocking OpenAI chat; expanded external/sandbox "
                "quality oracles; independent fail-safe parser/publication controls; "
                "heldout detail sealed; no performance fields"
            ),
            warmups=0,
            repetitions=args.repetitions,
            profiler={"used": False, "reason": "non-performance quality lane"},
            hipcc_version=hipcc_version,
        )
        if args.require_clean_provenance and provenance["dirty"]:
            raise AgenticQuality2Error("expanded quality provenance must be clean")
        summary["hipengine_artifact_provenance"] = provenance
        _atomic_write_json(args.records_json, records)
        _atomic_write_json(args.json, summary)
    except (AgenticBenchmarkError, AgenticQuality2Error, json.JSONDecodeError, OSError) as exc:
        print(f"expanded agentic quality benchmark rejected: {exc}", file=sys.stderr)
        return 2
    overall = summary["aggregation"]["overall"]
    print(
        f"Expanded quality collected: {overall['passed']}/{overall['scored_denominator']} "
        f"scored observations; blocked={overall['blocked_sandbox']} -> {args.json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
