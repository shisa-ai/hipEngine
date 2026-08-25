"""Independent boundary taxonomy for agentic quality observations."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from hipengine.benchmark.agentic import AgenticBenchmarkError, AgenticWorkloadSuite
from hipengine.benchmark.agentic_quality import _arguments_match_schema

_TOOL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_RAW_MARKERS = ("<think", "</think", "<tool_call", "</tool_call")
_RUNTIME_OUTCOMES = {
    "parser_mismatch",
    "runtime_error",
    "template_or_tokenizer_mismatch",
}


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _canonical_sha256(value: Any) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _text_sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def parse_independent_tool_envelope(raw_text: str) -> dict[str, Any]:
    """Parse one generic Qwen tool envelope without using the server parser."""

    text = str(raw_text)
    matches = list(_TOOL_BLOCK_RE.finditer(text))
    if not matches:
        stripped = text.strip()
        if not stripped:
            return {
                "accepted": False,
                "error": "no_tool_call",
                "content": "",
                "tool_calls": [],
            }
        if "<tool_call" in stripped or "</tool_call" in stripped:
            return {
                "accepted": False,
                "error": "malformed_envelope",
                "content": stripped,
                "tool_calls": [],
            }
        if not stripped.startswith("{"):
            return {
                "accepted": False,
                "error": "no_tool_call",
                "content": stripped,
                "tool_calls": [],
            }
        candidates = ((stripped, ""),)
    else:
        content_parts: list[str] = []
        cursor = 0
        candidates: list[tuple[str, str]] = []
        for match in matches:
            content_parts.append(text[cursor : match.start()])
            candidates.append((match.group(1).strip(), match.group(0)))
            cursor = match.end()
        content_parts.append(text[cursor:])
        content = "".join(content_parts).strip()
        candidates = tuple((body, raw) for body, raw in candidates)

    calls: list[dict[str, Any]] = []
    for body, raw_block in candidates:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            return {
                "accepted": False,
                "error": "invalid_json",
                "content": text.strip(),
                "tool_calls": [],
                "json_error": {
                    "position": int(exc.pos),
                    "message": str(exc.msg),
                },
            }
        if not isinstance(payload, Mapping):
            return {
                "accepted": False,
                "error": "malformed_envelope",
                "content": text.strip(),
                "tool_calls": [],
            }
        function = payload.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            arguments = function.get("arguments")
        else:
            name = payload.get("name")
            arguments = payload.get("arguments")
        if not isinstance(name, str) or not name.strip():
            return {
                "accepted": False,
                "error": "malformed_envelope",
                "content": text.strip(),
                "tool_calls": [],
            }
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                return {
                    "accepted": False,
                    "error": "invalid_json",
                    "content": text.strip(),
                    "tool_calls": [],
                    "json_error": {
                        "position": int(exc.pos),
                        "message": str(exc.msg),
                    },
                }
        if not isinstance(arguments, Mapping):
            return {
                "accepted": False,
                "error": "invalid_json",
                "content": text.strip(),
                "tool_calls": [],
            }
        calls.append(
            {
                "name": name.strip(),
                "arguments": copy.deepcopy(dict(arguments)),
                "raw_sha256": _canonical_sha256(raw_block or body),
            }
        )
    return {
        "accepted": True,
        "error": None,
        "content": content if matches else "",
        "tool_calls": calls,
    }


def _raw_response_error(raw_response: Mapping[str, Any]) -> str | None:
    error = raw_response.get("error")
    if error is not None:
        return str(error)
    choices = raw_response.get("choices")
    if not _is_sequence(choices) or not choices:
        return "response omits choices"
    return None


def _public_projection(record: Mapping[str, Any]) -> tuple[str | None, Mapping[str, Any] | None]:
    quality = record.get("quality")
    if not isinstance(quality, Mapping):
        raise AgenticBenchmarkError("taxonomy record quality must be an object")
    selected = quality.get("selected_tool")
    arguments = quality.get("arguments")
    return (
        None if selected is None else str(selected),
        arguments if isinstance(arguments, Mapping) else None,
    )


def _parser_projection_matches(
    parsed: Mapping[str, Any],
    record: Mapping[str, Any],
) -> bool:
    quality = record["quality"]
    calls = parsed.get("tool_calls")
    if not _is_sequence(calls):
        return False
    if len(calls) != int(quality.get("tool_call_count", 0)):
        return False
    if not calls:
        return quality.get("selected_tool") is None
    if len(calls) != 1 or not isinstance(calls[0], Mapping):
        return False
    selected, arguments = _public_projection(record)
    return calls[0].get("name") == selected and calls[0].get("arguments") == arguments


def classify_agentic_quality_observation(
    suite: AgenticWorkloadSuite,
    record: Mapping[str, Any],
    *,
    raw_response: Mapping[str, Any],
    raw_text: str,
    prompt_token_roundtrip: bool,
) -> dict[str, Any]:
    """Classify one normalized row at its earliest observable bad boundary."""

    workload_id = str(record.get("workload_id", ""))
    turn_index = int(record.get("turn_index", -1))
    if workload_id not in suite.workloads:
        raise AgenticBenchmarkError(f"taxonomy record has unknown workload {workload_id!r}")
    turns = suite.workloads[workload_id]["turns"]
    if turn_index < 0 or turn_index >= len(turns):
        raise AgenticBenchmarkError("taxonomy record turn_index is outside workload")
    quality = record.get("quality")
    output = record.get("output")
    finish = record.get("finish")
    if not all(isinstance(value, Mapping) for value in (quality, output, finish)):
        raise AgenticBenchmarkError("taxonomy record quality/output/finish must be objects")
    expected = turns[turn_index]
    expected_tool = str(expected["expected_tool"])
    expected_arguments = expected["expected_arguments"]
    parsed = parse_independent_tool_envelope(raw_text)
    runtime_error = _raw_response_error(raw_response)
    selected, arguments = _public_projection(record)
    raw_choices = raw_response.get("choices")
    raw_choice = (
        raw_choices[0]
        if _is_sequence(raw_choices) and raw_choices and isinstance(raw_choices[0], Mapping)
        else {}
    )
    backend_finish_details = raw_choice.get("finish_details")
    backend_finish_details = (
        copy.deepcopy(dict(backend_finish_details))
        if isinstance(backend_finish_details, Mapping)
        else None
    )
    finish_reason = str(finish.get("reason", ""))
    public_content = str(output.get("public_content", ""))
    raw_markup_leaked = bool(output.get("raw_markup_leaked", False))
    tool_call_count = int(quality.get("tool_call_count", 0))
    contributing: list[str] = []

    if runtime_error is not None:
        primary = "runtime_error"
        boundary = "http_or_generation_response"
        contributing.append(runtime_error)
    elif not prompt_token_roundtrip:
        primary = "template_or_tokenizer_mismatch"
        boundary = "rendered_prompt_token_roundtrip"
    elif parsed.get("accepted") and not _parser_projection_matches(parsed, record):
        primary = "parser_mismatch"
        boundary = "raw_envelope_to_public_projection"
    elif not parsed.get("accepted") and tool_call_count:
        primary = "parser_mismatch"
        boundary = "raw_envelope_to_public_projection"
        contributing.append(str(parsed.get("error") or "independent_parse_failed"))
    elif raw_markup_leaked or any(marker in public_content for marker in _RAW_MARKERS):
        primary = "raw_markup_leak"
        boundary = "public_message_projection"
    elif public_content.strip() and tool_call_count:
        primary = "content_alongside_tool_call"
        boundary = "public_message_projection"
    elif not parsed.get("accepted"):
        parse_error = str(parsed.get("error") or "unresolved")
        if finish_reason == "length":
            primary = "length_exhausted"
            boundary = "model_generation_limit"
            contributing.append(parse_error)
        elif parse_error in {"invalid_json", "malformed_envelope"}:
            primary = parse_error
            boundary = "model_generated_envelope"
        elif tool_call_count == 0:
            primary = "no_tool_call"
            boundary = "model_tool_selection"
        else:
            primary = "unresolved"
            boundary = "insufficient_raw_parse_evidence"
    elif len(parsed["tool_calls"]) != 1:
        primary = "malformed_envelope"
        boundary = "model_generated_envelope"
        contributing.append("multiple_tool_calls")
    elif selected not in suite.tools:
        primary = "undeclared_tool"
        boundary = "model_tool_selection"
    elif not bool(quality.get("arguments_json_valid", False)):
        primary = "invalid_json"
        boundary = "model_generated_arguments"
    elif arguments is None or not _arguments_match_schema(
        arguments,
        suite.tools[selected]["parameters"],
    ):
        primary = "schema_violation"
        boundary = "model_generated_arguments"
    elif selected != expected_tool:
        primary = "wrong_tool"
        boundary = "model_tool_selection"
    else:
        external = quality.get("external_oracle")
        external_passed = (
            isinstance(external, Mapping) and external.get("passed") is True
        )
        if isinstance(external, Mapping) and not external_passed:
            contributing.append("external_oracle_failure")
            if arguments != expected_arguments:
                primary = "wrong_arguments"
                boundary = "model_generated_arguments"
            else:
                primary = "external_oracle_failure"
                boundary = "tool_execution_or_oracle"
        elif arguments != expected_arguments and external is None:
            primary = "wrong_arguments"
            boundary = "model_generated_arguments"
        else:
            primary = "passed"
            boundary = "none"

    owner = (
        "none"
        if primary == "passed"
        else "runtime_implementation"
        if primary in _RUNTIME_OUTCOMES
        else "unresolved"
        if primary == "unresolved"
        else "model_quality"
    )
    return {
        "primary_outcome": primary,
        "owner": owner,
        "earliest_bad_boundary": boundary,
        "contributing_causes": contributing,
        "independent_parser": {
            "accepted": bool(parsed.get("accepted")),
            "error": parsed.get("error"),
            "projection_matches_public": _parser_projection_matches(parsed, record),
            "tool_call_count": len(parsed.get("tool_calls", ())),
            "raw_text_sha256": _text_sha256(str(raw_text)),
        },
        "prompt_token_roundtrip": bool(prompt_token_roundtrip),
        "expected": {
            "tool": expected_tool,
            "arguments": copy.deepcopy(dict(expected_arguments)),
        },
        "observed": {
            "tool": selected,
            "arguments": None if arguments is None else copy.deepcopy(dict(arguments)),
            "finish_reason": finish_reason,
            "finish_detail_reason": finish.get("detail_reason"),
            "generated_token_ids_sha256": output.get("generated_token_ids_sha256"),
            "schema_valid": bool(quality.get("schema_valid", False)),
            "repair_count": int(quality.get("repair_count", 0)),
            "backend_finish_details": backend_finish_details,
            "public_content_sha256": _text_sha256(public_content),
            "external_oracle": (
                None
                if not isinstance(quality.get("external_oracle"), Mapping)
                else {
                    key: quality["external_oracle"].get(key)
                    for key in (
                        "kind",
                        "evaluated",
                        "passed",
                        "result_sha256",
                        "expected_result_sha256",
                        "patch_applied",
                        "tests_passed",
                        "error",
                    )
                }
            ),
        },
    }


__all__ = [
    "classify_agentic_quality_observation",
    "parse_independent_tool_envelope",
]
