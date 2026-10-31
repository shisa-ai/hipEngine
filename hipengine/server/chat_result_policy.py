"""Pure tool and structured chat-result validation.

This module owns the result-side checks applied after chat output has been
parsed: tool-call validation against declared tools, structured-output failure
reasons, guided-patch shape validation, and the JSON-object length prefix check.
Callers pass explicit normalized policy records plus the pure parser output;
this module does not import requests, the HTTP application, an engine, or any
application state.  Request normalization and branch short-circuits stay in the
application so inactive fields are not evaluated early.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import re
from typing import Any

from hipengine.generation.constraints import JsonObjectConstraintState
from hipengine.server.chat_output import (
    _ParsedChatOutput,
    _ParsedToolCall,
    _TOOL_CALL_END_MARKER,
    _TOOL_CALL_START_MARKER,
    _find_marker,
    _valid_tool_call_blocks,
)
from hipengine.server.json_schema import _validate_json_schema_value


_PATCH_FENCE_RE = re.compile(r"\A\s*```(?P<label>[^\n`]*)\n(?P<body>.*?)\n```\s*\Z", re.DOTALL)
_UNIFIED_DIFF_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@(?: .*)?$")
_UNIFIED_DIFF_METADATA_PREFIXES = (
    "diff --git ",
    "index ",
    "new file mode ",
    "deleted file mode ",
    "old mode ",
    "new mode ",
    "similarity index ",
    "dissimilarity index ",
    "rename from ",
    "rename to ",
    "copy from ",
    "copy to ",
)
_UNIFIED_DIFF_BINARY_PREFIXES = ("Binary files ", "GIT binary patch")


@dataclass(frozen=True, slots=True)
class ToolResultPolicy:
    """Normalized tool result policy for one chat request."""

    tools: tuple[Mapping[str, Any], ...]
    choice_mode: str
    choice_name: str | None
    parallel_tool_calls: bool | None
    strict: bool


@dataclass(frozen=True, slots=True)
class StructuredResultPolicy:
    """Normalized structured result policy for the active validation branch."""

    response_format_mode: str | None = None
    response_format_schema: Mapping[str, Any] | None = None
    guided_json_mode: str | None = None
    guided_json_schema: Mapping[str, Any] | None = None
    guided_regex: str | None = None
    guided_choice: tuple[str, ...] = ()
    guided_patch_fenced_policy: str | None = None


@dataclass(frozen=True, slots=True)
class ToolValidationResult:
    parsed: _ParsedChatOutput
    failure_reason: str | None = None

    @property
    def failed(self) -> bool:
        return self.failure_reason is not None


def validate_tool_result(
    policy: ToolResultPolicy,
    parsed: _ParsedChatOutput,
    raw_text: str,
) -> ToolValidationResult:
    if not policy.tools:
        return ToolValidationResult(parsed)
    if parsed.tool_parser_name is not None:
        malformed_blocks = parsed.invalid_tool_call_blocks if policy.strict else ()
        unparseable_blocks = parsed.invalid_tool_call_blocks
    else:
        malformed_blocks = (
            _malformed_tool_call_blocks(raw_text, tools=policy.tools)
            if policy.strict
            else ()
        )
        unparseable_blocks = _unparseable_tool_call_blocks(raw_text, tools=policy.tools)
    if policy.strict:
        if policy.choice_mode == "none":
            if parsed.tool_calls or malformed_blocks:
                return _tool_validation_failure("invalid_tool_call")
            return ToolValidationResult(parsed)
        if malformed_blocks:
            return _tool_validation_failure("invalid_tool_call")
    if unparseable_blocks:
        return _tool_validation_failure("invalid_tool_call")
    if not parsed.tool_calls:
        if policy.strict and policy.choice_mode in {"required", "function"}:
            return _tool_validation_failure("tool_required_not_satisfied")
        return ToolValidationResult(parsed)
    if len(parsed.tool_calls) > 1 and policy.parallel_tool_calls is False:
        return _tool_validation_failure("invalid_tool_call")

    tools_by_name = tool_map_by_name(policy.tools)
    for call in parsed.tool_calls:
        if (
            policy.choice_mode == "function"
            and policy.choice_name is not None
            and call.name != policy.choice_name
        ):
            return _tool_validation_failure("invalid_tool_call")
        tool = tools_by_name.get(call.name)
        if tool is None:
            return _tool_validation_failure("invalid_tool_call")
        if policy.strict:
            schema = tool_parameters_schema(tool)
            if schema is not None:
                schema_error = _validate_json_schema_value(
                    _tool_call_arguments_value(call),
                    schema,
                    path=f"{call.name}.arguments",
                )
                if schema_error is not None:
                    return _tool_validation_failure("schema_violation")
    if (
        policy.strict
        and policy.choice_mode == "function"
        and policy.choice_name is not None
        and not any(call.name == policy.choice_name for call in parsed.tool_calls)
    ):
        return _tool_validation_failure("tool_required_not_satisfied")
    return ToolValidationResult(parsed)


def _tool_validation_failure(reason: str) -> ToolValidationResult:
    return ToolValidationResult(_ParsedChatOutput(text="", tool_calls=()), str(reason))


def tool_choice_mode(
    tool_choice: str | Mapping[str, Any] | None,
) -> tuple[str, str | None]:
    if tool_choice is None:
        return "auto", None
    if isinstance(tool_choice, str):
        value = tool_choice.strip().lower()
        if value in {"none", "auto", "required"}:
            return value, None
        return "function", tool_choice.strip()
    if isinstance(tool_choice, Mapping):
        choice_type = str(tool_choice.get("type", "")).strip().lower()
        function = tool_choice.get("function")
        if choice_type == "function" and isinstance(function, Mapping):
            name = function.get("name")
            if isinstance(name, str) and name:
                return "function", name
    return "auto", None


def tool_map_by_name(
    tools: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    mapped: dict[str, Mapping[str, Any]] = {}
    for tool in tools:
        function = tool_function(tool)
        name = function.get("name")
        if isinstance(name, str) and name:
            mapped[name] = tool
    return mapped


def tool_function(tool: Mapping[str, Any]) -> Mapping[str, Any]:
    function = tool.get("function")
    return function if isinstance(function, Mapping) else {}


def tool_parameters_schema(tool: Mapping[str, Any]) -> Mapping[str, Any] | None:
    function = tool_function(tool)
    parameters = function.get("parameters")
    return parameters if isinstance(parameters, Mapping) else None


def _tool_call_arguments_value(call: _ParsedToolCall) -> Any:
    try:
        return json.loads(call.arguments)
    except Exception:
        return call.arguments


def _malformed_tool_call_blocks(
    text: str,
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str, ...]:
    return _invalid_tool_call_blocks(text, tools=tools)


def _unparseable_tool_call_blocks(
    text: str,
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str, ...]:
    return _invalid_tool_call_blocks(text, tools=tools)


def _invalid_tool_call_blocks(
    text: str,
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str, ...]:
    lowered = text.lower()
    invalid: list[str] = []
    cursor = 0
    blocks = _valid_tool_call_blocks(text, tools=tools)
    for block in blocks:
        invalid.extend(
            _invalid_tool_call_blocks_between(
                text, lowered, cursor, block.start, flag_unclosed=False
            )
        )
        cursor = block.end
    invalid.extend(
        _invalid_tool_call_blocks_between(
            text, lowered, cursor, len(text), flag_unclosed=True
        )
    )
    return tuple(invalid)


def _invalid_tool_call_blocks_between(
    text: str,
    lowered: str,
    start: int,
    end: int,
    *,
    flag_unclosed: bool,
) -> tuple[str, ...]:
    invalid: list[str] = []
    cursor = start
    while cursor < end:
        open_index = _find_marker(lowered, _TOOL_CALL_START_MARKER, cursor, end)
        if open_index < 0:
            break
        close_index = _find_marker(
            lowered,
            _TOOL_CALL_END_MARKER,
            open_index + len(_TOOL_CALL_START_MARKER),
            end,
        )
        if close_index < 0:
            if flag_unclosed:
                invalid.append(text[open_index:end])
            break
        close_end = close_index + len(_TOOL_CALL_END_MARKER)
        invalid.append(text[open_index:close_end])
        cursor = close_end
    return tuple(invalid)


def parse_json_text(text: str) -> tuple[bool, Any]:
    try:
        return True, json.loads(str(text).strip())
    except Exception:
        return False, None


def response_format_failure(
    policy: StructuredResultPolicy,
    value: Any,
) -> str | None:
    mode = policy.response_format_mode
    if mode not in {"json_object", "json_schema"}:
        return None
    if mode == "json_object":
        return None if isinstance(value, dict) else "schema_violation"
    schema = policy.response_format_schema
    if schema is None:
        return "schema_violation"
    return None if _validate_json_schema_value(value, schema, path="$") is None else "schema_violation"


def guided_json_failure(
    policy: StructuredResultPolicy,
    value: Any,
) -> str | None:
    mode = policy.guided_json_mode
    if mode not in {"json_object", "json_schema"}:
        return None
    if mode == "json_object":
        return None if isinstance(value, dict) else "schema_violation"
    schema = policy.guided_json_schema
    if schema is None:
        return "schema_violation"
    return None if _validate_json_schema_value(value, schema, path="$") is None else "schema_violation"


def guided_regex_failure(
    policy: StructuredResultPolicy,
    text: str,
) -> str | None:
    pattern = policy.guided_regex
    if pattern is None:
        return "schema_violation"
    try:
        valid = re.fullmatch(pattern, str(text).strip()) is not None
    except re.error:
        valid = False
    return None if valid else "schema_violation"


def guided_choice_failure(
    policy: StructuredResultPolicy,
    text: str,
) -> str | None:
    return None if str(text).strip() in policy.guided_choice else "schema_violation"


def guided_patch_failure(
    policy: StructuredResultPolicy,
    text: str,
) -> str | None:
    valid = _is_valid_guided_patch_text(
        text,
        fenced_policy=policy.guided_patch_fenced_policy or "optional",
    )
    return None if valid else "schema_violation"


def structured_length_failure(text: str) -> str | None:
    state = JsonObjectConstraintState().observe_text(text)
    return "schema_violation" if state.invalid else None


def json_schema_root_object(schema: Mapping[str, Any] | None) -> bool:
    if not isinstance(schema, Mapping):
        return False
    schema_type = schema.get("type")
    if schema_type == "object":
        return True
    return any(key in schema for key in ("properties", "required", "additionalProperties"))


def _is_valid_guided_patch_text(text: str, *, fenced_policy: str = "optional") -> bool:
    extracted = _extract_guided_patch_body(str(text))
    if extracted is None:
        return False
    patch, is_fenced = extracted
    if fenced_policy == "required" and not is_fenced:
        return False
    if fenced_policy == "forbidden" and is_fenced:
        return False
    return _is_valid_unified_diff(patch)


def _extract_guided_patch_body(text: str) -> tuple[str, bool] | None:
    stripped = str(text).strip()
    if not stripped:
        return None
    match = _PATCH_FENCE_RE.match(stripped)
    if match is None:
        return stripped, False
    label = match.group("label").strip().lower()
    if label and label not in {"diff", "patch"}:
        return None
    return match.group("body").strip(), True


def _is_valid_unified_diff(text: str) -> bool:
    lines = str(text).strip().splitlines()
    if not lines:
        return False
    index = 0
    file_count = 0
    hunk_count = 0
    while index < len(lines):
        while index < len(lines) and _is_unified_diff_metadata_line(lines[index]):
            index += 1
        if index < len(lines) and lines[index].startswith(_UNIFIED_DIFF_BINARY_PREFIXES):
            return False
        if index >= len(lines) or not lines[index].startswith("--- "):
            return False
        index += 1
        if index >= len(lines) or not lines[index].startswith("+++ "):
            return False
        file_count += 1
        index += 1
        file_hunks = 0
        while index < len(lines) and _UNIFIED_DIFF_HUNK_RE.match(lines[index]):
            file_hunks += 1
            hunk_count += 1
            index += 1
            body_lines = 0
            while index < len(lines):
                line = lines[index]
                if _UNIFIED_DIFF_HUNK_RE.match(line):
                    break
                if line.startswith("--- ") or _is_unified_diff_metadata_line(line):
                    break
                if not _is_unified_diff_body_line(line):
                    return False
                body_lines += 1
                index += 1
            if body_lines == 0:
                return False
        if file_hunks == 0:
            return False
    return file_count > 0 and hunk_count > 0


def _is_unified_diff_metadata_line(line: str) -> bool:
    return line.startswith(_UNIFIED_DIFF_METADATA_PREFIXES)


def _is_unified_diff_body_line(line: str) -> bool:
    return (
        bool(line)
        and line[0] in {" ", "+", "-"}
        or line.startswith("\\ No newline at end of file")
    )
