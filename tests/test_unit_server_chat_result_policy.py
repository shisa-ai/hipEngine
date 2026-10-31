"""Tool and structured chat-result validation golden contract.

The fixture was captured from ``16864d1f9`` before the result-validation owner
was extracted.  It pins the exact failure strings, parsed results, public
exception fields, and the control-helper call order that keeps request
normalization lazy on inactive, length, and tool-call branches.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from hipengine.server import api, chat_result_policy
from hipengine.server.chat_output import _ParsedChatOutput, _ParsedToolCall
from hipengine.server.schemas import ChatCompletionRequest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "chat_result_policy_golden.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
SOURCE_HEAD = "16864d1f9"

CONTROL_HELPERS = (
    "_response_format_mode",
    "_response_format_json_schema",
    "_guided_json_mode_from_value",
    "_guided_json_schema",
    "_guided_regex_result_validation",
    "_guided_regex_pattern",
    "_guided_choice_result_validation",
    "_guided_choice_values",
    "_guided_patch_result_validation",
    "_guided_patch_fenced_policy",
    "_structured_json_object_prefix_validation",
    "json_schema_root_object",
)
TRACE_LABELS = {"json_schema_root_object": "_json_schema_root_object"}


def _request(data: dict) -> ChatCompletionRequest:
    return ChatCompletionRequest(**data)


def _parsed_from_fixture(data: dict) -> _ParsedChatOutput:
    return _ParsedChatOutput(
        text=data["text"],
        tool_calls=tuple(_ParsedToolCall(**item) for item in data["tool_calls"]),
        reasoning_initially_open=data["reasoning_initially_open"],
        tool_parser_name=data["tool_parser_name"],
        invalid_tool_call_blocks=tuple(data["invalid_tool_call_blocks"]),
    )


def _parsed_json(value: _ParsedChatOutput) -> dict:
    return {
        "text": value.text,
        "tool_calls": [
            {
                "id": item.id,
                "name": item.name,
                "arguments": item.arguments,
                "raw_text": item.raw_text,
            }
            for item in value.tool_calls
        ],
        "reasoning_initially_open": value.reasoning_initially_open,
        "tool_parser_name": value.tool_parser_name,
        "invalid_tool_call_blocks": list(value.invalid_tool_call_blocks),
    }


def _tool_policy(request: ChatCompletionRequest) -> chat_result_policy.ToolResultPolicy:
    if not request.tools:
        return chat_result_policy.ToolResultPolicy(
            tools=(),
            choice_mode="auto",
            choice_name=None,
            parallel_tool_calls=request.parallel_tool_calls,
            strict=False,
        )
    mode, name = chat_result_policy.tool_choice_mode(request.tool_choice)
    strict = False
    if mode in {"none", "required", "function"}:
        strict = True
    elif request.parallel_tool_calls is not None:
        strict = True
    else:
        strict = any(
            chat_result_policy.tool_function(tool).get("strict") is True
            for tool in request.tools
        )
    return chat_result_policy.ToolResultPolicy(
        tools=tuple(request.tools),
        choice_mode=mode,
        choice_name=name,
        parallel_tool_calls=request.parallel_tool_calls,
        strict=strict,
    )


def _install_control_trace(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    trace: list[str] = []
    for name in CONTROL_HELPERS:
        original = getattr(api, name)

        def wrapper(*args, __name=name, __original=original, **kwargs):
            trace.append(TRACE_LABELS.get(__name, __name))
            return __original(*args, **kwargs)

        monkeypatch.setattr(api, name, wrapper)
    return trace


def _exception_fields(exc: BaseException) -> dict:
    return {
        "raised": True,
        "type": type(exc).__name__,
        "message": str(exc),
        "status_code": getattr(exc, "status_code", None),
        "code": getattr(exc, "code", None),
        "param": getattr(exc, "param", None),
        "error_type": getattr(exc, "error_type", None),
    }


def test_fixture_was_captured_from_the_pre_move_head() -> None:
    assert FIXTURE["source_head"] == SOURCE_HEAD


@pytest.mark.parametrize("name", sorted(FIXTURE["tool_cases"]))
def test_tool_result_application_wrapper_matches_old_head(name: str) -> None:
    case = FIXTURE["tool_cases"][name]
    result = api._validate_chat_tool_result(
        _request(case["request"]),
        _parsed_from_fixture(case["parsed"]),
        case["raw_text"],
    )
    assert result.failure_reason == case["expected"]["failure_reason"]
    assert _parsed_json(result.parsed) == case["expected"]["parsed"]


@pytest.mark.parametrize("name", sorted(FIXTURE["tool_cases"]))
def test_tool_result_owner_matches_old_head(name: str) -> None:
    case = FIXTURE["tool_cases"][name]
    request = _request(case["request"])
    result = chat_result_policy.validate_tool_result(
        _tool_policy(request),
        _parsed_from_fixture(case["parsed"]),
        case["raw_text"],
    )
    assert result.failure_reason == case["expected"]["failure_reason"]
    assert _parsed_json(result.parsed) == case["expected"]["parsed"]


@pytest.mark.parametrize("name", sorted(FIXTURE["structured_cases"]))
def test_structured_application_wrapper_matches_old_head(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = FIXTURE["structured_cases"][name]
    trace = _install_control_trace(monkeypatch)
    request = _request(case["request"])
    try:
        if case["function"] == "structured_output":
            value = api._structured_output_failure_reason(
                request, case["text"], case["finish_reason"]
            )
        else:
            value = api._structured_length_failure_reason(
                request, case["text"], case["finish_reason"]
            )
        actual = {"raised": False, "failure_reason": value}
    except BaseException as exc:  # the fixture pins public exception fields
        actual = _exception_fields(exc)
    assert trace == case["trace"]
    assert actual == case["expected"]


def test_response_format_owner_validation() -> None:
    object_policy = chat_result_policy.StructuredResultPolicy(
        response_format_mode="json_object"
    )
    parsed, value = chat_result_policy.parse_json_text('{"a": 1}')
    assert parsed is True
    assert chat_result_policy.response_format_failure(object_policy, value) is None
    parsed, value = chat_result_policy.parse_json_text("[1]")
    assert parsed is True
    assert (
        chat_result_policy.response_format_failure(object_policy, value)
        == "schema_violation"
    )
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
    }
    schema_policy = chat_result_policy.StructuredResultPolicy(
        response_format_mode="json_schema",
        response_format_schema=schema,
    )
    parsed, value = chat_result_policy.parse_json_text('{"a": 1}')
    assert parsed is True
    assert chat_result_policy.response_format_failure(schema_policy, value) is None
    parsed, value = chat_result_policy.parse_json_text('{"a": "x"}')
    assert parsed is True
    assert (
        chat_result_policy.response_format_failure(schema_policy, value)
        == "schema_violation"
    )
    parsed, value = chat_result_policy.parse_json_text("not json")
    assert parsed is False


def test_guided_json_owner_validation() -> None:
    object_policy = chat_result_policy.StructuredResultPolicy(
        guided_json_mode="json_object"
    )
    parsed, value = chat_result_policy.parse_json_text('{"a": 1}')
    assert parsed is True
    assert chat_result_policy.guided_json_failure(object_policy, value) is None
    parsed, value = chat_result_policy.parse_json_text("[1]")
    assert parsed is True
    assert (
        chat_result_policy.guided_json_failure(object_policy, value)
        == "schema_violation"
    )
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
    }
    schema_policy = chat_result_policy.StructuredResultPolicy(
        guided_json_mode="json_schema",
        guided_json_schema=schema,
    )
    parsed, value = chat_result_policy.parse_json_text('{"a": 1}')
    assert parsed is True
    assert chat_result_policy.guided_json_failure(schema_policy, value) is None
    parsed, value = chat_result_policy.parse_json_text('{"a": "x"}')
    assert parsed is True
    assert (
        chat_result_policy.guided_json_failure(schema_policy, value)
        == "schema_violation"
    )


def test_guided_regex_choice_and_patch_owner_validation() -> None:
    regex_policy = chat_result_policy.StructuredResultPolicy(guided_regex="^a+$")
    assert chat_result_policy.guided_regex_failure(regex_policy, "aaa") is None
    assert (
        chat_result_policy.guided_regex_failure(regex_policy, "b")
        == "schema_violation"
    )
    choice_policy = chat_result_policy.StructuredResultPolicy(
        guided_choice=("a", "b")
    )
    assert chat_result_policy.guided_choice_failure(choice_policy, "a") is None
    assert (
        chat_result_policy.guided_choice_failure(choice_policy, "c")
        == "schema_violation"
    )
    diff = "--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new"
    fenced = f"```diff\n{diff}\n```"
    required = chat_result_policy.StructuredResultPolicy(
        guided_patch_fenced_policy="required"
    )
    assert chat_result_policy.guided_patch_failure(required, fenced) is None
    assert (
        chat_result_policy.guided_patch_failure(required, diff)
        == "schema_violation"
    )
    forbidden = chat_result_policy.StructuredResultPolicy(
        guided_patch_fenced_policy="forbidden"
    )
    assert chat_result_policy.guided_patch_failure(forbidden, diff) is None
    assert (
        chat_result_policy.guided_patch_failure(forbidden, fenced)
        == "schema_violation"
    )


def test_structured_length_owner_validation_and_root_object() -> None:
    assert chat_result_policy.structured_length_failure("[") == "schema_violation"
    assert chat_result_policy.structured_length_failure('{"a":') is None
    assert chat_result_policy.json_schema_root_object({"type": "object"}) is True
    assert chat_result_policy.json_schema_root_object({"properties": {}}) is True
    assert chat_result_policy.json_schema_root_object({"type": "array"}) is False
    assert chat_result_policy.json_schema_root_object(None) is False


def test_chat_result_policy_owner_has_no_application_or_request_dependencies() -> None:
    tree = ast.parse(Path(chat_result_policy.__file__).read_text(encoding="utf-8"))
    forbidden_imports = {
        "hipengine.server.api",
        "fastapi",
        "starlette",
        "hipengine.server.schemas",
        "hipengine.server.tool_stream",
    }
    forbidden_names = {
        "ChatCompletionRequest",
        "CompletionRequest",
        "OpenAIHTTPError",
        "ServerConfig",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden_imports, node.module
            for alias in node.names:
                assert alias.name not in forbidden_names, alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden_imports, alias.name
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert not isinstance(node, ast.AsyncFunctionDef), node.name
            arguments = [
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            ]
            assert not ({argument.arg for argument in arguments} & {
                "app",
                "config",
                "request",
                "engine",
            }), node.name
