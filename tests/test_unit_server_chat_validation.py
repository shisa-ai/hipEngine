"""Single-source chat validation contract for chat and snapshot consumers."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from hipengine.server.api import OpenAIHTTPError, render_chat_prompt
from hipengine.server.chat_sessions import SessionStateError
from hipengine.server.chat_validation import (
    CHAT_MESSAGE_ROLES,
    ChatValidationError,
    chat_message_role,
)
from hipengine.server.session_snapshots import (
    CHAT_SESSION_SNAPSHOT_SCHEMA,
    SessionSnapshotModel,
    restore_chat_session_from_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
API_PATH = REPO_ROOT / "hipengine" / "server" / "api.py"
SNAPSHOT_PATH = REPO_ROOT / "hipengine" / "server" / "session_snapshots.py"
VALIDATION_PATH = REPO_ROOT / "hipengine" / "server" / "chat_validation.py"

SHARED_FUNCTIONS = {
    "chat_message_role",
    "validate_chat_message_tool_fields",
    "chat_message_tool_calls",
    "chat_message_tool_call",
    "validate_chat_message_tool_transcript",
    "validate_chat_message_tool_transcript_entry",
}

_MODEL = SessionSnapshotModel(
    id="fake-model",
    backend="test-backend",
    quant="test-quant",
)


def _defined_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def _imported_from(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.update(alias.name for alias in node.names)
    return names


def _snapshot() -> dict[str, Any]:
    return {
        "object": "hipengine.session.snapshot",
        "schema": CHAT_SESSION_SNAPSHOT_SCHEMA,
        "model": {
            "id": "fake-model",
            "backend": "test-backend",
            "quant": "test-quant",
        },
        "tokenizer": {
            "name": "FakeTokenizer",
            "tokenize": True,
            "detokenize": True,
            "count_tokens": True,
        },
        "session": {
            "id": "sess",
            "storage": "app_local_transcript",
            "resident_state_reuse": False,
            "includes_transcript": True,
            "created": 10,
            "updated": 20,
        },
        "resident_state_reuse": False,
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_shared_validator_raises_one_typed_error_shape() -> None:
    with pytest.raises(ChatValidationError) as caught:
        chat_message_role("critic", param="messages[0].role")

    error = caught.value
    assert error.status_code == 400
    assert error.code == "invalid_request"
    assert error.param == "messages[0].role"
    assert error.message == (
        f"messages[0].role must be one of: {', '.join(CHAT_MESSAGE_ROLES)}"
    )


def test_chat_prompt_translates_shared_role_error_to_http() -> None:
    with pytest.raises(OpenAIHTTPError) as caught:
        render_chat_prompt([{"role": "critic", "content": "hi"}])

    error = caught.value
    assert error.status_code == 400
    assert error.code == "invalid_request"
    assert error.param == "messages[0].role"
    assert error.message == (
        f"messages[0].role must be one of: {', '.join(CHAT_MESSAGE_ROLES)}"
    )


def test_chat_prompt_translates_shared_tool_field_error_to_http() -> None:
    with pytest.raises(OpenAIHTTPError) as caught:
        render_chat_prompt(
            [
                {
                    "role": "user",
                    "content": "hi",
                    "tool_calls": [
                        {
                            "id": "call_bad",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                }
            ]
        )

    error = caught.value
    assert error.status_code == 400
    assert error.code == "invalid_request"
    assert error.param == "messages[0].tool_calls"
    assert error.message == (
        "messages[0].tool_calls is only supported on assistant messages"
    )


def test_chat_prompt_translates_shared_tool_call_error_to_http() -> None:
    with pytest.raises(OpenAIHTTPError) as caught:
        render_chat_prompt(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_bad",
                            "type": "custom",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                }
            ]
        )

    error = caught.value
    assert error.status_code == 400
    assert error.code == "invalid_request"
    assert error.param == "messages[0].tool_calls[0].type"
    assert error.message == (
        "chat message messages[0].tool_calls[0].type must be 'function'"
    )


def test_chat_prompt_translates_shared_transcript_error_to_http() -> None:
    with pytest.raises(OpenAIHTTPError) as caught:
        render_chat_prompt(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_read",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "user", "content": "skip the tool"},
            ]
        )

    error = caught.value
    assert error.status_code == 400
    assert error.code == "invalid_request"
    assert error.param == "messages[1].role"
    assert error.message == (
        "chat message messages[1].role must be 'tool' until prior assistant "
        "tool calls are resolved"
    )


def test_snapshot_restore_translates_shared_tool_call_error_to_state_error() -> None:
    snapshot = _snapshot()
    snapshot["messages"] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_bad",
                    "type": "custom",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        }
    ]

    with pytest.raises(SessionStateError) as caught:
        restore_chat_session_from_snapshot(
            "sess",
            snapshot,
            schema=CHAT_SESSION_SNAPSHOT_SCHEMA,
            model=_MODEL,
            current_tokenizer_metadata=lambda: None,
        )

    error = caught.value
    assert error.status_code == 400
    assert error.code == "invalid_request"
    assert error.param == "messages[0].tool_calls[0].type"
    assert error.message == (
        "session snapshot messages[0].tool_calls[0].type must be 'function'"
    )


def test_shared_chat_validation_is_the_single_source() -> None:
    shared = _defined_names(VALIDATION_PATH)
    api = _defined_names(API_PATH)
    snapshot = _defined_names(SNAPSHOT_PATH)

    assert SHARED_FUNCTIONS <= shared
    assert SHARED_FUNCTIONS.isdisjoint(api)
    assert SHARED_FUNCTIONS.isdisjoint(snapshot)
    assert "CHAT_MESSAGE_ROLES" in shared
    assert "CHAT_MESSAGE_ROLES" not in api
    assert "CHAT_MESSAGE_ROLES" not in snapshot
    assert {
        "_chat_message_role",
        "_validate_chat_message_tool_fields",
        "_chat_message_tool_calls",
        "_chat_message_tool_call",
        "_validate_chat_message_tool_transcript",
        "_validate_chat_message_tool_transcript_entry",
    }.isdisjoint(api)
    assert _imported_from(API_PATH, "hipengine.server.chat_validation") == {
        "ChatValidationError",
        "chat_message_role",
        "chat_message_tool_calls",
        "validate_chat_message_tool_fields",
        "validate_chat_message_tool_transcript_entry",
    }
    assert _imported_from(SNAPSHOT_PATH, "hipengine.server.chat_validation") == {
        "ChatValidationError",
        "chat_message_role",
        "chat_message_tool_calls",
        "validate_chat_message_tool_fields",
        "validate_chat_message_tool_transcript",
    }


def test_shared_validation_owner_is_pure() -> None:
    source = VALIDATION_PATH.read_text(encoding="utf-8")
    modules = _imported_modules(VALIDATION_PATH)

    assert "app.state" not in source
    assert "ServerConfig" not in source
    assert "fastapi" not in modules
    assert "hipengine.server.api" not in modules
    assert "hipengine.llm" not in modules
    assert "hipengine.server.config" not in modules
