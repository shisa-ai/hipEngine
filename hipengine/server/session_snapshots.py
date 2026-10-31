"""Pure chat-session snapshot payload construction and restore validation.

The owner receives explicit model identity and a narrow tokenizer-metadata
operation. It does not import the HTTP application, an engine, server
configuration or application state. Restore failures are typed
``SessionStateError`` values that the application's existing session-error seam
translates.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hipengine.server.chat_sessions import (
    ChatSessionRecord,
    SessionStateError,
    _chat_session_message_copy,
    session_metadata,
)
from hipengine.server.chat_validation import (
    ChatValidationError,
    chat_message_role,
    chat_message_tool_calls,
    validate_chat_message_tool_fields,
    validate_chat_message_tool_transcript,
)

CHAT_SESSION_SNAPSHOT_SCHEMA = "hipengine.chat_session_snapshot.v1"


@dataclass(frozen=True)
class SessionSnapshotModel:
    id: str
    backend: str
    quant: str

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "backend": self.backend,
            "quant": self.quant,
        }


def build_chat_session_snapshot(
    record: ChatSessionRecord,
    *,
    schema: str,
    model: SessionSnapshotModel,
    tokenizer_metadata: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "object": "hipengine.session.snapshot",
        "schema": schema,
        "model": model.as_dict(),
        "tokenizer": tokenizer_metadata(),
        "session": {
            **session_metadata(record),
            "includes_transcript": True,
        },
        "resident_state_reuse": False,
        "messages": [_chat_session_message_copy(message) for message in record.messages],
    }


def restore_chat_session_from_snapshot(
    session_id: str,
    snapshot: Mapping[str, Any],
    *,
    schema: str,
    model: SessionSnapshotModel,
    current_tokenizer_metadata: Callable[[], Mapping[str, Any] | None],
) -> ChatSessionRecord:
    if snapshot.get("object") != "hipengine.session.snapshot":
        raise _snapshot_error(
            400,
            "session snapshot object must be hipengine.session.snapshot",
            param="object",
        )
    if snapshot.get("schema") != schema:
        raise _snapshot_error(
            400,
            f"session snapshot schema must be {schema!r}",
            param="schema",
        )
    if snapshot.get("resident_state_reuse") is not False:
        raise _snapshot_error(
            400,
            "session snapshot resident_state_reuse must be false",
            param="resident_state_reuse",
        )
    model_payload = snapshot.get("model")
    if not isinstance(model_payload, Mapping):
        raise _snapshot_error(
            400,
            "session snapshot model must be an object",
            param="model",
        )
    for key, expected in model.as_dict().items():
        if model_payload.get(key) != expected:
            raise _snapshot_error(
                400,
                f"session snapshot model.{key} is incompatible with this server",
                param=f"model.{key}",
            )
    _validate_snapshot_tokenizer_metadata(
        snapshot.get("tokenizer"),
        current_tokenizer_metadata=current_tokenizer_metadata,
    )
    session = snapshot.get("session")
    if not isinstance(session, Mapping):
        raise _snapshot_error(
            400,
            "session snapshot session must be an object",
            param="session",
        )
    if session.get("storage") != "app_local_transcript":
        raise _snapshot_error(
            400,
            "session snapshot storage must be app_local_transcript",
            param="session.storage",
        )
    if session.get("resident_state_reuse") is not False:
        raise _snapshot_error(
            400,
            "session snapshot resident_state_reuse must be false",
            param="session.resident_state_reuse",
        )
    if session.get("includes_transcript") is not True:
        raise _snapshot_error(
            400,
            "session snapshot includes_transcript must be true",
            param="session.includes_transcript",
        )
    snapshot_id = session.get("id")
    if snapshot_id != session_id:
        raise _snapshot_error(
            400,
            "session snapshot id must match the restore path",
            param="session.id",
        )
    try:
        messages = _chat_session_snapshot_messages(snapshot.get("messages"))
    except ChatValidationError as exc:
        raise _snapshot_validation_error(exc) from exc
    created = _chat_session_snapshot_time(session.get("created"), param="session.created")
    updated = _chat_session_snapshot_time(session.get("updated"), param="session.updated")
    return ChatSessionRecord(
        id=session_id,
        messages=messages,
        created=created,
        updated=updated,
    )


def _snapshot_error(status_code: int, message: str, *, param: str) -> SessionStateError:
    return SessionStateError(
        status_code,
        message,
        code="invalid_request",
        param=param,
    )


def _snapshot_validation_error(exc: ChatValidationError) -> SessionStateError:
    return SessionStateError(
        exc.status_code,
        exc.message,
        code=exc.code,
        param=exc.param,
    )


def _validate_snapshot_tokenizer_metadata(
    value: Any,
    *,
    current_tokenizer_metadata: Callable[[], Mapping[str, Any] | None],
) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise _snapshot_error(
            400,
            "session snapshot tokenizer must be an object",
            param="tokenizer",
        )
    expected = current_tokenizer_metadata()
    if expected is None:
        return
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise _snapshot_error(
                400,
                f"session snapshot tokenizer.{key} is incompatible with this server",
                param=f"tokenizer.{key}",
            )


def _chat_session_snapshot_messages(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _snapshot_error(
            400,
            "session snapshot messages must be an array",
            param="messages",
        )
    messages = tuple(
        _chat_session_snapshot_message(item, index=index)
        for index, item in enumerate(value)
    )
    validate_chat_message_tool_transcript(messages, context="session snapshot")
    return messages


def _chat_session_snapshot_message(value: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _snapshot_error(
            400,
            f"session snapshot messages[{index}] must be an object",
            param=f"messages[{index}]",
        )
    allowed = {"role", "content", "name", "tool_call_id", "tool_calls"}
    extra = sorted(str(key) for key in value if str(key) not in allowed)
    if extra:
        raise _snapshot_error(
            400,
            f"session snapshot messages[{index}].{extra[0]} is not supported",
            param=f"messages[{index}].{extra[0]}",
        )
    role = chat_message_role(value.get("role"), param=f"messages[{index}].role")
    validate_chat_message_tool_fields(
        role,
        tool_calls=value.get("tool_calls"),
        tool_call_id=value.get("tool_call_id"),
        param_prefix=f"messages[{index}]",
    )
    payload: dict[str, Any] = {"role": role}
    if "content" in value:
        payload["content"] = _chat_session_snapshot_content(
            value.get("content"),
            message_index=index,
        )
    for key in ("name", "tool_call_id"):
        if key in value:
            payload[key] = _chat_session_snapshot_string(
                value.get(key),
                param=f"messages[{index}].{key}",
            )
    if "tool_calls" in value:
        payload["tool_calls"] = chat_message_tool_calls(
            value.get("tool_calls"),
            message_index=index,
            context="session snapshot",
        )
    return payload


def _chat_session_snapshot_string(value: Any, *, param: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _snapshot_error(
            400,
            f"session snapshot {param} must be a non-empty string",
            param=param,
        )
    return value


def _chat_session_snapshot_content(
    value: Any,
    *,
    message_index: int,
) -> str | list[Any] | None:
    param = f"messages[{message_index}].content"
    if value is None or isinstance(value, str):
        return value
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _snapshot_error(
            400,
            f"session snapshot {param} must be text, null, or text content parts",
            param=param,
        )
    parts: list[Any] = []
    for part_index, part in enumerate(value):
        part_param = f"{param}[{part_index}]"
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, Mapping):
            raise _snapshot_error(
                400,
                f"session snapshot {part_param} must be a text object",
                param=part_param,
            )
        part_type = part.get("type", "text")
        if part_type != "text":
            raise _snapshot_error(
                400,
                f"session snapshot {part_param}.type must be 'text'",
                param=part_param,
            )
        text = part.get("text", "")
        if not isinstance(text, str):
            raise _snapshot_error(
                400,
                f"session snapshot {part_param}.text must be a string",
                param=f"{part_param}.text",
            )
        parts.append(dict(part))
    return parts


def _chat_session_snapshot_time(value: Any, *, param: str) -> float:
    if isinstance(value, bool):
        value = None
    if not isinstance(value, (int, float)) or float(value) < 0:
        raise _snapshot_error(
            400,
            f"session snapshot {param} must be a non-negative timestamp",
            param=param,
        )
    return float(value)
