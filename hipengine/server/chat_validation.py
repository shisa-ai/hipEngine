"""Pure chat-message role and tool-transcript validation.

Both the HTTP chat-prompt path and the session-snapshot restore path consume
this owner. It raises typed ``ChatValidationError`` values; callers translate
them at their existing HTTP or snapshot error seam. It imports no FastAPI
application, engine, server config or application state.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

CHAT_MESSAGE_ROLES = ("assistant", "developer", "system", "tool", "user")
CHAT_MESSAGE_ROLE_SET = frozenset(CHAT_MESSAGE_ROLES)


class ChatValidationError(Exception):
    """Typed chat-message validation failure translated by the caller."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        code: str,
        param: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.message = message
        self.code = code
        self.param = param
        super().__init__(message)


def _validation_error(message: str, *, param: str) -> ChatValidationError:
    return ChatValidationError(
        400,
        message,
        code="invalid_request",
        param=param,
    )


def chat_message_role(value: Any, *, param: str) -> str:
    if not isinstance(value, str):
        raise _validation_error(
            f"{param} must be one of: {', '.join(CHAT_MESSAGE_ROLES)}",
            param=param,
        )
    role = value.strip()
    if role not in CHAT_MESSAGE_ROLE_SET:
        raise _validation_error(
            f"{param} must be one of: {', '.join(CHAT_MESSAGE_ROLES)}",
            param=param,
        )
    return role


def validate_chat_message_tool_fields(
    role: str,
    *,
    tool_calls: Any,
    tool_call_id: Any,
    param_prefix: str,
) -> None:
    if tool_calls is not None and role != "assistant":
        raise _validation_error(
            f"{param_prefix}.tool_calls is only supported on assistant messages",
            param=f"{param_prefix}.tool_calls",
        )
    if tool_call_id is not None and role != "tool":
        raise _validation_error(
            f"{param_prefix}.tool_call_id is only supported on tool messages",
            param=f"{param_prefix}.tool_call_id",
        )
    if role == "tool" and (not isinstance(tool_call_id, str) or not tool_call_id.strip()):
        raise _validation_error(
            f"{param_prefix}.tool_call_id is required for tool messages",
            param=f"{param_prefix}.tool_call_id",
        )


def chat_message_tool_calls(
    value: Any,
    *,
    message_index: int,
    context: str,
) -> list[dict[str, Any]]:
    param = f"messages[{message_index}].tool_calls"
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _validation_error(
            f"{context} {param} must be an array",
            param=param,
        )
    return [
        chat_message_tool_call(
            call,
            message_index=message_index,
            tool_index=tool_index,
            context=context,
        )
        for tool_index, call in enumerate(value)
    ]


def chat_message_tool_call(
    value: Any,
    *,
    message_index: int,
    tool_index: int,
    context: str,
) -> dict[str, Any]:
    param = f"messages[{message_index}].tool_calls[{tool_index}]"
    if not isinstance(value, Mapping):
        raise _validation_error(
            f"{context} {param} must be an object",
            param=param,
        )
    allowed = {"id", "type", "function"}
    extra = sorted(str(key) for key in value if str(key) not in allowed)
    if extra:
        raise _validation_error(
            f"{context} {param}.{extra[0]} is not supported",
            param=f"{param}.{extra[0]}",
        )
    call_id = value.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        raise _validation_error(
            f"{context} {param}.id must be a non-empty string",
            param=f"{param}.id",
        )
    call_type = value.get("type")
    if call_type != "function":
        raise _validation_error(
            f"{context} {param}.type must be 'function'",
            param=f"{param}.type",
        )
    function = value.get("function")
    function_param = f"{param}.function"
    if not isinstance(function, Mapping):
        raise _validation_error(
            f"{context} {function_param} must be an object",
            param=function_param,
        )
    function_allowed = {"name", "arguments"}
    function_extra = sorted(str(key) for key in function if str(key) not in function_allowed)
    if function_extra:
        raise _validation_error(
            f"{context} {function_param}.{function_extra[0]} is not supported",
            param=f"{function_param}.{function_extra[0]}",
        )
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise _validation_error(
            f"{context} {function_param}.name must be a non-empty string",
            param=f"{function_param}.name",
        )
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise _validation_error(
            f"{context} {function_param}.arguments must be a string",
            param=f"{function_param}.arguments",
        )
    try:
        json.loads(arguments)
    except Exception as exc:
        raise _validation_error(
            f"{context} {function_param}.arguments must be valid JSON",
            param=f"{function_param}.arguments",
        ) from exc
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def validate_chat_message_tool_transcript(
    messages: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> None:
    pending_tool_call_ids: set[str] = set()
    seen_tool_call_ids: set[str] = set()
    for index, message in enumerate(messages):
        tool_calls = message.get("tool_calls")
        validate_chat_message_tool_transcript_entry(
            str(message.get("role") or ""),
            tool_calls=(
                tool_calls
                if isinstance(tool_calls, Sequence)
                and not isinstance(tool_calls, (str, bytes))
                else []
            ),
            tool_call_id=message.get("tool_call_id"),
            message_index=index,
            context=context,
            pending_tool_call_ids=pending_tool_call_ids,
            seen_tool_call_ids=seen_tool_call_ids,
        )


def validate_chat_message_tool_transcript_entry(
    role: str,
    *,
    tool_calls: Sequence[Mapping[str, Any]],
    tool_call_id: Any,
    message_index: int,
    context: str,
    pending_tool_call_ids: set[str],
    seen_tool_call_ids: set[str],
) -> None:
    if pending_tool_call_ids and role != "tool":
        raise _validation_error(
            (
                f"{context} messages[{message_index}].role must be 'tool' until prior "
                "assistant tool calls are resolved"
            ),
            param=f"messages[{message_index}].role",
        )
    if role == "assistant":
        for tool_index, tool_call in enumerate(tool_calls):
            call_id = str(tool_call.get("id") or "")
            if call_id in seen_tool_call_ids:
                param = f"messages[{message_index}].tool_calls[{tool_index}].id"
                raise _validation_error(
                    f"{context} {param} duplicates a prior assistant tool call id",
                    param=param,
                )
            seen_tool_call_ids.add(call_id)
            pending_tool_call_ids.add(call_id)
        return
    if role != "tool":
        return
    call_id = str(tool_call_id or "").strip()
    if call_id not in pending_tool_call_ids:
        raise _validation_error(
            (
                f"{context} messages[{message_index}].tool_call_id must reference a "
                "prior unconsumed assistant tool call"
            ),
            param=f"messages[{message_index}].tool_call_id",
        )
    pending_tool_call_ids.remove(call_id)
