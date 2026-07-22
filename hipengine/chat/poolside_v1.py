"""Poolside Laguna ``poolside_v1`` chat and reasoning contracts.

The reasoning boundary follows vLLM's Poolside-specific parser at
``vllm-project/vllm@61c9ef986a807aa3b9c6ccd25bb223b8f4116ac7``: prompt
history is scanned backwards only as far as the current ``<assistant>`` token.
The renderer is a torch-free transcription of Laguna S 2.1's frozen GGUF Jinja
template; model execution never imports Transformers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

POOLSIDE_V1_CONTRACT = "poolside_v1"
POOLSIDE_V1_DEFAULT_SYSTEM = (
    "You are a helpful, conversationally-fluent assistant made by Poolside. "
    "You are here to be helpful to users through natural language conversations."
)

_REASONING_START = "<think>"
_REASONING_END = "</think>"
_ASSISTANT_START = "<assistant>"
_MISSING = object()


class PoolsideV1ReasoningParser:
    """Determine whether a Poolside assistant generation starts in reasoning."""

    def __init__(self, tokenizer: Any) -> None:
        token_to_id = getattr(tokenizer, "token_to_id", None)
        if not isinstance(token_to_id, Mapping):
            raise ValueError("Poolside tokenizer must expose a token_to_id mapping")
        self._tokenizer = tokenizer
        self.start_token_id = _required_token_id(token_to_id, _REASONING_START)
        self.end_token_id = _required_token_id(token_to_id, _REASONING_END)
        self.assistant_token_id = _required_token_id(token_to_id, _ASSISTANT_START)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Mirror vLLM while ignoring markers before the current assistant turn."""

        for token_id in reversed(input_ids):
            current = int(token_id)
            if current == self.start_token_id:
                return False
            if current == self.end_token_id:
                return True
            if current == self.assistant_token_id:
                return False
        return False

    def initially_open_ids(self, input_ids: Sequence[int]) -> bool:
        """Return whether generated text begins inside an open thinking span."""

        for token_id in reversed(input_ids):
            current = int(token_id)
            if current == self.start_token_id:
                return True
            if current in {self.end_token_id, self.assistant_token_id}:
                return False
        return False

    def initially_open(self, prompt: str) -> bool:
        encoder = getattr(self._tokenizer, "encode", None)
        if not callable(encoder):
            raise ValueError("Poolside tokenizer must expose encode(text)")
        return self.initially_open_ids(tuple(int(token) for token in encoder(str(prompt))))


def render_poolside_v1_chat(
    messages: Sequence[Any],
    *,
    tools: Sequence[Any] | None = None,
    enable_thinking: bool = False,
    add_generation_prompt: bool = True,
) -> str:
    """Render the frozen Laguna S 2.1 GGUF chat template exactly."""

    remaining = list(messages)
    system_message = POOLSIDE_V1_DEFAULT_SYSTEM
    if remaining and _field(remaining[0], "role", "") == "system":
        system_message = _string_content(_field(remaining[0], "content", ""))
        remaining = remaining[1:]

    rendered = ["〈|EOS|〉"]
    has_system = bool(system_message and system_message.strip())
    if has_system or tools or enable_thinking:
        system_parts = ["<system>"]
        if has_system:
            system_parts.append(system_message.rstrip())
            if tools:
                system_parts.append("\n\n")
        if tools:
            system_parts.extend(
                [
                    "### Tools\n\n",
                    "You may call functions to assist with the user query.\n",
                    "All available function signatures are listed below:\n",
                    "<available_tools>\n",
                ]
            )
            for tool in tools:
                system_parts.append(_json_dumps(_plain_value(tool)) + "\n")
            system_parts.append("</available_tools>")
        system_parts.append("</system>\n")
        rendered.append("".join(system_parts))

    for message in remaining:
        role = str(_field(message, "role", ""))
        content = _string_content(_field(message, "content", ""))
        if role == "user":
            rendered.append(f"<user>{content}</user>\n")
        elif role == "assistant":
            rendered.append(
                _render_assistant_message(
                    message,
                    content=content,
                    enable_thinking=bool(enable_thinking),
                )
            )
        elif role == "tool":
            rendered.append(f"<tool_response>{content}</tool_response>\n")
        elif role == "system":
            rendered.append(f"<system>{content}</system>\n")

    if add_generation_prompt:
        marker = _REASONING_START if enable_thinking else _REASONING_END
        rendered.append(f"{_ASSISTANT_START}{marker}")
    return "".join(rendered)


def _render_assistant_message(
    message: Any,
    *,
    content: str,
    enable_thinking: bool,
) -> str:
    reasoning = _field(message, "reasoning", None)
    if not isinstance(reasoning, str):
        reasoning = _field(message, "reasoning_content", "")
    if not isinstance(reasoning, str):
        reasoning = ""

    rendered = [_ASSISTANT_START]
    if enable_thinking:
        rendered.extend((_REASONING_START, reasoning, _REASONING_END))
    else:
        rendered.append(_REASONING_END)
    if content:
        rendered.append(content)

    tool_calls = _field(message, "tool_calls", None)
    if tool_calls:
        for tool_call in tool_calls:
            function = _field(tool_call, "function", {})
            name = str(_field(function, "name", ""))
            arguments = _field(function, "arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Poolside assistant tool-call arguments must be a JSON object"
                    ) from exc
            if not isinstance(arguments, Mapping):
                raise ValueError("Poolside assistant tool-call arguments must be an object")
            rendered.extend(("<tool_call>", name))
            for key, value in arguments.items():
                rendered.extend(("<arg_key>", str(key), "</arg_key><arg_value>"))
                rendered.append(value if isinstance(value, str) else _json_dumps(value, ensure_ascii=False))
                rendered.append("</arg_value>")
            rendered.append("</tool_call>")
    rendered.append("</assistant>\n")
    return "".join(rendered)


def _field(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    direct = getattr(value, name, _MISSING)
    if direct is not _MISSING:
        return direct
    extra = getattr(value, "model_extra", None)
    if isinstance(extra, Mapping):
        return extra.get(name, default)
    return default


def _string_content(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_value(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _plain_value(model_dump(exclude_none=True))
    return value


def _json_dumps(value: Any, *, ensure_ascii: bool = True) -> str:
    return json.dumps(value, ensure_ascii=ensure_ascii).replace("<", "\\u003c").replace(
        ">", "\\u003e"
    ).replace("&", "\\u0026").replace("'", "\\u0027")


def _required_token_id(token_to_id: Mapping[str, Any], token: str) -> int:
    value = token_to_id.get(token)
    if value is None:
        raise ValueError(f"Poolside tokenizer must contain the atomic token {token!r}")
    return int(value)


__all__ = [
    "POOLSIDE_V1_CONTRACT",
    "PoolsideV1ReasoningParser",
    "render_poolside_v1_chat",
]
