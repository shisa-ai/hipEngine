"""Pure chat reasoning and tool-output parsing.

This module owns the reasoning-tag splitter, terminal-marker stripping, the
tool-envelope parsers (legacy JSON and the Qwen XML function envelope), the
marker/residue constants they use, and the parser result dataclasses.

This module takes text, declared tool schemas and, for reasoning splitting, an
``initially_open`` flag. It has no application or request state and uses only
the standard library.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import re
import uuid
from typing import Any


_TOOL_CALL_START_MARKER = "<tool_call>"
_TOOL_CALL_END_MARKER = "</tool_call>"
_CHAT_TEMPLATE_TERMINAL_MARKERS = ("<|im_end|>", "<turn|>")
_CHAT_TEMPLATE_RESIDUE_MARKERS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>")
_CHAT_TEMPLATE_RESIDUE_ROLES = ("assistant", "developer", "system", "tool", "user")


@dataclass(frozen=True)
class _ReasoningSplit:
    content: str
    reasoning_content: str


@dataclass(frozen=True)
class _ParsedToolCall:
    id: str
    name: str
    arguments: str
    raw_text: str = ""


@dataclass(frozen=True)
class _ParsedChatOutput:
    text: str
    tool_calls: tuple[_ParsedToolCall, ...]
    reasoning_initially_open: bool = False
    tool_parser_name: str | None = None
    invalid_tool_call_blocks: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ReasoningPart:
    field: str
    text: str
    source_start: int
    source_end: int


_REASONING_OPEN_TAGS = ("<think>", "<|channel>thought\n")
_REASONING_CLOSE_TAGS = ("</think>", "<channel|>")


class _ReasoningSplitter:
    """Incrementally split thinking tags, including prompt-opened reasoning."""

    def __init__(self, *, initially_open: bool = False) -> None:
        self._buffer = ""
        self._buffer_start = 0
        self._input_length = 0
        self._in_reasoning = bool(initially_open)

    @property
    def pending_text_length(self) -> int:
        return len(self._buffer)

    @property
    def pending_source_start(self) -> int:
        return self._buffer_start

    def feed(self, text: str) -> list[tuple[str, str]]:
        return [(part.field, part.text) for part in self.feed_parts(text)]

    def finish(self) -> list[tuple[str, str]]:
        return [(part.field, part.text) for part in self.finish_parts()]

    def feed_parts(self, text: str) -> list[_ReasoningPart]:
        if not text:
            return []
        if not self._buffer:
            self._buffer_start = self._input_length
        self._buffer += text
        self._input_length += len(text)
        return self._drain(final=False)

    def finish_parts(self) -> list[_ReasoningPart]:
        parts = self._drain(final=True)
        if parts and parts[-1].field == "content":
            stripped = _strip_chat_terminal_markers(parts[-1].text)
            if stripped != parts[-1].text:
                parts[-1] = _ReasoningPart(
                    field="content",
                    text=stripped,
                    source_start=parts[-1].source_start,
                    source_end=parts[-1].source_start + len(stripped),
                )
        return parts

    def _drain(self, *, final: bool) -> list[_ReasoningPart]:
        outputs: list[_ReasoningPart] = []
        while self._buffer:
            markers = tuple(
                (self._buffer.find(tag), tag, next_state)
                for tags, next_state in (
                    (_REASONING_OPEN_TAGS, True), (_REASONING_CLOSE_TAGS, False)
                )
                for tag in tags
            )
            found = min((item for item in markers if item[0] >= 0), default=None)
            if found is not None:
                index, tag, next_state = found
                self._append(outputs, self._buffer[:index], self._buffer_start)
                self._buffer = self._buffer[index + len(tag) :]
                self._buffer_start += index + len(tag)
                self._in_reasoning = next_state
                continue
            if final:
                self._append(outputs, self._buffer, self._buffer_start)
                self._buffer_start += len(self._buffer)
                self._buffer = ""
                break
            keep = max(
                _tag_suffix_len(self._buffer, tag)
                for tag in (*_REASONING_OPEN_TAGS, *_REASONING_CLOSE_TAGS)
            )
            # A terminal marker can straddle stream chunks. Hold both partial
            # and complete trailing markers until finish_parts can strip them;
            # once emitted, a client cannot retract them.
            keep = max(keep, *(
                len(tag) if self._buffer.endswith(tag)
                else _tag_suffix_len(self._buffer, tag)
                for tag in _CHAT_TEMPLATE_TERMINAL_MARKERS
            ))
            emit_len = len(self._buffer) - keep
            if emit_len > 0:
                self._append(outputs, self._buffer[:emit_len], self._buffer_start)
                self._buffer = self._buffer[emit_len:]
                self._buffer_start += emit_len
            break
        return outputs

    def _append(self, outputs: list[_ReasoningPart], text: str, source_start: int) -> None:
        if text:
            field = "reasoning_content" if self._in_reasoning else "content"
            outputs.append(
                _ReasoningPart(
                    field=field,
                    text=text,
                    source_start=int(source_start),
                    source_end=int(source_start) + len(text),
                )
            )


def _tag_suffix_len(text: str, tag: str) -> int:
    max_len = min(len(tag) - 1, len(text))
    for length in range(max_len, 0, -1):
        if tag.startswith(text[-length:]):
            return length
    return 0


def _classify_chat_length_phase(text: str) -> str:
    if any(_tag_suffix_len(text, tag) for tag in _REASONING_OPEN_TAGS):
        return "reasoning"
    in_reasoning = max(text.rfind(tag) for tag in _REASONING_OPEN_TAGS) > max(
        text.rfind(tag) for tag in _REASONING_CLOSE_TAGS
    )
    if in_reasoning:
        if any(_tag_suffix_len(text, tag) for tag in _REASONING_CLOSE_TAGS):
            return "closing_think"
        return "reasoning"
    if _has_unclosed_tool_call(text):
        return "tool_call"
    if _looks_like_partial_json(text):
        return "structured"
    return "answer"


def _has_unclosed_tool_call(text: str) -> bool:
    lowered = text.lower()
    if _tag_suffix_len(lowered, "<tool_call>") or _tag_suffix_len(lowered, "</tool_call>"):
        return True
    open_index = lowered.rfind("<tool_call>")
    close_index = lowered.rfind("</tool_call>")
    return open_index > close_index


def _looks_like_partial_json(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return False
    try:
        json.loads(stripped)
    except Exception:
        return True
    return False


def _split_reasoning(text: str, *, initially_open: bool = False) -> _ReasoningSplit:
    splitter = _ReasoningSplitter(initially_open=initially_open)
    parts = splitter.feed(text) + splitter.finish()
    content = "".join(part for field, part in parts if field == "content")
    reasoning = "".join(part for field, part in parts if field == "reasoning_content")
    return _ReasoningSplit(content=content, reasoning_content=reasoning)


@dataclass(frozen=True)
class _ToolCallBlock:
    start: int
    end: int
    parsed: _ParsedToolCall


def _find_marker(lowered: str, marker: str, start: int, end: int | None = None) -> int:
    if end is None:
        return lowered.find(marker, start)
    return lowered.find(marker, start, end)


def _find_valid_tool_call_block(
    text: str,
    lowered: str,
    start_at: int,
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> _ToolCallBlock | None:
    start = _find_marker(lowered, _TOOL_CALL_START_MARKER, start_at)
    while start >= 0:
        body_start = start + len(_TOOL_CALL_START_MARKER)
        end = _find_marker(lowered, _TOOL_CALL_END_MARKER, body_start)
        while end >= 0:
            raw_text = text[start : end + len(_TOOL_CALL_END_MARKER)]
            parsed = _parsed_tool_call_from_block_body(
                text[body_start:end],
                raw_text=raw_text,
                tools=tools,
            )
            if parsed is not None:
                return _ToolCallBlock(
                    start=start,
                    end=end + len(_TOOL_CALL_END_MARKER),
                    parsed=parsed,
                )
            end = _find_marker(lowered, _TOOL_CALL_END_MARKER, end + len(_TOOL_CALL_END_MARKER))
        start = _find_marker(lowered, _TOOL_CALL_START_MARKER, body_start)
    return None


def _valid_tool_call_blocks(
    text: str,
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[_ToolCallBlock, ...]:
    lowered = text.lower()
    blocks: list[_ToolCallBlock] = []
    cursor = 0
    while cursor < len(text):
        block = _find_valid_tool_call_block(text, lowered, cursor, tools=tools)
        if block is None:
            break
        blocks.append(block)
        cursor = block.end
    return tuple(blocks)


def _is_chat_template_control_residue(text: str) -> bool:
    """Return whether text contains only leaked Qwen template controls/roles."""

    residue = str(text)
    cursor = 0
    saw_marker = False
    role_allowed = False
    while cursor < len(residue):
        if residue[cursor].isspace():
            cursor += 1
            continue
        marker = next(
            (item for item in _CHAT_TEMPLATE_RESIDUE_MARKERS if residue.startswith(item, cursor)),
            None,
        )
        if marker is not None:
            cursor += len(marker)
            saw_marker = True
            role_allowed = role_allowed or marker == "<|im_start|>"
            continue
        if role_allowed:
            role = next(
                (item for item in _CHAT_TEMPLATE_RESIDUE_ROLES if residue.startswith(item, cursor)),
                None,
            )
            if role is not None:
                role_end = cursor + len(role)
                if role_end == len(residue) or residue[role_end] in "\r\n":
                    cursor = role_end
                    role_allowed = False
                    continue
        return False
    return saw_marker


def _strip_chat_template_terminal_edges(text: str) -> str:
    """Remove chat-template residue only around a parsed tool envelope."""

    stripped = str(text).strip()
    while stripped:
        previous = stripped
        for marker in _CHAT_TEMPLATE_TERMINAL_MARKERS:
            if stripped.startswith(marker):
                stripped = stripped[len(marker) :].lstrip()
            if stripped.endswith(marker):
                stripped = stripped[: -len(marker)].rstrip()
        if stripped == previous:
            break
    if _is_chat_template_control_residue(stripped):
        return ""
    return stripped


def _is_incomplete_duplicate_tool_prefix_control_residue(text: str, *, tool_name: str) -> bool:
    """Return whether text is an unfinished duplicate prefix plus Qwen controls."""

    residue = str(text).strip()
    prefix = _tool_call_name_prefix_text(tool_name)
    if not residue.startswith(prefix):
        return False
    residue = residue[len(prefix) :].lstrip()
    if residue.startswith("{"):
        residue = residue[1:].lstrip()
    return _is_chat_template_control_residue(residue)


def _parse_chat_tool_calls(
    text: str,
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> _ParsedChatOutput:
    calls: list[_ParsedToolCall] = []
    text_parts: list[str] = []
    last_end = 0
    for block in _valid_tool_call_blocks(text, tools=tools):
        text_parts.append(text[last_end : block.start])
        calls.append(block.parsed)
        last_end = block.end
    text_parts.append(text[last_end:])
    if not calls:
        stripped = text.strip()
        parsed = _parsed_tool_call_from_json(stripped, raw_text=stripped)
        if parsed is not None:
            return _ParsedChatOutput(text="", tool_calls=(parsed,))
    visible_text = "".join(text_parts).strip()
    if calls:
        if _is_incomplete_duplicate_tool_prefix_control_residue(
            visible_text,
            tool_name=calls[0].name,
        ):
            visible_text = ""
        else:
            visible_text = _strip_chat_template_terminal_edges(visible_text)
    return _ParsedChatOutput(text=visible_text, tool_calls=tuple(calls))


def _parsed_tool_call_from_block_body(
    raw: str,
    *,
    raw_text: str = "",
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> _ParsedToolCall | None:
    stripped = raw.strip()
    parsed = _parsed_tool_call_from_json(stripped, raw_text=raw_text)
    if parsed is not None:
        return parsed
    repaired = _strip_duplicate_tool_call_start(stripped)
    if repaired != stripped:
        parsed = _parsed_tool_call_from_json(repaired, raw_text=raw_text)
        if parsed is not None:
            return parsed
    return _parsed_tool_call_from_function_xml(repaired, raw_text=raw_text, tools=tools)


_XML_FUNCTION_ENVELOPE_RE = re.compile(
    r"<function=(?P<name>[^<>\n]+?)>\s*\n(?P<body>.*?)</function>\s*\Z",
    re.DOTALL,
)
_XML_PARAMETER_RE = re.compile(
    r"<parameter=(?P<key>[^<>\n]+?)>\n?(?P<value>.*?)\n?</parameter>",
    re.DOTALL,
)


def _parsed_tool_call_from_function_xml(
    raw: str,
    *,
    raw_text: str = "",
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> _ParsedToolCall | None:
    """Lift the Qwen3.5-family XML function envelope into a tool call.

    Qwen3.5-and-later checkpoints (including the Qwen3.8-Flash-Next lineage)
    are trained on ``<tool_call>\n<function=name>\n<parameter=key>\nvalue\n"
    "</parameter>\n</function>\n</tool_call>`` envelopes rather than the
    legacy JSON body. String-typed parameters stay verbatim (multi-line
    safe); other parameters decode as JSON and fall back to the raw string,
    matching llama.cpp's Qwen3-Coder-family parser semantics.
    """

    envelope = _XML_FUNCTION_ENVELOPE_RE.fullmatch(raw.strip())
    if envelope is None:
        return None
    name = envelope.group("name").strip()
    if not name:
        return None
    body = envelope.group("body")
    arguments: dict[str, Any] = {}
    cursor = 0
    for parameter in _XML_PARAMETER_RE.finditer(body):
        if body[cursor : parameter.start()].strip():
            return None
        key = parameter.group("key").strip()
        if not key:
            return None
        raw_value = parameter.group("value")
        arguments[key] = _xml_parameter_value(
            raw_value,
            string_typed=_xml_parameter_is_string_typed(tools, tool_name=name, key=key),
        )
        cursor = parameter.end()
    if body[cursor:].strip():
        return None
    return _ParsedToolCall(
        id=f"call_{uuid.uuid4().hex[:24]}",
        name=name,
        arguments=_tool_arguments_json(arguments),
        raw_text=str(raw_text),
    )


def _xml_parameter_value(raw_value: str, *, string_typed: bool) -> Any:
    if string_typed:
        return raw_value
    try:
        return json.loads(raw_value)
    except Exception:
        return raw_value


def _xml_parameter_is_string_typed(
    tools: Sequence[Mapping[str, Any]] | None,
    *,
    tool_name: str,
    key: str,
) -> bool:
    schema = _tool_parameter_property_schema(tools, tool_name=tool_name, key=key)
    if schema is None:
        return False
    return _json_schema_resolves_to_string(schema)


def _tool_parameter_property_schema(
    tools: Sequence[Mapping[str, Any]] | None,
    *,
    tool_name: str,
    key: str,
) -> Mapping[str, Any] | None:
    for tool in tools or ():
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        source = function if isinstance(function, Mapping) else tool
        if str(source.get("name", "")) != tool_name:
            continue
        parameters = source.get("parameters")
        if not isinstance(parameters, Mapping):
            return None
        properties = parameters.get("properties")
        if not isinstance(properties, Mapping):
            return None
        schema = properties.get(key)
        return schema if isinstance(schema, Mapping) else None
    return None


def _json_schema_resolves_to_string(schema: Mapping[str, Any]) -> bool:
    """Mirror llama.cpp's resolves_to_string for XML argument typing."""

    schema_type = schema.get("type")
    if isinstance(schema_type, str) and schema_type == "string":
        return True
    if isinstance(schema_type, Sequence) and not isinstance(schema_type, (str, bytes)):
        if any(str(item) == "string" for item in schema_type):
            return True
    enum = schema.get("enum")
    if isinstance(enum, Sequence) and not isinstance(enum, (str, bytes)) and enum:
        return all(isinstance(item, str) for item in enum)
    for combinator in ("oneOf", "anyOf"):
        alternatives = schema.get(combinator)
        if isinstance(alternatives, Sequence) and not isinstance(alternatives, (str, bytes)):
            if any(
                isinstance(alternative, Mapping) and _json_schema_resolves_to_string(alternative)
                for alternative in alternatives
            ):
                return True
    components = schema.get("allOf")
    if isinstance(components, Sequence) and not isinstance(components, (str, bytes)) and components:
        return all(
            isinstance(component, Mapping) and _json_schema_resolves_to_string(component)
            for component in components
        )
    return False


def _strip_duplicate_tool_call_start(text: str) -> str:
    stripped = text.lstrip()
    marker_len = len(_TOOL_CALL_START_MARKER)
    if not stripped[:marker_len].lower() == _TOOL_CALL_START_MARKER:
        return text
    return stripped[marker_len:].lstrip()


def _parsed_tool_call_from_json(raw: str, *, raw_text: str = "") -> _ParsedToolCall | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    return _parsed_tool_call_from_mapping(payload, raw_text=raw_text)


def _parsed_tool_call_from_mapping(payload: Mapping[str, Any], *, raw_text: str = "") -> _ParsedToolCall | None:
    function = payload.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = payload.get("name")
        arguments = payload.get("arguments", {})
    if not isinstance(name, str) or not name:
        return None
    return _ParsedToolCall(
        id=f"call_{uuid.uuid4().hex[:24]}",
        name=name,
        arguments=_tool_arguments_json(arguments),
        raw_text=str(raw_text),
    )


def _tool_arguments_json(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except Exception:
            return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    if arguments is None:
        arguments = {}
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


def _strip_chat_terminal_markers(text: str) -> str:
    """Remove terminal markers, not whitespace generated before them.

    Whitespace following an actual terminal marker is template residue. Without
    that marker it is model output and must match streaming text and logprobs.
    """

    stripped = str(text)
    while True:
        candidate = stripped.rstrip()
        marker = next((marker for marker in _CHAT_TEMPLATE_TERMINAL_MARKERS
                       if marker and candidate.endswith(marker)), None)
        if marker is None:
            return stripped
        stripped = candidate[:-len(marker)]


def _tool_call_name_prefix_text(name: str) -> str:
    return (
        f"{_TOOL_CALL_START_MARKER}"
        f'{{"name":{json.dumps(str(name), ensure_ascii=False, separators=(",", ":"))},"arguments":'
    )
