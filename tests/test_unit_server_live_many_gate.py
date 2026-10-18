"""Which ``n>1`` chat requests may take the live-many (speculative) path.

The readiness checklist requires that multiple choices, tool/structured
responses, and unsupported sampling either work through existing contracts or
select a named supported fallback. ``_chat_live_many_streaming_allowed``
(``hipengine/server/api.py:12563``) is the gate that decides whether an
``n>1`` chat request is expanded onto the live-many path -- the one that can
run the speculative route -- or resolved through a single-choice route.

It had no direct coverage. Every branch below is a route decision, not a
refusal: returning False sends the request down a route the server supports
and reports in the response's route decision, so a False here is the "named
supported fallback" arm of the acceptance rather than a silent downgrade. The
explicit-refusal arm (a named capability error) lives in
``_speculative_mtp_route_for_request`` and is covered separately.

Values are exercised on both sides of the ``n`` threshold and at an unrelated
width, per the project rule against validating only the configured boundary.
"""

from __future__ import annotations

import pytest

from hipengine.server.api import (
    ChatCompletionRequest,
    _chat_live_many_streaming_allowed,
)

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]

_JSON_SCHEMA_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
    },
}


def _request(**values) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test", messages=[{"role": "user", "content": "hi"}], **values
    )


def test_live_many_gate_requires_more_than_one_choice() -> None:
    """Both sides of the width threshold, plus an unrelated width."""

    assert _chat_live_many_streaming_allowed(_request(n=1)) is False
    assert _chat_live_many_streaming_allowed(_request(n=2)) is True
    assert _chat_live_many_streaming_allowed(_request(n=7)) is True


@pytest.mark.parametrize(
    "exclusion",
    [
        pytest.param({"tools": _TOOLS}, id="tools"),
        pytest.param({"tools": _TOOLS, "tool_choice": "auto"}, id="tools-and-choice"),
        pytest.param({"logprobs": True}, id="logprobs"),
        pytest.param({"logprobs": True, "top_logprobs": 5}, id="top_logprobs"),
        pytest.param({"response_format": {"type": "json_object"}}, id="json_object"),
        pytest.param({"response_format": _JSON_SCHEMA_FORMAT}, id="json_schema"),
        pytest.param({"guided_json": {"type": "object"}}, id="guided_json"),
        pytest.param({"guided_regex": "^a+$"}, id="guided_regex"),
        pytest.param({"guided_choice": ["a", "b"]}, id="guided_choice"),
        pytest.param({"stop": "END"}, id="stop-string"),
        pytest.param({"stop": ["END", "STOP"]}, id="stop-list"),
        pytest.param({"continuation_id": "abc"}, id="continuation"),
    ],
)
def test_live_many_gate_excludes_each_structured_or_stateful_axis(
    exclusion: dict,
) -> None:
    """Every exclusion axis resolves the request off the live-many path.

    A tool call, a structured/guided response, per-token logprob metadata, a
    stop string, or a continuation all need per-row machinery the multi-choice
    speculative expansion does not carry, so the request keeps a route that
    can serve it.
    """

    assert _chat_live_many_streaming_allowed(_request(n=2, **exclusion)) is False


@pytest.mark.parametrize(
    "harmless",
    [
        pytest.param({"tools": []}, id="empty-tools"),
        pytest.param({"tools": None}, id="null-tools"),
        pytest.param({"logprobs": False}, id="logprobs-false"),
        pytest.param({"stop": []}, id="empty-stop-list"),
        pytest.param({"stop": None}, id="null-stop"),
        pytest.param({"continuation_id": None}, id="null-continuation"),
        # ``tool_choice`` constrains which of ``tools`` may be called, so on
        # its own it carries no constraint and must not exclude the request.
        pytest.param({"tool_choice": "auto"}, id="tool-choice-without-tools"),
        pytest.param({"tool_choice": "none"}, id="tool-choice-none-without-tools"),
    ],
)
def test_live_many_gate_keeps_the_path_for_falsy_optional_fields(
    harmless: dict,
) -> None:
    """Present-but-empty optional fields must not exclude a request.

    These are the cases where a client sends a field explicitly set to its
    default; treating them as exclusions would drop multi-choice speculative
    serving for requests that carry no constraint at all.
    """

    assert _chat_live_many_streaming_allowed(_request(n=2, **harmless)) is True


def test_live_many_gate_exclusions_apply_at_every_width() -> None:
    """An excluded request is excluded at the unrelated width too.

    Guards against an implementation that special-cases a particular width
    rather than testing the constraint itself.
    """

    assert _chat_live_many_streaming_allowed(_request(n=7, tools=_TOOLS)) is False
    assert (
        _chat_live_many_streaming_allowed(
            _request(n=7, response_format={"type": "json_object"})
        )
        is False
    )
    assert (
        _chat_live_many_streaming_allowed(_request(n=7, logprobs=True)) is False
    )
