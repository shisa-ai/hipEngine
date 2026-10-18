"""Gemma 4 chat-template rendering.

The canonical Gemma 4 template reads ``enable_thinking`` as a bare Jinja
variable with a ``false`` fallback. That fallback is not neutral: the false
branch emits an empty thought channel, which tells the model to skip reasoning.
Reaching it by omission is the defect these tests exist to catch, so they assert
on the rendered markers rather than only on the absence of an exception.
"""

from __future__ import annotations

import pytest

from hipengine.chat.gemma4 import render_gemma4_chat

# The reasoning markers the canonical template emits. ``<|think|>`` opens a
# system-level thinking turn; the empty channel pair is what suppresses it.
_THINK_MARKER = "<|think|>"
_EMPTY_THOUGHT_CHANNEL = "<|channel>thought\n<channel|>"

# A template shaped like the canonical one: it must be the *artifact's* template
# that decides, and the canonical one reads the flag as a bare variable.
_TEMPLATE = (
    "{{ bos_token }}"
    "{%- if enable_thinking -%}<|turn>system\n<|think|>\n<turn|>\n{%- endif -%}"
    "{%- for m in messages -%}<|turn>{{ m['role'] }}\n{{ m['content'] }}<turn|>\n{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|turn>model\n"
    "{%- if not enable_thinking -%}<|channel>thought\n<channel|>{%- endif -%}"
    "{%- endif -%}"
)

_MESSAGES = [{"role": "user", "content": "What is 2 + 2?"}]


def test_thinking_is_enabled_by_default() -> None:
    """Omitting the flag must not silently suppress reasoning.

    The template's own default is ``false``, so a renderer that simply forwards
    whatever the caller passed produces a non-reasoning prompt. This is the
    regression that matters: the model still answers, just without thinking.
    """

    rendered = render_gemma4_chat(_TEMPLATE, _MESSAGES)

    assert _THINK_MARKER in rendered, (
        "the default rendering omitted the thinking marker, so reasoning was "
        f"silently disabled: {rendered!r}"
    )
    assert _EMPTY_THOUGHT_CHANNEL not in rendered


def test_thinking_can_be_turned_off_explicitly() -> None:
    """An explicit opt-out must reach the template.

    Passing the flag only to have it ignored would make the parameter useless.
    """

    rendered = render_gemma4_chat(_TEMPLATE, _MESSAGES, enable_thinking=False)

    assert _THINK_MARKER not in rendered
    assert _EMPTY_THOUGHT_CHANNEL in rendered, (
        "enable_thinking=False did not produce the empty thought channel, so the "
        f"flag was not forwarded: {rendered!r}"
    )


def test_a_system_turn_precedes_the_user_turn() -> None:
    """A system message is rendered before the user message, not after.

    The content strings must be distinctive: a substring like "hi" also occurs
    inside the ``<|think|>`` marker, so ``str.index`` would find the marker first
    and the test would compare positions in the wrong strings.
    """

    rendered = render_gemma4_chat(
        _TEMPLATE,
        [
            {"role": "system", "content": "SYSTEM-TURN-CONTENT"},
            {"role": "user", "content": "USER-TURN-CONTENT"},
        ],
    )

    assert rendered.index("SYSTEM-TURN-CONTENT") < rendered.index("USER-TURN-CONTENT")


def test_a_missing_template_is_refused() -> None:
    """An artifact with no template raises rather than formatting something else."""

    with pytest.raises(ValueError, match="chat_template"):
        render_gemma4_chat(None, _MESSAGES)


def test_a_broken_template_reports_the_cause() -> None:
    """A template that cannot render surfaces its own error, not a bare crash."""

    # A syntax error, not an undefined lookup: Jinja's default Undefined renders
    # as the empty string, so a missing variable would not raise at all.
    with pytest.raises(ValueError, match="could not be rendered"):
        render_gemma4_chat("{% for m in %}", _MESSAGES)


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_server_protocol_uses_the_artifact_template(enable_thinking) -> None:
    from types import SimpleNamespace

    from hipengine.generation.gemma4_gguf import Gemma4GGUFGenerator
    from hipengine.server.api import (
        ChatCompletionRequest,
        _render_chat_prompt_with_model_protocol,
        _thinking_control_from_request,
    )

    generator = Gemma4GGUFGenerator("/unused.gguf", object(), object())
    generator._tokenizer = SimpleNamespace(chat_template=_TEMPLATE)
    request = ChatCompletionRequest(
        model="test", messages=_MESSAGES,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )
    rendered = _render_chat_prompt_with_model_protocol(
        request,
        thinking=_thinking_control_from_request(request, chat_default_max_tokens=None),
        engine=generator,
        validate_tool_transcript=True,
    )
    assert rendered == render_gemma4_chat(
        _TEMPLATE, _MESSAGES, enable_thinking=enable_thinking,
    )
    assert "<|im_start|>" not in rendered


@pytest.mark.parametrize("split", range(60))
def test_server_splits_gemma_thought_channels_across_chunks(split) -> None:
    from hipengine.server.api import _ReasoningSplitter, _strip_chat_terminal_markers

    text = "<|channel>thought\nLet me check.<channel|>Paris.<turn|>"
    parser = _ReasoningSplitter()
    parts = parser.feed(text[:split]) + parser.feed(text[split:]) + parser.finish()
    assert "".join(value for field, value in parts if field == "reasoning_content") == "Let me check."
    content = "".join(value for field, value in parts if field == "content")
    assert content == "Paris."
    assert _strip_chat_terminal_markers("Paris.<turn|>") == "Paris."


def test_server_tools_fail_with_the_missing_parser_capability() -> None:
    from hipengine.generation.gemma4_gguf import Gemma4GGUFGenerator

    generator = Gemma4GGUFGenerator("/unused.gguf", object(), object())
    with pytest.raises(NotImplementedError, match="tool-call parsing"):
        generator.render_chat_prompt(_MESSAGES, tools=[{"type": "function"}])

    from hipengine.server.api import (
        ChatCompletionRequest, OpenAIHTTPError,
        _render_chat_prompt_with_model_protocol, _thinking_control_from_request,
    )
    request = ChatCompletionRequest(
        model="test", messages=_MESSAGES,
        tools=[{"type": "function", "function": {
            "name": "lookup", "parameters": {"type": "object", "properties": {}},
        }}],
    )
    with pytest.raises(OpenAIHTTPError) as error:
        _render_chat_prompt_with_model_protocol(
            request, engine=generator, validate_tool_transcript=True,
            thinking=_thinking_control_from_request(request, chat_default_max_tokens=None),
        )
    assert error.value.status_code == 400
    assert "tool-call parsing" in str(error.value.message)


def test_gemma_reasoning_spans_and_length_phase_agree() -> None:
    from hipengine.server.api import _classify_chat_length_phase, _reasoning_text_segments

    text = "<|channel>thought\nCheck.<channel|>Paris."
    segments = _reasoning_text_segments(text)
    assert [(field, text[start:end]) for field, start, end in segments] == [
        ("reasoning_content", "Check."), ("content", "Paris."),
    ]
    assert _classify_chat_length_phase("<|channel>thought\nCheck.") == "reasoning"
    assert _classify_chat_length_phase("<|channel>thought\nCheck.<chan") == "closing_think"
    assert _classify_chat_length_phase(text) == "answer"
