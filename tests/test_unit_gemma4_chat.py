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
