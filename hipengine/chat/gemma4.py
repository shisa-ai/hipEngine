"""Gemma 4's embedded-template renderer.

The canonical Gemma 4 chat template reads several names as bare Jinja
variables and gives each a ``| default(...)`` fallback. That fallback is not
neutral: omitting ``enable_thinking`` renders an *empty* thought channel
(``<|channel>thought\\n<channel|>``), which instructs the model to skip
reasoning entirely rather than leaving the choice open. A caller that never
passes the flag therefore gets a silently non-reasoning model.

This module passes those names explicitly so the rendering reflects a decision
instead of a Jinja default.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from jinja2.sandbox import ImmutableSandboxedEnvironment


def _raise_template_error(message: object) -> None:
    raise ValueError(str(message))


@lru_cache(maxsize=8)
def _compile_template(source: str):
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    environment.globals["raise_exception"] = _raise_template_error
    return environment.from_string(source)


def render_gemma4_chat(
    template: str | None,
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    enable_thinking: bool = True,
    preserve_thinking: bool = False,
    add_generation_prompt: bool = True,
    bos_token: str = "<bos>",
    eos_token: str = "<eos>",
) -> str:
    """Render a Gemma 4 chat template.

    ``enable_thinking`` defaults to True. Gemma 4 is a reasoning model, so
    reasoning is the behaviour to preserve unless a caller asks otherwise; the
    template's own default is the opposite, and reaching it by omission is the
    failure this signature exists to prevent.

    ``preserve_thinking`` only affects how prior tool-call turns are replayed.
    Its default matches the upstream template.
    """

    if not template:
        raise ValueError(
            "the artifact carries no tokenizer.chat_template, so a chat prompt "
            "cannot be rendered; pass preformatted text instead"
        )

    normalized = [dict(message) for message in messages]
    kwargs: dict[str, Any] = {
        "messages": normalized,
        "tools": list(tools) if tools else None,
        "enable_thinking": bool(enable_thinking),
        "preserve_thinking": bool(preserve_thinking),
        "add_generation_prompt": bool(add_generation_prompt),
        "bos_token": bos_token,
        "eos_token": eos_token,
    }
    try:
        return _compile_template(template).render(**kwargs)
    except Exception as exc:  # noqa: BLE001 - re-raised with template context
        raise ValueError(
            f"the artifact's chat template could not be rendered ({exc})"
        ) from exc
