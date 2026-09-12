"""Shared request and greedy-decode contract for the Surya OCR generators.

Surya OCR is a deterministic transcription model, so both the CPU-reference
and the HIP generators implement greedy decoding only. This module makes that
contract explicit and uniform:

* Supported controls are ``max_tokens``, ``ignore_eos``, ``eos_token_id``,
  ``stop_token_ids``, ``deadline_at``, and ``cancellation_token``.
* Every other sampling/constraint control must be left at its neutral default;
  a non-default value raises :class:`SuryaRequestError` instead of being
  silently ignored.
* Prompt-plus-output capacity is validated before any model work runs.
* The decode loop stops before issuing a decode step for a token that can no
  longer be extended, so the last requested token never triggers a wasted
  forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from hipengine.generation.deadline import raise_if_generation_deadline_expired


class SuryaRequestError(ValueError):
    """Raised when a Surya request uses a control the greedy path cannot honor."""


# (field, neutral default). Any deviation from the default is rejected.
_UNSUPPORTED_CONTROLS: tuple[tuple[str, Any], ...] = (
    ("temperature", 0.0),
    ("top_p", 1.0),
    ("top_k", 0),
    ("min_p", 0.0),
    ("repetition_penalty", 1.0),
    ("presence_penalty", 0.0),
    ("frequency_penalty", 0.0),
    ("logit_bias", ()),
    ("suppress_token_ids", ()),
    ("min_tokens", 0),
    ("stop_token_sequences", ()),
    ("forced_tokens_pending", ()),
    ("post_thinking_forced_tokens_pending", ()),
    ("force_sequence_completion_token_sequences", ()),
    ("thinking_close_token_ids", ()),
    ("thinking_hard_token_cap", None),
    ("thinking_soft_close_window", 0),
    ("json_object_close_forcing", False),
    ("tool_call_constraint", None),
    ("grammar", None),
    ("logprobs", False),
    ("top_logprobs", 0),
    ("kv_storage", "auto"),
    ("kv_scale_dtype", "fp16"),
    ("kv_scale_granularity", "per_token_head"),
)


@dataclass(frozen=True)
class SuryaGreedySettings:
    """Resolved greedy-decoding settings for one Surya request."""

    max_tokens: int
    ignore_eos: bool
    eos_token_ids: frozenset[int]
    stop_token_ids: frozenset[int]


def _is_neutral(value: Any, neutral: Any) -> bool:
    if neutral is None:
        return value is None
    if isinstance(neutral, bool):
        return bool(value) is neutral
    if isinstance(neutral, (int, float)):
        try:
            return float(value) == float(neutral)
        except (TypeError, ValueError):
            return False
    if isinstance(neutral, tuple):
        return tuple(value or ()) == neutral
    return value == neutral


def resolve_surya_greedy_settings(request: Any, spec: Any) -> SuryaGreedySettings:
    """Validate a Surya request and resolve its greedy-decoding settings."""

    for field, neutral in _UNSUPPORTED_CONTROLS:
        value = getattr(request, field, None)
        if _is_neutral(value, neutral):
            continue
        raise SuryaRequestError(
            f"Surya OCR greedy decoding does not support {field!r}={value!r}; "
            f"leave it at the default {neutral!r} or use a sampling-capable model"
        )

    max_tokens = int(getattr(request, "max_tokens", 0))
    if max_tokens < 0:
        raise SuryaRequestError("max_tokens must be non-negative")

    eos_override = getattr(request, "eos_token_id", None)
    eos = eos_override if eos_override is not None else getattr(spec, "eos_token_id", None)
    eos_ids = frozenset() if eos is None else frozenset({int(eos)})
    stop_ids = frozenset(
        int(token) for token in (getattr(request, "stop_token_ids", ()) or ())
    )
    return SuryaGreedySettings(
        max_tokens=max_tokens,
        ignore_eos=bool(getattr(request, "ignore_eos", False)),
        eos_token_ids=eos_ids,
        stop_token_ids=stop_ids,
    )


def check_prompt_capacity(
    prompt_len: int,
    max_tokens: int,
    max_seq: int,
    *,
    hint: str | None = None,
) -> None:
    """Reject a request whose prompt plus output cannot fit in the context.

    ``hint`` names the knob that raises the limit, so the failure says what to
    change instead of only what did not fit.
    """

    if prompt_len <= 0:
        raise SuryaRequestError("prompt produced no tokens")
    if prompt_len + max_tokens > max_seq:
        suffix = "" if hint is None else f"; {hint}"
        raise SuryaRequestError(
            f"prompt ({prompt_len}) + max_tokens ({max_tokens}) exceeds the "
            f"runner context capacity ({max_seq}); reduce the prompt or "
            f"max_tokens{suffix}"
        )


def greedy_decode_tokens(
    logits: np.ndarray,
    settings: SuryaGreedySettings,
    step_fn: Callable[[int, int], np.ndarray],
    request: Any,
) -> tuple[list[int], str]:
    """Greedy decode to a stop token or the token limit.

    ``step_fn(token_id, step_index)`` returns the next-step logits. Returns
    ``(token_ids, finish_reason)`` where the stop/EOS token itself is not
    appended (special tokens are stripped at detokenization, so the text is
    unaffected) and ``finish_reason`` is one of ``"stop"``, ``"eos"``, or
    ``"length"``.
    """

    generated: list[int] = []
    if settings.max_tokens <= 0:
        return generated, "length"

    finish_reason = "length"
    for step in range(settings.max_tokens):
        raise_if_generation_deadline_expired(request)
        next_token = int(np.argmax(logits))
        if next_token in settings.stop_token_ids:
            finish_reason = "stop"
            break
        if not settings.ignore_eos and next_token in settings.eos_token_ids:
            finish_reason = "eos"
            break
        generated.append(next_token)
        if step + 1 >= settings.max_tokens:
            # No further token can be requested: skip the trailing decode.
            break
        logits = step_fn(next_token, step)
    return generated, finish_reason
