"""``usage.prompt_tokens`` has one authoritative producer: the generation output.

A vision request's prompt is text *plus* image tokens, so the server's text
tokenizer cannot recover it; only the generator knows the length it actually
prefilled. ``GenerationOutput.prompt_tokens`` therefore owns the reported count:

* a stated count (including an explicit ``0``) is used unchanged,
* ``None`` means "this producer cannot say", and only then does the server fall
  back to counting the prepared prompt itself,
* the diagnostic mirrors -- ``DecodeState.prompt_tokens`` (which defaults to
  ``0``) and ``telemetry.usage["prompt_tokens"]`` -- never authorize a number,
  because neither distinguishes "zero" from "unknown".

The fallback stays for legacy plugins that return no count at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

from hipengine.generation.registry import (
    DecodeState,
    FinishDetails,
    GenerationOutput,
    GenerationTelemetry,
)
from hipengine.server.api import _coerce_generation_output, _usage


class _WordCountingEngine:
    """A text-only tokenizer: it cannot see image tokens."""

    def count_tokens(self, text: str) -> int:
        return len(str(text).split())


def _output(
    *,
    prompt_tokens: int | None,
    decode_state_prompt_tokens: int = 0,
    telemetry_usage: dict | None = None,
    generated_token_ids: tuple[int, ...] = (7, 8),
    timing: dict | None = None,
) -> GenerationOutput:
    return GenerationOutput(
        text="reply",
        prompt_tokens=prompt_tokens,
        generated_token_ids=generated_token_ids,
        telemetry=GenerationTelemetry(
            decode_state=DecodeState(prompt_tokens=decode_state_prompt_tokens),
            usage=telemetry_usage,
            timing=timing,
        ),
    )


def _usage_for(
    prompts: Sequence[str],
    details: Sequence[GenerationOutput] | None,
    *,
    outputs: Sequence[str] | None = None,
) -> dict:
    texts = tuple(outputs) if outputs is not None else ("reply",) * len(prompts)
    return _usage(
        _WordCountingEngine(),
        tuple(prompts),
        list(texts),
        details=None if details is None else tuple(details),
    )


def test_usage_prefers_the_reported_prompt_count_over_the_text_tokenizer() -> None:
    """A rendered vision prompt is 3 words but 7 model tokens."""

    usage = _usage_for(("describe this now",), (_output(prompt_tokens=7),))

    assert usage["prompt_tokens"] == 7
    assert usage["total_tokens"] == 9


def test_usage_preserves_an_explicit_zero_prompt_count() -> None:
    """Zero is a count, not a missing value: do not substitute the fallback."""

    usage = _usage_for(("one two three",), (_output(prompt_tokens=0),))

    assert usage["prompt_tokens"] == 0
    assert usage["total_tokens"] == 2


def test_usage_falls_back_to_the_prepared_prompt_for_a_legacy_plugin() -> None:
    """A producer that states nothing keeps the server-side count."""

    usage = _usage_for(("one two three four",), (_output(prompt_tokens=None),))

    assert usage["prompt_tokens"] == 4


def test_usage_falls_back_when_no_detail_is_available() -> None:
    usage = _usage_for(("one two", "three"), None)

    assert usage["prompt_tokens"] == 3


def test_usage_mixes_reported_and_fallback_rows_in_one_batch() -> None:
    prompts = ("one two three", "four five six seven")
    details = (_output(prompt_tokens=7), _output(prompt_tokens=None))

    usage = _usage_for(prompts, details)

    assert usage["prompt_tokens"] == 11


def test_usage_ignores_telemetry_usage_without_the_output_field() -> None:
    """The telemetry mirror is diagnostic; it must not become a second owner."""

    detail = _output(prompt_tokens=None, telemetry_usage={"prompt_tokens": 17})

    usage = _usage_for(("one two three",), (detail,))

    assert usage["prompt_tokens"] == 3


def test_usage_ignores_a_decode_state_prompt_count() -> None:
    """``DecodeState.prompt_tokens`` defaults to zero, so it cannot be an owner."""

    detail = _output(prompt_tokens=None, decode_state_prompt_tokens=9)

    usage = _usage_for(("one two three",), (detail,))

    assert usage["prompt_tokens"] == 3


def test_usage_falls_back_for_every_row_when_details_do_not_align() -> None:
    prompts = ("one two three", "four five")
    details = (_output(prompt_tokens=7),)

    usage = _usage_for(prompts, details)

    assert usage["prompt_tokens"] == 5


def test_usage_keeps_reported_counts_beside_speculative_accounting() -> None:
    """Speculative rows report acceptance details; the prompt count is unaffected."""

    detail = _output(
        prompt_tokens=11,
        timing={"mtp_generated_draft_tokens": 4.0, "mtp_accepted_draft_tokens": 3.0},
    )

    usage = _usage_for(("one two three",), (detail,))

    assert usage["prompt_tokens"] == 11
    assert usage["completion_tokens_details"] == {
        "accepted_prediction_tokens": 3,
        "rejected_prediction_tokens": 1,
    }


def test_usage_accepts_a_token_id_prompt_without_a_reported_count() -> None:
    """Token-id prompts are counted by length, not by the text tokenizer."""

    usage = _usage_for(((11, 12, 13, 14, 15),), (_output(prompt_tokens=None),))

    assert usage["prompt_tokens"] == 5


def test_coerced_plugin_output_preserves_the_reported_prompt_count() -> None:
    """A duck-typed plugin output must not lose its exact count in coercion."""

    plugin_output = SimpleNamespace(
        text="reply",
        generated_token_ids=(7,),
        finish_details=FinishDetails(reason="length", length_limit=1),
        prompt_tokens=5,
    )

    coerced = _coerce_generation_output(plugin_output)

    assert coerced.prompt_tokens == 5
    assert _usage(
        _WordCountingEngine(),
        ("one two",),
        ["reply"],
        details=(coerced,),
    )["prompt_tokens"] == 5


def test_coerced_legacy_plugin_output_keeps_an_unknown_count() -> None:
    coerced = _coerce_generation_output(SimpleNamespace(text="reply", prompt_tokens=None))

    assert coerced.prompt_tokens is None
