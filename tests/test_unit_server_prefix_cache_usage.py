"""``usage.prompt_tokens_details.cached_tokens`` reports measured prefix reuse.

The field is OpenAI/vLLM-compatible: it counts prompt tokens the engine served
from its prefix cache, as a subset of ``prompt_tokens``. It is read from the
backend's per-request ``prefix_cache`` diagnostics, so a request whose outputs
carried no such telemetry must report no field at all -- reporting zero there
would claim a cache that was never consulted.
"""

from __future__ import annotations

from collections.abc import Sequence

from hipengine.generation.registry import GenerationOutput, GenerationTelemetry
from hipengine.server.api import _usage


class _WordCountingEngine:
    def count_tokens(self, text: str) -> int:
        return len(str(text).split())


def _output(*, diagnostics: dict | None, prompt_tokens: int = 4) -> GenerationOutput:
    return GenerationOutput(
        text="reply",
        telemetry=GenerationTelemetry.from_decode_counts(
            prompt_tokens=prompt_tokens,
            generated_tokens=2,
            diagnostics=diagnostics,
        ),
    )


def _prefix(reused_tokens: int, *, prompt_tokens: int = 4, reason: str | None = None) -> dict:
    return {
        "prefix_cache": {
            "mode": "radix",
            "block_size_tokens": 256,
            "eligible": True,
            "lookup": True,
            "hit": bool(reused_tokens),
            "reused_tokens": reused_tokens,
            "executed_prefill_tokens": max(0, prompt_tokens - reused_tokens),
            "fallback_reason": reason,
        }
    }


def _usage_for(
    prompts: Sequence[str],
    details: Sequence[GenerationOutput] | None,
) -> dict:
    return _usage(
        _WordCountingEngine(),
        tuple(prompts),
        ["reply"] * len(prompts),
        details=details,
    )


def test_usage_reports_measured_prefix_reuse() -> None:
    usage = _usage_for(("one two three four",), (_output(diagnostics=_prefix(3)),))

    assert usage["prompt_tokens"] == 4
    assert usage["prompt_tokens_details"] == {"cached_tokens": 3}


def test_usage_reports_a_consulted_cache_that_missed_as_zero() -> None:
    """A miss is a measurement: the cache looked and reused nothing."""

    usage = _usage_for(("one two three four",), (_output(diagnostics=_prefix(0, reason="miss")),))

    assert usage["prompt_tokens_details"] == {"cached_tokens": 0}


def test_usage_omits_cached_tokens_without_prefix_telemetry() -> None:
    without_diagnostics = _usage_for(("one two three four",), (_output(diagnostics=None),))
    other_diagnostics = _usage_for(
        ("one two three four",),
        (_output(diagnostics={"specdec2_mtp2": {"cycles": 3}}),),
    )
    without_details = _usage_for(("one two three four",), None)

    assert "prompt_tokens_details" not in without_diagnostics
    assert "prompt_tokens_details" not in other_diagnostics
    assert "prompt_tokens_details" not in without_details


def test_usage_clamps_reuse_to_the_prompt_it_came_from() -> None:
    usage = _usage_for(("one two three four",), (_output(diagnostics=_prefix(9)),))

    assert usage["prompt_tokens_details"] == {"cached_tokens": 4}


def test_usage_sums_reuse_across_the_outputs_of_one_request() -> None:
    prompts = ("one two three four", "five six seven eight")
    details = (
        _output(diagnostics=_prefix(2), prompt_tokens=4),
        _output(diagnostics=_prefix(3), prompt_tokens=4),
    )

    usage = _usage_for(prompts, details)

    assert usage["prompt_tokens"] == 8
    assert usage["prompt_tokens_details"] == {"cached_tokens": 5}


def test_usage_reports_cache_reuse_beside_mtp_acceptance() -> None:
    """The new field must not disturb the MTP acceptance fields beside it."""

    detail = GenerationOutput(
        text="reply",
        telemetry=GenerationTelemetry.from_decode_counts(
            prompt_tokens=4,
            generated_tokens=2,
            timing={"mtp_generated_draft_tokens": 4.0, "mtp_accepted_draft_tokens": 3.0},
            diagnostics=_prefix(2),
        ),
    )

    usage = _usage_for(("one two three four",), (detail,))

    assert usage["prompt_tokens_details"] == {"cached_tokens": 2}
    assert usage["completion_tokens_details"] == {
        "accepted_prediction_tokens": 3,
        "rejected_prediction_tokens": 1,
    }
