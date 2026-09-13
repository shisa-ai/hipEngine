from __future__ import annotations

import pytest

from scripts.laguna_poolside_prefill_control import (
    _completion_payload,
    _response_row,
    _summarize_rows,
    _timing_order,
)


def test_poolside_prefill_payload_is_exact_token_prompt_only() -> None:
    payload = _completion_payload((1, 2, 3))

    assert payload["prompt"] == [1, 2, 3]
    assert payload["n_predict"] == 0
    assert payload["cache_prompt"] is False
    assert payload["stream"] is False
    assert payload["return_tokens"] is True


def test_poolside_prefill_response_uses_native_prompt_timing() -> None:
    row = _response_row(
        length=128,
        repetition=2,
        response={
            "tokens": [81],
            "tokens_predicted": 1,
            "timings": {
                "prompt_n": 128,
                "prompt_ms": 2000.0,
                "predicted_n": 1,
                "predicted_ms": 0.001,
            },
        },
        wall_seconds=2.25,
    )

    assert row["valid_prompt_count"] is True
    assert row["no_post_prompt_decode"] is True
    assert row["prompt_seconds"] == pytest.approx(2.0)
    assert row["prompt_tok_s"] == pytest.approx(64.0)
    assert row["wall_seconds"] == pytest.approx(2.25)


def test_poolside_prefill_summary_is_balanced_and_fail_closed() -> None:
    rows = [
        {
            "length": 128,
            "repetition": 0,
            "prompt_n": 128,
            "prompt_seconds": 2.0,
            "prompt_tok_s": 64.0,
            "wall_seconds": 2.2,
            "valid_prompt_count": True,
            "no_post_prompt_decode": True,
        },
        {
            "length": 128,
            "repetition": 1,
            "prompt_n": 128,
            "prompt_seconds": 1.6,
            "prompt_tok_s": 80.0,
            "wall_seconds": 1.8,
            "valid_prompt_count": True,
            "no_post_prompt_decode": True,
        },
        {
            "length": 128,
            "repetition": 2,
            "prompt_n": 128,
            "prompt_seconds": 1.8,
            "prompt_tok_s": 128 / 1.8,
            "wall_seconds": 2.0,
            "valid_prompt_count": True,
            "no_post_prompt_decode": True,
        },
    ]

    summary = _summarize_rows(rows, lengths=(128,), repetitions=3)

    assert summary["128"]["median_prompt_seconds"] == pytest.approx(1.8)
    assert summary["128"]["median_prompt_tok_s"] == pytest.approx(128 / 1.8)
    assert summary["128"]["all_prompt_counts_exact"] is True
    assert summary["128"]["all_prompt_only"] is True

    rows[2]["valid_prompt_count"] = False
    with pytest.raises(ValueError, match="prompt count"):
        _summarize_rows(rows, lengths=(128,), repetitions=3)


def test_poolside_prefill_timing_order_alternates() -> None:
    lengths = (128, 512, 1024, 4096)

    assert _timing_order(lengths, 0) == lengths
    assert _timing_order(lengths, 1) == tuple(reversed(lengths))
    assert _timing_order(lengths, 2) == lengths
