from __future__ import annotations

import pytest

from scripts.laguna_poolside_ar_bench import _aggregate, _response_row


def test_poolside_response_row_uses_native_timing_ownership() -> None:
    row = _response_row(
        prompt={
            "id": "p",
            "category": "code",
            "prompt_tokens": 10,
            "token_ids_sha256": "prompt-hash",
        },
        horizon=4,
        repetition=1,
        response={
            "tokens": [1, 2, 3, 4],
            "tokens_predicted": 4,
            "stop_type": "limit",
            "timings": {
                "prompt_n": 10,
                "prompt_ms": 500.0,
                "predicted_n": 4,
                "predicted_ms": 200.0,
            },
        },
        wall_seconds=0.8,
        hipengine_ids=[1, 2, 9, 4],
    )

    assert row["valid_token_count"] is True
    assert row["valid_prompt_count"] is True
    assert row["prompt_tok_s"] == pytest.approx(20.0)
    assert row["predicted_tok_s"] == pytest.approx(20.0)
    assert row["wall_output_tok_s"] == pytest.approx(5.0)
    assert row["matches_hipengine"] is False
    assert row["matching_hipengine_prefix_tokens"] == 2


def test_poolside_aggregate_is_weighted_and_fail_closed() -> None:
    rows = [
        {
            "prompt_n": 10,
            "prompt_seconds": 1.0,
            "predicted_n": 4,
            "predicted_seconds": 0.2,
            "wall_seconds": 1.3,
            "valid_token_count": True,
            "valid_prompt_count": True,
            "matches_hipengine": True,
        },
        {
            "prompt_n": 20,
            "prompt_seconds": 1.0,
            "predicted_n": 4,
            "predicted_seconds": 0.3,
            "wall_seconds": 1.5,
            "valid_token_count": True,
            "valid_prompt_count": True,
            "matches_hipengine": False,
        },
    ]

    aggregate = _aggregate(rows)

    assert aggregate["prompt_tok_s"] == pytest.approx(15.0)
    assert aggregate["predicted_tok_s"] == pytest.approx(16.0)
    assert aggregate["wall_output_tok_s"] == pytest.approx(8 / 2.8)
    assert aggregate["valid_token_counts"] is True
    assert aggregate["valid_prompt_counts"] is True
    assert aggregate["hipengine_exact_runs"] == 1

    rows[1]["valid_token_count"] = False
    assert _aggregate(rows)["valid_token_counts"] is False
