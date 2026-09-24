"""Accounting tests for the Gemma 4 llama.cpp same-artifact reference bench.

The regression these cover: the comparator's value is its exact accounting —
a response that truncated the prompt, returned fewer tokens than requested, or
reported non-positive timings must be rejected rather than averaged into a
reference number the campaign target is computed from. The decode denominator
must match the hipEngine harness convention: outputs - 1 transitions over the
predicted time.
"""

from __future__ import annotations

import pytest

from scripts.gemma4_llamacpp_reference_bench import reference_row, tokenizer_probe


def _response(**overrides):
    payload = {
        "tokens": list(range(128)),
        "tokens_evaluated": 1024,
        "truncated": False,
        "timings": {
            "prompt_n": 1024,
            "prompt_ms": 1600.0,
            "predicted_n": 128,
            "predicted_ms": 1120.0,
        },
    }
    payload.update(overrides)
    return payload


def test_reference_row_uses_transitions_over_predicted_time():
    row = reference_row(_response(), prompt_ids=list(range(1024)), outputs=128)

    assert row["prompt_tokens"] == 1024
    assert row["generated_tokens"] == 128
    assert row["decode_forwards"] == 127
    assert row["prefill_s"] == pytest.approx(1.6)
    assert row["decode_s"] == pytest.approx(1.12)
    assert row["decode_tps"] == pytest.approx(127 / 1.12)
    assert row["prefill_tps"] == pytest.approx(1024 / 1.6)


def test_reference_row_rejects_truncated_prompt_and_short_output():
    with pytest.raises(ValueError, match="truncated"):
        reference_row(_response(truncated=True), prompt_ids=list(range(1024)), outputs=128)
    with pytest.raises(ValueError, match="tokens_evaluated"):
        reference_row(
            _response(tokens_evaluated=1023), prompt_ids=list(range(1024)), outputs=128
        )
    with pytest.raises(ValueError, match="output"):
        reference_row(
            _response(tokens=list(range(100))), prompt_ids=list(range(1024)), outputs=128
        )
    with pytest.raises(ValueError, match="predicted_n"):
        reference_row(
            _response(**{"timings": {
                "prompt_n": 1024, "prompt_ms": 1600.0,
                "predicted_n": 127, "predicted_ms": 1120.0,
            }}),
            prompt_ids=list(range(1024)),
            outputs=128,
        )


def test_reference_row_rejects_nonpositive_timings():
    with pytest.raises(ValueError, match="timing"):
        reference_row(
            _response(**{"timings": {
                "prompt_n": 1024, "prompt_ms": 0.0,
                "predicted_n": 128, "predicted_ms": 1120.0,
            }}),
            prompt_ids=list(range(1024)),
            outputs=128,
        )


def test_tokenizer_probe_accepts_exact_and_bos_prefixed_spellings():
    ids = [5, 6, 7]
    assert tokenizer_probe([5, 6, 7], ids)["match"] is True
    bos = tokenizer_probe([2, 5, 6, 7], ids)
    assert bos["match"] is True and bos["bos"] == 2


def test_tokenizer_probe_reports_the_first_divergence():
    miss = tokenizer_probe([5, 6, 9], [5, 6, 7])
    assert miss["match"] is False
    assert miss["first_divergence"] == 2
    short = tokenizer_probe([5, 6], [5, 6, 7])
    assert short["match"] is False and short["expected"] == 3
    assert short["first_divergence"] == 2