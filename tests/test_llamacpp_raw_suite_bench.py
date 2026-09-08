import pytest

from scripts.llamacpp_raw_suite_bench import completion_row


def response():
    return dict(tokens=[4, 5, 6], tokens_evaluated=2, truncated=False,
                timings=dict(predicted_n=3, predicted_ms=20, prompt_ms=4))


def test_transition_denominator_and_prompt_accounting():
    row = completion_row(response(), [1, 2], 3)
    assert row["decode_transitions"] == 2
    assert row["decode_tok_s"] == 100
    assert row["prefill_tok_s"] == 500
    assert row["output_ids"] == [4, 5, 6]


@pytest.mark.parametrize("key,value", [
    ("tokens", [4, 5]), ("tokens_evaluated", 1), ("truncated", True),
    ("timings", dict(predicted_n=3, predicted_ms=0, prompt_ms=4)),
])
def test_rejects_incomplete_or_invalid_response(key, value):
    payload = response()
    payload[key] = value
    with pytest.raises(ValueError):
        completion_row(payload, [1, 2], 3)
