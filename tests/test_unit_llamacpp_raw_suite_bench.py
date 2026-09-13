import pytest

from scripts.llamacpp_raw_suite_bench import completion_row, require_idle_memory, speculation_args


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


def test_busy_card_fails_admission():
    require_idle_memory(28_000_256, 128)
    with pytest.raises(RuntimeError, match="not idle"):
        require_idle_memory(2 << 30, 128)


def test_adaptive_floor_is_explicit_without_changing_fork_default():
    assert speculation_args("ar", None) == []
    assert speculation_args("mtp", None) == ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
    default = speculation_args("adaptive", None)
    assert "--spec-draft-n-min-adaptive" not in default
    assert speculation_args("adaptive", 1) == default + ["--spec-draft-n-min-adaptive", "1"]


@pytest.mark.parametrize("mode,floor", [("mtp", 1), ("ar", 1), ("adaptive", 0), ("adaptive", 4)])
def test_invalid_adaptive_floor_rejected(mode, floor):
    with pytest.raises(ValueError):
        speculation_args(mode, floor)
