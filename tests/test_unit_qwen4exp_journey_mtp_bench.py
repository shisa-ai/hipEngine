import pytest

from scripts.qwen4exp_journey_mtp_bench import (
    summarize, validate_output, wait_server, validate_local_mtp,
)


def test_output_requires_exact_token_count():
    with pytest.raises(ValueError, match="token count"):
        validate_output([1], 2)
    with pytest.raises(ValueError, match="integer"):
        validate_output([True, 2], 2)
    assert validate_output([1, 2], 2) == [1, 2]


def test_summary_uses_total_wall_and_checks_every_repeat():
    rows = [
        {"id": "a", "category": "code", "split": "heldout", "mode": mode,
         "repetition": rep, "seconds": seconds, "tokens": tokens}
        for rep in range(2)
        for mode, seconds, tokens in (("ar", 2, [1, 2]), ("mtp", 1, [1, 2]))
    ]
    result = summarize(rows, expected_ids=["a"], repetitions=2)
    assert result["full"]["mtp_over_ar"] == 2
    assert result["full"]["mtp_request_tok_s"] == 2
    assert result["exact_vs_own_ar"]
    rows[-1]["tokens"] = [1, 3]
    result = summarize(rows, expected_ids=["a"], repetitions=2)
    assert not result["exact_vs_own_ar"]
    assert not result["deterministic"]


def test_summary_rejects_missing_and_duplicate_cells():
    row = {"id": "a", "category": "code", "split": "train", "mode": "ar",
           "repetition": 0, "seconds": 1, "tokens": [1]}
    with pytest.raises(ValueError, match="matrix"):
        summarize([row], expected_ids=["a"], repetitions=1)
    with pytest.raises(ValueError, match="duplicate"):
        summarize([row, row], expected_ids=["a"], repetitions=1)


def test_dead_server_fails_before_health_wait():
    class DeadServer:
        def poll(self):
            return -6

    with pytest.raises(RuntimeError, match="exited.*-6"):
        wait_server(DeadServer(), "127.0.0.1", 1, 600)


def test_mtp_requires_real_proposals():
    with pytest.raises(ValueError, match="proposals"):
        validate_local_mtp({})
    with pytest.raises(ValueError, match="proposals"):
        validate_local_mtp({"proposed_draft_tokens": 0})
    validate_local_mtp({"proposed_draft_tokens": 2, "accepted_draft_tokens": 0})
