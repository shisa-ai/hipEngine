import pytest

from scripts.qwen4exp_conservative_cost import summarize_cost


def samples():
    return [
        dict(case_id="code-p512", category="code", prompt_tokens=512,
             mode=mode, output_token_ids_sha256=mode, prefill_ms=1000,
             decode_ms=1000, decode_transitions=128, client_wall_s=2,
             prefill_tok_s=512, decode_tok_s=128)
        for mode in ("before", "after") for _ in range(3)
    ]


def test_cost_allows_cross_arm_drift_but_reports_it():
    result = summarize_cost(samples(), 3)
    assert result["correctness"]["within_mode_deterministic"]
    assert not result["correctness"]["cross_mode_output_exact"]


def test_cost_still_rejects_within_arm_nondeterminism():
    rows = samples()
    rows[0]["output_token_ids_sha256"] = "different"
    with pytest.raises(ValueError, match="within-mode"):
        summarize_cost(rows, 3)
