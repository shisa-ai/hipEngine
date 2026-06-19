from __future__ import annotations

from scripts.gguf_mtp_bench import compute_speculative_metrics


def test_compute_speculative_metrics_counts_visible_accepted_tokens() -> None:
    """Accepted draft tokens are visible outputs, not just diagnostic accepts."""
    cycles = [
        {"accepted": False, "ar_decode_ms": 50.0, "mtp_draft_ms": 7.0},
        {"accepted": True, "ar_decode_ms": 50.0, "mtp_draft_ms": 7.0},
        {"accepted": False, "ar_decode_ms": 50.0, "mtp_draft_ms": 7.0},
        {"accepted": False, "ar_decode_ms": 50.0, "mtp_draft_ms": 7.0},
        {"accepted": False, "ar_decode_ms": 50.0, "mtp_draft_ms": 7.0},
    ]

    metrics = compute_speculative_metrics(cycles)

    assert metrics["total_drafts"] == 5
    assert metrics["total_accepted"] == 1
    assert metrics["verify_cycle_count"] == 5
    assert metrics["total_output_tokens"] == 6
    assert metrics["accept_per_draft"] == 0.2
    assert metrics["accepted_per_output"] == 1 / 6
    assert metrics["visible_tokens_per_cycle"] == 1.2
    assert metrics["avg_cycle_ms"] == 57.0
    assert metrics["avg_ms_per_visible_token"] == 47.5
    assert metrics["tokens_per_sec"] == 1000.0 / 47.5
    assert metrics["ar_baseline_tokens_per_sec"] == 20.0
    assert metrics["speedup_vs_ar_visible"] == (1000.0 / 47.5) / 20.0
    assert metrics["denominators"] == {
        "accept_per_draft": "accepted_draft_tokens / generated_draft_tokens",
        "accepted_per_output": "accepted_draft_tokens / visible_output_token_count",
        "visible_tokens_per_cycle": "visible_output_token_count / verify_cycle_count",
        "tokens_per_sec": "visible_output_token_count / total_cycle_wall_time",
    }
