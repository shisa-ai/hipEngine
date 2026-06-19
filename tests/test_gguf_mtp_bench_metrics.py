from __future__ import annotations

import numpy as np
import pytest

from scripts.gguf_mtp_bench import (
    compute_speculative_metrics,
    llama_cpp_acceptance_from_target_samples,
    select_topk_tokens,
    validate_draft_n_max,
)


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
    assert metrics["ar_baseline_tokens_per_sec"] == 1000.0 * 6 / 250.0
    assert metrics["speedup_vs_ar_visible"] == (1000.0 / 47.5) / (1000.0 * 6 / 250.0)
    assert metrics["denominators"] == {
        "accept_per_draft": "accepted_draft_tokens / generated_draft_tokens",
        "accepted_per_output": "accepted_draft_tokens / visible_output_token_count",
        "visible_tokens_per_cycle": "visible_output_token_count / verify_cycle_count",
        "tokens_per_sec": "visible_output_token_count / total_cycle_wall_time",
    }


def test_select_topk_tokens_returns_descending_tokens_and_greedy() -> None:
    logits = np.array([0.1, 4.0, -1.0, 2.5, 3.0], dtype=np.float32)

    greedy, top3 = select_topk_tokens(logits, k=3)

    assert greedy == 1
    assert top3 == [1, 4, 3]


def test_validate_draft_n_max_accepts_b1_and_b2_only() -> None:
    assert validate_draft_n_max(1) == 1
    assert validate_draft_n_max(2) == 2
    with pytest.raises(ValueError, match="B3-B4"):
        validate_draft_n_max(3)


def test_llama_cpp_acceptance_counts_corrective_target_after_reject() -> None:
    summary = llama_cpp_acceptance_from_target_samples([10, 11], [20])

    assert summary["accepted_draft_tokens"] == 0
    assert summary["visible_output_tokens"] == 1
    assert summary["output_tokens"] == [20]
    assert summary["comparison_target_tokens"] == [20]
    assert summary["pending_hidden_row_index"] == 0


def test_llama_cpp_acceptance_counts_partial_prefix_plus_corrective() -> None:
    summary = llama_cpp_acceptance_from_target_samples([10, 11], [10, 20])

    assert summary["accepted_draft_tokens"] == 1
    assert summary["visible_output_tokens"] == 2
    assert summary["output_tokens"] == [10, 20]
    assert summary["comparison_target_tokens"] == [10, 20]
    assert summary["pending_hidden_row_index"] == 1


def test_llama_cpp_acceptance_requires_corrective_after_full_accept() -> None:
    summary = llama_cpp_acceptance_from_target_samples([10, 11], [10, 11, 20])

    assert summary["accepted_draft_tokens"] == 2
    assert summary["visible_output_tokens"] == 3
    assert summary["output_tokens"] == [10, 11, 20]
    assert summary["comparison_target_tokens"] == [10, 11]
    assert summary["pending_hidden_row_index"] == 2

    with pytest.raises(ValueError, match="corrective target"):
        llama_cpp_acceptance_from_target_samples([10], [10])
