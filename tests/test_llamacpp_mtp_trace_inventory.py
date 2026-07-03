from __future__ import annotations

from scripts.llamacpp_mtp_trace_inventory import (
    infer_trace_checkpoint_coverage,
    next_action,
    summarize_alignment,
    summarize_trace,
)


def test_inventory_detects_token_only_trace_lacks_numeric_checkpoints() -> None:
    trace = {
        "schema": 1,
        "kind": "llamacpp_mtp_draft_candidate_trace",
        "prompt_tokens": 17,
        "summary": {"draft_n": 2, "draft_n_accepted": 1},
        "metadata": {"request": {"draft_n_max": 2}},
        "calls": [
            {
                "call_count": 1,
                "generated": 2,
                "accepted": 1,
                "accept_generated": 2,
                "hist_size": 17,
                "candidates": [
                    {"pos": 0, "rank": 0, "token_id": 271, "piece": "!", "prob": 0.9},
                    {"pos": 1, "rank": 0, "token_id": 2500, "piece": " How", "prob": 0.8},
                ],
            }
        ],
    }
    checkpoint = {"capture": {"position": 16, "layer_id": 3, "layer_type": "full_attention"}}

    summary = summarize_trace(trace)
    coverage = infer_trace_checkpoint_coverage(trace)
    alignment = summarize_alignment(summary, checkpoint)

    assert summary["prompt_tokens"] == 17
    assert summary["calls"][0]["top_by_pos"]["0"]["token_id"] == 271
    assert coverage["supports_token_draft_alignment"] is True
    assert coverage["has_numeric_layer_checkpoints"] is False
    assert alignment["prompt_tokens_match"] is True
    assert (
        next_action(coverage, alignment)
        == "capture_llamacpp_numeric_layer_checkpoint_or_add_llama_trace_tap"
    )


def test_inventory_detects_numeric_checkpoint_arrays() -> None:
    trace = {
        "prompt_tokens": 17,
        "calls": [],
        "numeric_checkpoint": {
            "hidden_in_f32": [float(i) for i in range(16)],
            "attn_out_f32": [float(i) * 0.5 for i in range(16)],
        },
    }

    coverage = infer_trace_checkpoint_coverage(trace)

    assert coverage["has_numeric_layer_checkpoints"] is True
    assert "$.numeric_checkpoint.hidden_in_f32" in coverage["known_checkpoint_array_paths"]
    assert coverage["numeric_array_paths_len_ge_16"][0]["count"] == 16


def test_alignment_reports_prompt_length_mismatch() -> None:
    trace_summary = {"prompt_tokens": 18}
    checkpoint = {"capture": {"position": 16}}

    alignment = summarize_alignment(trace_summary, checkpoint)

    assert alignment["prompt_tokens_match"] is False
