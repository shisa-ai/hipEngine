from __future__ import annotations

from scripts.mtp_target_score_compare import build_comparison


def test_target_score_compare_reports_pair_margin_flip() -> None:
    hip = {
        "cycles": [
            {
                "cycle": 3,
                "draft_tokens": [11, 567],
                "target_tokens": [11, 567, 8940],
                "target_lm_head_score_rows": [
                    {
                        "row": 2,
                        "input_token": 567,
                        "target_token": 8940,
                        "top_k": [
                            {"rank": 1, "token": 8940, "score": 25.8, "delta_vs_top1": 0.0},
                            {"rank": 2, "token": 668, "score": 25.3, "delta_vs_top1": -0.5},
                        ],
                        "candidate_scores": [],
                    }
                ],
            }
        ]
    }
    llama_rows = [
        {
            "task_id": 9,
            "cycle": 3,
            "draft_token_ids": [11, 567],
            "sampled_token_ids": [11, 567, 668],
            "target_sample_trace": [
                {
                    "row": 2,
                    "sampled_token": 668,
                    "top_k": [
                        {"rank": 1, "token_id": 668, "logit": 25.55, "margin_from_top": 0.0},
                        {"rank": 2, "token_id": 8940, "logit": 25.54, "margin_from_top": 0.01},
                    ],
                    "candidate_scores": [],
                }
            ],
        }
    ]

    result = build_comparison(
        hip=hip,
        llama_rows=llama_rows,
        cycle=3,
        row=2,
        task_id=9,
        candidate_tokens=[8940, 668],
        pair=(8940, 668),
    )

    assert result["performance_claim"] is False
    assert result["hip"]["sampled_token"] == 8940
    assert result["llama"]["sampled_token"] == 668
    assert result["pair_margin"] == {
        "token_a": 8940,
        "token_b": 668,
        "definition": "token_a_score - token_b_score",
        "hip": 0.5,
        "llama": -0.010000000000001563,
        "delta_hip_minus_llama": 0.5100000000000016,
    }
    by_token = {row["token"]: row for row in result["token_scores"]}
    assert by_token[8940]["hip"]["rank"] == 1
    assert by_token[8940]["llama"]["rank"] == 2
    assert by_token[668]["hip"]["rank"] == 2
    assert by_token[668]["llama"]["rank"] == 1
