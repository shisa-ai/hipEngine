from __future__ import annotations

import pytest

from scripts.q8t16_batch_route_perf import (
    KIND,
    revalidate_generated_id_policy,
    summarize_by_configuration,
)


def test_summarize_by_configuration_keeps_matched_pair_ratios() -> None:
    runs = [
        {"configuration": "c2", "label": "strict", "pair": 1, "decode_tok_s": 75.0},
        {"configuration": "c2", "label": "candidate", "pair": 1, "decode_tok_s": 76.5},
        {"configuration": "c2", "label": "candidate", "pair": 2, "decode_tok_s": 73.5},
        {"configuration": "c2", "label": "strict", "pair": 2, "decode_tok_s": 75.0},
        {"configuration": "c4", "label": "strict", "pair": 1, "decode_tok_s": 100.0},
        {"configuration": "c4", "label": "candidate", "pair": 1, "decode_tok_s": 101.0},
        {"configuration": "c4", "label": "candidate", "pair": 2, "decode_tok_s": 98.0},
        {"configuration": "c4", "label": "strict", "pair": 2, "decode_tok_s": 100.0},
        {"configuration": "c8", "label": "strict", "pair": 1, "decode_tok_s": 150.0},
        {"configuration": "c8", "label": "candidate", "pair": 1, "decode_tok_s": 153.0},
        {"configuration": "c8", "label": "candidate", "pair": 2, "decode_tok_s": 147.0},
        {"configuration": "c8", "label": "strict", "pair": 2, "decode_tok_s": 150.0},
    ]

    summary = summarize_by_configuration(runs)

    assert summary["c2"]["candidate_over_strict"] == pytest.approx(1.0)
    assert summary["c2"]["paired_ratios"] == pytest.approx([1.02, 0.98])
    assert summary["c4"]["candidate_over_strict"] == pytest.approx(0.995)
    assert summary["c4"]["paired_ratios"] == pytest.approx([1.01, 0.98])
    assert summary["c8"]["candidate_over_strict"] == pytest.approx(1.0)
    assert summary["c8"]["paired_ratios"] == pytest.approx([1.02, 0.98])


def test_revalidate_generated_id_policy_keeps_mismatch_diagnostic() -> None:
    runs = []
    for pair in range(1, 8):
        runs.extend(
            [
                {
                    "configuration": "c2",
                    "label": "strict",
                    "pair": pair,
                    "decode_tok_s": 75.0,
                    "trajectory_sha256": [f"strict-{pair}"],
                },
                {
                    "configuration": "c2",
                    "label": "candidate",
                    "pair": pair,
                    "decode_tok_s": 76.0,
                    "trajectory_sha256": [f"candidate-{pair}"],
                },
            ]
        )
    source = {
        "kind": KIND,
        "status": "invalid_or_screen_only",
        "measurement_valid": False,
        "performance_claim": False,
        "protocol": {
            "counterbalanced_pairs": 7,
            "prompt_tokens": 512,
            "decode_steps": 128,
            "configurations": ["c2"],
        },
        "provenance": {"dirty": False},
        "memory": {"teardown_exact": True},
        "runs": runs,
    }

    result = revalidate_generated_id_policy(
        source,
        source_sha256="abc",
        command=["python", "revalidate"],
        revalidator_commit="deadbeef",
    )

    assert result["measurement_valid"] is True
    assert result["performance_claim"] is True
    assert result["generated_id_equality"] == {
        "binding": False,
        "reason": (
            "production-profile free-running equality is diagnostic; "
            "strict-teacher full logits are the binding quality comparison"
        ),
        "all_trajectories_exact": False,
    }
    assert result["policy_revalidation"]["source_measurement_valid"] is False
