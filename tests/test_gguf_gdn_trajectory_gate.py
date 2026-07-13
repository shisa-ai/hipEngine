from __future__ import annotations

import numpy as np

from scripts.gguf_gdn_trajectory_gate import (
    _aggregate_gate,
    _compare_trajectories,
    _summarize_decode_measurements,
)


def _step(token_id: int, logits: list[float]) -> dict[str, object]:
    return {
        "token_id": token_id,
        "logits": np.asarray(logits, dtype=np.float32),
    }


def test_compare_trajectories_requires_every_token_and_logit_gate() -> None:
    baseline = [_step(4, [0.0, 1.0]), _step(5, [1.0, 0.0])]
    candidate = [_step(4, [0.0, 0.99]), _step(5, [0.99, 0.0])]

    accepted = _compare_trajectories(baseline, candidate)

    assert accepted["passed"] is True
    assert accepted["tokens_exact"] is True
    assert accepted["transitions_total"] == 2
    assert accepted["transitions_passed"] == 2
    assert accepted["top1_agreement_min"] == 1.0

    rejected = _compare_trajectories(
        baseline,
        [_step(4, [0.0, 0.99]), _step(7, [0.99, 0.0])],
    )
    assert rejected["passed"] is False
    assert rejected["tokens_exact"] is False
    assert rejected["first_token_divergence"] == 1


def test_decode_summary_uses_paired_balanced_walls_and_exact_trajectories() -> None:
    measurements = [
        {
            "repetition": 0,
            "order": ["fused", "chain_wave32_tree"],
            "modes": {
                "fused": {"wall_ms": 200.0, "token_ids": [1, 2, 3]},
                "chain_wave32_tree": {
                    "wall_ms": 198.0,
                    "token_ids": [1, 2, 3],
                },
            },
        },
        {
            "repetition": 1,
            "order": ["chain_wave32_tree", "fused"],
            "modes": {
                "chain_wave32_tree": {
                    "wall_ms": 202.0,
                    "token_ids": [1, 2, 3],
                },
                "fused": {"wall_ms": 204.0, "token_ids": [1, 2, 3]},
            },
        },
    ]

    summary = _summarize_decode_measurements(
        measurements,
        decode_steps=128,
        candidate_mode="chain_wave32_tree",
    )

    assert summary["statistics"]["fused"]["median_ms"] == 202.0
    assert summary["statistics"]["chain_wave32_tree"]["median_ms"] == 200.0
    assert summary["paired_candidate_minus_baseline_ms"] == [-2.0, -2.0]
    assert summary["trajectories_exact"] is True
    assert summary["candidate_wins"] is True
    assert summary["candidate_speedup_vs_baseline"] == 1.01


def test_decode_summary_rejects_a_token_divergence_even_when_faster() -> None:
    measurements = [
        {
            "repetition": 0,
            "order": ["fused", "chain_wave32_tree"],
            "modes": {
                "fused": {"wall_ms": 200.0, "token_ids": [1, 2, 3]},
                "chain_wave32_tree": {
                    "wall_ms": 190.0,
                    "token_ids": [1, 8, 3],
                },
            },
        }
    ]

    summary = _summarize_decode_measurements(
        measurements,
        decode_steps=128,
        candidate_mode="chain_wave32_tree",
    )

    assert summary["trajectories_exact"] is False
    assert summary["candidate_wins"] is True


def test_aggregate_gate_has_no_decode_regression_allowance() -> None:
    prompt = {
        "correctness": {
            "passed": True,
            "transitions": [
                {
                    "passed": True,
                    "kl_mean": 1.0e-6,
                    "kl_max": 1.0e-6,
                    "top1_agreement": 1.0,
                }
            ],
        },
        "decode_performance": {
            "trajectories_exact": True,
            "statistics": {
                "fused": {"median_ms": 100.0},
                "chain_wave32_tree": {"median_ms": 100.01},
            },
            "paired_candidate_minus_baseline_ms": [0.01],
        },
    }

    summary = _aggregate_gate(
        [prompt],
        candidate_mode="chain_wave32_tree",
        decode_steps=128,
    )

    assert summary["correctness_passed"] is True
    assert summary["decode_non_regressive"] is False
    assert summary["passed"] is False
