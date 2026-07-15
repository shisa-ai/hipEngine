from __future__ import annotations

import numpy as np

from scripts.gguf_gdn_trajectory_gate import (
    _aggregate_gate,
    _classify_gate,
    _compare_trajectories,
    _summarize_decode_measurements,
    build_parser,
)


def test_parser_accepts_exact_lds_candidate_modes() -> None:
    for mode in ("chain_lds32", "chain_lds64", "chain_lds32_direct", "chain_wy8"):
        args = build_parser().parse_args(
            ["--candidate-mode", mode, "--json", "/tmp/out.json"]
        )
        assert args.candidate_mode == mode


def test_parser_accepts_explicit_current_default_baseline() -> None:
    args = build_parser().parse_args(
        [
            "--baseline-mode",
            "chain_lds32",
            "--candidate-mode",
            "chain_lds32_direct",
            "--json",
            "/tmp/out.json",
        ]
    )
    assert args.baseline_mode == "chain_lds32"
    assert args.candidate_mode == "chain_lds32_direct"


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


def test_decode_summary_supports_current_default_baseline() -> None:
    measurements = [
        {
            "repetition": 0,
            "order": ["chain_lds32", "chain_lds32_direct"],
            "modes": {
                "chain_lds32": {"wall_ms": 200.0, "token_ids": [1, 2, 3]},
                "chain_lds32_direct": {
                    "wall_ms": 198.0,
                    "token_ids": [1, 2, 3],
                },
            },
        },
        {
            "repetition": 1,
            "order": ["chain_lds32_direct", "chain_lds32"],
            "modes": {
                "chain_lds32_direct": {
                    "wall_ms": 202.0,
                    "token_ids": [1, 2, 3],
                },
                "chain_lds32": {"wall_ms": 204.0, "token_ids": [1, 2, 3]},
            },
        },
    ]
    summary = _summarize_decode_measurements(
        measurements,
        decode_steps=128,
        baseline_mode="chain_lds32",
        candidate_mode="chain_lds32_direct",
    )
    assert summary["baseline_mode"] == "chain_lds32"
    assert summary["statistics"]["chain_lds32"]["median_ms"] == 202.0
    assert summary["statistics"]["chain_lds32_direct"]["median_ms"] == 200.0
    assert summary["paired_candidate_minus_baseline_ms"] == [-2.0, -2.0]
    assert summary["trajectories_exact"] is True


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


def test_aggregate_gate_supports_current_default_baseline() -> None:
    prompt = {
        "correctness": {
            "passed": True,
            "transitions": [
                {
                    "passed": True,
                    "kl_mean": 0.0,
                    "kl_max": 0.0,
                    "top1_agreement": 1.0,
                }
            ],
        },
        "decode_performance": {
            "trajectories_exact": True,
            "statistics": {
                "chain_lds32": {"median_ms": 100.0},
                "chain_lds32_direct": {"median_ms": 99.0},
            },
            "paired_candidate_minus_baseline_ms": [-1.0],
        },
    }
    summary = _aggregate_gate(
        [prompt],
        baseline_mode="chain_lds32",
        candidate_mode="chain_lds32_direct",
        decode_steps=128,
    )
    assert summary["baseline_mode"] == "chain_lds32"
    assert summary["decode_non_regressive"] is True
    assert summary["passed"] is True


def test_clean_trajectory_divergence_is_a_correctness_rejection() -> None:
    classification = _classify_gate(
        {
            "passed": False,
            "correctness_passed": False,
            "trajectory_tokens_exact": False,
        },
        provenance={"dirty": False},
        candidate_mode="chain_wave32_tree",
    )

    assert classification["status"] == "rejected_correctness"
    assert classification["measurement_valid"] is True
    assert classification["performance_comparison_valid"] is False
    assert classification["gate_passed"] is False
