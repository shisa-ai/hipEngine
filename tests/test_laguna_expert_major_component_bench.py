from __future__ import annotations

from scripts.laguna_expert_major_component_bench import (
    CANDIDATE_MODES,
    MODES,
    _evaluate,
    _mode_order,
)


def test_component_bisection_mode_order_is_deterministic_permutation() -> None:
    assert set(MODES) == {
        "adaptive_grouped_smallm_fused",
        "adaptive_expert_major_gate_up_comp",
        "adaptive_expert_major_down_comp",
        "adaptive_expert_major_wmma_comp",
    }
    assert set(CANDIDATE_MODES) == set(MODES[1:])
    orders = [_mode_order(prompt_index=2, repetition=rep) for rep in range(4)]
    assert all(set(order) == set(MODES) for order in orders)
    assert len({order[0] for order in orders}) == len(MODES)
    assert orders == [_mode_order(prompt_index=2, repetition=rep) for rep in range(4)]


def test_component_bisection_selects_only_quality_safe_faster_mode() -> None:
    performance = {
        "modes": {
            MODES[0]: {"prefill_tok_s": 70.0},
            CANDIDATE_MODES[0]: {"prefill_tok_s": 110.0},
            CANDIDATE_MODES[1]: {"prefill_tok_s": 90.0},
            CANDIDATE_MODES[2]: {"prefill_tok_s": 130.0},
        },
        "speedups_vs_retained": {
            CANDIDATE_MODES[0]: 110.0 / 70.0,
            CANDIDATE_MODES[1]: 90.0 / 70.0,
            CANDIDATE_MODES[2]: 130.0 / 70.0,
        },
    }
    quality = {
        CANDIDATE_MODES[0]: {
            "pass": True,
            "max_kl_divergence": 0.04,
            "top1_agreement": 0.98,
        },
        CANDIDATE_MODES[1]: {
            "pass": False,
            "max_kl_divergence": 0.08,
            "top1_agreement": 0.99,
        },
        CANDIDATE_MODES[2]: {
            "pass": False,
            "max_kl_divergence": 0.5,
            "top1_agreement": 0.98,
        },
    }

    result = _evaluate(performance, quality)

    assert result["pass"] is True
    assert result["passing_modes"] == [CANDIDATE_MODES[0]]
    assert result["selected_mode"] == CANDIDATE_MODES[0]
    assert result["modes"][CANDIDATE_MODES[0]]["pass"] is True
    assert result["modes"][CANDIDATE_MODES[1]]["failed_checks"] == [
        "teacher_forced_quality_failed"
    ]


def test_component_bisection_rejects_quality_pass_that_is_not_faster() -> None:
    candidate = CANDIDATE_MODES[0]
    performance = {
        "modes": {
            MODES[0]: {"prefill_tok_s": 70.0},
            candidate: {"prefill_tok_s": 69.0},
        },
        "speedups_vs_retained": {candidate: 69.0 / 70.0},
    }
    quality = {
        candidate: {
            "pass": True,
            "max_kl_divergence": 0.01,
            "top1_agreement": 1.0,
        }
    }

    result = _evaluate(performance, quality)

    assert result["pass"] is False
    assert result["selected_mode"] is None
    assert result["modes"][candidate]["failed_checks"] == [
        "prefill_not_faster"
    ]
