from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.laguna_expert_major_component_bench import (
    CANDIDATE_MODES,
    LAYER_CANDIDATE_MODES,
    LAYER_MODES,
    MODES,
    _bisection_modes,
    _evaluate,
    _load_source_rejection,
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


def test_layer_family_bisection_uses_only_architecture_scopes() -> None:
    assert _bisection_modes("components") == (MODES, CANDIDATE_MODES)
    assert _bisection_modes("layer_families") == (
        LAYER_MODES,
        LAYER_CANDIDATE_MODES,
    )
    assert LAYER_CANDIDATE_MODES == (
        "adaptive_expert_major_wmma_comp_global",
        "adaptive_expert_major_wmma_comp_swa",
        "adaptive_expert_major_wmma_comp",
    )
    orders = [
        _mode_order(1, repetition, modes=LAYER_MODES) for repetition in range(4)
    ]
    assert all(set(order) == set(LAYER_MODES) for order in orders)
    assert len({order[0] for order in orders}) == len(LAYER_MODES)


def test_layer_family_bisection_requires_published_component_rejection(
    tmp_path,
) -> None:
    path = tmp_path / "component-rejection.json"
    artifact = {
        "kind": "hipengine_laguna_expert_major_component_bisection",
        "status": "component_bisection_rejected",
        "pass": False,
        "model": {"sha256": "model-sha"},
        "repo": {"revision": "component-revision"},
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")
    args = SimpleNamespace(
        source_rejection=path,
        model_sha256="model-sha",
        bisection="layer_families",
    )

    result = _load_source_rejection(args)

    assert result["pass"] is True
    assert result["source_kind"] == artifact["kind"]
    assert result["revision"] == "component-revision"

    artifact["kind"] = "hipengine_laguna_prefill_expert_major_wmma_category"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="source expert-major rejection"):
        _load_source_rejection(args)


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
