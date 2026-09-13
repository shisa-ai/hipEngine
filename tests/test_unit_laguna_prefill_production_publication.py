from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.laguna_prefill_production_publication import (
    CAPABILITY_DEFAULTS,
    EXPECTED_LENGTHS,
    _expected_candidate_modes,
    summarize_publication,
)


def _defaults() -> dict:
    return {name: expected for name, (_, expected) in CAPABILITY_DEFAULTS.items()}


def _quality() -> dict:
    defaults = _defaults()
    quality_modes = _expected_candidate_modes(defaults)
    categories = {
        name: {
            "finite": True,
            "steps": 80,
            "max_kl_divergence": 0.04,
            "top1_agreement": 0.95,
            "top1_matches": 76,
        }
        for name in ("code", "general_en", "general_ja", "mixed_ja_en")
    }
    return {
        "kind": "hipengine_laguna_prefill_prefill_350_category",
        "status": "retained_category_gate",
        "pass": True,
        "performance_claim": True,
        "repo": {"tracked_clean": True},
        "promotion": {"pass": True, "failed_checks": []},
        "protocol": {
            "prefill_lane_configurations": {
                "prefill_350_candidate": {
                    **quality_modes,
                }
            }
        },
        "quality": {
            "teacher_forced": {
                "pass": True,
                "steps": 320,
                "max_kl_divergence": 0.04,
                "top1_agreement": 0.95,
                "top1_matches": 304,
                "categories": categories,
            },
            "poolside_oracle": {"pass": True},
            "free_running": {"same_mode_repeat_deterministic": True},
        },
        "memory": {
            "tracked_before": {"active_allocations": 0, "current_allocated_bytes": 0},
            "tracked_after": {"active_allocations": 0, "current_allocated_bytes": 0},
        },
        "provenance": {"model_fingerprint": {"value": "model"}},
    }


def _matrix() -> dict:
    rows = []
    aggregates = {}
    for length in EXPECTED_LENGTHS:
        samples = []
        for repetition in range(3):
            tok_s = 355.0 if length == 512 else 300.0
            seconds = length / tok_s
            samples.append(seconds)
            rows.append(
                {
                    "matrix_rows": 512,
                    "length": length,
                    "repetition": repetition,
                    "prefill_seconds": seconds,
                    "prefill_tok_s": tok_s,
                    "next_token_id": length,
                    "next_token_logit_hex": "0x1.0p+0",
                    "logits_sha256": f"logits-{length}",
                    "final_hidden_sha256": f"hidden-{length}",
                    "post_layer_hidden_sha256": f"post-{length}",
                    "kv_sha256": f"kv-{length}",
                    "final_position": length - 1,
                    "session_tracked_returned_to_baseline": True,
                }
            )
        aggregates[str(length)] = {
            "median_seconds": samples[1],
            "median_tok_s": 355.0 if length == 512 else 300.0,
        }
    return {
        "kind": "hipengine_laguna_ar_o3_matrix_chunk_screen",
        "status": "measured_rejected",
        "pass": False,
        "performance_claim": False,
        "repo": {"tracked_clean": True},
        "platform": {"backend": "hip_gfx1151", "target_arch": "gfx1151"},
        "protocol": {
            "lengths": list(EXPECTED_LENGTHS),
            "matrix_rows": [128, 256, 512],
            "attention_rows": 128,
            "repetitions": 3,
        },
        "decision": {
            "failed_checks": [
                "matrix_policy_outputs_or_state_not_exact",
                "no_larger_policy_improves_every_length",
            ]
        },
        "correctness": {
            "same_mode_repeat_deterministic": True,
            "tracked_returned_to_baseline": True,
        },
        "memory": {
            "tracked_before": {"active_allocations": 0, "current_allocated_bytes": 0},
            "tracked_after": {"active_allocations": 0, "current_allocated_bytes": 0},
        },
        "rows": rows,
        "aggregate": {"512": {"lengths": aggregates}},
        "provenance": {"model_fingerprint": {"value": "model"}},
    }


def test_laguna_production_publication_accepts_bound_quality_and_speed() -> None:
    artifact = summarize_publication(_quality(), _matrix(), defaults=_defaults())

    assert artifact["pass"] is True
    assert artifact["status"] == "retained_production_default"
    assert artifact["headline"]["median_tok_s"] == pytest.approx(355.0)
    assert artifact["quality"]["teacher_forced_steps"] == 320
    assert artifact["decision"]["failed_checks"] == []


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("slow", "pp512_median_below_target"),
        ("mode", "quality_candidate_modes_do_not_match_defaults"),
        ("nondeterministic", "matrix_same_mode_not_deterministic"),
        ("quality", "teacher_forced_quality_failed"),
        ("historical_failure", "unexpected_historical_matrix_gate_outcome"),
    ],
)
def test_laguna_production_publication_fails_closed(
    mutation: str, failed_check: str
) -> None:
    quality = _quality()
    matrix = _matrix()
    defaults = _defaults()
    if mutation == "slow":
        matrix["aggregate"]["512"]["lengths"]["512"]["median_tok_s"] = 349.0
    elif mutation == "mode":
        defaults["selected_gate_up_mode"] = "wrong"
    elif mutation == "nondeterministic":
        matrix["correctness"]["same_mode_repeat_deterministic"] = False
    elif mutation == "quality":
        quality["quality"]["teacher_forced"]["pass"] = False
    elif mutation == "historical_failure":
        matrix["decision"]["failed_checks"].append("unexpected")

    artifact = summarize_publication(quality, matrix, defaults=defaults)

    assert artifact["pass"] is False
    assert artifact["performance_claim"] is False
    assert failed_check in artifact["decision"]["failed_checks"]
