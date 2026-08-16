from __future__ import annotations

from pathlib import Path

import pytest

from scripts import execution_profile_gguf_c1_route_gate as quality_gate
from scripts import execution_profile_gguf_c1_route_perf as perf


def _quality(tmp_path: Path, *, status: str = "passed"):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    return model, {
        "kind": quality_gate.KIND,
        "status": status,
        "measurement_valid": True,
        "candidate": {"name": "router_f32w_coop_persistent"},
        "protocol": {
            "complete_prompt_and_heldout_suite": True,
            "model": str(model.resolve()),
        },
    }


def test_validate_quality_artifact_requires_complete_passing_candidate(tmp_path: Path) -> None:
    model, artifact = _quality(tmp_path)
    perf.validate_quality_artifact(
        artifact,
        candidate_name="router_f32w_coop_persistent",
        model=model,
    )

    artifact["protocol"]["complete_prompt_and_heldout_suite"] = False
    with pytest.raises(perf.PerformanceGateError, match="complete prompt suite"):
        perf.validate_quality_artifact(
            artifact,
            candidate_name="router_f32w_coop_persistent",
            model=model,
        )


def test_summarize_samples_reports_median_gain_and_repeatability() -> None:
    summary = perf.summarize_samples(
        {"strict": [10.0, 11.0, 12.0], "candidate": [12.0, 13.0, 14.0]},
        {
            "strict": [[1, 2], [1, 2], [1, 2]],
            "candidate": [[1, 2], [1, 2], [1, 2]],
        },
    )

    assert summary["strict"]["median_tok_s"] == 11.0
    assert summary["candidate"]["median_tok_s"] == 13.0
    assert summary["candidate_vs_strict_pct"] == pytest.approx(18.18181818)
    assert summary["candidate_faster"] is True
    assert summary["candidate_generated_ids_repeatable"] is True
    assert summary["strict_candidate_generated_ids_equal"] is True


def test_summarize_natural_samples_reports_aggregate_and_paired_wins() -> None:
    rows = [
        {
            "routes": {
                "strict": {
                    "tok_s": 10.0,
                    "elapsed_seconds": 1.0,
                    "generated_token_ids": [1],
                },
                "candidate": {
                    "tok_s": 12.5,
                    "elapsed_seconds": 0.8,
                    "generated_token_ids": [1],
                },
            }
        },
        {
            "routes": {
                "strict": {
                    "tok_s": 8.0,
                    "elapsed_seconds": 1.25,
                    "generated_token_ids": [2],
                },
                "candidate": {
                    "tok_s": 10.0,
                    "elapsed_seconds": 1.0,
                    "generated_token_ids": [3],
                },
            }
        },
    ]

    summary = perf.summarize_natural_samples(rows, decode_steps=10)

    assert summary["strict"]["aggregate_tok_s"] == pytest.approx(20 / 2.25)
    assert summary["candidate"]["aggregate_tok_s"] == pytest.approx(20 / 1.8)
    assert summary["candidate_vs_strict_pct"] == pytest.approx(25.0)
    assert summary["paired_wins"] == 2
    assert summary["strict_candidate_generated_ids_equal_prompts"] == 1


def test_summarize_samples_flags_candidate_id_drift() -> None:
    summary = perf.summarize_samples(
        {"strict": [10.0, 10.0, 10.0], "candidate": [11.0, 11.0, 11.0]},
        {
            "strict": [[1], [1], [1]],
            "candidate": [[1], [2], [1]],
        },
    )

    assert summary["candidate_generated_ids_repeatable"] is False
