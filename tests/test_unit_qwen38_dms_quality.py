from __future__ import annotations

import json

import pytest

from scripts.qwen38_dms_quality import load_scenario_thresholds, summarize_quality


def test_quality_summary_reports_mean_tail_max_top1_and_outer_floor() -> None:
    rows = [
        {"kl": 0.001, "top1_match": True},
        {"kl": 0.010, "top1_match": True},
        {"kl": 0.020, "top1_match": False},
    ]

    summary = summarize_quality(rows)

    assert summary["rows"] == 3
    assert summary["kl_mean"] == pytest.approx(0.031 / 3)
    assert summary["kl_max"] == pytest.approx(0.020)
    assert summary["top1_agreement"] == pytest.approx(2 / 3)
    assert summary["outer_floor_passed"] is False


def test_replay_threshold_loader_binds_sidecar_and_required_scenarios(tmp_path) -> None:
    path = tmp_path / "replay.json"
    path.write_text(
        json.dumps(
            {
                "sidecar_sha256": "a" * 64,
                "calibration": {
                    "scenarios": {
                        "cr2": {"threshold": 2.0},
                        "cr4": {"threshold": 0.0},
                        "cr8": {"threshold": -2.0},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert load_scenario_thresholds(
        path,
        expected_sidecar_sha256="a" * 64,
    ) == {"no_evict": None, "cr2": 2.0, "cr4": 0.0, "cr8": -2.0}
    with pytest.raises(ValueError, match="different sidecar"):
        load_scenario_thresholds(path, expected_sidecar_sha256="b" * 64)
