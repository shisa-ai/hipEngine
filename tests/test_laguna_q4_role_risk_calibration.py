from __future__ import annotations

import numpy as np
import pytest

from scripts.laguna_q4_role_risk_calibration import (
    _FEATURE_THRESHOLDS,
    _activation_risk_features,
    _aggregate_role_error,
    _bf16_bits_to_f32,
    _repair_economics,
    _select_prompts,
    _threshold_sweep,
)


def test_bf16_bits_to_f32_preserves_bf16_values() -> None:
    bits = np.asarray([0x0000, 0x3F80, 0xC020, 0x7F80], dtype=np.uint16)

    converted = _bf16_bits_to_f32(bits)

    assert converted.tolist() == [0.0, 1.0, -2.5, float("inf")]


def test_activation_risk_features_detect_half_scale_imbalance() -> None:
    hidden = np.ones((2, 32), dtype=np.float32)
    hidden[1, :16] = 16.0

    features = _activation_risk_features(hidden)

    assert features["half_scale_ratio_max"].tolist() == [1.0, 16.0]
    assert features["half_scale_ratio_fraction_gt2"].tolist() == [0.0, 1.0]
    assert features["d4_vs_d8_delta_relative_l2"][0] == pytest.approx(0.0)
    assert features["d4_vs_d8_delta_relative_l2"][1] > 0.0


def test_role_error_aggregates_compact_routes_to_source_rows() -> None:
    gate_d8 = np.asarray([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    gate_d4 = gate_d8.copy()
    gate_d4[0, 0] += 1.0
    gate_d4[2, 1] += 2.0
    up_d8 = np.ones_like(gate_d8)

    result = _aggregate_role_error(
        gate_d4,
        gate_d8,
        up_d8,
        np.asarray([1, 0, 1], dtype=np.int64),
        np.asarray([1.0, 1.0, 0.5], dtype=np.float32),
        rows=2,
    )

    assert result["gate_error_sq"].tolist() == [0.0, 5.0]
    assert result["route_weighted_intermediate_error_sq"][0] == 0.0
    assert result["route_weighted_intermediate_error_sq"][1] > 0.0


def test_threshold_sweep_reports_repair_and_error_coverage() -> None:
    rows = _threshold_sweep(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([0.0, 1.0, 1.0, 8.0]),
        thresholds=(3.0, 4.0),
    )

    assert rows[0]["repair_fraction"] == 0.5
    assert rows[0]["error_mass_coverage"] == pytest.approx(0.9)
    assert rows[1]["repair_fraction"] == 0.25
    assert rows[1]["error_mass_coverage"] == pytest.approx(0.8)
    assert rows[1]["severe_row_coverage"] == 1.0


def test_calibration_thresholds_are_fixed_and_transferable() -> None:
    assert 0.01 in _FEATURE_THRESHOLDS["d4_vs_d8_delta_max_abs"]
    assert 1.0 in _FEATURE_THRESHOLDS["activation_abs_max"]
    assert all(
        tuple(sorted(thresholds)) == tuple(thresholds)
        for thresholds in _FEATURE_THRESHOLDS.values()
    )


def test_repair_economics_counts_experts_and_mmq_padding() -> None:
    economics = _repair_economics(
        np.asarray([False, True, False, True]),
        np.asarray([0, 1, 2, 3, 1, 3], dtype=np.int64),
        np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
        expert_count=3,
        tile_rows=2,
    )

    assert economics == {
        "repair_source_rows": 2,
        "repair_route_rows": 4,
        "repair_active_experts": 3,
        "full_active_experts": 3,
        "repair_padded_rows": 6,
        "full_padded_rows": 6,
        "active_expert_fraction": 1.0,
        "padded_row_fraction": 1.0,
    }


def test_select_prompts_declares_calibration_subset() -> None:
    prompts = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

    assert [
        row["id"]
        for row in _select_prompts(
            prompts,
            prompt_ids=("c", "a"),
            prompt_count=1,
        )
    ] == ["c", "a"]
    assert [
        row["id"]
        for row in _select_prompts(
            prompts,
            prompt_ids=None,
            prompt_count=2,
        )
    ] == ["a", "b"]
