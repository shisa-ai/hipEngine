from __future__ import annotations

import pytest

from scripts.qwen38_iu4_s4_gate_up_leaf import (
    _implementation_quality_metrics,
    _kernel_resource_assessment,
)


def test_implementation_quality_names_binding_roof_and_percentages() -> None:
    metrics = _implementation_quality_metrics(
        rows=2,
        core_executed_tops=9.819225649841309,
        candidate_effective_weight_gbps=153.9048551562041,
        control_effective_weight_gbps=218.34868192082143,
    )

    assert metrics["binding_roof"] == "memory"
    assert metrics["binding_roof_value"] == pytest.approx(221.0)
    assert metrics["binding_roof_unit"] == "GB/s"
    assert metrics["candidate_effective_weight_gbps"] == pytest.approx(153.9048551562041)
    assert metrics["control_effective_weight_gbps"] == pytest.approx(218.34868192082143)
    assert metrics["candidate_fraction_of_memory_roof"] == pytest.approx(0.6964020595)
    assert metrics["candidate_percent_of_memory_roof"] == pytest.approx(69.64020595)
    assert metrics["candidate_fraction_of_arithmetic_roof"] == pytest.approx(
        9.819225649841309 / 109.715
    )
    assert metrics["candidate_fraction_of_binding_roof"] == pytest.approx(0.6964020595)


def test_implementation_quality_switches_to_arithmetic_roof_at_m128() -> None:
    metrics = _implementation_quality_metrics(
        rows=128,
        core_executed_tops=12.54,
        candidate_effective_weight_gbps=196.5,
        control_effective_weight_gbps=98.0,
    )

    assert metrics["binding_roof"] == "arithmetic"
    assert metrics["binding_roof_value"] == pytest.approx(109.715)
    assert metrics["binding_roof_unit"] == "TOPS"
    assert metrics["candidate_fraction_of_binding_roof"] == pytest.approx(12.54 / 109.715)


def test_low_vgpr_single_accumulator_wmma_is_flagged() -> None:
    assessment = _kernel_resource_assessment(
        accumulators_per_wave=1,
        vgpr_count=24,
    )

    assert assessment["accumulators_per_wave"] == 1
    assert assessment["vgpr_anomaly"] is True
    assert assessment["vgpr_anomaly_threshold"] == 64
    assert "unblocked" in assessment["vgpr_anomaly_reason"]


def test_blocked_wmma_resource_shape_is_not_flagged() -> None:
    assessment = _kernel_resource_assessment(
        accumulators_per_wave=8,
        vgpr_count=88,
    )

    assert assessment["vgpr_anomaly"] is False
