from __future__ import annotations

import pytest

from scripts.qwen38_iu4_prefill_ab import (
    _candidate_metrics,
    _family_amdahl_projection,
)


def test_candidate_metrics_charge_one_weight_sweep_per_256_rows() -> None:
    metrics = _candidate_metrics(
        rows=512,
        hidden=5120,
        out_features=17408,
        inclusive_ms=3.0,
        core_ms=2.9,
        pair_bytes=89_407_488,
    )

    assert metrics["padded_rows"] == 512
    assert metrics["weight_sweeps"] == 2
    assert metrics["accumulators_per_wave"] == 16
    assert metrics["core_executed_tops"] == pytest.approx(
        4 * 512 * 5120 * 17408 / 0.0029 / 1e12
    )
    assert metrics["core_effective_weight_gbps"] == pytest.approx(
        2 * 89_407_488 / 2.9 / 1e6
    )
    assert metrics["percent_of_iu4_arithmetic_roof"] == pytest.approx(
        100.0 * metrics["core_executed_tops"] / 109.715
    )


def test_family_amdahl_projection_replaces_only_gate_up_family() -> None:
    projection = _family_amdahl_projection(
        control_layer_ms=8.0,
        candidate_layer_ms=3.0,
        layers=64,
        prompt_tokens=512,
        complete_prefill_tok_s=400.0,
    )

    assert projection["complete_prefill_ms"] == pytest.approx(1280.0)
    assert projection["control_family_ms"] == pytest.approx(512.0)
    assert projection["candidate_family_ms"] == pytest.approx(192.0)
    assert projection["control_family_wall_share"] == pytest.approx(0.4)
    assert projection["projected_complete_prefill_ms"] == pytest.approx(960.0)
    assert projection["projected_complete_prefill_speedup"] == pytest.approx(4.0 / 3.0)
    assert projection["projected_complete_prefill_tok_s"] == pytest.approx(512 / 0.96)
