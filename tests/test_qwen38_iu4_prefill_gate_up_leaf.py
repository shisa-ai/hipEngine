from __future__ import annotations

import pytest

from scripts.qwen38_iu4_prefill_gate_up_leaf import _prefill_control_metrics


def test_prefill_control_metrics_report_bandwidth_and_f16_roof() -> None:
    metrics = _prefill_control_metrics(
        rows=128,
        hidden=5120,
        out_features=17408,
        median_ms=2.0,
        pair_bytes=133_693_440,
    )

    expected_useful_ops = 4 * 128 * 5120 * 17408
    expected_executed_ops = 4 * 256 * 5120 * 17408
    assert metrics["wmma_useful_ops"] == expected_useful_ops
    assert metrics["wmma_executed_ops"] == expected_executed_ops
    assert metrics["wmma_row_blocks"] == 1
    assert metrics["wmma_executed_rows"] == 256
    assert metrics["wmma_row_utilization"] == pytest.approx(0.5)
    assert metrics["executed_payload_bytes"] == 133_693_440
    assert metrics["effective_payload_gbps"] == pytest.approx(66.84672)
    assert metrics["executed_tflops"] == pytest.approx(
        expected_executed_ops / 0.002 / 1e12
    )
    assert metrics["f16_wmma_roof_tflops"] == pytest.approx(55.066)
    assert metrics["fraction_of_f16_wmma_roof"] == pytest.approx(
        metrics["executed_tflops"] / 55.066
    )
    assert metrics["percent_of_f16_wmma_roof"] == pytest.approx(
        100.0 * metrics["fraction_of_f16_wmma_roof"]
    )


def test_prefill_control_metrics_count_each_256_row_weight_sweep() -> None:
    metrics = _prefill_control_metrics(
        rows=1024,
        hidden=5120,
        out_features=17408,
        median_ms=16.0,
        pair_bytes=133_693_440,
    )

    assert metrics["wmma_row_blocks"] == 4
    assert metrics["wmma_executed_rows"] == 1024
    assert metrics["wmma_row_utilization"] == pytest.approx(1.0)
    assert metrics["executed_payload_bytes"] == 4 * 133_693_440
    assert metrics["effective_payload_gbps"] == pytest.approx(33.42336)
