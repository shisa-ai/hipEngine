from __future__ import annotations

import pytest

from scripts.benchmark_moonshine_cuda_continuous import (
    WorkloadResult,
    parse_batches,
    timing_summary,
)


def test_continuous_benchmark_timing_summary_keeps_raw_samples() -> None:
    samples = [
        WorkloadResult(100.0, (10.0, 20.0), {"a": (2,)}),
        WorkloadResult(80.0, (8.0, 16.0), {"a": (2,)}),
        WorkloadResult(120.0, (12.0, 24.0), {"a": (2,)}),
    ]
    result = timing_summary(samples, request_count=2)
    assert result["wall_ms_raw"] == [100.0, 80.0, 120.0]
    assert result["requests_per_s_raw"] == [20.0, 25.0, 100.0 / 6.0]
    assert result["request_latency_ms_raw"] == [
        [10.0, 20.0],
        [8.0, 16.0],
        [12.0, 24.0],
    ]
    assert result["wall_median_ms"] == 100.0
    assert result["requests_per_s_median"] == 20.0
    assert result["request_latency_p50_ms"] == 14.0


def test_continuous_benchmark_batch_parser_rejects_invalid_values() -> None:
    assert parse_batches("2,4,8") == (2, 4, 8)
    with pytest.raises(Exception, match="unique positive"):
        parse_batches("2,2")
    with pytest.raises(Exception, match="unique positive"):
        parse_batches("0")
