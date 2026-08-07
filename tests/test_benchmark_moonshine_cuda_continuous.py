from __future__ import annotations

import pytest

from scripts.benchmark_moonshine_cuda_continuous import (
    WorkloadResult,
    _gpu_processes,
    parse_batches,
    staggered_arrival_steps,
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


def test_continuous_benchmark_exclusivity_filters_selected_gpu(monkeypatch) -> None:
    outputs = iter(
        (
            "0, GPU-zero\n1, GPU-one\n2, GPU-two\n",
            "GPU-one, 10, other, 100\nGPU-two, 11, other, 200\n",
        )
    )
    monkeypatch.setattr(
        "scripts.benchmark_moonshine_cuda_continuous.subprocess.check_output",
        lambda *_args, **_kwargs: next(outputs),
    )
    assert _gpu_processes(0) == []


def test_continuous_benchmark_staggered_arrival_schedule() -> None:
    assert staggered_arrival_steps(8, 4, 2) == (0, 0, 0, 0, 1, 3, 5, 7)
    assert staggered_arrival_steps(2, 2, 1) == (0, 0)
    with pytest.raises(ValueError, match="interval_steps"):
        staggered_arrival_steps(2, 1, 0)
    with pytest.raises(ValueError, match="inconsistent"):
        staggered_arrival_steps(2, 3, 1)


def test_continuous_benchmark_batch_parser_rejects_invalid_values() -> None:
    assert parse_batches("2,4,8") == (2, 4, 8)
    with pytest.raises(Exception, match="unique positive"):
        parse_batches("2,2")
    with pytest.raises(Exception, match="unique positive"):
        parse_batches("0")
