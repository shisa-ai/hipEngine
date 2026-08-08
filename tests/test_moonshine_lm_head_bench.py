from __future__ import annotations

import json

import pytest

from scripts.moonshine_lm_head_bench import (
    _load_profiler_summary,
    improvement_percent,
    summarize_samples,
)


def test_moonshine_lm_head_bench_summarizes_raw_samples_and_improvement() -> None:
    summary = summarize_samples([4.0, 1.0, 3.0, 2.0])
    assert summary == {
        "samples_us": [4.0, 1.0, 3.0, 2.0],
        "median_us": 2.5,
        "mean_us": 2.5,
        "p95_us": 4.0,
        "min_us": 1.0,
        "max_us": 4.0,
        "stdev_us": pytest.approx(1.2909944487358056),
    }
    assert improvement_percent(250.0, 200.0) == pytest.approx(20.0)
    with pytest.raises(ValueError, match="samples"):
        summarize_samples([])
    with pytest.raises(ValueError, match="baseline"):
        improvement_percent(0.0, 1.0)


def test_moonshine_lm_head_bench_profiler_summary_requires_both_kernels(
    tmp_path,
) -> None:
    path = tmp_path / "profiler.json"
    payload = {
        "status": "passed",
        "observed_kernels": [
            "moonshine_f16_lm_head_projection_wave8_top1_kernel(...)",
            "moonshine_lm_head_top1_reduce_kernel(...)",
        ],
        "durations_ns": [170000, 5000],
    }
    path.write_text(json.dumps(payload))
    assert _load_profiler_summary(path) == payload

    path.write_text(
        json.dumps(
            {
                "status": "incomplete",
                "observed_kernels": [
                    "moonshine_f16_lm_head_projection_wave8_top1_kernel(...)"
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="missing"):
        _load_profiler_summary(path)
