from __future__ import annotations

import pytest

from scripts.gguf_prefix_reuse_bench import _distribution, _summarize_comparison


def _row(mode: str, ttft_ms: float, token_id: int, live_pages: int) -> dict:
    return {
        "mode": mode,
        "continuation_ttft_ms": ttft_ms,
        "continuation_token_id": token_id,
        "prefix_usable_hits_delta": 1 if mode == "radix" else 0,
        "prefix_admission_fallbacks_delta": 0,
        "prefix_reused_tokens": 256 if mode == "radix" else 0,
        "refcounted_pages": live_pages,
        "final_refcounted_pages": 0,
    }


def test_distribution_reports_required_e2e_statistics() -> None:
    summary = _distribution([10.0, 20.0, 30.0])

    assert summary["samples"] == [10.0, 20.0, 30.0]
    assert summary["count"] == 3
    assert summary["median"] == 20.0
    assert summary["p95"] == pytest.approx(29.0)
    assert summary["min"] == 10.0
    assert summary["max"] == 30.0
    assert summary["stdev"] == pytest.approx(8.16496580927726)


def test_comparison_requires_exact_outputs_hits_savings_and_final_drain() -> None:
    baseline = [_row("off", value, 264, 4) for value in (100.0, 110.0, 120.0)]
    radix = [_row("radix", value, 264, 3) for value in (20.0, 22.0, 24.0)]

    summary = _summarize_comparison(baseline, radix, prefix_tokens=256)

    assert summary["passed"] is True
    assert summary["correctness_exact"] is True
    assert summary["radix_hit_rate"] == 1.0
    assert summary["saved_live_pages"] == 1
    assert summary["ttft_speedup"] == 5.0
    assert summary["ttft_reduction_percent"] == 80.0

    radix[1]["continuation_token_id"] = 999
    assert _summarize_comparison(baseline, radix, prefix_tokens=256)["passed"] is False
