from __future__ import annotations

import pytest

from scripts.gguf_prefix_reuse_bench import (
    _correctness_prerequisite_matches,
    _distribution,
    _paired_delta_distribution,
    _summarize_comparison,
    build_parser,
)


def _row(
    mode: str,
    ttft_ms: float,
    token_id: int,
    live_pages: int,
    *,
    source_lifecycle: str = "active",
) -> dict:
    completed_radix = source_lifecycle == "completed" and mode == "radix"
    return {
        "mode": mode,
        "continuation_ttft_ms": ttft_ms,
        "continuation_token_id": token_id,
        "prefix_usable_hits_delta": 1 if mode == "radix" else 0,
        "prefix_admission_fallbacks_delta": 0,
        "prefix_reused_tokens": 256 if mode == "radix" else 0,
        "refcounted_pages": live_pages,
        "final_refcounted_pages": 0,
        "prefix_snapshot_hit": completed_radix,
        "cache_refcount_after_source_release": 1 if completed_radix else 0,
        "cache_refcount_after_continuation_release": 1 if completed_radix else 0,
        "snapshot_evicted": completed_radix,
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


def test_paired_delta_distribution_does_not_turn_one_lazy_growth_into_a_claim() -> None:
    baseline = [{"bytes": value} for value in (100, 300, 300)]
    radix = [{"bytes": value} for value in (100, 100, 300)]

    summary = _paired_delta_distribution(baseline, radix, key="bytes")

    assert summary["samples"] == [0.0, 200.0, 0.0]
    assert summary["median"] == 0.0
    assert summary["all_positive"] is False


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


def test_completed_comparison_accepts_zero_live_page_savings_but_requires_snapshot_lifecycle() -> None:
    baseline = [
        _row("off", value, 264, 2, source_lifecycle="completed")
        for value in (100.0, 110.0, 120.0)
    ]
    radix = [
        _row("radix", value, 264, 2, source_lifecycle="completed")
        for value in (20.0, 22.0, 24.0)
    ]

    summary = _summarize_comparison(
        baseline,
        radix,
        prefix_tokens=256,
        source_lifecycle="completed",
    )
    assert summary["passed"] is True
    assert summary["saved_live_pages"] == 0
    assert summary["snapshot_lifecycle_exact"] is True

    radix[0]["prefix_snapshot_hit"] = False
    assert not _summarize_comparison(
        baseline,
        radix,
        prefix_tokens=256,
        source_lifecycle="completed",
    )["passed"]


def test_completed_economics_requires_matching_correctness_artifact_and_cli() -> None:
    completed = {
        "kind": "gguf_completed_prefix_reuse_correctness_gate",
        "passed": True,
        "workload": {"source_lifecycle": "completed"},
    }
    assert _correctness_prerequisite_matches(completed, "completed")
    assert not _correctness_prerequisite_matches(completed, "active")

    args = build_parser().parse_args(["--source-lifecycle", "completed"])
    assert args.source_lifecycle == "completed"
