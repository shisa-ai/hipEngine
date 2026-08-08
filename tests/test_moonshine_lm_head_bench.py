from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

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


def test_retained_moonshine_lm_head_artifact_replays_decision_arithmetic() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact = json.loads(
        (
            repo_root
            / "benchmarks/results/2026-08-08-gfx1151-moonshine-lm-head-wave8-top1-retained.json"
        ).read_text()
    )
    assert artifact["kind"] == "hipengine_moonshine_lm_head_ab"
    assert artifact["status"] == "accepted_kernel_microbenchmark"
    assert artifact["performance_claim"] is True
    assert artifact["decision"]["retain_candidate"] is True
    assert artifact["decision"]["runtime_default_change"] is False
    provenance = artifact["provenance"]
    assert provenance["dirty"] is False
    assert provenance["untracked_count"] == 0
    assert provenance["hipengine_commit"].startswith("382a49de8")

    routes = artifact["timing"]["routes"]
    for scope in ("event_us", "wall_us"):
        baseline = routes["wave8_argmax"][scope]
        candidate = routes["wave8_top1"][scope]
        assert statistics.median(baseline["samples_us"]) == pytest.approx(
            baseline["median_us"]
        )
        assert statistics.median(candidate["samples_us"]) == pytest.approx(
            candidate["median_us"]
        )
        assert candidate["median_us"] < baseline["median_us"]
        assert candidate["p95_us"] < baseline["p95_us"]
    assert artifact["timing"]["event_improvement_percent"] == pytest.approx(
        improvement_percent(
            routes["wave8_argmax"]["event_us"]["median_us"],
            routes["wave8_top1"]["event_us"]["median_us"],
        )
    )
    assert artifact["timing"]["wall_improvement_percent"] == pytest.approx(
        improvement_percent(
            routes["wave8_argmax"]["wall_us"]["median_us"],
            routes["wave8_top1"]["wall_us"]["median_us"],
        )
    )

    state_gate = artifact["model_state_gate"]
    assert state_gate["status"] == "passed"
    assert state_gate["token_positions_exact"] == 194
    assert state_gate["full_fp16_logit_pairs_byte_exact"] == 8
    assert state_gate["final_hidden_pairs_byte_exact"] == 8
    assert state_gate["complete_self_kv_byte_exact"] is True
    assert state_gate["complete_cross_kv_byte_exact"] is True
    assert state_gate["teardown_returned_to_baseline"] is True

    for relative, expected in artifact["source_files"].items():
        actual = hashlib.sha256((repo_root / relative).read_bytes()).hexdigest()
        assert actual == expected
