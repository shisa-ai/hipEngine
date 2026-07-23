from __future__ import annotations

from scripts.laguna_wmma16_down_prefill_bench import (
    BASELINE_MODE,
    CANDIDATE_MODE,
    MODES,
    PROFILE_ROWS,
    _comparison_summary,
    _mode_order,
)


def _fixture(*, candidate_seconds: float = 1.0, kl: float = 0.01, agree: bool = True):
    samples = {
        BASELINE_MODE: {row: [2.0, 2.0] for row in PROFILE_ROWS},
        CANDIDATE_MODE: {row: [candidate_seconds, candidate_seconds] for row in PROFILE_ROWS},
    }
    next_tokens = {
        BASELINE_MODE: {row: [7, 7] for row in PROFILE_ROWS},
        CANDIDATE_MODE: {
            row: [7, 7] if agree else [8, 8] for row in PROFILE_ROWS
        },
    }
    quality = {
        row: [
            {
                "kl_divergence": kl,
                "baseline_top1": 7,
                "candidate_top1": 7 if agree else 8,
                "top1_agreement": agree,
                "finite": True,
            }
            for _ in range(2)
        ]
        for row in PROFILE_ROWS
    }
    return samples, next_tokens, quality


def test_laguna_wmma16_screen_counterbalances_modes() -> None:
    assert _mode_order(0, 0) == MODES
    assert _mode_order(0, 1) == tuple(reversed(MODES))
    assert _mode_order(1, 0) == tuple(reversed(MODES))


def test_laguna_wmma16_screen_accepts_faster_quality_safe_candidate() -> None:
    samples, next_tokens, quality = _fixture()
    summary = _comparison_summary(samples, next_tokens, quality, rows=PROFILE_ROWS)

    assert summary["correctness"]["pass"] is True
    assert summary["screen"]["pass"] is True
    assert summary["screen"]["aggregate_speedup"] == 2.0


def test_laguna_wmma16_screen_rejects_quality_regression() -> None:
    samples, next_tokens, quality = _fixture(kl=0.06, agree=False)
    summary = _comparison_summary(samples, next_tokens, quality, rows=PROFILE_ROWS)

    assert summary["correctness"]["pass"] is False
    assert "max_kl_above_0.05" in summary["screen"]["failed_checks"]
    assert "top1_or_next_id_mismatch" in summary["screen"]["failed_checks"]


def test_laguna_wmma16_screen_rejects_non_improving_shape() -> None:
    samples, next_tokens, quality = _fixture(candidate_seconds=2.1)
    summary = _comparison_summary(samples, next_tokens, quality, rows=PROFILE_ROWS)

    assert summary["screen"]["pass"] is False
    assert summary["screen"]["slower_rows"] == [256, 512]
    assert "wmma16_not_faster_at_every_shape" in summary["screen"]["failed_checks"]
