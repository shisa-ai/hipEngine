from __future__ import annotations

from scripts.laguna_wmma16_down_category_bench import (
    _aggregate,
    _expanded_prompt_tokens,
    _free_running_summary,
    _promotion_gate,
    _quality_summary,
)
from scripts.laguna_wmma16_down_prefill_bench import (
    BASELINE_MODE,
    CANDIDATE_MODE,
    PROFILE_ROWS,
)

HORIZONS = (16, 32)
CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")


def _runs() -> list[dict]:
    result = []
    for repetition in range(2):
        for rows in PROFILE_ROWS:
            for category in CATEGORIES:
                prompt_id = f"{category}_prompt"
                generated = list(range(32))
                for mode, prefill_seconds in (
                    (BASELINE_MODE, 2.0),
                    (CANDIDATE_MODE, 1.8),
                ):
                    result.append(
                        {
                            "repetition": repetition,
                            "rows": rows,
                            "prompt_id": prompt_id,
                            "category": category,
                            "mode": mode,
                            "prompt_tokens": rows,
                            "prefill_seconds": prefill_seconds,
                            "checkpoints": {
                                "16": {
                                    "output_tokens": 16,
                                    "decode_forward_calls": 15,
                                    "decode_seconds": 1.0,
                                    "total_seconds": prefill_seconds + 1.0,
                                    "generated_token_ids": generated[:16],
                                    "generated_ids_sha256": "h16",
                                },
                                "32": {
                                    "output_tokens": 32,
                                    "decode_forward_calls": 31,
                                    "decode_seconds": 2.0,
                                    "total_seconds": prefill_seconds + 2.0,
                                    "generated_token_ids": generated,
                                    "generated_ids_sha256": "h32",
                                },
                            },
                        }
                    )
    return result


def _quality(*, kl: float = 0.01, agree: bool = True) -> list[dict]:
    return [
        {
            "category": category,
            "finite": True,
            "kl_divergence": kl,
            "top1_agreement": agree,
        }
        for category in CATEGORIES
        for _ in range(2)
    ]


def test_expanded_category_prompt_repeats_without_leading_bos() -> None:
    prompt = {"token_ids": (1, 2, 3)}

    assert _expanded_prompt_tokens(prompt, 2) == (1, 2)
    assert _expanded_prompt_tokens(prompt, 8) == (1, 2, 3, 2, 3, 2, 3, 2)


def test_wmma16_category_policy_accepts_faster_exact_trajectories() -> None:
    runs = _runs()
    aggregate = _aggregate(runs, rows=PROFILE_ROWS, horizons=HORIZONS)
    quality = _quality_summary(_quality())
    free_running = _free_running_summary(runs, HORIZONS)
    promotion = _promotion_gate(
        aggregate,
        quality,
        free_running,
        rows=PROFILE_ROWS,
        horizons=HORIZONS,
    )

    assert quality["pass"] is True
    assert free_running["pass"] is True
    assert aggregate["comparison"]["overall"]["prefill_speedup"] > 1.0
    assert promotion["pass"] is True


def test_wmma16_category_quality_rejects_excess_kl_and_top1() -> None:
    summary = _quality_summary(_quality(kl=0.06, agree=False))

    assert summary["pass"] is False
    assert "max_kl_above_0.05" in summary["failed_checks"]
    assert "suite_top1_below_0.9" in summary["failed_checks"]
    assert "code_top1_below_0.9" in summary["failed_checks"]


def test_wmma16_category_free_running_rejects_mode_mismatch() -> None:
    runs = _runs()
    candidate = next(row for row in runs if row["mode"] == CANDIDATE_MODE)
    candidate["checkpoints"]["32"]["generated_token_ids"] = [999] * 32

    summary = _free_running_summary(runs, HORIZONS)

    assert summary["pass"] is False
    assert summary["mismatches"]


def test_wmma16_category_policy_rejects_category_prefill_regression() -> None:
    runs = _runs()
    for row in runs:
        if row["mode"] == CANDIDATE_MODE and row["category"] == "code":
            row["prefill_seconds"] = 2.2
            for checkpoint in row["checkpoints"].values():
                checkpoint["total_seconds"] = 2.2 + checkpoint["decode_seconds"]
    aggregate = _aggregate(runs, rows=PROFILE_ROWS, horizons=HORIZONS)
    promotion = _promotion_gate(
        aggregate,
        _quality_summary(_quality()),
        _free_running_summary(runs, HORIZONS),
        rows=PROFILE_ROWS,
        horizons=HORIZONS,
    )

    assert promotion["pass"] is False
    assert "rows_256_code_prefill_not_faster" in promotion["failed_checks"]
    assert "rows_512_code_prefill_not_faster" in promotion["failed_checks"]
