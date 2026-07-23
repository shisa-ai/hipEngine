from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.laguna_selected_q8_category_bench import (
    MODES,
    _aggregate,
    _mode_order,
    _paired_free_running,
    _promotion_gate,
    _teacher_forced_quality,
)

HORIZONS = (16, 32)
CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prompt_index, category in enumerate(CATEGORIES):
        prompt_id = f"prompt_{prompt_index}"
        for repetition in range(3):
            generated = list(range(32))
            for mode in MODES:
                prefill = 2.0 if mode == "split" else 1.8
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "category": category,
                        "prompt_tokens": 96,
                        "mode": mode,
                        "repetition": repetition,
                        "prefill_seconds": prefill,
                        "ttft_seconds": prefill,
                        "checkpoints": {
                            str(horizon): {
                                "generated_token_ids": generated[:horizon],
                                "generated_ids_sha256": f"{mode}-{prompt_id}-{horizon}",
                                "decode_forward_calls": horizon - 1,
                                "decode_seconds": float(horizon - 1),
                                "output_tokens": horizon,
                                "total_seconds": prefill + horizon - 1,
                                "decode_tok_s": 1.0,
                                "e2e_output_tok_s": horizon / (prefill + horizon - 1),
                            }
                            for horizon in HORIZONS
                        },
                    }
                )
    return rows


def _teacher_rows(*, top1_agreement: float = 1.0, max_kl: float = 1e-3):
    rows = []
    for index, category in enumerate(CATEGORIES):
        steps = []
        for step in range(32):
            matched = step / 32 < top1_agreement
            steps.append(
                {
                    "index": step,
                    "kl_divergence": max_kl,
                    "split_top1": step,
                    "q8_dp4a_top1": step if matched else step + 100,
                    "top1_agreement": matched,
                    "finite": True,
                }
            )
        rows.append(
            {"prompt_id": f"prompt_{index}", "category": category, "steps": steps}
        )
    return rows


def test_selected_q8_category_mode_order_is_counterbalanced() -> None:
    for index in range(10):
        assert _mode_order(index, 1) == tuple(reversed(_mode_order(index, 0)))
        assert _mode_order(index, 2) == _mode_order(index, 0)


def test_selected_q8_category_gate_accepts_quality_and_full_model_win() -> None:
    rows = _rows()
    free_running = _paired_free_running(rows, HORIZONS)
    teacher = _teacher_forced_quality(_teacher_rows())
    aggregate = _aggregate(rows, HORIZONS)
    promotion = _promotion_gate(
        aggregate,
        free_running,
        teacher,
        {"pass": True},
        {"pass": True},
        horizons=HORIZONS,
        recovered=True,
    )

    assert free_running["all_pairs_exact"] is True
    assert teacher["pass"] is True
    assert teacher["top1_agreement"] == 1.0
    assert aggregate["q8_dp4a_vs_split"]["prefill_speedup"] == pytest.approx(2.0 / 1.8)
    assert promotion == {
        "pass": True,
        "failed_checks": [],
        "policy": promotion["policy"],
    }


def test_selected_q8_quality_fails_closed_on_category_or_kl() -> None:
    low_top1 = _teacher_forced_quality(_teacher_rows(top1_agreement=0.875))
    assert low_top1["pass"] is False
    assert "general_en_top1_below_0.9" in low_top1["failed_checks"]

    high_kl = _teacher_forced_quality(_teacher_rows(max_kl=0.051))
    assert high_kl["pass"] is False
    assert "max_kl_above_0.05" in high_kl["failed_checks"]


def test_selected_q8_free_running_ids_are_diagnostic_not_quality_substitute() -> None:
    rows = _rows()
    mismatch_rows = deepcopy(rows)
    candidate = next(row for row in mismatch_rows if row["mode"] == "q8_dp4a")
    candidate["checkpoints"]["32"]["generated_token_ids"][-1] = 999
    free_running = _paired_free_running(mismatch_rows, HORIZONS)

    assert free_running["all_pairs_exact"] is False
    gate = _promotion_gate(
        _aggregate(mismatch_rows, HORIZONS),
        free_running,
        _teacher_forced_quality(_teacher_rows()),
        {"pass": True},
        {"pass": True},
        horizons=HORIZONS,
        recovered=True,
    )
    assert gate["pass"] is True

    nondeterministic_rows = deepcopy(rows)
    repeat = next(
        row
        for row in nondeterministic_rows
        if row["mode"] == "q8_dp4a" and row["repetition"] == 1
    )
    repeat["checkpoints"]["32"]["generated_ids_sha256"] = "changed"
    deterministic_gate = _promotion_gate(
        _aggregate(nondeterministic_rows, HORIZONS),
        _paired_free_running(nondeterministic_rows, HORIZONS),
        _teacher_forced_quality(_teacher_rows()),
        {"pass": True},
        {"pass": True},
        horizons=HORIZONS,
        recovered=True,
    )
    assert deterministic_gate["pass"] is False
    assert "free_running_repeat_not_deterministic" in deterministic_gate["failed_checks"]


def test_selected_q8_category_gate_rejects_regression_or_missing_screen() -> None:
    rows = _rows()
    for row in rows:
        if row["mode"] == "q8_dp4a" and row["category"] == "general_ja":
            row["prefill_seconds"] = 2.1
            row["ttft_seconds"] = 2.1
            for checkpoint in row["checkpoints"].values():
                checkpoint["total_seconds"] = 2.1 + checkpoint["decode_seconds"]
                checkpoint["e2e_output_tok_s"] = (
                    checkpoint["output_tokens"] / checkpoint["total_seconds"]
                )
    aggregate = _aggregate(rows, HORIZONS)
    gate = _promotion_gate(
        aggregate,
        _paired_free_running(rows, HORIZONS),
        _teacher_forced_quality(_teacher_rows()),
        {"pass": True},
        {"pass": False},
        horizons=HORIZONS,
        recovered=True,
    )

    assert gate["pass"] is False
    assert "general_ja_prefill_regressed" in gate["failed_checks"]
    assert "inclusive_screen_failed" in gate["failed_checks"]
