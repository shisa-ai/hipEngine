from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scripts.laguna_grouped_down_category_bench import (
    F16_WMMA_COMP_SWA_COMPARISON,
    GROUPED_COMBINE_COMPARISON,
    MODES,
    _aggregate,
    _load_shape_screen,
    _mode_order,
    _paired_free_running,
    _promotion_gate,
    _teacher_forced_quality,
)

HORIZONS = (16, 32)
CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")


def _rows(
    *,
    modes: tuple[str, str] = MODES,
    baseline_prefill: float = 2.0,
    candidate_prefill: float = 1.8,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prompt_index, category in enumerate(CATEGORIES):
        prompt_id = f"prompt_{prompt_index}"
        for repetition in range(3):
            generated = list(range(32))
            for mode in modes:
                prefill = baseline_prefill if mode == modes[0] else candidate_prefill
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
                    "direct_top1": step,
                    "adaptive_grouped_smallm_top1": step if matched else step + 100,
                    "top1_agreement": matched,
                    "finite": True,
                }
            )
        rows.append(
            {"prompt_id": f"prompt_{index}", "category": category, "steps": steps}
        )
    return rows


def test_grouped_down_category_mode_order_is_counterbalanced() -> None:
    for index in range(10):
        assert _mode_order(index, 1) == tuple(reversed(_mode_order(index, 0)))
        assert _mode_order(index, 2) == _mode_order(index, 0)


def test_grouped_combine_category_accepts_exact_nonregressive_wall() -> None:
    comparison = GROUPED_COMBINE_COMPARISON
    rows = _rows(
        modes=comparison.modes,
        baseline_prefill=2.0,
        candidate_prefill=2.001,
    )
    free_running = _paired_free_running(
        rows,
        HORIZONS,
        comparison=comparison,
    )
    aggregate = _aggregate(rows, HORIZONS, comparison=comparison)
    gate = _promotion_gate(
        aggregate,
        free_running,
        _teacher_forced_quality(_teacher_rows()),
        {"pass": True},
        {"pass": True},
        horizons=HORIZONS,
        recovered=True,
        comparison=comparison,
    )

    assert _mode_order(0, 0, comparison=comparison) == comparison.modes
    assert aggregate[comparison.aggregate_key]["prefill_speedup"] == pytest.approx(
        2.0 / 2.001
    )
    assert free_running["all_pairs_exact"] is True
    assert gate["pass"] is True
    assert gate["policy"]["performance"].startswith("aggregate/category prefill >=0.995x")


def test_grouped_combine_category_loads_matching_screen(tmp_path) -> None:
    screen = tmp_path / "combine-screen.json"
    screen.write_text(
        json.dumps(
            {
                "kind": GROUPED_COMBINE_COMPARISON.screen_kind,
                "status": GROUPED_COMBINE_COMPARISON.screen_status,
                "pass": True,
                "screen": {
                    "pass": True,
                    "regressed_rows": [],
                    "effective_speedup": 0.9997,
                },
                "model": {"sha256": "model-sha"},
                "repo": {"revision": "candidate-revision"},
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(shape_screen=screen, model_sha256="model-sha")

    result = _load_shape_screen(
        args,
        comparison=GROUPED_COMBINE_COMPARISON,
    )

    assert result["pass"] is True
    assert result["comparison"] == "grouped_combine"
    assert result["aggregate_speedup"] == pytest.approx(0.9997)


def test_f16_wmma_comp_swa_category_requires_matching_compensated_screen(
    tmp_path,
) -> None:
    screen = tmp_path / "f16-comp-screen.json"
    artifact = {
        "kind": F16_WMMA_COMP_SWA_COMPARISON.screen_kind,
        "status": F16_WMMA_COMP_SWA_COMPARISON.screen_status,
        "pass": True,
        "summary": {"pass": True, "failed_checks": []},
        "protocol": {"candidate_variant": "wmma_comp"},
        "repo": {"revision": "candidate-revision"},
    }
    screen.write_text(json.dumps(artifact), encoding="utf-8")
    args = SimpleNamespace(shape_screen=screen, model_sha256="model-sha")

    result = _load_shape_screen(
        args,
        comparison=F16_WMMA_COMP_SWA_COMPARISON,
    )
    assert result["pass"] is True
    assert result["comparison"] == "f16_wmma_comp_swa"
    assert result["model_sha256"] is None
    assert result["candidate_variant"] == "wmma_comp"

    artifact["protocol"]["candidate_variant"] = "wmma"
    screen.write_text(json.dumps(artifact), encoding="utf-8")
    assert _load_shape_screen(
        args,
        comparison=F16_WMMA_COMP_SWA_COMPARISON,
    )["pass"] is False


def test_f16_wmma_comp_swa_reports_but_does_not_require_exact_trajectories() -> None:
    comparison = F16_WMMA_COMP_SWA_COMPARISON
    rows = _rows(
        modes=comparison.modes,
        baseline_prefill=2.0,
        candidate_prefill=1.0,
    )
    for row in rows:
        if row["mode"] == comparison.modes[1]:
            for checkpoint in row["checkpoints"].values():
                checkpoint["generated_token_ids"][-1] = 999
    free_running = _paired_free_running(
        rows,
        HORIZONS,
        comparison=comparison,
    )
    gate = _promotion_gate(
        _aggregate(rows, HORIZONS, comparison=comparison),
        free_running,
        _teacher_forced_quality(_teacher_rows()),
        {"pass": True},
        {"pass": True},
        horizons=HORIZONS,
        recovered=True,
        comparison=comparison,
    )

    assert free_running["all_pairs_exact"] is False
    assert free_running["same_mode_repeat_deterministic"] is True
    assert gate["pass"] is True
    assert gate["policy"]["free_running_ids"].startswith("report complete")


def test_grouped_down_category_gate_accepts_quality_and_full_model_win() -> None:
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
    comparison = aggregate["adaptive_grouped_smallm_vs_direct"]
    assert comparison["prefill_speedup"] == pytest.approx(2.0 / 1.8)
    assert promotion == {
        "pass": True,
        "failed_checks": [],
        "policy": promotion["policy"],
    }


def test_grouped_down_quality_fails_closed_on_category_or_kl() -> None:
    low_top1 = _teacher_forced_quality(_teacher_rows(top1_agreement=0.875))
    assert low_top1["pass"] is False
    assert "general_en_top1_below_0.9" in low_top1["failed_checks"]

    high_kl = _teacher_forced_quality(_teacher_rows(max_kl=0.051))
    assert high_kl["pass"] is False
    assert "max_kl_above_0.05" in high_kl["failed_checks"]


def test_grouped_down_free_running_ids_are_required() -> None:
    rows = _rows()
    mismatch_rows = deepcopy(rows)
    candidate = next(
        row for row in mismatch_rows if row["mode"] == "adaptive_grouped_smallm"
    )
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
    assert gate["pass"] is False
    assert "free_running_pairs_not_exact" in gate["failed_checks"]

    nondeterministic_rows = deepcopy(rows)
    repeat = next(
        row
        for row in nondeterministic_rows
        if row["mode"] == "adaptive_grouped_smallm" and row["repetition"] == 1
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


def test_grouped_down_category_gate_rejects_regression_or_missing_screen() -> None:
    rows = _rows()
    for row in rows:
        if (
            row["mode"] == "adaptive_grouped_smallm"
            and row["category"] == "general_ja"
        ):
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
    assert "shape_screen_failed" in gate["failed_checks"]
