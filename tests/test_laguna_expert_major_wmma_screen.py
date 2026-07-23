from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.laguna_expert_major_wmma_screen import (
    EXACT_STATE_FIELDS,
    MODES,
    ROWS,
    _load_leaf_screen,
    _mode_order,
    _quality,
    _summarize,
)


def _records(*, candidate_speedups: dict[int, float] | None = None):
    speedups = candidate_speedups or {
        32: 0.95,
        55: 0.98,
        64: 0.99,
        122: 1.10,
        128: 1.12,
        256: 1.25,
        512: 1.50,
    }
    records = []
    comparisons = []
    for row in ROWS:
        for repetition in range(3):
            for mode in MODES:
                seconds = 2.0 if mode == "retained" else 2.0 / speedups[row]
                record = {
                    "rows": row,
                    "mode": mode,
                    "repetition": repetition,
                    "prefill_seconds": seconds + repetition * 0.001,
                }
                for field in EXACT_STATE_FIELDS:
                    if field == "final_position":
                        record[field] = row - 1
                    elif field == "next_token_id":
                        record[field] = 7 if mode == "retained" else 8
                    else:
                        record[field] = f"{mode}-{row}-{field}"
                records.append(record)
            comparisons.append(
                {
                    "rows": row,
                    "repetition": repetition,
                    "finite": True,
                    "kl_divergence": 0.01,
                    "top1_agreement": True,
                    "cursor_exact": True,
                }
            )
    return records, comparisons


def test_mode_order_counterbalances_every_shape_and_repetition() -> None:
    assert _mode_order(0, 0) == MODES
    assert _mode_order(1, 0) == tuple(reversed(MODES))
    assert _mode_order(0, 1) == tuple(reversed(MODES))


def test_summary_selects_smallest_nonregressive_threshold() -> None:
    records, comparisons = _records()
    summary = _summarize(records, comparisons)

    assert summary["pass"] is True
    assert summary["quality"]["pass"] is True
    assert summary["threshold"]["selected_rows"] == 122
    assert summary["threshold"]["policies"]["64"]["eligible"] is False
    assert summary["threshold"]["policies"]["122"]["eligible"] is True
    assert summary["shapes"]["512"]["expert_major_comp_vs_retained_speedup"] == pytest.approx(1.5, rel=1.0e-3)


def test_summary_rejects_quality_or_missing_positive_threshold() -> None:
    records, comparisons = _records(
        candidate_speedups={row: 0.9 for row in ROWS}
    )
    comparisons[0] = {
        **comparisons[0],
        "kl_divergence": 0.06,
        "top1_agreement": False,
    }
    summary = _summarize(records, comparisons)

    assert summary["pass"] is False
    assert summary["threshold"]["selected_rows"] is None
    assert summary["failed_checks"] == [
        "no_quality_safe_nonregressive_adaptive_threshold"
    ]


def test_summary_falls_back_past_a_quality_failing_shape() -> None:
    records, comparisons = _records()
    for comparison in comparisons:
        if comparison["rows"] == 122:
            comparison["top1_agreement"] = False
    summary = _summarize(records, comparisons)

    assert summary["pass"] is True
    assert summary["explicit_candidate_quality"]["pass"] is False
    assert summary["threshold"]["policies"]["122"]["quality"]["pass"] is False
    assert summary["threshold"]["selected_rows"] == 128
    assert summary["quality"]["pass"] is True
    assert summary["quality"]["top1_agreement"] == 1.0


def test_quality_reports_finite_kl_and_top1() -> None:
    reference = [0.0, 1.0, -1.0]
    candidate = [0.01, 0.99, -1.0]
    result = _quality(reference, candidate)
    assert result["finite"] is True
    assert result["kl_divergence"] < 0.05
    assert result["top1_agreement"] is True


def test_leaf_screen_requires_passing_bound_variant(tmp_path: Path) -> None:
    path = tmp_path / "leaf.json"
    path.write_text(
        json.dumps(
            {
                "kind": "hipengine_laguna_expert_major_wmma_leaf_screen",
                "status": "leaf_screen_passed",
                "pass": True,
                "candidate": {
                    "registry_variant": "selected_t16_expert_major_wmma_comp_bf16_bf16_out"
                },
            }
        ),
        encoding="utf-8",
    )
    assert _load_leaf_screen(path)["pass"] is True
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["candidate"]["registry_variant"] = "wrong"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="passing leaf artifact"):
        _load_leaf_screen(path)
