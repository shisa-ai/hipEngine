from __future__ import annotations

import json
from pathlib import Path


_ARTIFACT = (
    Path(__file__).parents[1]
    / "benchmarks/results/2026-08-15-gfx1151-qwen35-08b-prompt-threshold-sweep.json"
)


def test_qwen35_08b_threshold_sweep_preserves_route_scope() -> None:
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    assert payload["status"] == "retained_diagnostic"
    assert payload["source"]["tracked_source_clean"] is True
    assert payload["source"]["all_children_clean"] is True
    assert payload["integrity"] == {
        "children": 187,
        "all_finite": True,
        "all_final_token_ids": [9707],
        "all_expected_scratch_positions": True,
    }

    q4 = payload["results"]["q4"]["comparisons_vs_current"]
    assert q4["pre_x2"]["512"]["ratio"] > 1.5
    assert 0.95 < q4["pre_x2"]["513"]["ratio"] < 1.05
    assert 0.95 < q4["pre_x2"]["4096"]["ratio"] < 1.05
    assert all(row["ratio"] > 1.0 for row in q4["strict_x2"].values())

    q8 = payload["results"]["q8"]["comparisons_vs_current"]["strict_x2"]
    assert all(row["ratio"] > 1.0 for row in q8.values())
    assert q8["512"]["paired_wins"] == q8["512"]["blocks"] == 4
