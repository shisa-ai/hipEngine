from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.mtp_paro_verifier_review_aggregate import aggregate


def _capture(path: Path, *, prompt: str, category: str, match: bool) -> Path:
    row = {
        "cycle": 1,
        "output_offset": 0,
        "row": 0,
        "row_role": "root",
        "decision_role": "draft_acceptance_or_reject_correction",
        "category": category,
        "shape": "c2_b1",
        "transition": "prefill_to_verify",
        "context": 10,
        "position": 10,
        "strict_selected_for_commit": True,
        "candidate_selected_for_commit": True,
        "cycle_task_decision_mismatch": not match,
        "strict_top1": 1,
        "candidate_top1": 1 if match else 2,
        "kl": 1.0e-4,
        "top1_equal": match,
        "top5_overlap": 1.0,
        "max_abs_logit_delta": 0.1,
        "strict_margin": 0.05,
    }
    mismatch = [] if match else [row]
    decision = [] if match else [
        {
            "cycle": 1,
            "output_offset": 0,
            "strict_accepted": 1,
            "candidate_accepted": 0,
            "strict_bonus": 3,
            "candidate_bonus": 2,
        }
    ]
    payload = {
        "model": "fixture-model",
        "backend": "hip_gfx1100",
        "status": "passed" if match else "rejected",
        "prompt": {
            "name": prompt,
            "category": category,
            "split": "heldout",
            "render": "raw",
        },
        "manifests": {"candidate_review_sha256": "a" * 64},
        "aggregate": {"rows": 1},
        "scope_failures": [],
        "rows": [row],
        "review": {
            "top1_mismatch_rows": mismatch,
            "task_decision_mismatches": decision,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_aggregate_preserves_failed_category_and_task_review(tmp_path: Path) -> None:
    first = _capture(tmp_path / "code.json", prompt="code", category="code", match=True)
    second = _capture(
        tmp_path / "general.json",
        prompt="general",
        category="general_en",
        match=False,
    )

    result = aggregate([first, second])

    assert result["status"] == "manual_review_required"
    assert result["coverage"]["prompts"] == 2
    assert result["coverage"]["categories"] == ["code", "general_en"]
    assert result["aggregate"]["top1_matches"] == 1
    assert result["aggregate"]["top1_mismatches"] == 1
    assert result["checks"]["top1"] is False
    assert result["checks"]["task_decision_proxy"] is False
    assert result["scopes"]["category"]["code"]["passed"] is True
    assert result["scopes"]["category"]["general_en"]["passed"] is False
    assert result["review"]["top1_mismatch_rows"][0]["prompt"] == "general"


def test_aggregate_rejects_mixed_candidate_manifests(tmp_path: Path) -> None:
    first = _capture(tmp_path / "first.json", prompt="first", category="code", match=True)
    second = _capture(tmp_path / "second.json", prompt="second", category="general_en", match=True)
    payload = json.loads(second.read_text(encoding="utf-8"))
    payload["manifests"]["candidate_review_sha256"] = "b" * 64
    second.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate review manifest"):
        aggregate([first, second])
