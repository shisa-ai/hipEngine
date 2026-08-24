from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.mtp_paro_verifier_repeat_review import aggregate_repeats
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
        "capture_sha256": "c" * 64,
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


def test_repeat_aggregate_uses_one_quality_trajectory_and_checks_hashes(
    tmp_path: Path,
) -> None:
    paths = []
    for prompt, category in (("code", "code"), ("general", "general_en")):
        base = _capture(
            tmp_path / f"{prompt}-base.json",
            prompt=prompt,
            category=category,
            match=True,
        )
        payload = base.read_text(encoding="utf-8")
        for repeat in range(3):
            path = tmp_path / f"run{repeat + 1}-{prompt}.json"
            path.write_text(payload, encoding="utf-8")
            paths.append(path)

    result = aggregate_repeats(paths, expected_repeats=3)

    assert result["coverage"]["rows"] == 2
    assert result["coverage"]["capture_files"] == 6
    assert result["repeat_determinism"]["passed"] is True
    assert result["review"]["numerical_and_repeat_gates_passed"] is True
    assert result["status"] == "numerical_repeat_pass_task_review_pending"


def test_repeat_aggregate_rejects_capture_hash_drift(tmp_path: Path) -> None:
    paths = []
    for repeat in range(3):
        path = _capture(
            tmp_path / f"run{repeat + 1}.json",
            prompt="code",
            category="code",
            match=True,
        )
        if repeat == 2:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["capture_sha256"] = "d" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    result = aggregate_repeats(paths, expected_repeats=3)

    assert result["repeat_determinism"]["passed"] is False
    assert result["repeat_determinism"]["failed_prompts"] == ["code"]
    assert result["status"] == "numerical_or_repeat_gate_failed"


def test_aggregate_rejects_mixed_candidate_manifests(tmp_path: Path) -> None:
    first = _capture(tmp_path / "first.json", prompt="first", category="code", match=True)
    second = _capture(tmp_path / "second.json", prompt="second", category="general_en", match=True)
    payload = json.loads(second.read_text(encoding="utf-8"))
    payload["manifests"]["candidate_review_sha256"] = "b" * 64
    second.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate review manifest"):
        aggregate([first, second])
