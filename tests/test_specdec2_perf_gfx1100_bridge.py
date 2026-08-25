from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.specdec2_perf_gfx1100_bridge import (
    CANONICAL_PROMPTS,
    REQUIRED_TOP_LEVEL_STAGES,
    aggregate_bridge_rows,
    atomic_write_json,
    build_execution_plan,
    validate_bridge_rows,
)


_MANIFEST = "a" * 64
_STRICT_MANIFEST = "b" * 64
_COMMIT = "c" * 40


def _rows(*, runs: int = 3) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_index in range(runs):
        for prompt_index, prompt in enumerate(CANONICAL_PROMPTS):
            order = build_execution_plan(
                lane="gguf",
                profiles=("strict",),
                candidate_budgets=(3,),
                runs=runs,
                prompt_ids=tuple(item[0] for item in CANONICAL_PROMPTS),
            )[run_index * len(CANONICAL_PROMPTS) + prompt_index]["arm_order"]
            for order_index, arm in enumerate(order):
                complete = {
                    "true_ar": 2.0,
                    "direct": 1.0,
                    "staged": 1.1,
                }[arm]
                decode = complete * 0.6
                known = complete * 0.9
                stages = {
                    stage: 0.0 for stage in REQUIRED_TOP_LEVEL_STAGES
                }
                stages["target_prefill"] = complete * 0.2
                stages["cycle_total"] = decode
                stages["terminal_reclaim"] = complete * 0.1
                row = {
                    "schema": 1,
                    "lane": "gguf",
                    "arm": arm,
                    "profile": "strict",
                    "prompt_id": prompt[0],
                    "category": prompt[1],
                    "split": prompt[2],
                    "run_index": run_index,
                    "order_index": order_index,
                    "concurrency": 1,
                    "candidate_budget": 3,
                    "realized_candidate_budget": 0 if arm == "true_ar" else 3,
                    "max_tokens": 25,
                    "generated_token_ids": [11, 12, 13],
                    "timing": {
                        "complete_request_seconds": complete,
                        "decode_only_seconds": decode,
                        "top_level_stage_seconds": stages,
                        "unattributed_seconds": complete - known,
                        "timing_owner_id": (
                            f"gguf:strict:{run_index}:{prompt[0]}:{arm}:c1:k3"
                        ),
                        "timing_owner": True,
                        "timing_scope": "request",
                    },
                    "route": {
                        "realized": arm,
                        "true_autoregressive_path": arm == "true_ar",
                        "staged_generation2": arm == "staged",
                        "direct_control": arm == "direct",
                        "physical_proposal_widths": [1],
                        "physical_target_rows": [4] if arm != "true_ar" else [1],
                    },
                    "manifests": {
                        "selected_sha256": _MANIFEST,
                        "strict_sha256": _STRICT_MANIFEST,
                    },
                    "provenance": {
                        "commit": _COMMIT,
                        "staged_dirty": False,
                        "unstaged_dirty": False,
                        "untracked_dirty": False,
                    },
                }
                rows.append(row)
    return rows


def test_execution_plan_counterbalances_ar_and_staged_without_prompt_content() -> None:
    prompt_ids = tuple(item[0] for item in CANONICAL_PROMPTS[:2])
    plan = build_execution_plan(
        lane="paro",
        profiles=("production",),
        candidate_budgets=(1,),
        runs=2,
        prompt_ids=prompt_ids,
    )

    assert [row["arm_order"] for row in plan] == [
        ("true_ar", "direct", "staged"),
        ("staged", "direct", "true_ar"),
        ("staged", "direct", "true_ar"),
        ("true_ar", "direct", "staged"),
    ]
    assert [row["prompt_id"] for row in plan] == [*prompt_ids, *prompt_ids]


def test_bridge_contract_accepts_complete_clean_strict_packet() -> None:
    rows = _rows()

    validated = validate_bridge_rows(
        rows,
        lane="gguf",
        profiles=("strict",),
        candidate_budgets=(3,),
        runs=3,
        max_tokens=25,
        require_full_suite=True,
        strict_generated_ids=True,
    )
    aggregate = aggregate_bridge_rows(validated)

    assert len(validated) == 90
    cell = aggregate["cells"]["gguf:strict:c1:k3"]
    assert cell["arms"]["true_ar"]["complete_request_seconds"] == 60.0
    assert cell["arms"]["direct"]["complete_request_seconds"] == 30.0
    assert cell["arms"]["staged"]["complete_request_seconds"] == 33.0
    assert cell["complete_speedup_vs_true_ar"]["direct"] == 2.0
    assert cell["complete_speedup_vs_true_ar"]["staged"] == pytest.approx(
        60.0 / 33.0
    )
    assert cell["staged_speedup_vs_direct"] == pytest.approx(30.0 / 33.0)


def test_bridge_contract_rejects_missing_or_duplicated_timing_owner() -> None:
    missing = _rows()
    missing[0]["timing"]["timing_owner"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="exactly one timing owner"):
        validate_bridge_rows(missing, lane="gguf", profiles=("strict",), candidate_budgets=(3,), runs=3, max_tokens=25)

    duplicated = _rows()
    duplicated[1]["timing"]["timing_owner_id"] = duplicated[0]["timing"]["timing_owner_id"]  # type: ignore[index]
    with pytest.raises(ValueError, match="duplicate timing owner"):
        validate_bridge_rows(duplicated, lane="gguf", profiles=("strict",), candidate_budgets=(3,), runs=3, max_tokens=25)


def test_bridge_contract_rejects_incomplete_suite_and_invalid_ar_denominator() -> None:
    incomplete = [
        row for row in _rows() if row["prompt_id"] != "mixed_ja_en_review"
    ]
    with pytest.raises(ValueError, match="canonical prompt suite"):
        validate_bridge_rows(incomplete, lane="gguf", profiles=("strict",), candidate_budgets=(3,), runs=3, max_tokens=25)

    invalid_ar = _rows()
    ar = next(row for row in invalid_ar if row["arm"] == "true_ar")
    ar["route"]["true_autoregressive_path"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="true AR denominator"):
        validate_bridge_rows(invalid_ar, lane="gguf", profiles=("strict",), candidate_budgets=(3,), runs=3, max_tokens=25)


def test_bridge_contract_rejects_dirty_provenance_and_bad_manifest() -> None:
    dirty = _rows()
    dirty[0]["provenance"]["unstaged_dirty"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="dirty provenance"):
        validate_bridge_rows(dirty, lane="gguf", profiles=("strict",), candidate_budgets=(3,), runs=3, max_tokens=25)

    malformed = _rows()
    malformed[0]["manifests"]["selected_sha256"] = "bad"  # type: ignore[index]
    with pytest.raises(ValueError, match="manifest"):
        validate_bridge_rows(malformed, lane="gguf", profiles=("strict",), candidate_budgets=(3,), runs=3, max_tokens=25)


def test_bridge_contract_rejects_missing_stage_and_stage_overflow() -> None:
    missing = _rows()
    del missing[0]["timing"]["top_level_stage_seconds"]["resident_owner_transition"]  # type: ignore[index]
    with pytest.raises(ValueError, match="top-level timing stages"):
        validate_bridge_rows(missing, lane="gguf", profiles=("strict",), candidate_budgets=(3,), runs=3, max_tokens=25)

    overflow = _rows()
    overflow[0]["timing"]["unattributed_seconds"] = 2.0  # type: ignore[index]
    with pytest.raises(ValueError, match="reconcile"):
        validate_bridge_rows(overflow, lane="gguf", profiles=("strict",), candidate_budgets=(3,), runs=3, max_tokens=25)


def test_atomic_checkpoint_replaces_complete_json_without_temp_leak(tmp_path: Path) -> None:
    output = tmp_path / "bridge.json"
    atomic_write_json(output, {"checkpoint": 1})
    atomic_write_json(output, {"checkpoint": 2, "rows": [1, 2]})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "checkpoint": 2,
        "rows": [1, 2],
    }
    assert list(tmp_path.iterdir()) == [output]
