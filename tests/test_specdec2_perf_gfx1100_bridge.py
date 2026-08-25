from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.specdec2_perf_gfx1100_bridge import (
    CANONICAL_PROMPTS,
    REQUIRED_TOP_LEVEL_STAGES,
    aggregate_bridge_rows,
    atomic_write_json,
    attach_paro_direct_rows,
    build_execution_plan,
    validate_bridge_rows,
)
from scripts.specdec2_perf_gfx1100_child import (
    build_bridge_row,
    resolve_arm_timing,
    validate_child_scope,
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


def test_child_scope_is_paro_k1_only_and_dense_uses_shared_bridge() -> None:
    validate_child_scope(lane="paro", profile="production", candidate_budget=1)
    validate_child_scope(lane="paro", profile="strict", candidate_budget=1)
    with pytest.raises(ValueError, match="PARO staged bridge is K1-only"):
        validate_child_scope(lane="paro", profile="production", candidate_budget=2)
    with pytest.raises(ValueError, match="dense uses the shared bridge"):
        validate_child_scope(lane="gguf", profile="strict", candidate_budget=1)


def test_paro_direct_attachment_uses_raw_ids_manifests_and_activation_timing(
    tmp_path: Path,
) -> None:
    timing = resolve_arm_timing(
        complete_request_seconds=1.0,
        output_timing={},
        scheduler_observability={"prefill_seconds": 0.25, "decode_seconds": 0.5},
    )
    common = {
        "lane": "paro",
        "profile": "production",
        "prompt_id": "code_merge_intervals",
        "run_index": 0,
        "candidate_budget": 1,
        "max_tokens": 24,
        "generated_token_ids": (1, 2, 3),
        "timing": timing,
        "selected_manifest_sha256": _MANIFEST,
        "strict_manifest_sha256": _STRICT_MANIFEST,
        "commit": _COMMIT,
    }
    loaded = {
        "prompt_ids": ["code_merge_intervals"],
        "runs": 1,
        "max_tokens": 24,
        "rows": [
            build_bridge_row(
                **common,
                arm="true_ar",
                order_index=0,
                physical_target_rows=(1,),
                physical_proposal_widths=(),
                route_name="true_ar",
            ),
            build_bridge_row(
                **common,
                arm="staged",
                order_index=2,
                physical_target_rows=(2,),
                physical_proposal_widths=(1,),
                route_name="eager",
            ),
        ],
    }
    raw = tmp_path / "smoke.json"
    raw.write_text(
        json.dumps(
            {
                "status": "passed",
                "exact_ar_match": True,
                "decode_tokens": 24,
                "candidate_budget": 1,
                "ar_tokens": [1, 2, 3],
                "mtp_tokens": [1, 2, 3],
                "execution_profile": "production",
                "execution_profile_manifest_sha256": _MANIFEST,
                "execution_profile_strict_manifest_sha256": _STRICT_MANIFEST,
                "mtp": {
                    "target_prefill_seconds": 0.2,
                    "proposal_prefill_seconds": 0.1,
                    "decode_seconds": 0.3,
                    "verify_seconds": 0.25,
                    "proposal_decode_update_seconds": 0.04,
                },
            }
        ),
        encoding="utf-8",
    )
    economics_child = tmp_path / "economics.json"
    economics_child.write_text(
        json.dumps(
            {"by_budget": {"1": {"runs": [{"run_idx": 1, "smoke_json": str(raw)}]}}}
        ),
        encoding="utf-8",
    )
    economics = {
        "execution_profile": "production",
        "decode_tokens": 24,
        "runs_per_prompt": 1,
        "repo": {
            "hipengine_commit": _COMMIT,
            "staged_dirty": False,
            "unstaged_dirty": False,
            "untracked_dirty": False,
        },
        "results": [
            {
                "name": "code_merge_intervals",
                "tokenization_seconds": 0.05,
                "economics_json": str(economics_child),
            }
        ],
    }

    attached = attach_paro_direct_rows(
        loaded,
        economics,
        profile="production",
        require_full_suite=False,
    )

    assert len(attached["rows"]) == 3
    direct = next(row for row in attached["rows"] if row["arm"] == "direct")
    assert direct["generated_token_ids"] == [1, 2, 3]
    assert direct["timing"]["complete_request_seconds"] == pytest.approx(0.65)
    assert direct["timing"]["top_level_stage_seconds"]["tokenize"] == 0.05
    assert direct["timing"]["top_level_stage_seconds"]["target_prefill"] == 0.2
    assert direct["timing"]["top_level_stage_seconds"]["provider_prompt_prime"] == 0.1
    assert direct["timing"]["top_level_stage_seconds"]["cycle_total"] == 0.3
    assert direct["reload_boundary"]["unavoidable_reload"] is True
    assert attached["aggregate"]["cells"]["paro:production:c1:k1"][
        "staged_speedup_vs_direct"
    ] == pytest.approx(0.65)


def test_paro_direct_attachment_rejects_source_commit_mismatch(tmp_path: Path) -> None:
    loaded = {
        "prompt_ids": ["code_merge_intervals"],
        "runs": 1,
        "max_tokens": 24,
        "rows": [],
    }
    economics = {
        "execution_profile": "production",
        "decode_tokens": 24,
        "runs_per_prompt": 1,
        "repo": {
            "hipengine_commit": "d" * 40,
            "staged_dirty": False,
            "unstaged_dirty": False,
            "untracked_dirty": False,
        },
        "results": [],
    }
    # Source equality fails before incomplete-row validation.
    loaded["provenance"] = {"commit": _COMMIT}
    with pytest.raises(ValueError, match="source commit"):
        attach_paro_direct_rows(
            loaded,
            economics,
            profile="production",
            require_full_suite=False,
        )


def test_child_timing_uses_nonoverlapping_scheduler_windows_and_residual() -> None:
    timing = resolve_arm_timing(
        complete_request_seconds=1.0,
        output_timing={
            "specdec2_mtp2_prompt_prime_ms": 100.0,
            "specdec2_mtp2_proposal_ms": 50.0,
            "specdec2_mtp2_target_ms": 400.0,
            "specdec2_mtp2_provider_update_ms": 10.0,
        },
        scheduler_observability={
            "prefill_seconds": 0.2,
            "decode_seconds": 0.6,
            "queue_seconds": 0.01,
        },
    )

    assert timing["decode_only_seconds"] == 0.6
    assert timing["top_level_stage_seconds"]["target_prefill"] == 0.1
    assert timing["top_level_stage_seconds"]["provider_prompt_prime"] == 0.1
    assert timing["top_level_stage_seconds"]["cycle_total"] == 0.6
    assert timing["top_level_stage_seconds"]["resident_owner_transition"] == 0.0
    assert timing["unattributed_seconds"] == pytest.approx(0.2)
    assert timing["cycle_detail_seconds"] == {
        "proposal": 0.05,
        "target_verify": 0.4,
        "provider_update": 0.01,
    }


def test_child_row_reports_realized_route_manifests_and_physical_shape() -> None:
    timing = resolve_arm_timing(
        complete_request_seconds=1.0,
        output_timing={},
        scheduler_observability={"prefill_seconds": 0.25, "decode_seconds": 0.5},
    )
    row = build_bridge_row(
        lane="gguf",
        arm="staged",
        profile="strict",
        prompt_id="code_merge_intervals",
        run_index=0,
        order_index=2,
        candidate_budget=3,
        max_tokens=25,
        generated_token_ids=(1, 2, 3),
        timing=timing,
        selected_manifest_sha256=_MANIFEST,
        strict_manifest_sha256=_STRICT_MANIFEST,
        commit=_COMMIT,
        physical_target_rows=(4,),
        physical_proposal_widths=(1,),
        route_name="graph",
    )

    assert row["route"] == {
        "realized": "graph",
        "true_autoregressive_path": False,
        "staged_generation2": True,
        "direct_control": False,
        "physical_proposal_widths": [1],
        "physical_target_rows": [4],
    }
    assert row["timing"]["timing_owner_id"].endswith(
        ":staged:c1:k3"
    )
    assert row["manifests"]["selected_sha256"] == _MANIFEST
