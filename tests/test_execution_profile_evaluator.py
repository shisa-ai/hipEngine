from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from hipengine.benchmark.execution_profiles import (
    EXECUTION_PROFILE_CAPTURE_KIND,
    EXECUTION_PROFILE_CONTROL_CAPTURE_KIND,
    EXECUTION_PROFILE_CONTROL_FIXTURE_KIND,
    EXECUTION_PROFILE_EVALUATION_KIND,
    Bf16NoninferiorityThresholds,
    ControlRecord,
    EvaluationThresholds,
    RowDescriptor,
    RunCapture,
    build_execution_profile_artifact,
    compare_bf16_noninferiority,
    compare_control_records,
    compare_profile_logits,
    compare_repeat_captures,
    compare_request_results,
    load_control_capture,
    load_run_capture_manifest,
    qwen36_rows_from_teacher_fixture,
    summarize_scenario,
    validate_execution_profile_artifact,
)
from hipengine.execution_profiles import (
    VariantSelection,
    build_variant_manifest,
    manifest_sha256,
)


def _rows() -> tuple[RowDescriptor, ...]:
    return (
        RowDescriptor("scenario-a", 0, "request-a", 0, "code", "c2", "steady"),
        RowDescriptor("scenario-a", 0, "request-b", 0, "general_en", "c2", "steady"),
        RowDescriptor("scenario-a", 1, "request-a", 1, "code", "c1", "c2_to_c1"),
    )


def _controls() -> tuple[ControlRecord, ...]:
    return (
        ControlRecord(
            scenario_id="scenario-a",
            scenario_step=0,
            work_class="DECODE_STEP",
            request_id="request-a",
            physical_slot=2,
            execution_row=0,
            physical_width=2,
            input_token_id=10,
            position=8,
            context_length=9,
            active=True,
            active_mask_hash="mask-c2",
            mask_manifest_hash="all-masks-a-0",
            publication_ordinal=8,
            transaction_id="none",
            transaction_phase="NONE",
            accepted_token_count=1,
            route_decision_hash="route-a-0",
            route_scatter_owner_hash="scatter-a-0",
            route_owner_request_id="request-a",
            route_top_k=8,
            kv_base_offset=100,
            kv_live_count=9,
            kv_token_position=8,
            kv_evict=False,
            kv_values_finite=True,
            kv_append_ordinal=8,
            state_owner_request_id="request-a",
            state_update_ordinal=8,
            state_values_finite=True,
            rng_owner_request_id="request-a",
            rng_seed=101,
            rng_counter=3,
            route_values_finite=True,
            graph_bucket="decode-c2",
        ),
        ControlRecord(
            scenario_id="scenario-a",
            scenario_step=0,
            work_class="DECODE_STEP",
            request_id="request-b",
            physical_slot=5,
            execution_row=1,
            physical_width=2,
            input_token_id=20,
            position=3,
            context_length=4,
            active=True,
            active_mask_hash="mask-c2",
            mask_manifest_hash="all-masks-b-0",
            publication_ordinal=3,
            transaction_id="none",
            transaction_phase="NONE",
            accepted_token_count=1,
            route_decision_hash="route-b-0",
            route_scatter_owner_hash="scatter-b-0",
            route_owner_request_id="request-b",
            route_top_k=8,
            kv_base_offset=200,
            kv_live_count=4,
            kv_token_position=3,
            kv_evict=False,
            kv_values_finite=True,
            kv_append_ordinal=3,
            state_owner_request_id="request-b",
            state_update_ordinal=3,
            state_values_finite=True,
            rng_owner_request_id="request-b",
            rng_seed=202,
            rng_counter=7,
            route_values_finite=True,
            graph_bucket="decode-c2",
        ),
        ControlRecord(
            scenario_id="scenario-a",
            scenario_step=1,
            work_class="DECODE_STEP",
            request_id="request-a",
            physical_slot=0,
            execution_row=0,
            physical_width=1,
            input_token_id=11,
            position=9,
            context_length=10,
            active=True,
            active_mask_hash="mask-c1",
            mask_manifest_hash="all-masks-a-1",
            publication_ordinal=9,
            transaction_id="none",
            transaction_phase="NONE",
            accepted_token_count=1,
            route_decision_hash="route-a-1",
            route_scatter_owner_hash="scatter-a-1",
            route_owner_request_id="request-a",
            route_top_k=8,
            kv_base_offset=100,
            kv_live_count=10,
            kv_token_position=9,
            kv_evict=False,
            kv_values_finite=True,
            kv_append_ordinal=9,
            state_owner_request_id="request-a",
            state_update_ordinal=9,
            state_values_finite=True,
            rng_owner_request_id="request-a",
            rng_seed=101,
            rng_counter=4,
            route_values_finite=True,
            graph_bucket="decode-c1",
        ),
    )


def _logits() -> np.ndarray:
    return np.asarray(
        [
            [4.0, 2.0, 0.0, -1.0],
            [0.0, 4.0, 1.0, -1.0],
            [3.0, 2.0, 0.0, -2.0],
        ],
        dtype=np.float32,
    )


def _capture(
    *,
    profile: str,
    logits: np.ndarray | None = None,
    controls: tuple[ControlRecord, ...] | None = None,
    selected: tuple[int, ...] = (0, 1, 0),
    repeat_index: int = 0,
) -> RunCapture:
    return RunCapture(
        execution_profile=profile,
        scenario_id="scenario-a",
        run_id=f"{profile}-run-{repeat_index}",
        variant_manifest_sha256=manifest_sha256(_manifest(profile)),
        repeat_index=repeat_index,
        rows=_rows(),
        logits=_logits() if logits is None else logits,
        selected_token_ids=selected,
        controls=_controls() if controls is None else controls,
    )


def _alternate_capture(
    *,
    profile: str,
    scenario_id: str,
    run_id: str,
) -> tuple[RunCapture, tuple[ControlRecord, ...]]:
    rows = tuple(replace(row, scenario_id=scenario_id) for row in _rows())
    controls = tuple(replace(control, scenario_id=scenario_id) for control in _controls())
    return (
        RunCapture(
            execution_profile=profile,
            scenario_id=scenario_id,
            run_id=run_id,
            variant_manifest_sha256=manifest_sha256(_manifest(profile)),
            repeat_index=0,
            rows=rows,
            logits=_logits(),
            selected_token_ids=(0, 1, 0),
            controls=controls,
        ),
        controls,
    )


def _manifest(profile: str = "production") -> dict[str, object]:
    return build_variant_manifest(
        profile=profile,
        backend="hip_gfx1100",
        model="qwen3_5_moe",
        quant="w4_paro",
        kv_policy="paged_bf16",
        graph_policy="decode-c1-c2",
        selections=(
            VariantSelection(
                layer="paged_attn_decode",
                scope="c1-c2",
                selected_variant="online" if profile == "production" else "exact",
                strict_fallback_variant="exact",
                evidence_artifact="benchmarks/results/example.json",
            ),
        ),
    )


def test_default_thresholds_match_retained_production_policy() -> None:
    assert EvaluationThresholds().to_dict() == {
        "mean_kl_max": 1e-3,
        "p95_kl_max": 5e-3,
        "p99_kl_max": 2e-2,
        "max_kl_max": 5e-2,
        "top1_min": 0.99,
        "per_scope_top1_min": 0.97,
        "review_kl": 2e-2,
    }
    bf16_thresholds = Bf16NoninferiorityThresholds()
    assert bf16_thresholds.mean_kl_delta_max == 1e-3
    assert bf16_thresholds.top1_drop_max == 1e-2


def test_profile_logit_summary_reports_tails_scopes_and_review_rows() -> None:
    strict = _logits()
    candidate = strict.copy()
    candidate[1] = np.asarray([0.0, 3.5, 1.5, -1.0], dtype=np.float32)
    thresholds = EvaluationThresholds(
        mean_kl_max=1.0,
        p95_kl_max=1.0,
        p99_kl_max=1.0,
        max_kl_max=1.0,
        top1_min=0.0,
        per_scope_top1_min=0.0,
        review_kl=0.0,
    )

    result = compare_profile_logits(strict, candidate, _rows(), thresholds=thresholds)

    assert result["summary"]["rows"] == 3
    assert result["summary"]["kl_p95"] > 0
    assert result["summary"]["kl_p99"] >= result["summary"]["kl_p95"]
    assert result["summary"]["kl_max"] >= result["summary"]["kl_p99"]
    assert set(result["by_scope"]) == {"category", "shape", "transition"}
    assert result["by_scope"]["category"]["general_en"]["rows"] == 1
    assert result["hard_gates_passed"] is True
    assert result["requires_outlier_review"] is True
    assert result["eligible_for_automatic_admission"] is False
    assert result["rows_over_review_boundary"][0]["request_id"] == "request-b"
    assert result["top1_mismatch_rows"] == []


def test_profile_logit_summary_attributes_top1_mismatch_below_review_boundary() -> None:
    strict = _logits()
    candidate = strict.copy()
    candidate[2] = np.asarray([2.9, 3.1, 0.0, -2.0], dtype=np.float32)
    result = compare_profile_logits(
        strict,
        candidate,
        _rows(),
        thresholds=EvaluationThresholds(
            mean_kl_max=1.0,
            p95_kl_max=1.0,
            p99_kl_max=1.0,
            max_kl_max=1.0,
            top1_min=0.0,
            per_scope_top1_min=0.0,
            review_kl=1.0,
        ),
    )

    assert result["rows_over_review_boundary"] == []
    assert len(result["top1_mismatch_rows"]) == 1
    mismatch = result["top1_mismatch_rows"][0]
    assert mismatch["request_id"] == "request-a"
    assert mismatch["teacher_step"] == 1
    assert mismatch["row_index"] == 2
    assert mismatch["top1_equal"] is False
    assert mismatch["strict_top1_token_id"] == 0
    assert mismatch["candidate_top1_token_id"] == 1
    assert mismatch["strict_top1_candidate_rank"] == 2
    assert mismatch["strict_margin"] == pytest.approx(1.0)
    assert mismatch["max_abs_logit_delta"] == pytest.approx(1.1)


def test_profile_logit_summary_turns_nonfinite_candidate_into_failed_gate() -> None:
    candidate = _logits()
    candidate[0, 0] = np.nan

    result = compare_profile_logits(_logits(), candidate, _rows())

    assert result["finite"] is False
    assert result["hard_gates_passed"] is False
    assert result["eligible_for_automatic_admission"] is False


def test_control_comparator_localizes_width_transition_state_owner_bug() -> None:
    expected = _controls()
    actual = list(expected)
    actual[2] = replace(actual[2], state_owner_request_id="request-b")

    result = compare_control_records(expected, tuple(actual))

    assert result["passed"] is False
    assert result["mismatches"] == [
        {
            "key": ["scenario-a", 1, "DECODE_STEP", "request-a"],
            "field": "state_owner_request_id",
            "expected": "request-a",
            "actual": "request-b",
        }
    ]

    route_changed = list(expected)
    route_changed[2] = replace(route_changed[2], route_decision_hash="near-tie-route")
    route_result = compare_control_records(
        expected,
        tuple(route_changed),
        diagnostic_fields=("route_decision_hash",),
    )
    assert route_result["passed"] is True
    assert route_result["mismatches"] == []
    assert route_result["diagnostic_mismatches"][0]["field"] == "route_decision_hash"

    summary = summarize_scenario(expected)
    assert summary["width_sequence"] == [2, 1]
    assert summary["width_transition_count"] == 1
    assert summary["compaction_count"] == 1
    assert summary["ragged_steps"] == [0]


def test_scenario_summary_covers_delayed_admission_cancellation_and_reclaim() -> None:
    controls = list(_controls())
    controls.extend(
        [
            replace(
                controls[2],
                scenario_step=2,
                physical_width=2,
                position=10,
                context_length=11,
                kv_live_count=11,
                kv_token_position=10,
                kv_append_ordinal=10,
                state_update_ordinal=10,
                rng_counter=5,
            ),
            replace(
                controls[1],
                scenario_step=2,
                work_class="ADMIT",
                request_id="request-c",
                physical_slot=4,
                execution_row=1,
                physical_width=2,
                input_token_id=30,
                position=0,
                context_length=1,
                active_mask_hash="mask-admit-c2",
                route_decision_hash="route-c-0",
                route_scatter_owner_hash="scatter-c-0",
                route_owner_request_id="request-c",
                kv_base_offset=300,
                kv_live_count=1,
                kv_token_position=0,
                kv_append_ordinal=0,
                state_owner_request_id="request-c",
                state_update_ordinal=0,
                rng_owner_request_id="request-c",
                rng_seed=303,
                rng_counter=0,
            ),
            replace(
                controls[2],
                scenario_step=3,
                work_class="CANCEL",
                active=False,
                position=10,
                context_length=11,
                kv_live_count=11,
                kv_token_position=10,
                kv_append_ordinal=10,
                state_update_ordinal=10,
                rng_counter=5,
            ),
            replace(
                controls[1],
                scenario_step=3,
                request_id="request-c",
                physical_slot=0,
                execution_row=0,
                physical_width=1,
                input_token_id=31,
                position=1,
                context_length=2,
                active_mask_hash="mask-c1",
                route_decision_hash="route-c-1",
                route_scatter_owner_hash="scatter-c-1",
                route_owner_request_id="request-c",
                kv_base_offset=300,
                kv_live_count=2,
                kv_token_position=1,
                kv_append_ordinal=1,
                state_owner_request_id="request-c",
                state_update_ordinal=1,
                rng_owner_request_id="request-c",
                rng_seed=303,
                rng_counter=1,
                graph_bucket="decode-c1",
            ),
        ]
    )

    summary = summarize_scenario(tuple(controls))

    assert summary["width_sequence"] == [2, 1, 2, 1]
    assert summary["delayed_admission_steps"] == [2]
    assert summary["cancellation_steps"] == [3]
    assert summary["sparse_retirement_steps"] == [1, 3]
    assert summary["request_count"] == 3


def test_repeat_and_request_invariance_checks_are_logical_request_aligned() -> None:
    baseline = _capture(profile="production")
    repeat = _capture(profile="production", repeat_index=1)
    changed = _logits()
    changed[2, 0] += np.float32(0.25)
    bad_repeat = _capture(profile="production", logits=changed, repeat_index=2)

    deterministic = compare_repeat_captures(baseline, (repeat, bad_repeat))
    assert deterministic["passed"] is False
    assert deterministic["repeats"][0]["logits_exact"] is True
    assert deterministic["repeats"][1]["logits_exact"] is False

    # A different scenario/slot layout is aligned by request + teacher step.
    other_rows = tuple(replace(row, scenario_id="scenario-neighbors") for row in _rows())
    other_controls = tuple(
        replace(control, scenario_id="scenario-neighbors", physical_slot=control.physical_slot + 3)
        for control in _controls()
    )
    other = RunCapture(
        execution_profile="batch_invariant",
        scenario_id="scenario-neighbors",
        run_id="batch-invariant-neighbor-run",
        variant_manifest_sha256=manifest_sha256(_manifest("batch_invariant")),
        repeat_index=0,
        rows=other_rows,
        logits=_logits(),
        selected_token_ids=(0, 1, 0),
        controls=other_controls,
    )
    invariant = compare_request_results(
        _capture(profile="batch_invariant"),
        (other,),
        request_ids=("request-a",),
    )
    assert invariant["passed"] is True
    assert invariant["comparisons"][0]["rows_compared"] == 2

    self_comparison = compare_request_results(
        baseline,
        (baseline,),
        request_ids=("request-a",),
    )
    assert self_comparison["passed"] is False
    assert self_comparison["comparisons"][0]["independent_run"] is False
    assert self_comparison["comparisons"][0]["distinct_scenario"] is False


def test_execution_profile_artifact_requires_all_production_gates() -> None:
    strict_expected = tuple(
        replace(control, graph_bucket=f"strict-{control.graph_bucket}")
        for control in _controls()
    )
    strict = _capture(profile="strict", controls=strict_expected)
    candidate = _capture(profile="production")
    repeat = _capture(profile="production", repeat_index=1)
    isolation, isolation_controls = _alternate_capture(
        profile="production",
        scenario_id="scenario-neighbor-substitution",
        run_id="production-isolation-run",
    )

    artifact = build_execution_profile_artifact(
        variant_manifest=_manifest("production"),
        strict_manifest=_manifest("strict"),
        arithmetic_class="T2",
        strict_capture=strict,
        candidate_capture=candidate,
        expected_controls=_controls(),
        strict_expected_controls=strict_expected,
        comparison_expected_controls={
            "scenario-neighbor-substitution": isolation_controls
        },
        repeat_captures=(repeat,),
        isolation_captures=(isolation,),
        batch_invariant_captures=(),
        task_results={"category_suite": {"passed": True, "evidence": "synthetic"}},
    )

    assert artifact["kind"] == EXECUTION_PROFILE_EVALUATION_KIND
    assert artifact["execution_profile"] == "production"
    assert artifact["generated_id_equality"]["binding"] is False
    assert artifact["control_semantics"]["passed"] is True
    assert artifact["quality"]["eligible_for_automatic_admission"] is True
    assert artifact["decision"] == {
        "status": "passed",
        "eligible_for_automatic_admission": True,
        "binding_gates_passed": True,
    }
    assert validate_execution_profile_artifact(artifact) == artifact
    forged = json.loads(json.dumps(artifact))
    forged["quality"]["hard_gates_passed"] = False
    with pytest.raises(ValueError, match="inconsistent with gate sections"):
        validate_execution_profile_artifact(forged)

    no_tasks = build_execution_profile_artifact(
        variant_manifest=_manifest("production"),
        strict_manifest=_manifest("strict"),
        arithmetic_class="T2",
        strict_capture=strict,
        candidate_capture=candidate,
        expected_controls=_controls(),
        strict_expected_controls=strict_expected,
        comparison_expected_controls={
            "scenario-neighbor-substitution": isolation_controls
        },
        repeat_captures=(repeat,),
        isolation_captures=(isolation,),
        task_results={},
    )
    assert no_tasks["task_quality"]["status"] == "missing"
    assert no_tasks["decision"]["status"] == "failed"
    assert no_tasks["decision"]["eligible_for_automatic_admission"] is False


def test_batch_invariant_artifact_requires_controls_for_comparison_scenarios() -> None:
    other_rows = tuple(replace(row, scenario_id="scenario-neighbors") for row in _rows())
    other_controls = tuple(
        replace(control, scenario_id="scenario-neighbors", physical_slot=control.physical_slot + 3)
        for control in _controls()
    )
    other = RunCapture(
        execution_profile="batch_invariant",
        scenario_id="scenario-neighbors",
        run_id="batch-invariant-neighbor-artifact-run",
        variant_manifest_sha256=manifest_sha256(_manifest("batch_invariant")),
        repeat_index=0,
        rows=other_rows,
        logits=_logits(),
        selected_token_ids=(0, 1, 0),
        controls=other_controls,
    )
    candidate = _capture(profile="batch_invariant")
    repeat = _capture(profile="batch_invariant", repeat_index=1)
    common = {
        "variant_manifest": _manifest("batch_invariant"),
        "strict_manifest": _manifest("strict"),
        "arithmetic_class": "T0",
        "strict_capture": _capture(profile="strict"),
        "candidate_capture": candidate,
        "expected_controls": _controls(),
        "strict_expected_controls": _controls(),
        "repeat_captures": (repeat,),
        "isolation_captures": (other,),
        "batch_invariant_captures": (other,),
        "task_results": {"category_suite": True},
    }

    missing = build_execution_profile_artifact(**common)
    assert missing["batch_invariance"]["control_semantics"]["passed"] is False
    assert missing["decision"]["status"] == "failed"

    complete = build_execution_profile_artifact(
        **common,
        comparison_expected_controls={"scenario-neighbors": other_controls},
    )
    assert complete["batch_invariance"]["control_semantics"]["passed"] is True
    assert complete["decision"]["status"] == "passed"


def test_bf16_noninferiority_is_reported_by_category() -> None:
    strict = _logits()
    candidate = strict.copy()
    candidate[1] = np.asarray([0.0, 3.5, 1.5, -1.0], dtype=np.float32)

    result = compare_bf16_noninferiority(
        strict,
        strict,
        candidate,
        _rows(),
        thresholds=Bf16NoninferiorityThresholds(
            mean_kl_delta_max=0.0,
            top1_drop_max=0.0,
        ),
    )

    assert result["passed"] is False
    assert result["mean_kl_delta"] > 0.0
    assert set(result["by_category"]) == {"code", "general_en"}
    assert result["by_category"]["general_en"]["passed"] is False


def test_capture_manifest_loads_external_logits_and_checks_hash(tmp_path) -> None:
    logits_path = tmp_path / "strict.npy"
    np.save(logits_path, _logits())
    logits_sha = hashlib.sha256(logits_path.read_bytes()).hexdigest()
    payload = {
        "kind": EXECUTION_PROFILE_CAPTURE_KIND,
        "schema_version": 1,
        "execution_profile": "strict",
        "scenario_id": "scenario-a",
        "run_id": "strict-run-0",
        "variant_manifest_sha256": manifest_sha256(_manifest("strict")),
        "repeat_index": 0,
        "logits_path": logits_path.name,
        "logits_sha256": logits_sha,
        "rows": [row.to_dict() for row in _rows()],
        "selected_token_ids": [0, 1, 0],
        "controls": [control.to_dict() for control in _controls()],
    }
    manifest_path = tmp_path / "capture.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    capture = load_run_capture_manifest(manifest_path)
    assert isinstance(capture.logits, np.memmap)
    assert capture.execution_profile == "strict"
    assert capture.sha256() == _capture(profile="strict").sha256()

    controls_path = tmp_path / "actual-controls.json"
    controls_path.write_text(
        json.dumps(
            {
                "kind": EXECUTION_PROFILE_CONTROL_CAPTURE_KIND,
                "schema_version": 1,
                "scenario_id": "scenario-a",
                "run_id": "strict-run-0",
                "controls": [control.to_dict() for control in _controls()],
            }
        ),
        encoding="utf-8",
    )
    run_id, controls = load_control_capture(controls_path)
    assert run_id == capture.run_id
    assert controls == _controls()

    payload["logits_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="logits_sha256 mismatch"):
        load_run_capture_manifest(manifest_path)


def test_execution_profile_gate_cli_writes_valid_artifact(tmp_path) -> None:
    logits_path = tmp_path / "logits.npy"
    np.save(logits_path, _logits())
    logits_sha = hashlib.sha256(logits_path.read_bytes()).hexdigest()

    def write_capture(
        name: str,
        profile: str,
        repeat_index: int,
        *,
        scenario_id: str = "scenario-a",
    ) -> str:
        path = tmp_path / f"{name}.json"
        rows = [replace(row, scenario_id=scenario_id).to_dict() for row in _rows()]
        controls = [
            replace(control, scenario_id=scenario_id).to_dict()
            for control in _controls()
        ]
        path.write_text(
            json.dumps(
                {
                    "kind": EXECUTION_PROFILE_CAPTURE_KIND,
                    "schema_version": 1,
                    "execution_profile": profile,
                    "scenario_id": scenario_id,
                    "run_id": f"{profile}-{name}-cli-run-{repeat_index}",
                    "variant_manifest_sha256": manifest_sha256(_manifest(profile)),
                    "repeat_index": repeat_index,
                    "logits_path": logits_path.name,
                    "logits_sha256": logits_sha,
                    "rows": rows,
                    "selected_token_ids": [0, 1, 0],
                    "controls": controls,
                }
            ),
            encoding="utf-8",
        )
        return str(path)

    strict_capture = write_capture("strict", "strict", 0)
    candidate_capture = write_capture("candidate", "production", 0)
    repeat_capture = write_capture("repeat", "production", 1)
    isolation_capture = write_capture(
        "isolation",
        "production",
        0,
        scenario_id="scenario-neighbor-substitution",
    )
    strict_manifest = tmp_path / "strict-manifest.json"
    strict_manifest.write_text(json.dumps(_manifest("strict")), encoding="utf-8")
    candidate_manifest = tmp_path / "candidate-manifest.json"
    candidate_manifest.write_text(json.dumps(_manifest("production")), encoding="utf-8")
    controls_path = tmp_path / "controls.json"
    controls_path.write_text(
        json.dumps(
            {
                "kind": EXECUTION_PROFILE_CONTROL_FIXTURE_KIND,
                "schema_version": 1,
                "scenario_id": "scenario-a",
                "controls": [control.to_dict() for control in _controls()],
            }
        ),
        encoding="utf-8",
    )
    comparison_controls_path = tmp_path / "comparison-controls.json"
    comparison_controls_path.write_text(
        json.dumps(
            {
                "kind": EXECUTION_PROFILE_CONTROL_FIXTURE_KIND,
                "schema_version": 1,
                "scenario_id": "scenario-neighbor-substitution",
                "controls": [
                    replace(
                        control, scenario_id="scenario-neighbor-substitution"
                    ).to_dict()
                    for control in _controls()
                ],
            }
        ),
        encoding="utf-8",
    )
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps({"category_suite": True}), encoding="utf-8")
    output = tmp_path / "evaluation.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/execution_profile_gate.py",
            "--variant-manifest",
            str(candidate_manifest),
            "--strict-manifest",
            str(strict_manifest),
            "--strict-capture",
            strict_capture,
            "--candidate-capture",
            candidate_capture,
            "--expected-controls",
            str(controls_path),
            "--strict-expected-controls",
            str(controls_path),
            "--repeat-capture",
            repeat_capture,
            "--isolation-capture",
            isolation_capture,
            "--comparison-controls",
            str(comparison_controls_path),
            "--task-results",
            str(tasks_path),
            "--arithmetic-class",
            "T2",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert validate_execution_profile_artifact(artifact)["decision"]["status"] == "passed"


def test_qwen36_teacher_fixture_adapter_preserves_categories_and_labels() -> None:
    fixture = {
        "kind": "quant_quality_teacher_fixture",
        "teacher_steps": 2,
        "prompts": [
            {
                "id": "code-1",
                "category": "code",
                "teacher_token_ids": [7, 8],
            },
            {
                "id": "ja-1",
                "category": "general_ja",
                "teacher_token_ids": [9, 10],
            },
        ],
    }

    rows = qwen36_rows_from_teacher_fixture(fixture, scenario_id="qwen36-c1")

    assert [row.teacher_token_id for row in rows] == [7, 8, 9, 10]
    assert [row.category for row in rows] == ["code", "code", "general_ja", "general_ja"]
    assert [row.transition for row in rows] == [
        "prefill_to_c1",
        "steady",
        "prefill_to_c1",
        "steady",
    ]
    assert rows[2].scenario_step == 2


def test_capture_rejects_duplicate_logical_rows() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="duplicate logical row"):
        RunCapture(
            execution_profile="strict",
            scenario_id="bad",
            run_id="duplicate-row-run",
            variant_manifest_sha256=manifest_sha256(_manifest("strict")),
            repeat_index=0,
            rows=(rows[0], rows[0]),
            logits=np.zeros((2, 4), dtype=np.float32),
            selected_token_ids=(0, 0),
            controls=_controls(),
        )
