from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.execution_profiles import resolve_runtime_profile
from hipengine.speculative.paro_mtp_profiles import (
    FAST_VERIFIER_CANDIDATE_VARIANT,
    PARO_MTP_BACKEND,
    PARO_MTP_MODEL,
    PARO_MTP_MODEL_QUANT,
    VERIFIER_LAYER,
)
from scripts import mtp_paro_verifier_numerics
from scripts.mtp_paro_verifier_numerics import (
    _capture_result,
    _capture_sha256,
    _numerical_report,
    _review_manifests,
    _route_attestation,
    _row_review_diagnostic,
    _scope_summaries,
    _summary,
    _summary_line,
    _wilson_interval,
)


def _row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "category": "general_en",
        "shape": "c2_b1",
        "transition": "verify_to_verify",
        "row_role": "root",
        "decision_role": "draft_acceptance_or_reject_correction",
        "strict_selected_for_commit": True,
        "top5_overlap": 1.0,
        "strict_margin": 0.1,
        "kl": 1.0e-4,
        "top1_equal": True,
    }
    base.update(overrides)
    return base


def _rows(
    category: str, kls: list[float], *, top1: list[bool] | None = None
) -> list[dict[str, object]]:
    flags = top1 if top1 is not None else [True] * len(kls)
    return [
        _row(category=category, kl=kl, top1_equal=flag)
        for kl, flag in zip(kls, flags, strict=True)
    ]


def _report(rows: list[dict[str, object]], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "prefill_root_equal": True,
        "finite_logits": True,
        "decision_mismatches": (),
        "route_flags_attested": True,
    }
    kwargs.update(overrides)
    return _numerical_report(rows, **kwargs)  # type: ignore[arg-type]


def test_verifier_numerical_summary_reports_binding_tail_and_top1() -> None:
    result = _summary(
        np.asarray([0.0, 1.0e-3, 2.0e-2, 5.0e-2], dtype=np.float64),
        np.asarray([True, True, False, True], dtype=np.bool_),
    )

    assert result["rows"] == 4
    assert result["mean_kl"] == pytest.approx(0.01775)
    assert result["max_kl"] == 0.05
    assert result["top1_agreement"] == 0.75
    assert result["p99_kl"] > result["p95_kl"]


def test_summary_reports_median_and_p90_robust_statistics() -> None:
    values = np.asarray([1.0e-6] * 97 + [6.0e-2] * 3, dtype=np.float64)
    result = _summary(values, np.asarray([True] * 100, dtype=np.bool_))

    assert result["median_kl"] == pytest.approx(float(np.median(values)))
    assert result["p90_kl"] == pytest.approx(float(np.percentile(values, 90.0)))
    assert result["large_drift_rows"] == 3
    assert result["large_drift_rate"] == pytest.approx(0.03)


def test_top1_summary_reports_resolution_and_wilson_interval() -> None:
    result = _summary(
        np.zeros((68,), dtype=np.float64),
        np.asarray([True] * 67 + [False], dtype=np.bool_),
    )
    expected = 67 / 68
    low, high = _wilson_interval(67, 68)

    assert result["top1_agreement"] == pytest.approx(expected)
    assert result["top1_matches"] == 67
    assert result["top1_mismatches"] == 1
    assert result["top1_wilson95_low"] == pytest.approx(low)
    assert result["top1_wilson95_high"] == pytest.approx(high)
    assert low < expected < high


def test_row_review_diagnostic_localizes_near_tie_flip() -> None:
    strict = np.asarray([5.0, 4.8, 1.0, 0.0], dtype=np.float32)
    candidate = np.asarray([4.9, 5.0, 1.0, 0.0], dtype=np.float32)
    result = _row_review_diagnostic(strict, candidate, top_k=3)

    assert result["strict_topk"] == [0, 1, 2]
    assert result["candidate_topk"] == [1, 0, 2]
    assert result["strict_margin"] == pytest.approx(0.2)
    assert result["candidate_margin"] == pytest.approx(0.1)
    assert result["strict_top1_candidate_rank"] == 2
    assert result["candidate_top1_strict_rank"] == 2
    assert result["strict_gap_to_candidate_top1"] == pytest.approx(0.2)
    assert result["candidate_gap_to_strict_top1"] == pytest.approx(0.1)


def test_scope_summaries_bind_each_row_role() -> None:
    rows = [
        _row(row_role="root", kl=1.0e-4, top1_equal=False),
        _row(
            row_role="draft_candidate",
            strict_selected_for_commit=False,
            kl=1.0e-4,
            top1_equal=True,
        ),
    ]
    result = _scope_summaries(rows)

    assert result["row_role"]["root"]["top1_agreement"] == 0.0
    assert result["row_role"]["root"]["binding"] is False
    assert result["row_role"]["root"]["passed"] is False
    assert result["row_role"]["draft_candidate"]["top1_agreement"] == 1.0
    assert result["row_role"]["draft_candidate"]["binding"] is False
    assert result["row_role"]["draft_candidate"]["passed"] is True
    assert result["category"]["general_en"]["binding"] is True


def test_scope_summaries_bind_the_top1_floor_and_report_scope_kl_as_advisory() -> None:
    rows = _rows("general_ja", [2.0e-3, 2.0e-3, 2.0e-3])

    result = _scope_summaries(rows)
    scope = result["category"]["general_ja"]

    # EXECUTION-PROFILES 6.1: a declared scope binds on its top-1 floor only.
    assert scope["binding"] is True
    assert scope["passed"] is True
    assert scope["mean_kl"] > scope["global_limits"]["mean_kl_max"]
    assert scope["kl_within_global_limits"] is False
    assert scope["advisory"] == "scope_kl_advisory"


def test_scope_summaries_fail_the_top1_floor_with_one_miss_in_a_three_row_scope() -> None:
    rows = _rows("code", [1.0e-6, 1.0e-6, 1.0e-6], top1=[True, True, False])

    scope = _scope_summaries(rows)["category"]["code"]

    assert scope["top1_agreement"] == pytest.approx(2 / 3)
    assert scope["passed"] is False
    assert scope["kl_within_global_limits"] is True
    assert scope["advisory"] is None


def test_capture_hash_binds_prompt_manifest_rows_and_cycles() -> None:
    kwargs = {
        "prompt": {"name": "fixture", "tokens": 4},
        "candidate_manifest_sha256": "a" * 64,
        "rows": [{"row": 0, "kl": 1.0e-4}],
        "cycles": [{"cycle": 1, "root": 7}],
    }
    first = _capture_sha256(**kwargs)
    second = _capture_sha256(**kwargs)
    changed = _capture_sha256(**{**kwargs, "cycles": [{"cycle": 1, "root": 8}]})

    assert first == second
    assert len(first) == 64
    assert changed != first


def test_registered_production_manifest_selects_fast_verifier_candidate() -> None:
    result = _review_manifests()
    production = resolve_runtime_profile(
        model=PARO_MTP_MODEL,
        backend=PARO_MTP_BACKEND,
        quant=PARO_MTP_MODEL_QUANT,
        profile="production",
    )
    selections = result["registered_production_manifest"]["selections"]
    verifier = next(row for row in selections if row["layer"] == VERIFIER_LAYER)

    assert result["registered_production_sha256"] == production.manifest_sha256
    assert result["candidate_verifier_variant"] == FAST_VERIFIER_CANDIDATE_VARIANT
    assert verifier["selected_variant"] == FAST_VERIFIER_CANDIDATE_VARIANT
    assert verifier["scope"] == "graph_off_fixed_chain"
    assert verifier["strict_fallback_variant"] == "b1_graph_off_strict_exact"
    assert result["candidate_registered"] is True
    assert result["candidate_certified"] is True
    assert result["candidate_selected_by_production"] is True
    assert result["capture_manifest_sha256"] == production.manifest_sha256
    assert "candidate_registered_but_uncertified" not in result
    assert len(result["registered_strict_sha256"]) == 64


def test_legacy_review_manifest_is_retained_but_labelled_review_only() -> None:
    result = _review_manifests()
    legacy = result["legacy_review_manifest"]

    assert result["candidate_review_sha256"] == result["legacy_review_manifest_sha256"]
    assert legacy["describes_execution"] is False
    assert legacy["scope_note"] == "review_only_scope_not_a_registered_selection"
    assert len(result["candidate_review_sha256"]) == 64


def test_current_sparse_moe_policy_computes_robust_metrics() -> None:
    rows = _rows("code", [1.0e-6] * 97 + [6.0e-2] * 3)

    envelope = _report(rows)["envelopes"]["current_sparse_moe_policy"]
    aggregate = envelope["aggregate"]

    assert envelope["binding"] is True
    assert envelope["source"] == "docs/EXECUTION-PROFILES.md 2.11"
    assert aggregate["rows"] == 100
    assert aggregate["median_kl"] == pytest.approx(1.0e-6)
    assert aggregate["p90_kl"] == pytest.approx(1.0e-6)
    assert aggregate["large_drift_rows"] == 3
    assert aggregate["large_drift_rate"] == pytest.approx(0.03)
    assert aggregate["median_kl_within_limit"] is True
    assert aggregate["p90_kl_within_limit"] is True
    assert aggregate["large_drift_rate_within_limit"] is True
    assert aggregate["top1_within_limit"] is True


def test_current_sparse_moe_policy_reports_per_category_damage_above_the_budget() -> None:
    rows = _rows("code", [1.0e-6] * 9 + [6.0e-2]) + _rows("general_en", [1.0e-6] * 40)

    envelope = _report(rows)["envelopes"]["current_sparse_moe_policy"]

    assert envelope["aggregate"]["rows"] == 50
    assert envelope["aggregate"]["large_drift_rate"] == pytest.approx(0.02)
    assert envelope["aggregate"]["large_drift_rate_within_limit"] is True
    assert envelope["per_category"]["code"]["large_drift_rate"] == pytest.approx(0.1)
    assert envelope["per_category"]["code"]["large_drift_rate_within_limit"] is False
    assert envelope["per_category"]["general_en"]["large_drift_rate_within_limit"] is True
    assert envelope["per_category"]["code"]["limits"]["large_drift_rate_max"] == pytest.approx(0.08)


def test_current_sparse_moe_policy_reports_a_global_median_above_the_limit() -> None:
    rows = _rows("code", [2.0e-3] * 20)

    report = _report(rows)
    envelope = report["envelopes"]["current_sparse_moe_policy"]
    median = envelope["observed_comparisons"]["global"]["median_kl"]

    assert median["observed"] == pytest.approx(2.0e-3)
    assert median["sample_within_limit"] is False
    assert "global.median_kl" in envelope["sample_limits_exceeded"]
    assert envelope["status"] == "inconclusive"
    assert report["status"] == "inconclusive"


def test_legacy_mean_tail_envelope_sample_is_reported_by_name() -> None:
    # The 2026-08-16 mean/tail sample exceeds its mean KL limit while the
    # current sparse-MoE sample does not: both are reported by name and neither
    # decides admission.
    rows = _rows("general_en", [1.0e-6, 1.0e-6, 9.0e-3])

    report = _report(rows)
    legacy = report["envelopes"]["legacy_mean_tail_kl_envelope"]
    current = report["envelopes"]["current_sparse_moe_policy"]

    assert legacy["binding"] is False
    assert legacy["thresholds"]["mean_kl_max"] == pytest.approx(1.0e-3)
    assert legacy["observed_comparisons"]["mean_kl"]["sample_within_limit"] is False
    assert legacy["observed_comparisons"]["mean_kl"]["decides_policy"] is False
    assert legacy["sample_limits_exceeded"] == ["mean_kl", "p95_kl"]
    assert legacy["status"] == "inconclusive"
    assert current["sample_limits_exceeded"] == []
    assert current["status"] == "inconclusive"
    assert report["status"] == "inconclusive"
    assert report["policy_decision"] == "inconclusive"
    assert "mean_kl" not in report["execution_checks"]
    assert report["envelope_disagreement"]["disagree"] is True
    assert "binding 2.11" in report["envelope_disagreement"]["note"]


def test_envelopes_agree_when_both_exceed_sample_limits() -> None:
    report = _report(_rows("code", [1.0e-6] * 10, top1=[True] * 5 + [False] * 5))
    comparison = report["envelope_disagreement"]
    assert comparison["current_sample_limits_exceeded"]
    assert comparison["legacy_sample_limits_exceeded"]
    assert comparison["disagree"] is False


def test_scope_kl_advisory_does_not_decide_admission() -> None:
    rows = _rows("general_en", [1.0e-6, 1.0e-6, 9.0e-3])

    report = _report(rows)
    advisories = report["envelopes"]["legacy_mean_tail_kl_envelope"]["scope_kl_advisories"]

    assert report["scope_failures"] == []
    assert report["scope_kl_advisories"] == advisories
    category_advisories = [
        entry for entry in advisories if entry["dimension"] == "category"
    ]
    assert [entry["value"] for entry in category_advisories] == ["general_en"]
    assert {entry["dimension"] for entry in advisories} == {
        "category",
        "shape",
        "transition",
    }
    assert category_advisories[0]["mean_kl"] == pytest.approx(
        (1.0e-6 + 1.0e-6 + 9.0e-3) / 3.0
    )
    scoped = report["envelopes"]["current_sparse_moe_policy"]["observed_comparisons"]
    assert scoped["per_category"]["general_en"]["median_kl"]["decides_policy"] is False
    assert report["status"] == "inconclusive"


def test_envelope_disagreement_is_false_when_both_samples_agree() -> None:
    report = _report(_rows("code", [1.0e-6] * 10))

    assert report["envelopes"]["legacy_mean_tail_kl_envelope"]["sample_limits_exceeded"] == []
    assert report["envelopes"]["current_sparse_moe_policy"]["sample_limits_exceeded"] == []
    assert report["envelope_disagreement"]["disagree"] is False


def test_status_is_not_full_qualification() -> None:
    rows = _rows("code", [1.0e-6] * 40) + _rows("general_en", [2.0e-6] * 40)

    report = _report(rows)
    qualification = report["qualification"]

    assert report["status"] == "inconclusive"
    assert "metric_status" not in report
    assert report["policy_decision"] == "inconclusive"
    assert qualification["status"] == "not_qualified"
    assert qualification["full_qualification"] is False
    assert qualification["policy_decision"] == "inconclusive"
    assert qualification["binding_envelope"] == "current_sparse_moe_policy"
    assert qualification["scope"] == "b1_target_verifier_teacher_forced_diagnostic"
    assert qualification["rate_gates_decided"] is False
    assert qualification["full_protocol_reached"] is False
    requires = " ".join(qualification["full_qualification_requires"])
    assert "repeat" in requires
    assert "isolation" in requires
    assert "task" in requires
    uncovered = " ".join(qualification["does_not_cover"])
    assert "B2" in uncovered
    assert "public adapter" in uncovered
    assert "admission" in uncovered


def test_rate_gates_are_never_decided_at_any_row_count() -> None:
    for count in (1, 266, 1548, 2000):
        report = _report(_rows("code", [1.0e-6] * count))
        envelope = report["envelopes"]["current_sparse_moe_policy"]
        policy = envelope["rate_gate_policy"]

        assert policy["decided"] is False
        assert "6.4" in policy["reason"]
        assert policy["gates"] == ["top1_agreement", "large_drift_rate"]
        assert envelope["observed_comparisons"]["global"]["top1_agreement"][
            "decides_policy"
        ] is False
        assert report["qualification"]["rate_gates_decided"] is False


def test_rate_gate_observation_reports_an_observed_top1_miss() -> None:
    rows = _rows("code", [1.0e-6] * 10, top1=[True] * 9 + [False])

    envelope = _report(rows)["envelopes"]["current_sparse_moe_policy"]
    top1 = envelope["observed_comparisons"]["global"]["top1_agreement"]

    assert top1["mismatches"] == 1
    assert top1["observed"] == pytest.approx(0.9)
    assert top1["sample_within_limit"] is False
    assert top1["decides_policy"] is False
    assert top1["wilson95"][0] < 0.9 < top1["wilson95"][1]
    assert envelope["status"] == "inconclusive"


def test_unreconcilable_top1_interval_is_reported_but_not_decided() -> None:
    rows = _rows("code", [1.0e-6] * 10, top1=[True] * 5 + [False] * 5)

    envelope = _report(rows)["envelopes"]["current_sparse_moe_policy"]
    top1 = envelope["observed_comparisons"]["global"]["top1_agreement"]

    # Even an interval that lies entirely below the limit decides nothing here.
    assert top1["wilson95"][1] < top1["limit"]
    assert top1["decides_policy"] is False
    assert envelope["status"] == "inconclusive"
    assert _report(rows)["status"] == "inconclusive"


def test_unreconcilable_flip_damage_interval_is_reported_but_not_decided() -> None:
    rows = _rows("code", [1.0e-6] * 2 + [6.0e-2] * 8)

    envelope = _report(rows)["envelopes"]["current_sparse_moe_policy"]
    damage = envelope["observed_comparisons"]["global"]["large_drift_rate"]

    assert damage["rows_above"] == 8
    assert damage["wilson95"][0] > damage["limit"]
    assert damage["decides_policy"] is False
    assert envelope["status"] == "inconclusive"


def test_route_attestation_matches_the_registered_binder(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_MTP_CHAIN_ATTN_MODE", "sentinel")

    result = _route_attestation()

    strict = result["arms"]["strict"]
    fast = result["arms"]["fast"]
    assert result["matches_registered_binder"] is True
    assert strict["mismatches"] == {}
    assert fast["mismatches"] == {}
    moe_flag = "HIPENGINE_QWEN35_MOE_C1_FORCE_SMALL_BATCH_SHARED_EXPERT"
    assert strict["registered_binder_flags"][moe_flag] == "1"
    assert strict["bound_flags"]["HIPENGINE_LINEAR_OUT_C1_EXACT_ROWS"] == "1"
    assert fast["registered_binder_flags"]["HIPENGINE_LINEAR_OUT_C1_EXACT_ROWS"] == "0"
    assert fast["bound_flags"][moe_flag] == "0"
    # Nothing the harness pins sits outside the registered binder: the retired
    # fused-LM-head selector is gone rather than merely forced off.
    assert result["harness_pins_outside_binder"] == []
    assert result["arms"]["strict"]["pins_outside_binder"] == {}
    assert result["arms"]["fast"]["pins_outside_binder"] == {}
    # The sandboxed binder probe must restore the ambient environment.
    assert result["source"].startswith("registered runtime profile binder")
    import os

    assert os.environ["HIPENGINE_MTP_CHAIN_ATTN_MODE"] == "sentinel"


def test_route_attestation_detects_arm_flag_drift(monkeypatch) -> None:
    drifted = dict(mtp_paro_verifier_numerics._FAST_FLAGS)
    drifted["HIPENGINE_QWEN35_MOE_C1_FORCE_SMALL_BATCH_SHARED_EXPERT"] = "1"
    monkeypatch.setattr(mtp_paro_verifier_numerics, "_FAST_FLAGS", drifted)

    result = _route_attestation()

    assert result["matches_registered_binder"] is False
    assert result["arms"]["fast"]["mismatches"] == {
        "HIPENGINE_QWEN35_MOE_C1_FORCE_SMALL_BATCH_SHARED_EXPERT": {
            "bound": "1",
            "registered": "0",
        }
    }


def _capture_fixture(**overrides: object) -> dict[str, object]:
    rows = _rows("general_en", [1.0e-6, 2.0e-6])
    cycle = {
        "cycle": 1,
        "output_offset": 0,
        "context": 8,
        "root": 11,
        "candidate": 12,
        "strict_accepted": 1,
        "strict_bonus": 13,
        "strict_commit_row": 1,
        "strict_commit_position": 9,
        "strict_target_top1": [11, 12],
        "committed": [11, 12],
        "candidate_accepted": 1,
        "candidate_bonus": 13,
        "candidate_commit_row": 1,
        "task_decision_mismatch": False,
        "strict_and_candidate_logits_finite": True,
        "runner_reported_finite_logits": True,
        "runner_reported_gpu_accept_match_cpu": True,
    }
    kwargs: dict[str, object] = {
        "row_metrics": rows,
        "replay_cycles": [cycle],
        "decision_mismatch": None,
        "decision_mismatches": [],
        "manifests_before": _review_manifests(),
        "attestation": _route_attestation(),
        "observed_flags": {
            "strict": dict(mtp_paro_verifier_numerics._STRICT_FLAGS),
            "fast": dict(mtp_paro_verifier_numerics._FAST_FLAGS),
        },
        "ambient_env": {"HIPENGINE_W4_MULTI_ROW_PACK8": "0"},
        "prompt_metadata": {
            "name": "fixture",
            "category": "general_en",
            "split": "heldout",
            "render": "raw",
            "prompt_tokens": 8,
            "decode_tokens": 2,
        },
        "strict_root": 11,
        "candidate_root": 11,
        "logits_finite_all": True,
        "model": "fixture-model",
        "backend": "hip_gfx1100",
        "seconds": 0.5,
    }
    kwargs.update(overrides)
    return _capture_result(**kwargs)  # type: ignore[arg-type]


def test_capture_result_assembles_named_envelopes_and_attestation() -> None:
    result = _capture_fixture()

    assert result["schema"] == "hipengine.paro_mtp_verifier_numerics.v5"
    assert result["status"] == "inconclusive"
    assert result["policy_decision"] == "inconclusive"
    assert "metric_status" not in result
    assert result["qualification"]["status"] == "not_qualified"
    assert result["qualification"]["full_qualification"] is False
    assert result["capture_binding"] == "registered_production_manifest_sha256"
    assert result["capture_sha256"] == _capture_sha256(
        prompt=result["prompt"],
        candidate_manifest_sha256=result["manifests"]["registered_production_sha256"],
        rows=result["rows"],
        cycles=result["cycles"],
    )
    assert result["manifests_stable_during_capture"] is True
    assert result["execution_checks"]["manifests_stable_during_capture"] is True
    assert result["execution_checks"]["route_flags_attested"] is True
    assert result["execution_failures"] == []
    assert result["route_attestation"]["attested"] is True
    assert result["route_attestation"]["observed_matches_bound"] == {
        "strict": True,
        "fast": True,
    }
    assert result["ambient_env"]["present"] is True
    assert result["ambient_env"]["route_env_at_start"] == {}
    assert result["runner_reported_metadata"]["all_cycles_logits_finite"] is True
    assert result["envelopes"]["current_sparse_moe_policy"]["binding"] is True
    assert result["envelopes"]["legacy_mean_tail_kl_envelope"]["binding"] is False
    assert result["sample_scope"]["policy_decision"] == "inconclusive"
    assert result["thresholds_envelope"] == "legacy_mean_tail_kl_envelope"
    assert result["review"]["binding_envelope"] == "current_sparse_moe_policy"
    assert result["review"]["policy_decision"] == "inconclusive"
    assert result["review"]["rate_gates_decided"] is False


def test_capture_result_fails_a_drifted_arm_and_reports_it() -> None:
    drifted = dict(mtp_paro_verifier_numerics._FAST_FLAGS)
    drifted["HIPENGINE_QWEN35_MOE_C1_FORCE_SMALL_BATCH_SHARED_EXPERT"] = "1"
    result = _capture_fixture(
        observed_flags={
            "strict": dict(mtp_paro_verifier_numerics._STRICT_FLAGS),
            "fast": drifted,
        },
    )

    assert result["route_attestation"]["attested"] is False
    assert result["route_attestation"]["observed_matches_bound"]["fast"] is False
    assert result["execution_checks"]["route_flags_attested"] is False
    assert result["execution_failures"] == ["route_flags_attested"]
    assert result["status"] == "failed"
    assert result["policy_decision"] == "inconclusive"


def test_summary_line_reports_the_status_and_named_samples() -> None:
    result = _capture_fixture()

    line = _summary_line(result)

    assert line["status"] == "inconclusive"
    assert line["policy_decision"] == "inconclusive"
    assert line["qualification_status"] == "not_qualified"
    assert line["binding_envelope"] == "current_sparse_moe_policy"
    assert line["current_sample_limits_exceeded"] == []
    assert line["legacy_sample_limits_exceeded"] == []
    assert line["rate_gates_decided"] is False
    assert line["execution_failures"] == []
    assert line["rows"] == 2
    assert line["route_flags_attested"] is True


def test_describe_scope_cli_reports_unqualified_scope(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["mtp_paro_verifier_numerics.py", "--describe-scope"])

    assert mtp_paro_verifier_numerics.main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "hipengine.paro_mtp_verifier_numerics.scope.v2"
    assert payload["qualification"]["status"] == "not_qualified"
    assert payload["qualification"]["full_qualification"] is False
    assert payload["manifests"]["candidate_selected_by_production"] is True
    assert payload["manifests"]["candidate_certified"] is True
    assert payload["envelopes"]["current_sparse_moe_policy"]["binding"] is True
    assert payload["envelopes"]["legacy_mean_tail_kl_envelope"]["binding"] is False
    assert payload["route_attestation"]["matches_registered_binder"] is True
    assert payload["backend_scope"]["restriction"] == "gfx1100_only_diagnostic"


def test_verifier_capture_reads_materialized_verify_logits(monkeypatch) -> None:
    copied: list[tuple[int, int]] = []

    def fake_copy(_host_ptr, buffer, nbytes, *, runtime):
        del runtime
        copied.append((int(buffer.ptr), int(nbytes)))

    monkeypatch.setattr(mtp_paro_verifier_numerics, "copy_device_to_host", fake_copy)
    session = SimpleNamespace(
        vocab_size=4,
        verify_lm_logits=SimpleNamespace(ptr=0xABC000),
        batch_lm_logits=SimpleNamespace(ptr=0xBAD000),
        runtime=object(),
    )
    logits = mtp_paro_verifier_numerics._copy_logits(session, rows=2)

    assert logits.shape == (2, 4)
    assert copied == [(0xABC000, 2 * 4 * 4)]


def test_reduced_run_rate_interval_never_decides_a_rate_gate() -> None:
    # EXECUTION-PROFILES 6.4 rule 1: top-1 is a raw miss count with an interval,
    # never a pass/fail. Five misses in ten rows is as far from the limit as a
    # reduced sample can get, and it still must not reject.
    top1_rows = _rows("code", [1.0e-6] * 10, top1=[True] * 5 + [False] * 5)
    damage_rows = _rows("code", [1.0e-6] * 2 + [6.0e-2] * 8)

    for rows in (top1_rows, damage_rows):
        report = _report(rows)
        envelope = report["envelopes"]["current_sparse_moe_policy"]
        comparisons = envelope["observed_comparisons"]["global"]

        assert comparisons["top1_agreement"]["decides_policy"] is False
        assert comparisons["large_drift_rate"]["decides_policy"] is False
        low, high = comparisons["top1_agreement"]["wilson95"]
        assert 0.0 <= low < high <= 1.0
        damage_low, damage_high = comparisons["large_drift_rate"]["wilson95"]
        assert 0.0 <= damage_low < damage_high <= 1.0
        assert envelope["rate_gate_policy"]["decided"] is False
        assert envelope["status"] == "inconclusive"
        assert envelope["policy_decision"] == "inconclusive"
        assert report["status"] == "inconclusive"
        assert report["policy_decision"] == "inconclusive"

    # The unreconcilable observations are still reported by name.
    top1_comparison = _report(top1_rows)["envelopes"]["current_sparse_moe_policy"][
        "observed_comparisons"
    ]["global"]["top1_agreement"]
    assert top1_comparison["mismatches"] == 5
    assert top1_comparison["observed"] == pytest.approx(0.5)
    assert top1_comparison["sample_within_limit"] is False
    damage_comparison = _report(damage_rows)["envelopes"]["current_sparse_moe_policy"][
        "observed_comparisons"
    ]["global"]["large_drift_rate"]
    assert damage_comparison["rows_above"] == 8
    assert damage_comparison["observed"] == pytest.approx(0.8)
    assert damage_comparison["sample_within_limit"] is False


def test_no_rate_gate_reaches_the_deciding_execution_checks() -> None:
    rows = _rows("code", [1.0e-6] * 10, top1=[True] * 5 + [False] * 5)

    report = _report(rows)

    assert set(report["execution_checks"]) == {
        "prefill_root_equal",
        "finite_logits",
        "finite_kl",
        "task_decisions",
        "route_flags_attested",
    }
    assert report["execution_failures"] == []
    # Only execution failures decide: no metric or rate gate is a check.
    assert not any(
        name.startswith(("median_kl", "p90_kl", "top1", "large_drift"))
        for name in report["execution_checks"]
    )
    assert "rate_gates_reconcilable_with_limits" not in report["execution_checks"]


def test_single_prompt_row_count_never_declares_the_full_protocol() -> None:
    # The canonical full protocol is 12 cases x 129 transitions across the
    # prompt suite; 1548 rows from one prompt is not that.
    for count in (1548, 1600, 3096):
        report = _report(_rows("code", [1.0e-6] * count))
        sample_scope = report["sample_scope"]

        assert sample_scope["rows"] == count
        assert sample_scope["prompts"] == 1
        assert sample_scope["full_protocol_rows"] == 1548
        assert sample_scope["full_protocol_reached"] is False
        assert sample_scope["policy_decision"] == "inconclusive"
        assert report["policy_decision"] == "inconclusive"
        assert report["status"] == "inconclusive"
        assert report["qualification"]["full_qualification"] is False
        assert report["qualification"]["rate_gates_decided"] is False
        for envelope in report["envelopes"].values():
            assert envelope["policy_decision"] == "inconclusive"
            assert envelope["status"] == "inconclusive"
        assert report["envelopes"]["current_sparse_moe_policy"]["rate_gate_policy"][
            "decided"
        ] is False


def test_continuous_sample_above_a_limit_is_an_observation_not_a_verdict() -> None:
    report = _report(_rows("code", [2.0e-3] * 20))
    envelope = report["envelopes"]["current_sparse_moe_policy"]
    median = envelope["observed_comparisons"]["global"]["median_kl"]

    assert median["observed"] == pytest.approx(2.0e-3)
    assert median["limit"] == pytest.approx(1.0e-3)
    assert median["sample_within_limit"] is False
    assert median["decides_policy"] is False
    assert "global.median_kl" in envelope["sample_limits_exceeded"]
    assert envelope["status"] == "inconclusive"
    assert report["status"] == "inconclusive"
    assert report["policy_decision"] == "inconclusive"
    # No calibrated control scope and no screen-to-full ratio interval exist in
    # this batch, so 6.4 rule 3 forbids reading the sample against the limit.
    assert report["sample_scope"]["control_scopes"] == []
    assert report["sample_scope"]["screen_to_full_ratio_interval"] is None
    assert report["qualification"]["full_qualification"] is False


def test_execution_failures_still_fail_the_diagnostic() -> None:
    rows = _rows("code", [1.0e-6] * 10)

    for kwargs, name in (
        ({"finite_logits": False}, "finite_logits"),
        ({"route_flags_attested": False}, "route_flags_attested"),
        ({"decision_mismatches": [{"cycle": 1}]}, "task_decisions"),
        ({"prefill_root_equal": False}, "prefill_root_equal"),
        ({"extra_checks": {"manifests_stable_during_capture": False}},
         "manifests_stable_during_capture"),
    ):
        report = _report(rows, **kwargs)
        assert report["status"] == "failed"
        assert report["execution_failures"] == [name]
        assert report["policy_decision"] == "inconclusive"


def test_backend_other_than_gfx1100_is_refused_by_diagnostic_scope() -> None:
    assert (
        mtp_paro_verifier_numerics._require_diagnostic_backend(PARO_MTP_BACKEND)
        == PARO_MTP_BACKEND
    )

    with pytest.raises(mtp_paro_verifier_numerics.DiagnosticScopeError) as excinfo:
        mtp_paro_verifier_numerics._require_diagnostic_backend("hip_gfx1151")

    message = str(excinfo.value)
    assert "diagnostic scope" in message
    assert PARO_MTP_BACKEND in message
    assert "hip_gfx1151" in message
    # The refusal is a diagnostic-scope limit, not model/runtime admission.
    assert "not model" in message


def test_cli_refuses_a_non_gfx1100_backend_before_capture(monkeypatch, capsys, tmp_path) -> None:
    out = tmp_path / "capture.json"

    def _must_not_run(**_kwargs):
        raise AssertionError("capture must not start for a non-gfx1100 backend")

    monkeypatch.setattr(mtp_paro_verifier_numerics, "run_sequential", _must_not_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mtp_paro_verifier_numerics.py",
            "--model",
            str(tmp_path / "model"),
            "--prompt-name",
            "fixture",
            "--out",
            str(out),
            "--backend",
            "hip_gfx1151",
        ],
    )

    assert mtp_paro_verifier_numerics.main() == 3
    assert not out.exists()
    assert "diagnostic scope" in capsys.readouterr().err


def test_capture_result_records_backend_scope() -> None:
    result = _capture_fixture()
    scope = result["backend_scope"]

    assert scope["backend"] == PARO_MTP_BACKEND
    assert scope["required_backend"] == PARO_MTP_BACKEND
    assert scope["manifest_backend"] == PARO_MTP_BACKEND
    assert scope["restriction"] == "gfx1100_only_diagnostic"
    assert scope["refusal"] == "DiagnosticScopeError"
    assert "not model" in scope["note"]
    # A capture result cannot claim another backend either.
    with pytest.raises(mtp_paro_verifier_numerics.DiagnosticScopeError):
        _capture_fixture(backend="hip_gfx1151")


def test_cli_exit_code_matches_the_overall_diagnostic_status(
    monkeypatch, capsys, tmp_path
) -> None:
    result = _capture_fixture()
    monkeypatch.setattr(
        mtp_paro_verifier_numerics,
        "run_sequential",
        lambda **_kwargs: dict(result),
    )
    argv = [
        "mtp_paro_verifier_numerics.py",
        "--model",
        str(tmp_path / "model"),
        "--prompt-name",
        "fixture",
        "--out",
        str(tmp_path / "capture.json"),
    ]

    monkeypatch.setattr(sys, "argv", argv)
    assert mtp_paro_verifier_numerics.main() == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "inconclusive"
    assert printed["policy_decision"] == "inconclusive"
    assert printed["qualification_status"] == "not_qualified"

    failed = _capture_fixture(logits_finite_all=False)
    monkeypatch.setattr(
        mtp_paro_verifier_numerics,
        "run_sequential",
        lambda **_kwargs: dict(failed),
    )
    monkeypatch.setattr(sys, "argv", argv)
    assert mtp_paro_verifier_numerics.main() == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "failed"
    assert printed["execution_failures"] == ["finite_logits"]


def test_describe_scope_states_the_exit_codes_and_backend_scope(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["mtp_paro_verifier_numerics.py", "--describe-scope"])

    assert mtp_paro_verifier_numerics.main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["backend_scope"]["restriction"] == "gfx1100_only_diagnostic"
    assert payload["qualification"]["policy_decision"] == "inconclusive"
    assert payload["qualification"]["rate_gates_decided"] is False
    assert payload["sample_scope"]["full_protocol_reached"] is False
    assert payload["sample_scope"]["policy_decision"] == "inconclusive"
    exit_codes = payload["exit_codes"]
    assert "inconclusive" in exit_codes["0"]
    assert "failed" in exit_codes["1"]
    assert "diagnostic scope" in exit_codes["3"]


def test_observed_comparison_states_the_sample_without_deciding() -> None:
    at_most = mtp_paro_verifier_numerics._observed_comparison(
        observed=2.0e-3, limit=1.0e-3, at_least=False
    )
    at_least = mtp_paro_verifier_numerics._observed_comparison(
        observed=0.97, limit=0.96, at_least=True, mismatches=3
    )

    assert at_most == {
        "observed": 2.0e-3,
        "limit": 1.0e-3,
        "direction": "at_most",
        "sample_within_limit": False,
        "decides_policy": False,
    }
    assert at_least["direction"] == "at_least"
    assert at_least["sample_within_limit"] is True
    assert at_least["decides_policy"] is False
    assert at_least["mismatches"] == 3


def test_run_sequential_refuses_a_non_gfx1100_backend_before_loading(monkeypatch) -> None:
    def _must_not_load(_path):
        raise AssertionError("capture must not load a suite for a non-gfx1100 backend")

    monkeypatch.setattr(mtp_paro_verifier_numerics, "_load_prompt_suite", _must_not_load)

    with pytest.raises(mtp_paro_verifier_numerics.DiagnosticScopeError):
        mtp_paro_verifier_numerics.run_sequential(
            model=Path("fixture-model"),
            prompts_file=Path("fixture-prompts.jsonl"),
            prompt_name="fixture",
            prompt_render="raw",
            decode_tokens=4,
            backend="cuda_sm86",
        )


# ---------------------------------------------------------------------------
# Retired-route and evaluator-entry guarantees
# ---------------------------------------------------------------------------


def test_verifier_routes_do_not_configure_retired_fused_lm_head() -> None:
    retired = "HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD"
    assert retired not in mtp_paro_verifier_numerics._ROUTE_FLAGS
    for flags in (
        mtp_paro_verifier_numerics._STRICT_FLAGS,
        mtp_paro_verifier_numerics._FAST_FLAGS,
    ):
        assert retired not in flags
        assert set(flags) == set(mtp_paro_verifier_numerics._ROUTE_FLAGS)


def test_simultaneous_evaluator_is_absent() -> None:
    assert not hasattr(mtp_paro_verifier_numerics, "run")
    assert callable(mtp_paro_verifier_numerics.run_sequential)


def test_cli_dispatches_sequential_evaluator(tmp_path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    result = _capture_fixture()

    def fake_run_sequential(**kwargs):
        calls.append(kwargs)
        return dict(result)

    monkeypatch.setattr(
        mtp_paro_verifier_numerics, "run_sequential", fake_run_sequential
    )
    out = tmp_path / "gate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mtp_paro_verifier_numerics",
            "--model",
            "/models/gguf/fixture.gguf",
            "--prompt-name",
            "fixture",
            "--decode-tokens",
            "4",
            "--out",
            str(out),
        ],
    )

    assert mtp_paro_verifier_numerics.main() == 0

    assert len(calls) == 1
    assert calls[0]["prompt_name"] == "fixture"
    assert calls[0]["decode_tokens"] == 4
    assert calls[0]["backend"] == "hip_gfx1100"
    assert calls[0]["prompt_tokens_file"] is None
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "inconclusive"
