from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import pytest

from scripts.mtp_paro_verifier_repeat_review import aggregate_repeats
from scripts.mtp_paro_verifier_review_aggregate import aggregate, main

_ENVELOPES = ("current_sparse_moe_policy", "legacy_mean_tail_kl_envelope")


def _row(
    *,
    category: str = "code",
    kl: float = 1.0e-4,
    top1_equal: bool = True,
    decision_mismatch: bool = False,
) -> dict[str, Any]:
    return {
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
        "cycle_task_decision_mismatch": decision_mismatch,
        "strict_top1": 1,
        "candidate_top1": 1 if top1_equal else 2,
        "kl": kl,
        "top1_equal": top1_equal,
        "top5_overlap": 1.0,
        "max_abs_logit_delta": 0.1,
        "strict_margin": 0.05,
    }


def _capture(
    path: Path,
    *,
    prompt: str = "code",
    category: str = "code",
    rows: Sequence[dict[str, Any]] | None = None,
    execution_failures: Sequence[str] = (),
    status: str = "inconclusive",
    policy_decision: str = "inconclusive",
    capture_sha256: str = "c" * 64,
    capture_manifest_sha256: str = "a" * 64,
    candidate_review_sha256: str = "b" * 64,
    route_attested: bool = True,
    manifests_stable: bool = True,
    full_qualification: bool = False,
    schema: str = "hipengine.paro_mtp_verifier_numerics.v5",
    with_execution_failures: bool = True,
) -> Path:
    capture_rows = list(rows) if rows is not None else [_row(category=category)]
    mismatches = [row for row in capture_rows if not row["top1_equal"]]
    decisions = (
        [
            {
                "cycle": 1,
                "output_offset": 0,
                "strict_accepted": 1,
                "candidate_accepted": 0,
                "strict_bonus": 3,
                "candidate_bonus": 2,
            }
        ]
        if any(row["cycle_task_decision_mismatch"] for row in capture_rows)
        else []
    )
    payload: dict[str, Any] = {
        "schema": schema,
        "model": "fixture-model",
        "backend": "hip_gfx1100",
        "status": status,
        "policy_decision": policy_decision,
        "execution_checks": {"finite_kl": "finite_kl" not in execution_failures},
        "qualification": {
            "status": "not_qualified",
            "full_qualification": full_qualification,
            "policy_decision": policy_decision,
        },
        "capture_sha256": capture_sha256,
        "capture_binding": "registered_production_manifest_sha256",
        "manifests": {
            "candidate_review_sha256": candidate_review_sha256,
            "capture_manifest_sha256": capture_manifest_sha256,
        },
        "route_attestation": {
            "attested": route_attested,
            "matches_registered_binder": route_attested,
        },
        "manifests_stable_during_capture": manifests_stable,
        "backend_scope": {"backend": "hip_gfx1100", "restriction": "gfx1100_only_diagnostic"},
        "prompt": {"name": prompt, "category": category, "split": "heldout", "render": "raw"},
        "aggregate": {"rows": len(capture_rows)},
        "scope_failures": [],
        "rows": capture_rows,
        "review": {
            "top1_mismatch_rows": mismatches,
            "task_decision_mismatches": decisions,
        },
    }
    if with_execution_failures:
        payload["execution_failures"] = list(execution_failures)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_low_drift_aggregate_is_observational_and_never_admits(tmp_path: Path) -> None:
    first = _capture(tmp_path / "code.json", prompt="code", category="code")
    second = _capture(tmp_path / "general.json", prompt="general", category="general_en")

    result = aggregate([first, second])

    assert result["schema"] == "hipengine.paro_mtp_verifier_review_matrix.v2"
    assert result["status"] == "inconclusive"
    assert result["policy_decision"] == "inconclusive"
    assert result["coverage"]["prompts"] == 2
    assert result["aggregate"]["top1_matches"] == 2
    assert result["aggregate"]["rows"] == 2
    assert result["qualification"]["status"] == "not_qualified"
    assert result["qualification"]["full_qualification"] is False
    assert result["qualification"]["automatic_admission"] is False
    assert result["qualification"]["policy_decision"] == "inconclusive"
    for name in _ENVELOPES:
        envelope = result["envelopes"][name]
        assert envelope["status"] == "inconclusive"
        assert envelope["policy_decision"] == "inconclusive"
        assert envelope["sample_limits_exceeded"] == []
    current = result["envelopes"]["current_sparse_moe_policy"]["observed_comparisons"]["global"]
    assert set(current) == {
        "median_kl",
        "p90_kl",
        "top1_agreement",
        "large_drift_rate",
    }
    for entry in current.values():
        assert entry["decides_policy"] is False
        assert entry["sample_within_limit"] is True
    assert result["checks"]
    assert all(result["checks"].values())
    assert result["review"]["policy_decision"] == "inconclusive"
    assert result["review"]["source_execution_failures"] == []
    assert result["sample_scope"]["full_protocol_reached"] is False
    assert "eligible_for_automatic_admission" not in result["review"]
    assert "automatic_admission_threshold_unchanged" not in result["review"]
    assert "automatic_admission_passed" not in json.dumps(result)


def test_aggregate_reports_both_named_envelopes_as_observations(tmp_path: Path) -> None:
    # The 2026-08-16 mean/p95 sample exceeds its limits while the binding 2.11
    # sample does not: both are reported by name and neither decides anything.
    rows = [_row(kl=1.0e-4)] * 4 + [_row(kl=9.0e-3)]
    capture = _capture(tmp_path / "code.json", prompt="code", category="code", rows=rows)

    result = aggregate([capture])

    current = result["envelopes"]["current_sparse_moe_policy"]
    legacy = result["envelopes"]["legacy_mean_tail_kl_envelope"]
    assert current["binding"] is True
    assert legacy["binding"] is False
    assert current["status"] == "inconclusive"
    assert legacy["status"] == "inconclusive"
    assert current["sample_limits_exceeded"] == []
    assert legacy["sample_limits_exceeded"] == ["mean_kl", "p95_kl"]
    assert legacy["observed_comparisons"]["mean_kl"]["sample_within_limit"] is False
    assert legacy["observed_comparisons"]["mean_kl"]["decides_policy"] is False
    assert result["envelope_disagreement"]["disagree"] is True
    assert result["review"]["current_sample_limits_exceeded"] == []
    assert result["review"]["legacy_sample_limits_exceeded"] == ["mean_kl", "p95_kl"]
    assert result["status"] == "inconclusive"
    assert result["policy_decision"] == "inconclusive"


def test_aggregate_propagates_source_execution_failures(tmp_path: Path) -> None:
    clean = _capture(tmp_path / "general.json", prompt="general", category="general_en")
    failed = _capture(
        tmp_path / "code.json",
        prompt="code",
        category="code",
        execution_failures=["finite_logits"],
        status="failed",
    )

    result = aggregate([clean, failed])

    assert result["status"] == "failed"
    assert result["review"]["source_execution_failures"] == [
        {"prompt": "code", "check": "finite_logits"}
    ]
    assert result["review"]["aggregation_failures"] == []
    assert result["qualification"]["execution_failure_count"] == 1
    assert result["prompts"]["code"]["execution_failures"] == ["finite_logits"]
    assert result["prompts"]["code"]["status"] == "failed"
    assert result["prompts"]["general"]["execution_failures"] == []
    # An execution failure is not a numerical observation: the numbers stay
    # reported, and the policy decision stays inconclusive.
    assert result["policy_decision"] == "inconclusive"
    assert result["review"]["policy_decision"] == "inconclusive"
    median = result["envelopes"]["current_sparse_moe_policy"]["observed_comparisons"][
        "global"
    ]["median_kl"]
    assert median["sample_within_limit"] is True
    assert median["decides_policy"] is False
    assert result["envelopes"]["current_sparse_moe_policy"]["sample_limits_exceeded"] == []


def test_legacy_source_status_is_not_read_as_an_execution_failure(tmp_path: Path) -> None:
    # A pre-v5 capture recorded one combined status. It must not be re-read as an
    # execution failure, and the report must say the distinction is unknown.
    capture = _capture(
        tmp_path / "legacy.json",
        prompt="code",
        category="code",
        status="rejected",
        schema="hipengine.paro_mtp_verifier_numerics.v4",
        with_execution_failures=False,
    )

    result = aggregate([capture])

    assert result["status"] == "inconclusive"
    assert result["review"]["source_execution_failures"] == []
    assert result["provenance"]["execution_failures_known_all"] is False
    assert result["provenance"]["schemas"] == ["hipengine.paro_mtp_verifier_numerics.v4"]
    assert result["prompts"]["code"]["execution_failures_known"] is False


def test_many_rows_do_not_imply_the_full_protocol_or_qualification(tmp_path: Path) -> None:
    rows = [_row(kl=1.0e-6) for _ in range(1600)]
    capture = _capture(tmp_path / "long.json", prompt="long", category="code", rows=rows)

    result = aggregate([capture])

    assert result["sample_scope"]["rows"] == 1600
    assert result["sample_scope"]["prompts"] == 1
    assert result["sample_scope"]["full_protocol_rows"] == 1548
    assert result["sample_scope"]["full_protocol_reached"] is False
    assert result["sample_scope"]["control_scopes"] == []
    assert result["sample_scope"]["screen_to_full_ratio_interval"] is None
    assert result["sample_scope"]["policy_decision"] == "inconclusive"
    assert result["qualification"]["full_protocol_reached"] is False
    assert result["qualification"]["full_qualification"] is False
    assert result["qualification"]["automatic_admission"] is False
    assert result["qualification"]["rate_gates_decided"] is False
    assert result["envelopes"]["current_sparse_moe_policy"]["rate_gate_policy"][
        "decided"
    ] is False
    assert result["status"] == "inconclusive"
    assert result["policy_decision"] == "inconclusive"


def test_aggregate_preserves_failed_category_and_task_review(tmp_path: Path) -> None:
    first = _capture(tmp_path / "code.json", prompt="code", category="code")
    second = _capture(
        tmp_path / "general.json",
        prompt="general",
        category="general_en",
        rows=[_row(category="general_en", top1_equal=False, decision_mismatch=True)],
    )

    result = aggregate([first, second])

    assert result["status"] == "inconclusive"
    assert result["coverage"]["prompts"] == 2
    assert result["coverage"]["categories"] == ["code", "general_en"]
    assert result["aggregate"]["top1_matches"] == 1
    assert result["aggregate"]["top1_mismatches"] == 1
    assert result["checks"]["top1"] is False
    assert result["checks"]["task_decision_proxy"] is False
    assert result["checks"]["per_scope"] is False
    assert result["scopes"]["category"]["code"]["passed"] is True
    assert result["scopes"]["category"]["general_en"]["passed"] is False
    assert result["review"]["top1_mismatch_rows"][0]["prompt"] == "general"
    assert len(result["review"]["task_decision_mismatches"]) == 1
    assert "global.top1_agreement" in result["envelopes"]["current_sparse_moe_policy"][
        "sample_limits_exceeded"
    ]
    assert "per_category.general_en.top1_agreement" in result["envelopes"][
        "current_sparse_moe_policy"
    ]["sample_limits_exceeded"]
    # A missed top-1 sample is an observation, not an admission verdict.
    assert result["policy_decision"] == "inconclusive"


def test_aggregate_records_source_provenance(tmp_path: Path) -> None:
    first = _capture(
        tmp_path / "code.json",
        prompt="code",
        category="code",
        capture_sha256="1" * 64,
    )
    second = _capture(
        tmp_path / "general.json",
        prompt="general",
        category="general_en",
        capture_sha256="2" * 64,
        route_attested=False,
        manifests_stable=False,
    )

    result = aggregate([first, second])
    provenance = result["provenance"]

    assert [entry["capture_sha256"] for entry in provenance["source_captures"]] == [
        "1" * 64,
        "2" * 64,
    ]
    assert provenance["schemas"] == ["hipengine.paro_mtp_verifier_numerics.v5"]
    assert provenance["mixed_schemas"] is False
    assert provenance["route_flags_attested_all"] is False
    assert provenance["manifests_stable_all"] is False
    assert provenance["execution_failures_known_all"] is True
    assert result["manifests"]["candidate_review_sha256"] == "b" * 64
    # A recorded route drift or manifest drift is an execution finding: it fails
    # status, and the numerical observations stay reported beside it.
    assert result["status"] == "failed"
    assert result["review"]["source_execution_failures"] == [
        {"prompt": "general", "check": "route_flags_attested"},
        {"prompt": "general", "check": "manifests_stable_during_capture"},
    ]
    assert result["policy_decision"] == "inconclusive"
    assert result["envelopes"]["current_sparse_moe_policy"]["sample_limits_exceeded"] == []


def test_aggregate_fails_on_an_inconsistent_source_row_count(tmp_path: Path) -> None:
    capture = _capture(tmp_path / "truncated.json", prompt="code", category="code")
    payload = json.loads(capture.read_text(encoding="utf-8"))
    payload["aggregate"]["rows"] = 9
    capture.write_text(json.dumps(payload), encoding="utf-8")

    result = aggregate([capture])

    assert result["status"] == "failed"
    assert result["review"]["aggregation_failures"] == ["source_row_count_mismatch"]
    assert result["review"]["source_execution_failures"] == []
    assert result["provenance"]["source_captures"][0]["row_count_consistent"] is False
    assert result["policy_decision"] == "inconclusive"


def test_aggregate_rejects_mixed_candidate_manifests(tmp_path: Path) -> None:
    first = _capture(tmp_path / "first.json", prompt="first", category="code")
    second = _capture(
        tmp_path / "second.json",
        prompt="second",
        category="general_en",
        candidate_review_sha256="9" * 64,
    )

    with pytest.raises(ValueError, match="candidate review manifest"):
        aggregate([first, second])


def test_aggregate_rejects_mixed_registered_capture_manifests(tmp_path: Path) -> None:
    first = _capture(tmp_path / "first.json", prompt="first", category="code")
    second = _capture(
        tmp_path / "second.json",
        prompt="second",
        category="general_en",
        capture_manifest_sha256="8" * 64,
    )

    with pytest.raises(ValueError, match="registered production capture manifest"):
        aggregate([first, second])


def test_aggregate_rejects_duplicate_prompt_captures(tmp_path: Path) -> None:
    first = _capture(tmp_path / "first.json", prompt="code", category="code")
    second = _capture(tmp_path / "second.json", prompt="code", category="code")

    with pytest.raises(ValueError, match="duplicate prompt capture"):
        aggregate([first, second])


def test_cli_exit_code_distinguishes_clean_from_execution_failed(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    clean = _capture(tmp_path / "code.json", prompt="code", category="code")
    out = tmp_path / "matrix.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["mtp_paro_verifier_review_aggregate.py", "--inputs", str(clean), "--out", str(out)],
    )

    assert main() == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "inconclusive"
    assert printed["policy_decision"] == "inconclusive"
    assert printed["full_qualification"] is False
    assert printed["automatic_admission"] is False
    assert printed["full_protocol_reached"] is False
    assert out.exists()

    failed = _capture(
        tmp_path / "failed.json",
        prompt="failed",
        category="general_en",
        execution_failures=["route_flags_attested"],
        status="failed",
    )
    out_failed = tmp_path / "matrix-failed.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mtp_paro_verifier_review_aggregate.py",
            "--inputs",
            str(failed),
            "--out",
            str(out_failed),
        ],
    )

    assert main() == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "failed"
    assert printed["execution_failure_count"] == 1
    assert printed["policy_decision"] == "inconclusive"
    assert out_failed.exists()


def test_repeat_aggregate_uses_one_quality_trajectory_and_checks_hashes(
    tmp_path: Path,
) -> None:
    paths = []
    for prompt, category in (("code", "code"), ("general", "general_en")):
        base = _capture(
            tmp_path / f"{prompt}-base.json",
            prompt=prompt,
            category=category,
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
    assert "numerical_and_repeat_gates_passed" not in result["review"]
    assert result["policy_decision"] == "inconclusive"
    assert result["qualification"]["automatic_admission"] is False
    assert result["status"] == "inconclusive"


def test_repeat_aggregate_rejects_capture_hash_drift(tmp_path: Path) -> None:
    paths = []
    for repeat in range(3):
        path = _capture(tmp_path / f"run{repeat + 1}.json", prompt="code", category="code")
        if repeat == 2:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["capture_sha256"] = "d" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    result = aggregate_repeats(paths, expected_repeats=3)

    assert result["repeat_determinism"]["passed"] is False
    assert result["repeat_determinism"]["failed_prompts"] == ["code"]
    assert result["status"] == "failed"


def test_repeat_aggregate_keeps_source_execution_failures_visible(tmp_path: Path) -> None:
    base = _capture(
        tmp_path / "base.json",
        prompt="code",
        category="code",
        execution_failures=["finite_logits"],
        status="failed",
    )
    payload = base.read_text(encoding="utf-8")
    paths = []
    for repeat in range(3):
        path = tmp_path / f"run{repeat + 1}.json"
        path.write_text(payload, encoding="utf-8")
        paths.append(path)

    result = aggregate_repeats(paths, expected_repeats=3)

    assert result["status"] == "failed"
    assert result["review"]["source_execution_failures"] == [
        {"prompt": "code", "check": "finite_logits"}
    ]


def test_aggregate_envelopes_agree_when_both_samples_exceed_limits(tmp_path: Path) -> None:
    path = _capture(tmp_path / "both.json", rows=[_row(top1_equal=False)])
    result = aggregate([path])
    assert result["envelope_disagreement"]["current_sample_limits_exceeded"]
    assert result["envelope_disagreement"]["legacy_sample_limits_exceeded"]
    assert result["envelope_disagreement"]["disagree"] is False


def test_repeat_numerical_observations_do_not_become_policy_verdicts(tmp_path: Path) -> None:
    paths = [_capture(tmp_path / f"run{i}.json", rows=[_row(kl=0.1)]) for i in range(3)]
    result = aggregate_repeats(paths)
    assert result["repeat_determinism"]["passed"] is True
    assert result["status"] == "inconclusive"
    assert result["policy_decision"] == "inconclusive"
    assert result["envelopes"]["current_sparse_moe_policy"]["sample_limits_exceeded"]


def test_repeat_execution_failure_in_noncanonical_capture_is_not_lost(tmp_path: Path) -> None:
    paths = [_capture(tmp_path / f"run{i}.json") for i in range(3)]
    _capture(paths[-1], execution_failures=["route_flags_attested"], status="failed")
    result = aggregate_repeats(paths)
    assert result["status"] == "failed"
    assert result["execution_checks"]["all_repeat_executions"] is False
    assert result["review"]["source_execution_failures"] == [
        {"prompt": "code", "check": "route_flags_attested"}
    ]
    assert result["qualification"]["execution_failure_count"] == 1


def test_repeat_rejects_duplicate_resolved_input_paths(tmp_path: Path) -> None:
    path = _capture(tmp_path / "run.json")
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="duplicate.*path"):
        aggregate_repeats([path, alias, path])


@pytest.mark.parametrize("bad_hash", [None, "", "xyz", "g" * 64])
def test_repeat_rejects_unusable_capture_hashes(tmp_path: Path, bad_hash) -> None:
    paths = [_capture(tmp_path / f"run{i}.json", capture_sha256=bad_hash) for i in range(3)]
    with pytest.raises(ValueError, match="SHA-256"):
        aggregate_repeats(paths)


def test_repeat_hash_case_does_not_change_the_digest(tmp_path: Path) -> None:
    paths = [_capture(tmp_path / f"run{i}.json", capture_sha256=("aB" * 32 if i else "Ab" * 32))
             for i in range(3)]
    assert aggregate_repeats(paths)["repeat_determinism"]["passed"] is True


def test_noncanonical_aggregation_failures_are_summarized(tmp_path: Path) -> None:
    paths = [_capture(tmp_path / f"run{i}.json") for i in range(3)]
    payload = json.loads(paths[-1].read_text())
    payload["aggregate"]["rows"] = 2
    paths[-1].write_text(json.dumps(payload))
    result = aggregate_repeats(paths)
    assert result["status"] == "failed"
    assert result["review"]["aggregation_failures"] == ["source_row_count_mismatch"]
    assert result["qualification"]["aggregation_failures"] == ["source_row_count_mismatch"]


def test_noncanonical_attestation_and_unknown_execution_summary_covers_all_repeats(tmp_path: Path):
    paths = [_capture(tmp_path / f"run{i}.json") for i in range(3)]
    _capture(paths[-1], route_attested=False, manifests_stable=False,
             with_execution_failures=False)
    result = aggregate_repeats(paths)
    assert result["status"] == "failed"
    assert result["provenance"]["route_flags_attested_all"] is False
    assert result["provenance"]["manifests_stable_all"] is False
    assert result["provenance"]["execution_failures_known_all"] is False
    assert result["qualification"]["execution_failure_count"] == 2
