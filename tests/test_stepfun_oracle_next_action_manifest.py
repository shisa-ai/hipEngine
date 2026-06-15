from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_correctness_status import (
    SOURCE_ARTIFACT_MISMATCH_EXIT_CODE,
    _stable_json_sha256,
)
from scripts.stepfun_oracle_blocker_diagnosis import build_oracle_blocker_diagnosis
from scripts.stepfun_oracle_evidence_consistency_check import (
    build_oracle_evidence_consistency_check,
)
from scripts.stepfun_oracle_next_action_manifest import (
    build_oracle_next_action_manifest,
    main,
    verify_oracle_next_action_manifest,
)
from test_stepfun_oracle_blocker_diagnosis import _write_inputs, _write_json  # type: ignore[import-not-found]


def _write_host_logit_margin(path: Path) -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "artifact_kind": "stepfun_host_top_logit_margin",
            "status": "passed",
            "ready": True,
            "expected_next_token_id": 369,
            "expected_next_token_text": " |",
            "top1_matches_expected_id": True,
            "top1_to_top2_margin": 0.8150444030761719,
            "top1_to_top5_margin": 3.5056095123291016,
            "host_top_token_ids": [369, 5, 15251, 223, 201],
            "generated_token_context": {
                "generated_first_token_id": 671,
                "generated_token_in_host_top_tokens": False,
                "generated_token_host_rank": None,
            },
        },
    )


def _write_manifest_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    backend_matrix, token_mismatch, rank_check, top_roundtrip, prompt_roundtrip = _write_inputs(
        tmp_path
    )
    diagnosis = tmp_path / "diagnosis.json"
    consistency = tmp_path / "consistency.json"
    host_margin = tmp_path / "host-logit-margin.json"
    diagnosis_payload = build_oracle_blocker_diagnosis(
        backend_matrix_artifact=backend_matrix,
        token_mismatch_artifact=token_mismatch,
        rank_check_artifact=rank_check,
        top_token_roundtrip_artifact=top_roundtrip,
        prompt_token_roundtrip_artifact=prompt_roundtrip,
        artifact_date="2030-01-24",
    )
    _write_json(diagnosis, diagnosis_payload)
    consistency_payload = build_oracle_evidence_consistency_check(
        diagnosis_artifact=diagnosis,
        backend_matrix_artifact=backend_matrix,
        token_mismatch_artifact=token_mismatch,
        rank_check_artifact=rank_check,
        top_token_roundtrip_artifact=top_roundtrip,
        prompt_token_roundtrip_artifact=prompt_roundtrip,
        artifact_date="2030-01-24",
    )
    _write_json(consistency, consistency_payload)
    _write_host_logit_margin(host_margin)
    return diagnosis, consistency, host_margin


def test_stepfun_oracle_next_action_manifest_consolidates_investigation_handoff(
    tmp_path: Path,
) -> None:
    diagnosis, consistency, host_margin = _write_manifest_inputs(tmp_path)

    report = build_oracle_next_action_manifest(
        diagnosis_artifact=diagnosis,
        consistency_artifact=consistency,
        host_logit_margin_artifact=host_margin,
        artifact_date="2030-01-25",
    )

    assert report["schema_version"] == 1
    assert report["artifact_kind"] == "stepfun_oracle_next_action_manifest"
    assert report["date"] == "2030-01-25"
    assert report["status"] == "blocked"
    assert report["ready"] is False
    assert report["investigation_ready"] is True
    assert report["missing_preconditions"] == []
    assert report["next_action_kind"] == "logits_backend_parity_investigation"
    assert report["next_action"] == (
        "investigate logits/backend parity for the canonical Vulkan executed oracle; "
        "do not claim oracle parity until generated_text_matches_target passes"
    )
    assert report["target"] == {
        "canonical_backend": "vulkan",
        "readiness_gate": "oracle_parity",
        "unresolved_evidence_gap": "generated_text_matches_target",
        "expected_next_token_id": 369,
        "expected_next_token_text": " |",
        "generated_first_token_id": 671,
        "generated_text": "The\n\n",
        "generated_text_token_ids": [671, 271],
        "generated_text_stripped_token_ids": [671],
        "generated_first_token_matches_expected_id": False,
        "generated_token_in_host_top_tokens": False,
        "generated_token_host_rank": None,
        "host_top_token_ids": [369, 5, 15251, 223, 201],
        "host_top1_to_top2_margin": 0.8150444030761719,
        "host_top1_to_top5_margin": 3.5056095123291016,
    }
    assert report["ruled_out_causes"] == [
        "prompt_token_drift",
        "host_top_token_text_label_drift",
        "expected_next_token_text_tokenization_drift",
    ]
    assert report["active_findings"] == [
        "generated_text_mismatch",
        "generated_token_absent_from_host_top_tokens",
        "hip_oracle_timeout",
    ]
    assert report["comparison_only_findings"] == ["hip_oracle_timeout"]
    assert [item["name"] for item in report["preconditions"]] == [
        "oracle_diagnosis_status_blocked",
        "oracle_evidence_consistency_matches",
        "host_logit_margin_passes",
        "expected_token_is_host_top1",
        "generated_text_mismatch_is_active",
    ]
    assert all(item["passed"] is True for item in report["preconditions"])
    required_paths = {item["path"] for item in report["required_evidence_inputs"]}
    assert str(diagnosis) in required_paths
    assert str(consistency) in required_paths
    assert str(host_margin) in required_paths
    assert len(report["required_evidence_inputs"]) == 8
    assert report["acceptance_to_clear_oracle_parity"] == [
        "canonical generated_text_matches_target evidence passes",
        "oracle_parity_ready becomes true in stepfun_correctness_status",
        "retained oracle/status/handoff artifacts are refreshed and verify as match",
        "full StepFun guard passes",
        "WORKLOG and docs/STEPFUN.md record the new evidence without performance claims",
    ]
    assert report["non_goals_for_next_action"] == [
        "KV-backed decode wiring",
        "e2e generation readiness claim",
        "StepFun throughput or latency claim",
        "NVFP4, vision, or MTP work",
    ]
    assert report["blocked_gates"] == [
        "oracle_parity",
        "kv_backed_decode",
        "e2e_inference",
    ]
    assert report["no_claim_policy"] == {
        "oracle_parity_claim_allowed": False,
        "kv_backed_decode_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": (
            "This manifest describes the next investigation target; it does not "
            "resolve generated_text_matches_target."
        ),
    }


def test_stepfun_oracle_next_action_manifest_reports_missing_preconditions(
    tmp_path: Path,
) -> None:
    diagnosis, consistency, host_margin = _write_manifest_inputs(tmp_path)
    consistency_payload = json.loads(consistency.read_text())
    consistency_payload["status"] = "mismatch"
    _write_json(consistency, consistency_payload)
    margin_payload = json.loads(host_margin.read_text())
    margin_payload["top1_matches_expected_id"] = False
    _write_json(host_margin, margin_payload)

    report = build_oracle_next_action_manifest(
        diagnosis_artifact=diagnosis,
        consistency_artifact=consistency,
        host_logit_margin_artifact=host_margin,
    )

    assert report["status"] == "blocked"
    assert report["investigation_ready"] is False
    assert report["missing_preconditions"] == [
        "oracle_evidence_consistency_matches",
        "expected_token_is_host_top1",
    ]


def test_stepfun_oracle_next_action_manifest_cli_writes_report(tmp_path: Path) -> None:
    diagnosis, consistency, host_margin = _write_manifest_inputs(tmp_path)
    output = tmp_path / "next-action.json"

    rc = main(
        [
            "--diagnosis-artifact",
            str(diagnosis),
            "--consistency-artifact",
            str(consistency),
            "--host-logit-margin-artifact",
            str(host_margin),
            "--artifact-date",
            "2030-01-26",
            "--output",
            str(output),
            "--pretty",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["date"] == "2030-01-26"
    assert payload["status"] == "blocked"
    assert payload["investigation_ready"] is True
    assert payload["target"]["unresolved_evidence_gap"] == "generated_text_matches_target"


def test_stepfun_oracle_next_action_manifest_cli_compact_modes(tmp_path: Path) -> None:
    diagnosis, consistency, host_margin = _write_manifest_inputs(tmp_path)
    output = tmp_path / "compact.json"
    base_args = [
        "--diagnosis-artifact",
        str(diagnosis),
        "--consistency-artifact",
        str(consistency),
        "--host-logit-margin-artifact",
        str(host_margin),
    ]

    assert main([*base_args, "--status-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == "blocked"
    assert main([*base_args, "--investigation-ready-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) is True
    assert main([*base_args, "--next-action-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()).startswith("investigate logits/backend parity")
    assert main([*base_args, "--required-inputs-only", "--output", str(output)]) == 0
    assert len(json.loads(output.read_text())) == 8


def test_stepfun_oracle_next_action_manifest_verifies_persisted_artifact(
    tmp_path: Path,
) -> None:
    diagnosis, consistency, host_margin = _write_manifest_inputs(tmp_path)
    artifact = tmp_path / "next-action.json"
    output = tmp_path / "verify.json"
    base_args = [
        "--diagnosis-artifact",
        str(diagnosis),
        "--consistency-artifact",
        str(consistency),
        "--host-logit-margin-artifact",
        str(host_margin),
        "--artifact-date",
        "2030-01-27",
    ]
    current = build_oracle_next_action_manifest(
        diagnosis_artifact=diagnosis,
        consistency_artifact=consistency,
        host_logit_margin_artifact=host_margin,
        artifact_date="2030-01-27",
    )
    artifact.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")

    verification = verify_oracle_next_action_manifest(
        artifact,
        current_manifest=current,
    )
    assert verification["status"] == "match"
    assert verification["all_match"] is True
    assert verification["verification_failures"] == []
    assert verification["persisted_artifact_sha256"] == _stable_json_sha256(current)
    assert verification["current_artifact_sha256"] == _stable_json_sha256(current)
    assert verification["persisted_status"] == "blocked"
    assert verification["current_status"] == "blocked"
    assert verification["persisted_investigation_ready"] is True
    assert verification["current_investigation_ready"] is True
    assert verification["persisted_next_action"] == current["next_action"]
    assert verification["current_next_action"] == current["next_action"]

    assert (
        main(
            [
                *base_args,
                "--verify-manifest",
                str(artifact),
                "--verification-status-only",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text()) == "match"

    drifted = dict(current)
    drifted["investigation_ready"] = False
    artifact.write_text(json.dumps(drifted, indent=2, sort_keys=True) + "\n")
    mismatch = verify_oracle_next_action_manifest(
        artifact,
        current_manifest=current,
    )
    assert mismatch["status"] == "mismatch"
    assert mismatch["all_match"] is False
    assert mismatch["verification_failure_count"] == 1
    assert mismatch["verification_failures"][0]["name"] == (
        "oracle_next_action_manifest_drift"
    )
    assert (
        main(
            [
                *base_args,
                "--verify-manifest",
                str(artifact),
                "--verification-failures-only",
                "--output",
                str(output),
            ]
        )
        == SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
    )
    assert json.loads(output.read_text())[0]["name"] == (
        "oracle_next_action_manifest_drift"
    )
