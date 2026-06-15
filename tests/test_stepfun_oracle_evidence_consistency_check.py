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
    main,
    verify_oracle_evidence_consistency_check,
)
from test_stepfun_oracle_blocker_diagnosis import _write_inputs, _write_json  # type: ignore[import-not-found]


def _write_diagnosis(
    path: Path,
    *,
    backend_matrix: Path,
    token_mismatch: Path,
    rank_check: Path,
    top_roundtrip: Path,
    prompt_roundtrip: Path,
) -> None:
    diagnosis = build_oracle_blocker_diagnosis(
        backend_matrix_artifact=backend_matrix,
        token_mismatch_artifact=token_mismatch,
        rank_check_artifact=rank_check,
        top_token_roundtrip_artifact=top_roundtrip,
        prompt_token_roundtrip_artifact=prompt_roundtrip,
        artifact_date="2030-01-19",
    )
    _write_json(path, diagnosis)


def _write_consistent_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    backend_matrix, token_mismatch, rank_check, top_roundtrip, prompt_roundtrip = _write_inputs(
        tmp_path
    )
    diagnosis = tmp_path / "diagnosis.json"
    _write_diagnosis(
        diagnosis,
        backend_matrix=backend_matrix,
        token_mismatch=token_mismatch,
        rank_check=rank_check,
        top_roundtrip=top_roundtrip,
        prompt_roundtrip=prompt_roundtrip,
    )
    return diagnosis, backend_matrix, token_mismatch, rank_check, top_roundtrip, prompt_roundtrip


def test_stepfun_oracle_evidence_consistency_check_accepts_consistent_bundle(
    tmp_path: Path,
) -> None:
    diagnosis, backend_matrix, token_mismatch, rank_check, top_roundtrip, prompt_roundtrip = (
        _write_consistent_inputs(tmp_path)
    )

    report = build_oracle_evidence_consistency_check(
        diagnosis_artifact=diagnosis,
        backend_matrix_artifact=backend_matrix,
        token_mismatch_artifact=token_mismatch,
        rank_check_artifact=rank_check,
        top_token_roundtrip_artifact=top_roundtrip,
        prompt_token_roundtrip_artifact=prompt_roundtrip,
        artifact_date="2030-01-20",
    )

    assert report["schema_version"] == 1
    assert report["artifact_kind"] == "stepfun_oracle_evidence_consistency_check"
    assert report["date"] == "2030-01-20"
    assert report["status"] == "match"
    assert report["ready"] is True
    assert report["oracle_parity_ready"] is False
    assert report["check_count"] == 16
    assert report["passed_check_count"] == 16
    assert report["inconsistency_count"] == 0
    assert report["inconsistencies"] == []
    assert all(check["passed"] is True for check in report["checks"])
    assert report["checked_artifacts"]["rank_check"]["path"] == str(rank_check)
    assert report["checked_artifacts"]["rank_check"]["status"] == "failed"
    assert report["blocked_gates"] == [
        "oracle_parity",
        "kv_backed_decode",
        "e2e_inference",
    ]
    assert report["next_action"] == (
        "investigate logits/backend parity for the canonical Vulkan executed oracle; "
        "do not claim oracle parity until generated_text_matches_target passes"
    )
    assert report["no_claim_policy"] == {
        "oracle_parity_claim_allowed": False,
        "kv_backed_decode_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": (
            "This artifact only checks retained oracle evidence consistency; "
            "generated_text_matches_target remains unresolved."
        ),
    }


def test_stepfun_oracle_evidence_consistency_check_reports_stale_rank_artifact(
    tmp_path: Path,
) -> None:
    diagnosis, backend_matrix, token_mismatch, rank_check, top_roundtrip, prompt_roundtrip = (
        _write_consistent_inputs(tmp_path)
    )
    rank_payload = json.loads(rank_check.read_text())
    rank_payload.update(
        {
            "generated_first_token_id": 999,
            "generated_token_in_host_top_tokens": True,
            "generated_token_host_rank": 6,
        }
    )
    _write_json(rank_check, rank_payload)

    report = build_oracle_evidence_consistency_check(
        diagnosis_artifact=diagnosis,
        backend_matrix_artifact=backend_matrix,
        token_mismatch_artifact=token_mismatch,
        rank_check_artifact=rank_check,
        top_token_roundtrip_artifact=top_roundtrip,
        prompt_token_roundtrip_artifact=prompt_roundtrip,
    )

    assert report["status"] == "mismatch"
    assert report["ready"] is False
    names = [item["name"] for item in report["inconsistencies"]]
    assert "diagnosis_rank_check_sha_matches_live_artifact" in names
    assert "generated_first_token_id_consistent" in names
    assert "generated_token_absent_from_host_top_tokens_consistent" in names
    assert report["inconsistency_count"] == 3


def test_stepfun_oracle_evidence_consistency_check_reports_top_id_mismatch(
    tmp_path: Path,
) -> None:
    diagnosis, backend_matrix, token_mismatch, rank_check, top_roundtrip, prompt_roundtrip = (
        _write_consistent_inputs(tmp_path)
    )
    top_payload = json.loads(top_roundtrip.read_text())
    top_payload["host_top_token_ids"] = [369, 5, 15251, 223, 999]
    _write_json(top_roundtrip, top_payload)

    report = build_oracle_evidence_consistency_check(
        diagnosis_artifact=diagnosis,
        backend_matrix_artifact=backend_matrix,
        token_mismatch_artifact=token_mismatch,
        rank_check_artifact=rank_check,
        top_token_roundtrip_artifact=top_roundtrip,
        prompt_token_roundtrip_artifact=prompt_roundtrip,
    )

    assert report["status"] == "mismatch"
    names = [item["name"] for item in report["inconsistencies"]]
    assert "diagnosis_top_token_roundtrip_sha_matches_live_artifact" in names
    assert "host_top_token_ids_consistent" in names


def test_stepfun_oracle_evidence_consistency_check_cli_writes_report(
    tmp_path: Path,
) -> None:
    diagnosis, backend_matrix, token_mismatch, rank_check, top_roundtrip, prompt_roundtrip = (
        _write_consistent_inputs(tmp_path)
    )
    output = tmp_path / "consistency.json"

    rc = main(
        [
            "--diagnosis-artifact",
            str(diagnosis),
            "--backend-matrix-artifact",
            str(backend_matrix),
            "--token-mismatch-artifact",
            str(token_mismatch),
            "--rank-check-artifact",
            str(rank_check),
            "--top-token-roundtrip-artifact",
            str(top_roundtrip),
            "--prompt-token-roundtrip-artifact",
            str(prompt_roundtrip),
            "--artifact-date",
            "2030-01-21",
            "--output",
            str(output),
            "--pretty",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["date"] == "2030-01-21"
    assert payload["status"] == "match"
    assert payload["inconsistency_count"] == 0


def test_stepfun_oracle_evidence_consistency_check_cli_compact_modes(
    tmp_path: Path,
) -> None:
    diagnosis, backend_matrix, token_mismatch, rank_check, top_roundtrip, prompt_roundtrip = (
        _write_consistent_inputs(tmp_path)
    )
    output = tmp_path / "compact.json"
    base_args = [
        "--diagnosis-artifact",
        str(diagnosis),
        "--backend-matrix-artifact",
        str(backend_matrix),
        "--token-mismatch-artifact",
        str(token_mismatch),
        "--rank-check-artifact",
        str(rank_check),
        "--top-token-roundtrip-artifact",
        str(top_roundtrip),
        "--prompt-token-roundtrip-artifact",
        str(prompt_roundtrip),
    ]

    assert main([*base_args, "--status-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == "match"
    assert main([*base_args, "--inconsistencies-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == []
    assert main([*base_args, "--checks-only", "--output", str(output)]) == 0
    assert len(json.loads(output.read_text())) == 16


def test_stepfun_oracle_evidence_consistency_check_verifies_persisted_artifact(
    tmp_path: Path,
) -> None:
    diagnosis, backend_matrix, token_mismatch, rank_check, top_roundtrip, prompt_roundtrip = (
        _write_consistent_inputs(tmp_path)
    )
    artifact = tmp_path / "consistency.json"
    output = tmp_path / "verify.json"
    base_args = [
        "--diagnosis-artifact",
        str(diagnosis),
        "--backend-matrix-artifact",
        str(backend_matrix),
        "--token-mismatch-artifact",
        str(token_mismatch),
        "--rank-check-artifact",
        str(rank_check),
        "--top-token-roundtrip-artifact",
        str(top_roundtrip),
        "--prompt-token-roundtrip-artifact",
        str(prompt_roundtrip),
        "--artifact-date",
        "2030-01-22",
    ]
    current = build_oracle_evidence_consistency_check(
        diagnosis_artifact=diagnosis,
        backend_matrix_artifact=backend_matrix,
        token_mismatch_artifact=token_mismatch,
        rank_check_artifact=rank_check,
        top_token_roundtrip_artifact=top_roundtrip,
        prompt_token_roundtrip_artifact=prompt_roundtrip,
        artifact_date="2030-01-22",
    )
    artifact.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")

    verification = verify_oracle_evidence_consistency_check(
        artifact,
        current_report=current,
    )
    assert verification["status"] == "match"
    assert verification["all_match"] is True
    assert verification["verification_failures"] == []
    assert verification["persisted_artifact_sha256"] == _stable_json_sha256(current)
    assert verification["current_artifact_sha256"] == _stable_json_sha256(current)
    assert verification["persisted_status"] == "match"
    assert verification["current_status"] == "match"
    assert verification["persisted_ready"] is True
    assert verification["current_ready"] is True
    assert verification["persisted_inconsistency_count"] == 0
    assert verification["current_inconsistency_count"] == 0

    assert (
        main(
            [
                *base_args,
                "--verify-consistency",
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
    drifted["ready"] = False
    artifact.write_text(json.dumps(drifted, indent=2, sort_keys=True) + "\n")
    mismatch = verify_oracle_evidence_consistency_check(
        artifact,
        current_report=current,
    )
    assert mismatch["status"] == "mismatch"
    assert mismatch["all_match"] is False
    assert mismatch["verification_failure_count"] == 1
    assert mismatch["verification_failures"][0]["name"] == (
        "oracle_evidence_consistency_drift"
    )
    assert (
        main(
            [
                *base_args,
                "--verify-consistency",
                str(artifact),
                "--verification-failures-only",
                "--output",
                str(output),
            ]
        )
        == SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
    )
    assert json.loads(output.read_text())[0]["name"] == (
        "oracle_evidence_consistency_drift"
    )
