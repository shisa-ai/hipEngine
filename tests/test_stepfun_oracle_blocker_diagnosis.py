from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_oracle_blocker_diagnosis import build_oracle_blocker_diagnosis, main


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True))


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    backend_matrix = tmp_path / "backend-matrix.json"
    token_mismatch = tmp_path / "token-mismatch.json"
    rank_check = tmp_path / "rank-check.json"
    top_roundtrip = tmp_path / "top-roundtrip.json"
    prompt_roundtrip = tmp_path / "prompt-roundtrip.json"
    _write_json(
        backend_matrix,
        {
            "schema_version": 1,
            "artifact_kind": "stepfun_llamacpp_oracle_backend_matrix",
            "status": "blocked",
            "backend_outcomes": [
                {"backend": "vulkan", "outcome": "executed_token_mismatch"},
                {"backend": "hip", "outcome": "timeout"},
            ],
        },
    )
    _write_json(
        token_mismatch,
        {
            "schema_version": 1,
            "artifact_kind": "stepfun_llamacpp_oracle_token_mismatch",
            "status": "failed",
            "missing_evidence": ["generated_text_matches_target"],
            "expected_next_token_id": 369,
            "expected_next_token_text": " |",
            "generated_text": "The\n\n",
            "text_matches_expected_stripped": False,
            "tokenization_diagnostic": {
                "expected_next_token_text_single_token_matches_expected_id": True,
                "expected_next_token_text_token_ids": [369],
                "generated_text_token_ids": [671, 271],
                "generated_text_stripped_token_ids": [671],
            },
        },
    )
    _write_json(
        rank_check,
        {
            "schema_version": 1,
            "artifact_kind": "stepfun_llamacpp_oracle_rank_check",
            "status": "failed",
            "expected_next_token_id": 369,
            "expected_next_token_text": " |",
            "generated_first_token_id": 671,
            "generated_token_in_host_top_tokens": False,
            "generated_token_host_rank": None,
            "host_top_token_ids": [369, 5, 15251, 223, 201],
        },
    )
    _write_json(
        top_roundtrip,
        {
            "schema_version": 1,
            "artifact_kind": "stepfun_host_top_token_roundtrip",
            "status": "passed",
            "all_top_token_texts_roundtrip": True,
            "host_top_token_ids": [369, 5, 15251, 223, 201],
            "mismatch_count": 0,
        },
    )
    _write_json(
        prompt_roundtrip,
        {
            "schema_version": 1,
            "artifact_kind": "stepfun_prompt_token_roundtrip",
            "status": "passed",
            "prompt_tokenization_matches_host_input_ids": True,
            "host_input_id_count": 23,
            "llama_token_count": 23,
            "first_mismatch": None,
        },
    )
    return backend_matrix, token_mismatch, rank_check, top_roundtrip, prompt_roundtrip


def test_stepfun_oracle_blocker_diagnosis_summarizes_ruled_out_and_active_causes(
    tmp_path: Path,
) -> None:
    backend_matrix, token_mismatch, rank_check, top_roundtrip, prompt_roundtrip = _write_inputs(
        tmp_path
    )

    report = build_oracle_blocker_diagnosis(
        backend_matrix_artifact=backend_matrix,
        token_mismatch_artifact=token_mismatch,
        rank_check_artifact=rank_check,
        top_token_roundtrip_artifact=top_roundtrip,
        prompt_token_roundtrip_artifact=prompt_roundtrip,
        artifact_date="2030-01-17",
    )

    assert report["schema_version"] == 1
    assert report["artifact_kind"] == "stepfun_oracle_blocker_diagnosis"
    assert report["date"] == "2030-01-17"
    assert report["status"] == "blocked"
    assert report["ready"] is False
    assert report["oracle_parity_ready"] is False
    assert report["ruled_out_cause_count"] == 3
    assert report["all_tokenizer_prompt_drift_causes_ruled_out"] is True
    assert report["ruled_out_causes"] == [
        {
            "cause": "prompt_token_drift",
            "ruled_out": True,
            "evidence_artifact": str(prompt_roundtrip),
            "host_input_id_count": 23,
            "llama_token_count": 23,
            "first_mismatch": None,
        },
        {
            "cause": "host_top_token_text_label_drift",
            "ruled_out": True,
            "evidence_artifact": str(top_roundtrip),
            "host_top_token_ids": [369, 5, 15251, 223, 201],
            "mismatch_count": 0,
        },
        {
            "cause": "expected_next_token_text_tokenization_drift",
            "ruled_out": True,
            "evidence_artifact": str(token_mismatch),
            "expected_next_token_id": 369,
            "expected_next_token_text": " |",
            "expected_next_token_text_token_ids": [369],
        },
    ]
    assert report["active_blocker_count"] == 3
    generated_mismatch, rank_absent, hip_timeout = report["active_findings"]
    assert generated_mismatch["finding"] == "generated_text_mismatch"
    assert generated_mismatch["active"] is True
    assert generated_mismatch["missing_evidence"] == ["generated_text_matches_target"]
    assert generated_mismatch["expected_next_token_id"] == 369
    assert generated_mismatch["generated_first_token_id"] == 671
    assert generated_mismatch["generated_first_token_matches_expected_id"] is False
    assert rank_absent["finding"] == "generated_token_absent_from_host_top_tokens"
    assert rank_absent["active"] is True
    assert rank_absent["host_top_token_ids"] == [369, 5, 15251, 223, 201]
    assert rank_absent["generated_token_host_rank"] is None
    assert hip_timeout["finding"] == "hip_oracle_timeout"
    assert hip_timeout["active"] is True
    assert hip_timeout["backend_outcomes"] == [
        {"backend": "vulkan", "outcome": "executed_token_mismatch"},
        {"backend": "hip", "outcome": "timeout"},
    ]
    assert report["diagnosis"] == (
        "prompt/tokenizer drift ruled out; generated-text mismatch remains"
    )
    assert report["next_action"] == (
        "investigate logits/backend parity for the canonical Vulkan executed oracle; "
        "do not claim oracle parity until generated_text_matches_target passes"
    )
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
            "This diagnosis consolidates retained evidence and explicitly leaves "
            "generated_text_matches_target unresolved."
        ),
    }
    assert report["evidence_artifacts"]["rank_check"]["path"] == str(rank_check)
    assert report["evidence_artifacts"]["rank_check"]["status"] == "failed"


def test_stepfun_oracle_blocker_diagnosis_reports_unruled_prompt_drift(
    tmp_path: Path,
) -> None:
    backend_matrix, token_mismatch, rank_check, top_roundtrip, prompt_roundtrip = _write_inputs(
        tmp_path
    )
    prompt_payload = json.loads(prompt_roundtrip.read_text())
    prompt_payload.update(
        {
            "status": "failed",
            "prompt_tokenization_matches_host_input_ids": False,
            "first_mismatch": {
                "index": 4,
                "host_token_id": 12345,
                "llama_token_id": 77666,
                "kind": "token_id_mismatch",
            },
        }
    )
    _write_json(prompt_roundtrip, prompt_payload)

    report = build_oracle_blocker_diagnosis(
        backend_matrix_artifact=backend_matrix,
        token_mismatch_artifact=token_mismatch,
        rank_check_artifact=rank_check,
        top_token_roundtrip_artifact=top_roundtrip,
        prompt_token_roundtrip_artifact=prompt_roundtrip,
    )

    assert report["ruled_out_cause_count"] == 2
    assert report["all_tokenizer_prompt_drift_causes_ruled_out"] is False
    assert report["ruled_out_causes"][0]["cause"] == "prompt_token_drift"
    assert report["ruled_out_causes"][0]["ruled_out"] is False
    assert report["diagnosis"] == (
        "prompt/tokenizer drift not fully ruled out; generated-text mismatch remains"
    )


def test_stepfun_oracle_blocker_diagnosis_cli_writes_report(tmp_path: Path) -> None:
    backend_matrix, token_mismatch, rank_check, top_roundtrip, prompt_roundtrip = _write_inputs(
        tmp_path
    )
    output = tmp_path / "diagnosis.json"

    rc = main(
        [
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
            "2030-01-18",
            "--output",
            str(output),
            "--pretty",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["date"] == "2030-01-18"
    assert payload["status"] == "blocked"
    assert payload["all_tokenizer_prompt_drift_causes_ruled_out"] is True
    assert payload["active_blocker_count"] == 3


def test_stepfun_oracle_blocker_diagnosis_cli_compact_modes(tmp_path: Path) -> None:
    backend_matrix, token_mismatch, rank_check, top_roundtrip, prompt_roundtrip = _write_inputs(
        tmp_path
    )
    output = tmp_path / "compact.json"
    base_args = [
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
    assert json.loads(output.read_text()) == "blocked"
    assert main([*base_args, "--ruled-out-causes-only", "--output", str(output)]) == 0
    assert len(json.loads(output.read_text())) == 3
    assert main([*base_args, "--active-findings-only", "--output", str(output)]) == 0
    assert len(json.loads(output.read_text())) == 3
    assert main([*base_args, "--next-action-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()).startswith("investigate logits/backend parity")
