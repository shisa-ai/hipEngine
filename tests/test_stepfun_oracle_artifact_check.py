from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_oracle_artifact_check import build_oracle_check_report, main


def _write_prompt(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "prompt_length": 23,
                "next_token_id": 369,
                "next_token_text": " |",
                "next_token_logit": 19.158626556396484,
            },
            sort_keys=True,
        )
    )


def _oracle_artifact() -> dict[str, object]:
    return {
        "status": "executed",
        "returncode": 0,
        "llama_cli": "/tmp/llama-cli",
        "llama_cpp_version": "version: test (deadbeef)",
        "model": "/data/models/gguf/Step-3.7-flash-Q3_K_L-00001-of-00003.gguf",
        "command_shell": "/tmp/llama-cli --model stepfun.gguf --predict 1 --temp 0",
        "prompt_length": 23,
        "n_predict": 1,
        "expected_next_token_id": 369,
        "expected_next_token_text": " |",
        "expected_next_token_logit": 19.158626556396484,
        "expected_top_tokens": [
            {
                "rank": 1,
                "token_id": 369,
                "token_text": " |",
                "logit": 19.158626556396484,
            },
            {"rank": 2, "token_id": 5, "token_text": "#", "logit": 18.343582153320312},
        ],
        "stdout": " |",
        "stderr": "",
        "generated_text": " |",
        "text_matches_expected_exact": True,
        "text_matches_expected_stripped": True,
        "oracle_blocker_kind": None,
        "oracle_blocker_detail": None,
        "step35_supported": True,
        "timeout_termination": None,
    }


def test_stepfun_oracle_artifact_check_accepts_successful_oracle(
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    _write_prompt(prompt)
    oracle.write_text(json.dumps(_oracle_artifact(), sort_keys=True))

    report = build_oracle_check_report(oracle, prompt_artifact=prompt)

    assert report["status"] == "passed"
    summary = report["oracle_summary"]
    assert summary["ready"] is True
    assert summary["oracle_status"] == "executed"
    assert summary["oracle_returncode"] == 0
    assert summary["expected_next_token_id"] == 369
    assert summary["generated_text"] == " |"
    assert summary["text_matches_expected_exact"] is True
    assert summary["missing_evidence"] == []
    assert summary["no_claim_policy"] == {
        "llama_cpp_oracle_success_artifact_claim_allowed": True,
        "oracle_parity_claim_allowed": True,
        "kv_backed_decode_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": (
            "Oracle validation checks only the retained llama.cpp one-token artifact; "
            "KV-backed decode and benchmark gates remain separate."
        ),
    }
    assert {record["name"]: record["ready"] for record in report["evidence_checks"]} == {
        "oracle_success_status": True,
        "oracle_returncode_zero": True,
        "oracle_binary_metadata_recorded": True,
        "step35_supported_by_oracle": True,
        "no_timeout_or_oracle_blocker": True,
        "prompt_length_matches_target": True,
        "n_predict_one": True,
        "expected_token_metadata_matches_target": True,
        "top_token_metadata_matches_target": True,
        "top_token_logit_matches_target": True,
        "generated_text_nonempty": True,
        "generated_text_matches_target": True,
    }
    assert report["readiness_impact"] == {
        "llama_cpp_oracle_success_artifact": True,
        "oracle_parity": True,
        "kv_backed_decode_ready": False,
        "e2e_inference_ready": False,
        "reason": (
            "This report can satisfy only the oracle parity side of StepFun e2e readiness; "
            "KV trace/token evidence must also pass."
        ),
    }
    assert len(report["report_sha256"]) == 64


def test_stepfun_oracle_artifact_check_rejects_blocked_oracle(
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    _write_prompt(prompt)
    blocked = _oracle_artifact()
    blocked.update(
        {
            "status": "executed",
            "returncode": 1,
            "stdout": "",
            "generated_text": "",
            "stderr": "unknown model architecture: 'step35'",
            "text_matches_expected_exact": False,
            "text_matches_expected_stripped": False,
            "oracle_blocker_kind": "llama_cpp_missing_step35_architecture",
            "oracle_blocker_detail": "local llama.cpp build reports unknown model architecture: 'step35'",
            "step35_supported": False,
        }
    )
    oracle.write_text(json.dumps(blocked, sort_keys=True))

    report = build_oracle_check_report(oracle, prompt_artifact=prompt)

    assert report["status"] == "failed"
    summary = report["oracle_summary"]
    assert summary["ready"] is False
    assert summary["oracle_blocker_kind"] == "llama_cpp_missing_step35_architecture"
    assert summary["missing_evidence"] == [
        "oracle_returncode_zero",
        "step35_supported_by_oracle",
        "no_timeout_or_oracle_blocker",
        "generated_text_nonempty",
        "generated_text_matches_target",
    ]
    checks = {record["name"]: record for record in report["evidence_checks"]}
    assert checks["oracle_success_status"]["ready"] is True
    assert checks["oracle_returncode_zero"]["current"] == 1
    assert checks["step35_supported_by_oracle"]["current"] == {
        "step35_supported": False,
        "oracle_blocker_kind": "llama_cpp_missing_step35_architecture",
    }
    assert checks["generated_text_nonempty"]["current"] == {
        "generated_text_len": 0,
        "stdout_len": 0,
        "stderr_len": len("unknown model architecture: 'step35'"),
    }
    assert report["readiness_impact"]["oracle_parity"] is False


def test_stepfun_oracle_artifact_check_cli_compact_modes(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    _write_prompt(prompt)
    oracle.write_text(json.dumps(_oracle_artifact(), sort_keys=True))
    summary_output = tmp_path / "summary.json"
    sha_output = tmp_path / "sha.json"
    status_output = tmp_path / "status.json"

    rc = main(
        [
            "--artifact",
            str(oracle),
            "--prompt-artifact",
            str(prompt),
            "--summary-only",
            "--output",
            str(summary_output),
            "--pretty",
        ]
    )
    assert rc == 0
    summary = json.loads(summary_output.read_text())
    assert summary["status"] == "passed"
    assert summary["generated_text"] == " |"

    rc = main(
        [
            "--artifact",
            str(oracle),
            "--prompt-artifact",
            str(prompt),
            "--summary-only",
            "--sha-only",
            "--output",
            str(sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert len(json.loads(sha_output.read_text())) == 64

    bad = _oracle_artifact()
    bad["returncode"] = 1
    oracle.write_text(json.dumps(bad, sort_keys=True))
    rc = main(
        [
            "--artifact",
            str(oracle),
            "--prompt-artifact",
            str(prompt),
            "--status-only",
            "--fail-on-missing",
            "--output",
            str(status_output),
            "--pretty",
        ]
    )
    assert rc == 2
    assert json.loads(status_output.read_text()) == "failed"
