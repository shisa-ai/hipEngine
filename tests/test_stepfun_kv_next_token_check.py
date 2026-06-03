from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_kv_next_token_check import build_next_token_check_report, main


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


def _valid_artifact() -> dict[str, object]:
    return {
        "status": "passed",
        "execution_path": "kv_backed_decode",
        "kv_backed_decode": True,
        "kv_cache_used": True,
        "streaming_runner_ready": True,
        "host_composed_layer_prefix": False,
        "prompt_length": 23,
        "next_token_id": 369,
        "next_token_text": " |",
        "next_token_logit": 19.158626556396484,
    }


def test_stepfun_kv_next_token_check_accepts_kv_backed_artifact(
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    artifact = tmp_path / "kv-next-token.json"
    _write_prompt(prompt)
    artifact.write_text(json.dumps(_valid_artifact(), sort_keys=True))

    report = build_next_token_check_report(artifact, prompt_artifact=prompt)

    assert report["status"] == "passed"
    summary = report["next_token_summary"]
    assert summary["ready"] is True
    assert summary["expected_next_token_id"] == 369
    assert summary["observed_next_token_id"] == 369
    assert summary["observed_next_token_text"] == " |"
    assert summary["execution_path"] == "kv_backed_decode"
    assert summary["kv_backed_decode"] is True
    assert summary["kv_cache_used"] is True
    assert summary["streaming_runner_ready"] is True
    assert summary["missing_evidence"] == []
    assert summary["no_claim_policy"] == {
        "kv_backed_next_token_artifact_claim_allowed": True,
        "kv_backed_decode_claim_allowed": False,
        "oracle_parity_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": (
            "KV next-token validation checks only the retained one-token artifact; "
            "KV trace, oracle parity, and benchmark gates remain separate."
        ),
    }
    assert {record["name"]: record["ready"] for record in report["evidence_checks"]} == {
        "artifact_success_status": True,
        "kv_backed_runtime_path": True,
        "streaming_runner_ready": True,
        "not_host_composed_layer_prefix": True,
        "prompt_length_matches_target": True,
        "next_token_id_matches_target": True,
        "next_token_text_matches_target": True,
        "next_token_logit_recorded_finite": True,
        "next_token_logit_within_tolerance": True,
    }
    assert report["readiness_impact"] == {
        "kv_backed_next_token_artifact": True,
        "kv_backed_decode_ready": False,
        "e2e_inference_ready": False,
        "reason": (
            "This report can satisfy only the retained KV-backed next-token artifact; "
            "the streaming loop readiness, KV trace artifact, and oracle parity must also pass."
        ),
    }
    assert len(report["report_sha256"]) == 64


def test_stepfun_kv_next_token_check_rejects_host_composed_or_wrong_token(
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    artifact = tmp_path / "kv-next-token.json"
    _write_prompt(prompt)
    bad = _valid_artifact()
    bad.update(
        {
            "status": "completed",
            "execution_path": "layer_prefix_prompt_logits_probe",
            "kv_backed_decode": False,
            "kv_cache_used": False,
            "streaming_runner_ready": False,
            "host_composed_layer_prefix": True,
            "next_token_id": 5,
            "next_token_text": "#",
            "next_token_logit": "nan",
        }
    )
    artifact.write_text(json.dumps(bad, sort_keys=True))

    report = build_next_token_check_report(artifact, prompt_artifact=prompt)

    assert report["status"] == "failed"
    summary = report["next_token_summary"]
    assert summary["ready"] is False
    assert summary["missing_evidence"] == [
        "kv_backed_runtime_path",
        "streaming_runner_ready",
        "not_host_composed_layer_prefix",
        "next_token_id_matches_target",
        "next_token_text_matches_target",
        "next_token_logit_recorded_finite",
        "next_token_logit_within_tolerance",
    ]
    checks = {record["name"]: record for record in report["evidence_checks"]}
    assert checks["artifact_success_status"]["ready"] is True
    assert checks["kv_backed_runtime_path"]["current"] == {
        "execution_path": "layer_prefix_prompt_logits_probe",
        "kv_backed_decode": False,
        "kv_cache_used": False,
    }
    assert checks["not_host_composed_layer_prefix"]["current"] == {
        "host_composed_layer_prefix": True,
        "host_composed": None,
        "layer_prefix_host_composed": None,
    }
    assert report["readiness_impact"]["kv_backed_next_token_artifact"] is False


def test_stepfun_kv_next_token_check_cli_compact_modes(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    artifact = tmp_path / "kv-next-token.json"
    _write_prompt(prompt)
    artifact.write_text(json.dumps(_valid_artifact(), sort_keys=True))
    summary_output = tmp_path / "summary.json"
    sha_output = tmp_path / "sha.json"
    status_output = tmp_path / "status.json"

    rc = main(
        [
            "--artifact",
            str(artifact),
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
    assert summary["observed_next_token_id"] == 369

    rc = main(
        [
            "--artifact",
            str(artifact),
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

    bad = _valid_artifact()
    bad["next_token_id"] = 5
    artifact.write_text(json.dumps(bad, sort_keys=True))
    rc = main(
        [
            "--artifact",
            str(artifact),
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
