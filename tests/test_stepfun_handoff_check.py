from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_correctness_status import _stable_json_sha256, build_status
from scripts.stepfun_final_blocker_manifest import build_final_blocker_manifest
from scripts.stepfun_handoff_check import build_handoff_check, main, verify_handoff_report
from test_stepfun_correctness_status import (  # type: ignore[import-not-found]
    _write_docs,
    _write_oracle_artifact,
    _write_prompt_artifact,
    _write_resource_artifact,
)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_artifact = tmp_path / "correctness-status.json"
    manifest_artifact = tmp_path / "final-blocker-manifest.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    manifest = build_final_blocker_manifest(status)
    status_artifact.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    manifest_artifact.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return prompt, oracle, docs, resource, status_artifact, manifest_artifact


def test_stepfun_handoff_check_reports_verified_blocked_state(tmp_path: Path) -> None:
    prompt, oracle, docs, resource, status_artifact, manifest_artifact = _write_inputs(
        tmp_path
    )

    report = build_handoff_check(
        prompt_artifact=prompt,
        oracle_artifact=oracle,
        resource_artifact=resource,
        docs=docs,
        status_artifact=status_artifact,
        manifest_artifact=manifest_artifact,
    )

    assert report["schema_version"] == 1
    assert report["status"] == "blocked_verified"
    assert report["verification_failures"] == []
    assert report["verification_failures_sha256"] == _stable_json_sha256([])
    assert report["summary_sha256"] == _stable_json_sha256(report["summary"])
    assert report["artifact_verification_sha256"] == _stable_json_sha256(
        report["artifact_verification"]
    )
    assert report["readiness_summary_sha256"] == _stable_json_sha256(
        report["readiness_summary"]
    )
    assert report["readiness_summary"]["status"] == "blocked"
    assert report["readiness_summary"]["oracle_parity"] is False
    assert report["readiness_summary"]["kv_backed_decode_ready"] is False
    assert report["readiness_summary"]["e2e_inference_ready"] is False
    assert report["final_blocker_manifest_summary_sha256"] == _stable_json_sha256(
        report["final_blocker_manifest_summary"]
    )
    assert report["exit_code_policy_sha256"] == _stable_json_sha256(
        report["exit_code_policy"]
    )
    assert report["exit_code_policy"] == {
        "schema_version": 1,
        "ready": 0,
        "mismatch": 1,
        "blocked_verified_with_fail_on_blocked": 2,
        "current_without_fail_on_blocked": 0,
        "current_with_fail_on_blocked": 2,
        "status": "blocked_verified",
        "verified": True,
        "ready_status": False,
        "fail_on_blocked_option": "--fail-on-blocked",
    }
    assert report["final_blocker_manifest_summary"] == {
        "remaining_blocker_count": 2,
        "remaining_blocker_kinds": [
            "oracle_parity_blocked",
            "kv_backed_decode_not_wired",
        ],
        "blocked_gates": [
            "oracle_parity",
            "kv_backed_decode",
            "e2e_inference",
        ],
        "no_claim_policy": build_final_blocker_manifest(
            build_status(prompt, oracle, docs, resource_artifact=resource)
        )["no_claim_policy"],
    }
    assert report["digest_summary_sha256"] == _stable_json_sha256(
        report["digest_summary"]
    )
    assert report["digest_summary"] == {
        "schema_version": 1,
        "status": "blocked_verified",
        "summary_sha256": report["summary_sha256"],
        "artifact_verification_sha256": report["artifact_verification_sha256"],
        "readiness_summary_sha256": report["readiness_summary_sha256"],
        "final_blocker_manifest_summary_sha256": report[
            "final_blocker_manifest_summary_sha256"
        ],
        "exit_code_policy_sha256": report["exit_code_policy_sha256"],
        "verification_failures_sha256": report["verification_failures_sha256"],
        "source_artifacts_sha256": report["summary"]["source_artifacts_sha256"],
        "manifest_sha256": report["summary"]["manifest_sha256"],
    }
    assert report["artifact_verification"] == {
        "schema_version": 1,
        "status": "match",
        "all_match": True,
        "correctness_status": {
            "artifact": str(status_artifact),
            "status": "match",
            "all_match": True,
            "source_artifacts_all_match": True,
            "checked_count": report["correctness_status_verification"][
                "checked_count"
            ],
            "verification_failures_sha256": report[
                "correctness_status_verification"
            ]["verification_failures_sha256"],
        },
        "final_blocker_manifest": {
            "artifact": str(manifest_artifact),
            "status": "match",
            "all_match": True,
            "persisted_manifest_sha256": report[
                "final_blocker_manifest_verification"
            ]["persisted_manifest_sha256"],
            "current_manifest_sha256": report[
                "final_blocker_manifest_verification"
            ]["current_manifest_sha256"],
            "verification_failure_count": 0,
        },
    }
    assert report["correctness_status_verification"]["status"] == "match"
    assert report["final_blocker_manifest_verification"]["status"] == "match"
    summary = report["summary"]
    assert summary == {
        "schema_version": 1,
        "status": "blocked_verified",
        "verified": True,
        "ready": False,
        "blocked_verified": True,
        "remaining_blocker_count": 2,
        "remaining_blocker_kinds": [
            "oracle_parity_blocked",
            "kv_backed_decode_not_wired",
        ],
        "blocked_gates": [
            "oracle_parity",
            "kv_backed_decode",
            "e2e_inference",
        ],
        "open_or_partial_items_p0_p12": 2,
        "status_artifact": str(status_artifact),
        "manifest_artifact": str(manifest_artifact),
        "source_artifacts_sha256": report["readiness_summary"][
            "source_artifacts_sha256"
        ],
        "manifest_sha256": report["final_blocker_manifest_verification"][
            "current_manifest_sha256"
        ],
        "verification_failure_count": 0,
    }
    assert report["final_blocker_manifest_summary"]["no_claim_policy"][
        "performance_claim_allowed"
    ] is False


def test_stepfun_handoff_check_cli_compact_outputs(capsys, tmp_path: Path) -> None:
    prompt, oracle, docs, resource, status_artifact, manifest_artifact = _write_inputs(
        tmp_path
    )
    full_output = tmp_path / "handoff-check.json"
    report_verification_output = tmp_path / "handoff-report-verification.json"
    report_verification_status_output = tmp_path / "handoff-report-verification-status.json"
    report_verification_failures_output = tmp_path / "handoff-report-verification-failures.json"
    summary_output = tmp_path / "handoff-summary.json"
    summary_sha_output = tmp_path / "handoff-summary-sha.json"
    artifact_verification_output = tmp_path / "handoff-artifact-verification.json"
    artifact_verification_sha_output = tmp_path / "handoff-artifact-verification-sha.json"
    readiness_summary_output = tmp_path / "handoff-readiness-summary.json"
    readiness_summary_sha_output = tmp_path / "handoff-readiness-summary-sha.json"
    final_blocker_summary_output = tmp_path / "handoff-final-blocker-summary.json"
    final_blocker_summary_sha_output = tmp_path / "handoff-final-blocker-summary-sha.json"
    exit_code_policy_output = tmp_path / "handoff-exit-code-policy.json"
    exit_code_policy_sha_output = tmp_path / "handoff-exit-code-policy-sha.json"
    digest_summary_output = tmp_path / "handoff-digest-summary.json"
    digest_summary_sha_output = tmp_path / "handoff-digest-summary-sha.json"
    status_output = tmp_path / "handoff-status.json"
    failures_output = tmp_path / "handoff-failures.json"
    failures_sha_output = tmp_path / "handoff-failures-sha.json"
    expected = build_handoff_check(
        prompt_artifact=prompt,
        oracle_artifact=oracle,
        resource_artifact=resource,
        docs=docs,
        status_artifact=status_artifact,
        manifest_artifact=manifest_artifact,
    )

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--output",
            str(full_output),
            "--pretty",
        ]
    )
    assert rc == 0
    full_payload = json.loads(full_output.read_text())
    assert full_payload == expected
    assert full_payload["status"] == "blocked_verified"
    assert full_payload["artifact_verification"]["status"] == "match"
    assert full_payload["verification_failures"] == []
    report_verification = verify_handoff_report(full_output, current_report=expected)
    assert report_verification["status"] == "match"
    assert report_verification["all_match"] is True
    assert report_verification["verification_failures"] == []
    assert report_verification["persisted_status"] == "blocked_verified"
    assert report_verification["current_status"] == "blocked_verified"

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--verify-handoff-report",
            str(full_output),
            "--output",
            str(report_verification_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(report_verification_output.read_text()) == report_verification

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--verify-handoff-report",
            str(full_output),
            "--report-verification-status-only",
            "--output",
            str(report_verification_status_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(report_verification_status_output.read_text()) == "match"

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--verify-handoff-report",
            str(full_output),
            "--report-verification-failures-only",
            "--output",
            str(report_verification_failures_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(report_verification_failures_output.read_text()) == []

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--summary-only",
            "--output",
            str(summary_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(summary_output.read_text()) == expected["summary"]

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--summary-sha-only",
            "--output",
            str(summary_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(summary_sha_output.read_text()) == expected["summary_sha256"]

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--artifact-verification-only",
            "--output",
            str(artifact_verification_output),
            "--pretty",
        ]
    )
    assert rc == 0
    artifact_verification_payload = json.loads(
        artifact_verification_output.read_text()
    )
    assert artifact_verification_payload == expected["artifact_verification"]
    assert artifact_verification_payload["status"] == "match"
    assert artifact_verification_payload["correctness_status"]["status"] == "match"
    assert artifact_verification_payload["final_blocker_manifest"]["status"] == "match"

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--artifact-verification-sha-only",
            "--output",
            str(artifact_verification_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(artifact_verification_sha_output.read_text()) == expected[
        "artifact_verification_sha256"
    ]

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--readiness-summary-only",
            "--output",
            str(readiness_summary_output),
            "--pretty",
        ]
    )
    assert rc == 0
    readiness_summary_payload = json.loads(readiness_summary_output.read_text())
    assert readiness_summary_payload == expected["readiness_summary"]
    assert readiness_summary_payload["status"] == "blocked"
    assert readiness_summary_payload["oracle_parity"] is False
    assert readiness_summary_payload["kv_backed_decode_ready"] is False
    assert readiness_summary_payload["e2e_inference_ready"] is False

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--readiness-summary-sha-only",
            "--output",
            str(readiness_summary_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(readiness_summary_sha_output.read_text()) == expected[
        "readiness_summary_sha256"
    ]

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--exit-code-policy-only",
            "--output",
            str(exit_code_policy_output),
            "--pretty",
        ]
    )
    assert rc == 0
    exit_code_policy_payload = json.loads(exit_code_policy_output.read_text())
    assert exit_code_policy_payload == expected["exit_code_policy"]
    assert exit_code_policy_payload["current_without_fail_on_blocked"] == 0
    assert exit_code_policy_payload["current_with_fail_on_blocked"] == 2

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--exit-code-policy-sha-only",
            "--output",
            str(exit_code_policy_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(exit_code_policy_sha_output.read_text()) == expected[
        "exit_code_policy_sha256"
    ]

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--final-blocker-summary-only",
            "--output",
            str(final_blocker_summary_output),
            "--pretty",
        ]
    )
    assert rc == 0
    final_blocker_summary_payload = json.loads(
        final_blocker_summary_output.read_text()
    )
    assert final_blocker_summary_payload == expected[
        "final_blocker_manifest_summary"
    ]
    assert final_blocker_summary_payload["remaining_blocker_kinds"] == [
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    ]
    assert final_blocker_summary_payload["no_claim_policy"][
        "performance_claim_allowed"
    ] is False

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--final-blocker-summary-sha-only",
            "--output",
            str(final_blocker_summary_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(final_blocker_summary_sha_output.read_text()) == expected[
        "final_blocker_manifest_summary_sha256"
    ]

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--digest-summary-only",
            "--output",
            str(digest_summary_output),
            "--pretty",
        ]
    )
    assert rc == 0
    digest_summary_payload = json.loads(digest_summary_output.read_text())
    assert digest_summary_payload == expected["digest_summary"]
    assert digest_summary_payload["status"] == "blocked_verified"
    assert digest_summary_payload["manifest_sha256"] == expected["summary"][
        "manifest_sha256"
    ]

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--digest-summary-sha-only",
            "--output",
            str(digest_summary_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(digest_summary_sha_output.read_text()) == expected[
        "digest_summary_sha256"
    ]

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--status-only",
            "--output",
            str(status_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(status_output.read_text()) == "blocked_verified"

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--failures-only",
            "--output",
            str(failures_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(failures_output.read_text()) == []

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--failures-sha-only",
            "--output",
            str(failures_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(failures_sha_output.read_text()) == expected[
        "verification_failures_sha256"
    ]

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--fail-on-blocked",
            "--status-only",
        ]
    )
    assert rc == 2
    assert json.loads(capsys.readouterr().out) == "blocked_verified"


def test_stepfun_handoff_check_verifies_default_persisted_report_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prompt, oracle, docs, resource, status_artifact, manifest_artifact = _write_inputs(
        tmp_path
    )
    default_report = tmp_path / "benchmarks/results/2026-05-31-stepfun-q3kl-handoff-check.json"
    default_report.parent.mkdir(parents=True)
    current_report = build_handoff_check(
        prompt_artifact=prompt,
        oracle_artifact=oracle,
        resource_artifact=resource,
        docs=docs,
        status_artifact=status_artifact,
        manifest_artifact=manifest_artifact,
    )
    default_report.write_text(json.dumps(current_report, indent=2, sort_keys=True) + "\n")
    status_output = tmp_path / "default-report-verification-status.json"
    monkeypatch.chdir(tmp_path)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--verify-handoff-report",
            "--report-verification-status-only",
            "--output",
            str(status_output),
            "--pretty",
        ]
    )

    assert rc == 0
    assert json.loads(status_output.read_text()) == "match"


def test_stepfun_handoff_check_reports_handoff_report_mismatch(
    capsys,
    tmp_path: Path,
) -> None:
    prompt, oracle, docs, resource, status_artifact, manifest_artifact = _write_inputs(
        tmp_path
    )
    full_output = tmp_path / "handoff-check.json"
    failures_output = tmp_path / "handoff-report-failures.json"
    current_report = build_handoff_check(
        prompt_artifact=prompt,
        oracle_artifact=oracle,
        resource_artifact=resource,
        docs=docs,
        status_artifact=status_artifact,
        manifest_artifact=manifest_artifact,
    )
    full_output.write_text(json.dumps(current_report, indent=2, sort_keys=True) + "\n")
    persisted = json.loads(full_output.read_text())
    persisted["status"] = "stale"
    full_output.write_text(json.dumps(persisted, indent=2, sort_keys=True) + "\n")

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--verify-handoff-report",
            str(full_output),
            "--report-verification-failures-only",
            "--output",
            str(failures_output),
            "--pretty",
        ]
    )

    assert rc == 1
    assert json.loads(failures_output.read_text()) == [
        {
            "name": "handoff_report_drift",
            "expected_sha256": _stable_json_sha256(current_report),
            "actual_sha256": _stable_json_sha256(persisted),
            "evidence": (
                "Persisted handoff-check report differs from current "
                "prompt/oracle/resource/docs/status/manifest inputs."
            ),
        }
    ]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_stepfun_handoff_check_reports_manifest_mismatch(
    capsys,
    tmp_path: Path,
) -> None:
    prompt, oracle, docs, resource, status_artifact, manifest_artifact = _write_inputs(
        tmp_path
    )
    persisted = json.loads(manifest_artifact.read_text())
    persisted["status"] = "stale"
    manifest_artifact.write_text(json.dumps(persisted, indent=2, sort_keys=True) + "\n")
    failures_output = tmp_path / "handoff-failures.json"

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-artifact",
            str(status_artifact),
            "--manifest-artifact",
            str(manifest_artifact),
            "--failures-only",
            "--output",
            str(failures_output),
            "--pretty",
        ]
    )

    assert rc == 1
    failures = json.loads(failures_output.read_text())
    assert failures == [
        {
            "name": "final_blocker_manifest_mismatch",
            "status": "mismatch",
            "verification_failures": [
                {
                    "actual_sha256": _stable_json_sha256(persisted),
                    "evidence": (
                        "Persisted final-blocker manifest differs from current "
                        "prompt/oracle/resource/docs inputs."
                    ),
                    "expected_sha256": build_handoff_check(
                        prompt_artifact=prompt,
                        oracle_artifact=oracle,
                        resource_artifact=resource,
                        docs=docs,
                        status_artifact=status_artifact,
                        manifest_artifact=manifest_artifact,
                    )["final_blocker_manifest_verification"]["current_manifest_sha256"],
                    "name": "final_blocker_manifest_drift",
                }
            ],
        }
    ]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
