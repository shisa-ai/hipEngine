from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_correctness_status import _stable_json_sha256, build_status
from scripts.stepfun_final_blocker_manifest import build_final_blocker_manifest
from scripts.stepfun_handoff_check import build_handoff_check, main
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
    summary_output = tmp_path / "handoff-summary.json"
    summary_sha_output = tmp_path / "handoff-summary-sha.json"
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
