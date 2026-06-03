from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_correctness_status import _stable_json_sha256, build_status
from scripts.stepfun_final_blocker_manifest import build_final_blocker_manifest, main
from test_stepfun_correctness_status import (  # type: ignore[import-not-found]
    _write_docs,
    _write_oracle_artifact,
    _write_prompt_artifact,
    _write_resource_artifact,
)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    return prompt, oracle, docs, resource


def test_stepfun_final_blocker_manifest_joins_oracle_and_kv_evidence(
    tmp_path: Path,
) -> None:
    prompt, oracle, docs, resource = _write_inputs(tmp_path)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    manifest = build_final_blocker_manifest(status)

    assert manifest["schema_version"] == 1
    assert manifest["status"] == "blocked"
    provenance = manifest["status_provenance"]
    assert provenance["source_artifacts_sha256"] == status["source_artifacts_sha256"]
    assert provenance["status_integrity_sha256"] == status["status_integrity_sha256"]
    assert provenance["handoff_summary_sha256"] == status["handoff_summary_sha256"]
    assert provenance["readiness_summary_sha256"] == status["readiness_summary_sha256"]
    assert provenance["next_action_commands_sha256"] == status[
        "next_action_commands_sha256"
    ]
    assert provenance["source_artifacts"]["oracle"]["path"] == str(oracle)
    assert manifest["open_or_partial_items_p0_p12"] == 2
    assert manifest["remaining_blocker_count"] == 2
    assert manifest["remaining_blocker_kinds"] == [
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    ]
    assert manifest["blocked_gates"] == [
        "oracle_parity",
        "kv_backed_decode",
        "e2e_inference",
    ]
    assert manifest["entry_count"] == 2
    assert manifest["artifact_count"] == 3
    assert manifest["entries_sha256"] == _stable_json_sha256(manifest["entries"])
    assert manifest["artifacts_to_collect_sha256"] == _stable_json_sha256(
        manifest["artifacts_to_collect"]
    )
    assert manifest["compact_output_modes"] == {
        "sha_only": "manifest_sha256",
        "entries_only": "entries",
        "entries_sha_only": "entries_sha256",
        "artifacts_only": "artifacts_to_collect",
        "artifacts_sha_only": "artifacts_to_collect_sha256",
        "verification_status_only": "verification.status",
        "verification_failures_only": "verification.verification_failures",
    }
    assert manifest["all_entries_have_success_criteria"] is True
    assert manifest["all_entries_have_recommended_commands"] is True
    assert manifest["no_claim_policy"] == status["remaining_blockers_report"][
        "no_claim_policy"
    ]

    entries = {entry["blocker_kind"]: entry for entry in manifest["entries"]}
    oracle_entry = entries["oracle_parity_blocked"]
    assert oracle_entry["readiness_gate"] == "oracle_parity"
    assert oracle_entry["gate_ready"] is False
    assert oracle_entry["first_missing_evidence"] == "oracle_completed_successfully"
    assert "oracle_parity is true" in oracle_entry["success_criteria"]
    oracle_artifact = oracle_entry["artifact_handoff"]
    assert oracle_artifact == manifest["artifacts_to_collect"][0]
    assert oracle_artifact["name"] == "llama_cpp_oracle_success_artifact"
    assert oracle_artifact["path"] == str(oracle)
    assert oracle_artifact["current_blocker_kind"] == (
        "llama_cpp_missing_step35_architecture"
    )
    assert oracle_artifact["partial_output_handoff_safe"] is True

    kv_entry = entries["kv_backed_decode_not_wired"]
    assert kv_entry["readiness_gate"] == "kv_backed_decode"
    assert kv_entry["gate_ready"] is False
    assert kv_entry["first_missing_evidence"] == "streaming_runner_ready_flags"
    assert "kv_backed_decode_ready is true" in kv_entry["success_criteria"]
    kv_artifact = kv_entry["artifact_handoff"]
    assert kv_artifact["name"] == "kv_backed_decode_runtime_artifacts"
    assert kv_artifact["first_streaming_runner_blocker"] == (
        "streaming_decode_loop_not_wired"
    )
    assert kv_artifact["launch_trace_operation_count"] == 135
    assert kv_artifact["launch_trace_non_executable"] is True
    assert kv_artifact["required_artifact_names"] == [
        "kv_kernel_trace_artifact",
        "kv_backed_next_token_artifact",
    ]
    assert manifest["artifacts_to_collect"][1:] == kv_artifact["required_artifacts"]


def test_stepfun_final_blocker_manifest_cli_outputs_payload_and_sha(
    capsys,
    tmp_path: Path,
) -> None:
    prompt, oracle, docs, resource = _write_inputs(tmp_path)
    output = tmp_path / "final-blocker-manifest.json"
    sha_output = tmp_path / "final-blocker-manifest-sha.json"

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
            "--output",
            str(output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out == ""
    payload = json.loads(output.read_text())
    expected = build_final_blocker_manifest(
        build_status(prompt, oracle, docs, resource_artifact=resource)
    )
    assert payload == expected

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
            "--output",
            str(sha_output),
            "--sha-only",
            "--pretty",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(sha_output.read_text()) == _stable_json_sha256(expected)


def test_stepfun_final_blocker_manifest_cli_compact_outputs(
    capsys,
    tmp_path: Path,
) -> None:
    prompt, oracle, docs, resource = _write_inputs(tmp_path)
    entries_output = tmp_path / "final-blocker-entries.json"
    entries_sha_output = tmp_path / "final-blocker-entries-sha.json"
    artifacts_output = tmp_path / "final-blocker-artifacts.json"
    artifacts_sha_output = tmp_path / "final-blocker-artifacts-sha.json"
    expected = build_final_blocker_manifest(
        build_status(prompt, oracle, docs, resource_artifact=resource)
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
            "--entries-only",
            "--output",
            str(entries_output),
            "--pretty",
        ]
    )
    assert rc == 0
    entries_payload = json.loads(entries_output.read_text())
    assert entries_payload == expected["entries"]
    assert [entry["blocker_kind"] for entry in entries_payload] == [
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
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
            "--entries-sha-only",
            "--output",
            str(entries_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(entries_sha_output.read_text()) == expected["entries_sha256"]

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
            "--artifacts-only",
            "--output",
            str(artifacts_output),
            "--pretty",
        ]
    )
    assert rc == 0
    artifacts_payload = json.loads(artifacts_output.read_text())
    assert artifacts_payload == expected["artifacts_to_collect"]
    assert [artifact["name"] for artifact in artifacts_payload] == [
        "llama_cpp_oracle_success_artifact",
        "kv_kernel_trace_artifact",
        "kv_backed_next_token_artifact",
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
            "--artifacts-sha-only",
            "--output",
            str(artifacts_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(artifacts_sha_output.read_text()) == expected[
        "artifacts_to_collect_sha256"
    ]

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_stepfun_final_blocker_manifest_cli_verifies_persisted_manifest(
    capsys,
    tmp_path: Path,
) -> None:
    prompt, oracle, docs, resource = _write_inputs(tmp_path)
    output = tmp_path / "final-blocker-manifest.json"
    verification_output = tmp_path / "final-blocker-verification.json"
    status_output = tmp_path / "final-blocker-verification-status.json"
    failures_output = tmp_path / "final-blocker-verification-failures.json"

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
            "--output",
            str(output),
            "--pretty",
        ]
    )
    assert rc == 0

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
            "--verify-manifest",
            str(output),
            "--output",
            str(verification_output),
            "--pretty",
        ]
    )
    assert rc == 0
    verification = json.loads(verification_output.read_text())
    assert verification["schema_version"] == 1
    assert verification["status"] == "match"
    assert verification["all_match"] is True
    assert verification["verification_failures"] == []
    assert verification["persisted_manifest_sha256"] == _stable_json_sha256(
        json.loads(output.read_text())
    )
    assert verification["current_manifest_sha256"] == verification[
        "persisted_manifest_sha256"
    ]
    assert verification["current_status_provenance"] == verification[
        "persisted_status_provenance"
    ]

    persisted = json.loads(output.read_text())
    persisted["status"] = "stale"
    output.write_text(json.dumps(persisted, sort_keys=True) + "\n")
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
            "--verify-manifest",
            str(output),
            "--output",
            str(status_output),
            "--verification-status-only",
            "--pretty",
        ]
    )
    assert rc == 1
    assert json.loads(status_output.read_text()) == "mismatch"

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
            "--verify-manifest",
            str(output),
            "--output",
            str(failures_output),
            "--verification-failures-only",
            "--pretty",
        ]
    )
    assert rc == 1
    failures = json.loads(failures_output.read_text())
    assert failures == [
        {
            "actual_sha256": _stable_json_sha256(persisted),
            "evidence": (
                "Persisted final-blocker manifest differs from current "
                "prompt/oracle/resource/docs inputs."
            ),
            "expected_sha256": verification["current_manifest_sha256"],
            "name": "final_blocker_manifest_drift",
        }
    ]

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
