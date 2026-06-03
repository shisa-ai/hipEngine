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
    assert manifest["status_provenance_sha256"] == _stable_json_sha256(provenance)
    assert manifest["recommended_commands_handoff_sha256"] == _stable_json_sha256(
        manifest["recommended_commands_handoff"]
    )
    assert [
        record["recommended_command_kind"]
        for record in manifest["recommended_commands_handoff"]
    ] == ["oracle_helper_long_timeout_command", "resource_plan_refresh_command"]
    assert manifest["recommended_commands_handoff"][0][
        "writes_partial_output_before_launch"
    ] is True
    assert manifest["recommended_commands_handoff"][0]["partial_output_status"] == (
        "running"
    )
    assert manifest["recommended_commands_handoff"][1]["resource_artifact"] == str(
        resource
    )
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
    assert manifest["artifact_status_handoff_sha256"] == _stable_json_sha256(
        manifest["artifact_status_handoff"]
    )
    assert manifest["missing_artifacts_handoff_sha256"] == _stable_json_sha256(
        manifest["missing_artifacts_handoff"]
    )
    assert manifest["all_required_artifacts_satisfied"] is False
    assert manifest["missing_artifact_count"] == 3
    assert [record["name"] for record in manifest["artifact_status_handoff"]] == [
        "llama_cpp_oracle_success_artifact",
        "kv_kernel_trace_artifact",
        "kv_backed_next_token_artifact",
    ]
    assert [record["name"] for record in manifest["missing_artifacts_handoff"]] == [
        "llama_cpp_oracle_success_artifact",
        "kv_kernel_trace_artifact",
        "kv_backed_next_token_artifact",
    ]
    assert manifest["artifact_status_handoff"][0]["artifact_file_present"] is True
    assert manifest["artifact_status_handoff"][0]["evidence_satisfied"] is False
    assert manifest["artifact_status_handoff"][0]["missing_reason"] == (
        "oracle_completed_successfully"
    )
    assert manifest["artifact_status_handoff"][1]["artifact_file_present"] is False
    assert manifest["artifact_status_handoff"][1]["readiness_gate"] == (
        "kv_backed_decode"
    )
    assert manifest["artifact_status_handoff"][1]["recommended_command_kind"] == (
        "resource_plan_refresh_command"
    )
    assert manifest["artifact_status_handoff"][1]["validator_command_kind"] == (
        "kv_trace_check_command"
    )
    assert "scripts/stepfun_kv_trace_check.py" in manifest[
        "artifact_status_handoff"
    ][1]["validator_command"]
    assert manifest["artifact_status_handoff"][1][
        "validator_command_sha256"
    ] == _stable_json_sha256(
        manifest["artifact_status_handoff"][1]["validator_command"]
    )
    assert [
        family["name"]
        for family in manifest["artifact_status_handoff"][1][
            "validator_expected_kernel_families"
        ]
    ] == [
        "prompt_kv_write",
        "decode_kv_write",
        "decode_attention_context",
        "decode_attention_gate_reduce",
    ]
    assert manifest["artifact_status_handoff"][1]["missing_reason"] == (
        "kv_kernel_trace_artifact_missing"
    )
    assert manifest["success_criteria_handoff_sha256"] == _stable_json_sha256(
        manifest["success_criteria_handoff"]
    )
    assert manifest["no_claim_policy_sha256"] == _stable_json_sha256(
        manifest["no_claim_policy"]
    )
    assert manifest["gate_status_handoff_sha256"] == _stable_json_sha256(
        manifest["gate_status_handoff"]
    )
    assert [gate["readiness_gate"] for gate in manifest["gate_status_handoff"]] == [
        "oracle_parity",
        "kv_backed_decode",
        "e2e_inference",
    ]
    assert manifest["gate_status_handoff"][0]["blocker_kind"] == (
        "oracle_parity_blocked"
    )
    assert manifest["gate_status_handoff"][1]["blocker_kind"] == (
        "kv_backed_decode_not_wired"
    )
    assert manifest["gate_status_handoff"][2] == {
        "readiness_gate": "e2e_inference",
        "ready": False,
        "blocked_by": ["oracle_parity", "kv_backed_decode"],
        "required_evidence": status["readiness_gates"]["e2e_inference"][
            "required_evidence"
        ],
        "blocker_kind": None,
        "first_missing_evidence": None,
        "success_criteria": [],
    }
    assert manifest["no_claim_policy"]["performance_claim_allowed"] is False
    assert manifest["no_claim_policy"]["e2e_inference_claim_allowed"] is False
    entries = {entry["blocker_kind"]: entry for entry in manifest["entries"]}
    success_criteria_by_blocker = {
        record["blocker_kind"]: record
        for record in manifest["success_criteria_handoff"]
    }
    assert success_criteria_by_blocker["oracle_parity_blocked"] == {
        "blocker_kind": "oracle_parity_blocked",
        "readiness_gate": "oracle_parity",
        "gate_ready": False,
        "first_missing_evidence": "oracle_completed_successfully",
        "success_criteria": entries["oracle_parity_blocked"]["success_criteria"],
        "recommended_command_kind": entries["oracle_parity_blocked"][
            "recommended_command_kind"
        ],
        "recommended_command_sha256": entries["oracle_parity_blocked"][
            "recommended_command_sha256"
        ],
    }
    assert success_criteria_by_blocker["kv_backed_decode_not_wired"] == {
        "blocker_kind": "kv_backed_decode_not_wired",
        "readiness_gate": "kv_backed_decode",
        "gate_ready": False,
        "first_missing_evidence": "streaming_runner_ready_flags",
        "success_criteria": entries["kv_backed_decode_not_wired"][
            "success_criteria"
        ],
        "recommended_command_kind": entries["kv_backed_decode_not_wired"][
            "recommended_command_kind"
        ],
        "recommended_command_sha256": entries["kv_backed_decode_not_wired"][
            "recommended_command_sha256"
        ],
    }
    assert manifest["compact_output_modes"] == {
        "sha_only": "manifest_sha256",
        "entries_only": "entries",
        "entries_sha_only": "entries_sha256",
        "artifacts_only": "artifacts_to_collect",
        "artifacts_sha_only": "artifacts_to_collect_sha256",
        "artifact_status_only": "artifact_status_handoff",
        "artifact_status_sha_only": "artifact_status_handoff_sha256",
        "missing_artifacts_only": "missing_artifacts_handoff",
        "missing_artifacts_sha_only": "missing_artifacts_handoff_sha256",
        "success_criteria_only": "success_criteria_handoff",
        "success_criteria_sha_only": "success_criteria_handoff_sha256",
        "no_claim_policy_only": "no_claim_policy",
        "no_claim_policy_sha_only": "no_claim_policy_sha256",
        "gate_status_only": "gate_status_handoff",
        "gate_status_sha_only": "gate_status_handoff_sha256",
        "status_provenance_only": "status_provenance",
        "status_provenance_sha_only": "status_provenance_sha256",
        "recommended_commands_only": "recommended_commands_handoff",
        "recommended_commands_sha_only": "recommended_commands_handoff_sha256",
        "verification_status_only": "verification.status",
        "verification_failures_only": "verification.verification_failures",
    }
    assert manifest["all_entries_have_success_criteria"] is True
    assert manifest["all_entries_have_recommended_commands"] is True
    assert manifest["no_claim_policy"] == status["remaining_blockers_report"][
        "no_claim_policy"
    ]

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
    assert kv_artifact["required_artifacts"][0]["validator_command_kind"] == (
        "kv_trace_check_command"
    )
    assert kv_artifact["required_artifacts"][0]["validator_success_status"] == (
        "passed"
    )
    assert kv_artifact["required_artifacts"][0]["validator_failure_exit_code"] == 2
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
    artifact_status_output = tmp_path / "final-blocker-artifact-status.json"
    artifact_status_sha_output = tmp_path / "final-blocker-artifact-status-sha.json"
    missing_artifacts_output = tmp_path / "final-blocker-missing-artifacts.json"
    missing_artifacts_sha_output = tmp_path / "final-blocker-missing-artifacts-sha.json"
    success_criteria_output = tmp_path / "final-blocker-success-criteria.json"
    success_criteria_sha_output = tmp_path / "final-blocker-success-criteria-sha.json"
    no_claim_output = tmp_path / "final-blocker-no-claim-policy.json"
    no_claim_sha_output = tmp_path / "final-blocker-no-claim-policy-sha.json"
    gate_status_output = tmp_path / "final-blocker-gate-status.json"
    gate_status_sha_output = tmp_path / "final-blocker-gate-status-sha.json"
    provenance_output = tmp_path / "final-blocker-status-provenance.json"
    provenance_sha_output = tmp_path / "final-blocker-status-provenance-sha.json"
    commands_output = tmp_path / "final-blocker-recommended-commands.json"
    commands_sha_output = tmp_path / "final-blocker-recommended-commands-sha.json"
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
            "--artifact-status-only",
            "--output",
            str(artifact_status_output),
            "--pretty",
        ]
    )
    assert rc == 0
    artifact_status_payload = json.loads(artifact_status_output.read_text())
    assert artifact_status_payload == expected["artifact_status_handoff"]
    assert artifact_status_payload[0]["artifact_file_present"] is True
    assert artifact_status_payload[0]["missing_reason"] == (
        "oracle_completed_successfully"
    )
    assert artifact_status_payload[1]["missing_reason"] == (
        "kv_kernel_trace_artifact_missing"
    )
    assert artifact_status_payload[1]["validator_command_kind"] == (
        "kv_trace_check_command"
    )
    assert "<kv_kernel_trace_artifact.csv-or-json>" in artifact_status_payload[1][
        "validator_command"
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
            "--artifact-status-sha-only",
            "--output",
            str(artifact_status_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(artifact_status_sha_output.read_text()) == expected[
        "artifact_status_handoff_sha256"
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
            "--missing-artifacts-only",
            "--output",
            str(missing_artifacts_output),
            "--pretty",
        ]
    )
    assert rc == 0
    missing_artifacts_payload = json.loads(missing_artifacts_output.read_text())
    assert missing_artifacts_payload == expected["missing_artifacts_handoff"]
    assert [record["name"] for record in missing_artifacts_payload] == [
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
            "--missing-artifacts-sha-only",
            "--output",
            str(missing_artifacts_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(missing_artifacts_sha_output.read_text()) == expected[
        "missing_artifacts_handoff_sha256"
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
            "--success-criteria-only",
            "--output",
            str(success_criteria_output),
            "--pretty",
        ]
    )
    assert rc == 0
    success_criteria_payload = json.loads(success_criteria_output.read_text())
    assert success_criteria_payload == expected["success_criteria_handoff"]
    assert [record["readiness_gate"] for record in success_criteria_payload] == [
        "oracle_parity",
        "kv_backed_decode",
    ]
    assert "oracle_parity is true" in success_criteria_payload[0]["success_criteria"]
    assert "kv_backed_decode_ready is true" in success_criteria_payload[1][
        "success_criteria"
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
            "--success-criteria-sha-only",
            "--output",
            str(success_criteria_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(success_criteria_sha_output.read_text()) == expected[
        "success_criteria_handoff_sha256"
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
            "--no-claim-policy-only",
            "--output",
            str(no_claim_output),
            "--pretty",
        ]
    )
    assert rc == 0
    no_claim_payload = json.loads(no_claim_output.read_text())
    assert no_claim_payload == expected["no_claim_policy"]
    assert no_claim_payload["performance_claim_allowed"] is False
    assert no_claim_payload["e2e_inference_claim_allowed"] is False

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
            "--no-claim-policy-sha-only",
            "--output",
            str(no_claim_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(no_claim_sha_output.read_text()) == expected[
        "no_claim_policy_sha256"
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
            "--gate-status-only",
            "--output",
            str(gate_status_output),
            "--pretty",
        ]
    )
    assert rc == 0
    gate_status_payload = json.loads(gate_status_output.read_text())
    assert gate_status_payload == expected["gate_status_handoff"]
    assert [gate["readiness_gate"] for gate in gate_status_payload] == [
        "oracle_parity",
        "kv_backed_decode",
        "e2e_inference",
    ]
    assert gate_status_payload[2]["blocked_by"] == [
        "oracle_parity",
        "kv_backed_decode",
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
            "--gate-status-sha-only",
            "--output",
            str(gate_status_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(gate_status_sha_output.read_text()) == expected[
        "gate_status_handoff_sha256"
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
            "--status-provenance-only",
            "--output",
            str(provenance_output),
            "--pretty",
        ]
    )
    assert rc == 0
    provenance_payload = json.loads(provenance_output.read_text())
    assert provenance_payload == expected["status_provenance"]
    assert provenance_payload["source_artifacts"]["prompt"]["path"] == str(prompt)
    assert provenance_payload["source_artifacts"]["oracle"]["path"] == str(oracle)
    assert provenance_payload["source_artifacts"]["text_resource"]["path"] == str(
        resource
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
            "--status-provenance-sha-only",
            "--output",
            str(provenance_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(provenance_sha_output.read_text()) == expected[
        "status_provenance_sha256"
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
            "--recommended-commands-only",
            "--output",
            str(commands_output),
            "--pretty",
        ]
    )
    assert rc == 0
    commands_payload = json.loads(commands_output.read_text())
    assert commands_payload == expected["recommended_commands_handoff"]
    assert [record["blocker_kind"] for record in commands_payload] == [
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    ]
    assert commands_payload[0]["recommended_command_kind"] == (
        "oracle_helper_long_timeout_command"
    )
    assert "stepfun_llamacpp_oracle.py" in commands_payload[0][
        "recommended_command"
    ]
    assert commands_payload[0]["writes_partial_output_before_launch"] is True
    assert commands_payload[1]["recommended_command_kind"] == (
        "resource_plan_refresh_command"
    )
    assert "stepfun_gguf_load_smoke.py" in commands_payload[1][
        "recommended_command"
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
            "--recommended-commands-sha-only",
            "--output",
            str(commands_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(commands_sha_output.read_text()) == expected[
        "recommended_commands_handoff_sha256"
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
