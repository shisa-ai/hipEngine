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
