from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_kv_blocker_status import build_kv_blocker_status, main
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
    _write_docs(docs)
    _write_resource_artifact(resource)
    return prompt, oracle, docs, resource


def test_stepfun_kv_blocker_status_summarizes_missing_kv_artifacts(
    tmp_path: Path,
) -> None:
    prompt, oracle, docs, resource = _write_inputs(tmp_path)

    artifact = build_kv_blocker_status(
        prompt_artifact=prompt,
        oracle_artifact=oracle,
        docs=docs,
        resource_artifact=resource,
        artifact_date="2030-01-02",
    )

    assert artifact["schema_version"] == 1
    assert artifact["artifact_kind"] == "stepfun_kv_backed_decode_blocker_status"
    assert artifact["date"] == "2030-01-02"
    assert artifact["status"] == "blocked"
    assert artifact["readiness_gate"] == "kv_backed_decode"
    assert artifact["kv_decode_dispatch_ready"] is True
    assert artifact["kv_backed_decode_ready"] is False
    assert artifact["e2e_inference_ready"] is False
    assert artifact["blocked_count"] == 2
    assert artifact["blocked_artifact_names"] == [
        "kv_kernel_trace_artifact",
        "kv_backed_next_token_artifact",
    ]
    assert artifact["missing_artifact_paths"] == [
        "benchmarks/results/2026-05-31-stepfun-q3kl-kv-kernel-trace.json",
        "benchmarks/results/2026-05-31-stepfun-q3kl-kv-backed-next-token.json",
    ]
    assert artifact["validator_command_kinds"] == [
        "kv_trace_check_command",
        "kv_next_token_check_command",
    ]
    assert artifact["producer_command_kinds"] == ["resource_plan_refresh_command"]
    source_status = artifact["streaming_runner_source_status"]
    assert source_status["schema_version"] == 1
    assert source_status["kv_decode_run_plan_present"] is True
    assert source_status["source"] == "kv_decode_run_plan"
    assert source_status["ready"] is False
    assert source_status["executable"] is False
    assert source_status["blocked_by"] == "streaming_decode_loop_not_wired"
    assert source_status["next_action"] == "wire_streaming_decode_loop"
    assert source_status["blocker_count"] == 3
    assert source_status["blueprint_operation_count"] == 135
    assert source_status["blueprint_stage_count"] == 4
    assert source_status["kernel_trace_blocker_name"] == "kv_kernel_trace_artifact_missing"
    assert source_status["last_blocker_name"] == "kv_backed_next_token_artifact_missing"
    assert source_status["required_artifact_names"] == [
        "kv_kernel_trace_artifact",
        "kv_backed_next_token_artifact",
    ]
    assert source_status["no_kernel_launches"] is True
    assert artifact["no_claim_policy"] == {
        "kv_backed_decode_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": (
            "This artifact records missing KV evidence only; it does not contain "
            "a kernel trace or a KV-backed next-token result."
        ),
    }
    assert [record["status"] for record in artifact["blocked_records"]] == [
        "missing",
        "missing",
    ]
    assert [record["validator_missing_evidence"] for record in artifact["blocked_records"]] == [
        ["artifact_file_present"],
        ["artifact_file_present"],
    ]


def test_stepfun_kv_blocker_status_cli_writes_artifact(tmp_path: Path) -> None:
    prompt, oracle, docs, resource = _write_inputs(tmp_path)
    output = tmp_path / "kv-blocker.json"

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--docs",
            str(docs),
            "--resource-artifact",
            str(resource),
            "--artifact-date",
            "2030-01-03",
            "--output",
            str(output),
            "--pretty",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["date"] == "2030-01-03"
    assert payload["status"] == "blocked"
    assert payload["blocked_count"] == 2
    assert payload["streaming_runner_source_status"]["blocked_by"] == (
        "streaming_decode_loop_not_wired"
    )
    assert payload["streaming_runner_source_status"]["next_action"] == (
        "wire_streaming_decode_loop"
    )
    assert payload["missing_artifact_paths"] == [
        "benchmarks/results/2026-05-31-stepfun-q3kl-kv-kernel-trace.json",
        "benchmarks/results/2026-05-31-stepfun-q3kl-kv-backed-next-token.json",
    ]
