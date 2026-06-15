from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_correctness_status import (
    SOURCE_ARTIFACT_MISMATCH_EXIT_CODE,
    _stable_json_sha256,
)
from scripts.stepfun_kv_blocker_status import (
    build_kv_blocker_status,
    main,
    verify_kv_blocker_status_artifact,
)
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
    wiring_map = artifact["runtime_wiring_map"]
    assert wiring_map["schema_version"] == 1
    assert wiring_map["source"] == "static_runtime_symbol_map"
    assert wiring_map["runner_file"] == "hipengine/runtime/stepfun_gguf_runner.py"
    assert wiring_map["planner_entrypoint"]["symbol"] == (
        "StepFunShortContextDecodePlanner.plan_kv_decode_chat"
    )
    assert wiring_map["resource_plan_entrypoint"]["symbol"] == (
        "StepFunTextDecodeResourcePlan.kv_decode_launch_schedule"
    )
    assert [entry["symbol"] for entry in wiring_map["device_input_entrypoints"]] == [
        "StepFunKVDecodeRunPlan.upload_decode_inputs",
        "StepFunKVDecodeRunPlan.decode_input_upload_plan",
    ]
    assert [entry["symbol"] for entry in wiring_map["metadata_only_trace_entrypoints"]] == [
        "StepFunKVDecodeRunPlan.streaming_decode_loop_blueprint",
        "StepFunKVDecodeRunPlan.streaming_decode_launch_trace",
        "StepFunKVDecodeRunPlan.streaming_decode_loop_status",
    ]
    assert wiring_map["current_host_composed_prompt_smoke"]["symbol"] == (
        "StepFunResidentSession.layer_prefix_prompt_logits_probe_bf16"
    )
    assert wiring_map["session_contract_entrypoint"]["symbol"] == (
        "StepFunResidentSession.kv_streaming_decode_contract"
    )
    assert wiring_map["missing_execution_entrypoint"]["owner"] == "StepFunResidentSession"
    assert wiring_map["missing_execution_entrypoint"]["expected_symbol"] == (
        "StepFunResidentSession.decode_one_token_kv_bf16"
    )
    assert wiring_map["missing_execution_entrypoint"]["expected_signature"] == (
        "decode_one_token_kv_bf16(run_plan: StepFunKVDecodeRunPlan, *, "
        "kv_cache: StepFunKVCacheAllocation, runtime: HipRuntime | None = None, "
        "stream: int = 0) -> dict[str, object]"
    )
    assert wiring_map["missing_execution_entrypoint"]["required_runtime_steps"] == [
        "validate the run_plan backend/layer count with kv_streaming_decode_contract",
        "upload input token IDs plus KVLiveSpans base_offsets/live_counts/token_positions",
        "launch prompt KV writes for every layer using gguf_step35 mixed_bf16_prompt_spans",
        "launch one-token decode KV writes for every layer using gguf_step35 mixed_bf16_spans",
        "launch gated paged decode attention for every layer using bf16_split_k_gate_f32_spans",
        "compute final next-token logits from resident output_norm/lm_head without host-composed layer-prefix outputs",
        "retain benchmarks/results/2026-05-31-stepfun-q3kl-kv-kernel-trace.json and benchmarks/results/2026-05-31-stepfun-q3kl-kv-backed-next-token.json",
    ]
    assert wiring_map["missing_execution_entrypoint"]["required_artifacts"] == [
        "benchmarks/results/2026-05-31-stepfun-q3kl-kv-kernel-trace.json",
        "benchmarks/results/2026-05-31-stepfun-q3kl-kv-backed-next-token.json",
    ]
    assert wiring_map["next_action"] == "wire_streaming_decode_loop"
    assert wiring_map["no_claim_policy"] == {
        "kv_backed_decode_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": "This is an entrypoint map, not an executable KV-backed decode run.",
    }
    validation = artifact["runtime_wiring_symbol_validation"]
    assert validation["schema_version"] == 1
    assert validation["source"] == "ast_symbol_validation"
    assert validation["file"] == "hipengine/runtime/stepfun_gguf_runner.py"
    assert validation["all_symbols_present"] is True
    assert validation["missing_symbols"] == []
    assert validation["symbol_count"] == 9
    assert {record["symbol"] for record in validation["symbols"]} == {
        "StepFunShortContextDecodePlanner.plan_kv_decode_chat",
        "StepFunTextDecodeResourcePlan.kv_decode_launch_schedule",
        "StepFunKVDecodeRunPlan.upload_decode_inputs",
        "StepFunKVDecodeRunPlan.decode_input_upload_plan",
        "StepFunKVDecodeRunPlan.streaming_decode_loop_blueprint",
        "StepFunKVDecodeRunPlan.streaming_decode_launch_trace",
        "StepFunKVDecodeRunPlan.streaming_decode_loop_status",
        "StepFunResidentSession.layer_prefix_prompt_logits_probe_bf16",
        "StepFunResidentSession.kv_streaming_decode_contract",
    }
    assert all(record["present"] is True for record in validation["symbols"])
    assert validation["missing_execution_owner"] == {
        "owner": "StepFunResidentSession",
        "class_present": True,
        "present": True,
    }
    assert validation["future_execution_entrypoint"]["symbol"] == (
        "StepFunResidentSession.decode_one_token_kv_bf16"
    )
    assert validation["future_execution_entrypoint"]["class_present"] is True
    assert validation["future_execution_entrypoint"]["method_present"] is True
    assert validation["future_execution_entrypoint"]["surface_present"] is True
    assert validation["future_execution_entrypoint"]["present"] is True
    assert validation["future_execution_entrypoint"]["executable"] is False
    assert validation["future_execution_entrypoint"]["ready"] is False
    assert validation["future_execution_entrypoint"]["blocked_by"] == (
        "streaming_decode_loop_not_wired"
    )
    assert validation["future_execution_entrypoint"][
        "expected_missing_until_streaming_loop_wired"
    ] is False
    assert validation["future_execution_entrypoint"]["required_runtime_steps"] == (
        wiring_map["missing_execution_entrypoint"]["required_runtime_steps"]
    )
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
    assert payload["runtime_wiring_map"]["missing_execution_entrypoint"]["owner"] == (
        "StepFunResidentSession"
    )
    assert payload["runtime_wiring_map"]["missing_execution_entrypoint"]["expected_symbol"] == (
        "StepFunResidentSession.decode_one_token_kv_bf16"
    )
    assert payload["runtime_wiring_map"]["next_action"] == "wire_streaming_decode_loop"
    assert payload["runtime_wiring_symbol_validation"]["all_symbols_present"] is True
    assert payload["runtime_wiring_symbol_validation"]["future_execution_entrypoint"][
        "surface_present"
    ] is True
    assert payload["runtime_wiring_symbol_validation"]["future_execution_entrypoint"][
        "executable"
    ] is False
    assert payload["runtime_wiring_symbol_validation"]["missing_symbols"] == []
    assert payload["missing_artifact_paths"] == [
        "benchmarks/results/2026-05-31-stepfun-q3kl-kv-kernel-trace.json",
        "benchmarks/results/2026-05-31-stepfun-q3kl-kv-backed-next-token.json",
    ]


def test_stepfun_kv_blocker_status_verifies_persisted_artifact(
    tmp_path: Path,
) -> None:
    prompt, oracle, docs, resource = _write_inputs(tmp_path)
    artifact_path = tmp_path / "kv-blocker.json"
    output = tmp_path / "verify.json"
    base_args = [
        "--prompt-artifact",
        str(prompt),
        "--oracle-artifact",
        str(oracle),
        "--docs",
        str(docs),
        "--resource-artifact",
        str(resource),
        "--artifact-date",
        "2030-01-04",
    ]
    current = build_kv_blocker_status(
        prompt_artifact=prompt,
        oracle_artifact=oracle,
        docs=docs,
        resource_artifact=resource,
        artifact_date="2030-01-04",
    )
    artifact_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")

    verification = verify_kv_blocker_status_artifact(
        artifact_path,
        current_artifact=current,
    )
    assert verification["status"] == "match"
    assert verification["all_match"] is True
    assert verification["verification_failures"] == []
    assert verification["persisted_artifact_sha256"] == _stable_json_sha256(current)
    assert verification["current_artifact_sha256"] == _stable_json_sha256(current)
    assert verification["persisted_status"] == "blocked"
    assert verification["current_status"] == "blocked"
    assert verification["persisted_blocked_count"] == 2
    assert verification["current_blocked_count"] == 2
    assert verification["persisted_missing_artifact_paths"] == current[
        "missing_artifact_paths"
    ]
    assert verification["current_missing_artifact_paths"] == current[
        "missing_artifact_paths"
    ]

    assert (
        main(
            [
                *base_args,
                "--verify-blocker-status",
                str(artifact_path),
                "--verification-status-only",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text()) == "match"

    drifted = dict(current)
    drifted["blocked_count"] = 1
    artifact_path.write_text(json.dumps(drifted, indent=2, sort_keys=True) + "\n")
    mismatch = verify_kv_blocker_status_artifact(
        artifact_path,
        current_artifact=current,
    )
    assert mismatch["status"] == "mismatch"
    assert mismatch["all_match"] is False
    assert mismatch["verification_failure_count"] == 1
    assert mismatch["verification_failures"][0]["name"] == "kv_blocker_status_drift"
    assert (
        main(
            [
                *base_args,
                "--verify-blocker-status",
                str(artifact_path),
                "--verification-failures-only",
                "--output",
                str(output),
            ]
        )
        == SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
    )
    assert json.loads(output.read_text())[0]["name"] == "kv_blocker_status_drift"
