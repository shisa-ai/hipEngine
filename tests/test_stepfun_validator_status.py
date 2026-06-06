from __future__ import annotations

import json
from pathlib import Path

from scripts import stepfun_correctness_status as status_mod
from scripts.stepfun_validator_status import build_validator_status_report, main

PROMPT_KV = "hipengine_qwen35_write_paged_kv_mixed_value_bf16_prompt_spans"
DECODE_KV = "hipengine_qwen35_write_paged_kv_mixed_value_bf16_spans"
ATTN_CONTEXT = "hipengine_qwen35_paged_full_attn_decode_split_k_context_bf16_spans"
ATTN_REDUCE = "hipengine_qwen35_paged_full_attn_decode_split_k_reduce_gate_f32"


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


def _write_resource(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "kv_decode_run_plan": {
                    "streaming_decode_launch_trace": {"layer_count": 1},
                    "kv_decode_launch_operation_count": 4,
                }
            },
            sort_keys=True,
        )
    )


def _write_oracle(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
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
                    }
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
            },
            sort_keys=True,
        )
    )


def _write_timeout_oracle(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "timeout",
                "returncode": None,
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
                    }
                ],
                "stdout": "",
                "stderr": "",
                "generated_text": "",
                "text_matches_expected_exact": False,
                "text_matches_expected_stripped": False,
                "oracle_blocker_kind": "llama_cpp_oracle_timeout",
                "oracle_blocker_detail": (
                    "llama.cpp oracle timed out before producing a comparable token"
                ),
                "step35_supported": True,
                "timeout_termination": {"timeout_reached": True},
            },
            sort_keys=True,
        )
    )


def _write_trace(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "kernel_names": [
                    PROMPT_KV,
                    DECODE_KV,
                    ATTN_CONTEXT,
                    ATTN_REDUCE,
                ]
            },
            sort_keys=True,
        )
    )


def _write_token(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
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
            },
            sort_keys=True,
        )
    )


def _manifest(oracle: Path, trace: Path, token: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "recommended_commands_handoff": [
            {
                "readiness_gate": "oracle_parity",
                "recommended_command_kind": "oracle_helper_long_timeout_command",
                "recommended_command": "python3 scripts/run_stepfun_oracle.py --long-timeout",
                "recommended_command_sha256": "oracle-producer-sha",
                "recommended_command_reason": "rerun llama.cpp oracle with long timeout",
                "writes_partial_output_before_launch": True,
                "partial_output_path": str(oracle),
                "partial_output_status": "running",
                "partial_output_overwrite_policy": "overwrite_on_execute_or_timeout",
                "partial_output_supervisor_signal_handoff_safe": True,
            },
            {
                "readiness_gate": "kv_backed_decode",
                "recommended_command_kind": "resource_plan_refresh_command",
                "recommended_command": "python3 scripts/refresh_stepfun_kv_artifacts.py",
                "recommended_command_sha256": "kv-producer-sha",
                "recommended_command_reason": "refresh KV trace and token artifacts",
            },
        ],
        "artifacts_to_collect": [
            {
                "name": "llama_cpp_oracle_success_artifact",
                "readiness_gate": "oracle_parity",
                "path": str(oracle),
                "partial_output_handoff_safe": True,
                "partial_output_supervisor_signal_handoff_safe": True,
                "partial_output_supervisor_signal_contract": {
                    "handled_signals": ["SIGTERM", "SIGINT"],
                    "handler_scope": "while_llama_cli_subprocess_is_running",
                    "cleanup_method": "os.killpg",
                    "cleanup_signal": "SIGKILL",
                    "cleanup_path": "supervisor_signal_killpg_then_communicate",
                    "timeout_status": "timeout",
                    "timeout_blocker_kind": "llama_cpp_oracle_timeout",
                    "timeout_termination_supervisor_signal_received": True,
                    "partial_output_overwrite_policy": "overwrite_on_execute_or_timeout",
                },
            }
        ],
        "validator_commands_handoff": [
            {
                "artifact_name": "llama_cpp_oracle_success_artifact",
                "readiness_gate": "oracle_parity",
                "required_for": "oracle_completed_successfully",
                "validator_command_kind": "oracle_artifact_check_command",
                "validator_artifact_path": str(oracle),
                "validator_command_concrete": f"python3 scripts/stepfun_oracle_artifact_check.py --artifact {oracle}",
                "validator_command_concrete_sha256": "oracle-command-sha",
            },
            {
                "artifact_name": "kv_kernel_trace_artifact",
                "readiness_gate": "kv_backed_decode",
                "required_for": "kv_kernel_trace_artifact_missing",
                "validator_command_kind": "kv_trace_check_command",
                "validator_artifact_path": str(trace),
                "validator_command_concrete": f"python3 scripts/stepfun_kv_trace_check.py --trace {trace}",
                "validator_command_concrete_sha256": "trace-command-sha",
            },
            {
                "artifact_name": "kv_backed_next_token_artifact",
                "readiness_gate": "kv_backed_decode",
                "required_for": "kv_backed_next_token_artifact_missing",
                "validator_command_kind": "kv_next_token_check_command",
                "validator_artifact_path": str(token),
                "validator_command_concrete": f"python3 scripts/stepfun_kv_next_token_check.py --artifact {token}",
                "validator_command_concrete_sha256": "token-command-sha",
            },
        ],
    }


def test_stepfun_validator_status_reports_all_passed(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    resource = tmp_path / "resource.json"
    oracle = tmp_path / "oracle.json"
    trace = tmp_path / "trace.json"
    token = tmp_path / "token.json"
    _write_prompt(prompt)
    _write_resource(resource)
    _write_oracle(oracle)
    _write_trace(trace)
    _write_token(token)

    report = build_validator_status_report(
        _manifest(oracle, trace, token),
        prompt_artifact=prompt,
        resource_artifact=resource,
    )

    assert report["status"] == "passed"
    summary = report["validator_status_summary"]
    assert summary["ready"] is True
    assert summary["validator_count"] == 3
    assert summary["passed_count"] == 3
    assert summary["missing_count"] == 0
    assert summary["failed_count"] == 0
    assert summary["validator_command_kinds"] == [
        "oracle_artifact_check_command",
        "kv_trace_check_command",
        "kv_next_token_check_command",
    ]
    assert summary["validator_results_sha256"] == report["validator_results_sha256"]
    assert summary["blocked_validator_results_sha256"] == report[
        "blocked_validator_results_sha256"
    ]
    assert summary["blocked_evidence_summary"] == []
    assert summary["blocked_evidence_summary_sha256"] == report[
        "blocked_evidence_summary_sha256"
    ]
    assert summary["blocked_evidence_by_gate"] == []
    assert summary["blocked_evidence_by_gate_sha256"] == report[
        "blocked_evidence_by_gate_sha256"
    ]
    assert summary["blocked_readiness_gates"] == []
    assert summary["blocked_readiness_gates_sha256"] == report[
        "blocked_readiness_gates_sha256"
    ]
    assert report["blocked_readiness_gates"] == []
    assert summary["next_blocked_gate"] is None
    assert summary["next_blocked_gate_readiness_gate"] is None
    assert summary["next_blocked_gate_sha256"] == report["next_blocked_gate_sha256"]
    assert summary["selected_blocked_gate_readiness_gate"] is None
    assert summary["selected_blocked_gate_found"] is False
    assert summary["selected_blocked_gate"] is None
    assert summary["selected_blocked_gate_sha256"] == report[
        "selected_blocked_gate_sha256"
    ]
    assert summary["selected_blocked_gate_artifact_names"] is None
    assert summary["selected_blocked_gate_artifact_count"] is None
    assert summary["selected_blocked_gate_artifact_names_sha256"] == report[
        "selected_blocked_gate_artifact_names_sha256"
    ]
    assert summary["selected_blocked_gate_blocked_count"] is None
    assert summary["selected_blocked_gate_status_counts"] is None
    assert summary["selected_blocked_gate_status_counts_sha256"] == report[
        "selected_blocked_gate_status_counts_sha256"
    ]
    assert summary["selected_blocked_gate_producer_commands"] is None
    assert summary["selected_blocked_gate_producer_command_count"] is None
    assert summary["selected_blocked_gate_producer_commands_sha256"] == report[
        "selected_blocked_gate_producer_commands_sha256"
    ]
    assert summary["selected_blocked_gate_validator_commands"] is None
    assert summary["selected_blocked_gate_validator_command_count"] is None
    assert summary["selected_blocked_gate_validator_commands_sha256"] == report[
        "selected_blocked_gate_validator_commands_sha256"
    ]
    assert summary["selected_blocked_gate_missing_evidence"] is None
    assert summary["selected_blocked_gate_missing_evidence_count"] is None
    assert summary["selected_blocked_gate_missing_evidence_sha256"] == report[
        "selected_blocked_gate_missing_evidence_sha256"
    ]
    assert report["selected_blocked_gate"] is None
    assert report["blocked_evidence_summary"] == []
    assert report["blocked_evidence_by_gate"] == []
    assert report["next_blocked_gate"] is None
    assert summary["next_blocker_artifact_name"] is None
    assert summary["next_blocker_readiness_gate"] is None
    assert summary["next_blocker_status"] is None
    assert summary["next_blocker_reason"] is None
    assert summary["next_blocker_command"] is None
    assert summary["next_blocker_command_kind"] is None
    assert summary["next_blocker_command_sha256"] == report[
        "next_blocker_command_sha256"
    ]
    assert summary["next_producer_command_kind"] is None
    assert summary["next_producer_command"] is None
    assert summary["next_producer_command_sha256"] == report[
        "next_producer_command_sha256"
    ]
    assert summary["next_action_available"] is False
    assert summary["next_action_available"] == report["next_action_available"]
    assert summary["next_action_artifact_name"] is None
    assert summary["next_action_artifact_name"] == report["next_action_artifact_name"]
    assert summary["next_action_readiness_gate"] is None
    assert summary["next_action_readiness_gate"] == report[
        "next_action_readiness_gate"
    ]
    assert summary["next_action_status"] is None
    assert summary["next_action_status"] == report["next_action_status"]
    assert summary["next_action_reason"] is None
    assert summary["next_action_reason"] == report["next_action_reason"]
    assert summary["next_action_missing_evidence_count"] == report[
        "next_action_missing_evidence_count"
    ]
    assert summary["next_action_first_missing_evidence"] is None
    assert summary["next_action_first_missing_evidence"] == report[
        "next_action_first_missing_evidence"
    ]
    assert summary["next_action_last_missing_evidence"] is None
    assert summary["next_action_last_missing_evidence"] == report[
        "next_action_last_missing_evidence"
    ]
    assert summary["next_action_artifact_file_present_missing"] is None
    assert summary["next_action_artifact_file_present_missing"] == report[
        "next_action_artifact_file_present_missing"
    ]
    assert summary["next_action_oracle_success_status_missing"] is None
    assert summary["next_action_oracle_success_status_missing"] == report[
        "next_action_oracle_success_status_missing"
    ]
    assert summary["next_action_generated_text_matches_target_missing"] is None
    assert summary["next_action_generated_text_matches_target_missing"] == report[
        "next_action_generated_text_matches_target_missing"
    ]
    assert summary["next_action_missing_evidence_sha256"] == report[
        "next_action_missing_evidence_sha256"
    ]
    assert summary["next_action_validator_command_kind"] is None
    assert summary["next_action_validator_command_kind"] == report[
        "next_action_validator_command_kind"
    ]
    assert summary["next_action_validator_command"] is None
    assert summary["next_action_validator_command"] == report[
        "next_action_validator_command"
    ]
    assert summary["next_action_validator_command_sha256"] == status_mod._stable_json_sha256(
        None
    )
    assert summary["next_action_validator_command_sha256"] == report[
        "next_action_validator_command_sha256"
    ]
    assert summary["next_action_producer_command_kind"] is None
    assert summary["next_action_producer_command_kind"] == report[
        "next_action_producer_command_kind"
    ]
    assert summary["next_action_producer_command"] is None
    assert summary["next_action_producer_command"] == report[
        "next_action_producer_command"
    ]
    assert summary["next_action_producer_command_sha256"] == status_mod._stable_json_sha256(
        None
    )
    assert summary["next_action_producer_command_sha256"] == report[
        "next_action_producer_command_sha256"
    ]
    assert summary["next_action_validator_summary_sha256"] == report[
        "next_action_validator_summary_sha256"
    ]
    assert summary["next_action_validator_summary_status"] is None
    assert summary["next_action_validator_summary_status"] == report[
        "next_action_validator_summary_status"
    ]
    assert summary["next_action_validator_summary_ready"] is None
    assert summary["next_action_validator_summary_ready"] == report[
        "next_action_validator_summary_ready"
    ]
    assert summary["next_action_validator_summary_oracle_status"] is None
    assert summary["next_action_validator_summary_oracle_status"] == report[
        "next_action_validator_summary_oracle_status"
    ]
    assert summary["next_action_validator_summary_oracle_blocker_kind"] is None
    assert summary["next_action_validator_summary_oracle_blocker_kind"] == report[
        "next_action_validator_summary_oracle_blocker_kind"
    ]
    assert summary["next_action_oracle_expected_token"] is None
    assert summary["next_action_oracle_expected_token"] == report[
        "next_action_oracle_expected_token"
    ]
    assert summary["next_action_oracle_expected_token_sha256"] == report[
        "next_action_oracle_expected_token_sha256"
    ]
    assert summary["next_action_expected_next_token_id"] is None
    assert summary["next_action_expected_next_token_id"] == report[
        "next_action_expected_next_token_id"
    ]
    assert summary["next_action_expected_next_token_text"] is None
    assert summary["next_action_expected_next_token_text"] == report[
        "next_action_expected_next_token_text"
    ]
    assert summary["next_action_expected_next_token_logit"] is None
    assert summary["next_action_expected_next_token_logit"] == report[
        "next_action_expected_next_token_logit"
    ]
    assert summary["next_action_oracle_generated_text"] is None
    assert summary["next_action_oracle_generated_text"] == report[
        "next_action_oracle_generated_text"
    ]
    assert summary["next_action_oracle_generated_text_sha256"] == report[
        "next_action_oracle_generated_text_sha256"
    ]
    assert summary["next_action_generated_text"] is None
    assert summary["next_action_generated_text"] == report[
        "next_action_generated_text"
    ]
    assert summary["next_action_generated_text_len"] is None
    assert summary["next_action_generated_text_len"] == report[
        "next_action_generated_text_len"
    ]
    assert summary["next_action_generated_text_matches_expected_exact"] is None
    assert summary["next_action_generated_text_matches_expected_exact"] == report[
        "next_action_generated_text_matches_expected_exact"
    ]
    assert summary["next_action_generated_text_matches_expected_stripped"] is None
    assert summary["next_action_generated_text_matches_expected_stripped"] == report[
        "next_action_generated_text_matches_expected_stripped"
    ]
    assert summary["next_action_sha256"] == report["next_action_sha256"]
    assert summary["next_blocker_sha256"] == report["next_blocker_sha256"]
    assert report["blocked_validator_results"] == []
    assert report["next_blocker"] is None
    assert report["next_blocker_artifact_name"] is None
    assert report["next_blocker_readiness_gate"] is None
    assert report["next_blocker_status"] is None
    assert report["next_blocker_reason"] is None
    assert report["next_blocker_command"] is None
    assert report["next_blocker_command_kind"] is None
    assert report["next_producer_command_kind"] is None
    assert report["next_producer_command"] is None
    assert report["next_action"] is None
    assert summary["no_claim_policy"]["validator_artifacts_passed"] is True
    assert summary["no_claim_policy"]["e2e_inference_claim_allowed"] is False
    assert [record["status"] for record in report["validator_results"]] == [
        "passed",
        "passed",
        "passed",
    ]
    assert report["readiness_impact"] == {
        "validator_artifacts_passed": True,
        "oracle_parity": False,
        "kv_backed_decode_ready": False,
        "e2e_inference_ready": False,
        "reason": (
            "This report is an aggregate validator execution status; it does not by itself "
            "mark StepFun e2e inference ready."
        ),
    }
    assert len(report["report_sha256"]) == 64


def test_stepfun_validator_status_reports_missing_artifact(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    resource = tmp_path / "resource.json"
    oracle = tmp_path / "oracle.json"
    trace = tmp_path / "missing-trace.json"
    token = tmp_path / "token.json"
    _write_prompt(prompt)
    _write_resource(resource)
    _write_oracle(oracle)
    _write_token(token)

    report = build_validator_status_report(
        _manifest(oracle, trace, token),
        prompt_artifact=prompt,
        resource_artifact=resource,
        selected_blocked_gate_name="kv_backed_decode",
    )

    assert report["status"] == "blocked"
    summary = report["validator_status_summary"]
    assert summary["ready"] is False
    assert summary["passed_count"] == 2
    assert summary["missing_count"] == 1
    assert summary["failed_count"] == 0
    results = {record["artifact_name"]: record for record in report["validator_results"]}
    expected_missing_trace = {
        "artifact_name": "kv_kernel_trace_artifact",
        "readiness_gate": "kv_backed_decode",
        "required_for": "kv_kernel_trace_artifact_missing",
        "validator_command_kind": "kv_trace_check_command",
        "validator_artifact_path": str(trace),
        "validator_command_concrete": f"python3 scripts/stepfun_kv_trace_check.py --trace {trace}",
        "validator_command_concrete_sha256": "trace-command-sha",
        "status": "missing",
        "ready": False,
        "reason": "artifact_file_missing",
        "validator_missing_evidence": ["artifact_file_present"],
        "validator_missing_evidence_count": 1,
        "producer_command_kind": "resource_plan_refresh_command",
        "producer_command": "python3 scripts/refresh_stepfun_kv_artifacts.py",
        "producer_command_sha256": "kv-producer-sha",
        "producer_command_reason": "refresh KV trace and token artifacts",
    }
    assert results["kv_kernel_trace_artifact"] == expected_missing_trace
    assert report["blocked_validator_results"] == [expected_missing_trace]
    expected_blocked_summary = [
        {
            "artifact_name": "kv_kernel_trace_artifact",
            "readiness_gate": "kv_backed_decode",
            "status": "missing",
            "reason": "artifact_file_missing",
            "validator_artifact_path": str(trace),
            "validator_command_kind": "kv_trace_check_command",
            "validator_command_sha256": "trace-command-sha",
            "producer_command_kind": "resource_plan_refresh_command",
            "producer_command_sha256": "kv-producer-sha",
            "missing_evidence": ["artifact_file_present"],
            "missing_evidence_count": 1,
            "validator_summary_sha256": None,
        }
    ]
    expected_by_gate = [
        {
            "readiness_gate": "kv_backed_decode",
            "artifact_names": ["kv_kernel_trace_artifact"],
            "blocked_count": 1,
            "status_counts": {"missing": 1},
            "missing_evidence": ["artifact_file_present"],
            "producer_command_kinds": ["resource_plan_refresh_command"],
            "producer_command_sha256s": ["kv-producer-sha"],
            "validator_command_kinds": ["kv_trace_check_command"],
            "validator_command_sha256s": ["trace-command-sha"],
            "missing_evidence_count": 1,
        }
    ]
    assert report["blocked_evidence_summary"] == expected_blocked_summary
    assert report["blocked_evidence_by_gate"] == expected_by_gate
    assert summary["blocked_evidence_summary"] == expected_blocked_summary
    assert summary["blocked_evidence_by_gate"] == expected_by_gate
    assert summary["blocked_evidence_summary_sha256"] == report[
        "blocked_evidence_summary_sha256"
    ]
    assert summary["blocked_evidence_by_gate_sha256"] == report[
        "blocked_evidence_by_gate_sha256"
    ]
    assert summary["blocked_readiness_gates"] == ["kv_backed_decode"]
    assert summary["blocked_readiness_gates_sha256"] == report[
        "blocked_readiness_gates_sha256"
    ]
    assert report["blocked_readiness_gates"] == ["kv_backed_decode"]
    assert report["next_blocked_gate"] == expected_by_gate[0]
    assert summary["next_blocked_gate"] == expected_by_gate[0]
    assert summary["next_blocked_gate_readiness_gate"] == "kv_backed_decode"
    assert summary["next_blocked_gate_sha256"] == report["next_blocked_gate_sha256"]
    assert summary["selected_blocked_gate_readiness_gate"] == "kv_backed_decode"
    assert summary["selected_blocked_gate_found"] is True
    assert summary["selected_blocked_gate"] == expected_by_gate[0]
    assert summary["selected_blocked_gate_sha256"] == report[
        "selected_blocked_gate_sha256"
    ]
    assert summary["selected_blocked_gate_artifact_names"] == [
        "kv_kernel_trace_artifact"
    ]
    assert summary["selected_blocked_gate_artifact_count"] == 1
    assert summary["selected_blocked_gate_artifact_names_sha256"] == report[
        "selected_blocked_gate_artifact_names_sha256"
    ]
    assert summary["selected_blocked_gate_blocked_count"] == 1
    assert summary["selected_blocked_gate_status_counts"] == {"missing": 1}
    assert summary["selected_blocked_gate_status_counts_sha256"] == report[
        "selected_blocked_gate_status_counts_sha256"
    ]
    assert summary["selected_blocked_gate_producer_commands"] == [
        "python3 scripts/refresh_stepfun_kv_artifacts.py"
    ]
    assert summary["selected_blocked_gate_producer_command_count"] == 1
    assert summary["selected_blocked_gate_producer_commands_sha256"] == report[
        "selected_blocked_gate_producer_commands_sha256"
    ]
    assert summary["selected_blocked_gate_validator_commands"] == [
        expected_missing_trace["validator_command_concrete"]
    ]
    assert summary["selected_blocked_gate_validator_command_count"] == 1
    assert summary["selected_blocked_gate_validator_commands_sha256"] == report[
        "selected_blocked_gate_validator_commands_sha256"
    ]
    assert summary["selected_blocked_gate_missing_evidence"] == [
        "artifact_file_present"
    ]
    assert summary["selected_blocked_gate_missing_evidence_count"] == 1
    assert summary["selected_blocked_gate_missing_evidence_sha256"] == report[
        "selected_blocked_gate_missing_evidence_sha256"
    ]
    assert report["selected_blocked_gate"] == expected_by_gate[0]
    assert report["next_blocker"] == expected_missing_trace
    assert summary["next_blocker_artifact_name"] == "kv_kernel_trace_artifact"
    assert summary["next_blocker_readiness_gate"] == "kv_backed_decode"
    assert summary["next_blocker_status"] == "missing"
    assert summary["next_blocker_reason"] == "artifact_file_missing"
    assert summary["next_blocker_command"] == expected_missing_trace[
        "validator_command_concrete"
    ]
    assert summary["next_blocker_command_kind"] == "kv_trace_check_command"
    assert summary["next_blocker_command_sha256"] == report[
        "next_blocker_command_sha256"
    ]
    assert summary["next_producer_command_kind"] == "resource_plan_refresh_command"
    assert summary["next_producer_command"] == (
        "python3 scripts/refresh_stepfun_kv_artifacts.py"
    )
    assert summary["next_producer_command_sha256"] == report[
        "next_producer_command_sha256"
    ]
    assert report["next_blocker_artifact_name"] == "kv_kernel_trace_artifact"
    assert report["next_blocker_readiness_gate"] == "kv_backed_decode"
    assert report["next_blocker_status"] == "missing"
    assert report["next_blocker_reason"] == "artifact_file_missing"
    assert report["next_blocker_command"] == expected_missing_trace[
        "validator_command_concrete"
    ]
    assert report["next_blocker_command_kind"] == "kv_trace_check_command"
    assert report["next_producer_command_kind"] == "resource_plan_refresh_command"
    assert report["next_producer_command"] == (
        "python3 scripts/refresh_stepfun_kv_artifacts.py"
    )
    expected_next_action = {
        "artifact_name": "kv_kernel_trace_artifact",
        "readiness_gate": "kv_backed_decode",
        "status": "missing",
        "reason": "artifact_file_missing",
        "validator_command_kind": "kv_trace_check_command",
        "validator_command": expected_missing_trace["validator_command_concrete"],
        "validator_command_sha256": report["next_blocker_command_sha256"],
        "producer_command_kind": "resource_plan_refresh_command",
        "producer_command": "python3 scripts/refresh_stepfun_kv_artifacts.py",
        "producer_command_sha256": report["next_producer_command_sha256"],
        "validator_artifact_path": str(trace),
        "validator_missing_evidence": ["artifact_file_present"],
        "validator_missing_evidence_count": 1,
    }
    assert report["next_action"] == expected_next_action
    assert summary["next_action_available"] is True
    assert summary["next_action_available"] == report["next_action_available"]
    assert summary["next_action_artifact_name"] == "kv_kernel_trace_artifact"
    assert summary["next_action_artifact_name"] == report["next_action_artifact_name"]
    assert summary["next_action_readiness_gate"] == "kv_backed_decode"
    assert summary["next_action_readiness_gate"] == report[
        "next_action_readiness_gate"
    ]
    assert summary["next_action_status"] == "missing"
    assert summary["next_action_status"] == report["next_action_status"]
    assert summary["next_action_reason"] == "artifact_file_missing"
    assert summary["next_action_reason"] == report["next_action_reason"]
    assert summary["next_action_missing_evidence_count"] == report[
        "next_action_missing_evidence_count"
    ]
    assert summary["next_action_first_missing_evidence"] == "artifact_file_present"
    assert summary["next_action_first_missing_evidence"] == report[
        "next_action_first_missing_evidence"
    ]
    assert summary["next_action_last_missing_evidence"] == "artifact_file_present"
    assert summary["next_action_last_missing_evidence"] == report[
        "next_action_last_missing_evidence"
    ]
    assert summary["next_action_artifact_file_present_missing"] is True
    assert summary["next_action_artifact_file_present_missing"] == report[
        "next_action_artifact_file_present_missing"
    ]
    assert summary["next_action_oracle_success_status_missing"] is False
    assert summary["next_action_oracle_success_status_missing"] == report[
        "next_action_oracle_success_status_missing"
    ]
    assert summary["next_action_generated_text_matches_target_missing"] is False
    assert summary["next_action_generated_text_matches_target_missing"] == report[
        "next_action_generated_text_matches_target_missing"
    ]
    assert summary["next_action_missing_evidence_sha256"] == report[
        "next_action_missing_evidence_sha256"
    ]
    assert summary["next_action_validator_command_kind"] == "kv_trace_check_command"
    assert summary["next_action_validator_command_kind"] == report[
        "next_action_validator_command_kind"
    ]
    assert summary["next_action_validator_command"] == expected_missing_trace[
        "validator_command_concrete"
    ]
    assert summary["next_action_validator_command"] == report[
        "next_action_validator_command"
    ]
    assert summary["next_action_validator_command_sha256"] == status_mod._stable_json_sha256(
        expected_missing_trace["validator_command_concrete"]
    )
    assert summary["next_action_validator_command_sha256"] == report[
        "next_action_validator_command_sha256"
    ]
    assert summary["next_action_producer_command_kind"] == "resource_plan_refresh_command"
    assert summary["next_action_producer_command_kind"] == report[
        "next_action_producer_command_kind"
    ]
    assert summary["next_action_producer_command"] == (
        "python3 scripts/refresh_stepfun_kv_artifacts.py"
    )
    assert summary["next_action_producer_command"] == report[
        "next_action_producer_command"
    ]
    assert summary["next_action_producer_command_sha256"] == status_mod._stable_json_sha256(
        "python3 scripts/refresh_stepfun_kv_artifacts.py"
    )
    assert summary["next_action_producer_command_sha256"] == report[
        "next_action_producer_command_sha256"
    ]
    assert summary["next_action_validator_summary_sha256"] == report[
        "next_action_validator_summary_sha256"
    ]
    assert summary["next_action_validator_summary_status"] is None
    assert summary["next_action_validator_summary_status"] == report[
        "next_action_validator_summary_status"
    ]
    assert summary["next_action_validator_summary_ready"] is None
    assert summary["next_action_validator_summary_ready"] == report[
        "next_action_validator_summary_ready"
    ]
    assert summary["next_action_validator_summary_oracle_status"] is None
    assert summary["next_action_validator_summary_oracle_status"] == report[
        "next_action_validator_summary_oracle_status"
    ]
    assert summary["next_action_validator_summary_oracle_blocker_kind"] is None
    assert summary["next_action_validator_summary_oracle_blocker_kind"] == report[
        "next_action_validator_summary_oracle_blocker_kind"
    ]
    assert summary["next_action_sha256"] == report["next_action_sha256"]
    assert summary["next_blocker_sha256"] == report["next_blocker_sha256"]
    assert summary["blocked_validator_results_sha256"] == report[
        "blocked_validator_results_sha256"
    ]


def test_stepfun_validator_status_next_action_includes_oracle_partial_output_handoff(
    tmp_path: Path,
) -> None:
    from scripts import stepfun_correctness_status as status_mod

    prompt = tmp_path / "prompt.json"
    resource = tmp_path / "resource.json"
    oracle = tmp_path / "oracle-timeout.json"
    trace = tmp_path / "trace.json"
    token = tmp_path / "token.json"
    _write_prompt(prompt)
    _write_resource(resource)
    _write_timeout_oracle(oracle)
    _write_trace(trace)
    _write_token(token)

    report = build_validator_status_report(
        _manifest(oracle, trace, token),
        prompt_artifact=prompt,
        resource_artifact=resource,
    )

    assert report["status"] == "blocked"
    assert len(report["blocked_evidence_summary"]) == 1
    assert report["blocked_evidence_summary"][0]["artifact_name"] == (
        "llama_cpp_oracle_success_artifact"
    )
    assert report["blocked_evidence_summary"][0]["status"] == "failed"
    assert report["blocked_evidence_summary"][0]["missing_evidence_count"] == 5
    assert report["blocked_evidence_by_gate"][0]["readiness_gate"] == "oracle_parity"
    assert report["blocked_evidence_by_gate"][0]["artifact_names"] == [
        "llama_cpp_oracle_success_artifact"
    ]
    assert report["blocked_evidence_by_gate"][0]["status_counts"] == {"failed": 1}
    assert report["blocked_evidence_by_gate"][0]["missing_evidence_count"] == 5
    assert report["blocked_readiness_gates"] == ["oracle_parity"]
    assert report["validator_status_summary"]["blocked_readiness_gates"] == [
        "oracle_parity"
    ]
    assert report["next_blocked_gate"] == report["blocked_evidence_by_gate"][0]
    assert report["validator_status_summary"]["next_blocked_gate"] == (
        report["blocked_evidence_by_gate"][0]
    )
    assert report["validator_status_summary"][
        "next_blocked_gate_readiness_gate"
    ] == "oracle_parity"
    assert report["validator_status_summary"][
        "selected_blocked_gate_found"
    ] is False
    assert report["validator_status_summary"]["selected_blocked_gate"] is None
    assert report["validator_status_summary"][
        "selected_blocked_gate_artifact_names"
    ] is None
    assert report["validator_status_summary"][
        "selected_blocked_gate_blocked_count"
    ] is None
    assert report["validator_status_summary"][
        "selected_blocked_gate_status_counts"
    ] is None
    assert report["validator_status_summary"][
        "selected_blocked_gate_producer_commands"
    ] is None
    assert report["validator_status_summary"][
        "selected_blocked_gate_validator_commands"
    ] is None
    assert report["validator_status_summary"][
        "selected_blocked_gate_missing_evidence"
    ] is None
    assert report["next_blocker"]["artifact_name"] == (
        "llama_cpp_oracle_success_artifact"
    )
    assert report["next_blocker"]["validator_summary"]["oracle_status"] == "timeout"
    assert report["next_blocker"]["validator_summary"]["oracle_returncode"] is None
    assert report["next_blocker"]["validator_summary"]["oracle_blocker_kind"] == (
        "llama_cpp_oracle_timeout"
    )
    assert report["next_blocker"]["validator_summary"]["generated_text_len"] == 0
    assert report["next_blocker"][
        "producer_writes_partial_output_before_launch"
    ] is True
    assert report["next_blocker"]["producer_partial_output_path"] == str(oracle)
    assert report["next_blocker"]["producer_partial_output_status"] == "running"
    assert report["next_blocker"]["producer_partial_output_overwrite_policy"] == (
        "overwrite_on_execute_or_timeout"
    )
    assert report["next_blocker"][
        "producer_partial_output_supervisor_signal_handoff_safe"
    ] is True
    assert report["next_blocker"]["artifact_partial_output_handoff_safe"] is True
    assert report["next_blocker"][
        "artifact_partial_output_supervisor_signal_handoff_safe"
    ] is True
    supervisor_contract = report["next_blocker"][
        "artifact_partial_output_supervisor_signal_contract"
    ]
    assert supervisor_contract["handled_signals"] == ["SIGTERM", "SIGINT"]
    assert supervisor_contract["cleanup_method"] == "os.killpg"
    assert supervisor_contract["timeout_blocker_kind"] == (
        "llama_cpp_oracle_timeout"
    )
    assert supervisor_contract["partial_output_overwrite_policy"] == (
        "overwrite_on_execute_or_timeout"
    )
    summary = report["validator_status_summary"]
    assert summary["next_action_validator_summary_sha256"] == (
        status_mod._stable_json_sha256(report["next_blocker"]["validator_summary"])
    )
    assert summary["next_action_validator_summary_sha256"] == report[
        "next_action_validator_summary_sha256"
    ]
    assert summary["next_action_validator_summary_status"] == "failed"
    assert summary["next_action_validator_summary_status"] == report[
        "next_action_validator_summary_status"
    ]
    assert summary["next_action_validator_summary_ready"] is False
    assert summary["next_action_validator_summary_ready"] == report[
        "next_action_validator_summary_ready"
    ]
    assert summary["next_action_validator_summary_oracle_status"] == "timeout"
    assert summary["next_action_validator_summary_oracle_status"] == report[
        "next_action_validator_summary_oracle_status"
    ]
    assert summary["next_action_validator_summary_oracle_blocker_kind"] == (
        "llama_cpp_oracle_timeout"
    )
    assert summary["next_action_validator_summary_oracle_blocker_kind"] == report[
        "next_action_validator_summary_oracle_blocker_kind"
    ]
    expected_no_claim_policy = report["next_blocker"]["validator_summary"][
        "no_claim_policy"
    ]
    assert summary["next_action_no_claim_policy"] == expected_no_claim_policy
    assert summary["next_action_no_claim_policy"] == report[
        "next_action_no_claim_policy"
    ]
    assert summary["next_action_no_claim_policy_sha256"] == (
        status_mod._stable_json_sha256(expected_no_claim_policy)
    )
    assert summary["next_action_no_claim_policy_sha256"] == report[
        "next_action_no_claim_policy_sha256"
    ]
    assert summary["next_action_oracle_parity_claim_allowed"] is False
    assert summary["next_action_oracle_parity_claim_allowed"] == report[
        "next_action_oracle_parity_claim_allowed"
    ]
    assert summary["next_action_kv_backed_decode_claim_allowed"] is False
    assert summary["next_action_kv_backed_decode_claim_allowed"] == report[
        "next_action_kv_backed_decode_claim_allowed"
    ]
    assert summary["next_action_e2e_inference_claim_allowed"] is False
    assert summary["next_action_e2e_inference_claim_allowed"] == report[
        "next_action_e2e_inference_claim_allowed"
    ]
    assert summary["next_action_performance_claim_allowed"] is False
    assert summary["next_action_performance_claim_allowed"] == report[
        "next_action_performance_claim_allowed"
    ]
    expected_oracle_token = {
        "expected_next_token_id": report["next_blocker"]["validator_summary"][
            "expected_next_token_id"
        ],
        "expected_next_token_text": report["next_blocker"]["validator_summary"][
            "expected_next_token_text"
        ],
        "expected_next_token_logit": report["next_blocker"]["validator_summary"][
            "expected_next_token_logit"
        ],
    }
    assert summary["next_action_oracle_expected_token"] == expected_oracle_token
    assert summary["next_action_oracle_expected_token"] == report[
        "next_action_oracle_expected_token"
    ]
    assert summary["next_action_oracle_expected_token_sha256"] == (
        status_mod._stable_json_sha256(expected_oracle_token)
    )
    assert summary["next_action_oracle_expected_token_sha256"] == report[
        "next_action_oracle_expected_token_sha256"
    ]
    assert summary["next_action_expected_next_token_id"] == expected_oracle_token[
        "expected_next_token_id"
    ]
    assert summary["next_action_expected_next_token_id"] == report[
        "next_action_expected_next_token_id"
    ]
    assert summary["next_action_expected_next_token_text"] == expected_oracle_token[
        "expected_next_token_text"
    ]
    assert summary["next_action_expected_next_token_text"] == report[
        "next_action_expected_next_token_text"
    ]
    assert summary["next_action_expected_next_token_logit"] == expected_oracle_token[
        "expected_next_token_logit"
    ]
    assert summary["next_action_expected_next_token_logit"] == report[
        "next_action_expected_next_token_logit"
    ]
    expected_generated_text = {
        "generated_text": report["next_blocker"]["validator_summary"][
            "generated_text"
        ],
        "generated_text_len": report["next_blocker"]["validator_summary"][
            "generated_text_len"
        ],
        "text_matches_expected_exact": report["next_blocker"][
            "validator_summary"
        ]["text_matches_expected_exact"],
        "text_matches_expected_stripped": report["next_blocker"][
            "validator_summary"
        ]["text_matches_expected_stripped"],
    }
    assert summary["next_action_oracle_generated_text"] == expected_generated_text
    assert summary["next_action_oracle_generated_text"] == report[
        "next_action_oracle_generated_text"
    ]
    assert summary["next_action_oracle_generated_text_sha256"] == (
        status_mod._stable_json_sha256(expected_generated_text)
    )
    assert summary["next_action_oracle_generated_text_sha256"] == report[
        "next_action_oracle_generated_text_sha256"
    ]
    assert summary["next_action_generated_text"] == ""
    assert summary["next_action_generated_text"] == report[
        "next_action_generated_text"
    ]
    assert summary["next_action_generated_text_len"] == 0
    assert summary["next_action_generated_text_len"] == report[
        "next_action_generated_text_len"
    ]
    assert summary["next_action_generated_text_matches_expected_exact"] is False
    assert summary["next_action_generated_text_matches_expected_exact"] == report[
        "next_action_generated_text_matches_expected_exact"
    ]
    assert summary["next_action_generated_text_matches_expected_stripped"] is False
    assert summary["next_action_generated_text_matches_expected_stripped"] == report[
        "next_action_generated_text_matches_expected_stripped"
    ]
    expected_provenance = {
        "artifact": report["next_blocker"]["validator_summary"]["artifact"],
        "artifact_sha256": report["next_blocker"]["validator_summary"][
            "artifact_sha256"
        ],
        "prompt_artifact": report["next_blocker"]["validator_summary"][
            "prompt_artifact"
        ],
        "prompt_artifact_sha256": report["next_blocker"]["validator_summary"][
            "prompt_artifact_sha256"
        ],
        "evidence_checks_sha256": report["next_blocker"]["validator_summary"][
            "evidence_checks_sha256"
        ],
    }
    assert summary["next_action_oracle_artifact_provenance"] == expected_provenance
    assert summary["next_action_oracle_artifact_provenance"] == report[
        "next_action_oracle_artifact_provenance"
    ]
    assert summary["next_action_oracle_artifact_provenance_sha256"] == (
        status_mod._stable_json_sha256(expected_provenance)
    )
    assert summary["next_action_oracle_artifact_provenance_sha256"] == report[
        "next_action_oracle_artifact_provenance_sha256"
    ]
    assert summary["next_action_oracle_artifact_path"] == str(oracle)
    assert summary["next_action_oracle_artifact_path"] == report[
        "next_action_oracle_artifact_path"
    ]
    assert summary["next_action_oracle_artifact_sha256"] == expected_provenance[
        "artifact_sha256"
    ]
    assert summary["next_action_oracle_artifact_sha256"] == report[
        "next_action_oracle_artifact_sha256"
    ]
    assert summary["next_action_prompt_artifact_path"] == str(prompt)
    assert summary["next_action_prompt_artifact_path"] == report[
        "next_action_prompt_artifact_path"
    ]
    assert summary["next_action_prompt_artifact_sha256"] == expected_provenance[
        "prompt_artifact_sha256"
    ]
    assert summary["next_action_prompt_artifact_sha256"] == report[
        "next_action_prompt_artifact_sha256"
    ]
    assert summary["next_action_evidence_checks_sha256"] == expected_provenance[
        "evidence_checks_sha256"
    ]
    assert summary["next_action_evidence_checks_sha256"] == report[
        "next_action_evidence_checks_sha256"
    ]
    expected_presence = {
        "oracle_artifact_present": True,
        "prompt_artifact_present": True,
    }
    assert summary["next_action_oracle_artifact_presence"] == expected_presence
    assert summary["next_action_oracle_artifact_presence"] == report[
        "next_action_oracle_artifact_presence"
    ]
    assert summary["next_action_oracle_artifact_presence_sha256"] == (
        status_mod._stable_json_sha256(expected_presence)
    )
    assert summary["next_action_oracle_artifact_presence_sha256"] == report[
        "next_action_oracle_artifact_presence_sha256"
    ]
    assert summary["next_action_oracle_artifact_present"] is True
    assert summary["next_action_oracle_artifact_present"] == report[
        "next_action_oracle_artifact_present"
    ]
    assert summary["next_action_prompt_artifact_present"] is True
    assert summary["next_action_prompt_artifact_present"] == report[
        "next_action_prompt_artifact_present"
    ]
    expected_handoff = {
        "producer_writes_partial_output_before_launch": True,
        "producer_partial_output_path": str(oracle),
        "producer_partial_output_status": "running",
        "producer_partial_output_overwrite_policy": "overwrite_on_execute_or_timeout",
        "producer_partial_output_supervisor_signal_handoff_safe": True,
        "artifact_partial_output_handoff_safe": True,
        "artifact_partial_output_supervisor_signal_handoff_safe": True,
        "artifact_partial_output_supervisor_signal_contract": supervisor_contract,
    }
    assert summary["next_action_partial_output_handoff"] == expected_handoff
    assert summary["next_action_partial_output_handoff"] == report[
        "next_action_partial_output_handoff"
    ]
    assert summary["next_action_partial_output_handoff_sha256"] == (
        status_mod._stable_json_sha256(expected_handoff)
    )
    assert summary["next_action_partial_output_handoff_sha256"] == report[
        "next_action_partial_output_handoff_sha256"
    ]
    assert summary["next_action_partial_output_path"] == str(oracle)
    assert summary["next_action_partial_output_path"] == report[
        "next_action_partial_output_path"
    ]
    assert summary["next_action_partial_output_status"] == "running"
    assert summary["next_action_partial_output_status"] == report[
        "next_action_partial_output_status"
    ]
    assert summary["next_action_missing_evidence"] == report["next_blocker"][
        "validator_missing_evidence"
    ]
    assert summary["next_action_missing_evidence_count"] == 5
    assert summary["next_action_missing_evidence_count"] == report[
        "next_action_missing_evidence_count"
    ]
    assert summary["next_action_first_missing_evidence"] == "oracle_success_status"
    assert summary["next_action_first_missing_evidence"] == report[
        "next_action_first_missing_evidence"
    ]
    assert summary["next_action_last_missing_evidence"] == report["next_blocker"][
        "validator_missing_evidence"
    ][-1]
    assert summary["next_action_last_missing_evidence"] == report[
        "next_action_last_missing_evidence"
    ]
    assert summary["next_action_artifact_file_present_missing"] is False
    assert summary["next_action_artifact_file_present_missing"] == report[
        "next_action_artifact_file_present_missing"
    ]
    assert summary["next_action_oracle_success_status_missing"] is True
    assert summary["next_action_oracle_success_status_missing"] == report[
        "next_action_oracle_success_status_missing"
    ]
    assert summary["next_action_generated_text_matches_target_missing"] is True
    assert summary["next_action_generated_text_matches_target_missing"] == report[
        "next_action_generated_text_matches_target_missing"
    ]
    assert summary["next_action_missing_evidence_sha256"] == (
        status_mod._stable_json_sha256(
            report["next_blocker"]["validator_missing_evidence"]
        )
    )
    assert summary["next_action_missing_evidence_sha256"] == report[
        "next_action_missing_evidence_sha256"
    ]
    next_action = report["next_action"]
    assert next_action["artifact_name"] == "llama_cpp_oracle_success_artifact"
    assert next_action["validator_summary"] == report["next_blocker"][
        "validator_summary"
    ]
    assert next_action["validator_summary"]["missing_evidence"] == [
        "oracle_success_status",
        "oracle_returncode_zero",
        "no_timeout_or_oracle_blocker",
        "generated_text_nonempty",
        "generated_text_matches_target",
    ]
    assert next_action["producer_writes_partial_output_before_launch"] is True
    assert next_action["producer_partial_output_path"] == str(oracle)
    assert next_action["producer_partial_output_status"] == "running"
    assert next_action["producer_partial_output_overwrite_policy"] == (
        "overwrite_on_execute_or_timeout"
    )
    assert next_action[
        "producer_partial_output_supervisor_signal_handoff_safe"
    ] is True
    assert next_action["artifact_partial_output_handoff_safe"] is True
    assert next_action[
        "artifact_partial_output_supervisor_signal_handoff_safe"
    ] is True
    assert next_action["artifact_partial_output_supervisor_signal_contract"] == (
        supervisor_contract
    )


def test_stepfun_validator_status_cli_compact_modes(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    resource = tmp_path / "resource.json"
    oracle = tmp_path / "oracle.json"
    trace = tmp_path / "trace.json"
    token = tmp_path / "token.json"
    manifest = tmp_path / "manifest.json"
    summary_output = tmp_path / "summary.json"
    results_output = tmp_path / "results.json"
    results_sha_output = tmp_path / "results-sha.json"
    blocked_output = tmp_path / "blocked.json"
    blocked_sha_output = tmp_path / "blocked-sha.json"
    blocked_evidence_output = tmp_path / "blocked-evidence.json"
    blocked_evidence_sha_output = tmp_path / "blocked-evidence-sha.json"
    blocked_evidence_by_gate_output = tmp_path / "blocked-evidence-by-gate.json"
    blocked_evidence_by_gate_sha_output = tmp_path / "blocked-evidence-by-gate-sha.json"
    blocked_readiness_gates_output = tmp_path / "blocked-readiness-gates.json"
    blocked_readiness_gates_sha_output = tmp_path / "blocked-readiness-gates-sha.json"
    selected_blocked_gate_output = tmp_path / "selected-blocked-gate.json"
    selected_blocked_gate_sha_output = tmp_path / "selected-blocked-gate-sha.json"
    selected_blocked_gate_found_output = tmp_path / "selected-blocked-gate-found.json"
    selected_blocked_gate_missing_found_output = tmp_path / "selected-blocked-gate-missing-found.json"
    selected_blocked_gate_artifacts_output = tmp_path / "selected-blocked-gate-artifacts.json"
    selected_blocked_gate_artifacts_sha_output = tmp_path / "selected-blocked-gate-artifacts-sha.json"
    selected_blocked_gate_artifact_count_output = tmp_path / "selected-blocked-gate-artifact-count.json"
    selected_blocked_gate_blocked_count_output = tmp_path / "selected-blocked-gate-blocked-count.json"
    selected_blocked_gate_status_counts_output = tmp_path / "selected-blocked-gate-status-counts.json"
    selected_blocked_gate_status_counts_sha_output = tmp_path / "selected-blocked-gate-status-counts-sha.json"
    selected_blocked_gate_producer_commands_output = tmp_path / "selected-blocked-gate-producer-commands.json"
    selected_blocked_gate_producer_commands_sha_output = tmp_path / "selected-blocked-gate-producer-commands-sha.json"
    selected_blocked_gate_producer_command_count_output = tmp_path / "selected-blocked-gate-producer-command-count.json"
    selected_blocked_gate_validator_commands_output = tmp_path / "selected-blocked-gate-validator-commands.json"
    selected_blocked_gate_validator_commands_sha_output = tmp_path / "selected-blocked-gate-validator-commands-sha.json"
    selected_blocked_gate_validator_command_count_output = tmp_path / "selected-blocked-gate-validator-command-count.json"
    selected_blocked_gate_missing_output = tmp_path / "selected-blocked-gate-missing.json"
    selected_blocked_gate_missing_count_output = tmp_path / "selected-blocked-gate-missing-count.json"
    selected_blocked_gate_missing_sha_output = tmp_path / "selected-blocked-gate-missing-sha.json"
    next_blocked_gate_output = tmp_path / "next-blocked-gate.json"
    next_blocked_gate_sha_output = tmp_path / "next-blocked-gate-sha.json"
    next_blocker_output = tmp_path / "next-blocker.json"
    next_blocker_artifact_name_output = tmp_path / "next-blocker-artifact-name.json"
    next_blocker_readiness_gate_output = tmp_path / "next-blocker-readiness-gate.json"
    next_blocker_status_output = tmp_path / "next-blocker-status.json"
    next_blocker_reason_output = tmp_path / "next-blocker-reason.json"
    next_blocker_sha_output = tmp_path / "next-blocker-sha.json"
    next_command_output = tmp_path / "next-command.json"
    next_command_kind_output = tmp_path / "next-command-kind.json"
    next_command_sha_output = tmp_path / "next-command-sha.json"
    next_producer_command_output = tmp_path / "next-producer-command.json"
    next_producer_command_kind_output = tmp_path / "next-producer-command-kind.json"
    next_producer_command_sha_output = tmp_path / "next-producer-command-sha.json"
    next_action_output = tmp_path / "next-action.json"
    next_action_sha_output = tmp_path / "next-action-sha.json"
    next_action_available_output = tmp_path / "next-action-available.json"
    next_action_artifact_name_output = tmp_path / "next-action-artifact-name.json"
    next_action_readiness_gate_output = tmp_path / "next-action-readiness-gate.json"
    next_action_status_output = tmp_path / "next-action-status.json"
    next_action_reason_output = tmp_path / "next-action-reason.json"
    next_action_validator_command_kind_output = tmp_path / (
        "next-action-validator-command-kind.json"
    )
    next_action_validator_command_output = tmp_path / "next-action-validator-command.json"
    next_action_validator_command_sha_output = tmp_path / (
        "next-action-validator-command-sha.json"
    )
    next_action_producer_command_kind_output = tmp_path / (
        "next-action-producer-command-kind.json"
    )
    next_action_producer_command_output = tmp_path / "next-action-producer-command.json"
    next_action_producer_command_sha_output = tmp_path / (
        "next-action-producer-command-sha.json"
    )
    sha_output = tmp_path / "sha.json"
    status_output = tmp_path / "status.json"
    _write_prompt(prompt)
    _write_resource(resource)
    _write_oracle(oracle)
    _write_trace(trace)
    _write_token(token)
    manifest.write_text(json.dumps(_manifest(oracle, trace, token), sort_keys=True))

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--summary-only",
            "--output",
            str(summary_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(summary_output.read_text())["status"] == "passed"

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--summary-only",
            "--sha-only",
            "--output",
            str(sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert len(json.loads(sha_output.read_text())) == 64

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--results-only",
            "--output",
            str(results_output),
            "--pretty",
        ]
    )
    assert rc == 0
    results_payload = json.loads(results_output.read_text())
    assert [record["status"] for record in results_payload] == [
        "passed",
        "passed",
        "passed",
    ]

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--results-sha-only",
            "--output",
            str(results_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert len(json.loads(results_sha_output.read_text())) == 64

    trace.unlink()
    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-only",
            "--output",
            str(blocked_output),
            "--pretty",
        ]
    )
    assert rc == 0
    blocked_payload = json.loads(blocked_output.read_text())
    assert [record["artifact_name"] for record in blocked_payload] == [
        "kv_kernel_trace_artifact"
    ]
    assert blocked_payload[0]["status"] == "missing"

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-sha-only",
            "--output",
            str(blocked_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert len(json.loads(blocked_sha_output.read_text())) == 64

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-summary-only",
            "--output",
            str(blocked_evidence_output),
            "--pretty",
        ]
    )
    assert rc == 0
    blocked_evidence_payload = json.loads(blocked_evidence_output.read_text())
    assert blocked_evidence_payload == [
        {
            "artifact_name": "kv_kernel_trace_artifact",
            "readiness_gate": "kv_backed_decode",
            "status": "missing",
            "reason": "artifact_file_missing",
            "validator_artifact_path": str(trace),
            "validator_command_kind": "kv_trace_check_command",
            "validator_command_sha256": "trace-command-sha",
            "producer_command_kind": "resource_plan_refresh_command",
            "producer_command_sha256": "kv-producer-sha",
            "missing_evidence": ["artifact_file_present"],
            "missing_evidence_count": 1,
            "validator_summary_sha256": None,
        }
    ]

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-summary-sha-only",
            "--output",
            str(blocked_evidence_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert len(json.loads(blocked_evidence_sha_output.read_text())) == 64

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-by-gate-only",
            "--output",
            str(blocked_evidence_by_gate_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(blocked_evidence_by_gate_output.read_text()) == [
        {
            "readiness_gate": "kv_backed_decode",
            "artifact_names": ["kv_kernel_trace_artifact"],
            "blocked_count": 1,
            "status_counts": {"missing": 1},
            "missing_evidence": ["artifact_file_present"],
            "producer_command_kinds": ["resource_plan_refresh_command"],
            "producer_command_sha256s": ["kv-producer-sha"],
            "validator_command_kinds": ["kv_trace_check_command"],
            "validator_command_sha256s": ["trace-command-sha"],
            "missing_evidence_count": 1,
        }
    ]

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-by-gate-sha-only",
            "--output",
            str(blocked_evidence_by_gate_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert len(json.loads(blocked_evidence_by_gate_sha_output.read_text())) == 64

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-readiness-gates-only",
            "--output",
            str(blocked_readiness_gates_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(blocked_readiness_gates_output.read_text()) == [
        "kv_backed_decode"
    ]

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-readiness-gates-sha-only",
            "--output",
            str(blocked_readiness_gates_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert len(json.loads(blocked_readiness_gates_sha_output.read_text())) == 64

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-gate",
            "kv_backed_decode",
            "--blocked-evidence-gate-only",
            "--output",
            str(selected_blocked_gate_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(selected_blocked_gate_output.read_text()) == {
        "readiness_gate": "kv_backed_decode",
        "artifact_names": ["kv_kernel_trace_artifact"],
        "blocked_count": 1,
        "status_counts": {"missing": 1},
        "missing_evidence": ["artifact_file_present"],
        "producer_command_kinds": ["resource_plan_refresh_command"],
        "producer_command_sha256s": ["kv-producer-sha"],
        "validator_command_kinds": ["kv_trace_check_command"],
        "validator_command_sha256s": ["trace-command-sha"],
        "missing_evidence_count": 1,
    }

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-gate",
            "kv_backed_decode",
            "--blocked-evidence-gate-sha-only",
            "--output",
            str(selected_blocked_gate_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert len(json.loads(selected_blocked_gate_sha_output.read_text())) == 64

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-gate",
            "kv_backed_decode",
            "--blocked-evidence-gate-found-only",
            "--output",
            str(selected_blocked_gate_found_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(selected_blocked_gate_found_output.read_text()) is True

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-gate",
            "not_a_gate",
            "--blocked-evidence-gate-found-only",
            "--output",
            str(selected_blocked_gate_missing_found_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(selected_blocked_gate_missing_found_output.read_text()) is False

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-gate",
            "kv_backed_decode",
            "--blocked-evidence-gate-artifacts-only",
            "--output",
            str(selected_blocked_gate_artifacts_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(selected_blocked_gate_artifacts_output.read_text()) == [
        "kv_kernel_trace_artifact"
    ]

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-gate",
            "kv_backed_decode",
            "--blocked-evidence-gate-artifacts-sha-only",
            "--output",
            str(selected_blocked_gate_artifacts_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert len(json.loads(selected_blocked_gate_artifacts_sha_output.read_text())) == 64

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-gate",
            "kv_backed_decode",
            "--blocked-evidence-gate-artifact-count-only",
            "--output",
            str(selected_blocked_gate_artifact_count_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(selected_blocked_gate_artifact_count_output.read_text()) == 1

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-gate",
            "kv_backed_decode",
            "--blocked-evidence-gate-blocked-count-only",
            "--output",
            str(selected_blocked_gate_blocked_count_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(selected_blocked_gate_blocked_count_output.read_text()) == 1

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-gate",
            "kv_backed_decode",
            "--blocked-evidence-gate-status-counts-only",
            "--output",
            str(selected_blocked_gate_status_counts_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(selected_blocked_gate_status_counts_output.read_text()) == {
        "missing": 1
    }

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-gate",
            "kv_backed_decode",
            "--blocked-evidence-gate-status-counts-sha-only",
            "--output",
            str(selected_blocked_gate_status_counts_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert len(json.loads(selected_blocked_gate_status_counts_sha_output.read_text())) == 64

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-gate",
            "kv_backed_decode",
            "--blocked-evidence-gate-producer-commands-only",
            "--output",
            str(selected_blocked_gate_producer_commands_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(selected_blocked_gate_producer_commands_output.read_text()) == [
        "python3 scripts/refresh_stepfun_kv_artifacts.py"
    ]

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-gate",
            "kv_backed_decode",
            "--blocked-evidence-gate-producer-commands-sha-only",
            "--output",
            str(selected_blocked_gate_producer_commands_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert len(json.loads(selected_blocked_gate_producer_commands_sha_output.read_text())) == 64

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-gate",
            "kv_backed_decode",
            "--blocked-evidence-gate-producer-command-count-only",
            "--output",
            str(selected_blocked_gate_producer_command_count_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(selected_blocked_gate_producer_command_count_output.read_text()) == 1

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-gate",
            "kv_backed_decode",
            "--blocked-evidence-gate-validator-commands-only",
            "--output",
            str(selected_blocked_gate_validator_commands_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(selected_blocked_gate_validator_commands_output.read_text()) == [
        f"python3 scripts/stepfun_kv_trace_check.py --trace {trace}"
    ]

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-gate",
            "kv_backed_decode",
            "--blocked-evidence-gate-validator-commands-sha-only",
            "--output",
            str(selected_blocked_gate_validator_commands_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert len(json.loads(selected_blocked_gate_validator_commands_sha_output.read_text())) == 64

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-gate",
            "kv_backed_decode",
            "--blocked-evidence-gate-validator-command-count-only",
            "--output",
            str(selected_blocked_gate_validator_command_count_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(selected_blocked_gate_validator_command_count_output.read_text()) == 1

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-gate",
            "kv_backed_decode",
            "--blocked-evidence-gate-missing-evidence-only",
            "--output",
            str(selected_blocked_gate_missing_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(selected_blocked_gate_missing_output.read_text()) == [
        "artifact_file_present"
    ]

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-gate",
            "kv_backed_decode",
            "--blocked-evidence-gate-missing-evidence-count-only",
            "--output",
            str(selected_blocked_gate_missing_count_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(selected_blocked_gate_missing_count_output.read_text()) == 1

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--blocked-evidence-gate",
            "kv_backed_decode",
            "--blocked-evidence-gate-missing-evidence-sha-only",
            "--output",
            str(selected_blocked_gate_missing_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert len(json.loads(selected_blocked_gate_missing_sha_output.read_text())) == 64

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-blocked-gate-only",
            "--output",
            str(next_blocked_gate_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(next_blocked_gate_output.read_text()) == {
        "readiness_gate": "kv_backed_decode",
        "artifact_names": ["kv_kernel_trace_artifact"],
        "blocked_count": 1,
        "status_counts": {"missing": 1},
        "missing_evidence": ["artifact_file_present"],
        "producer_command_kinds": ["resource_plan_refresh_command"],
        "producer_command_sha256s": ["kv-producer-sha"],
        "validator_command_kinds": ["kv_trace_check_command"],
        "validator_command_sha256s": ["trace-command-sha"],
        "missing_evidence_count": 1,
    }

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-blocked-gate-sha-only",
            "--output",
            str(next_blocked_gate_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert len(json.loads(next_blocked_gate_sha_output.read_text())) == 64

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-blocker-only",
            "--output",
            str(next_blocker_output),
            "--pretty",
        ]
    )
    assert rc == 0
    next_blocker_payload = json.loads(next_blocker_output.read_text())
    assert next_blocker_payload["artifact_name"] == "kv_kernel_trace_artifact"
    assert next_blocker_payload["status"] == "missing"
    assert next_blocker_payload["reason"] == "artifact_file_missing"

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-blocker-artifact-name-only",
            "--output",
            str(next_blocker_artifact_name_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(next_blocker_artifact_name_output.read_text()) == (
        "kv_kernel_trace_artifact"
    )

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-blocker-readiness-gate-only",
            "--output",
            str(next_blocker_readiness_gate_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(next_blocker_readiness_gate_output.read_text()) == (
        "kv_backed_decode"
    )

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-blocker-status-only",
            "--output",
            str(next_blocker_status_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(next_blocker_status_output.read_text()) == "missing"

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-blocker-reason-only",
            "--output",
            str(next_blocker_reason_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(next_blocker_reason_output.read_text()) == "artifact_file_missing"

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-command-only",
            "--output",
            str(next_command_output),
            "--pretty",
        ]
    )
    assert rc == 0
    next_command_payload = json.loads(next_command_output.read_text())
    assert next_command_payload == f"python3 scripts/stepfun_kv_trace_check.py --trace {trace}"

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-command-kind-only",
            "--output",
            str(next_command_kind_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(next_command_kind_output.read_text()) == "kv_trace_check_command"

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-command-sha-only",
            "--output",
            str(next_command_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert len(json.loads(next_command_sha_output.read_text())) == 64

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-producer-command-only",
            "--output",
            str(next_producer_command_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(next_producer_command_output.read_text()) == (
        "python3 scripts/refresh_stepfun_kv_artifacts.py"
    )

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-producer-command-kind-only",
            "--output",
            str(next_producer_command_kind_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(next_producer_command_kind_output.read_text()) == (
        "resource_plan_refresh_command"
    )

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-producer-command-sha-only",
            "--output",
            str(next_producer_command_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert len(json.loads(next_producer_command_sha_output.read_text())) == 64

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-action-only",
            "--output",
            str(next_action_output),
            "--pretty",
        ]
    )
    assert rc == 0
    next_action_payload = json.loads(next_action_output.read_text())
    assert next_action_payload["artifact_name"] == "kv_kernel_trace_artifact"
    assert next_action_payload["validator_command"] == next_command_payload
    assert next_action_payload["producer_command"] == (
        "python3 scripts/refresh_stepfun_kv_artifacts.py"
    )

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-action-available-only",
            "--output",
            str(next_action_available_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(next_action_available_output.read_text()) is True

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-action-artifact-name-only",
            "--output",
            str(next_action_artifact_name_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(next_action_artifact_name_output.read_text()) == (
        "kv_kernel_trace_artifact"
    )

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-action-readiness-gate-only",
            "--output",
            str(next_action_readiness_gate_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(next_action_readiness_gate_output.read_text()) == (
        "kv_backed_decode"
    )

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-action-status-only",
            "--output",
            str(next_action_status_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(next_action_status_output.read_text()) == "missing"

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-action-reason-only",
            "--output",
            str(next_action_reason_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(next_action_reason_output.read_text()) == "artifact_file_missing"

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-action-validator-command-kind-only",
            "--output",
            str(next_action_validator_command_kind_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(next_action_validator_command_kind_output.read_text()) == (
        "kv_trace_check_command"
    )

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-action-validator-command-only",
            "--output",
            str(next_action_validator_command_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(next_action_validator_command_output.read_text()) == (
        f"python3 scripts/stepfun_kv_trace_check.py --trace {trace}"
    )

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-action-validator-command-sha-only",
            "--output",
            str(next_action_validator_command_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(next_action_validator_command_sha_output.read_text()) == (
        status_mod._stable_json_sha256(
            f"python3 scripts/stepfun_kv_trace_check.py --trace {trace}"
        )
    )

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-action-producer-command-kind-only",
            "--output",
            str(next_action_producer_command_kind_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(next_action_producer_command_kind_output.read_text()) == (
        "resource_plan_refresh_command"
    )

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-action-producer-command-only",
            "--output",
            str(next_action_producer_command_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(next_action_producer_command_output.read_text()) == (
        "python3 scripts/refresh_stepfun_kv_artifacts.py"
    )

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-action-producer-command-sha-only",
            "--output",
            str(next_action_producer_command_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(next_action_producer_command_sha_output.read_text()) == (
        status_mod._stable_json_sha256("python3 scripts/refresh_stepfun_kv_artifacts.py")
    )

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-action-sha-only",
            "--output",
            str(next_action_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert len(json.loads(next_action_sha_output.read_text())) == 64

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--next-blocker-sha-only",
            "--output",
            str(next_blocker_sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert len(json.loads(next_blocker_sha_output.read_text())) == 64

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--prompt-artifact",
            str(prompt),
            "--resource-artifact",
            str(resource),
            "--status-only",
            "--fail-on-blocked",
            "--output",
            str(status_output),
            "--pretty",
        ]
    )
    assert rc == 2
    assert json.loads(status_output.read_text()) == "blocked"


def test_stepfun_validator_status_cli_next_action_validator_summary_modes(
    tmp_path: Path,
    capsys,
) -> None:
    from scripts import stepfun_correctness_status as status_mod

    prompt = tmp_path / "prompt.json"
    resource = tmp_path / "resource.json"
    oracle = tmp_path / "oracle-timeout.json"
    trace = tmp_path / "trace.json"
    token = tmp_path / "token.json"
    manifest = tmp_path / "manifest.json"
    _write_prompt(prompt)
    _write_resource(resource)
    _write_timeout_oracle(oracle)
    _write_trace(trace)
    _write_token(token)
    manifest.write_text(json.dumps(_manifest(oracle, trace, token), sort_keys=True))

    args = [
        "--manifest",
        str(manifest),
        "--prompt-artifact",
        str(prompt),
        "--resource-artifact",
        str(resource),
    ]

    rc = main([*args, "--next-action-validator-summary-only", "--pretty"])

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["oracle_status"] == "timeout"
    assert summary["oracle_returncode"] is None
    assert summary["oracle_blocker_kind"] == "llama_cpp_oracle_timeout"
    assert summary["generated_text_len"] == 0
    assert summary["missing_evidence"] == [
        "oracle_success_status",
        "oracle_returncode_zero",
        "no_timeout_or_oracle_blocker",
        "generated_text_nonempty",
        "generated_text_matches_target",
    ]

    rc = main([*args, "--next-action-validator-summary-sha-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == status_mod._stable_json_sha256(
        summary
    )

    rc = main([*args, "--next-action-validator-summary-status-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == "failed"

    rc = main([*args, "--next-action-validator-summary-ready-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) is False

    rc = main([*args, "--next-action-validator-summary-oracle-status-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == "timeout"

    rc = main([*args, "--next-action-validator-summary-oracle-blocker-kind-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == "llama_cpp_oracle_timeout"

    rc = main([*args, "--next-action-oracle-expected-token-only", "--pretty"])

    assert rc == 0
    oracle_expected_token = json.loads(capsys.readouterr().out)
    assert oracle_expected_token == {
        "expected_next_token_id": summary["expected_next_token_id"],
        "expected_next_token_text": summary["expected_next_token_text"],
        "expected_next_token_logit": summary["expected_next_token_logit"],
    }

    rc = main([*args, "--next-action-oracle-expected-token-sha-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == status_mod._stable_json_sha256(
        oracle_expected_token
    )

    rc = main([*args, "--next-action-expected-next-token-id-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == summary["expected_next_token_id"]

    rc = main([*args, "--next-action-expected-next-token-text-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == summary["expected_next_token_text"]

    rc = main([*args, "--next-action-expected-next-token-logit-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == summary["expected_next_token_logit"]

    rc = main([*args, "--next-action-oracle-generated-text-only", "--pretty"])

    assert rc == 0
    oracle_generated_text = json.loads(capsys.readouterr().out)
    assert oracle_generated_text == {
        "generated_text": summary["generated_text"],
        "generated_text_len": summary["generated_text_len"],
        "text_matches_expected_exact": summary["text_matches_expected_exact"],
        "text_matches_expected_stripped": summary[
            "text_matches_expected_stripped"
        ],
    }

    rc = main([*args, "--next-action-oracle-generated-text-sha-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == status_mod._stable_json_sha256(
        oracle_generated_text
    )

    rc = main([*args, "--next-action-generated-text-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == summary["generated_text"]

    rc = main([*args, "--next-action-generated-text-len-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == summary["generated_text_len"]

    rc = main([*args, "--next-action-generated-text-matches-expected-exact-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) is False

    rc = main([*args, "--next-action-generated-text-matches-expected-stripped-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) is False

    rc = main([*args, "--next-action-oracle-artifact-provenance-only", "--pretty"])

    assert rc == 0
    artifact_provenance = json.loads(capsys.readouterr().out)
    assert artifact_provenance == {
        "artifact": summary["artifact"],
        "artifact_sha256": summary["artifact_sha256"],
        "prompt_artifact": summary["prompt_artifact"],
        "prompt_artifact_sha256": summary["prompt_artifact_sha256"],
        "evidence_checks_sha256": summary["evidence_checks_sha256"],
    }

    rc = main([*args, "--next-action-oracle-artifact-provenance-sha-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == status_mod._stable_json_sha256(
        artifact_provenance
    )

    rc = main([*args, "--next-action-oracle-artifact-path-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == str(oracle)

    rc = main([*args, "--next-action-oracle-artifact-sha-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == summary["artifact_sha256"]

    rc = main([*args, "--next-action-prompt-artifact-path-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == str(prompt)

    rc = main([*args, "--next-action-prompt-artifact-sha-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == summary["prompt_artifact_sha256"]

    rc = main([*args, "--next-action-evidence-checks-sha-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == summary["evidence_checks_sha256"]

    rc = main([*args, "--next-action-oracle-artifact-presence-only", "--pretty"])

    assert rc == 0
    artifact_presence = json.loads(capsys.readouterr().out)
    assert artifact_presence == {
        "oracle_artifact_present": True,
        "prompt_artifact_present": True,
    }

    rc = main([*args, "--next-action-oracle-artifact-presence-sha-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == status_mod._stable_json_sha256(
        artifact_presence
    )

    rc = main([*args, "--next-action-oracle-artifact-present-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) is True

    rc = main([*args, "--next-action-prompt-artifact-present-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) is True

    rc = main([*args, "--next-action-no-claim-policy-only", "--pretty"])

    assert rc == 0
    no_claim_policy = json.loads(capsys.readouterr().out)
    assert no_claim_policy == summary["no_claim_policy"]
    assert no_claim_policy["oracle_parity_claim_allowed"] is False
    assert no_claim_policy["kv_backed_decode_claim_allowed"] is False
    assert no_claim_policy["e2e_inference_claim_allowed"] is False
    assert no_claim_policy["performance_claim_allowed"] is False

    rc = main([*args, "--next-action-no-claim-policy-sha-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == status_mod._stable_json_sha256(
        no_claim_policy
    )

    rc = main([*args, "--next-action-oracle-parity-claim-allowed-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) is False

    rc = main([*args, "--next-action-kv-backed-decode-claim-allowed-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) is False

    rc = main([*args, "--next-action-e2e-inference-claim-allowed-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) is False

    rc = main([*args, "--next-action-performance-claim-allowed-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) is False

    rc = main([*args, "--next-action-partial-output-handoff-only", "--pretty"])

    assert rc == 0
    partial_handoff = json.loads(capsys.readouterr().out)
    assert partial_handoff["producer_partial_output_path"] == str(oracle)
    assert partial_handoff["producer_partial_output_status"] == "running"
    assert partial_handoff["producer_partial_output_supervisor_signal_handoff_safe"] is True
    assert partial_handoff["artifact_partial_output_handoff_safe"] is True
    assert partial_handoff["artifact_partial_output_supervisor_signal_handoff_safe"] is True
    assert partial_handoff["artifact_partial_output_supervisor_signal_contract"][
        "timeout_blocker_kind"
    ] == "llama_cpp_oracle_timeout"

    rc = main([*args, "--next-action-partial-output-handoff-sha-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == status_mod._stable_json_sha256(
        partial_handoff
    )

    rc = main([*args, "--next-action-partial-output-path-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == str(oracle)

    rc = main([*args, "--next-action-partial-output-status-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == "running"

    rc = main([*args, "--next-action-missing-evidence-only", "--pretty"])

    assert rc == 0
    missing_evidence = json.loads(capsys.readouterr().out)
    assert missing_evidence == summary["missing_evidence"]

    rc = main([*args, "--next-action-missing-evidence-count-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == len(missing_evidence)

    rc = main([*args, "--next-action-first-missing-evidence-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == missing_evidence[0]

    rc = main([*args, "--next-action-last-missing-evidence-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == missing_evidence[-1]

    rc = main([*args, "--next-action-artifact-file-present-missing-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == (
        "artifact_file_present" in missing_evidence
    )

    rc = main([*args, "--next-action-oracle-success-status-missing-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == (
        "oracle_success_status" in missing_evidence
    )

    rc = main([*args, "--next-action-generated-text-matches-target-missing-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == (
        "generated_text_matches_target" in missing_evidence
    )

    rc = main([*args, "--next-action-missing-evidence-sha-only"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == status_mod._stable_json_sha256(
        missing_evidence
    )
