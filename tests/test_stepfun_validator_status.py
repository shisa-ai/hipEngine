from __future__ import annotations

import json
from pathlib import Path

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
    assert summary["next_blocker_artifact_name"] is None
    assert summary["next_blocker_status"] is None
    assert summary["next_blocker_reason"] is None
    assert summary["next_blocker_sha256"] == report["next_blocker_sha256"]
    assert report["blocked_validator_results"] == []
    assert report["next_blocker"] is None
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
    }
    assert results["kv_kernel_trace_artifact"] == expected_missing_trace
    assert report["blocked_validator_results"] == [expected_missing_trace]
    assert report["next_blocker"] == expected_missing_trace
    assert summary["next_blocker_artifact_name"] == "kv_kernel_trace_artifact"
    assert summary["next_blocker_status"] == "missing"
    assert summary["next_blocker_reason"] == "artifact_file_missing"
    assert summary["next_blocker_sha256"] == report["next_blocker_sha256"]
    assert summary["blocked_validator_results_sha256"] == report[
        "blocked_validator_results_sha256"
    ]


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
    next_blocker_output = tmp_path / "next-blocker.json"
    next_blocker_sha_output = tmp_path / "next-blocker-sha.json"
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
