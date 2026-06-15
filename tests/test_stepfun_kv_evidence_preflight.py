from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_kv_evidence_preflight import build_kv_evidence_preflight, main


def _write_session_contract(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "artifact_kind": "stepfun_kv_session_streaming_decode_contract",
        "status": "blocked",
        "contract": {
            "blocked_by": "streaming_decode_loop_not_wired",
            "next_action": "wire_streaming_decode_loop",
            "no_kernel_launches": True,
            "launch_operation_count": 135,
            "required_artifacts": [
                "benchmarks/results/2026-05-31-stepfun-q3kl-kv-kernel-trace.json",
                "benchmarks/results/2026-05-31-stepfun-q3kl-kv-backed-next-token.json",
            ],
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True))


def _write_kv_blocker(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "artifact_kind": "stepfun_kv_backed_decode_blocker_status",
        "status": "blocked",
        "blocked_count": 2,
        "missing_artifact_paths": [
            "benchmarks/results/2026-05-31-stepfun-q3kl-kv-kernel-trace.json",
            "benchmarks/results/2026-05-31-stepfun-q3kl-kv-backed-next-token.json",
        ],
        "streaming_runner_source_status": {
            "blocked_by": "streaming_decode_loop_not_wired"
        },
        "runtime_wiring_symbol_validation": {"all_symbols_present": True},
    }
    path.write_text(json.dumps(payload, sort_keys=True))


def test_stepfun_kv_evidence_preflight_reports_missing_required_artifacts(
    tmp_path: Path,
) -> None:
    session_contract = tmp_path / "session-contract.json"
    kv_blocker = tmp_path / "kv-blocker.json"
    trace = tmp_path / "missing-trace.json"
    next_token = tmp_path / "missing-next-token.json"
    _write_session_contract(session_contract)
    _write_kv_blocker(kv_blocker)

    report = build_kv_evidence_preflight(
        session_contract_artifact=session_contract,
        kv_blocker_artifact=kv_blocker,
        trace_artifact=trace,
        next_token_artifact=next_token,
        artifact_date="2030-01-09",
    )

    assert report["schema_version"] == 1
    assert report["artifact_kind"] == "stepfun_kv_evidence_preflight"
    assert report["date"] == "2030-01-09"
    assert report["status"] == "blocked"
    assert report["metadata_contract_ready"] is True
    assert report["required_artifacts_present"] is False
    assert report["kv_backed_decode_ready"] is False
    assert report["next_action"] == "wire_streaming_decode_loop"
    assert report["missing_required_artifact_paths"] == [str(trace), str(next_token)]
    assert report["session_contract"]["ready"] is True
    assert report["session_contract"]["blocked_by"] == "streaming_decode_loop_not_wired"
    assert report["kv_blocker_status"]["ready"] is True
    assert report["kv_blocker_status"]["runtime_symbols_validated"] is True
    assert [record["ready"] for record in report["required_artifact_checks"]] == [
        False,
        False,
    ]
    assert [record["missing_evidence"] for record in report["required_artifact_checks"]] == [
        ["artifact_file_present"],
        ["artifact_file_present"],
    ]
    assert report["generator_commands"] == {
        "session_contract": (
            "python3 scripts/stepfun_kv_session_contract.py --default-output --pretty"
        ),
        "kv_blocker_status": (
            "python3 scripts/stepfun_kv_blocker_status.py --default-output --pretty"
        ),
    }
    assert report["no_claim_policy"] == {
        "kv_backed_decode_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": (
            "This preflight proves only that metadata-only contract evidence exists; "
            "the required KV trace and KV-backed next-token artifacts are still absent."
        ),
    }


def test_stepfun_kv_evidence_preflight_cli_writes_report(tmp_path: Path) -> None:
    session_contract = tmp_path / "session-contract.json"
    kv_blocker = tmp_path / "kv-blocker.json"
    trace = tmp_path / "missing-trace.json"
    next_token = tmp_path / "missing-next-token.json"
    output = tmp_path / "preflight.json"
    _write_session_contract(session_contract)
    _write_kv_blocker(kv_blocker)

    rc = main(
        [
            "--session-contract-artifact",
            str(session_contract),
            "--kv-blocker-artifact",
            str(kv_blocker),
            "--trace-artifact",
            str(trace),
            "--next-token-artifact",
            str(next_token),
            "--artifact-date",
            "2030-01-10",
            "--output",
            str(output),
            "--pretty",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["date"] == "2030-01-10"
    assert payload["status"] == "blocked"
    assert payload["missing_required_artifact_paths"] == [str(trace), str(next_token)]


def test_stepfun_kv_evidence_preflight_cli_compact_modes(tmp_path: Path) -> None:
    session_contract = tmp_path / "session-contract.json"
    kv_blocker = tmp_path / "kv-blocker.json"
    trace = tmp_path / "missing-trace.json"
    next_token = tmp_path / "missing-next-token.json"
    output = tmp_path / "compact.json"
    _write_session_contract(session_contract)
    _write_kv_blocker(kv_blocker)
    base_args = [
        "--session-contract-artifact",
        str(session_contract),
        "--kv-blocker-artifact",
        str(kv_blocker),
        "--trace-artifact",
        str(trace),
        "--next-token-artifact",
        str(next_token),
    ]

    assert main([*base_args, "--status-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == "blocked"
    assert main([*base_args, "--missing-paths-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == [str(trace), str(next_token)]
    assert main([*base_args, "--next-action-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == "wire_streaming_decode_loop"
