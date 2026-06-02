from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.stepfun_correctness_status import build_status, main


CANONICAL_ORACLE = Path(
    "benchmarks/results/2026-05-31-stepfun-q3kl-llamacpp-step35-timeout.json"
)
WRAPPER_TIMEOUT_ARTIFACT = Path(
    "benchmarks/results/2026-06-01-stepfun-q3kl-llamacpp-step35-180s-wrapper-timeout.json"
)


def test_stepfun_180s_oracle_wrapper_timeout_artifact_is_blocking_evidence() -> None:
    artifact = json.loads(WRAPPER_TIMEOUT_ARTIFACT.read_text())
    canonical = json.loads(CANONICAL_ORACLE.read_text())
    canonical_sha256 = hashlib.sha256(CANONICAL_ORACLE.read_bytes()).hexdigest()

    assert artifact["schema_version"] == 1
    assert artifact["status"] == "blocked"
    assert artifact["attempt_kind"] == "llamacpp_oracle_longer_timeout_wrapper_timeout"
    assert artifact["blocker_kind"] == "llama_cpp_oracle_timeout"
    assert artifact["script_timeout_s"] == 180.0
    assert artifact["wrapper_timeout_s"] == 240.0
    assert artifact["wrapper_result"] == "timeout"
    assert artifact["output_artifact"] == str(CANONICAL_ORACLE)

    # The longer attempt was interrupted by the outer pi wrapper before the helper
    # rewrote the canonical machine-readable oracle artifact. A later 60 s helper
    # refresh added timeout_termination metadata, so the wrapper's historical
    # after-attempt SHA is retained as evidence but no longer equals the current
    # canonical artifact SHA.
    assert artifact["output_artifact_status_after_attempt"] == canonical["status"] == "timeout"
    assert artifact["output_artifact_timeout_s_after_attempt"] == canonical["timeout_s"] == 60.0
    assert artifact["output_artifact_sha256_after_attempt"] != canonical_sha256
    assert canonical["timeout_termination"]["termination_method"] == "os.killpg"
    assert canonical["timeout_termination"]["termination_signal"] == "SIGKILL"

    command = artifact["command"]
    assert command[:2] == ["python3", "scripts/stepfun_llamacpp_oracle.py"]
    assert "--timeout-s" in command
    assert command[command.index("--timeout-s") + 1] == "180.0"
    assert "--diagnostic-logs" in command
    assert "--llama-arg=--device" in command
    assert "--llama-arg=none" in command
    assert "--llama-arg=--gpu-layers" in command
    assert "--llama-arg=0" in command

    assert artifact["readiness_impact"] == {
        "oracle_parity": False,
        "kv_backed_decode_ready": False,
        "e2e_inference_ready": False,
    }
    assert "before scripts/stepfun_llamacpp_oracle.py rewrote" in artifact["observed_result"]
    assert artifact["claim"] == "No oracle parity, e2e correctness, or performance claim is made."


def test_stepfun_correctness_status_tracks_oracle_wrapper_timeout_source_artifact() -> None:
    status = build_status(
        Path("benchmarks/results/2026-05-31-stepfun-q3kl-layer-prefix-all45-prompt-smoke.json"),
        CANONICAL_ORACLE,
    )
    source_record = status["source_artifacts"]["oracle_wrapper_timeout"]

    assert source_record["path"] == str(WRAPPER_TIMEOUT_ARTIFACT)
    assert source_record["exists"] is True
    assert source_record["size_bytes"] == len(WRAPPER_TIMEOUT_ARTIFACT.read_bytes())
    assert source_record["sha256"] == hashlib.sha256(
        WRAPPER_TIMEOUT_ARTIFACT.read_bytes()
    ).hexdigest()
    assert status["status_integrity"]["checks"]["source_artifacts_sha256"] is True


def test_stepfun_correctness_status_oracle_wrapper_timeout_source_only(tmp_path: Path) -> None:
    output = tmp_path / "oracle-wrapper-timeout-source.json"

    rc = main([
        "--oracle-wrapper-timeout-source-only",
        "--output",
        str(output),
        "--pretty",
    ])

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["path"] == str(WRAPPER_TIMEOUT_ARTIFACT)
    assert payload["exists"] is True
    assert payload["sha256"] == hashlib.sha256(
        WRAPPER_TIMEOUT_ARTIFACT.read_bytes()
    ).hexdigest()


def test_stepfun_correctness_status_oracle_wrapper_timeout_source_sha_only(
    tmp_path: Path,
) -> None:
    output = tmp_path / "oracle-wrapper-timeout-source-sha.json"

    rc = main([
        "--oracle-wrapper-timeout-source-sha-only",
        "--output",
        str(output),
        "--pretty",
    ])

    assert rc == 0
    assert json.loads(output.read_text()) == hashlib.sha256(
        WRAPPER_TIMEOUT_ARTIFACT.read_bytes()
    ).hexdigest()


def test_stepfun_correctness_status_verify_source_oracle_wrapper_timeout_source_only(
    tmp_path: Path,
) -> None:
    status_path = tmp_path / "status.json"
    output = tmp_path / "oracle-wrapper-timeout-source-verify.json"
    status = build_status(
        Path("benchmarks/results/2026-05-31-stepfun-q3kl-layer-prefix-all45-prompt-smoke.json"),
        CANONICAL_ORACLE,
    )
    status_path.write_text(json.dumps(status))

    rc = main([
        "--verify-source-artifacts",
        str(status_path),
        "--oracle-wrapper-timeout-source-only",
        "--output",
        str(output),
        "--pretty",
    ])

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["match"] is True
    assert payload["path"] == str(WRAPPER_TIMEOUT_ARTIFACT)
    assert payload["recorded"]["path"] == str(WRAPPER_TIMEOUT_ARTIFACT)
    assert payload["current"]["sha256"] == hashlib.sha256(
        WRAPPER_TIMEOUT_ARTIFACT.read_bytes()
    ).hexdigest()


def test_stepfun_oracle_wrapper_timeout_source_artifact_drift_is_detected(
    tmp_path: Path,
) -> None:
    status = build_status(
        Path("benchmarks/results/2026-05-31-stepfun-q3kl-layer-prefix-all45-prompt-smoke.json"),
        CANONICAL_ORACLE,
    )
    status["source_artifacts"]["oracle_wrapper_timeout"]["sha256"] = "stale"
    status_path = tmp_path / "status.json"
    verify_path = tmp_path / "verify.json"
    status_path.write_text(json.dumps(status))

    rc = main(["--verify-source-artifacts", str(status_path), "--output", str(verify_path)])

    assert rc == 1
    verification = json.loads(verify_path.read_text())
    assert verification["status"] == "mismatch"
    assert verification["all_match"] is False
    assert verification["source_artifacts_all_match"] is False
    assert verification["source_artifact_failed_records"] == ["oracle_wrapper_timeout"]
    assert verification["records"]["oracle_wrapper_timeout"]["match"] is False
    assert verification["records"]["oracle_wrapper_timeout"]["matches"]["sha256"] is False
    assert "source_artifacts_sha256" in verification["status_integrity"]["failed_checks"]
