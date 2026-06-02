from __future__ import annotations

import hashlib
import json
from pathlib import Path


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
    # rewrote the canonical machine-readable oracle artifact, so the canonical
    # 60 s timeout remains the source of truth for status automation.
    assert artifact["output_artifact_status_after_attempt"] == canonical["status"] == "timeout"
    assert artifact["output_artifact_timeout_s_after_attempt"] == canonical["timeout_s"] == 60.0
    assert artifact["output_artifact_sha256_after_attempt"] == canonical_sha256

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
