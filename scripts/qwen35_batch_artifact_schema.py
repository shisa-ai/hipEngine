from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_REQUIRED_WORKLOAD_FLAGS = (
    "native_compact_prefill",
    "native_caware_decode",
)
_REQUIRED_BATCH_EXECUTION_FLAGS = (
    "native_compact_prefill",
    "native_caware_decode",
    "throughput_claim_eligible",
)


def _mapping_at(payload: Mapping[str, Any], key: str, errors: list[str]) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{key} must be an object")
        return {}
    return value


def validate_cn_diagnostic_artifact_payload(payload: Mapping[str, Any]) -> None:
    """Validate c>N diagnostic/retained benchmark artifact labeling fields.

    This is intentionally a small schema guard for the fields that prevent c>N
    artifacts from being misread. It does not replace the full benchmark
    protocol in ``docs/BENCHMARK.md``; it only ensures every emitted c>N batch
    artifact distinguishes workload intent, execution path, correctness status,
    and throughput-claim eligibility.
    """

    errors: list[str] = []
    status = payload.get("status")
    if not isinstance(status, str) or not status:
        errors.append("status must be a non-empty string")

    workload = _mapping_at(payload, "workload", errors)
    for field in _REQUIRED_WORKLOAD_FLAGS:
        if not isinstance(workload.get(field), bool):
            errors.append(f"workload.{field} must be a bool")

    correctness = _mapping_at(payload, "correctness", errors)
    if not isinstance(correctness.get("passed"), bool):
        errors.append("correctness.passed must be a bool")

    execution = _mapping_at(payload, "execution", errors)
    batch_execution = execution.get("batch_execution")
    if not isinstance(batch_execution, Mapping):
        errors.append("execution.batch_execution must be an object")
        batch_execution = {}
    for field in _REQUIRED_BATCH_EXECUTION_FLAGS:
        if not isinstance(batch_execution.get(field), bool):
            errors.append(f"execution.batch_execution.{field} must be a bool")

    decision = _mapping_at(payload, "decision", errors)
    if not isinstance(decision.get("accepted"), bool):
        errors.append("decision.accepted must be a bool")

    performance_claim = payload.get("performance_claim")
    if not isinstance(performance_claim, bool):
        errors.append("performance_claim must be a bool")

    if errors:
        raise ValueError("invalid c>N diagnostic artifact payload: " + "; ".join(errors))


__all__ = ["validate_cn_diagnostic_artifact_payload"]
