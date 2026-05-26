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
_REQUIRED_ACCEPTED_OBSERVABILITY_FIELDS = (
    "admission_timestamps",
    "completion_timestamps",
    "request_latency_seconds",
)
_REQUIRED_ACCEPTED_POOL_FIELDS = (
    "dynamic_pool",
    "stable_block_id",
    "prefix_sharing",
)
_REQUIRED_ACCEPTED_PER_REQUEST_OBSERVABILITY_FIELDS = (
    "queue_seconds",
    "prefill_seconds",
    "decode_seconds",
    "kv_pages_owned",
    "kv_pages_peak",
    "bucket_key",
    "admission_blocked_reason",
    "finish_reason",
)
_REQUIRED_ACCEPTED_POOL_COUNTER_FIELDS = (
    "current_bytes",
    "high_water_observed_bytes",
    "grow_events",
    "grow_failures",
    "shrink_events",
    "free_pages",
    "refcounted_pages",
)
_REQUIRED_ACCEPTED_SCALING_BASELINES = (
    "c1_baseline",
    "serial_bridge_baseline",
)
_REQUIRED_ACCEPTED_SCALING_RATIOS = (
    "aggregate_vs_c1",
    "per_request_vs_c1",
    "aggregate_vs_serial_bridge",
    "per_request_vs_serial_bridge",
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

    accepted = bool(decision.get("accepted"))
    if accepted or status == "accepted" or performance_claim is True:
        _validate_accepted_retained_gates(payload, errors)
        _validate_accepted_correctness_gates(correctness, errors)
        _validate_accepted_scaling_gates(payload, errors)
        _validate_accepted_evidence_fields(payload, errors)

    if errors:
        raise ValueError("invalid c>N diagnostic artifact payload: " + "; ".join(errors))


def _validate_accepted_retained_gates(payload: Mapping[str, Any], errors: list[str]) -> None:
    if payload.get("status") != "accepted":
        errors.append("accepted retained artifact must have status='accepted'")
    if payload.get("performance_claim") is not True:
        errors.append("accepted retained artifact must set performance_claim=true")

    observability = _mapping_at(payload, "observability", errors)
    for field in _REQUIRED_ACCEPTED_OBSERVABILITY_FIELDS:
        if not isinstance(observability.get(field), Mapping):
            errors.append(f"observability.{field} must be an object for accepted artifacts")
    latency = observability.get("request_latency_seconds")
    if isinstance(latency, Mapping):
        if not _is_number(latency.get("p50")):
            errors.append("observability.request_latency_seconds.p50 must be numeric for accepted artifacts")
        if not _is_number(latency.get("p95")):
            errors.append("observability.request_latency_seconds.p95 must be numeric for accepted artifacts")
    per_request = observability.get("per_request")
    if not isinstance(per_request, Mapping) or not per_request:
        errors.append("observability.per_request must be a non-empty object for accepted artifacts")
    else:
        for row in per_request.values():
            _valid_request_observability(row, errors)

    memory = _mapping_at(payload, "memory", errors)
    for field in _REQUIRED_ACCEPTED_POOL_FIELDS:
        if not isinstance(memory.get(field), Mapping):
            errors.append(f"memory.{field} must be an object for accepted artifacts")
    dynamic_pool = memory.get("dynamic_pool")
    if isinstance(dynamic_pool, Mapping):
        if not isinstance(dynamic_pool.get("evidence"), str):
            errors.append("memory.dynamic_pool.evidence must be a string for accepted artifacts")
        pool_counters = dynamic_pool.get("pool_counters")
        if not isinstance(pool_counters, Mapping):
            errors.append("memory.dynamic_pool.pool_counters must be an object for accepted artifacts")
        else:
            for field in _REQUIRED_ACCEPTED_POOL_COUNTER_FIELDS:
                if not _is_number(pool_counters.get(field)):
                    errors.append(f"memory.dynamic_pool.pool_counters.{field} must be numeric for accepted artifacts")
    stable_block_id = memory.get("stable_block_id")
    if isinstance(stable_block_id, Mapping) and stable_block_id.get("passed") is not True:
        errors.append("memory.stable_block_id.passed must be true for accepted artifacts")
    prefix_sharing = memory.get("prefix_sharing")
    if isinstance(prefix_sharing, Mapping):
        if not isinstance(prefix_sharing.get("enabled"), bool):
            errors.append("memory.prefix_sharing.enabled must be a bool for accepted artifacts")
        if not _is_number(prefix_sharing.get("savings_bytes")):
            errors.append("memory.prefix_sharing.savings_bytes must be numeric for accepted artifacts")


def _validate_accepted_evidence_fields(payload: Mapping[str, Any], errors: list[str]) -> None:
    hardware = _mapping_at(payload, "hardware", errors)
    if not hardware:
        errors.append("hardware must be a non-empty object for accepted artifacts")
    software = _mapping_at(payload, "software", errors)
    if not isinstance(software.get("hipengine_commit"), str) or not software.get("hipengine_commit"):
        errors.append("software.hipengine_commit must be a non-empty string for accepted artifacts")
    if not isinstance(software.get("hipengine_dirty"), bool):
        errors.append("software.hipengine_dirty must be a bool for accepted artifacts")
    commands = _mapping_at(payload, "commands", errors)
    if not isinstance(commands.get("benchmark"), str) or not commands.get("benchmark"):
        errors.append("commands.benchmark must be a non-empty string for accepted artifacts")
    if not isinstance(commands.get("profiler"), str) or not commands.get("profiler"):
        errors.append("commands.profiler must be a non-empty string for accepted artifacts")
    profiler = _mapping_at(payload, "profiler", errors)
    if profiler.get("status") != "captured":
        errors.append("profiler.status must be 'captured' for accepted artifacts")
    if profiler.get("expected_kernels_present") is not True:
        errors.append("profiler.expected_kernels_present must be true for accepted artifacts")


def _validate_accepted_correctness_gates(correctness: Mapping[str, Any], errors: list[str]) -> None:
    if correctness.get("passed") is not True:
        errors.append("correctness.passed must be true for accepted artifacts")
    equality = correctness.get("generated_token_equality")
    if not isinstance(equality, Mapping):
        errors.append("correctness.generated_token_equality must be an object for accepted artifacts")
        return
    if equality.get("passed") is not True:
        errors.append("correctness.generated_token_equality.passed must be true for accepted artifacts")
    if equality.get("skipped") is not False:
        errors.append("correctness.generated_token_equality.skipped must be false for accepted artifacts")
    if not isinstance(equality.get("batch_sequences"), list):
        errors.append("correctness.generated_token_equality.batch_sequences must be a list for accepted artifacts")
    if not isinstance(equality.get("c1_sequences"), list):
        errors.append("correctness.generated_token_equality.c1_sequences must be a list for accepted artifacts")
    mismatches = equality.get("mismatches")
    if not isinstance(mismatches, list):
        errors.append("correctness.generated_token_equality.mismatches must be a list for accepted artifacts")
    elif mismatches:
        errors.append("correctness.generated_token_equality.mismatches must be empty for accepted artifacts")
    primitive = correctness.get("primitive_batch_correctness")
    if not isinstance(primitive, Mapping):
        errors.append("correctness.primitive_batch_correctness must be an object for accepted artifacts")
        return
    if primitive.get("passed") is not True:
        errors.append("correctness.primitive_batch_correctness.passed must be true for accepted artifacts")
    if not isinstance(primitive.get("artifact_path"), str) or not primitive.get("artifact_path"):
        errors.append("correctness.primitive_batch_correctness.artifact_path must be a non-empty string for accepted artifacts")
    if not isinstance(primitive.get("rows"), int):
        errors.append("correctness.primitive_batch_correctness.rows must be an int for accepted artifacts")


def _validate_accepted_scaling_gates(payload: Mapping[str, Any], errors: list[str]) -> None:
    scaling = _mapping_at(payload, "scaling", errors)
    if scaling.get("complete") is not True:
        errors.append("scaling.complete must be true for accepted artifacts")
    native = scaling.get("native")
    if not isinstance(native, Mapping):
        errors.append("scaling.native must be an object for accepted artifacts")
    else:
        for field in ("decode_tok_s_aggregate", "decode_tok_s_per_request"):
            if not _is_number(native.get(field)):
                errors.append(f"scaling.native.{field} must be numeric for accepted artifacts")
    for baseline_name in _REQUIRED_ACCEPTED_SCALING_BASELINES:
        baseline = scaling.get(baseline_name)
        if not isinstance(baseline, Mapping):
            errors.append(f"scaling.{baseline_name} must be an object for accepted artifacts")
            continue
        if not isinstance(baseline.get("artifact_path"), str) or not baseline.get("artifact_path"):
            errors.append(f"scaling.{baseline_name}.artifact_path must be a non-empty string for accepted artifacts")
        for field in ("decode_tok_s_aggregate", "decode_tok_s_per_request"):
            if not _is_number(baseline.get(field)):
                errors.append(f"scaling.{baseline_name}.{field} must be numeric for accepted artifacts")
    ratios = scaling.get("ratios")
    if not isinstance(ratios, Mapping):
        errors.append("scaling.ratios must be an object for accepted artifacts")
    else:
        for field in _REQUIRED_ACCEPTED_SCALING_RATIOS:
            if not _is_number(ratios.get(field)):
                errors.append(f"scaling.ratios.{field} must be numeric for accepted artifacts")


def _valid_request_observability(row: Any, errors: list[str]) -> bool:
    if not isinstance(row, Mapping):
        errors.append("observability.per_request entries must be objects for accepted artifacts")
        return False
    ok = True
    for field in _REQUIRED_ACCEPTED_PER_REQUEST_OBSERVABILITY_FIELDS:
        if field not in row:
            errors.append(f"observability.per_request.*.{field} is required for accepted artifacts")
            ok = False
    for field in ("queue_seconds", "prefill_seconds", "decode_seconds"):
        if field in row and not _is_number(row.get(field)):
            errors.append(f"observability.per_request.*.{field} must be numeric for accepted artifacts")
            ok = False
    for field in ("kv_pages_owned", "kv_pages_peak"):
        if field in row and not isinstance(row.get(field), int):
            errors.append(f"observability.per_request.*.{field} must be an int for accepted artifacts")
            ok = False
    if "bucket_key" in row and row.get("bucket_key") is not None and not isinstance(row.get("bucket_key"), str):
        errors.append("observability.per_request.*.bucket_key must be a string or null for accepted artifacts")
        ok = False
    if "admission_blocked_reason" in row and row.get("admission_blocked_reason") is not None and not isinstance(row.get("admission_blocked_reason"), str):
        errors.append("observability.per_request.*.admission_blocked_reason must be a string or null for accepted artifacts")
        ok = False
    if "finish_reason" in row and not isinstance(row.get("finish_reason"), str):
        errors.append("observability.per_request.*.finish_reason must be a string for accepted artifacts")
        ok = False
    return ok


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


__all__ = ["validate_cn_diagnostic_artifact_payload"]
