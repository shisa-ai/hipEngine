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
_REQUIRED_ACCEPTED_HARDWARE_FIELDS = (
    "gpu",
    "arch",
)
_REQUIRED_ACCEPTED_COMMAND_FIELDS = (
    "benchmark",
    "correctness_reference",
    "profiler",
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
        _validate_accepted_execution_gates(payload, errors)
        _validate_accepted_correctness_gates(payload, correctness, errors)
        _validate_accepted_measurement_gates(payload, errors)
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
    workload = _mapping_at(payload, "workload", errors)
    concurrency = workload.get("concurrency")
    concurrency_valid = isinstance(concurrency, int) and not isinstance(concurrency, bool) and concurrency > 1
    for field in _REQUIRED_ACCEPTED_OBSERVABILITY_FIELDS:
        if not isinstance(observability.get(field), Mapping):
            errors.append(f"observability.{field} must be an object for accepted artifacts")
    for field in ("admission_timestamps", "completion_timestamps"):
        row_map = observability.get(field)
        if concurrency_valid and isinstance(row_map, Mapping) and len(row_map) != concurrency:
            errors.append(f"observability.{field} length must match workload.concurrency for accepted artifacts")
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
        if concurrency_valid and len(per_request) != concurrency:
            errors.append("observability.per_request length must match workload.concurrency for accepted artifacts")
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


def _validate_accepted_execution_gates(payload: Mapping[str, Any], errors: list[str]) -> None:
    execution = _mapping_at(payload, "execution", errors)
    batch_execution = execution.get("batch_execution")
    if not isinstance(batch_execution, Mapping):
        errors.append("execution.batch_execution must be an object for accepted artifacts")
        return
    for field in _REQUIRED_BATCH_EXECUTION_FLAGS:
        if batch_execution.get(field) is not True:
            errors.append(f"execution.batch_execution.{field} must be true for accepted artifacts")
    path = batch_execution.get("path")
    if not isinstance(path, str) or not path:
        errors.append("execution.batch_execution.path must be a non-empty string for accepted artifacts")
    elif "serial" in path:
        errors.append("execution.batch_execution.path must not be a serial bridge for accepted artifacts")
    row_execution = batch_execution.get("row_execution")
    if not isinstance(row_execution, str) or not row_execution:
        errors.append("execution.batch_execution.row_execution must be a non-empty string for accepted artifacts")
    elif "serial" in row_execution or "fallback" in row_execution:
        errors.append("execution.batch_execution.row_execution must not contain serial or fallback for accepted artifacts")
    decode_execution = batch_execution.get("decode_execution")
    if isinstance(decode_execution, Mapping):
        if decode_execution.get("full_attention_decode_path") in {"per_row_splitk_fallback", "per_row_context_fallback"}:
            errors.append("execution.batch_execution.decode_execution.full_attention_decode_path must not be a per-row fallback for accepted artifacts")
        sampler_execution = decode_execution.get("sampler_execution")
        if isinstance(sampler_execution, Mapping) and sampler_execution.get("native_row_aware_lm_head") is not True:
            errors.append("execution.batch_execution.decode_execution.sampler_execution.native_row_aware_lm_head must be true for accepted artifacts")


def _validate_accepted_evidence_fields(payload: Mapping[str, Any], errors: list[str]) -> None:
    hardware = _mapping_at(payload, "hardware", errors)
    if not hardware:
        errors.append("hardware must be a non-empty object for accepted artifacts")
    for field in _REQUIRED_ACCEPTED_HARDWARE_FIELDS:
        if not isinstance(hardware.get(field), str) or not hardware.get(field):
            errors.append(f"hardware.{field} must be a non-empty string for accepted artifacts")
    software = _mapping_at(payload, "software", errors)
    if not isinstance(software.get("hipengine_commit"), str) or not software.get("hipengine_commit"):
        errors.append("software.hipengine_commit must be a non-empty string for accepted artifacts")
    if not isinstance(software.get("hipengine_dirty"), bool):
        errors.append("software.hipengine_dirty must be a bool for accepted artifacts")
    commands = _mapping_at(payload, "commands", errors)
    for field in _REQUIRED_ACCEPTED_COMMAND_FIELDS:
        if not isinstance(commands.get(field), str) or not commands.get(field):
            errors.append(f"commands.{field} must be a non-empty string for accepted artifacts")
    profiler = _mapping_at(payload, "profiler", errors)
    if profiler.get("status") != "captured":
        errors.append("profiler.status must be 'captured' for accepted artifacts")
    if profiler.get("expected_kernels_present") is not True:
        errors.append("profiler.expected_kernels_present must be true for accepted artifacts")


def _validate_accepted_correctness_gates(payload: Mapping[str, Any], correctness: Mapping[str, Any], errors: list[str]) -> None:
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
    batch_sequences = equality.get("batch_sequences")
    if not isinstance(batch_sequences, list):
        errors.append("correctness.generated_token_equality.batch_sequences must be a list for accepted artifacts")
    c1_sequences = equality.get("c1_sequences")
    if not isinstance(c1_sequences, list):
        errors.append("correctness.generated_token_equality.c1_sequences must be a list for accepted artifacts")
    workload = _mapping_at(payload, "workload", errors)
    concurrency = workload.get("concurrency")
    concurrency_valid = isinstance(concurrency, int) and not isinstance(concurrency, bool) and concurrency > 1
    if not concurrency_valid:
        errors.append("workload.concurrency must be an int > 1 for accepted artifacts")
    else:
        if isinstance(batch_sequences, list) and len(batch_sequences) != concurrency:
            errors.append("correctness.generated_token_equality.batch_sequences length must match workload.concurrency for accepted artifacts")
        if isinstance(c1_sequences, list) and len(c1_sequences) != concurrency:
            errors.append("correctness.generated_token_equality.c1_sequences length must match workload.concurrency for accepted artifacts")
    if isinstance(batch_sequences, list) and isinstance(c1_sequences, list) and batch_sequences != c1_sequences:
        errors.append("correctness.generated_token_equality.batch_sequences must equal c1_sequences for accepted artifacts")
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
    primitive_rows = primitive.get("rows")
    if not isinstance(primitive_rows, int) or isinstance(primitive_rows, bool):
        errors.append("correctness.primitive_batch_correctness.rows must be an int for accepted artifacts")
    if concurrency_valid and isinstance(primitive_rows, int) and not isinstance(primitive_rows, bool) and primitive_rows != concurrency:
        errors.append("correctness.primitive_batch_correctness.rows must match workload.concurrency for accepted artifacts")


def _validate_accepted_measurement_gates(payload: Mapping[str, Any], errors: list[str]) -> None:
    measurements = _mapping_at(payload, "measurements", errors)
    for field in ("decode_seconds", "decode_tok_s_aggregate", "decode_tok_s_per_request"):
        if not _is_positive_number(measurements.get(field)):
            errors.append(f"measurements.{field} must be positive numeric for accepted artifacts")
    decode_steps = measurements.get("decode_step_seconds")
    if not isinstance(decode_steps, Mapping):
        errors.append("measurements.decode_step_seconds must be an object for accepted artifacts")
        return
    samples = decode_steps.get("samples")
    if not isinstance(samples, list) or not samples:
        errors.append("measurements.decode_step_seconds.samples must be a non-empty list for accepted artifacts")
    elif any(not _is_positive_number(sample) for sample in samples):
        errors.append("measurements.decode_step_seconds.samples must contain only positive numbers for accepted artifacts")
    for field in ("median", "p95", "min", "max"):
        if not _is_positive_number(decode_steps.get(field)):
            errors.append(f"measurements.decode_step_seconds.{field} must be positive numeric for accepted artifacts")
    if not _is_nonnegative_number(decode_steps.get("stdev")):
        errors.append("measurements.decode_step_seconds.stdev must be non-negative numeric for accepted artifacts")


def _validate_accepted_scaling_gates(payload: Mapping[str, Any], errors: list[str]) -> None:
    scaling = _mapping_at(payload, "scaling", errors)
    workload = _mapping_at(payload, "workload", errors)
    concurrency = workload.get("concurrency")
    concurrency_valid = isinstance(concurrency, int) and not isinstance(concurrency, bool) and concurrency > 1
    prompt_tokens = workload.get("prompt_tokens_per_request")
    prompt_tokens_valid = isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) and prompt_tokens > 0
    if not prompt_tokens_valid:
        errors.append("workload.prompt_tokens_per_request must be an int > 0 for accepted artifacts")
    gen_tokens = workload.get("gen_tokens_per_request")
    gen_tokens_valid = isinstance(gen_tokens, int) and not isinstance(gen_tokens, bool) and gen_tokens > 0
    if not gen_tokens_valid:
        errors.append("workload.gen_tokens_per_request must be an int > 0 for accepted artifacts")
    if scaling.get("complete") is not True:
        errors.append("scaling.complete must be true for accepted artifacts")
    native = scaling.get("native")
    if not isinstance(native, Mapping):
        errors.append("scaling.native must be an object for accepted artifacts")
    else:
        for field in ("decode_tok_s_aggregate", "decode_tok_s_per_request"):
            if not _is_positive_number(native.get(field)):
                errors.append(f"scaling.native.{field} must be positive numeric for accepted artifacts")
    for baseline_name in _REQUIRED_ACCEPTED_SCALING_BASELINES:
        baseline = scaling.get(baseline_name)
        if not isinstance(baseline, Mapping):
            errors.append(f"scaling.{baseline_name} must be an object for accepted artifacts")
            continue
        if not isinstance(baseline.get("artifact_path"), str) or not baseline.get("artifact_path"):
            errors.append(f"scaling.{baseline_name}.artifact_path must be a non-empty string for accepted artifacts")
        status = baseline.get("status")
        if not isinstance(status, str) or not status:
            errors.append(f"scaling.{baseline_name}.status must be a non-empty string for accepted artifacts")
        elif status in {"missing", "invalid_json"}:
            errors.append(f"scaling.{baseline_name}.status must be usable for accepted artifacts")
        if baseline.get("reason") is not None:
            errors.append(f"scaling.{baseline_name}.reason must be null for accepted artifacts")
        for field in ("decode_tok_s_aggregate", "decode_tok_s_per_request"):
            if not _is_positive_number(baseline.get(field)):
                errors.append(f"scaling.{baseline_name}.{field} must be positive numeric for accepted artifacts")
        baseline_prompt_tokens = baseline.get("prompt_tokens_per_request")
        if not isinstance(baseline_prompt_tokens, int) or isinstance(baseline_prompt_tokens, bool):
            errors.append(f"scaling.{baseline_name}.prompt_tokens_per_request must be an int for accepted artifacts")
        elif prompt_tokens_valid and baseline_prompt_tokens != prompt_tokens:
            errors.append(f"scaling.{baseline_name}.prompt_tokens_per_request must match workload.prompt_tokens_per_request for accepted artifacts")
        baseline_gen_tokens = baseline.get("gen_tokens_per_request")
        if not isinstance(baseline_gen_tokens, int) or isinstance(baseline_gen_tokens, bool):
            errors.append(f"scaling.{baseline_name}.gen_tokens_per_request must be an int for accepted artifacts")
        elif gen_tokens_valid and baseline_gen_tokens != gen_tokens:
            errors.append(f"scaling.{baseline_name}.gen_tokens_per_request must match workload.gen_tokens_per_request for accepted artifacts")
    c1_baseline = scaling.get("c1_baseline")
    if isinstance(c1_baseline, Mapping):
        c1_concurrency = c1_baseline.get("workload_concurrency")
        if not isinstance(c1_concurrency, int) or isinstance(c1_concurrency, bool):
            errors.append("scaling.c1_baseline.workload_concurrency must be an int for accepted artifacts")
        elif c1_concurrency != 1:
            errors.append("scaling.c1_baseline.workload_concurrency must be 1 for accepted artifacts")
    serial_baseline = scaling.get("serial_bridge_baseline")
    if isinstance(serial_baseline, Mapping):
        serial_concurrency = serial_baseline.get("workload_concurrency")
        if not isinstance(serial_concurrency, int) or isinstance(serial_concurrency, bool):
            errors.append("scaling.serial_bridge_baseline.workload_concurrency must be an int for accepted artifacts")
        elif concurrency_valid and serial_concurrency != concurrency:
            errors.append("scaling.serial_bridge_baseline.workload_concurrency must match workload.concurrency for accepted artifacts")
    ratios = scaling.get("ratios")
    if not isinstance(ratios, Mapping):
        errors.append("scaling.ratios must be an object for accepted artifacts")
    else:
        for field in _REQUIRED_ACCEPTED_SCALING_RATIOS:
            if not _is_positive_number(ratios.get(field)):
                errors.append(f"scaling.ratios.{field} must be positive numeric for accepted artifacts")
        if isinstance(native, Mapping) and isinstance(c1_baseline, Mapping) and isinstance(serial_baseline, Mapping):
            _validate_scaling_ratio(
                "aggregate_vs_c1",
                ratios,
                native,
                "decode_tok_s_aggregate",
                c1_baseline,
                "decode_tok_s_aggregate",
                errors,
            )
            _validate_scaling_ratio(
                "per_request_vs_c1",
                ratios,
                native,
                "decode_tok_s_per_request",
                c1_baseline,
                "decode_tok_s_per_request",
                errors,
            )
            _validate_scaling_ratio(
                "aggregate_vs_serial_bridge",
                ratios,
                native,
                "decode_tok_s_aggregate",
                serial_baseline,
                "decode_tok_s_aggregate",
                errors,
            )
            _validate_scaling_ratio(
                "per_request_vs_serial_bridge",
                ratios,
                native,
                "decode_tok_s_per_request",
                serial_baseline,
                "decode_tok_s_per_request",
                errors,
            )


def _validate_scaling_ratio(
    field: str,
    ratios: Mapping[str, Any],
    numerator_payload: Mapping[str, Any],
    numerator_field: str,
    denominator_payload: Mapping[str, Any],
    denominator_field: str,
    errors: list[str],
) -> None:
    numerator = numerator_payload.get(numerator_field)
    denominator = denominator_payload.get(denominator_field)
    actual = ratios.get(field)
    if not (_is_number(numerator) and _is_number(denominator) and _is_number(actual)):
        return
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        errors.append(f"scaling.ratios.{field} denominator must be positive for accepted artifacts")
        return
    expected = float(numerator) / denominator_value
    tolerance = max(1e-9, abs(expected) * 1e-6)
    if abs(float(actual) - expected) > tolerance:
        errors.append(f"scaling.ratios.{field} must match scaling throughput fields for accepted artifacts")


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


def _is_positive_number(value: Any) -> bool:
    return _is_number(value) and float(value) > 0.0


def _is_nonnegative_number(value: Any) -> bool:
    return _is_number(value) and float(value) >= 0.0


__all__ = ["validate_cn_diagnostic_artifact_payload"]
