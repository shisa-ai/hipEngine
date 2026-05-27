from __future__ import annotations

import re
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
_REQUIRED_ACCEPTED_WORKLOAD_LABELS = (
    "model",
    "quant",
    "kv_storage_dtype",
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
_REQUIRED_ACCEPTED_HARDWARE_CAPTURE_FIELDS = (
    "rocminfo",
    "rocm_smi",
)
_REQUIRED_ACCEPTED_COMMAND_FIELDS = (
    "benchmark",
    "correctness_reference",
    "profiler",
)
_REQUIRED_ACCEPTED_ENVIRONMENT_COMMAND_FRAGMENTS = (
    "rocminfo",
    "rocm-smi",
    "hipcc --version",
    "git rev-parse HEAD",
    "git diff --quiet",
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
_COMMAND_BATCH_SIZE_RE = re.compile(r"(?:^|\s)--batch-size(?:=|\s+)(\d+)(?=\s|$)")
_COMMAND_DECODE_TOKENS_RE = re.compile(r"(?:^|\s)--decode-tokens(?:=|\s+)(\d+)(?=\s|$)")
_COMMAND_MAX_LAYERS_RE = re.compile(r"(?:^|\s)--max-layers(?:=|\s+)(\d+)(?=\s|$)")
_COMMAND_MODEL_RE = re.compile(r"(?:^|\s)--model(?:=|\s+)(\S+)(?=\s|$)")
_COMMAND_FIXTURE_RE = re.compile(r"(?:^|\s)--fixture(?:=|\s+)(\S+)(?=\s|$)")
_COMMAND_PROMPT_LENGTH_RE = re.compile(r"(?:^|\s)--prompt-length(?:=|\s+)(\d+)(?=\s|$)")
_COMMAND_JSON_RE = re.compile(r"(?:^|\s)--json(?:=|\s+)(\S+)(?=\s|$)")
_COMMAND_PROFILER_JSON_RE = re.compile(r"(?:^|\s)--profiler-json(?:=|\s+)(\S+)(?=\s|$)")
_COMMAND_C1_BASELINE_JSON_RE = re.compile(r"(?:^|\s)--c1-baseline-json(?:=|\s+)(\S+)(?=\s|$)")
_COMMAND_SERIAL_BRIDGE_JSON_RE = re.compile(r"(?:^|\s)--serial-bridge-json(?:=|\s+)(\S+)(?=\s|$)")
_COMMAND_PRIMITIVE_CORRECTNESS_JSON_RE = re.compile(r"(?:^|\s)--primitive-correctness-json(?:=|\s+)(\S+)(?=\s|$)")
_COMMAND_COMPILER_VERSION_FILE_RE = re.compile(r"(?:^|\s)--compiler-version-file(?:=|\s+)(\S+)(?=\s|$)")
_CORRECTNESS_ROWS_RE = re.compile(r"(?:^|\s)--rows(?:=|\s+)(\d+)(?=\s|$)")
_FULL_COMMIT_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", re.IGNORECASE)
_DISALLOWED_PROFILER_KERNEL_NAME_FRAGMENTS = ("serial", "fallback", "per_row", "per-row")


def _mapping_at(payload: Mapping[str, Any], key: str, errors: list[str]) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{key} must be an object")
        return {}
    return value


def _validate_command_workload_shape(command: str, *, field: str, payload: Mapping[str, Any], errors: list[str]) -> None:
    workload = payload.get("workload")
    if not isinstance(workload, Mapping):
        return

    expected_fields = (
        (_COMMAND_BATCH_SIZE_RE, "--batch-size", "concurrency"),
        (_COMMAND_PROMPT_LENGTH_RE, "--prompt-length", "prompt_tokens_per_request"),
        (_COMMAND_DECODE_TOKENS_RE, "--decode-tokens", "gen_tokens_per_request"),
        (_COMMAND_MAX_LAYERS_RE, "--max-layers", "max_layers"),
    )
    for pattern, flag, workload_key in expected_fields:
        match = pattern.search(command)
        if match is None:
            errors.append(f"commands.{field} must include {flag} <workload.{workload_key}> for accepted artifacts")
            continue
        expected = workload.get(workload_key)
        if isinstance(expected, int) and not isinstance(expected, bool) and int(match.group(1)) != expected:
            errors.append(f"commands.{field} {flag} must match workload.{workload_key} for accepted artifacts")


def _command_json_path(command: str) -> str | None:
    json_match = _COMMAND_JSON_RE.search(command)
    if json_match is None:
        return None
    return json_match.group(1).strip("'\"")


def _validate_command_json_artifact_path(command: str, *, field: str, errors: list[str]) -> None:
    json_path = _command_json_path(command)
    if json_path is None:
        errors.append(f"commands.{field} must include --json <benchmarks/results/...> for accepted artifacts")
        return
    _validate_benchmark_results_artifact_path(f"commands.{field} --json path", json_path, errors)


def _validate_command_json_matches_artifact_path(
    command: str,
    *,
    field: str,
    artifact_field: str,
    artifact_path: Any,
    errors: list[str],
) -> None:
    json_path = _command_json_path(command)
    if json_path is None:
        errors.append(f"commands.{field} must include --json <{artifact_field}> for accepted artifacts")
        return
    _validate_benchmark_results_artifact_path(f"commands.{field} --json path", json_path, errors)
    if isinstance(artifact_path, str) and artifact_path and json_path != artifact_path:
        errors.append(f"commands.{field} --json path must match {artifact_field} for accepted artifacts")


def _validate_command_model_fixture_flags(command: str, *, field: str, errors: list[str]) -> None:
    if _COMMAND_MODEL_RE.search(command) is None:
        errors.append(f"commands.{field} must include --model for accepted artifacts")
    if _COMMAND_FIXTURE_RE.search(command) is None:
        errors.append(f"commands.{field} must include --fixture for accepted artifacts")


def _validate_command_flag_matches_artifact_path(
    command: str,
    *,
    field: str,
    flag: str,
    pattern: re.Pattern[str],
    artifact_field: str,
    artifact_path: Any,
    errors: list[str],
) -> None:
    match = pattern.search(command)
    if match is None:
        errors.append(f"commands.{field} must include {flag} <{artifact_field}> for accepted artifacts")
        return
    command_path = match.group(1).strip("'\"")
    _validate_benchmark_results_artifact_path(f"commands.{field} {flag} path", command_path, errors)
    if isinstance(artifact_path, str) and artifact_path and command_path != artifact_path:
        errors.append(f"commands.{field} {flag} path must match {artifact_field} for accepted artifacts")


def _reference_artifact_paths(payload: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    correctness = payload.get("correctness")
    primitive = correctness.get("primitive_batch_correctness") if isinstance(correctness, Mapping) else None
    primitive_artifact_path = primitive.get("artifact_path") if isinstance(primitive, Mapping) else None
    scaling = payload.get("scaling")
    c1 = scaling.get("c1_baseline") if isinstance(scaling, Mapping) else None
    serial = scaling.get("serial_bridge_baseline") if isinstance(scaling, Mapping) else None
    c1_artifact_path = c1.get("artifact_path") if isinstance(c1, Mapping) else None
    serial_artifact_path = serial.get("artifact_path") if isinstance(serial, Mapping) else None
    return primitive_artifact_path, c1_artifact_path, serial_artifact_path


def _validate_retained_benchmark_reference_paths(command: str, *, field: str, payload: Mapping[str, Any], errors: list[str]) -> None:
    primitive_artifact_path, c1_artifact_path, serial_artifact_path = _reference_artifact_paths(payload)
    _validate_command_flag_matches_artifact_path(
        command,
        field=field,
        flag="--c1-baseline-json",
        pattern=_COMMAND_C1_BASELINE_JSON_RE,
        artifact_field="scaling.c1_baseline.artifact_path",
        artifact_path=c1_artifact_path,
        errors=errors,
    )
    _validate_command_flag_matches_artifact_path(
        command,
        field=field,
        flag="--serial-bridge-json",
        pattern=_COMMAND_SERIAL_BRIDGE_JSON_RE,
        artifact_field="scaling.serial_bridge_baseline.artifact_path",
        artifact_path=serial_artifact_path,
        errors=errors,
    )
    _validate_command_flag_matches_artifact_path(
        command,
        field=field,
        flag="--primitive-correctness-json",
        pattern=_COMMAND_PRIMITIVE_CORRECTNESS_JSON_RE,
        artifact_field="correctness.primitive_batch_correctness.artifact_path",
        artifact_path=primitive_artifact_path,
        errors=errors,
    )


def _validate_profiler_command_artifact_reference(command: str, profiler_artifact_path: str, errors: list[str]) -> None:
    profiler_json_match = _COMMAND_PROFILER_JSON_RE.search(command)
    if profiler_json_match is None:
        errors.append("commands.profiler must include --profiler-json <profiler.artifact_path> for accepted artifacts")
        return
    command_profiler_path = profiler_json_match.group(1).strip("'\"")
    if command_profiler_path != profiler_artifact_path:
        errors.append("commands.profiler --profiler-json path must match profiler.artifact_path for accepted artifacts")


def _has_disallowed_profiler_kernel_fragment(name: str) -> bool:
    lowered = name.lower()
    return any(fragment in lowered for fragment in _DISALLOWED_PROFILER_KERNEL_NAME_FRAGMENTS)


def _validate_expected_profiler_kernel_names(expected_kernel_names: list[Any], errors: list[str]) -> None:
    if not any(isinstance(name, str) and "batch" in name.lower() for name in expected_kernel_names):
        errors.append("profiler.expected_kernel_names must include at least one native batch kernel name for accepted artifacts")
    for name in expected_kernel_names:
        if isinstance(name, str) and _has_disallowed_profiler_kernel_fragment(name):
            errors.append("profiler.expected_kernel_names must not include serial/per-row/fallback kernel names for accepted artifacts")
            break


def _is_benchmark_results_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    return normalized.startswith("benchmarks/results/") or "/benchmarks/results/" in normalized


def _validate_benchmark_results_artifact_path(field: str, value: Any, errors: list[str]) -> None:
    if isinstance(value, str) and value and not _is_benchmark_results_path(value):
        errors.append(f"{field} must be under benchmarks/results for accepted artifacts")


def _validate_capture_context(
    field: str,
    value: Any,
    errors: list[str],
    *,
    command_fragment: str | None = None,
) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{field} must be an object for accepted artifacts")
        return
    command = value.get("command")
    if not isinstance(command, str) or not command:
        errors.append(f"{field}.command must be a non-empty string for accepted artifacts")
    elif command_fragment is not None and command_fragment not in command:
        errors.append(f"{field}.command must include {command_fragment} for accepted artifacts")
    if value.get("returncode") != 0:
        errors.append(f"{field}.returncode must be 0 for accepted artifacts")
    if not isinstance(value.get("output"), str) or not value.get("output"):
        errors.append(f"{field}.output must be a non-empty string for accepted artifacts")


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
        if not _is_positive_number(latency.get("p50")):
            errors.append("observability.request_latency_seconds.p50 must be positive numeric for accepted artifacts")
        if not _is_positive_number(latency.get("p95")):
            errors.append("observability.request_latency_seconds.p95 must be positive numeric for accepted artifacts")
        samples = latency.get("samples")
        if not isinstance(samples, list) or not samples:
            errors.append("observability.request_latency_seconds.samples must be a non-empty list for accepted artifacts")
        else:
            if any(not _is_positive_number(sample) for sample in samples):
                errors.append("observability.request_latency_seconds.samples must contain only positive numbers for accepted artifacts")
            if concurrency_valid and len(samples) != concurrency:
                errors.append("observability.request_latency_seconds.samples length must match workload.concurrency for accepted artifacts")
    per_request = observability.get("per_request")
    if not isinstance(per_request, Mapping) or not per_request:
        errors.append("observability.per_request must be a non-empty object for accepted artifacts")
    else:
        if concurrency_valid and len(per_request) != concurrency:
            errors.append("observability.per_request length must match workload.concurrency for accepted artifacts")
        per_request_keys = set(per_request.keys())
        for field in ("admission_timestamps", "completion_timestamps"):
            row_map = observability.get(field)
            if isinstance(row_map, Mapping) and set(row_map.keys()) != per_request_keys:
                errors.append(f"observability.{field} keys must match observability.per_request keys for accepted artifacts")
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
    workload = _mapping_at(payload, "workload", errors)
    batch_execution = execution.get("batch_execution")
    if not isinstance(batch_execution, Mapping):
        errors.append("execution.batch_execution must be an object for accepted artifacts")
        return
    for field in _REQUIRED_BATCH_EXECUTION_FLAGS:
        if batch_execution.get(field) is not True:
            errors.append(f"execution.batch_execution.{field} must be true for accepted artifacts")
    path = batch_execution.get("path")
    path_valid = isinstance(path, str) and bool(path)
    if not path_valid:
        errors.append("execution.batch_execution.path must be a non-empty string for accepted artifacts")
    elif "serial" in path:
        errors.append("execution.batch_execution.path must not be a serial bridge for accepted artifacts")
    scheduler_path = workload.get("scheduler_path")
    if not isinstance(scheduler_path, str) or not scheduler_path:
        errors.append("workload.scheduler_path must be a non-empty string for accepted artifacts")
    elif path_valid and scheduler_path != path:
        errors.append("workload.scheduler_path must match execution.batch_execution.path for accepted artifacts")
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
    scheduler_metadata = execution.get("scheduler_metadata")
    if not isinstance(scheduler_metadata, Mapping):
        errors.append("execution.scheduler_metadata must be an object for accepted artifacts")
    else:
        _validate_accepted_scheduler_metadata(scheduler_metadata, workload, errors)


def _validate_accepted_scheduler_metadata(
    scheduler_metadata: Mapping[str, Any],
    workload: Mapping[str, Any],
    errors: list[str],
) -> None:
    decode_shape_key = scheduler_metadata.get("decode_shape_key")
    if not isinstance(decode_shape_key, Mapping):
        errors.append("execution.scheduler_metadata.decode_shape_key must be an object for accepted artifacts")
    else:
        if decode_shape_key.get("mode") != "decode":
            errors.append("execution.scheduler_metadata.decode_shape_key.mode must be decode for accepted artifacts")
        active_c = decode_shape_key.get("active_c")
        concurrency = workload.get("concurrency")
        if isinstance(concurrency, int) and not isinstance(concurrency, bool) and active_c != concurrency:
            errors.append("execution.scheduler_metadata.decode_shape_key.active_c must match workload.concurrency for accepted artifacts")
        active_mask = decode_shape_key.get("active_mask")
        if not isinstance(active_mask, list) or not active_mask or any(not isinstance(item, bool) for item in active_mask):
            errors.append("execution.scheduler_metadata.decode_shape_key.active_mask must be a non-empty bool list for accepted artifacts")
    graph_stats = scheduler_metadata.get("graph_bucket_stats")
    if not isinstance(graph_stats, Mapping):
        errors.append("execution.scheduler_metadata.graph_bucket_stats must be an object for accepted artifacts")
    else:
        for field in ("entries", "hits", "misses"):
            if not isinstance(graph_stats.get(field), int) or isinstance(graph_stats.get(field), bool) or graph_stats.get(field) < 0:
                errors.append(f"execution.scheduler_metadata.graph_bucket_stats.{field} must be a non-negative int for accepted artifacts")
        entries = graph_stats.get("entries")
        if isinstance(entries, int) and not isinstance(entries, bool) and entries <= 0:
            errors.append("execution.scheduler_metadata.graph_bucket_stats.entries must be positive for accepted artifacts")


def _validate_accepted_evidence_fields(payload: Mapping[str, Any], errors: list[str]) -> None:
    hardware = _mapping_at(payload, "hardware", errors)
    if not hardware:
        errors.append("hardware must be a non-empty object for accepted artifacts")
    for field in _REQUIRED_ACCEPTED_HARDWARE_FIELDS:
        if not isinstance(hardware.get(field), str) or not hardware.get(field):
            errors.append(f"hardware.{field} must be a non-empty string for accepted artifacts")
    for field in _REQUIRED_ACCEPTED_HARDWARE_CAPTURE_FIELDS:
        command_fragment = "rocm-smi" if field == "rocm_smi" else field
        _validate_capture_context(
            f"hardware.{field}",
            hardware.get(field),
            errors,
            command_fragment=command_fragment,
        )
    hardware_arch = hardware.get("arch")
    rocminfo = hardware.get("rocminfo")
    if isinstance(hardware_arch, str) and hardware_arch and isinstance(rocminfo, Mapping):
        rocminfo_output = rocminfo.get("output")
        if isinstance(rocminfo_output, str) and hardware_arch not in rocminfo_output:
            errors.append("hardware.rocminfo.output must include hardware.arch for accepted artifacts")
    software = _mapping_at(payload, "software", errors)
    hipengine_commit = software.get("hipengine_commit")
    if not isinstance(hipengine_commit, str) or not hipengine_commit:
        errors.append("software.hipengine_commit must be a non-empty string for accepted artifacts")
    elif not _FULL_COMMIT_RE.fullmatch(hipengine_commit):
        errors.append("software.hipengine_commit must be a full commit hash for accepted artifacts")
    hipengine_dirty = software.get("hipengine_dirty")
    if not isinstance(hipengine_dirty, bool):
        errors.append("software.hipengine_dirty must be a bool for accepted artifacts")
    elif hipengine_dirty:
        errors.append("software.hipengine_dirty must be false for accepted artifacts")
    if not isinstance(software.get("hipcc_version"), str) or not software.get("hipcc_version"):
        errors.append("software.hipcc_version must be a non-empty string for accepted artifacts")
    commands = _mapping_at(payload, "commands", errors)
    for field in _REQUIRED_ACCEPTED_COMMAND_FIELDS:
        if not isinstance(commands.get(field), str) or not commands.get(field):
            errors.append(f"commands.{field} must be a non-empty string for accepted artifacts")
    environment_commands = commands.get("environment")
    if not _is_nonempty_string_list(environment_commands):
        errors.append("commands.environment must be a non-empty string list for accepted artifacts")
    elif isinstance(environment_commands, list):
        joined_environment_commands = "\n".join(environment_commands)
        for fragment in _REQUIRED_ACCEPTED_ENVIRONMENT_COMMAND_FRAGMENTS:
            if fragment not in joined_environment_commands:
                errors.append(f"commands.environment must include {fragment} for accepted artifacts")
    benchmark_command = commands.get("benchmark")
    if isinstance(benchmark_command, str):
        if "qwen35_batch_retained_bench.py" not in benchmark_command:
            errors.append("commands.benchmark must reference scripts/qwen35_batch_retained_bench.py for accepted artifacts")
        else:
            _validate_command_model_fixture_flags(benchmark_command, field="benchmark", errors=errors)
            _validate_command_workload_shape(benchmark_command, field="benchmark", payload=payload, errors=errors)
            _validate_command_json_artifact_path(benchmark_command, field="benchmark", errors=errors)
            _validate_retained_benchmark_reference_paths(benchmark_command, field="benchmark", payload=payload, errors=errors)
    correctness_command = commands.get("correctness_reference")
    if isinstance(correctness_command, str):
        if "qwen35_batch_correctness.py" not in correctness_command:
            errors.append("commands.correctness_reference must reference scripts/qwen35_batch_correctness.py for accepted artifacts")
        else:
            rows_match = _CORRECTNESS_ROWS_RE.search(correctness_command)
            if rows_match is None:
                errors.append("commands.correctness_reference must include --rows <workload.concurrency> for accepted artifacts")
            else:
                workload = payload.get("workload")
                concurrency = workload.get("concurrency") if isinstance(workload, Mapping) else None
                if isinstance(concurrency, int) and not isinstance(concurrency, bool) and int(rows_match.group(1)) != concurrency:
                    errors.append("commands.correctness_reference --rows must match workload.concurrency for accepted artifacts")
            correctness = payload.get("correctness")
            primitive = correctness.get("primitive_batch_correctness") if isinstance(correctness, Mapping) else None
            primitive_artifact_path = primitive.get("artifact_path") if isinstance(primitive, Mapping) else None
            _validate_command_json_matches_artifact_path(
                correctness_command,
                field="correctness_reference",
                artifact_field="correctness.primitive_batch_correctness.artifact_path",
                artifact_path=primitive_artifact_path,
                errors=errors,
            )
    profiler_command = commands.get("profiler")
    if isinstance(profiler_command, str):
        if "rocprofv3" not in profiler_command or "--kernel-trace" not in profiler_command:
            errors.append("commands.profiler must include rocprofv3 --kernel-trace for accepted artifacts")
        if "qwen35_batch_retained_bench.py" not in profiler_command:
            errors.append("commands.profiler must target scripts/qwen35_batch_retained_bench.py for accepted artifacts")
        else:
            _validate_command_model_fixture_flags(profiler_command, field="profiler", errors=errors)
            _validate_command_workload_shape(profiler_command, field="profiler", payload=payload, errors=errors)
            _validate_command_json_artifact_path(profiler_command, field="profiler", errors=errors)
            _validate_retained_benchmark_reference_paths(profiler_command, field="profiler", payload=payload, errors=errors)
            if "--require-cached-build" not in profiler_command:
                errors.append("commands.profiler must include --require-cached-build for accepted artifacts")
            if _COMMAND_COMPILER_VERSION_FILE_RE.search(profiler_command) is None:
                errors.append("commands.profiler must include --compiler-version-file for accepted artifacts")
    profiler = _mapping_at(payload, "profiler", errors)
    profiler_artifact_path = profiler.get("artifact_path")
    if not isinstance(profiler_artifact_path, str) or not profiler_artifact_path:
        errors.append("profiler.artifact_path must be a non-empty string for accepted artifacts")
    else:
        _validate_benchmark_results_artifact_path("profiler.artifact_path", profiler_artifact_path, errors)
        if isinstance(profiler_command, str):
            _validate_profiler_command_artifact_reference(profiler_command, profiler_artifact_path, errors)
    if profiler.get("status") != "captured":
        errors.append("profiler.status must be 'captured' for accepted artifacts")
    if profiler.get("expected_kernels_present") is not True:
        errors.append("profiler.expected_kernels_present must be true for accepted artifacts")
    expected_kernel_names = profiler.get("expected_kernel_names")
    if not _is_nonempty_string_list(expected_kernel_names):
        errors.append("profiler.expected_kernel_names must be a non-empty string list for accepted artifacts")
    elif isinstance(expected_kernel_names, list):
        _validate_expected_profiler_kernel_names(expected_kernel_names, errors)
    kernel_durations = profiler.get("kernel_durations_ns")
    if not isinstance(kernel_durations, Mapping) or not kernel_durations:
        errors.append("profiler.kernel_durations_ns must be a non-empty object for accepted artifacts")
    else:
        for kernel_name, duration_ns in kernel_durations.items():
            if isinstance(kernel_name, str) and _has_disallowed_profiler_kernel_fragment(kernel_name):
                errors.append("profiler.kernel_durations_ns must not include serial/per-row/fallback kernel names for accepted artifacts")
                break
            if isinstance(kernel_name, str) and kernel_name and not _is_positive_number(duration_ns):
                errors.append(f"profiler.kernel_durations_ns.{kernel_name} must be positive numeric for accepted artifacts")
        if isinstance(expected_kernel_names, list):
            for kernel_name in expected_kernel_names:
                if not isinstance(kernel_name, str) or not kernel_name:
                    continue
                if not _is_positive_number(kernel_durations.get(kernel_name)):
                    errors.append(f"profiler.kernel_durations_ns.{kernel_name} must be positive numeric for accepted artifacts")


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
    primitive_artifact_path = primitive.get("artifact_path")
    if not isinstance(primitive_artifact_path, str) or not primitive_artifact_path:
        errors.append("correctness.primitive_batch_correctness.artifact_path must be a non-empty string for accepted artifacts")
    else:
        _validate_benchmark_results_artifact_path(
            "correctness.primitive_batch_correctness.artifact_path",
            primitive_artifact_path,
            errors,
        )
    primitive_rows = primitive.get("rows")
    if not isinstance(primitive_rows, int) or isinstance(primitive_rows, bool):
        errors.append("correctness.primitive_batch_correctness.rows must be an int for accepted artifacts")
    if concurrency_valid and isinstance(primitive_rows, int) and not isinstance(primitive_rows, bool) and primitive_rows != concurrency:
        errors.append("correctness.primitive_batch_correctness.rows must match workload.concurrency for accepted artifacts")
    for field in ("append_key_mismatch", "append_value_mismatch"):
        value = primitive.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"correctness.primitive_batch_correctness.{field} must be an int for accepted artifacts")
        elif value != 0:
            errors.append(f"correctness.primitive_batch_correctness.{field} must be 0 for accepted artifacts")
    attn_vs_c1 = primitive.get("attn_batch_vs_c1_max_abs")
    if not _is_number(attn_vs_c1):
        errors.append("correctness.primitive_batch_correctness.attn_batch_vs_c1_max_abs must be numeric for accepted artifacts")
    elif float(attn_vs_c1) != 0.0:
        errors.append("correctness.primitive_batch_correctness.attn_batch_vs_c1_max_abs must be 0.0 for accepted artifacts")


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
    measurements = _mapping_at(payload, "measurements", errors)
    workload = _mapping_at(payload, "workload", errors)
    for field in _REQUIRED_ACCEPTED_WORKLOAD_LABELS:
        if not isinstance(workload.get(field), str) or not workload.get(field):
            errors.append(f"workload.{field} must be a non-empty string for accepted artifacts")
    kv_policy = workload.get("kv_policy")
    if not isinstance(kv_policy, Mapping) or not kv_policy:
        errors.append("workload.kv_policy must be a non-empty object for accepted artifacts")
    else:
        policy_storage_dtype = kv_policy.get("storage_dtype")
        if policy_storage_dtype != workload.get("kv_storage_dtype"):
            errors.append("workload.kv_policy.storage_dtype must match workload.kv_storage_dtype for accepted artifacts")
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
    max_layers = workload.get("max_layers")
    if not isinstance(max_layers, int) or isinstance(max_layers, bool):
        errors.append("workload.max_layers must be an int for accepted artifacts")
    elif max_layers != 40:
        errors.append("workload.max_layers must be 40 for accepted artifacts")
    if concurrency_valid and prompt_tokens_valid:
        _validate_workload_aggregate_tokens(
            "prompt_tokens_aggregate",
            workload,
            int(prompt_tokens) * int(concurrency),
            errors,
        )
    if concurrency_valid and gen_tokens_valid:
        _validate_workload_aggregate_tokens(
            "gen_tokens_aggregate",
            workload,
            int(gen_tokens) * int(concurrency),
            errors,
        )
    if concurrency_valid:
        _validate_prompt_lengths(workload, int(concurrency), int(prompt_tokens) if prompt_tokens_valid else None, errors)
        _validate_aggregate_per_request_rate("measurements", measurements, int(concurrency), errors)
    if scaling.get("complete") is not True:
        errors.append("scaling.complete must be true for accepted artifacts")
    native = scaling.get("native")
    if not isinstance(native, Mapping):
        errors.append("scaling.native must be an object for accepted artifacts")
    else:
        for field in ("decode_tok_s_aggregate", "decode_tok_s_per_request"):
            if not _is_positive_number(native.get(field)):
                errors.append(f"scaling.native.{field} must be positive numeric for accepted artifacts")
            _validate_matching_number(
                f"scaling.native.{field}",
                native,
                field,
                measurements,
                field,
                errors,
            )
    for baseline_name in _REQUIRED_ACCEPTED_SCALING_BASELINES:
        baseline = scaling.get(baseline_name)
        if not isinstance(baseline, Mapping):
            errors.append(f"scaling.{baseline_name} must be an object for accepted artifacts")
            continue
        baseline_artifact_path = baseline.get("artifact_path")
        if not isinstance(baseline_artifact_path, str) or not baseline_artifact_path:
            errors.append(f"scaling.{baseline_name}.artifact_path must be a non-empty string for accepted artifacts")
        else:
            _validate_benchmark_results_artifact_path(f"scaling.{baseline_name}.artifact_path", baseline_artifact_path, errors)
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
        else:
            _validate_aggregate_per_request_rate("scaling.c1_baseline", c1_baseline, c1_concurrency, errors)
    serial_baseline = scaling.get("serial_bridge_baseline")
    if isinstance(serial_baseline, Mapping):
        serial_concurrency = serial_baseline.get("workload_concurrency")
        if not isinstance(serial_concurrency, int) or isinstance(serial_concurrency, bool):
            errors.append("scaling.serial_bridge_baseline.workload_concurrency must be an int for accepted artifacts")
        elif concurrency_valid and serial_concurrency != concurrency:
            errors.append("scaling.serial_bridge_baseline.workload_concurrency must match workload.concurrency for accepted artifacts")
        else:
            _validate_aggregate_per_request_rate("scaling.serial_bridge_baseline", serial_baseline, serial_concurrency, errors)
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


def _validate_prompt_lengths(
    workload: Mapping[str, Any],
    concurrency: int,
    prompt_tokens_per_request: int | None,
    errors: list[str],
) -> None:
    prompt_lengths = workload.get("prompt_lengths")
    if not isinstance(prompt_lengths, list):
        errors.append("workload.prompt_lengths must be a list for accepted artifacts")
        return
    if len(prompt_lengths) != concurrency:
        errors.append("workload.prompt_lengths length must match workload.concurrency for accepted artifacts")
    if any(not isinstance(length, int) or isinstance(length, bool) or length <= 0 for length in prompt_lengths):
        errors.append("workload.prompt_lengths entries must be positive ints for accepted artifacts")
    elif prompt_tokens_per_request is not None and any(length != prompt_tokens_per_request for length in prompt_lengths):
        errors.append("workload.prompt_lengths entries must match workload.prompt_tokens_per_request for accepted artifacts")


def _validate_aggregate_per_request_rate(
    label: str,
    payload: Mapping[str, Any],
    concurrency: int,
    errors: list[str],
) -> None:
    aggregate = payload.get("decode_tok_s_aggregate")
    per_request = payload.get("decode_tok_s_per_request")
    if not (_is_number(aggregate) and _is_number(per_request)):
        return
    expected = float(per_request) * float(concurrency)
    tolerance = max(1e-9, abs(expected) * 1e-6)
    if abs(float(aggregate) - expected) > tolerance:
        errors.append(f"{label}.decode_tok_s_aggregate must match decode_tok_s_per_request times concurrency for accepted artifacts")


def _validate_workload_aggregate_tokens(
    field: str,
    workload: Mapping[str, Any],
    expected: int,
    errors: list[str],
) -> None:
    actual = workload.get(field)
    if not isinstance(actual, int) or isinstance(actual, bool):
        errors.append(f"workload.{field} must be an int for accepted artifacts")
    elif actual != expected:
        errors.append(f"workload.{field} must equal per-request tokens times workload.concurrency for accepted artifacts")


def _validate_matching_number(
    label: str,
    actual_payload: Mapping[str, Any],
    actual_field: str,
    expected_payload: Mapping[str, Any],
    expected_field: str,
    errors: list[str],
) -> None:
    actual = actual_payload.get(actual_field)
    expected = expected_payload.get(expected_field)
    if not (_is_number(actual) and _is_number(expected)):
        return
    expected_value = float(expected)
    tolerance = max(1e-9, abs(expected_value) * 1e-6)
    if abs(float(actual) - expected_value) > tolerance:
        errors.append(f"{label} must match measurements.{expected_field} for accepted artifacts")


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


def _is_nonempty_string_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) and bool(item) for item in value)


__all__ = ["validate_cn_diagnostic_artifact_payload"]
