from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import statistics
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from hipengine.dispatch import (
    ProjectionDispatchEvidence,
    batch_sampler_equality_payload_blockers,
    projection_dispatch_candidates_from_artifact,
    projection_dispatch_evidence_payload_blockers,
)
from hipengine.generation import GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS
from scripts.qwen35_batch_constants import (
    PROFILER_DISALLOWED_DIAGNOSTIC_KERNEL_NAME_FRAGMENTS,
    RETAINED_ARTIFACT_ACCEPTED_DECISION_REASON,
    RETAINED_ARTIFACT_ACCEPTED_MODE,
    RETAINED_ARTIFACT_ACCEPTED_NOTES,
    RETAINED_ARTIFACT_ACCEPTED_SUMMARY,
    RETAINED_ARTIFACT_VALIDATION_SUMMARY_TYPE,
    RETAINED_ARTIFACT_CORRECTNESS_REFERENCE_UNIQUE_FLAGS,
    RETAINED_ARTIFACT_CORRECTNESS_SCRIPT_ALLOWED_FLAGS,
    RETAINED_ARTIFACT_DISALLOWED_DIAGNOSTIC_COMMAND_FRAGMENTS,
    RETAINED_ARTIFACT_DISALLOWED_DIAGNOSTIC_EVIDENCE_FRAGMENTS,
    RETAINED_ARTIFACT_DISALLOWED_DIAGNOSTIC_TRACE_FIELD_FRAGMENTS,
    RETAINED_ARTIFACT_DISALLOWED_DIAGNOSTIC_TRACE_FIELD_NAMES,
    RETAINED_ARTIFACT_INT8_PRIMITIVE_GATE_FLAGS,
    RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_NUMPY_MAX_ABS_LIMIT,
    RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_SCRIPT,
    RETAINED_ARTIFACT_PROFILER_SYNTHESIZED_FIELDS,
    RETAINED_ARTIFACT_ROCPROF_COMMAND_FLAGS,
    RETAINED_ARTIFACT_ROCPROF_EXECUTABLE,
    RETAINED_ARTIFACT_ROCPROF_OUTPUT_FORMAT,
    RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SCHEMA,
    RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SEED,
    RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS,
    RETAINED_ARTIFACT_REQUIRED_PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES,
    RETAINED_ARTIFACT_REQUIRED_PROFILER_KERNEL_DURATION_CATEGORIES,
    RETAINED_ARTIFACT_REQUIRED_SCALING_BASELINES,
    RETAINED_ARTIFACT_RETAINED_BENCH_SCRIPT,
    RETAINED_ARTIFACT_RETAINED_BENCH_UNIQUE_FLAGS,
    RETAINED_ARTIFACT_RETAINED_GATE_FLAGS,
    RETAINED_ARTIFACT_RETAINED_GATE_LABELS,
    RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_UNIQUE_FLAGS,
    RETAINED_ARTIFACT_REQUIRED_SCALING_RATIOS,
    RETAINED_ARTIFACT_UNUSABLE_SCALING_BASELINE_STATUSES,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-finite constant {value!r}")


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
    "model_path",
    "fixture_path",
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
_REQUIRED_ACCEPTED_HARDWARE_CAPTURE_COMMAND_FRAGMENTS = {
    "rocminfo": ("rocminfo | grep -E", "Name:|gfx", "head -4"),
    "rocm_smi": ("rocm-smi", "--showmeminfo", "vram", "--showuse", "--showtemp"),
}
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
_REQUIRED_ACCEPTED_ENVIRONMENT_COMMANDS = (
    "rocminfo | grep -E 'Name:|gfx' | head -4",
    "rocm-smi --showmeminfo vram --showuse --showtemp",
    "hipcc --version",
    "git rev-parse HEAD",
    "git diff --quiet",
)
DISALLOWED_ACCEPTED_DIAGNOSTIC_COMMAND_FRAGMENTS = RETAINED_ARTIFACT_DISALLOWED_DIAGNOSTIC_COMMAND_FRAGMENTS
DISALLOWED_ACCEPTED_DIAGNOSTIC_EVIDENCE_FRAGMENTS = RETAINED_ARTIFACT_DISALLOWED_DIAGNOSTIC_EVIDENCE_FRAGMENTS
DISALLOWED_ACCEPTED_DIAGNOSTIC_TRACE_FIELD_NAMES = RETAINED_ARTIFACT_DISALLOWED_DIAGNOSTIC_TRACE_FIELD_NAMES
DISALLOWED_ACCEPTED_DIAGNOSTIC_TRACE_FIELD_FRAGMENTS = RETAINED_ARTIFACT_DISALLOWED_DIAGNOSTIC_TRACE_FIELD_FRAGMENTS
_REQUIRED_ACCEPTED_SCALING_BASELINES = RETAINED_ARTIFACT_REQUIRED_SCALING_BASELINES
_REQUIRED_ACCEPTED_SCALING_RATIOS = RETAINED_ARTIFACT_REQUIRED_SCALING_RATIOS
_UNUSABLE_ACCEPTED_SCALING_BASELINE_STATUSES = RETAINED_ARTIFACT_UNUSABLE_SCALING_BASELINE_STATUSES
_GRAPH_KERNEL_TIME_HISTOGRAM_BUCKET_SET = frozenset(GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS)


def _graph_kernel_time_histogram_bucket_ns(duration_ns: int) -> str:
    if duration_ns <= 10_000:
        return GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS[0]
    if duration_ns <= 100_000:
        return GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS[1]
    if duration_ns <= 1_000_000:
        return GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS[2]
    if duration_ns <= 10_000_000:
        return GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS[3]
    return GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS[4]


_COMMAND_BATCH_SIZE_RE = re.compile(r"(?:^|\s)--batch-size(?:=|\s+)(\d+)(?=\s|$)")
_COMMAND_DECODE_TOKENS_RE = re.compile(r"(?:^|\s)--decode-tokens(?:=|\s+)(\d+)(?=\s|$)")
_COMMAND_MAX_LAYERS_RE = re.compile(r"(?:^|\s)--max-layers(?:=|\s+)(\d+)(?=\s|$)")
_COMMAND_MODEL_RE = re.compile(r"(?:^|\s)--model(?:=|\s+)(\S+)(?=\s|$)")
_COMMAND_FIXTURE_RE = re.compile(r"(?:^|\s)--fixture(?:=|\s+)(\S+)(?=\s|$)")
_COMMAND_PROMPT_LENGTH_RE = re.compile(r"(?:^|\s)--prompt-length(?:=|\s+)(\d+)(?=\s|$)")
_COMMAND_JSON_RE = re.compile(r"(?:^|\s)--json(?:=|\s+)(\S+)(?=\s|$)")
_ROLLUP_LAST_UPDATED_RE = re.compile(r"(?im)^\s*Last updated\s*:?\s*(\d{4}-\d{2}-\d{2})\b")
_ROLLUP_DATED_CHANGELOG_RE = re.compile(r"(?m)^\s*(?:[-*]\s*)?(?:##\s*)?(\d{4}-\d{2}-\d{2})\b")
_ROLLUP_PERCENT_DELTA_RE = re.compile(r"[+-]?\d+(?:\.\d+)?\s*%")
_ROLLUP_OLD_NEW_RE = re.compile(r"[+-]?\d+(?:\.\d+)?(?:\s*[A-Za-z/_]+)?\s*(?:→|->)\s*[+-]?\d+(?:\.\d+)?")
_RETAINED_GATE_FLAGS = RETAINED_ARTIFACT_RETAINED_GATE_FLAGS
_RETAINED_GATE_LABELS = RETAINED_ARTIFACT_RETAINED_GATE_LABELS
_COMMAND_RETAINED_GATE_PATH_RES = tuple(
    re.compile(rf"(?:^|\s){re.escape(flag)}(?:=|\s+)(\S+)(?=\s|$)")
    for flag in _RETAINED_GATE_FLAGS
)
_COMMAND_C1_BASELINE_JSON_RE = _COMMAND_RETAINED_GATE_PATH_RES[0]
_COMMAND_SERIAL_BRIDGE_JSON_RE = _COMMAND_RETAINED_GATE_PATH_RES[1]
_COMMAND_PRIMITIVE_CORRECTNESS_JSON_RE = _COMMAND_RETAINED_GATE_PATH_RES[2]
_COMMAND_PROFILER_JSON_RE = _COMMAND_RETAINED_GATE_PATH_RES[3]
_COMMAND_COMPILER_VERSION_FILE_RE = re.compile(r"(?:^|\s)--compiler-version-file(?:=|\s+)(\S+)(?=\s|$)")
_COMMAND_OUTPUT_FORMAT_RE = re.compile(r"(?:^|\s)--output-format(?:=|\s+)(\S+)(?=\s|$)")
_COMMAND_TRACE_DIR_RE = re.compile(r"(?:^|\s)-d(?:=|\s+)(\S+)(?=\s|$)")
_CORRECTNESS_ROWS_RE = re.compile(r"(?:^|\s)--rows(?:=|\s+)(\d+)(?=\s|$)")
_CORRECTNESS_SEED_RE = re.compile(r"(?:^|\s)--seed(?:=|\s+)(\d+)(?=\s|$)")
_PRIMITIVE_CORRECTNESS_SCRIPT = RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_SCRIPT
_RETAINED_BENCH_SCRIPT = RETAINED_ARTIFACT_RETAINED_BENCH_SCRIPT
_RETAINED_BENCH_UNIQUE_FLAGS = RETAINED_ARTIFACT_RETAINED_BENCH_UNIQUE_FLAGS
_DEVICE_METADATA_ENV_KEYS = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
_COMMAND_ENV_KEYS = _DEVICE_METADATA_ENV_KEYS
_RETAINED_PROFILED_COMMAND_UNIQUE_FLAGS = RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_UNIQUE_FLAGS
_INT8_PRIMITIVE_GATE_FLAGS = RETAINED_ARTIFACT_INT8_PRIMITIVE_GATE_FLAGS
_CORRECTNESS_REFERENCE_UNIQUE_FLAGS = RETAINED_ARTIFACT_CORRECTNESS_REFERENCE_UNIQUE_FLAGS
_CORRECTNESS_SCRIPT_ALLOWED_FLAGS = RETAINED_ARTIFACT_CORRECTNESS_SCRIPT_ALLOWED_FLAGS
_FULL_COMMIT_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", re.IGNORECASE)
_ACCEPTED_HARDWARE_ARCH_RE = re.compile(r"gfx[0-9a-f]+", re.IGNORECASE)
_ROCPROF_COMMAND_FLAGS = RETAINED_ARTIFACT_ROCPROF_COMMAND_FLAGS
_ROCPROF_EXECUTABLE = RETAINED_ARTIFACT_ROCPROF_EXECUTABLE
_ROCPROF_OUTPUT_FORMAT = RETAINED_ARTIFACT_ROCPROF_OUTPUT_FORMAT
_DISALLOWED_PROFILER_KERNEL_NAME_FRAGMENTS = PROFILER_DISALLOWED_DIAGNOSTIC_KERNEL_NAME_FRAGMENTS
_REQUIRED_PROFILER_KERNEL_DURATION_CATEGORIES = RETAINED_ARTIFACT_REQUIRED_PROFILER_KERNEL_DURATION_CATEGORIES
_REQUIRED_PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES = RETAINED_ARTIFACT_REQUIRED_PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES
_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS = RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS
_REQUIRED_PRIMITIVE_CORRECTNESS_SCHEMA = RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SCHEMA
_REQUIRED_PRIMITIVE_CORRECTNESS_SEED = RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SEED
_PRIMITIVE_CORRECTNESS_NUMPY_MAX_ABS_LIMIT = RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_NUMPY_MAX_ABS_LIMIT
_ALLOWED_PROFILER_SYNTHESIZED_FIELDS = RETAINED_ARTIFACT_PROFILER_SYNTHESIZED_FIELDS
DECODE_EXECUTION_DIAGNOSTIC_TRACE_FIELDS = (
    "_decode_full_attention_trace",
    "_decode_linear_input_trace",
    "_decode_linear_output_trace",
    "_decode_linear_stage_trace",
    "decode_full_attention",
    "decode_full_attention_trace",
    "decode_full_context_oracle",
    "decode_full_kv_samples",
    "decode_linear_input_trace",
    "decode_linear_inputs",
    "decode_linear_output_trace",
    "decode_linear_outputs",
    "decode_linear_stage_trace",
    "decode_linear_stages",
)


def _disallowed_accepted_diagnostic_trace_field_reasons(
    label: str,
    *,
    exact_decode_fields: bool = False,
) -> list[str]:
    reasons: list[str] = []
    for field_name in DISALLOWED_ACCEPTED_DIAGNOSTIC_TRACE_FIELD_NAMES:
        if field_name in label:
            reasons.append(field_name)
    for field_name in DECODE_EXECUTION_DIAGNOSTIC_TRACE_FIELDS:
        decode_field_matches = (label == field_name) if exact_decode_fields else (field_name in label)
        if decode_field_matches:
            reasons.append(field_name)
    for fragment in DISALLOWED_ACCEPTED_DIAGNOSTIC_TRACE_FIELD_FRAGMENTS:
        if fragment in label:
            reasons.append(fragment)
    return reasons


def _append_disallowed_accepted_diagnostic_text_errors(value: str, *, path: str, errors: list[str]) -> None:
    for fragment in DISALLOWED_ACCEPTED_DIAGNOSTIC_COMMAND_FRAGMENTS:
        if fragment in value:
            errors.append(f"{path} must not include diagnostic override {fragment} for accepted artifacts")
    for fragment in DISALLOWED_ACCEPTED_DIAGNOSTIC_EVIDENCE_FRAGMENTS:
        if fragment in value:
            errors.append(f"{path} must not include diagnostic evidence {fragment} for accepted artifacts")
    for trace_field_reason in _disallowed_accepted_diagnostic_trace_field_reasons(value):
        errors.append(f"{path} must not include diagnostic trace field {trace_field_reason} for accepted artifacts")


def _validate_no_disallowed_diagnostic_metadata(
    value: Any,
    *,
    path: str,
    errors: list[str],
) -> None:
    if path == "commands" or path.startswith("commands."):
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_label = str(key)
            child_path = f"{path}.{key_label}" if path else key_label
            for fragment in DISALLOWED_ACCEPTED_DIAGNOSTIC_COMMAND_FRAGMENTS:
                if fragment in key_label:
                    errors.append(f"{child_path} must not include diagnostic override {fragment} for accepted artifacts")
            for fragment in DISALLOWED_ACCEPTED_DIAGNOSTIC_EVIDENCE_FRAGMENTS:
                if fragment in key_label:
                    errors.append(f"{child_path} must not include diagnostic evidence {fragment} for accepted artifacts")
            for trace_field_reason in _disallowed_accepted_diagnostic_trace_field_reasons(
                key_label,
                exact_decode_fields=True,
            ):
                errors.append(f"{child_path} must not include diagnostic trace field {trace_field_reason} for accepted artifacts")
            _validate_no_disallowed_diagnostic_metadata(child, path=child_path, errors=errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_disallowed_diagnostic_metadata(child, path=f"{path}[{index}]", errors=errors)
    elif isinstance(value, str):
        _append_disallowed_accepted_diagnostic_text_errors(value, path=path, errors=errors)



def _mapping_at(payload: Mapping[str, Any], key: str, errors: list[str]) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{key} must be an object")
        return {}
    return value


def _flag_token_matches(token: str, flag: str) -> bool:
    return token == flag or token.startswith(f"{flag}=")


def _validate_unique_flags(argv: list[str], flags: tuple[str, ...], *, field: str, errors: list[str]) -> None:
    for flag in flags:
        count = sum(1 for token in argv if _flag_token_matches(token, flag))
        if count > 1:
            errors.append(f"{field} must not repeat {flag} for accepted artifacts")


def _validate_command_unique_flags(command: str, flags: tuple[str, ...], *, field: str, errors: list[str]) -> None:
    try:
        argv = shlex.split(command)
    except ValueError:
        errors.append(f"commands.{field} must be shell-parseable for accepted artifacts")
        return
    _validate_unique_flags(argv, flags, field=f"commands.{field}", errors=errors)


def _validate_retained_bench_unique_flags(command: str, *, field: str, errors: list[str]) -> None:
    _validate_command_unique_flags(command, _RETAINED_PROFILED_COMMAND_UNIQUE_FLAGS, field=field, errors=errors)


def _is_python_executable(token: str) -> bool:
    return re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(token).name) is not None


def _is_env_assignment_token(token: str) -> bool:
    key, sep, _value = token.partition("=")
    return bool(
        sep
        and key
        and (key[0].isalpha() or key[0] == "_")
        and all(ch.isalnum() or ch == "_" for ch in key)
    )


def _strip_command_env_prefix(argv: Sequence[str]) -> list[str]:
    idx = 0
    if argv and Path(argv[0]).name == "env":
        idx = 1
    while idx < len(argv) and _is_env_assignment_token(argv[idx]):
        idx += 1
    if idx == 0:
        return list(argv)
    return list(argv[idx:])


def _command_device_env_assignments(argv: Sequence[str]) -> dict[str, str]:
    idx = 0
    if argv and Path(argv[0]).name == "env":
        idx = 1
    assignments: dict[str, str] = {}
    while idx < len(argv) and _is_env_assignment_token(argv[idx]):
        key, _sep, value = argv[idx].partition("=")
        if key in _COMMAND_ENV_KEYS:
            assignments[key] = value
        idx += 1
    return assignments


def _validate_command_device_env_assignments_unique(argv: Sequence[str], *, field: str, errors: list[str]) -> None:
    idx = 0
    if argv and Path(argv[0]).name == "env":
        idx = 1
    seen: set[str] = set()
    while idx < len(argv) and _is_env_assignment_token(argv[idx]):
        key, _sep, _value = argv[idx].partition("=")
        if key in _COMMAND_ENV_KEYS:
            if key in seen:
                errors.append(f"commands.{field} device env prefix {key} must appear at most once for accepted artifacts")
            seen.add(key)
        idx += 1


def _retained_bench_command_device_env(command: str) -> dict[str, str]:
    try:
        return _command_device_env_assignments(shlex.split(command))
    except ValueError:
        return {}


def _validate_device_env_assignments_nonblank(assignments: Mapping[str, str], *, field: str, errors: list[str]) -> None:
    for key, value in assignments.items():
        if not value.strip():
            errors.append(f"commands.{field} device env prefix {key} must be non-blank for accepted artifacts")


def _validate_device_env_assignments_have_metadata(
    assignments: Mapping[str, str],
    *,
    field: str,
    requirements: Mapping[str, str],
    errors: list[str],
) -> None:
    for key in assignments:
        if key not in requirements:
            errors.append(
                f"commands.{field} device env prefix {key} must be recorded in hardware.visible_device.env "
                "or correctness.primitive_batch_correctness.device.env for accepted artifacts"
            )


def _validate_device_env_metadata(env: Any, *, prefix: str, errors: list[str]) -> None:
    if not isinstance(env, Mapping):
        errors.append(f"{prefix}.env must be an object for accepted artifacts")
        return
    for key in _DEVICE_METADATA_ENV_KEYS:
        value = env.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            errors.append(f"{prefix}.env.{key} must be a non-empty string when present for accepted artifacts")
        elif isinstance(value, str) and not value.strip():
            errors.append(f"{prefix}.env.{key} must be a non-blank string when present for accepted artifacts")


def _validate_device_runtime_metadata(device: Mapping[str, Any], *, prefix: str, errors: list[str]) -> None:
    for field in ("hipGetDeviceCount_error", "hipGetDevice_error", "hipDeviceGetName_error"):
        value = device.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{prefix}.{field} must be an int for accepted artifacts")
        elif value != 0:
            errors.append(f"{prefix}.{field} must be 0 for accepted artifacts")
    visible_count = device.get("visible_device_count")
    if not isinstance(visible_count, int) or isinstance(visible_count, bool):
        errors.append(f"{prefix}.visible_device_count must be an int for accepted artifacts")
    elif visible_count <= 0:
        errors.append(f"{prefix}.visible_device_count must be positive for accepted artifacts")
    current_device = device.get("current_device")
    if not isinstance(current_device, int) or isinstance(current_device, bool):
        errors.append(f"{prefix}.current_device must be an int for accepted artifacts")
    elif current_device < 0:
        errors.append(f"{prefix}.current_device must be non-negative for accepted artifacts")
    device_name = device.get("device_name")
    if not isinstance(device_name, str) or not device_name:
        errors.append(f"{prefix}.device_name must be a non-empty string for accepted artifacts")


def _validate_device_env_requirements(
    device: Any,
    *,
    prefix: str,
    requirements: Mapping[str, str],
    errors: list[str],
) -> bool:
    if not requirements:
        return False
    if not isinstance(device, Mapping):
        errors.append(f"{prefix} must be an object when command device env prefixes are required for accepted artifacts")
        return False
    env = device.get("env")
    if not isinstance(env, Mapping):
        errors.append(f"{prefix}.env must be an object when command device env prefixes are required for accepted artifacts")
        return False
    for key, value in requirements.items():
        if env.get(key) != value:
            errors.append(f"{prefix}.env.{key} must include {key}={value} for accepted artifacts")
    return True


def _validate_hardware_visible_device_env_requirements(
    hardware: Mapping[str, Any],
    requirements: Mapping[str, str],
    errors: list[str],
) -> None:
    visible_device = hardware.get("visible_device")
    if _validate_device_env_requirements(
        visible_device,
        prefix="hardware.visible_device",
        requirements=requirements,
        errors=errors,
    ):
        _validate_device_runtime_metadata(
            visible_device,
            prefix="hardware.visible_device",
            errors=errors,
        )


def _validate_primitive_device_env_requirements(
    payload: Mapping[str, Any],
    requirements: Mapping[str, str],
    errors: list[str],
) -> None:
    correctness = payload.get("correctness")
    primitive = correctness.get("primitive_batch_correctness") if isinstance(correctness, Mapping) else None
    primitive_device = primitive.get("device") if isinstance(primitive, Mapping) else None
    _validate_device_env_requirements(
        primitive_device,
        prefix="correctness.primitive_batch_correctness.device",
        requirements=requirements,
        errors=errors,
    )


def _validate_hardware_visible_device_matches_primitive(payload: Mapping[str, Any], errors: list[str]) -> None:
    hardware = payload.get("hardware")
    visible_device = hardware.get("visible_device") if isinstance(hardware, Mapping) else None
    correctness = payload.get("correctness")
    primitive = correctness.get("primitive_batch_correctness") if isinstance(correctness, Mapping) else None
    primitive_device = primitive.get("device") if isinstance(primitive, Mapping) else None
    if not isinstance(visible_device, Mapping) or not isinstance(primitive_device, Mapping):
        return
    for field in ("visible_device_count", "current_device"):
        visible_value = visible_device.get(field)
        primitive_value = primitive_device.get(field)
        if visible_value is not None and primitive_value is not None and visible_value != primitive_value:
            errors.append(f"hardware.visible_device.{field} must match correctness.primitive_batch_correctness.device.{field} for accepted artifacts")
    visible_name = visible_device.get("device_name")
    primitive_name = primitive_device.get("device_name")
    if (
        isinstance(visible_name, str)
        and visible_name
        and isinstance(primitive_name, str)
        and primitive_name
        and _normalized_gpu_label(visible_name) != _normalized_gpu_label(primitive_name)
    ):
        errors.append("hardware.visible_device.device_name must match correctness.primitive_batch_correctness.device.device_name for accepted artifacts")


def _script_invocation_device_env_prefix_argv(command: str, script: str) -> list[str] | None:
    try:
        argv = shlex.split(command)
    except ValueError:
        return None
    for script_index, token in enumerate(argv):
        if token != script or script_index == 0:
            continue
        python_index = script_index - 1
        if not _is_python_executable(argv[python_index]):
            continue
        assignment_start = python_index
        while assignment_start > 0 and _is_env_assignment_token(argv[assignment_start - 1]):
            assignment_start -= 1
        has_env_command_prefix = assignment_start > 0 and Path(argv[assignment_start - 1]).name == "env"
        if assignment_start != 0 and not has_env_command_prefix:
            return None
        prefix_start = assignment_start - 1 if has_env_command_prefix else assignment_start
        return list(argv[prefix_start:python_index])
    return None


def _script_invocation_device_env_assignments(command: str, script: str) -> dict[str, str]:
    prefix_argv = _script_invocation_device_env_prefix_argv(command, script)
    if prefix_argv is None:
        return {}
    return _command_device_env_assignments(prefix_argv)


def _validate_retained_bench_command_target(command: str, *, field: str, errors: list[str]) -> None:
    try:
        argv = shlex.split(command)
    except ValueError:
        errors.append(f"commands.{field} must be shell-parseable for accepted artifacts")
        return
    command_argv = _strip_command_env_prefix(argv)
    if (
        len(command_argv) < 2
        or not _is_python_executable(command_argv[0])
        or command_argv[1] != _RETAINED_BENCH_SCRIPT
    ):
        errors.append(
            f"commands.{field} must start with python scripts/qwen35_batch_retained_bench.py for accepted artifacts"
        )


def _embedded_python_script_argv(command: str, script: str, *, field: str, errors: list[str]) -> list[str] | None:
    try:
        argv = shlex.split(command)
    except ValueError:
        errors.append(f"commands.{field} must be shell-parseable for accepted artifacts")
        return None
    script_indices = [idx for idx, token in enumerate(argv) if token == script]
    if len(script_indices) != 1:
        errors.append(f"commands.{field} must include exactly one python {script} invocation for accepted artifacts")
        return None
    script_index = script_indices[0]
    if script_index == 0 or not _is_python_executable(argv[script_index - 1]):
        errors.append(f"commands.{field} must invoke {script} with python for accepted artifacts")
        return None
    return argv[script_index - 1 :]


def _env_assignment_prefix_contains(argv: Sequence[str], python_index: int, *, key: str, value: str) -> bool:
    assignment_start = python_index
    while assignment_start > 0 and _is_env_assignment_token(argv[assignment_start - 1]):
        assignment_start -= 1
    if assignment_start == python_index:
        return False
    has_env_command_prefix = assignment_start > 0 and Path(argv[assignment_start - 1]).name == "env"
    if assignment_start != 0 and not has_env_command_prefix:
        return False
    return f"{key}={value}" in argv[assignment_start:python_index]


def _command_script_invocation_has_env_assignment(command: str, script: str, *, key: str, value: str) -> bool:
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    for script_index, token in enumerate(argv):
        if token != script or script_index == 0:
            continue
        python_index = script_index - 1
        if _is_python_executable(argv[python_index]) and _env_assignment_prefix_contains(
            argv,
            python_index,
            key=key,
            value=value,
        ):
            return True
    return False


def _accepted_device_selection_env_requirements(payload: Mapping[str, Any], errors: list[str]) -> dict[str, str]:
    requirements: dict[str, str] = {}
    hardware = payload.get("hardware")
    visible_device = hardware.get("visible_device") if isinstance(hardware, Mapping) else None
    correctness = payload.get("correctness")
    primitive = correctness.get("primitive_batch_correctness") if isinstance(correctness, Mapping) else None
    primitive_device = primitive.get("device") if isinstance(primitive, Mapping) else None
    sources: tuple[tuple[str, Any], ...] = (
        ("hardware.visible_device.env", visible_device),
        ("correctness.primitive_batch_correctness.device.env", primitive_device),
    )
    for source_path, source in sources:
        env = source.get("env") if isinstance(source, Mapping) else None
        if not isinstance(env, Mapping):
            continue
        for key in _DEVICE_METADATA_ENV_KEYS:
            value = env.get(key)
            if value is None:
                continue
            if not isinstance(value, str) or not value:
                errors.append(f"{source_path}.{key} must be a non-empty string for accepted artifacts")
                continue
            if not value.strip():
                errors.append(f"{source_path}.{key} must be a non-blank string for accepted artifacts")
                continue
            existing = requirements.get(key)
            if existing is not None and existing != value:
                if key == "HIP_VISIBLE_DEVICES":
                    errors.append(
                        "hardware.visible_device.env.HIP_VISIBLE_DEVICES must match primitive device env for accepted artifacts"
                    )
                else:
                    errors.append(f"hardware.visible_device.env.{key} must match primitive device env for accepted artifacts")
            else:
                requirements[key] = value
    return requirements


def _validate_command_device_selection_env(
    command: str,
    *,
    field: str,
    script: str,
    requirements: Mapping[str, str],
    errors: list[str],
) -> None:
    for key, value in requirements.items():
        if not _command_script_invocation_has_env_assignment(command, script, key=key, value=value):
            errors.append(
                f"commands.{field} must include {key}={value} before {script} for accepted artifacts"
            )


def _validate_correctness_script_argv_shape(argv: list[str] | None, *, field: str, errors: list[str]) -> None:
    if argv is None:
        return
    idx = 2
    while idx < len(argv):
        token = argv[idx]
        matched_inline_flag = next((flag for flag in _CORRECTNESS_SCRIPT_ALLOWED_FLAGS if token.startswith(f"{flag}=")), None)
        if matched_inline_flag is not None:
            if token == f"{matched_inline_flag}=":
                errors.append(f"commands.{field} {matched_inline_flag} must include a value for accepted artifacts")
            idx += 1
            continue
        if token in _CORRECTNESS_SCRIPT_ALLOWED_FLAGS:
            if idx + 1 >= len(argv) or argv[idx + 1].startswith("--"):
                errors.append(f"commands.{field} {token} must include a value for accepted artifacts")
                idx += 1
            else:
                idx += 2
            continue
        errors.append("commands.correctness_reference python script argv must only include --rows/--seed/--json for accepted artifacts")
        return


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


def _argv_value(argv: list[str], flag: str) -> str | None:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


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


def _validate_command_model_fixture_flags(command: str, *, field: str, workload: Mapping[str, Any], errors: list[str]) -> None:
    model_match = _COMMAND_MODEL_RE.search(command)
    if model_match is None:
        errors.append(f"commands.{field} must include --model for accepted artifacts")
    else:
        model_path = workload.get("model_path")
        command_model = model_match.group(1).strip("'\"")
        if isinstance(model_path, str) and model_path and command_model != model_path:
            errors.append(f"commands.{field} --model must match workload.model_path for accepted artifacts")
    fixture_match = _COMMAND_FIXTURE_RE.search(command)
    if fixture_match is None:
        errors.append(f"commands.{field} must include --fixture for accepted artifacts")
    else:
        fixture_path = workload.get("fixture_path")
        command_fixture = fixture_match.group(1).strip("'\"")
        if isinstance(fixture_path, str) and fixture_path and command_fixture != fixture_path:
            errors.append(f"commands.{field} --fixture must match workload.fixture_path for accepted artifacts")


def _command_pattern_value(command: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(command)
    if match is None:
        return None
    return match.group(1).strip("'\"")


def _validate_profiled_command_matches_benchmark_model_fixture(profiled_command: str, benchmark_command: str, errors: list[str]) -> None:
    for flag, pattern in (("--model", _COMMAND_MODEL_RE), ("--fixture", _COMMAND_FIXTURE_RE)):
        profiled_value = _command_pattern_value(profiled_command, pattern)
        benchmark_value = _command_pattern_value(benchmark_command, pattern)
        if profiled_value is not None and benchmark_value is not None and profiled_value != benchmark_value:
            errors.append(f"commands.profiler {flag} must match commands.benchmark {flag} for accepted artifacts")


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
    for flag, pattern, artifact_field, artifact_path in (
        (
            _RETAINED_GATE_FLAGS[0],
            _COMMAND_C1_BASELINE_JSON_RE,
            "scaling.c1_baseline.artifact_path",
            c1_artifact_path,
        ),
        (
            _RETAINED_GATE_FLAGS[1],
            _COMMAND_SERIAL_BRIDGE_JSON_RE,
            "scaling.serial_bridge_baseline.artifact_path",
            serial_artifact_path,
        ),
        (
            _RETAINED_GATE_FLAGS[2],
            _COMMAND_PRIMITIVE_CORRECTNESS_JSON_RE,
            "correctness.primitive_batch_correctness.artifact_path",
            primitive_artifact_path,
        ),
    ):
        _validate_command_flag_matches_artifact_path(
            command,
            field=field,
            flag=flag,
            pattern=pattern,
            artifact_field=artifact_field,
            artifact_path=artifact_path,
            errors=errors,
        )


def _validate_profiler_command_artifact_reference(command: str, profiler_artifact_path: str, errors: list[str]) -> None:
    profiler_json_match = _COMMAND_PROFILER_JSON_RE.search(command)
    profiler_json_flag = _RETAINED_GATE_FLAGS[3]
    if profiler_json_match is None:
        errors.append(f"commands.profiler must include {profiler_json_flag} <profiler.artifact_path> for accepted artifacts")
        return
    command_profiler_path = profiler_json_match.group(1).strip("'\"")
    if command_profiler_path != profiler_artifact_path:
        errors.append(f"commands.profiler {profiler_json_flag} path must match profiler.artifact_path for accepted artifacts")


def _has_disallowed_profiler_kernel_fragment(name: str) -> bool:
    lowered = name.lower()
    return any(fragment in lowered for fragment in _DISALLOWED_PROFILER_KERNEL_NAME_FRAGMENTS)


def _is_kernel_trace_csv_path(trace_file: str) -> bool:
    name = Path(trace_file).name.lower()
    return Path(trace_file).suffix.lower() == ".csv" and "kernel" in name and "trace" in name


def _path_has_parent_directory_component(path: str | Path) -> bool:
    return ".." in Path(path).parts


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _is_resolved_path_relative_to(path: str | Path, root: str | Path) -> bool:
    return _resolve_repo_path(path).is_relative_to(_resolve_repo_path(root))


def _path_has_symlink_parent(path: str | Path) -> bool:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            return True
        current = current.parent
    return False


def _path_has_non_directory_parent(path: str | Path) -> bool:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    current = path.parent
    while current != current.parent:
        if current.exists() and not current.is_dir():
            return True
        current = current.parent
    return False


def _profiler_kernel_duration_category(kernel_name: str) -> str:
    lowered = kernel_name.lower()
    if "graph" in lowered or "replay" in lowered:
        return "graph_replay"
    if "moe" in lowered or "expert" in lowered or "router" in lowered:
        return "moe"
    if "attn" in lowered or "attention" in lowered or "paged" in lowered or "kv" in lowered:
        return "attention"
    if "lm_head" in lowered or "sample" in lowered or "argmax" in lowered:
        return "sampling"
    projection_fragments = ("projection", "linear", "matmul", "gemm", "gemv", "mmq", "wmma")
    if any(fragment in lowered for fragment in projection_fragments):
        return "projection"
    return "other"


def _profiler_kernel_duration_category_sums(kernel_durations: Mapping[Any, Any]) -> dict[str, float]:
    categories = dict.fromkeys(_REQUIRED_PROFILER_KERNEL_DURATION_CATEGORIES, 0.0)
    for kernel_name, duration_ns in kernel_durations.items():
        if not isinstance(kernel_name, str) or not kernel_name or not _is_positive_number(duration_ns):
            continue
        categories[_profiler_kernel_duration_category(kernel_name)] += float(duration_ns)
    return categories


def _validate_expected_profiler_kernel_names(expected_kernel_names: list[Any], errors: list[str]) -> None:
    if len(set(expected_kernel_names)) != len(expected_kernel_names):
        errors.append("profiler.expected_kernel_names entries must be unique for accepted artifacts")
    if not any(isinstance(name, str) and "batch" in name.lower() for name in expected_kernel_names):
        errors.append("profiler.expected_kernel_names must include at least one native batch kernel name for accepted artifacts")
    for name in expected_kernel_names:
        if isinstance(name, str) and _has_disallowed_profiler_kernel_fragment(name):
            errors.append("profiler.expected_kernel_names must not include serial/per-row/fallback kernel names for accepted artifacts")
            break


def _validate_profiler_synthesized_fields(profiler: Mapping[str, Any], errors: list[str]) -> None:
    synthesized_fields = profiler.get("synthesized_fields")
    if not isinstance(synthesized_fields, list) or not all(isinstance(field, str) for field in synthesized_fields):
        errors.append("profiler.synthesized_fields must be a string list for accepted artifacts")
        return
    if len(set(synthesized_fields)) != len(synthesized_fields):
        errors.append("profiler.synthesized_fields must not contain duplicates for accepted artifacts")
    unknown_fields = sorted(set(synthesized_fields) - set(_ALLOWED_PROFILER_SYNTHESIZED_FIELDS))
    if unknown_fields:
        errors.append("profiler.synthesized_fields must only name known synthesized profiler fields for accepted artifacts")


def _path_text_contains_parent_traversal(value: str) -> bool:
    return ".." in value.replace("\\", "/").split("/")


def _is_benchmark_results_path(value: str) -> bool:
    results_root = (Path.cwd() / "benchmarks" / "results").resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        return candidate.resolve().is_relative_to(results_root)
    except OSError:
        return False


def _benchmark_results_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    marker = "benchmarks/results/"
    if normalized.startswith(marker):
        return normalized
    marker_index = normalized.find("/" + marker)
    if marker_index >= 0:
        return normalized[marker_index + 1 :]
    return normalized


def _validate_benchmark_results_artifact_path(field: str, value: Any, errors: list[str]) -> None:
    if isinstance(value, str) and value:
        if not _is_benchmark_results_path(value):
            errors.append(f"{field} must be under benchmarks/results for accepted artifacts")
        if _path_text_contains_parent_traversal(value):
            errors.append(f"{field} must not contain parent traversal for accepted artifacts")


def _path_has_benchmark_results_symlink_parent(path: Path) -> bool:
    results_root = (Path.cwd() / "benchmarks" / "results").resolve()
    current = path.parent
    while True:
        try:
            current_resolved = current.resolve()
        except OSError:
            return False
        if not current_resolved.is_relative_to(results_root):
            return False
        if current.is_symlink():
            return True
        if current == current.parent:
            return False
        current = current.parent


def _load_benchmark_results_json_artifact(field: str, value: str, errors: list[str]) -> Mapping[str, Any] | None:
    if not _is_benchmark_results_path(value):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.suffix.lower() != ".json":
        errors.append(f"{field} must point to a .json artifact for accepted artifacts")
        return None
    if path.is_symlink():
        errors.append(f"{field} must point to a regular JSON artifact, not a symlink, for accepted artifacts")
        return None
    if _path_has_benchmark_results_symlink_parent(path):
        errors.append(f"{field} parent directories must not be symlinks for accepted artifacts")
        return None
    if not path.exists():
        errors.append(f"{field} must point to an existing JSON artifact for accepted artifacts")
        return None
    if not path.is_file():
        errors.append(f"{field} must point to a regular JSON artifact for accepted artifacts")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except OSError as exc:
        errors.append(f"{field} must point to a readable JSON artifact for accepted artifacts: {exc}")
        return None
    except (json.JSONDecodeError, ValueError) as exc:
        errors.append(f"{field} must point to a valid JSON artifact for accepted artifacts: {exc}")
        return None
    if not isinstance(payload, Mapping):
        errors.append(f"{field} must point to a JSON object artifact for accepted artifacts")
        return None
    return payload


def _artifact_row_count(payload: Mapping[str, Any]) -> Any:
    rows = payload.get("rows")
    if rows is not None:
        return rows
    workload = payload.get("workload")
    if isinstance(workload, Mapping):
        return workload.get("concurrency")
    return None


def _artifact_is_accepted(payload: Mapping[str, Any]) -> bool:
    if payload.get("accepted") is True or payload.get("passed") is True or payload.get("status") == "accepted":
        return True
    decision = payload.get("decision")
    return isinstance(decision, Mapping) and decision.get("accepted") is True


def _looks_like_hipcc_version(value: str) -> bool:
    lower = value.lower()
    return any(marker in lower for marker in ("hip version", "hipcc", "amd clang", "clang version"))


def _looks_like_amd_gpu_label(value: str) -> bool:
    lower = value.lower()
    return any(marker in lower for marker in ("amd", "radeon", "instinct"))


def _normalized_gpu_label(value: str) -> str:
    return " ".join(value.lower().split())


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


def validate_cn_diagnostic_rollup_evidence(payload: Mapping[str, Any]) -> None:
    """Validate post-run benchmark rollup evidence for an accepted c>N artifact.

    This check intentionally reads the live rollup files and is not called by
    the retained benchmark emitter: the artifact has to exist before humans can
    update ``benchmarks/README.md`` and ``benchmarks/CHANGELOG.md``. Run this
    after the rollup docs are updated, before treating a c>N retained row as a
    promoted performance claim.
    """

    errors: list[str] = []
    try:
        validate_cn_diagnostic_artifact_payload(payload)
    except ValueError as exc:
        errors.append(str(exc))

    artifact_path = payload.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        errors.append("artifact_path must be a non-empty string for rollup evidence")
    else:
        _validate_benchmark_results_artifact_path("artifact_path", artifact_path, errors)

    rollup = payload.get("benchmark_rollup")
    if not isinstance(rollup, Mapping):
        errors.append("benchmark_rollup must be an object for rollup evidence")
        rollup = {}
    if rollup.get("artifact_path") != artifact_path:
        errors.append("benchmark_rollup.artifact_path must match artifact_path for rollup evidence")
    if rollup.get("source_artifact_path") != artifact_path:
        errors.append("benchmark_rollup.source_artifact_path must match artifact_path for rollup evidence")
    readme_text = _validate_rollup_file_mentions_artifact(
        "benchmark_rollup.readme_path",
        rollup.get("readme_path"),
        expected_path="benchmarks/README.md",
        artifact_path=artifact_path,
        errors=errors,
    )
    readme_last_updated_date = _validate_rollup_readme_metadata(readme_text, errors)
    changelog_text = _validate_rollup_file_mentions_artifact(
        "benchmark_rollup.changelog_path",
        rollup.get("changelog_path"),
        expected_path="benchmarks/CHANGELOG.md",
        artifact_path=artifact_path,
        errors=errors,
    )
    _validate_rollup_changelog_metadata(changelog_text, artifact_path, readme_last_updated_date, errors)

    if errors:
        raise ValueError("invalid c>N benchmark rollup evidence: " + "; ".join(errors))


def _validate_rollup_file_mentions_artifact(
    field: str,
    value: Any,
    *,
    expected_path: str,
    artifact_path: Any,
    errors: list[str],
) -> str | None:
    if value != expected_path:
        errors.append(f"{field} must be {expected_path} for rollup evidence")
        return None
    path = Path(value)
    try:
        text = path.read_text()
    except FileNotFoundError:
        errors.append(f"{field} file does not exist for rollup evidence")
        return None
    if not isinstance(artifact_path, str) or not artifact_path:
        return text
    normalized_text = text.replace("\\", "/")
    normalized_artifact_path = artifact_path.replace("\\", "/")
    if normalized_artifact_path not in normalized_text:
        errors.append(f"{field} must mention artifact_path for rollup evidence")
    return text


def _validate_rollup_readme_metadata(text: str | None, errors: list[str]) -> str | None:
    if text is None:
        return None
    match = _ROLLUP_LAST_UPDATED_RE.search(text)
    if match is None:
        errors.append("benchmark_rollup.readme_path must include Last updated YYYY-MM-DD metadata for rollup evidence")
        return None
    return match.group(1)


def _validate_rollup_changelog_metadata(
    text: str | None,
    artifact_path: Any,
    readme_last_updated_date: str | None,
    errors: list[str],
) -> None:
    if text is None:
        return
    if _ROLLUP_DATED_CHANGELOG_RE.search(text) is None:
        errors.append("benchmark_rollup.changelog_path must include a dated YYYY-MM-DD entry for rollup evidence")
    if not isinstance(artifact_path, str) or not artifact_path:
        return
    normalized_artifact_path = artifact_path.replace("\\", "/")
    artifact_lines = [line for line in text.replace("\\", "/").splitlines() if normalized_artifact_path in line]
    if not artifact_lines:
        return
    dated_artifact_entries = [(_ROLLUP_DATED_CHANGELOG_RE.search(line), line) for line in artifact_lines]
    dated_artifact_lines = [line for match, line in dated_artifact_entries if match is not None]
    if not dated_artifact_lines:
        errors.append("benchmark_rollup.changelog_path artifact entry must include YYYY-MM-DD date for rollup evidence")
        return
    dated_artifact_dates = [match.group(1) for match, _line in dated_artifact_entries if match is not None]
    if readme_last_updated_date is not None and readme_last_updated_date not in dated_artifact_dates:
        errors.append("benchmark_rollup.changelog_path artifact entry date must match benchmark_rollup.readme_path Last updated date for rollup evidence")
    if not any(_ROLLUP_OLD_NEW_RE.search(line) for line in dated_artifact_lines):
        errors.append("benchmark_rollup.changelog_path artifact entry must include numeric old→new metric marker for rollup evidence")
    if not any(_ROLLUP_PERCENT_DELTA_RE.search(line) for line in dated_artifact_lines):
        errors.append("benchmark_rollup.changelog_path artifact entry must include percent delta for rollup evidence")


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
    _validate_claimed_generated_token_equality(payload, correctness, workload, errors)

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
        _validate_no_disallowed_diagnostic_metadata(payload, path="", errors=errors)
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
    if payload.get("schema") != 3 or isinstance(payload.get("schema"), bool):
        errors.append("schema must be 3 for accepted artifacts")
    if payload.get("mode") != RETAINED_ARTIFACT_ACCEPTED_MODE:
        errors.append(f"mode must be {RETAINED_ARTIFACT_ACCEPTED_MODE} for accepted artifacts")
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary:
        errors.append("summary must be a non-empty string for accepted artifacts")
    elif summary != RETAINED_ARTIFACT_ACCEPTED_SUMMARY:
        errors.append(f"summary must be {RETAINED_ARTIFACT_ACCEPTED_SUMMARY} for accepted artifacts")
    if payload.get("performance_claim") is not True:
        errors.append("accepted retained artifact must set performance_claim=true")
    decision = payload.get("decision")
    if not isinstance(decision, Mapping) or decision.get("accepted") is not True:
        errors.append("accepted retained artifact must set decision.accepted=true")
    elif decision.get("reason") != RETAINED_ARTIFACT_ACCEPTED_DECISION_REASON:
        errors.append(f"accepted retained artifact decision.reason must be {RETAINED_ARTIFACT_ACCEPTED_DECISION_REASON}")
    notes = payload.get("notes")
    if not _is_nonempty_string_list(notes):
        errors.append("notes must be a non-empty string list for accepted artifacts")
    elif isinstance(notes, list):
        for note in RETAINED_ARTIFACT_ACCEPTED_NOTES:
            if note not in notes:
                errors.append(f"notes must include {note!r} for accepted artifacts")

    observability = _mapping_at(payload, "observability", errors)
    workload = _mapping_at(payload, "workload", errors)
    concurrency = workload.get("concurrency")
    concurrency_valid = isinstance(concurrency, int) and not isinstance(concurrency, bool) and concurrency > 1
    run_tag = payload.get("run_tag")
    if not isinstance(run_tag, str) or not run_tag:
        errors.append("run_tag must be a non-empty string for accepted artifacts")
    elif concurrency_valid and run_tag != f"qwen35-paro-c{concurrency}-native-retained":
        errors.append("run_tag must match qwen35-paro-c<workload.concurrency>-native-retained for accepted artifacts")
    timestamp = payload.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        errors.append("timestamp must be a non-empty ISO-8601 string for accepted artifacts")
    else:
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            errors.append("timestamp must be a parseable ISO-8601 timestamp for accepted artifacts")
        else:
            if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() is None:
                errors.append("timestamp must include timezone offset for accepted artifacts")
    prompt_tokens = workload.get("prompt_tokens_per_request")
    gen_tokens = workload.get("gen_tokens_per_request")
    workload_shape = workload.get("shape")
    if not isinstance(workload_shape, str) or not workload_shape:
        errors.append("workload.shape must be a non-empty string for accepted artifacts")
    elif (
        concurrency_valid
        and isinstance(prompt_tokens, int)
        and not isinstance(prompt_tokens, bool)
        and isinstance(gen_tokens, int)
        and not isinstance(gen_tokens, bool)
        and workload_shape != f"c={concurrency} prompt={prompt_tokens} decode={gen_tokens}"
    ):
        errors.append("workload.shape must match c=<workload.concurrency> prompt=<workload.prompt_tokens_per_request> decode=<workload.gen_tokens_per_request> for accepted artifacts")
    artifact_rows = payload.get("rows")
    if not isinstance(artifact_rows, int) or isinstance(artifact_rows, bool):
        errors.append("rows must be an int for accepted artifacts")
    elif concurrency_valid and artifact_rows != concurrency:
        errors.append("rows must match workload.concurrency for accepted artifacts")
    for field in _REQUIRED_ACCEPTED_OBSERVABILITY_FIELDS:
        if not isinstance(observability.get(field), Mapping):
            errors.append(f"observability.{field} must be an object for accepted artifacts")
    for field in ("admission_timestamps", "completion_timestamps"):
        row_map = observability.get(field)
        if concurrency_valid and isinstance(row_map, Mapping) and len(row_map) != concurrency:
            errors.append(f"observability.{field} length must match workload.concurrency for accepted artifacts")
        if isinstance(row_map, Mapping) and any(not _is_finite_number(value) for value in row_map.values()):
            errors.append(f"observability.{field} values must be finite numeric for accepted artifacts")
    admission_timestamps = observability.get("admission_timestamps")
    completion_timestamps = observability.get("completion_timestamps")
    if isinstance(admission_timestamps, Mapping) and isinstance(completion_timestamps, Mapping):
        for request_id, admission_timestamp in admission_timestamps.items():
            completion_timestamp = completion_timestamps.get(request_id)
            if _is_finite_number(admission_timestamp) and _is_finite_number(completion_timestamp) and float(completion_timestamp) <= float(admission_timestamp):
                errors.append("observability.completion_timestamps must be greater than admission_timestamps for accepted artifacts")
                break
    latency_samples: list[Any] | None = None
    latency = observability.get("request_latency_seconds")
    if isinstance(latency, Mapping):
        p50 = latency.get("p50")
        p95 = latency.get("p95")
        if not _is_positive_number(p50):
            errors.append("observability.request_latency_seconds.p50 must be positive numeric for accepted artifacts")
        if not _is_positive_number(p95):
            errors.append("observability.request_latency_seconds.p95 must be positive numeric for accepted artifacts")
        if _is_positive_number(p50) and _is_positive_number(p95) and float(p95) < float(p50):
            errors.append("observability.request_latency_seconds.p95 must be >= p50 for accepted artifacts")
        samples = latency.get("samples")
        if not isinstance(samples, list) or not samples:
            errors.append("observability.request_latency_seconds.samples must be a non-empty list for accepted artifacts")
        else:
            latency_samples = samples
            if any(not _is_positive_number(sample) for sample in samples):
                errors.append("observability.request_latency_seconds.samples must contain only positive numbers for accepted artifacts")
            else:
                ordered_samples = sorted(float(sample) for sample in samples)
                midpoint = len(ordered_samples) // 2
                expected_p50 = (
                    ordered_samples[midpoint]
                    if len(ordered_samples) % 2 == 1
                    else (ordered_samples[midpoint - 1] + ordered_samples[midpoint]) / 2.0
                )
                p95_index = min(len(ordered_samples) - 1, math.ceil(0.95 * len(ordered_samples)) - 1)
                expected_p95 = ordered_samples[p95_index]
                if _is_positive_number(p50) and not _numbers_close(float(p50), expected_p50):
                    errors.append("observability.request_latency_seconds.p50 must match request_latency_seconds.samples median for accepted artifacts")
                if _is_positive_number(p95) and not _numbers_close(float(p95), expected_p95):
                    errors.append("observability.request_latency_seconds.p95 must match request_latency_seconds.samples p95 for accepted artifacts")
            if concurrency_valid and len(samples) != concurrency:
                errors.append("observability.request_latency_seconds.samples length must match workload.concurrency for accepted artifacts")
    per_request = observability.get("per_request")
    if not isinstance(per_request, Mapping) or not per_request:
        errors.append("observability.per_request must be a non-empty object for accepted artifacts")
    else:
        if concurrency_valid and len(per_request) != concurrency:
            errors.append("observability.per_request length must match workload.concurrency for accepted artifacts")
        per_request_keys = set(per_request.keys())
        if concurrency_valid and per_request_keys != {str(request_id) for request_id in range(int(concurrency))}:
            errors.append("observability.per_request keys must match workload.concurrency row ids for accepted artifacts")
        for field in ("admission_timestamps", "completion_timestamps"):
            row_map = observability.get(field)
            if isinstance(row_map, Mapping) and set(row_map.keys()) != per_request_keys:
                errors.append(f"observability.{field} keys must match observability.per_request keys for accepted artifacts")
        if isinstance(admission_timestamps, Mapping) and isinstance(completion_timestamps, Mapping) and isinstance(latency_samples, list):
            ordered_request_ids = sorted(
                per_request_keys,
                key=lambda key: (0, int(key)) if isinstance(key, str) and key.isdigit() else (1, str(key)),
            )
            for index, request_id in enumerate(ordered_request_ids):
                if index >= len(latency_samples):
                    break
                admission_timestamp = admission_timestamps.get(request_id)
                completion_timestamp = completion_timestamps.get(request_id)
                latency_sample = latency_samples[index]
                if _is_finite_number(admission_timestamp) and _is_finite_number(completion_timestamp) and _is_positive_number(latency_sample):
                    expected_latency = float(completion_timestamp) - float(admission_timestamp)
                    tolerance = max(1e-9, abs(expected_latency) * 1e-6)
                    if expected_latency > 0.0 and abs(float(latency_sample) - expected_latency) > tolerance:
                        errors.append("observability.request_latency_seconds.samples must match completion_timestamps minus admission_timestamps for accepted artifacts")
                        break
        if isinstance(admission_timestamps, Mapping) and isinstance(completion_timestamps, Mapping):
            for request_id, row in per_request.items():
                admission_timestamp = admission_timestamps.get(request_id)
                completion_timestamp = completion_timestamps.get(request_id)
                if not isinstance(row, Mapping) or not _is_finite_number(admission_timestamp) or not _is_finite_number(completion_timestamp):
                    continue
                request_latency = float(completion_timestamp) - float(admission_timestamp)
                timing_fields = (row.get("queue_seconds"), row.get("prefill_seconds"), row.get("decode_seconds"))
                if request_latency > 0.0 and all(_is_nonnegative_number(value) for value in timing_fields):
                    component_total = sum(float(value) for value in timing_fields)
                    if component_total - request_latency > max(1e-9, request_latency * 1e-6):
                        errors.append("observability.per_request timing components must not exceed completion minus admission for accepted artifacts")
                        break
        expected_active_c = int(concurrency) if concurrency_valid else None
        expected_mode = None
        expected_context_bucket = None
        expected_context_buckets: set[int] = set()
        expected_active_mask = None
        expected_top_k = None
        expected_experts_per_token = None
        expected_replay_steps = None
        expected_draft_depth = None
        execution = payload.get("execution")
        scheduler_metadata = execution.get("scheduler_metadata") if isinstance(execution, Mapping) else None
        decode_shape_key = scheduler_metadata.get("decode_shape_key") if isinstance(scheduler_metadata, Mapping) else None
        if isinstance(decode_shape_key, Mapping):
            mode = decode_shape_key.get("mode")
            if isinstance(mode, str) and mode:
                expected_mode = mode
            context_bucket = decode_shape_key.get("context_bucket")
            if isinstance(context_bucket, int) and not isinstance(context_bucket, bool):
                expected_context_bucket = int(context_bucket)
                expected_context_buckets.add(int(context_bucket))
            active_mask = decode_shape_key.get("active_mask")
            if isinstance(active_mask, list) and all(isinstance(active, bool) for active in active_mask):
                expected_active_mask = "".join("1" if active else "0" for active in active_mask)
            top_k = decode_shape_key.get("top_k")
            if isinstance(top_k, int) and not isinstance(top_k, bool):
                expected_top_k = int(top_k)
            experts_per_token = decode_shape_key.get("experts_per_token")
            if isinstance(experts_per_token, int) and not isinstance(experts_per_token, bool):
                expected_experts_per_token = int(experts_per_token)
            replay_steps = decode_shape_key.get("replay_steps")
            if isinstance(replay_steps, int) and not isinstance(replay_steps, bool):
                expected_replay_steps = int(replay_steps)
            draft_depth = decode_shape_key.get("draft_depth")
            if isinstance(draft_depth, int) and not isinstance(draft_depth, bool):
                expected_draft_depth = int(draft_depth)
        observed_decode_shape_keys = scheduler_metadata.get("decode_shape_keys_observed") if isinstance(scheduler_metadata, Mapping) else None
        if isinstance(observed_decode_shape_keys, list):
            for observed_shape_key in observed_decode_shape_keys:
                if not isinstance(observed_shape_key, Mapping):
                    continue
                observed_context_bucket = observed_shape_key.get("context_bucket")
                if isinstance(observed_context_bucket, int) and not isinstance(observed_context_bucket, bool):
                    expected_context_buckets.add(int(observed_context_bucket))
        workload_kv_dtype = workload.get("kv_storage_dtype")
        expected_kv_storage_dtype = workload_kv_dtype if isinstance(workload_kv_dtype, str) else None
        max_layers = workload.get("max_layers")
        expected_layer_plan = f"max_layers={max_layers}" if isinstance(max_layers, int) and not isinstance(max_layers, bool) else None
        for row in per_request.values():
            _valid_request_observability(
                row,
                errors,
                expected_active_c=expected_active_c,
                expected_mode=expected_mode,
                expected_context_bucket=expected_context_bucket,
                expected_context_buckets=expected_context_buckets,
                expected_active_mask=expected_active_mask,
                expected_kv_storage_dtype=expected_kv_storage_dtype,
                expected_layer_plan=expected_layer_plan,
                expected_top_k=expected_top_k,
                expected_experts_per_token=expected_experts_per_token,
                expected_replay_steps=expected_replay_steps,
                expected_draft_depth=expected_draft_depth,
            )

    memory = _mapping_at(payload, "memory", errors)
    if concurrency_valid:
        if memory.get("max_batch_size") != concurrency:
            errors.append("memory.max_batch_size must match workload.concurrency for accepted artifacts")
    prompt_tokens = workload.get("prompt_tokens_per_request")
    warmup_tokens = workload.get("warmup_decode_tokens")
    gen_tokens = workload.get("gen_tokens_per_request")
    if all(isinstance(value, int) and not isinstance(value, bool) for value in (prompt_tokens, warmup_tokens, gen_tokens)):
        expected_sequence_length = int(prompt_tokens) + int(warmup_tokens) + int(gen_tokens) + 1
        max_sequence_length = memory.get("max_sequence_length")
        if not isinstance(max_sequence_length, int) or isinstance(max_sequence_length, bool) or max_sequence_length < expected_sequence_length:
            errors.append("memory.max_sequence_length must cover workload prompt + warmup + decode tokens for accepted artifacts")
    if memory.get("kv_storage_dtype") != workload.get("kv_storage_dtype"):
        errors.append("memory.kv_storage_dtype must match workload.kv_storage_dtype for accepted artifacts")
    memory_kv_policy = memory.get("kv_policy")
    workload_kv_policy = workload.get("kv_policy")
    if not isinstance(memory_kv_policy, Mapping):
        errors.append("memory.kv_policy must be an object for accepted artifacts")
    elif isinstance(workload_kv_policy, Mapping) and dict(memory_kv_policy) != dict(workload_kv_policy):
        errors.append("memory.kv_policy must match workload.kv_policy for accepted artifacts")
    allocator_peak = memory.get("allocator_reserved_peak_bytes")
    if not _is_nonnegative_number(allocator_peak):
        errors.append("memory.allocator_reserved_peak_bytes must be finite non-negative numeric for accepted artifacts")
    allocator_stats = memory.get("allocator_memory_stats")
    if not isinstance(allocator_stats, Mapping):
        errors.append("memory.allocator_memory_stats must be an object for accepted artifacts")
    else:
        stats_current = allocator_stats.get("current_allocated_bytes")
        stats_peak = allocator_stats.get("peak_allocated_bytes")
        if not _is_nonnegative_number(stats_current):
            errors.append("memory.allocator_memory_stats.current_allocated_bytes must be finite non-negative numeric for accepted artifacts")
        if not _is_nonnegative_number(stats_peak):
            errors.append("memory.allocator_memory_stats.peak_allocated_bytes must be finite non-negative numeric for accepted artifacts")
        elif _is_nonnegative_number(allocator_peak) and int(stats_peak) != int(allocator_peak):
            errors.append("memory.allocator_memory_stats.peak_allocated_bytes must match allocator_reserved_peak_bytes for accepted artifacts")
        if _is_nonnegative_number(stats_current) and _is_nonnegative_number(stats_peak) and float(stats_current) > float(stats_peak):
            errors.append("memory.allocator_memory_stats.current_allocated_bytes must be <= peak_allocated_bytes for accepted artifacts")
        total_allocated = allocator_stats.get("total_allocated_bytes")
        total_freed = allocator_stats.get("total_freed_bytes")
        for field, value in (("total_allocated_bytes", total_allocated), ("total_freed_bytes", total_freed)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(f"memory.allocator_memory_stats.{field} must be a non-negative int for accepted artifacts")
        active_allocations = allocator_stats.get("active_allocations")
        peak_allocations = allocator_stats.get("peak_allocations")
        for field, value in (("active_allocations", active_allocations), ("peak_allocations", peak_allocations)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(f"memory.allocator_memory_stats.{field} must be a non-negative int for accepted artifacts")
        if (
            isinstance(active_allocations, int)
            and not isinstance(active_allocations, bool)
            and active_allocations >= 0
            and isinstance(peak_allocations, int)
            and not isinstance(peak_allocations, bool)
            and peak_allocations >= 0
            and active_allocations > peak_allocations
        ):
            errors.append("memory.allocator_memory_stats.active_allocations must be <= peak_allocations for accepted artifacts")
    for field in _REQUIRED_ACCEPTED_POOL_FIELDS:
        if not isinstance(memory.get(field), Mapping):
            errors.append(f"memory.{field} must be an object for accepted artifacts")
    dynamic_pool = memory.get("dynamic_pool")
    if isinstance(dynamic_pool, Mapping):
        if not isinstance(dynamic_pool.get("enabled"), bool):
            errors.append("memory.dynamic_pool.enabled must be a bool for accepted artifacts")
        evidence = dynamic_pool.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            errors.append("memory.dynamic_pool.evidence must be a non-empty string for accepted artifacts")
        pool_counters = dynamic_pool.get("pool_counters")
        if not isinstance(pool_counters, Mapping):
            errors.append("memory.dynamic_pool.pool_counters must be an object for accepted artifacts")
        else:
            for field in _REQUIRED_ACCEPTED_POOL_COUNTER_FIELDS:
                value = pool_counters.get(field)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    errors.append(f"memory.dynamic_pool.pool_counters.{field} must be a non-negative int for accepted artifacts")
            current_bytes = pool_counters.get("current_bytes")
            high_water_bytes = pool_counters.get("high_water_observed_bytes")
            if (
                _is_nonnegative_number(current_bytes)
                and _is_nonnegative_number(high_water_bytes)
                and float(high_water_bytes) < float(current_bytes)
            ):
                errors.append("memory.dynamic_pool.pool_counters.high_water_observed_bytes must be >= current_bytes for accepted artifacts")
        for field in ("grow_events", "shrink_events"):
            value = dynamic_pool.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(f"memory.dynamic_pool.{field} must be a non-negative int for accepted artifacts")
            elif isinstance(pool_counters, Mapping) and _is_nonnegative_number(pool_counters.get(field)) and value != int(pool_counters[field]):
                errors.append(f"memory.dynamic_pool.{field} must match memory.dynamic_pool.pool_counters.{field} for accepted artifacts")
    stable_block_id = memory.get("stable_block_id")
    if isinstance(stable_block_id, Mapping):
        if stable_block_id.get("passed") is not True:
            errors.append("memory.stable_block_id.passed must be true for accepted artifacts")
        audit = stable_block_id.get("audit")
        if not isinstance(audit, str) or not audit.strip():
            errors.append("memory.stable_block_id.audit must be a non-empty string for accepted artifacts")
    prefix_sharing = memory.get("prefix_sharing")
    if isinstance(prefix_sharing, Mapping):
        prefix_enabled = prefix_sharing.get("enabled")
        prefix_savings = prefix_sharing.get("savings_bytes")
        if not isinstance(prefix_enabled, bool):
            errors.append("memory.prefix_sharing.enabled must be a bool for accepted artifacts")
        if not _is_nonnegative_number(prefix_savings):
            errors.append("memory.prefix_sharing.savings_bytes must be finite non-negative numeric for accepted artifacts")
        elif prefix_enabled is False and float(prefix_savings) != 0.0:
            errors.append("memory.prefix_sharing.savings_bytes must be 0 when prefix sharing is disabled for accepted artifacts")


def _validate_accepted_execution_gates(payload: Mapping[str, Any], errors: list[str]) -> None:
    execution = _mapping_at(payload, "execution", errors)
    workload = _mapping_at(payload, "workload", errors)
    batch_execution = execution.get("batch_execution")
    if not isinstance(batch_execution, Mapping):
        errors.append("execution.batch_execution must be an object for accepted artifacts")
        return
    for field in _REQUIRED_WORKLOAD_FLAGS:
        if workload.get(field) is not True:
            errors.append(f"workload.{field} must be true for accepted artifacts")
    if batch_execution.get("scheduler_owned") is not True:
        errors.append("execution.batch_execution.scheduler_owned must be true for accepted artifacts")
    if batch_execution.get("blockers") != []:
        errors.append("execution.batch_execution.blockers must be empty for accepted artifacts")
    for field in _REQUIRED_BATCH_EXECUTION_FLAGS:
        if batch_execution.get(field) is not True:
            errors.append(f"execution.batch_execution.{field} must be true for accepted artifacts")
    for diagnostic_field in DECODE_EXECUTION_DIAGNOSTIC_TRACE_FIELDS:
        if diagnostic_field in batch_execution:
            errors.append(f"execution.batch_execution.{diagnostic_field} must be absent for native retained decode for accepted artifacts")
    native_prefill_plan = batch_execution.get("native_prefill_plan")
    if not isinstance(native_prefill_plan, Mapping):
        errors.append("execution.batch_execution.native_prefill_plan must be an object for accepted artifacts")
    else:
        if native_prefill_plan.get("path") != "single_request_native_full":
            errors.append("execution.batch_execution.native_prefill_plan.path must be single_request_native_full for accepted artifacts")
        if native_prefill_plan.get("full_layer_limit_native") is not True:
            errors.append("execution.batch_execution.native_prefill_plan.full_layer_limit_native must be true for accepted artifacts")
        if "first_unsupported_layer" not in native_prefill_plan or native_prefill_plan.get("first_unsupported_layer") is not None:
            errors.append("execution.batch_execution.native_prefill_plan.first_unsupported_layer must be null for accepted artifacts")
        if "first_unsupported_type" not in native_prefill_plan or native_prefill_plan.get("first_unsupported_type") is not None:
            errors.append("execution.batch_execution.native_prefill_plan.first_unsupported_type must be null for accepted artifacts")
        layer_limit = native_prefill_plan.get("layer_limit")
        max_layers = workload.get("max_layers")
        if isinstance(layer_limit, bool) or not isinstance(layer_limit, int):
            errors.append("execution.batch_execution.native_prefill_plan.layer_limit must be an int for accepted artifacts")
        elif isinstance(max_layers, int) and not isinstance(max_layers, bool) and layer_limit != max_layers:
            errors.append("execution.batch_execution.native_prefill_plan.layer_limit must match workload.max_layers for accepted artifacts")
        if native_prefill_plan.get("blockers") != []:
            errors.append("execution.batch_execution.native_prefill_plan.blockers must be empty for accepted artifacts")
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
    if not isinstance(decode_execution, Mapping):
        errors.append("execution.batch_execution.decode_execution must be an object for accepted artifacts")
    else:
        prompt_tokens_per_request = workload.get("prompt_tokens_per_request")
        max_full_attention_context = decode_execution.get("max_full_attention_context")
        max_full_attention_context_valid = isinstance(max_full_attention_context, int) and not isinstance(max_full_attention_context, bool)
        if not max_full_attention_context_valid:
            errors.append("execution.batch_execution.decode_execution.max_full_attention_context must be an int for accepted artifacts")
        elif (
            isinstance(prompt_tokens_per_request, int)
            and not isinstance(prompt_tokens_per_request, bool)
            and max_full_attention_context < prompt_tokens_per_request
        ):
            errors.append("execution.batch_execution.decode_execution.max_full_attention_context must cover workload.prompt_tokens_per_request for accepted artifacts")
        elif max_full_attention_context >= 1024:
            errors.append("execution.batch_execution.decode_execution.max_full_attention_context must be < 1024 until row-aware split-K native decode lands for accepted artifacts")
        native_full_attention_layers = decode_execution.get("native_full_attention_layers")
        if (
            isinstance(native_full_attention_layers, bool)
            or not isinstance(native_full_attention_layers, int)
            or native_full_attention_layers <= 0
        ):
            errors.append("execution.batch_execution.decode_execution.native_full_attention_layers must be a positive int for accepted artifacts")
        concurrency = workload.get("concurrency")
        decode_rows = decode_execution.get("rows")
        if isinstance(decode_rows, bool) or not isinstance(decode_rows, int):
            errors.append("execution.batch_execution.decode_execution.rows must be an int for accepted artifacts")
        elif isinstance(concurrency, int) and not isinstance(concurrency, bool) and decode_rows != concurrency:
            errors.append("execution.batch_execution.decode_execution.rows must match workload.concurrency for accepted artifacts")
        decode_slots = decode_execution.get("slots")
        if not isinstance(decode_slots, list):
            errors.append("execution.batch_execution.decode_execution.slots must be a list for accepted artifacts")
        elif isinstance(concurrency, int) and not isinstance(concurrency, bool):
            if len(decode_slots) != concurrency:
                errors.append("execution.batch_execution.decode_execution.slots length must match workload.concurrency for accepted artifacts")
            elif not all(isinstance(slot, int) and not isinstance(slot, bool) and slot >= 0 for slot in decode_slots):
                errors.append("execution.batch_execution.decode_execution.slots entries must be non-negative ints for accepted artifacts")
            elif len(set(decode_slots)) != len(decode_slots):
                errors.append("execution.batch_execution.decode_execution.slots entries must be unique for accepted artifacts")
        moe_decode_rows = decode_execution.get("moe_decode_rows")
        if isinstance(moe_decode_rows, bool) or not isinstance(moe_decode_rows, int):
            errors.append("execution.batch_execution.decode_execution.moe_decode_rows must be an int for accepted artifacts")
        elif isinstance(concurrency, int) and not isinstance(concurrency, bool) and moe_decode_rows != concurrency:
            errors.append("execution.batch_execution.decode_execution.moe_decode_rows must match workload.concurrency for accepted artifacts")
        moe_grouped_compact_layers = decode_execution.get("moe_grouped_compact_layers")
        if isinstance(moe_grouped_compact_layers, bool) or not isinstance(moe_grouped_compact_layers, int) or moe_grouped_compact_layers <= 0:
            errors.append("execution.batch_execution.decode_execution.moe_grouped_compact_layers must be a positive int for accepted artifacts")
        if decode_execution.get("moe_selected_c1_fallback_layers") != 0:
            errors.append("execution.batch_execution.decode_execution.moe_selected_c1_fallback_layers must be zero for accepted artifacts")
        if decode_execution.get("moe_decode_path") != "grouped_compact":
            errors.append("execution.batch_execution.decode_execution.moe_decode_path must be grouped_compact for accepted artifacts")
        if decode_execution.get("full_attention_decode_path") != "native_batch":
            errors.append("execution.batch_execution.decode_execution.full_attention_decode_path must be native_batch for accepted artifacts")
        full_attention_input_path = decode_execution.get("full_attention_input_decode_path")
        if full_attention_input_path not in {None, "native_batch"}:
            errors.append(
                "execution.batch_execution.decode_execution.full_attention_input_decode_path must be native_batch or absent for accepted artifacts"
            )
        full_attention_context_path = decode_execution.get("full_attention_context_decode_path")
        if full_attention_context_path not in {None, "native_batch"}:
            errors.append(
                "execution.batch_execution.decode_execution.full_attention_context_decode_path must be native_batch or absent for accepted artifacts"
            )
        post_attention_path = decode_execution.get("post_attention_decode_path")
        if post_attention_path not in {None, "native_batch"}:
            errors.append(
                "execution.batch_execution.decode_execution.post_attention_decode_path must be native_batch or absent for accepted artifacts"
            )
        linear_projection_path = decode_execution.get("linear_attention_projection_path")
        if linear_projection_path not in {None, "native_batch"}:
            errors.append(
                "execution.batch_execution.decode_execution.linear_attention_projection_path must be native_batch or absent for accepted artifacts"
            )
        linear_state_path = decode_execution.get("linear_attention_state_path")
        if linear_state_path not in {None, "native_segments"}:
            errors.append(
                "execution.batch_execution.decode_execution.linear_attention_state_path must be native_segments or absent for accepted artifacts"
            )
        linear_output_path = decode_execution.get("linear_attention_output_path")
        if linear_output_path not in {None, "native_batch", "batch_gemv"}:
            errors.append(
                "execution.batch_execution.decode_execution.linear_attention_output_path must be native_batch, batch_gemv, or absent for accepted artifacts"
            )
        if decode_execution.get("native_caware_decode") is not True:
            errors.append("execution.batch_execution.decode_execution.native_caware_decode must be true for accepted artifacts")
        for diagnostic_field in DECODE_EXECUTION_DIAGNOSTIC_TRACE_FIELDS:
            if diagnostic_field in decode_execution:
                errors.append(
                    f"execution.batch_execution.decode_execution.{diagnostic_field} must be absent for native retained decode for accepted artifacts"
                )
        _validate_accepted_decode_layer_executions(decode_execution, workload, errors)
        if decode_execution.get("blockers") != []:
            errors.append("execution.batch_execution.decode_execution.blockers must be empty for accepted artifacts")
        sampler_execution = decode_execution.get("sampler_execution")
        if not isinstance(sampler_execution, Mapping):
            errors.append("execution.batch_execution.decode_execution.sampler_execution must be an object for accepted artifacts")
        else:
            _validate_accepted_sampler_execution(sampler_execution, workload, errors)
    _validate_accepted_projection_dispatch(payload, batch_execution, workload, errors)
    scheduler_metadata = execution.get("scheduler_metadata")
    if not isinstance(scheduler_metadata, Mapping):
        errors.append("execution.scheduler_metadata must be an object for accepted artifacts")
    else:
        _validate_accepted_scheduler_metadata(scheduler_metadata, workload, errors)


def _validate_accepted_decode_layer_executions(
    decode_execution: Mapping[str, Any],
    workload: Mapping[str, Any],
    errors: list[str],
) -> None:
    layer_executions = decode_execution.get("layer_executions")
    if not isinstance(layer_executions, list) or not layer_executions:
        errors.append("execution.batch_execution.decode_execution.layer_executions must be a non-empty list for accepted artifacts")
        return
    decode_slots = decode_execution.get("slots")
    concurrency = workload.get("concurrency")
    prompt_tokens_per_request = workload.get("prompt_tokens_per_request")
    native_full_attention_layers = decode_execution.get("native_full_attention_layers")
    moe_grouped_compact_layers = decode_execution.get("moe_grouped_compact_layers")
    traced_native_full_attention_layers = 0
    traced_grouped_moe_layers = 0
    for index, layer in enumerate(layer_executions):
        label = f"execution.batch_execution.decode_execution.layer_executions[{index}]"
        if not isinstance(layer, Mapping):
            errors.append(f"{label} must be an object for accepted artifacts")
            continue
        layer_index = layer.get("layer_index")
        if isinstance(layer_index, bool) or not isinstance(layer_index, int) or layer_index < 0:
            errors.append(f"{label}.layer_index must be a non-negative int for accepted artifacts")
        layer_type = layer.get("layer_type")
        if layer_type not in {"linear_attention", "full_attention"}:
            errors.append(f"{label}.layer_type must be linear_attention or full_attention for accepted artifacts")
        rows = layer.get("rows")
        if isinstance(rows, bool) or not isinstance(rows, int):
            errors.append(f"{label}.rows must be an int for accepted artifacts")
        elif isinstance(concurrency, int) and not isinstance(concurrency, bool) and rows != concurrency:
            errors.append(f"{label}.rows must match workload.concurrency for accepted artifacts")
        slots = layer.get("slots")
        if isinstance(decode_slots, list) and slots != decode_slots:
            errors.append(f"{label}.slots must match decode_execution.slots for accepted artifacts")
        elif not isinstance(slots, list):
            errors.append(f"{label}.slots must be a list for accepted artifacts")
        if layer.get("native_caware_decode") is not True:
            errors.append(f"{label}.native_caware_decode must be true for accepted artifacts")
        moe_path = layer.get("moe_decode_path")
        if moe_path != "grouped_compact":
            errors.append(f"{label}.moe_decode_path must be grouped_compact for accepted artifacts")
        else:
            traced_grouped_moe_layers += 1
        full_attention_path = layer.get("full_attention_decode_path")
        if layer_type == "full_attention":
            if full_attention_path != "native_batch":
                errors.append(f"{label}.full_attention_decode_path must be native_batch for accepted artifacts")
            else:
                traced_native_full_attention_layers += 1
            max_context = layer.get("max_context")
            if isinstance(max_context, bool) or not isinstance(max_context, int):
                errors.append(f"{label}.max_context must be an int for accepted artifacts")
            elif (
                isinstance(prompt_tokens_per_request, int)
                and not isinstance(prompt_tokens_per_request, bool)
                and max_context < prompt_tokens_per_request
            ):
                errors.append(f"{label}.max_context must cover workload.prompt_tokens_per_request for accepted artifacts")
            elif max_context >= 1024:
                errors.append(f"{label}.max_context must be < 1024 until row-aware split-K native decode lands for accepted artifacts")
            if "num_splits_per_row" in layer:
                errors.append(f"{label}.num_splits_per_row must be absent for native retained decode for accepted artifacts")
            if "full_attention_input_decode_path" in layer:
                errors.append(f"{label}.full_attention_input_decode_path must be absent for native retained decode for accepted artifacts")
            if "full_attention_context_decode_path" in layer:
                errors.append(f"{label}.full_attention_context_decode_path must be absent for native retained decode for accepted artifacts")
            if "post_attention_decode_path" in layer:
                errors.append(f"{label}.post_attention_decode_path must be absent for native retained decode for accepted artifacts")
            if "attn_context_trace_source" in layer:
                errors.append(f"{label}.attn_context_trace_source must be absent for native retained decode for accepted artifacts")
        elif layer_type == "linear_attention":
            if full_attention_path != "not_applicable":
                errors.append(f"{label}.full_attention_decode_path must be not_applicable for accepted artifacts")
            linear_decode_path = layer.get("linear_attention_decode_path")
            if linear_decode_path not in {None, "native_batch_segments"}:
                errors.append(f"{label}.linear_attention_decode_path must be native_batch_segments or absent for accepted artifacts")
            linear_projection_path = layer.get("linear_attention_projection_path")
            if linear_projection_path not in {None, "native_batch"}:
                errors.append(f"{label}.linear_attention_projection_path must be native_batch or absent for accepted artifacts")
            linear_state_path = layer.get("linear_attention_state_path")
            if linear_state_path not in {None, "native_segments"}:
                errors.append(f"{label}.linear_attention_state_path must be native_segments or absent for accepted artifacts")
            linear_output_path = layer.get("linear_attention_output_path")
            if linear_output_path not in {None, "native_batch", "batch_gemv"}:
                errors.append(f"{label}.linear_attention_output_path must be native_batch, batch_gemv, or absent for accepted artifacts")
    if isinstance(native_full_attention_layers, int) and not isinstance(native_full_attention_layers, bool):
        if traced_native_full_attention_layers != native_full_attention_layers:
            errors.append("execution.batch_execution.decode_execution.layer_executions native full-attention count must match native_full_attention_layers for accepted artifacts")
    if isinstance(moe_grouped_compact_layers, int) and not isinstance(moe_grouped_compact_layers, bool):
        if traced_grouped_moe_layers != moe_grouped_compact_layers:
            errors.append("execution.batch_execution.decode_execution.layer_executions grouped MoE count must match moe_grouped_compact_layers for accepted artifacts")


def _validate_accepted_projection_dispatch(
    payload: Mapping[str, Any],
    batch_execution: Mapping[str, Any],
    workload: Mapping[str, Any],
    errors: list[str],
) -> None:
    projection_dispatch = batch_execution.get("projection_dispatch")
    if not isinstance(projection_dispatch, Mapping):
        errors.append("execution.batch_execution.projection_dispatch must be an object for accepted artifacts")
        return
    rows = projection_dispatch.get("rows")
    concurrency = workload.get("concurrency")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 1:
        errors.append("execution.batch_execution.projection_dispatch.rows must be an int > 1 for accepted artifacts")
    elif isinstance(concurrency, int) and not isinstance(concurrency, bool) and rows != concurrency:
        errors.append("execution.batch_execution.projection_dispatch.rows must match workload.concurrency for accepted artifacts")
    if projection_dispatch.get("path") != "benchmark_accepted_caware_projection":
        errors.append("execution.batch_execution.projection_dispatch.path must be benchmark_accepted_caware_projection for accepted artifacts")
    selected_candidate = projection_dispatch.get("selected_candidate")
    if not isinstance(selected_candidate, str) or not selected_candidate:
        errors.append("execution.batch_execution.projection_dispatch.selected_candidate must be a non-empty string for accepted artifacts")
    elif selected_candidate == "row_gemv":
        errors.append("execution.batch_execution.projection_dispatch.selected_candidate must not be row_gemv for accepted artifacts")
    if projection_dispatch.get("throughput_claim_eligible") is not True:
        errors.append("execution.batch_execution.projection_dispatch.throughput_claim_eligible must be true for accepted artifacts")
    if projection_dispatch.get("blockers") != []:
        errors.append("execution.batch_execution.projection_dispatch.blockers must be empty for accepted artifacts")
    selection = projection_dispatch.get("selection")
    if not isinstance(selection, Mapping):
        errors.append("execution.batch_execution.projection_dispatch.selection must be an object for accepted artifacts")
    else:
        for field in ("layer", "quant", "variant"):
            if not isinstance(selection.get(field), str) or not selection.get(field):
                errors.append(f"execution.batch_execution.projection_dispatch.selection.{field} must be a non-empty string for accepted artifacts")
        if selection.get("variant") == "row_gemv":
            errors.append("execution.batch_execution.projection_dispatch.selection.variant must not be row_gemv for accepted artifacts")
    evidence = projection_dispatch.get("evidence")
    dispatch_evidence: ProjectionDispatchEvidence | None = None
    if not isinstance(evidence, Mapping):
        errors.append("execution.batch_execution.projection_dispatch.evidence must be an object for accepted artifacts")
    else:
        try:
            dispatch_evidence = ProjectionDispatchEvidence.from_json_dict(evidence)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if dispatch_evidence.accepted is not True:
                errors.append("execution.batch_execution.projection_dispatch.evidence.accepted must be true for accepted artifacts")
            evidence_artifact_payload = _load_benchmark_results_json_artifact(
                "execution.batch_execution.projection_dispatch.evidence.artifact_path",
                dispatch_evidence.artifact_path,
                errors,
            )
            if evidence_artifact_payload is not None:
                if not _artifact_is_accepted(evidence_artifact_payload):
                    errors.append("execution.batch_execution.projection_dispatch.evidence.artifact_path artifact must be accepted for accepted artifacts")
                expected_rows = concurrency if isinstance(concurrency, int) and not isinstance(concurrency, bool) else rows
                if isinstance(expected_rows, int) and not isinstance(expected_rows, bool):
                    errors.extend(
                        f"{blocker} for accepted artifacts"
                        for blocker in projection_dispatch_evidence_payload_blockers(
                            evidence_artifact_payload,
                            dispatch_evidence,
                            rows=expected_rows,
                            label="execution.batch_execution.projection_dispatch.evidence.artifact_path",
                        )
                    )

    if "projection_dispatch_candidates" not in payload:
        errors.append("projection_dispatch_candidates must include selected projection candidate for accepted artifacts")
        return
    try:
        candidates = projection_dispatch_candidates_from_artifact(payload)
    except ValueError as exc:
        errors.append(str(exc))
        return
    if not candidates:
        errors.append("projection_dispatch_candidates must be non-empty for accepted artifacts")
        return
    if not isinstance(selected_candidate, str) or not selected_candidate:
        return
    matches = [candidate for candidate in candidates if candidate.name == selected_candidate]
    if not matches:
        errors.append("projection_dispatch_candidates must include selected_candidate for accepted artifacts")
        return
    candidate = matches[0]
    if isinstance(rows, int) and not isinstance(rows, bool) and not candidate.applies_to(rows):
        errors.append("projection_dispatch_candidates selected_candidate row bounds must include projection_dispatch.rows for accepted artifacts")
    if isinstance(selection, Mapping):
        expected_selection = candidate.selection.to_json_dict()
        actual_selection = {field: selection.get(field) for field in ("layer", "quant", "variant")}
        if actual_selection != expected_selection:
            errors.append("execution.batch_execution.projection_dispatch.selection must match selected projection_dispatch_candidates entry for accepted artifacts")
    if dispatch_evidence is not None:
        if candidate.evidence is None:
            errors.append("projection_dispatch_candidates selected_candidate evidence must be present for accepted artifacts")
        elif candidate.evidence.to_json_dict() != dispatch_evidence.to_json_dict():
            errors.append("execution.batch_execution.projection_dispatch.evidence must match selected projection_dispatch_candidates entry for accepted artifacts")


def _validate_accepted_sampler_execution(
    sampler_execution: Mapping[str, Any],
    workload: Mapping[str, Any],
    errors: list[str],
) -> None:
    concurrency = workload.get("concurrency")
    rows = sampler_execution.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int):
        errors.append("execution.batch_execution.decode_execution.sampler_execution.rows must be an int for accepted artifacts")
    elif isinstance(concurrency, int) and not isinstance(concurrency, bool) and rows != concurrency:
        errors.append("execution.batch_execution.decode_execution.sampler_execution.rows must match workload.concurrency for accepted artifacts")
    if sampler_execution.get("requested_mode") != "batched_lm_head":
        errors.append("execution.batch_execution.decode_execution.sampler_execution.requested_mode must be batched_lm_head for accepted artifacts")
    if sampler_execution.get("native_row_aware_lm_head") is not True:
        errors.append("execution.batch_execution.decode_execution.sampler_execution.native_row_aware_lm_head must be true for accepted artifacts")
    if sampler_execution.get("mode") != "batched_lm_head":
        errors.append("execution.batch_execution.decode_execution.sampler_execution.mode must be batched_lm_head for accepted artifacts")
    if sampler_execution.get("c2_equality_green") is not True:
        errors.append("execution.batch_execution.decode_execution.sampler_execution.c2_equality_green must be true for accepted artifacts")
    equality_rows = sampler_execution.get("equality_rows")
    if isinstance(equality_rows, bool) or not isinstance(equality_rows, int):
        errors.append("execution.batch_execution.decode_execution.sampler_execution.equality_rows must be an int for accepted artifacts")
    elif isinstance(concurrency, int) and not isinstance(concurrency, bool) and equality_rows != concurrency:
        errors.append("execution.batch_execution.decode_execution.sampler_execution.equality_rows must match workload.concurrency for accepted artifacts")
    equality_artifact = sampler_execution.get("equality_artifact")
    if not isinstance(equality_artifact, str) or not equality_artifact:
        errors.append("execution.batch_execution.decode_execution.sampler_execution.equality_artifact must be a non-empty string for accepted artifacts")
    else:
        equality_artifact_field = "execution.batch_execution.decode_execution.sampler_execution.equality_artifact"
        _validate_benchmark_results_artifact_path(
            equality_artifact_field,
            equality_artifact,
            errors,
        )
        equality_artifact_payload = _load_benchmark_results_json_artifact(equality_artifact_field, equality_artifact, errors)
        if equality_artifact_payload is not None:
            expected_rows = concurrency if isinstance(concurrency, int) and not isinstance(concurrency, bool) else rows
            if isinstance(expected_rows, int) and not isinstance(expected_rows, bool):
                errors.extend(
                    f"{blocker} for accepted artifacts"
                    for blocker in batch_sampler_equality_payload_blockers(
                        equality_artifact_payload,
                        rows=expected_rows,
                        label="execution.batch_execution.decode_execution.sampler_execution.equality_artifact",
                        expected_artifact_path=equality_artifact,
                    )
                )
    blockers = sampler_execution.get("blockers")
    if blockers != []:
        errors.append("execution.batch_execution.decode_execution.sampler_execution.blockers must be empty for accepted artifacts")


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
        active_mask_valid = isinstance(active_mask, list) and bool(active_mask) and not any(not isinstance(item, bool) for item in active_mask)
        if not active_mask_valid:
            errors.append("execution.scheduler_metadata.decode_shape_key.active_mask must be a non-empty bool list for accepted artifacts")
        elif isinstance(concurrency, int) and not isinstance(concurrency, bool):
            if len(active_mask) != concurrency:
                errors.append("execution.scheduler_metadata.decode_shape_key.active_mask length must match workload.concurrency for accepted artifacts")
            if sum(1 for active in active_mask if active) != concurrency:
                errors.append("execution.scheduler_metadata.decode_shape_key.active_mask true count must match workload.concurrency for accepted artifacts")
        context_bucket = decode_shape_key.get("context_bucket")
        if not isinstance(context_bucket, int) or isinstance(context_bucket, bool) or context_bucket <= 0:
            errors.append("execution.scheduler_metadata.decode_shape_key.context_bucket must be a positive int for accepted artifacts")
        else:
            prompt_lengths = workload.get("prompt_lengths")
            prompt_tokens_per_request = workload.get("prompt_tokens_per_request")
            required_context_bucket: int | None = None
            if isinstance(prompt_lengths, list) and prompt_lengths and all(isinstance(item, int) and not isinstance(item, bool) for item in prompt_lengths):
                required_context_bucket = max(prompt_lengths)
            elif isinstance(prompt_tokens_per_request, int) and not isinstance(prompt_tokens_per_request, bool):
                required_context_bucket = prompt_tokens_per_request
            if required_context_bucket is not None and context_bucket < required_context_bucket:
                errors.append("execution.scheduler_metadata.decode_shape_key.context_bucket must cover workload prompt length for accepted artifacts")
        key_kv_dtype = decode_shape_key.get("kv_storage_dtype")
        workload_kv_dtype = workload.get("kv_storage_dtype")
        if not isinstance(key_kv_dtype, str) or not key_kv_dtype.strip():
            errors.append("execution.scheduler_metadata.decode_shape_key.kv_storage_dtype must be a non-empty string for accepted artifacts")
        elif isinstance(workload_kv_dtype, str) and key_kv_dtype != workload_kv_dtype:
            errors.append("execution.scheduler_metadata.decode_shape_key.kv_storage_dtype must match workload.kv_storage_dtype for accepted artifacts")
        key_layer_plan = decode_shape_key.get("layer_plan")
        max_layers = workload.get("max_layers")
        expected_layer_plan = f"max_layers={max_layers}" if isinstance(max_layers, int) and not isinstance(max_layers, bool) else None
        if not isinstance(key_layer_plan, str) or not key_layer_plan.strip():
            errors.append("execution.scheduler_metadata.decode_shape_key.layer_plan must be a non-empty string for accepted artifacts")
        elif expected_layer_plan is not None and key_layer_plan != expected_layer_plan:
            errors.append("execution.scheduler_metadata.decode_shape_key.layer_plan must match workload.max_layers for accepted artifacts")
        for field in ("top_k", "experts_per_token", "draft_depth"):
            value = decode_shape_key.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(f"execution.scheduler_metadata.decode_shape_key.{field} must be a non-negative int for accepted artifacts")
        replay_steps = decode_shape_key.get("replay_steps")
        if not isinstance(replay_steps, int) or isinstance(replay_steps, bool) or replay_steps <= 0:
            errors.append("execution.scheduler_metadata.decode_shape_key.replay_steps must be a positive int for accepted artifacts")
        tree_shape = decode_shape_key.get("tree_shape")
        if not isinstance(tree_shape, list) or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in tree_shape):
            errors.append("execution.scheduler_metadata.decode_shape_key.tree_shape must be a list of non-negative ints for accepted artifacts")
    graph_stats = scheduler_metadata.get("graph_bucket_stats")
    if not isinstance(graph_stats, Mapping):
        errors.append("execution.scheduler_metadata.graph_bucket_stats must be an object for accepted artifacts")
    else:
        for field in ("entries", "hits", "misses"):
            if not isinstance(graph_stats.get(field), int) or isinstance(graph_stats.get(field), bool) or graph_stats.get(field) < 0:
                errors.append(f"execution.scheduler_metadata.graph_bucket_stats.{field} must be a non-negative int for accepted artifacts")
        entries = graph_stats.get("entries")
        hits = graph_stats.get("hits")
        misses = graph_stats.get("misses")
        if isinstance(entries, int) and not isinstance(entries, bool) and entries <= 0:
            errors.append("execution.scheduler_metadata.graph_bucket_stats.entries must be positive for accepted artifacts")
        if isinstance(hits, int) and not isinstance(hits, bool) and hits <= 0:
            errors.append("execution.scheduler_metadata.graph_bucket_stats.hits must be positive for accepted artifacts")
        replay_hit_rate = graph_stats.get("replay_hit_rate")
        replay_hit_rate_valid = _is_positive_number(replay_hit_rate) and float(replay_hit_rate) <= 1.0
        if not replay_hit_rate_valid:
            errors.append("execution.scheduler_metadata.graph_bucket_stats.replay_hit_rate must be finite positive <= 1 for accepted artifacts")
        if (
            replay_hit_rate_valid
            and isinstance(hits, int)
            and not isinstance(hits, bool)
            and isinstance(misses, int)
            and not isinstance(misses, bool)
            and hits + misses > 0
            and not _numbers_close(float(replay_hit_rate), float(hits) / float(hits + misses))
        ):
            errors.append("execution.scheduler_metadata.graph_bucket_stats.replay_hit_rate must match hits / (hits + misses) for accepted artifacts")
        if (
            isinstance(entries, int)
            and not isinstance(entries, bool)
            and isinstance(hits, int)
            and not isinstance(hits, bool)
            and isinstance(misses, int)
            and not isinstance(misses, bool)
            and entries > hits + misses
        ):
            errors.append("execution.scheduler_metadata.graph_bucket_stats.entries must be covered by hits plus misses for accepted artifacts")
        miss_reasons = graph_stats.get("miss_reasons")
        if not isinstance(miss_reasons, Mapping):
            errors.append("execution.scheduler_metadata.graph_bucket_stats.miss_reasons must be an object for accepted artifacts")
        else:
            miss_reason_total = _validate_non_negative_int_mapping(
                "execution.scheduler_metadata.graph_bucket_stats.miss_reasons",
                miss_reasons,
                errors,
            )
            misses = graph_stats.get("misses")
            if isinstance(misses, int) and not isinstance(misses, bool):
                if misses > 0 and not miss_reasons:
                    errors.append("execution.scheduler_metadata.graph_bucket_stats.miss_reasons must be non-empty when misses is positive")
                if miss_reason_total is not None and miss_reason_total != misses:
                    errors.append("execution.scheduler_metadata.graph_bucket_stats.miss_reasons counts must sum to misses")
        kernel_time_histogram = graph_stats.get("kernel_time_histogram_ns")
        if not isinstance(kernel_time_histogram, Mapping):
            errors.append("execution.scheduler_metadata.graph_bucket_stats.kernel_time_histogram_ns must be an object for accepted artifacts")
        else:
            invalid_histogram_buckets = sorted(
                str(key) for key in kernel_time_histogram if isinstance(key, str) and key not in _GRAPH_KERNEL_TIME_HISTOGRAM_BUCKET_SET
            )
            allowed = ", ".join(GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS)
            if invalid_histogram_buckets:
                errors.append(
                    f"execution.scheduler_metadata.graph_bucket_stats.kernel_time_histogram_ns keys must be one of {allowed} for accepted artifacts"
                )
            if set(kernel_time_histogram) != _GRAPH_KERNEL_TIME_HISTOGRAM_BUCKET_SET:
                errors.append(
                    f"execution.scheduler_metadata.graph_bucket_stats.kernel_time_histogram_ns must include exactly the fixed buckets {allowed} for accepted artifacts"
                )
            histogram_total = _validate_non_negative_int_mapping(
                "execution.scheduler_metadata.graph_bucket_stats.kernel_time_histogram_ns",
                kernel_time_histogram,
                errors,
            )
            if not kernel_time_histogram or histogram_total == 0:
                errors.append("execution.scheduler_metadata.graph_bucket_stats.kernel_time_histogram_ns must contain at least one observation for accepted artifacts")
            if (
                histogram_total is not None
                and isinstance(hits, int)
                and not isinstance(hits, bool)
                and hits > 0
                and histogram_total < hits
            ):
                errors.append("execution.scheduler_metadata.graph_bucket_stats.kernel_time_histogram_ns observation count must cover graph_bucket_stats.hits for accepted artifacts")


def _validate_non_negative_int_mapping(label: str, values: Mapping[str, Any], errors: list[str]) -> int | None:
    total = 0
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            errors.append(f"{label} keys must be non-empty strings")
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"{label}.{key} must be a non-negative int")
            return None
        total += value
    return total


def _validate_accepted_payload_artifact_path(payload: Mapping[str, Any], errors: list[str]) -> Any:
    artifact_path = payload.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        errors.append("artifact_path must be a non-empty string for accepted artifacts")
    else:
        _validate_benchmark_results_artifact_path("artifact_path", artifact_path, errors)
        if not artifact_path.replace("\\", "/").rsplit("/", 1)[-1].endswith(".json"):
            errors.append("artifact_path must end with .json for accepted artifacts")
    return artifact_path


def _validate_accepted_benchmark_rollup_declaration(
    payload: Mapping[str, Any],
    *,
    payload_artifact_path: Any,
    errors: list[str],
) -> None:
    rollup = payload.get("benchmark_rollup")
    if not isinstance(rollup, Mapping):
        errors.append("benchmark_rollup must be an object for accepted artifacts")
        return
    if rollup.get("artifact_path") != payload_artifact_path:
        errors.append("benchmark_rollup.artifact_path must match artifact_path for accepted artifacts")
    if rollup.get("source_artifact_path") != payload_artifact_path:
        errors.append("benchmark_rollup.source_artifact_path must match artifact_path for accepted artifacts")
    if rollup.get("readme_path") != "benchmarks/README.md":
        errors.append("benchmark_rollup.readme_path must be benchmarks/README.md for accepted artifacts")
    if rollup.get("changelog_path") != "benchmarks/CHANGELOG.md":
        errors.append("benchmark_rollup.changelog_path must be benchmarks/CHANGELOG.md for accepted artifacts")


def _validate_accepted_evidence_fields(payload: Mapping[str, Any], errors: list[str]) -> None:
    payload_artifact_path = _validate_accepted_payload_artifact_path(payload, errors)
    _validate_accepted_benchmark_rollup_declaration(payload, payload_artifact_path=payload_artifact_path, errors=errors)
    workload = _mapping_at(payload, "workload", errors)
    hardware = _mapping_at(payload, "hardware", errors)
    if not hardware:
        errors.append("hardware must be a non-empty object for accepted artifacts")
    for field in _REQUIRED_ACCEPTED_HARDWARE_FIELDS:
        if not isinstance(hardware.get(field), str) or not hardware.get(field):
            errors.append(f"hardware.{field} must be a non-empty string for accepted artifacts")
    hardware_gpu = hardware.get("gpu")
    if isinstance(hardware_gpu, str) and hardware_gpu and not _looks_like_amd_gpu_label(hardware_gpu):
        errors.append("hardware.gpu must identify an AMD/Radeon/Instinct GPU for accepted artifacts")
    hardware_arch = hardware.get("arch")
    if isinstance(hardware_arch, str) and hardware_arch and _ACCEPTED_HARDWARE_ARCH_RE.fullmatch(hardware_arch) is None:
        errors.append("hardware.arch must be a gfx* architecture string for accepted artifacts")
    visible_device = hardware.get("visible_device")
    if isinstance(visible_device, Mapping) and "env" in visible_device:
        _validate_device_env_metadata(visible_device.get("env"), prefix="hardware.visible_device", errors=errors)
    visible_device_name = visible_device.get("device_name") if isinstance(visible_device, Mapping) else None
    if (
        isinstance(hardware_gpu, str)
        and hardware_gpu
        and isinstance(visible_device_name, str)
        and visible_device_name
        and _normalized_gpu_label(hardware_gpu) != _normalized_gpu_label(visible_device_name)
    ):
        errors.append("hardware.visible_device.device_name must match hardware.gpu for accepted artifacts")
    for field in _REQUIRED_ACCEPTED_HARDWARE_CAPTURE_FIELDS:
        command_fragment = "rocm-smi" if field == "rocm_smi" else field
        _validate_capture_context(
            f"hardware.{field}",
            hardware.get(field),
            errors,
            command_fragment=command_fragment,
        )
        capture = hardware.get(field)
        command = capture.get("command") if isinstance(capture, Mapping) else None
        if isinstance(command, str):
            for fragment in _REQUIRED_ACCEPTED_HARDWARE_CAPTURE_COMMAND_FRAGMENTS[field]:
                if fragment not in command:
                    errors.append(f"hardware.{field}.command must include {fragment} for accepted artifacts")
    rocminfo = hardware.get("rocminfo")
    if isinstance(hardware_arch, str) and hardware_arch and isinstance(rocminfo, Mapping):
        rocminfo_output = rocminfo.get("output")
        if isinstance(rocminfo_output, str):
            if "Name:" not in rocminfo_output:
                errors.append("hardware.rocminfo.output must include Name: for accepted artifacts")
            if hardware_arch not in rocminfo_output:
                errors.append("hardware.rocminfo.output must include hardware.arch for accepted artifacts")
    rocm_smi = hardware.get("rocm_smi")
    if isinstance(rocm_smi, Mapping):
        rocm_smi_output = rocm_smi.get("output")
        if isinstance(rocm_smi_output, str):
            rocm_smi_output_lower = rocm_smi_output.lower()
            if "gpu" not in rocm_smi_output_lower or "vram" not in rocm_smi_output_lower:
                errors.append("hardware.rocm_smi.output must include GPU and VRAM markers for accepted artifacts")
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
    hipcc_version = software.get("hipcc_version")
    if not isinstance(hipcc_version, str) or not hipcc_version:
        errors.append("software.hipcc_version must be a non-empty string for accepted artifacts")
    elif not _looks_like_hipcc_version(hipcc_version):
        errors.append("software.hipcc_version must include a hipcc/HIP/clang version marker for accepted artifacts")
    commands = _mapping_at(payload, "commands", errors)
    for field in _REQUIRED_ACCEPTED_COMMAND_FIELDS:
        command_value = commands.get(field)
        if not isinstance(command_value, str) or not command_value:
            errors.append(f"commands.{field} must be a non-empty string for accepted artifacts")
        elif isinstance(command_value, str):
            _append_disallowed_accepted_diagnostic_text_errors(command_value, path=f"commands.{field}", errors=errors)
    device_selection_env_requirements = _accepted_device_selection_env_requirements(payload, errors)
    _validate_hardware_visible_device_env_requirements(hardware, device_selection_env_requirements, errors)
    _validate_primitive_device_env_requirements(payload, device_selection_env_requirements, errors)
    _validate_hardware_visible_device_matches_primitive(payload, errors)
    environment_commands = commands.get("environment")
    if not _is_nonempty_string_list(environment_commands):
        errors.append("commands.environment must be a non-empty string list for accepted artifacts")
    elif isinstance(environment_commands, list):
        joined_environment_commands = "\n".join(environment_commands)
        for fragment in _REQUIRED_ACCEPTED_ENVIRONMENT_COMMAND_FRAGMENTS:
            if fragment not in joined_environment_commands:
                errors.append(f"commands.environment must include {fragment} for accepted artifacts")
        for command in _REQUIRED_ACCEPTED_ENVIRONMENT_COMMANDS:
            if command not in environment_commands:
                errors.append(f"commands.environment must include exact command `{command}` for accepted artifacts")
        _append_disallowed_accepted_diagnostic_text_errors(
            joined_environment_commands,
            path="commands.environment",
            errors=errors,
        )
    benchmark_command = commands.get("benchmark")
    benchmark_device_env: dict[str, str] = {}
    if isinstance(benchmark_command, str):
        benchmark_device_env = _retained_bench_command_device_env(benchmark_command)
        try:
            benchmark_command_argv = shlex.split(benchmark_command)
        except ValueError:
            benchmark_command_argv = []
        _validate_command_device_env_assignments_unique(benchmark_command_argv, field="benchmark", errors=errors)
        _validate_device_env_assignments_nonblank(benchmark_device_env, field="benchmark", errors=errors)
        _validate_device_env_assignments_have_metadata(
            benchmark_device_env,
            field="benchmark",
            requirements=device_selection_env_requirements,
            errors=errors,
        )
        if _RETAINED_BENCH_SCRIPT not in benchmark_command:
            errors.append("commands.benchmark must reference scripts/qwen35_batch_retained_bench.py for accepted artifacts")
        else:
            _validate_retained_bench_command_target(benchmark_command, field="benchmark", errors=errors)
            _validate_retained_bench_unique_flags(benchmark_command, field="benchmark", errors=errors)
            _validate_command_model_fixture_flags(benchmark_command, field="benchmark", workload=workload, errors=errors)
            _validate_command_workload_shape(benchmark_command, field="benchmark", payload=payload, errors=errors)
            _validate_command_json_matches_artifact_path(
                benchmark_command,
                field="benchmark",
                artifact_field="artifact_path",
                artifact_path=payload_artifact_path,
                errors=errors,
            )
            _validate_retained_benchmark_reference_paths(benchmark_command, field="benchmark", payload=payload, errors=errors)
            _validate_command_device_selection_env(
                benchmark_command,
                field="benchmark",
                script=_RETAINED_BENCH_SCRIPT,
                requirements=device_selection_env_requirements,
                errors=errors,
            )
    correctness_command = commands.get("correctness_reference")
    if isinstance(correctness_command, str):
        correctness_command_lower = correctness_command.lower()
        if "generated-token equality" not in correctness_command_lower or "independent c=1" not in correctness_command_lower:
            errors.append("commands.correctness_reference must name generated-token equality vs independent c=1 for accepted artifacts")
        if _PRIMITIVE_CORRECTNESS_SCRIPT not in correctness_command:
            errors.append("commands.correctness_reference must reference scripts/qwen35_batch_correctness.py for accepted artifacts")
        else:
            correctness_script_argv = _embedded_python_script_argv(
                correctness_command,
                _PRIMITIVE_CORRECTNESS_SCRIPT,
                field="correctness_reference",
                errors=errors,
            )
            correctness_script_command = shlex.join(correctness_script_argv) if correctness_script_argv is not None else correctness_command
            _validate_correctness_script_argv_shape(correctness_script_argv, field="correctness_reference", errors=errors)
            _validate_command_unique_flags(correctness_command, _CORRECTNESS_REFERENCE_UNIQUE_FLAGS, field="correctness_reference", errors=errors)
            rows_match = _CORRECTNESS_ROWS_RE.search(correctness_script_command)
            if rows_match is None:
                errors.append("commands.correctness_reference must include --rows <workload.concurrency> for accepted artifacts")
            else:
                workload = payload.get("workload")
                concurrency = workload.get("concurrency") if isinstance(workload, Mapping) else None
                if isinstance(concurrency, int) and not isinstance(concurrency, bool) and int(rows_match.group(1)) != concurrency:
                    errors.append("commands.correctness_reference --rows must match workload.concurrency for accepted artifacts")
            correctness = payload.get("correctness")
            primitive = correctness.get("primitive_batch_correctness") if isinstance(correctness, Mapping) else None
            primitive_seed = primitive.get("seed") if isinstance(primitive, Mapping) else None
            seed_match = _CORRECTNESS_SEED_RE.search(correctness_script_command)
            if seed_match is None:
                errors.append("commands.correctness_reference must include --seed <correctness.primitive_batch_correctness.seed> for accepted artifacts")
            elif isinstance(primitive_seed, int) and not isinstance(primitive_seed, bool) and int(seed_match.group(1)) != primitive_seed:
                errors.append("commands.correctness_reference --seed must match correctness.primitive_batch_correctness.seed for accepted artifacts")
            primitive_artifact_path = primitive.get("artifact_path") if isinstance(primitive, Mapping) else None
            _validate_command_json_matches_artifact_path(
                correctness_script_command,
                field="correctness_reference",
                artifact_field="correctness.primitive_batch_correctness.artifact_path",
                artifact_path=primitive_artifact_path,
                errors=errors,
            )
            _validate_command_device_selection_env(
                correctness_command,
                field="correctness_reference",
                script=_PRIMITIVE_CORRECTNESS_SCRIPT,
                requirements=device_selection_env_requirements,
                errors=errors,
            )
            correctness_device_env_argv = _script_invocation_device_env_prefix_argv(
                correctness_command,
                _PRIMITIVE_CORRECTNESS_SCRIPT,
            )
            if correctness_device_env_argv is not None:
                _validate_command_device_env_assignments_unique(
                    correctness_device_env_argv,
                    field="correctness_reference",
                    errors=errors,
                )
            correctness_device_env = _script_invocation_device_env_assignments(correctness_command, _PRIMITIVE_CORRECTNESS_SCRIPT)
            _validate_device_env_assignments_nonblank(correctness_device_env, field="correctness_reference", errors=errors)
            _validate_device_env_assignments_have_metadata(
                correctness_device_env,
                field="correctness_reference",
                requirements=device_selection_env_requirements,
                errors=errors,
            )
            if correctness_device_env != benchmark_device_env:
                errors.append("commands.correctness_reference device env prefix must match commands.benchmark for accepted artifacts")
    profiler_command = commands.get("profiler")
    profiler_command_output_format: str | None = None
    profiler_command_trace_dir: str | None = None
    profiler_profiled_benchmark_command: str | None = None
    if isinstance(profiler_command, str):
        if _ROCPROF_EXECUTABLE not in profiler_command or _ROCPROF_COMMAND_FLAGS[0] not in profiler_command:
            errors.append("commands.profiler must include rocprofv3 --kernel-trace for accepted artifacts")
        try:
            profiler_command_argv = shlex.split(profiler_command)
        except ValueError:
            profiler_command_argv = []
            errors.append("commands.profiler must be shell-parseable for accepted artifacts")
        rocprof_command_argv: list[str] = []
        profiled_command_argv: list[str] = []
        if not profiler_command_argv or Path(profiler_command_argv[0]).name != _ROCPROF_EXECUTABLE:
            errors.append("commands.profiler must start with rocprofv3 for accepted artifacts")
        elif "--" not in profiler_command_argv:
            errors.append("commands.profiler must include rocprof command separator for accepted artifacts")
        else:
            separator_index = profiler_command_argv.index("--")
            rocprof_command_argv = profiler_command_argv[:separator_index]
            profiled_command_argv = profiler_command_argv[separator_index + 1 :]
            _validate_unique_flags(
                rocprof_command_argv,
                _ROCPROF_COMMAND_FLAGS,
                field="commands.profiler rocprof options",
                errors=errors,
            )
            if _ROCPROF_COMMAND_FLAGS[0] not in rocprof_command_argv:
                errors.append("commands.profiler must include --kernel-trace before rocprof separator for accepted artifacts")
            profiler_command_output_format = _argv_value(rocprof_command_argv, _ROCPROF_COMMAND_FLAGS[1])
            if profiler_command_output_format != _ROCPROF_OUTPUT_FORMAT:
                errors.append("commands.profiler must include --output-format csv before rocprof separator for accepted artifacts")
            profiler_command_trace_dir = _argv_value(rocprof_command_argv, _ROCPROF_COMMAND_FLAGS[2])
            if profiler_command_trace_dir is None:
                errors.append("commands.profiler must include -d <profiler.trace_dir> before rocprof separator for accepted artifacts")
        profiled_launch_argv = _strip_command_env_prefix(profiled_command_argv)
        if (
            len(profiled_launch_argv) < 2
            or not Path(profiled_launch_argv[0]).name.startswith("python")
            or profiled_launch_argv[1] != _RETAINED_BENCH_SCRIPT
        ):
            errors.append("commands.profiler must target scripts/qwen35_batch_retained_bench.py after rocprof separator for accepted artifacts")
        else:
            profiler_profiled_benchmark_command = shlex.join(profiled_command_argv)
            _validate_retained_bench_unique_flags(profiler_profiled_benchmark_command, field="profiler", errors=errors)
            _validate_command_model_fixture_flags(profiler_profiled_benchmark_command, field="profiler", workload=workload, errors=errors)
            if isinstance(benchmark_command, str):
                _validate_profiled_command_matches_benchmark_model_fixture(profiler_profiled_benchmark_command, benchmark_command, errors)
            _validate_command_workload_shape(profiler_profiled_benchmark_command, field="profiler", payload=payload, errors=errors)
            _validate_command_json_matches_artifact_path(
                profiler_profiled_benchmark_command,
                field="profiler",
                artifact_field="artifact_path",
                artifact_path=payload_artifact_path,
                errors=errors,
            )
            _validate_retained_benchmark_reference_paths(profiler_profiled_benchmark_command, field="profiler", payload=payload, errors=errors)
            profiler_profiled_device_env = _command_device_env_assignments(profiled_command_argv)
            _validate_command_device_env_assignments_unique(profiled_command_argv, field="profiler", errors=errors)
            _validate_device_env_assignments_nonblank(profiler_profiled_device_env, field="profiler", errors=errors)
            _validate_device_env_assignments_have_metadata(
                profiler_profiled_device_env,
                field="profiler",
                requirements=device_selection_env_requirements,
                errors=errors,
            )
            if profiler_profiled_device_env != benchmark_device_env:
                errors.append("commands.profiler device env prefix must match commands.benchmark for accepted artifacts")
            _validate_command_device_selection_env(
                profiler_command,
                field="profiler",
                script=_RETAINED_BENCH_SCRIPT,
                requirements=device_selection_env_requirements,
                errors=errors,
            )
            if "--require-cached-build" not in profiled_command_argv:
                errors.append("commands.profiler must include --require-cached-build after rocprof separator for accepted artifacts")
            compiler_version_match = _COMMAND_COMPILER_VERSION_FILE_RE.search(profiler_profiled_benchmark_command)
            if compiler_version_match is None:
                errors.append("commands.profiler must include --compiler-version-file after rocprof separator for accepted artifacts")
            else:
                _validate_benchmark_results_artifact_path(
                    "commands.profiler --compiler-version-file path",
                    compiler_version_match.group(1).strip("'\""),
                    errors,
                )
    profiler = _mapping_at(payload, "profiler", errors)
    profiler_artifact_path = profiler.get("artifact_path")
    if not isinstance(profiler_artifact_path, str) or not profiler_artifact_path:
        errors.append("profiler.artifact_path must be a non-empty string for accepted artifacts")
    else:
        _validate_benchmark_results_artifact_path("profiler.artifact_path", profiler_artifact_path, errors)
        if profiler_profiled_benchmark_command is not None:
            _validate_profiler_command_artifact_reference(profiler_profiled_benchmark_command, profiler_artifact_path, errors)
    profiler_source_artifact_path = profiler.get("source_artifact_path")
    if not isinstance(profiler_source_artifact_path, str) or not profiler_source_artifact_path:
        errors.append("profiler.source_artifact_path must be a non-empty string for accepted artifacts")
    elif isinstance(profiler_artifact_path, str) and profiler_artifact_path and profiler_source_artifact_path != profiler_artifact_path:
        errors.append("profiler.source_artifact_path must match artifact_path for accepted artifacts")
    if profiler.get("status") != "captured":
        errors.append("profiler.status must be 'captured' for accepted artifacts")
    profiler_output_format = profiler.get("output_format")
    if profiler_output_format != _ROCPROF_OUTPUT_FORMAT:
        errors.append("profiler.output_format must be 'csv' for accepted artifacts")
    elif profiler_command_output_format is not None and profiler_output_format != profiler_command_output_format:
        errors.append("profiler.output_format must match commands.profiler --output-format for accepted artifacts")
    profiler_trace_dir = profiler.get("trace_dir")
    if not isinstance(profiler_trace_dir, str) or not profiler_trace_dir:
        errors.append("profiler.trace_dir must be a non-empty string for accepted artifacts")
    elif _path_has_parent_directory_component(profiler_trace_dir):
        errors.append("profiler.trace_dir must not contain parent-directory components for accepted artifacts")
    else:
        profiler_trace_dir_path = Path(profiler_trace_dir)
        profiler_trace_dir_check_path = (
            profiler_trace_dir_path if profiler_trace_dir_path.is_absolute() else REPO_ROOT / profiler_trace_dir_path
        )
        if profiler_trace_dir_check_path.is_symlink():
            errors.append("profiler.trace_dir must not be a symlink for accepted artifacts")
        if _path_has_symlink_parent(profiler_trace_dir_check_path):
            errors.append("profiler.trace_dir parent directories must not be symlinks for accepted artifacts")
        if _path_has_non_directory_parent(profiler_trace_dir_check_path):
            errors.append("profiler.trace_dir parent directories must be directories for accepted artifacts")
        if profiler_command_trace_dir is not None and profiler_trace_dir != profiler_command_trace_dir:
            errors.append("profiler.trace_dir must match commands.profiler -d for accepted artifacts")
    profiler_trace_files = profiler.get("trace_files")
    if not _is_nonempty_string_list(profiler_trace_files):
        errors.append("profiler.trace_files must be a non-empty string list for accepted artifacts")
    elif isinstance(profiler_trace_files, list) and isinstance(profiler_trace_dir, str) and profiler_trace_dir:
        if len(set(profiler_trace_files)) != len(profiler_trace_files):
            errors.append("profiler.trace_files entries must be unique for accepted artifacts")
        for trace_file in profiler_trace_files:
            trace_path = Path(trace_file)
            if trace_path.suffix.lower() != ".csv":
                errors.append("profiler.trace_files entries must be CSV paths for accepted artifacts")
                break
            trace_check_path = trace_path if trace_path.is_absolute() else REPO_ROOT / trace_path
            if trace_check_path.is_symlink():
                errors.append("profiler.trace_files entries must not be symlinks for accepted artifacts")
                break
            if _path_has_symlink_parent(trace_check_path):
                errors.append("profiler.trace_files parent directories must not be symlinks for accepted artifacts")
                break
            if _path_has_non_directory_parent(trace_check_path):
                errors.append("profiler.trace_files parent directories must be directories for accepted artifacts")
                break
            if _path_has_parent_directory_component(trace_file):
                errors.append("profiler.trace_files must not contain parent-directory components for accepted artifacts")
                break
            if not _is_resolved_path_relative_to(trace_file, profiler_trace_dir):
                errors.append("profiler.trace_files must be under profiler.trace_dir for accepted artifacts")
                break
        if not any(_is_kernel_trace_csv_path(trace_file) for trace_file in profiler_trace_files):
            errors.append("profiler.trace_files must include a kernel-trace CSV path for accepted artifacts")
    _validate_profiler_synthesized_fields(profiler, errors)
    profiler_trace_kernel_names = profiler.get("trace_kernel_names")
    profiler_trace_kernel_names_valid = _is_stripped_nonempty_string_list(profiler_trace_kernel_names)
    if not profiler_trace_kernel_names_valid:
        errors.append("profiler.trace_kernel_names must be a non-empty string list for accepted artifacts")
    elif isinstance(profiler_trace_kernel_names, list):
        if len(set(profiler_trace_kernel_names)) != len(profiler_trace_kernel_names):
            errors.append("profiler.trace_kernel_names entries must be unique for accepted artifacts")
        if not any("batch" in kernel_name.lower() for kernel_name in profiler_trace_kernel_names):
            errors.append("profiler.trace_kernel_names must include at least one native batch kernel name for accepted artifacts")
        if any(_has_disallowed_profiler_kernel_fragment(kernel_name) for kernel_name in profiler_trace_kernel_names):
            errors.append("profiler.trace_kernel_names must not include serial/per-row/fallback kernel names for accepted artifacts")
    if profiler.get("expected_kernels_present") is not True:
        errors.append("profiler.expected_kernels_present must be true for accepted artifacts")
    expected_kernel_names = profiler.get("expected_kernel_names")
    if not _is_stripped_nonempty_string_list(expected_kernel_names):
        errors.append("profiler.expected_kernel_names must be a non-empty string list for accepted artifacts")
    elif isinstance(expected_kernel_names, list):
        _validate_expected_profiler_kernel_names(expected_kernel_names, errors)
    kernel_durations = profiler.get("kernel_durations_ns")
    if not isinstance(kernel_durations, Mapping) or not kernel_durations:
        errors.append("profiler.kernel_durations_ns must be a non-empty object for accepted artifacts")
    else:
        for kernel_name, duration_ns in kernel_durations.items():
            if not _is_stripped_nonempty_string(kernel_name):
                errors.append("profiler.kernel_durations_ns keys must be non-empty strings for accepted artifacts")
                break
            if _has_disallowed_profiler_kernel_fragment(kernel_name):
                errors.append("profiler.kernel_durations_ns must not include serial/per-row/fallback kernel names for accepted artifacts")
                break
            if not _is_positive_number(duration_ns):
                errors.append(f"profiler.kernel_durations_ns.{kernel_name} must be positive numeric for accepted artifacts")
        if profiler_trace_kernel_names_valid and isinstance(profiler_trace_kernel_names, list):
            missing_duration_names = sorted(
                kernel_name
                for kernel_name in kernel_durations
                if _is_stripped_nonempty_string(kernel_name)
                and not _has_disallowed_profiler_kernel_fragment(kernel_name)
                and kernel_name not in profiler_trace_kernel_names
            )
            if missing_duration_names:
                errors.append("profiler.trace_kernel_names must include profiler.kernel_durations_ns keys for accepted artifacts")
            unmeasured_trace_names = sorted(
                kernel_name
                for kernel_name in profiler_trace_kernel_names
                if _is_stripped_nonempty_string(kernel_name)
                and not _has_disallowed_profiler_kernel_fragment(kernel_name)
                and kernel_name not in kernel_durations
            )
            if unmeasured_trace_names:
                errors.append("profiler.kernel_durations_ns keys must include profiler.trace_kernel_names for accepted artifacts")
        _validate_profiler_kernel_duration_total(profiler, kernel_durations, errors)
        _validate_profiler_kernel_duration_shares(profiler, kernel_durations, errors)
        _validate_profiler_kernel_duration_categories(profiler, kernel_durations, errors)
        _validate_graph_replay_profiler_evidence(payload, profiler, kernel_durations, errors)
        _validate_sampler_profiler_evidence(payload, profiler, kernel_durations, errors)
        _validate_profiler_cpu_side_bottlenecks(profiler, errors)
        _validate_graph_histogram_profiler_coverage(payload, kernel_durations, errors)
        if isinstance(expected_kernel_names, list):
            for kernel_name in expected_kernel_names:
                if not _is_stripped_nonempty_string(kernel_name):
                    continue
                if not _is_positive_number(kernel_durations.get(kernel_name)):
                    errors.append(f"profiler.kernel_durations_ns.{kernel_name} must be positive numeric for accepted artifacts")
    _validate_projection_dispatch_profiler_evidence(payload, profiler, errors)
    _validate_optional_projection_dispatch_candidates(payload, errors)


def _sampler_profiler_name_matches(kernel_name: str) -> bool:
    lowered = kernel_name.lower()
    if any(fragment in lowered for fragment in ("serial", "per_row", "per-row", "fallback")):
        return False
    return "batch" in lowered and _profiler_kernel_duration_category(kernel_name) == "sampling"


def _validate_sampler_profiler_evidence(
    payload: Mapping[str, Any],
    profiler: Mapping[str, Any],
    kernel_durations: Mapping[Any, Any],
    errors: list[str],
) -> None:
    execution = payload.get("execution")
    batch_execution = execution.get("batch_execution") if isinstance(execution, Mapping) else None
    decode_execution = batch_execution.get("decode_execution") if isinstance(batch_execution, Mapping) else None
    sampler_execution = decode_execution.get("sampler_execution") if isinstance(decode_execution, Mapping) else None
    if not isinstance(sampler_execution, Mapping):
        return
    if sampler_execution.get("mode") != "batched_lm_head" or sampler_execution.get("native_row_aware_lm_head") is not True:
        return
    expected_kernel_names = profiler.get("expected_kernel_names")
    if isinstance(expected_kernel_names, list) and not any(
        isinstance(kernel_name, str) and _sampler_profiler_name_matches(kernel_name)
        for kernel_name in expected_kernel_names
    ):
        errors.append("profiler.expected_kernel_names must include a native batch sampler/lm_head kernel for accepted artifacts")
    trace_kernel_names = profiler.get("trace_kernel_names")
    if isinstance(trace_kernel_names, list) and not any(
        isinstance(kernel_name, str) and _sampler_profiler_name_matches(kernel_name)
        for kernel_name in trace_kernel_names
    ):
        errors.append("profiler.trace_kernel_names must include a native batch sampler/lm_head kernel for accepted artifacts")
    if not any(
        isinstance(kernel_name, str)
        and _sampler_profiler_name_matches(kernel_name)
        and _is_positive_number(duration_ns)
        for kernel_name, duration_ns in kernel_durations.items()
    ):
        errors.append("profiler.kernel_durations_ns must include a positive native batch sampler/lm_head duration for accepted artifacts")


def _validate_graph_replay_profiler_evidence(
    payload: Mapping[str, Any],
    profiler: Mapping[str, Any],
    kernel_durations: Mapping[Any, Any],
    errors: list[str],
) -> None:
    execution = payload.get("execution")
    scheduler_metadata = execution.get("scheduler_metadata") if isinstance(execution, Mapping) else None
    graph_stats = scheduler_metadata.get("graph_bucket_stats") if isinstance(scheduler_metadata, Mapping) else None
    replay_kernel_hits = graph_stats.get("replay_kernel_hits") if isinstance(graph_stats, Mapping) else None
    if isinstance(replay_kernel_hits, int) and not isinstance(replay_kernel_hits, bool):
        if replay_kernel_hits <= 0:
            return
    else:
        hits = graph_stats.get("hits") if isinstance(graph_stats, Mapping) else None
        if not isinstance(hits, int) or isinstance(hits, bool) or hits <= 0:
            return
    duration_categories = profiler.get("kernel_duration_categories_ns")
    graph_replay_duration = duration_categories.get("graph_replay") if isinstance(duration_categories, Mapping) else None
    if not _is_positive_number(graph_replay_duration):
        errors.append(
            "profiler.kernel_duration_categories_ns.graph_replay must be positive when graph_bucket_stats.hits is positive for accepted artifacts"
        )
    category_shares = profiler.get("kernel_duration_category_shares")
    graph_replay_share = category_shares.get("graph_replay") if isinstance(category_shares, Mapping) else None
    if not _is_positive_number(graph_replay_share):
        errors.append(
            "profiler.kernel_duration_category_shares.graph_replay must be positive when graph_bucket_stats.hits is positive for accepted artifacts"
        )
    expected_kernel_names = profiler.get("expected_kernel_names")
    if isinstance(expected_kernel_names, list) and not any(
        isinstance(kernel_name, str) and _profiler_kernel_duration_category(kernel_name) == "graph_replay"
        for kernel_name in expected_kernel_names
    ):
        errors.append(
            "profiler.expected_kernel_names must include a graph/replay kernel when graph_bucket_stats.hits is positive for accepted artifacts"
        )
    if not any(
        isinstance(kernel_name, str)
        and _profiler_kernel_duration_category(kernel_name) == "graph_replay"
        and _is_positive_number(duration_ns)
        for kernel_name, duration_ns in kernel_durations.items()
    ):
        errors.append(
            "profiler.kernel_durations_ns must include a positive graph/replay duration when graph_bucket_stats.hits is positive for accepted artifacts"
        )


def _validate_graph_histogram_profiler_coverage(payload: Mapping[str, Any], kernel_durations: Mapping[Any, Any], errors: list[str]) -> None:
    execution = payload.get("execution")
    scheduler_metadata = execution.get("scheduler_metadata") if isinstance(execution, Mapping) else None
    graph_stats = scheduler_metadata.get("graph_bucket_stats") if isinstance(scheduler_metadata, Mapping) else None
    histogram = graph_stats.get("kernel_time_histogram_ns") if isinstance(graph_stats, Mapping) else None
    if not isinstance(histogram, Mapping):
        return
    histogram_total = 0
    for count in histogram.values():
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            return
        histogram_total += int(count)
    profiler_integer_duration_count = 0
    expected_bucket_counts: dict[str, int] = {}
    for kernel_name, duration_ns in kernel_durations.items():
        if not isinstance(kernel_name, str) or not kernel_name or _has_disallowed_profiler_kernel_fragment(kernel_name):
            continue
        if _is_positive_number(duration_ns) and float(duration_ns).is_integer():
            profiler_integer_duration_count += 1
            bucket = _graph_kernel_time_histogram_bucket_ns(int(duration_ns))
            expected_bucket_counts[bucket] = expected_bucket_counts.get(bucket, 0) + 1
    if profiler_integer_duration_count > 0 and histogram_total < profiler_integer_duration_count:
        errors.append(
            "execution.scheduler_metadata.graph_bucket_stats.kernel_time_histogram_ns observation count must cover profiler.kernel_durations_ns for accepted artifacts"
        )
    for bucket, expected_count in sorted(expected_bucket_counts.items()):
        observed_count = histogram.get(bucket, 0)
        if not isinstance(observed_count, int) or isinstance(observed_count, bool) or observed_count < expected_count:
            errors.append(
                "execution.scheduler_metadata.graph_bucket_stats.kernel_time_histogram_ns bucket counts must cover profiler.kernel_durations_ns for accepted artifacts"
            )
            break


def _validate_profiler_kernel_duration_total(
    profiler: Mapping[str, Any],
    kernel_durations: Mapping[Any, Any],
    errors: list[str],
) -> None:
    total_duration = profiler.get("total_kernel_duration_ns")
    if not _is_positive_number(total_duration):
        errors.append("profiler.total_kernel_duration_ns must be positive numeric for accepted artifacts")
        return
    duration_sum = sum(
        float(duration_ns)
        for kernel_name, duration_ns in kernel_durations.items()
        if _is_stripped_nonempty_string(kernel_name) and _is_positive_number(duration_ns)
    )
    tolerance = max(1.0, duration_sum * 1e-6)
    if duration_sum > 0.0 and abs(float(total_duration) - duration_sum) > tolerance:
        errors.append("profiler.total_kernel_duration_ns must match sum(profiler.kernel_durations_ns) for accepted artifacts")


def _validate_profiler_kernel_duration_shares(
    profiler: Mapping[str, Any],
    kernel_durations: Mapping[Any, Any],
    errors: list[str],
) -> None:
    kernel_duration_shares = profiler.get("kernel_duration_shares")
    if not isinstance(kernel_duration_shares, Mapping) or not kernel_duration_shares:
        errors.append("profiler.kernel_duration_shares must be a non-empty object for accepted artifacts")
        return
    duration_keys = {
        kernel_name
        for kernel_name, duration_ns in kernel_durations.items()
        if _is_stripped_nonempty_string(kernel_name) and _is_positive_number(duration_ns)
    }
    share_keys = {
        kernel_name
        for kernel_name, duration_share in kernel_duration_shares.items()
        if _is_stripped_nonempty_string(kernel_name) and _is_positive_number(duration_share)
    }
    if any(not _is_stripped_nonempty_string(kernel_name) for kernel_name in kernel_duration_shares):
        errors.append("profiler.kernel_duration_shares keys must be non-empty strings for accepted artifacts")
    if duration_keys != share_keys:
        errors.append("profiler.kernel_duration_shares keys must match profiler.kernel_durations_ns for accepted artifacts")
    total_duration = profiler.get("total_kernel_duration_ns")
    if not _is_positive_number(total_duration):
        return
    share_sum = 0.0
    for kernel_name, duration_share in kernel_duration_shares.items():
        if isinstance(kernel_name, str) and _has_disallowed_profiler_kernel_fragment(kernel_name):
            errors.append("profiler.kernel_duration_shares must not include serial/per-row/fallback kernel names for accepted artifacts")
            break
        if not _is_stripped_nonempty_string(kernel_name) or not _is_positive_number(duration_share):
            errors.append(f"profiler.kernel_duration_shares.{kernel_name} must be positive numeric for accepted artifacts")
            continue
        share_sum += float(duration_share)
        duration_ns = kernel_durations.get(kernel_name)
        if _is_positive_number(duration_ns):
            expected_share = float(duration_ns) / float(total_duration)
            tolerance = max(1e-6, expected_share * 1e-6)
            if abs(float(duration_share) - expected_share) > tolerance:
                errors.append(
                    f"profiler.kernel_duration_shares.{kernel_name} must match "
                    "profiler.kernel_durations_ns/kernel total for accepted artifacts"
                )
    if share_sum > 0.0 and abs(share_sum - 1.0) > 1e-6:
        errors.append("profiler.kernel_duration_shares must sum to 1.0 for accepted artifacts")


def _validate_profiler_kernel_duration_categories(
    profiler: Mapping[str, Any],
    kernel_durations: Mapping[Any, Any],
    errors: list[str],
) -> None:
    duration_categories = profiler.get("kernel_duration_categories_ns")
    category_shares = profiler.get("kernel_duration_category_shares")
    if not isinstance(duration_categories, Mapping) or not duration_categories:
        errors.append("profiler.kernel_duration_categories_ns must be a non-empty object for accepted artifacts")
        return
    if not isinstance(category_shares, Mapping) or not category_shares:
        errors.append("profiler.kernel_duration_category_shares must be a non-empty object for accepted artifacts")
        return
    expected_keys = set(_REQUIRED_PROFILER_KERNEL_DURATION_CATEGORIES)
    duration_keys = {key for key in duration_categories if isinstance(key, str)}
    share_keys = {key for key in category_shares if isinstance(key, str)}
    if duration_keys != expected_keys:
        errors.append("profiler.kernel_duration_categories_ns keys must match required kernel duration categories")
    if share_keys != expected_keys:
        errors.append("profiler.kernel_duration_category_shares keys must match required kernel duration categories")
    if duration_keys == expected_keys:
        expected_categories = _profiler_kernel_duration_category_sums(kernel_durations)
        if any(
            _is_nonnegative_number(duration_categories.get(category))
            and abs(float(duration_categories[category]) - expected_duration) > max(1.0, expected_duration * 1e-6)
            for category, expected_duration in expected_categories.items()
        ):
            errors.append(
                "profiler.kernel_duration_categories_ns must match categorized profiler.kernel_durations_ns "
                "for accepted artifacts"
            )

    total_duration = profiler.get("total_kernel_duration_ns")
    duration_sum = 0.0
    share_sum = 0.0
    for category in _REQUIRED_PROFILER_KERNEL_DURATION_CATEGORIES:
        duration_ns = duration_categories.get(category)
        duration_share = category_shares.get(category)
        if not _is_nonnegative_number(duration_ns):
            errors.append(f"profiler.kernel_duration_categories_ns.{category} must be non-negative numeric for accepted artifacts")
            continue
        duration_sum += float(duration_ns)
        if not _is_nonnegative_number(duration_share):
            errors.append(f"profiler.kernel_duration_category_shares.{category} must be non-negative numeric for accepted artifacts")
            continue
        share_sum += float(duration_share)
        if _is_positive_number(total_duration):
            expected_share = float(duration_ns) / float(total_duration)
            tolerance = max(1e-6, expected_share * 1e-6)
            if abs(float(duration_share) - expected_share) > tolerance:
                errors.append(
                    f"profiler.kernel_duration_category_shares.{category} must match "
                    "profiler.kernel_duration_categories_ns/kernel total for accepted artifacts"
                )
    if _is_positive_number(total_duration):
        tolerance = max(1.0, float(total_duration) * 1e-6)
        if abs(duration_sum - float(total_duration)) > tolerance:
            errors.append("profiler.kernel_duration_categories_ns must sum to profiler.total_kernel_duration_ns for accepted artifacts")
        if abs(share_sum - 1.0) > 1e-6:
            errors.append("profiler.kernel_duration_category_shares must sum to 1.0 for accepted artifacts")


def _validate_profiler_cpu_side_bottlenecks(profiler: Mapping[str, Any], errors: list[str]) -> None:
    cpu_total = profiler.get("cpu_side_total_seconds")
    durations = profiler.get("cpu_side_bottlenecks_seconds")
    shares = profiler.get("cpu_side_bottleneck_shares")
    if not _is_positive_number(cpu_total):
        errors.append("profiler.cpu_side_total_seconds must be positive numeric for accepted artifacts")
        return
    if not isinstance(durations, Mapping) or not durations:
        errors.append("profiler.cpu_side_bottlenecks_seconds must be a non-empty object for accepted artifacts")
        return
    if not isinstance(shares, Mapping) or not shares:
        errors.append("profiler.cpu_side_bottleneck_shares must be a non-empty object for accepted artifacts")
        return
    expected_keys = set(_REQUIRED_PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES)
    duration_keys = {key for key in durations if isinstance(key, str)}
    share_keys = {key for key in shares if isinstance(key, str)}
    if duration_keys != expected_keys:
        errors.append("profiler.cpu_side_bottlenecks_seconds keys must match required CPU-side bottleneck categories")
    if share_keys != expected_keys:
        errors.append("profiler.cpu_side_bottleneck_shares keys must match required CPU-side bottleneck categories")

    duration_sum = 0.0
    share_sum = 0.0
    for category in _REQUIRED_PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES:
        duration_seconds = durations.get(category)
        duration_share = shares.get(category)
        if not _is_nonnegative_number(duration_seconds):
            errors.append(f"profiler.cpu_side_bottlenecks_seconds.{category} must be non-negative numeric for accepted artifacts")
            continue
        duration_sum += float(duration_seconds)
        if not _is_nonnegative_number(duration_share):
            errors.append(f"profiler.cpu_side_bottleneck_shares.{category} must be non-negative numeric for accepted artifacts")
            continue
        share_sum += float(duration_share)
        expected_share = float(duration_seconds) / float(cpu_total)
        tolerance = max(1e-6, expected_share * 1e-6)
        if abs(float(duration_share) - expected_share) > tolerance:
            errors.append(
                f"profiler.cpu_side_bottleneck_shares.{category} must match "
                "profiler.cpu_side_bottlenecks_seconds/cpu total for accepted artifacts"
            )
    tolerance = max(1e-9, float(cpu_total) * 1e-6)
    if abs(duration_sum - float(cpu_total)) > tolerance:
        errors.append("profiler.cpu_side_bottlenecks_seconds must sum to profiler.cpu_side_total_seconds for accepted artifacts")
    if abs(share_sum - 1.0) > 1e-6:
        errors.append("profiler.cpu_side_bottleneck_shares must sum to 1.0 for accepted artifacts")


def _validate_projection_dispatch_profiler_evidence(
    payload: Mapping[str, Any],
    profiler: Mapping[str, Any],
    errors: list[str],
) -> None:
    execution = payload.get("execution")
    batch_execution = execution.get("batch_execution") if isinstance(execution, Mapping) else None
    projection_dispatch = batch_execution.get("projection_dispatch") if isinstance(batch_execution, Mapping) else None
    if not isinstance(projection_dispatch, Mapping):
        return
    fragments: list[str] = []
    selected_candidate = projection_dispatch.get("selected_candidate")
    if isinstance(selected_candidate, str) and selected_candidate and selected_candidate != "row_gemv":
        fragments.append(selected_candidate.lower())
    selection = projection_dispatch.get("selection")
    variant = selection.get("variant") if isinstance(selection, Mapping) else None
    if isinstance(variant, str) and variant and variant != "row_gemv":
        fragments.append(variant.lower())
    if not fragments:
        return
    profiler_names: list[str] = []
    expected_kernel_names = profiler.get("expected_kernel_names")
    if isinstance(expected_kernel_names, list):
        expected_lower_names = [name.lower() for name in expected_kernel_names if isinstance(name, str) and name]
        if not any(fragment in name for fragment in fragments for name in expected_lower_names):
            errors.append("profiler.expected_kernel_names must include selected projection_dispatch candidate or variant for accepted artifacts")
        profiler_names.extend(name for name in expected_kernel_names if isinstance(name, str) and name)
    trace_kernel_names = profiler.get("trace_kernel_names")
    if isinstance(trace_kernel_names, list):
        trace_lower_names = [name.lower() for name in trace_kernel_names if isinstance(name, str) and name]
        if not any(fragment in name for fragment in fragments for name in trace_lower_names):
            errors.append("profiler.trace_kernel_names must include selected projection_dispatch candidate or variant for accepted artifacts")
        profiler_names.extend(name for name in trace_kernel_names if isinstance(name, str) and name)
    kernel_durations = profiler.get("kernel_durations_ns")
    if isinstance(kernel_durations, Mapping):
        duration_lower_names = [
            name.lower()
            for name, duration_ns in kernel_durations.items()
            if isinstance(name, str) and name and _is_positive_number(duration_ns)
        ]
        if not any(fragment in name for fragment in fragments for name in duration_lower_names):
            errors.append("profiler.kernel_durations_ns must include a positive selected projection_dispatch candidate or variant duration for accepted artifacts")
        profiler_names.extend(name for name in kernel_durations if isinstance(name, str) and name)
    if not profiler_names:
        return
    lowered_names = [name.lower() for name in profiler_names]
    if not any(fragment in name for fragment in fragments for name in lowered_names):
        errors.append("profiler kernel names must include selected projection_dispatch candidate or variant for accepted artifacts")


def _validate_optional_projection_dispatch_candidates(payload: Mapping[str, Any], errors: list[str]) -> None:
    if "projection_dispatch_candidates" not in payload:
        return
    try:
        projection_dispatch_candidates_from_artifact(payload)
    except ValueError as exc:
        errors.append(str(exc))


def _validate_int8_kv_primitive_layer_accuracy_gates(
    payload: Mapping[str, Any],
    correctness: Mapping[str, Any],
    errors: list[str],
) -> None:
    workload = payload.get("workload")
    workload = workload if isinstance(workload, Mapping) else {}
    if workload.get("kv_storage_dtype") != "int8_per_token_head":
        return
    evidence = correctness.get("int8_kv_primitive_layer_accuracy")
    if not isinstance(evidence, Mapping):
        errors.append("correctness.int8_kv_primitive_layer_accuracy must be an object for accepted int8_per_token_head artifacts")
        return
    expected_prompt = workload.get("prompt_tokens_per_request")
    if not isinstance(expected_prompt, int) or isinstance(expected_prompt, bool) or expected_prompt <= 0:
        expected_prompt = None
    expected_scale_dtype = None
    kv_policy = workload.get("kv_policy")
    if isinstance(kv_policy, Mapping):
        scale_format = kv_policy.get("scale_metadata_format")
        if isinstance(scale_format, Mapping) and isinstance(scale_format.get("scale_dtype"), str):
            expected_scale_dtype = scale_format.get("scale_dtype")
        elif isinstance(kv_policy.get("requested_scale_dtype"), str):
            expected_scale_dtype = kv_policy.get("requested_scale_dtype")
    if expected_scale_dtype is None and isinstance(workload.get("kv_scale_dtype"), str):
        expected_scale_dtype = workload.get("kv_scale_dtype")
    expected_scale_dtype = expected_scale_dtype or "fp16"
    gate_artifact_paths: dict[str, str] = {}
    for label, expected_device in (("cpu_reference", "cpu"), ("hip_gate", "hip")):
        entry = evidence.get(label)
        field = f"correctness.int8_kv_primitive_layer_accuracy.{label}"
        if not isinstance(entry, Mapping):
            errors.append(f"{field} must be an object for accepted int8_per_token_head artifacts")
            continue
        if entry.get("status") != "loaded":
            errors.append(f"{field}.status must be loaded for accepted int8_per_token_head artifacts")
        if entry.get("artifact_status") != "accepted":
            errors.append(f"{field}.artifact_status must be accepted for accepted int8_per_token_head artifacts")
        if entry.get("passed") is not True:
            errors.append(f"{field}.passed must be true for accepted int8_per_token_head artifacts")
        if entry.get("schema") != 1 or isinstance(entry.get("schema"), bool):
            errors.append(f"{field}.schema must be 1 for accepted int8_per_token_head artifacts")
        if entry.get("mode") != "qwen35_kv_int8_layer_accuracy":
            errors.append(f"{field}.mode must be qwen35_kv_int8_layer_accuracy for accepted int8_per_token_head artifacts")
        if entry.get("device") != expected_device:
            errors.append(f"{field}.device must be {expected_device} for accepted int8_per_token_head artifacts")
        artifact_path = entry.get("artifact_path")
        if not isinstance(artifact_path, str) or not artifact_path:
            errors.append(f"{field}.artifact_path must be a non-empty string for accepted int8_per_token_head artifacts")
        else:
            _validate_benchmark_results_artifact_path(f"{field}.artifact_path", artifact_path, errors)
            gate_artifact_paths[label] = artifact_path
        source_artifact_path = entry.get("source_artifact_path")
        if not isinstance(source_artifact_path, str) or not source_artifact_path:
            errors.append(f"{field}.source_artifact_path must be a non-empty string for accepted int8_per_token_head artifacts")
        elif isinstance(artifact_path, str) and artifact_path and source_artifact_path != artifact_path:
            errors.append(f"{field}.source_artifact_path must match artifact_path for accepted int8_per_token_head artifacts")
        shape = entry.get("shape")
        if not isinstance(shape, Mapping):
            errors.append(f"{field}.shape must be an object for accepted int8_per_token_head artifacts")
        else:
            if expected_prompt is not None and shape.get("contexts") != [expected_prompt, expected_prompt + 1]:
                errors.append(f"{field}.shape.contexts must match prompt and prompt+1 for accepted int8_per_token_head artifacts")
            for shape_field, expected_value in (("block_size", 256), ("num_q_heads", 16), ("num_kv_heads", 2), ("head_dim", 256)):
                if shape.get(shape_field) != expected_value or isinstance(shape.get(shape_field), bool):
                    errors.append(f"{field}.shape.{shape_field} must be {expected_value} for accepted int8_per_token_head artifacts")
            if shape.get("scale_dtype") != expected_scale_dtype:
                errors.append(f"{field}.shape.scale_dtype must match workload KV scale dtype for accepted int8_per_token_head artifacts")
        entry_kv_policy = entry.get("kv_policy")
        if not isinstance(entry_kv_policy, Mapping):
            errors.append(f"{field}.kv_policy must be an object for accepted int8_per_token_head artifacts")
        elif entry_kv_policy.get("storage_dtype") != "int8_per_token_head":
            errors.append(f"{field}.kv_policy.storage_dtype must be int8_per_token_head for accepted int8_per_token_head artifacts")
        blocked_reasons = entry.get("blocked_reasons")
        if not isinstance(blocked_reasons, list):
            errors.append(f"{field}.blocked_reasons must be a list for accepted int8_per_token_head artifacts")
        elif blocked_reasons:
            errors.append(f"{field}.blocked_reasons must be empty for accepted int8_per_token_head artifacts")
        correctness_failures = entry.get("correctness_failures")
        if not isinstance(correctness_failures, list):
            errors.append(f"{field}.correctness_failures must be a list for accepted int8_per_token_head artifacts")
        elif correctness_failures:
            errors.append(f"{field}.correctness_failures must be empty for accepted int8_per_token_head artifacts")
        command = entry.get("command")
        if not isinstance(command, str) or "scripts/qwen35_kv_int8_accuracy.py" not in command:
            errors.append(f"{field}.command must invoke scripts/qwen35_kv_int8_accuracy.py for accepted int8_per_token_head artifacts")
        else:
            if f"--device {expected_device}" not in command:
                errors.append(f"{field}.command must include --device {expected_device} for accepted int8_per_token_head artifacts")
            if expected_prompt is not None and f"--contexts {expected_prompt},{expected_prompt + 1}" not in command:
                errors.append(f"{field}.command must include retained prompt/context boundary coverage for accepted int8_per_token_head artifacts")
            if expected_device == "hip" and "--require-int8-hip" not in command:
                errors.append(f"{field}.command must include --require-int8-hip for accepted int8_per_token_head artifacts")
            if isinstance(artifact_path, str) and artifact_path:
                _validate_command_json_matches_artifact_path(
                    command,
                    field=field,
                    artifact_field=f"{field}.artifact_path",
                    artifact_path=artifact_path,
                    errors=errors,
                )
    commands = payload.get("commands")
    if not isinstance(commands, Mapping):
        return
    for command_field in ("benchmark", "profiler"):
        retained_command = commands.get(command_field)
        if not isinstance(retained_command, str):
            continue
        try:
            retained_argv = shlex.split(retained_command)
        except ValueError:
            continue
        for label, flag in zip(("cpu_reference", "hip_gate"), _INT8_PRIMITIVE_GATE_FLAGS, strict=True):
            artifact_path = gate_artifact_paths.get(label)
            flag_value = _argv_value(retained_argv, flag)
            if flag_value is None:
                errors.append(f"commands.{command_field} must include {flag} for accepted int8_per_token_head artifacts")
            elif isinstance(artifact_path, str) and artifact_path and flag_value != artifact_path:
                errors.append(f"commands.{command_field} {flag} must match correctness.int8_kv_primitive_layer_accuracy.{label}.artifact_path for accepted int8_per_token_head artifacts")


def _validate_claimed_generated_token_equality(
    payload: Mapping[str, Any],
    correctness: Mapping[str, Any],
    workload: Mapping[str, Any],
    errors: list[str],
) -> None:
    equality = correctness.get("generated_token_equality")
    if not isinstance(equality, Mapping) or equality.get("passed") is not True:
        return
    if equality.get("skipped") is not False:
        errors.append("correctness.generated_token_equality.skipped must be false when passed is true")
    oracle = correctness.get("oracle")
    oracle_lower = oracle.lower() if isinstance(oracle, str) else ""
    if (
        "generated-token" not in oracle_lower
        or "independent c=1" not in oracle_lower
        or ("equal" not in oracle_lower and "equality" not in oracle_lower)
    ):
        errors.append("correctness.oracle must name generated-token equality vs independent c=1 when generated_token_equality.passed is true")
    batch_sequences = equality.get("batch_sequences")
    c1_sequences = equality.get("c1_sequences")
    if not isinstance(batch_sequences, list):
        errors.append("correctness.generated_token_equality.batch_sequences must be a list when passed is true")
    if not isinstance(c1_sequences, list):
        errors.append("correctness.generated_token_equality.c1_sequences must be a list when passed is true")
    concurrency = workload.get("concurrency")
    concurrency_valid = isinstance(concurrency, int) and not isinstance(concurrency, bool) and concurrency > 1
    if not concurrency_valid:
        errors.append("workload.concurrency must be an int > 1 when generated_token_equality.passed is true")
    else:
        if isinstance(batch_sequences, list) and len(batch_sequences) != concurrency:
            errors.append("correctness.generated_token_equality.batch_sequences length must match workload.concurrency when passed is true")
        if isinstance(c1_sequences, list) and len(c1_sequences) != concurrency:
            errors.append("correctness.generated_token_equality.c1_sequences length must match workload.concurrency when passed is true")
    prompt_lengths = workload.get("prompt_lengths")
    prompt_lengths_valid = False
    if prompt_lengths is not None:
        if not isinstance(prompt_lengths, list):
            errors.append("workload.prompt_lengths must be a list when present and generated_token_equality.passed is true")
        elif concurrency_valid and len(prompt_lengths) != concurrency:
            errors.append("workload.prompt_lengths length must match workload.concurrency when generated_token_equality.passed is true")
        elif any(not isinstance(length, int) or isinstance(length, bool) or length <= 0 for length in prompt_lengths):
            errors.append("workload.prompt_lengths entries must be positive ints when generated_token_equality.passed is true")
        else:
            prompt_lengths_valid = True
    prompt_tokens_per_request = workload.get("prompt_tokens_per_request")
    prompt_tokens_per_request_valid = False
    if prompt_tokens_per_request is not None:
        if not isinstance(prompt_tokens_per_request, int) or isinstance(prompt_tokens_per_request, bool) or prompt_tokens_per_request <= 0:
            errors.append("workload.prompt_tokens_per_request must be a positive int when present and generated_token_equality.passed is true")
        elif prompt_lengths_valid and any(length != prompt_tokens_per_request for length in prompt_lengths):
            errors.append("workload.prompt_tokens_per_request must match every workload.prompt_lengths entry when generated_token_equality.passed is true")
        else:
            prompt_tokens_per_request_valid = True
    prompt_tokens_aggregate = workload.get("prompt_tokens_aggregate")
    if prompt_tokens_aggregate is not None:
        if not isinstance(prompt_tokens_aggregate, int) or isinstance(prompt_tokens_aggregate, bool) or prompt_tokens_aggregate < 0:
            errors.append("workload.prompt_tokens_aggregate must be a non-negative int when present and generated_token_equality.passed is true")
        elif prompt_lengths_valid and prompt_tokens_aggregate != sum(prompt_lengths):
            errors.append("workload.prompt_tokens_aggregate must equal sum(workload.prompt_lengths) when generated_token_equality.passed is true")
        elif concurrency_valid and prompt_tokens_per_request_valid and prompt_tokens_aggregate != int(concurrency) * int(prompt_tokens_per_request):
            errors.append("workload.prompt_tokens_aggregate must equal workload.concurrency times workload.prompt_tokens_per_request when generated_token_equality.passed is true")
        elif not prompt_lengths_valid and not (concurrency_valid and prompt_tokens_per_request_valid):
            errors.append("workload.prompt_tokens_aggregate must be backed by workload.prompt_lengths or workload.prompt_tokens_per_request when generated_token_equality.passed is true")
    gen_tokens = workload.get("gen_tokens_per_request")
    gen_tokens_valid = isinstance(gen_tokens, int) and not isinstance(gen_tokens, bool) and gen_tokens > 0
    if not gen_tokens_valid:
        errors.append("workload.gen_tokens_per_request must be a positive int when generated_token_equality.passed is true")
    gen_tokens_aggregate = workload.get("gen_tokens_aggregate")
    if gen_tokens_aggregate is not None:
        if not isinstance(gen_tokens_aggregate, int) or isinstance(gen_tokens_aggregate, bool) or gen_tokens_aggregate <= 0:
            errors.append("workload.gen_tokens_aggregate must be a positive int when present and generated_token_equality.passed is true")
        elif concurrency_valid and gen_tokens_valid and gen_tokens_aggregate != int(concurrency) * int(gen_tokens):
            errors.append("workload.gen_tokens_aggregate must equal workload.concurrency times workload.gen_tokens_per_request when generated_token_equality.passed is true")
    warmup_tokens = workload.get("warmup_decode_tokens")
    warmup_tokens_valid = isinstance(warmup_tokens, int) and not isinstance(warmup_tokens, bool) and warmup_tokens >= 0
    if not warmup_tokens_valid:
        errors.append("workload.warmup_decode_tokens must be a non-negative int when generated_token_equality.passed is true")
    expected_tokens = None
    if gen_tokens_valid and warmup_tokens_valid:
        expected_tokens = 1 + int(warmup_tokens) + int(gen_tokens)
    for label, sequences in (
        ("correctness.generated_token_equality.batch_sequences", batch_sequences),
        ("correctness.generated_token_equality.c1_sequences", c1_sequences),
    ):
        if not isinstance(sequences, list):
            continue
        for index, sequence in enumerate(sequences):
            if not isinstance(sequence, list):
                errors.append(f"{label}[{index}] must be a per-row token-id list when passed is true")
                continue
            if not sequence:
                errors.append(f"{label}[{index}] must be a non-empty per-row token-id list when passed is true")
            if expected_tokens is not None and len(sequence) != expected_tokens:
                errors.append(f"{label}[{index}] length must match seed plus workload.warmup_decode_tokens plus workload.gen_tokens_per_request when passed is true")
            if any(not _is_valid_token_id(token) for token in sequence):
                errors.append(f"{label}[{index}] must contain only non-negative token ids when passed is true")
    if isinstance(batch_sequences, list) and isinstance(c1_sequences, list) and batch_sequences != c1_sequences:
        errors.append("correctness.generated_token_equality.batch_sequences must equal c1_sequences when passed is true")
    _validate_claimed_execution_seed_tokens(
        payload,
        batch_sequences,
        int(concurrency) if concurrency_valid else None,
        errors,
    )
    _validate_claimed_execution_generated_tokens(
        payload,
        batch_sequences,
        int(concurrency) if concurrency_valid else None,
        int(gen_tokens) if gen_tokens_valid else None,
        errors,
    )
    _validate_claimed_execution_completed_tokens(
        payload,
        batch_sequences,
        int(concurrency) if concurrency_valid else None,
        int(gen_tokens) if gen_tokens_valid else None,
        int(warmup_tokens) if warmup_tokens_valid else None,
        errors,
    )
    mismatches = equality.get("mismatches")
    if not isinstance(mismatches, list):
        errors.append("correctness.generated_token_equality.mismatches must be a list when passed is true")
    elif mismatches:
        errors.append("correctness.generated_token_equality.mismatches must be empty when passed is true")


def _validate_claimed_execution_seed_tokens(
    payload: Mapping[str, Any],
    batch_sequences: Any,
    concurrency: int | None,
    errors: list[str],
) -> None:
    execution = payload.get("execution")
    seed_tokens = execution.get("seed_tokens") if isinstance(execution, Mapping) else None
    if seed_tokens is None:
        return
    if not isinstance(seed_tokens, Mapping):
        errors.append("execution.seed_tokens must be an object when generated_token_equality.passed is true")
        return
    if not seed_tokens:
        return
    if concurrency is not None and len(seed_tokens) != concurrency:
        errors.append("execution.seed_tokens length must match workload.concurrency when generated_token_equality.passed is true")
    if concurrency is None:
        return
    expected_row_keys = {str(row_index) for row_index in range(concurrency)}
    if set(seed_tokens.keys()) != expected_row_keys:
        errors.append("execution.seed_tokens keys must match workload.concurrency row ids when generated_token_equality.passed is true")
    for row_index in range(concurrency):
        row_key = str(row_index)
        token_id = _extract_claimed_token_id(
            seed_tokens.get(row_key),
            f"execution.seed_tokens.{row_key}",
            errors,
        )
        if token_id is None:
            continue
        if (
            isinstance(batch_sequences, list)
            and row_index < len(batch_sequences)
            and isinstance(batch_sequences[row_index], list)
            and batch_sequences[row_index]
            and token_id != batch_sequences[row_index][0]
        ):
            errors.append(f"execution.seed_tokens.{row_key} must match correctness.generated_token_equality.batch_sequences first token when generated_token_equality.passed is true")


def _validate_claimed_execution_generated_tokens(
    payload: Mapping[str, Any],
    batch_sequences: Any,
    concurrency: int | None,
    gen_tokens_per_request: int | None,
    errors: list[str],
) -> None:
    execution = payload.get("execution")
    generated_tokens = execution.get("generated_tokens") if isinstance(execution, Mapping) else None
    if generated_tokens is None:
        return
    if not isinstance(generated_tokens, Mapping):
        errors.append("execution.generated_tokens must be an object when generated_token_equality.passed is true")
        return
    if not generated_tokens:
        return
    if concurrency is not None and len(generated_tokens) != concurrency:
        errors.append("execution.generated_tokens length must match workload.concurrency when generated_token_equality.passed is true")
    if concurrency is None:
        return
    expected_row_keys = {str(row_index) for row_index in range(concurrency)}
    if set(generated_tokens.keys()) != expected_row_keys:
        errors.append("execution.generated_tokens keys must match workload.concurrency row ids when generated_token_equality.passed is true")
    has_any_generated_row = any(generated_tokens.get(str(row_index)) != [] for row_index in range(concurrency))
    for row_index in range(concurrency):
        row_key = str(row_index)
        row = generated_tokens.get(row_key)
        if row == []:
            if has_any_generated_row:
                errors.append(f"execution.generated_tokens.{row_key} must be non-empty when any generated-token metadata is present and generated_token_equality.passed is true")
            continue
        token_ids = _extract_claimed_generated_token_ids(
            row,
            f"execution.generated_tokens.{row_key}",
            errors,
        )
        if token_ids is None:
            continue
        if gen_tokens_per_request is not None and len(token_ids) != gen_tokens_per_request:
            errors.append(f"execution.generated_tokens.{row_key} length must match workload.gen_tokens_per_request when generated_token_equality.passed is true")
        if (
            gen_tokens_per_request is not None
            and isinstance(batch_sequences, list)
            and row_index < len(batch_sequences)
            and isinstance(batch_sequences[row_index], list)
        ):
            expected_suffix = batch_sequences[row_index][-gen_tokens_per_request:]
            if token_ids != expected_suffix:
                errors.append(f"execution.generated_tokens.{row_key} must match correctness.generated_token_equality.batch_sequences suffix when generated_token_equality.passed is true")


def _validate_claimed_execution_completed_tokens(
    payload: Mapping[str, Any],
    batch_sequences: Any,
    concurrency: int | None,
    gen_tokens_per_request: int | None,
    warmup_decode_tokens: int | None,
    errors: list[str],
) -> None:
    execution = payload.get("execution")
    completed = execution.get("completed") if isinstance(execution, Mapping) else None
    if completed is None or completed == []:
        return
    if not isinstance(completed, list):
        errors.append("execution.completed must be a list when generated_token_equality.passed is true")
        return
    if concurrency is not None and len(completed) != concurrency:
        errors.append("execution.completed length must match workload.concurrency when generated_token_equality.passed is true")
    workload = payload.get("workload")
    prompt_lengths = workload.get("prompt_lengths") if isinstance(workload, Mapping) else None
    has_any_completed_prompt_tokens = any(isinstance(row, Mapping) and "prompt_tokens" in row for row in completed)
    if has_any_completed_prompt_tokens and not isinstance(prompt_lengths, list):
        errors.append("workload.prompt_lengths must be a list when completed prompt metadata is present and generated_token_equality.passed is true")
    elif has_any_completed_prompt_tokens and any(
        not isinstance(length, int) or isinstance(length, bool) or length <= 0 for length in prompt_lengths
    ):
        errors.append("workload.prompt_lengths entries must be positive ints when completed prompt metadata is present and generated_token_equality.passed is true")
    seen_request_ids: set[int] = set()
    for index, row in enumerate(completed):
        if not isinstance(row, Mapping):
            errors.append(f"execution.completed[{index}] must be an object when generated_token_equality.passed is true")
            continue
        request_id = row.get("request_id")
        if not isinstance(request_id, int) or isinstance(request_id, bool):
            errors.append(f"execution.completed[{index}].request_id must be an int when generated_token_equality.passed is true")
            continue
        if concurrency is not None and (request_id < 0 or request_id >= concurrency):
            errors.append(f"execution.completed[{index}].request_id must be in workload.concurrency range when generated_token_equality.passed is true")
            continue
        if request_id in seen_request_ids:
            errors.append("execution.completed request_id values must be unique when generated_token_equality.passed is true")
        else:
            seen_request_ids.add(request_id)
        if row.get("finished") is not True:
            errors.append(f"execution.completed[{index}].finished must be true when generated_token_equality.passed is true")
        finish_reason = row.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason.strip():
            errors.append(f"execution.completed[{index}].finish_reason must be a non-empty string when generated_token_equality.passed is true")
        if "prompt_tokens" not in row and has_any_completed_prompt_tokens:
            errors.append(f"execution.completed[{index}].prompt_tokens must be present when any completed prompt metadata is present and generated_token_equality.passed is true")
        if "prompt_tokens" in row:
            prompt_token_ids = _extract_claimed_generated_token_ids(
                row.get("prompt_tokens"),
                f"execution.completed[{index}].prompt_tokens",
                errors,
            )
            if (
                prompt_token_ids is not None
                and isinstance(prompt_lengths, list)
                and request_id < len(prompt_lengths)
                and isinstance(prompt_lengths[request_id], int)
                and not isinstance(prompt_lengths[request_id], bool)
                and len(prompt_token_ids) != prompt_lengths[request_id]
            ):
                errors.append(f"execution.completed[{index}].prompt_tokens length must match workload.prompt_lengths when generated_token_equality.passed is true")
        if "generated_tokens" not in row:
            errors.append(f"execution.completed[{index}].generated_tokens must be present when generated_token_equality.passed is true")
            continue
        token_ids = _extract_claimed_generated_token_ids(
            row.get("generated_tokens"),
            f"execution.completed[{index}].generated_tokens",
            errors,
        )
        if token_ids is None:
            continue
        expected_completed_tokens = None
        if gen_tokens_per_request is not None:
            expected_completed_tokens = gen_tokens_per_request + (warmup_decode_tokens or 0)
        if expected_completed_tokens is not None and len(token_ids) != expected_completed_tokens:
            errors.append(f"execution.completed[{index}].generated_tokens length must match workload.warmup_decode_tokens plus workload.gen_tokens_per_request when generated_token_equality.passed is true")
        if (
            expected_completed_tokens is not None
            and isinstance(batch_sequences, list)
            and request_id < len(batch_sequences)
            and isinstance(batch_sequences[request_id], list)
        ):
            expected_suffix = batch_sequences[request_id][-expected_completed_tokens:]
            if token_ids != expected_suffix:
                errors.append(f"execution.completed[{index}].generated_tokens must match correctness.generated_token_equality.batch_sequences warmup+decode suffix when generated_token_equality.passed is true")
    if concurrency is not None:
        missing_request_ids = [str(request_id) for request_id in range(concurrency) if request_id not in seen_request_ids]
        if missing_request_ids:
            errors.append("execution.completed must include every request_id when generated_token_equality.passed is true")


def _extract_claimed_generated_token_ids(row: Any, label: str, errors: list[str]) -> list[int] | None:
    if not isinstance(row, list):
        errors.append(f"{label} must be a list when generated_token_equality.passed is true")
        return None
    token_ids: list[int] = []
    for index, item in enumerate(row):
        token_id = _extract_claimed_token_id(item, f"{label}[{index}]", errors)
        if token_id is None:
            return None
        token_ids.append(token_id)
    return token_ids


def _extract_claimed_token_id(item: Any, label: str, errors: list[str]) -> int | None:
    if _is_valid_token_id(item):
        return item
    if isinstance(item, int) and not isinstance(item, bool):
        errors.append(f"{label} must be a non-negative token id when generated_token_equality.passed is true")
        return None
    if isinstance(item, Mapping):
        token_id = item.get("token_id")
        if _is_valid_token_id(token_id):
            return token_id
        if isinstance(token_id, int) and not isinstance(token_id, bool):
            errors.append(f"{label}.token_id must be non-negative when generated_token_equality.passed is true")
            return None
    errors.append(f"{label} must be a token id or object with token_id when generated_token_equality.passed is true")
    return None


def _validate_primitive_device_metadata(device: Any, errors: list[str]) -> None:
    prefix = "correctness.primitive_batch_correctness.device"
    if not isinstance(device, Mapping):
        errors.append(f"{prefix} must be an object for accepted artifacts")
        return
    _validate_device_env_metadata(device.get("env"), prefix=prefix, errors=errors)
    _validate_device_runtime_metadata(device, prefix=prefix, errors=errors)


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
    gen_tokens = workload.get("gen_tokens_per_request")
    gen_tokens_valid = isinstance(gen_tokens, int) and not isinstance(gen_tokens, bool) and gen_tokens > 0
    warmup_tokens = workload.get("warmup_decode_tokens")
    warmup_tokens_valid = isinstance(warmup_tokens, int) and not isinstance(warmup_tokens, bool) and warmup_tokens >= 0
    if not warmup_tokens_valid:
        errors.append("workload.warmup_decode_tokens must be a non-negative int for accepted artifacts")
    equality_comparison = equality.get("comparison")
    if not isinstance(equality_comparison, str) or not equality_comparison:
        errors.append("correctness.generated_token_equality.comparison must be a non-empty string for accepted artifacts")
    elif equality_comparison != "native_batch_vs_independent_c1":
        errors.append("correctness.generated_token_equality.comparison must be native_batch_vs_independent_c1 for accepted artifacts")
    equality_rows = equality.get("rows")
    if not isinstance(equality_rows, int) or isinstance(equality_rows, bool):
        errors.append("correctness.generated_token_equality.rows must be an int for accepted artifacts")
    if not concurrency_valid:
        errors.append("workload.concurrency must be an int > 1 for accepted artifacts")
    else:
        if isinstance(equality_rows, int) and not isinstance(equality_rows, bool) and equality_rows != concurrency:
            errors.append("correctness.generated_token_equality.rows must match workload.concurrency for accepted artifacts")
        if isinstance(batch_sequences, list) and len(batch_sequences) != concurrency:
            errors.append("correctness.generated_token_equality.batch_sequences length must match workload.concurrency for accepted artifacts")
        if isinstance(c1_sequences, list) and len(c1_sequences) != concurrency:
            errors.append("correctness.generated_token_equality.c1_sequences length must match workload.concurrency for accepted artifacts")
    if gen_tokens_valid and warmup_tokens_valid:
        expected_equality_tokens = 1 + int(warmup_tokens) + int(gen_tokens)
        equality_tokens_per_sequence = equality.get("tokens_per_sequence")
        if not isinstance(equality_tokens_per_sequence, int) or isinstance(equality_tokens_per_sequence, bool):
            errors.append("correctness.generated_token_equality.tokens_per_sequence must be an int for accepted artifacts")
        elif equality_tokens_per_sequence != expected_equality_tokens:
            errors.append("correctness.generated_token_equality.tokens_per_sequence must match seed plus workload.warmup_decode_tokens plus workload.gen_tokens_per_request for accepted artifacts")
        equality_warmup_tokens = equality.get("warmup_decode_tokens")
        if not isinstance(equality_warmup_tokens, int) or isinstance(equality_warmup_tokens, bool):
            errors.append("correctness.generated_token_equality.warmup_decode_tokens must be an int for accepted artifacts")
        elif equality_warmup_tokens != int(warmup_tokens):
            errors.append("correctness.generated_token_equality.warmup_decode_tokens must match workload.warmup_decode_tokens for accepted artifacts")
        equality_gen_tokens = equality.get("gen_tokens_per_request")
        if not isinstance(equality_gen_tokens, int) or isinstance(equality_gen_tokens, bool):
            errors.append("correctness.generated_token_equality.gen_tokens_per_request must be an int for accepted artifacts")
        elif equality_gen_tokens != int(gen_tokens):
            errors.append("correctness.generated_token_equality.gen_tokens_per_request must match workload.gen_tokens_per_request for accepted artifacts")
        _validate_generated_token_sequence_lengths(
            "correctness.generated_token_equality.batch_sequences",
            batch_sequences,
            expected_equality_tokens,
            errors,
        )
        _validate_generated_token_sequence_lengths(
            "correctness.generated_token_equality.c1_sequences",
            c1_sequences,
            expected_equality_tokens,
            errors,
        )
        _validate_execution_seed_tokens(
            payload,
            batch_sequences,
            int(concurrency) if concurrency_valid else None,
            errors,
        )
        _validate_execution_generated_tokens(
            payload,
            batch_sequences,
            int(concurrency) if concurrency_valid else None,
            int(gen_tokens),
            errors,
        )
        _validate_execution_completed_tokens(
            payload,
            workload,
            int(concurrency) if concurrency_valid else None,
            int(gen_tokens),
            int(warmup_tokens),
            errors,
        )
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
    primitive_schema = primitive.get("schema")
    if (
        not isinstance(primitive_schema, int)
        or isinstance(primitive_schema, bool)
        or primitive_schema != _REQUIRED_PRIMITIVE_CORRECTNESS_SCHEMA
    ):
        errors.append("correctness.primitive_batch_correctness.schema must be 1 for accepted artifacts")
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
    primitive_source_artifact_path = primitive.get("source_artifact_path")
    if not isinstance(primitive_source_artifact_path, str) or not primitive_source_artifact_path:
        errors.append("correctness.primitive_batch_correctness.source_artifact_path must be a non-empty string for accepted artifacts")
    elif isinstance(primitive_artifact_path, str) and primitive_artifact_path and primitive_source_artifact_path != primitive_artifact_path:
        errors.append("correctness.primitive_batch_correctness.source_artifact_path must match artifact_path for accepted artifacts")
    primitive_rows = primitive.get("rows")
    if not isinstance(primitive_rows, int) or isinstance(primitive_rows, bool):
        errors.append("correctness.primitive_batch_correctness.rows must be an int for accepted artifacts")
    if concurrency_valid and isinstance(primitive_rows, int) and not isinstance(primitive_rows, bool) and primitive_rows != concurrency:
        errors.append("correctness.primitive_batch_correctness.rows must match workload.concurrency for accepted artifacts")
    primitive_seed = primitive.get("seed")
    if not isinstance(primitive_seed, int) or isinstance(primitive_seed, bool):
        errors.append("correctness.primitive_batch_correctness.seed must be an int for accepted artifacts")
    elif primitive_seed != _REQUIRED_PRIMITIVE_CORRECTNESS_SEED:
        errors.append("correctness.primitive_batch_correctness.seed must match scripts/qwen35_batch_correctness.py deterministic seed for accepted artifacts")
    primitive_context_lens = primitive.get("context_lens")
    if not isinstance(primitive_context_lens, list):
        errors.append("correctness.primitive_batch_correctness.context_lens must be a list for accepted artifacts")
    elif isinstance(primitive_rows, int) and not isinstance(primitive_rows, bool):
        max_context_len = _REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS["max_context_len"]
        expected_lens = [(idx % max_context_len) + 1 for idx in range(primitive_rows)]
        if len(primitive_context_lens) != primitive_rows:
            errors.append("correctness.primitive_batch_correctness.context_lens length must match rows for accepted artifacts")
        elif any(not isinstance(value, int) or isinstance(value, bool) for value in primitive_context_lens):
            errors.append("correctness.primitive_batch_correctness.context_lens values must be ints for accepted artifacts")
        elif primitive_context_lens != expected_lens:
            errors.append("correctness.primitive_batch_correctness.context_lens must match scripts/qwen35_batch_correctness.py fixture coverage for accepted artifacts")
    for field, expected_value in _REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS.items():
        value = primitive.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"correctness.primitive_batch_correctness.{field} must be an int for accepted artifacts")
        elif value != expected_value:
            errors.append(f"correctness.primitive_batch_correctness.{field} must match scripts/qwen35_batch_correctness.py fixture shape for accepted artifacts")
    for field in (
        "append_key_mismatch",
        "append_value_mismatch",
        "append_batch_aa_key_mismatch",
        "append_batch_aa_value_mismatch",
    ):
        value = primitive.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"correctness.primitive_batch_correctness.{field} must be an int for accepted artifacts")
        elif value != 0:
            errors.append(f"correctness.primitive_batch_correctness.{field} must be 0 for accepted artifacts")
    attn_batch_aa = primitive.get("attn_batch_aa_max_abs")
    if not _is_number(attn_batch_aa):
        errors.append("correctness.primitive_batch_correctness.attn_batch_aa_max_abs must be numeric for accepted artifacts")
    elif float(attn_batch_aa) != 0.0:
        errors.append("correctness.primitive_batch_correctness.attn_batch_aa_max_abs must be 0.0 for accepted artifacts")
    if primitive.get("aa_passed") is not True:
        errors.append("correctness.primitive_batch_correctness.aa_passed must be true for accepted artifacts")
    primitive_device = primitive.get("device")
    _validate_primitive_device_metadata(primitive_device, errors)
    hardware = payload.get("hardware")
    hardware_gpu = hardware.get("gpu") if isinstance(hardware, Mapping) else None
    primitive_device_name = primitive_device.get("device_name") if isinstance(primitive_device, Mapping) else None
    if (
        isinstance(hardware_gpu, str)
        and hardware_gpu
        and isinstance(primitive_device_name, str)
        and primitive_device_name
        and _normalized_gpu_label(hardware_gpu) != _normalized_gpu_label(primitive_device_name)
    ):
        errors.append("correctness.primitive_batch_correctness.device.device_name must match hardware.gpu for accepted artifacts")
    attn_vs_c1 = primitive.get("attn_batch_vs_c1_max_abs")
    if not _is_number(attn_vs_c1):
        errors.append("correctness.primitive_batch_correctness.attn_batch_vs_c1_max_abs must be numeric for accepted artifacts")
    elif float(attn_vs_c1) != 0.0:
        errors.append("correctness.primitive_batch_correctness.attn_batch_vs_c1_max_abs must be 0.0 for accepted artifacts")
    attn_vs_numpy = primitive.get("attn_batch_vs_numpy_max_abs")
    if not _is_number(attn_vs_numpy) or not math.isfinite(float(attn_vs_numpy)):
        errors.append("correctness.primitive_batch_correctness.attn_batch_vs_numpy_max_abs must be finite numeric for accepted artifacts")
    elif float(attn_vs_numpy) < 0.0 or float(attn_vs_numpy) > _PRIMITIVE_CORRECTNESS_NUMPY_MAX_ABS_LIMIT:
        errors.append("correctness.primitive_batch_correctness.attn_batch_vs_numpy_max_abs must be between 0.0 and 2e-5 for accepted artifacts")
    _validate_int8_kv_primitive_layer_accuracy_gates(payload, correctness, errors)


def _validate_generated_token_sequence_lengths(
    label: str,
    sequences: Any,
    expected_equality_tokens: int,
    errors: list[str],
) -> None:
    if not isinstance(sequences, list):
        return
    for index, sequence in enumerate(sequences):
        if not isinstance(sequence, list):
            errors.append(f"{label}[{index}] must be a list for accepted artifacts")
            continue
        if len(sequence) != expected_equality_tokens:
            errors.append(f"{label}[{index}] length must match seed plus workload.warmup_decode_tokens plus workload.gen_tokens_per_request for accepted artifacts")
        if any(not _is_valid_token_id(token) for token in sequence):
            errors.append(f"{label}[{index}] must contain only non-negative token ids for accepted artifacts")


def _validate_execution_seed_tokens(
    payload: Mapping[str, Any],
    batch_sequences: Any,
    concurrency: int | None,
    errors: list[str],
) -> None:
    execution = payload.get("execution")
    seed_tokens = execution.get("seed_tokens") if isinstance(execution, Mapping) else None
    if not isinstance(seed_tokens, Mapping):
        errors.append("execution.seed_tokens must be an object for accepted artifacts")
        return
    if concurrency is not None and len(seed_tokens) != concurrency:
        errors.append("execution.seed_tokens length must match workload.concurrency for accepted artifacts")
    if concurrency is None:
        return
    expected_row_keys = {str(row_index) for row_index in range(concurrency)}
    if set(seed_tokens.keys()) != expected_row_keys:
        errors.append("execution.seed_tokens keys must match workload.concurrency row ids for accepted artifacts")
    for row_index in range(concurrency):
        row_key = str(row_index)
        token_id = _extract_token_id(seed_tokens.get(row_key), f"execution.seed_tokens.{row_key}", errors)
        if token_id is None:
            continue
        if isinstance(batch_sequences, list) and row_index < len(batch_sequences) and isinstance(batch_sequences[row_index], list) and batch_sequences[row_index]:
            if token_id != batch_sequences[row_index][0]:
                errors.append(f"execution.seed_tokens.{row_key} must match correctness.generated_token_equality.batch_sequences first token for accepted artifacts")



def _validate_execution_generated_tokens(
    payload: Mapping[str, Any],
    batch_sequences: Any,
    concurrency: int | None,
    gen_tokens_per_request: int,
    errors: list[str],
) -> None:
    execution = payload.get("execution")
    generated_tokens = execution.get("generated_tokens") if isinstance(execution, Mapping) else None
    if not isinstance(generated_tokens, Mapping):
        errors.append("execution.generated_tokens must be an object for accepted artifacts")
        return
    if concurrency is not None and len(generated_tokens) != concurrency:
        errors.append("execution.generated_tokens length must match workload.concurrency for accepted artifacts")
    if concurrency is None:
        return
    expected_row_keys = {str(row_index) for row_index in range(concurrency)}
    if set(generated_tokens.keys()) != expected_row_keys:
        errors.append("execution.generated_tokens keys must match workload.concurrency row ids for accepted artifacts")
    for row_index in range(concurrency):
        row_key = str(row_index)
        row = generated_tokens.get(row_key)
        token_ids = _extract_generated_token_ids(row, f"execution.generated_tokens.{row_key}", errors)
        if token_ids is None:
            continue
        if len(token_ids) != gen_tokens_per_request:
            errors.append(f"execution.generated_tokens.{row_key} length must match workload.gen_tokens_per_request for accepted artifacts")
        if isinstance(batch_sequences, list) and row_index < len(batch_sequences) and isinstance(batch_sequences[row_index], list):
            expected_suffix = batch_sequences[row_index][-len(token_ids) :] if token_ids else []
            if token_ids != expected_suffix:
                errors.append(f"execution.generated_tokens.{row_key} must match correctness.generated_token_equality.batch_sequences suffix for accepted artifacts")


def _validate_execution_completed_tokens(
    payload: Mapping[str, Any],
    workload: Mapping[str, Any],
    concurrency: int | None,
    gen_tokens_per_request: int,
    warmup_decode_tokens: int,
    errors: list[str],
) -> None:
    execution = payload.get("execution")
    completed = execution.get("completed") if isinstance(execution, Mapping) else None
    if not isinstance(completed, list):
        errors.append("execution.completed must be a list for accepted artifacts")
        return
    if concurrency is not None and len(completed) != concurrency:
        errors.append("execution.completed length must match workload.concurrency for accepted artifacts")
    prompt_lengths = workload.get("prompt_lengths")
    observability = payload.get("observability")
    per_request = observability.get("per_request") if isinstance(observability, Mapping) else None
    completed_by_request: dict[int, list[int]] = {}
    for index, row in enumerate(completed):
        if not isinstance(row, Mapping):
            errors.append(f"execution.completed[{index}] must be an object for accepted artifacts")
            continue
        request_id = row.get("request_id")
        if not isinstance(request_id, int) or isinstance(request_id, bool):
            errors.append(f"execution.completed[{index}].request_id must be an int for accepted artifacts")
            continue
        if concurrency is not None and (request_id < 0 or request_id >= concurrency):
            errors.append(f"execution.completed[{index}].request_id must be in workload.concurrency range for accepted artifacts")
        prompt_token_ids = _extract_generated_token_ids(row.get("prompt_tokens"), f"execution.completed[{index}].prompt_tokens", errors)
        if (
            prompt_token_ids is not None
            and isinstance(prompt_lengths, list)
            and 0 <= request_id < len(prompt_lengths)
            and isinstance(prompt_lengths[request_id], int)
            and not isinstance(prompt_lengths[request_id], bool)
            and len(prompt_token_ids) != prompt_lengths[request_id]
        ):
            errors.append(f"execution.completed[{index}].prompt_tokens length must match workload.prompt_lengths for accepted artifacts")
        token_ids = _extract_generated_token_ids(row.get("generated_tokens"), f"execution.completed[{index}].generated_tokens", errors)
        if token_ids is None:
            continue
        expected_completed_tokens = int(warmup_decode_tokens) + int(gen_tokens_per_request)
        if len(token_ids) != expected_completed_tokens:
            errors.append(f"execution.completed[{index}].generated_tokens length must match workload.warmup_decode_tokens plus workload.gen_tokens_per_request for accepted artifacts")
        if request_id in completed_by_request:
            errors.append("execution.completed request_id values must be unique for accepted artifacts")
        completed_by_request[int(request_id)] = token_ids
        if row.get("finished") is not True:
            errors.append(f"execution.completed[{index}].finished must be true for accepted artifacts")
        finish_reason = row.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason.strip():
            errors.append(f"execution.completed[{index}].finish_reason must be a non-empty string for accepted artifacts")
        elif isinstance(per_request, Mapping):
            observed = per_request.get(str(request_id))
            if isinstance(observed, Mapping):
                observed_finish_reason = observed.get("finish_reason")
                if isinstance(observed_finish_reason, str) and observed_finish_reason != finish_reason:
                    errors.append(f"execution.completed request_id {request_id} finish_reason must match observability.per_request for accepted artifacts")
    if concurrency is None:
        return
    generated_tokens = execution.get("generated_tokens") if isinstance(execution, Mapping) else None
    for request_id in range(concurrency):
        if request_id not in completed_by_request:
            errors.append("execution.completed must include every request_id for accepted artifacts")
            continue
        if isinstance(generated_tokens, Mapping):
            token_ids = _extract_generated_token_ids(
                generated_tokens.get(str(request_id)),
                f"execution.generated_tokens.{request_id}",
                errors,
            )
            if token_ids is not None and completed_by_request[request_id][-len(token_ids):] != token_ids:
                errors.append(f"execution.completed request_id {request_id} generated_tokens suffix must match execution.generated_tokens for accepted artifacts")


def _extract_generated_token_ids(row: Any, label: str, errors: list[str]) -> list[int] | None:
    if not isinstance(row, list):
        errors.append(f"{label} must be a list for accepted artifacts")
        return None
    token_ids: list[int] = []
    for index, item in enumerate(row):
        token_id = _extract_token_id(item, f"{label}[{index}]", errors)
        if token_id is None:
            return None
        token_ids.append(token_id)
    return token_ids


def _is_valid_token_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _extract_token_id(item: Any, label: str, errors: list[str]) -> int | None:
    if _is_valid_token_id(item):
        return item
    if isinstance(item, int) and not isinstance(item, bool):
        errors.append(f"{label} must be a non-negative token id for accepted artifacts")
        return None
    if isinstance(item, Mapping):
        token_id = item.get("token_id")
        if _is_valid_token_id(token_id):
            return token_id
        if isinstance(token_id, int) and not isinstance(token_id, bool):
            errors.append(f"{label}.token_id must be non-negative for accepted artifacts")
            return None
    errors.append(f"{label} must be a token id or object with token_id for accepted artifacts")
    return None


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
    samples_valid = False
    if not isinstance(samples, list) or not samples:
        errors.append("measurements.decode_step_seconds.samples must be a non-empty list for accepted artifacts")
    elif any(not _is_positive_number(sample) for sample in samples):
        errors.append("measurements.decode_step_seconds.samples must contain only positive numbers for accepted artifacts")
    else:
        samples_valid = True
    for field in ("median", "p95", "min", "max"):
        if not _is_positive_number(decode_steps.get(field)):
            errors.append(f"measurements.decode_step_seconds.{field} must be positive numeric for accepted artifacts")
    if not _is_nonnegative_number(decode_steps.get("stdev")):
        errors.append("measurements.decode_step_seconds.stdev must be non-negative numeric for accepted artifacts")
    if samples_valid:
        sample_values = [float(sample) for sample in samples]
        ordered_samples = sorted(sample_values)
        p95_index = min(len(ordered_samples) - 1, math.ceil(0.95 * len(ordered_samples)) - 1)
        expected_values = {
            "median": float(statistics.median(sample_values)),
            "p95": ordered_samples[p95_index],
            "min": ordered_samples[0],
            "max": ordered_samples[-1],
            "stdev": statistics.stdev(sample_values) if len(sample_values) > 1 else 0.0,
        }
        for field, expected in expected_values.items():
            value = decode_steps.get(field)
            if _is_nonnegative_number(value) and not _numbers_close(float(value), float(expected)):
                errors.append(f"measurements.decode_step_seconds.{field} must match decode_step_seconds.samples for accepted artifacts")


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
        baseline_reference_artifact_path = baseline.get("reference_artifact_path")
        if not isinstance(baseline_reference_artifact_path, str) or not baseline_reference_artifact_path:
            errors.append(f"scaling.{baseline_name}.reference_artifact_path must be a non-empty string for accepted artifacts")
        elif isinstance(baseline_artifact_path, str) and baseline_artifact_path and baseline_reference_artifact_path != baseline_artifact_path:
            errors.append(f"scaling.{baseline_name}.reference_artifact_path must match artifact_path for accepted artifacts")
        status = baseline.get("status")
        if not isinstance(status, str) or not status:
            errors.append(f"scaling.{baseline_name}.status must be a non-empty string for accepted artifacts")
        elif status in _UNUSABLE_ACCEPTED_SCALING_BASELINE_STATUSES:
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
        for field in ("aggregate_vs_c1", "aggregate_vs_serial_bridge"):
            value = ratios.get(field)
            if _is_number(value) and float(value) <= 1.0:
                errors.append(f"scaling.ratios.{field} must be > 1.0 for accepted artifacts")


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


def _bucket_key_axis(bucket_key: str, axis: str) -> str | None:
    prefix = f"{axis}="
    for segment in bucket_key.split(":"):
        if segment.startswith(prefix):
            value = segment[len(prefix) :]
            return value if value else None
    return None


def _valid_request_observability(
    row: Any,
    errors: list[str],
    *,
    expected_active_c: int | None = None,
    expected_mode: str | None = None,
    expected_context_bucket: int | None = None,
    expected_context_buckets: set[int] | None = None,
    expected_active_mask: str | None = None,
    expected_kv_storage_dtype: str | None = None,
    expected_layer_plan: str | None = None,
    expected_top_k: int | None = None,
    expected_experts_per_token: int | None = None,
    expected_replay_steps: int | None = None,
    expected_draft_depth: int | None = None,
) -> bool:
    if not isinstance(row, Mapping):
        errors.append("observability.per_request entries must be objects for accepted artifacts")
        return False
    ok = True
    for field in _REQUIRED_ACCEPTED_PER_REQUEST_OBSERVABILITY_FIELDS:
        if field not in row:
            errors.append(f"observability.per_request.*.{field} is required for accepted artifacts")
            ok = False
    for field in ("queue_seconds", "prefill_seconds", "decode_seconds"):
        if field in row and not _is_nonnegative_number(row.get(field)):
            errors.append(f"observability.per_request.*.{field} must be finite non-negative numeric for accepted artifacts")
            ok = False
    for field in ("kv_pages_owned", "kv_pages_peak"):
        if field in row and (not isinstance(row.get(field), int) or isinstance(row.get(field), bool) or row.get(field) < 0):
            errors.append(f"observability.per_request.*.{field} must be a non-negative int for accepted artifacts")
            ok = False
    kv_pages_owned = row.get("kv_pages_owned")
    kv_pages_peak = row.get("kv_pages_peak")
    if (
        isinstance(kv_pages_owned, int)
        and not isinstance(kv_pages_owned, bool)
        and kv_pages_owned >= 0
        and isinstance(kv_pages_peak, int)
        and not isinstance(kv_pages_peak, bool)
        and kv_pages_peak >= 0
        and kv_pages_peak < kv_pages_owned
    ):
        errors.append("observability.per_request.*.kv_pages_peak must be >= kv_pages_owned for accepted artifacts")
        ok = False
    if "bucket_key" in row and row.get("bucket_key") is not None:
        bucket_key = row.get("bucket_key")
        if not isinstance(bucket_key, str) or not bucket_key.strip():
            errors.append("observability.per_request.*.bucket_key must be a non-empty string or null for accepted artifacts")
            ok = False
        else:
            mode_axis = bucket_key.split(":", 1)[0]
            c_axis = _bucket_key_axis(bucket_key, "c")
            ctx_axis = _bucket_key_axis(bucket_key, "ctx")
            mask_axis = _bucket_key_axis(bucket_key, "mask")
            kv_axis = _bucket_key_axis(bucket_key, "kv")
            layer_axis = _bucket_key_axis(bucket_key, "layers")
            top_k_axis = _bucket_key_axis(bucket_key, "top_k")
            experts_axis = _bucket_key_axis(bucket_key, "experts")
            replay_axis = _bucket_key_axis(bucket_key, "replay")
            draft_axis = _bucket_key_axis(bucket_key, "draft")
            if expected_mode is not None and mode_axis != expected_mode:
                errors.append("observability.per_request.*.bucket_key mode must match scheduler decode_shape_key.mode for accepted artifacts")
                ok = False
            if c_axis is None or ctx_axis is None or mask_axis is None:
                errors.append("observability.per_request.*.bucket_key must include c, context, and active-mask axes for accepted artifacts")
                ok = False
            else:
                if expected_active_c is not None and c_axis != str(expected_active_c):
                    errors.append("observability.per_request.*.bucket_key c axis must match workload.concurrency for accepted artifacts")
                    ok = False
                if expected_context_buckets:
                    expected_context_axes = {str(context_bucket) for context_bucket in expected_context_buckets}
                    if ctx_axis not in expected_context_axes:
                        errors.append("observability.per_request.*.bucket_key context axis must match scheduler decode_shape_key.context_bucket or an observed scheduler decode context bucket for accepted artifacts")
                        ok = False
                elif expected_context_bucket is not None and ctx_axis != str(expected_context_bucket):
                    errors.append("observability.per_request.*.bucket_key context axis must match scheduler decode_shape_key.context_bucket for accepted artifacts")
                    ok = False
                if expected_active_mask is not None and mask_axis != expected_active_mask:
                    errors.append("observability.per_request.*.bucket_key active-mask axis must match scheduler decode_shape_key.active_mask for accepted artifacts")
                    ok = False
            if kv_axis is None or layer_axis is None:
                errors.append("observability.per_request.*.bucket_key must include kv and layer-plan axes for accepted artifacts")
                ok = False
            else:
                if expected_kv_storage_dtype is not None and kv_axis != expected_kv_storage_dtype:
                    errors.append("observability.per_request.*.bucket_key kv axis must match workload.kv_storage_dtype for accepted artifacts")
                    ok = False
                if expected_layer_plan is not None and layer_axis != expected_layer_plan:
                    errors.append("observability.per_request.*.bucket_key layer-plan axis must match workload.max_layers for accepted artifacts")
                    ok = False
            if top_k_axis is None or experts_axis is None or replay_axis is None or draft_axis is None:
                errors.append("observability.per_request.*.bucket_key must include top-k, experts, replay, and draft axes for accepted artifacts")
                ok = False
            else:
                if expected_top_k is not None and top_k_axis != str(expected_top_k):
                    errors.append("observability.per_request.*.bucket_key top-k axis must match scheduler decode_shape_key.top_k for accepted artifacts")
                    ok = False
                if expected_experts_per_token is not None and experts_axis != str(expected_experts_per_token):
                    errors.append("observability.per_request.*.bucket_key experts axis must match scheduler decode_shape_key.experts_per_token for accepted artifacts")
                    ok = False
                if expected_replay_steps is not None and replay_axis != str(expected_replay_steps):
                    errors.append("observability.per_request.*.bucket_key replay axis must match scheduler decode_shape_key.replay_steps for accepted artifacts")
                    ok = False
                if expected_draft_depth is not None and draft_axis != str(expected_draft_depth):
                    errors.append("observability.per_request.*.bucket_key draft axis must match scheduler decode_shape_key.draft_depth for accepted artifacts")
                    ok = False
    if "admission_blocked_reason" in row and row.get("admission_blocked_reason") is not None and (not isinstance(row.get("admission_blocked_reason"), str) or not row.get("admission_blocked_reason").strip()):
        errors.append("observability.per_request.*.admission_blocked_reason must be a non-empty string or null for accepted artifacts")
        ok = False
    if "finish_reason" in row and (not isinstance(row.get("finish_reason"), str) or not row.get("finish_reason").strip()):
        errors.append("observability.per_request.*.finish_reason must be a non-empty string for accepted artifacts")
        ok = False
    return ok


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _numbers_close(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= max(1e-9, abs(expected) * 1e-6)


def _is_positive_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value)) and float(value) > 0.0


def _is_nonnegative_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value)) and float(value) >= 0.0


def _is_nonempty_string_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) and bool(item) for item in value)


def _is_stripped_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _is_stripped_nonempty_string_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_is_stripped_nonempty_string(item) for item in value)


def _load_json_value(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)


def _load_payload(path: Path) -> Mapping[str, Any]:
    payload = _load_json_value(path)
    if not isinstance(payload, Mapping):
        raise ValueError("artifact root must be an object")
    return payload


def _validation_summary(
    *,
    artifact_json: Path,
    mode: str,
    passed: bool,
    payload: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload = payload or {}
    artifact_json_text = _benchmark_results_relative_path(str(artifact_json))
    artifact_path = payload.get("artifact_path")
    status = payload.get("status") if passed else None
    performance_claim = payload.get("performance_claim") if passed else None
    benchmark_rollup = payload.get("benchmark_rollup") if passed else None
    return {
        "schema": 1,
        "summary_type": RETAINED_ARTIFACT_VALIDATION_SUMMARY_TYPE,
        "mode": mode,
        "passed": passed,
        "artifact_json": artifact_json_text,
        "source_artifact_path": artifact_json_text,
        "artifact_path": artifact_path if isinstance(artifact_path, str) else None,
        "status": status if isinstance(status, str) else None,
        "performance_claim": performance_claim if isinstance(performance_claim, bool) else None,
        "benchmark_rollup": benchmark_rollup if mode == "rollup_evidence" and isinstance(benchmark_rollup, Mapping) else None,
        "error": error,
    }


def validate_cn_diagnostic_validation_summary(summary: Mapping[str, Any]) -> None:
    errors: list[str] = []
    allowed_keys = {
        "schema",
        "summary_type",
        "mode",
        "passed",
        "artifact_json",
        "source_artifact_path",
        "artifact_path",
        "status",
        "performance_claim",
        "benchmark_rollup",
        "error",
    }
    unexpected_keys = sorted(str(key) for key in summary.keys() - allowed_keys)
    if unexpected_keys:
        errors.append("summary contains unexpected keys: " + ", ".join(unexpected_keys))
    if summary.get("schema") != 1 or isinstance(summary.get("schema"), bool):
        errors.append("summary.schema must be 1")
    if summary.get("summary_type") != RETAINED_ARTIFACT_VALIDATION_SUMMARY_TYPE:
        errors.append(f"summary.summary_type must be {RETAINED_ARTIFACT_VALIDATION_SUMMARY_TYPE}")
    mode = summary.get("mode")
    if mode not in {"artifact_schema", "rollup_evidence"}:
        errors.append("summary.mode must be artifact_schema or rollup_evidence")
    passed = summary.get("passed")
    if not isinstance(passed, bool):
        errors.append("summary.passed must be a bool")
    artifact_json = summary.get("artifact_json")
    if not isinstance(artifact_json, str) or not artifact_json:
        errors.append("summary.artifact_json must be a non-empty string")
    elif artifact_json.strip() != artifact_json:
        errors.append("summary.artifact_json must not contain leading or trailing whitespace")
    source_artifact_path = summary.get("source_artifact_path")
    if not isinstance(source_artifact_path, str) or not source_artifact_path:
        errors.append("summary.source_artifact_path must be a non-empty string")
    else:
        _validate_benchmark_results_artifact_path("summary.source_artifact_path", source_artifact_path, errors)
        if source_artifact_path.strip() != source_artifact_path:
            errors.append("summary.source_artifact_path must not contain leading or trailing whitespace")
        if _path_text_contains_parent_traversal(source_artifact_path):
            errors.append("summary.source_artifact_path must not contain parent traversal")
        if _benchmark_results_relative_path(source_artifact_path) != source_artifact_path.replace("\\", "/"):
            errors.append("summary.source_artifact_path must be a repo-relative benchmarks/results path")
        if not source_artifact_path.replace("\\", "/").rsplit("/", 1)[-1].endswith(".json"):
            errors.append("summary.source_artifact_path must end with .json")
    artifact_path = summary.get("artifact_path")
    if artifact_path is not None:
        if not isinstance(artifact_path, str) or not artifact_path:
            errors.append("summary.artifact_path must be a non-empty string or null")
        else:
            _validate_benchmark_results_artifact_path("summary.artifact_path", artifact_path, errors)
            if artifact_path.strip() != artifact_path:
                errors.append("summary.artifact_path must not contain leading or trailing whitespace")
            if _path_text_contains_parent_traversal(artifact_path):
                errors.append("summary.artifact_path must not contain parent traversal")
            if _benchmark_results_relative_path(artifact_path) != artifact_path.replace("\\", "/"):
                errors.append("summary.artifact_path must be a repo-relative benchmarks/results path or null")
            if not artifact_path.replace("\\", "/").rsplit("/", 1)[-1].endswith(".json"):
                errors.append("summary.artifact_path must end with .json when non-null")
    status = summary.get("status")
    if status is not None:
        if not isinstance(status, str) or not status.strip():
            errors.append("summary.status must be a non-empty string or null")
        elif status.strip() != status:
            errors.append("summary.status must not contain leading or trailing whitespace")
    performance_claim = summary.get("performance_claim")
    if performance_claim is not None and not isinstance(performance_claim, bool):
        errors.append("summary.performance_claim must be a bool or null")
    benchmark_rollup = summary.get("benchmark_rollup")
    if passed is False:
        if status is not None:
            errors.append("failed validation summary.status must be null")
        if performance_claim is not None:
            errors.append("failed validation summary.performance_claim must be null")
        if benchmark_rollup is not None:
            errors.append("failed validation summary.benchmark_rollup must be null")
    if passed is True:
        if status is None:
            errors.append("passed validation summary.status must be a non-empty string")
        if performance_claim is None:
            errors.append("passed validation summary.performance_claim must be a bool")
        if status == "accepted" and performance_claim is not True:
            errors.append("passed validation summary.status accepted requires performance_claim true")
        if performance_claim is True and status != "accepted":
            errors.append("passed validation summary.performance_claim true requires status accepted")
    if (passed is True or benchmark_rollup is not None) and not isinstance(artifact_path, str):
        errors.append("summary.artifact_path must be a non-empty string when summary.passed is true or summary.benchmark_rollup is present")
    if benchmark_rollup is not None:
        if mode != "rollup_evidence":
            errors.append("summary.benchmark_rollup requires summary.mode rollup_evidence")
        if not isinstance(benchmark_rollup, Mapping):
            errors.append("summary.benchmark_rollup must be an object or null")
        else:
            expected_rollup_keys = {"artifact_path", "source_artifact_path", "readme_path", "changelog_path"}
            unexpected_rollup_keys = sorted(str(key) for key in benchmark_rollup.keys() - expected_rollup_keys)
            if unexpected_rollup_keys:
                errors.append("summary.benchmark_rollup contains unexpected keys: " + ", ".join(unexpected_rollup_keys))
        if isinstance(benchmark_rollup, Mapping):
            for key in ("artifact_path", "source_artifact_path"):
                value = benchmark_rollup.get(key)
                if not isinstance(value, str) or not value:
                    errors.append(f"summary.benchmark_rollup.{key} must be a non-empty string")
                else:
                    if value.strip() != value:
                        errors.append(f"summary.benchmark_rollup.{key} must not contain leading or trailing whitespace")
                    if not _is_benchmark_results_path(value):
                        errors.append(f"summary.benchmark_rollup.{key} must be under benchmarks/results")
                    if _path_text_contains_parent_traversal(value):
                        errors.append(f"summary.benchmark_rollup.{key} must not contain parent traversal")
                    if _benchmark_results_relative_path(value) != value.replace("\\", "/"):
                        errors.append(f"summary.benchmark_rollup.{key} must be a repo-relative benchmarks/results path")
                    if not value.replace("\\", "/").rsplit("/", 1)[-1].endswith(".json"):
                        errors.append(f"summary.benchmark_rollup.{key} must end with .json")
            expected_doc_paths = {"readme_path": "benchmarks/README.md", "changelog_path": "benchmarks/CHANGELOG.md"}
            for key, expected_path in expected_doc_paths.items():
                value = benchmark_rollup.get(key)
                if not isinstance(value, str) or not value:
                    errors.append(f"summary.benchmark_rollup.{key} must be a non-empty string")
                else:
                    if value.strip() != value:
                        errors.append(f"summary.benchmark_rollup.{key} must not contain leading or trailing whitespace")
                    if value != expected_path:
                        errors.append(f"summary.benchmark_rollup.{key} must be {expected_path}")
        if isinstance(benchmark_rollup, Mapping) and isinstance(artifact_path, str):
            if benchmark_rollup.get("artifact_path") != artifact_path:
                errors.append("summary.benchmark_rollup.artifact_path must match summary.artifact_path")
            if benchmark_rollup.get("source_artifact_path") != artifact_path:
                errors.append("summary.benchmark_rollup.source_artifact_path must match summary.artifact_path")
    if mode == "rollup_evidence" and passed is True:
        if status != "accepted":
            errors.append("passed rollup evidence summary.status must be accepted")
        if performance_claim is not True:
            errors.append("passed rollup evidence summary.performance_claim must be true")
        if not isinstance(benchmark_rollup, Mapping):
            errors.append("passed rollup evidence summary.benchmark_rollup must be an object")
    if isinstance(artifact_json, str) and isinstance(source_artifact_path, str):
        if artifact_json.replace("\\", "/") != source_artifact_path.replace("\\", "/"):
            errors.append("summary.source_artifact_path must match summary.artifact_json")
    if mode in {"artifact_schema", "rollup_evidence"} and isinstance(artifact_json, str):
        if not _is_benchmark_results_path(artifact_json):
            errors.append("summary.artifact_json must be under benchmarks/results for validation summaries")
        elif _benchmark_results_relative_path(artifact_json) != artifact_json.replace("\\", "/"):
            errors.append("summary.artifact_json must be a repo-relative benchmarks/results path for validation summaries")
        if _path_text_contains_parent_traversal(artifact_json):
            errors.append("summary.artifact_json must not contain parent traversal for validation summaries")
        if not artifact_json.replace("\\", "/").rsplit("/", 1)[-1].endswith(".json"):
            errors.append("summary.artifact_json must end with .json for validation summaries")
    if (passed is True or isinstance(benchmark_rollup, Mapping)) and isinstance(artifact_json, str) and isinstance(artifact_path, str):
        normalized_artifact_json = artifact_json.replace("\\", "/")
        normalized_artifact_path = artifact_path.replace("\\", "/")
        if normalized_artifact_json != normalized_artifact_path and not normalized_artifact_json.endswith("/" + normalized_artifact_path):
            errors.append("summary.artifact_json must point to summary.artifact_path when summary.passed is true or summary.benchmark_rollup is present")
        if _benchmark_results_relative_path(normalized_artifact_json) != normalized_artifact_path:
            errors.append("summary.artifact_json benchmarks/results-relative path must match summary.artifact_path")
    error = summary.get("error")
    if passed is True and error is not None:
        errors.append("summary.error must be null when summary.passed is true")
    if passed is False:
        if not isinstance(error, str) or not error.strip():
            errors.append("summary.error must be a non-empty string when summary.passed is false")
        elif error.strip() != error:
            errors.append("summary.error must not contain leading or trailing whitespace")
    if errors:
        raise ValueError("invalid c>N validation summary: " + "; ".join(errors))


def _validation_summary_json(summary: Mapping[str, Any]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)


def _write_validation_summary(path: Path, summary: Mapping[str, Any]) -> None:
    validate_cn_diagnostic_validation_summary(summary)
    _validate_validation_summary_output_path(path, summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_validation_summary_json(summary) + "\n")


def _summary_json_path_context(path: Path) -> tuple[Path, Path]:
    results_root = (Path.cwd() / "benchmarks" / "results").resolve()
    candidate = path if path.is_absolute() else Path.cwd() / path
    return results_root, candidate


def _summary_json_path_is_in_current_results(path: Path, *, results_root: Path | None = None, candidate: Path | None = None) -> bool:
    if results_root is None or candidate is None:
        results_root, candidate = _summary_json_path_context(path)
    try:
        return candidate.resolve().is_relative_to(results_root)
    except OSError:
        return False


def _summary_json_path_parent_is_in_results(current: Path, results_root: Path) -> bool:
    try:
        return current.resolve().is_relative_to(results_root)
    except OSError:
        return False


def _summary_json_path_has_symlink_parent(path: Path, *, results_root: Path | None = None, candidate: Path | None = None) -> bool:
    if results_root is None or candidate is None:
        results_root, candidate = _summary_json_path_context(path)
    current = candidate.parent
    while _summary_json_path_parent_is_in_results(current, results_root):
        if current.is_symlink():
            return True
        if current.resolve() == results_root or current == current.parent:
            return False
        current = current.parent
    return False


def _summary_json_path_has_non_directory_parent(path: Path, *, results_root: Path | None = None, candidate: Path | None = None) -> bool:
    if results_root is None or candidate is None:
        results_root, candidate = _summary_json_path_context(path)
    current = candidate.parent
    while _summary_json_path_parent_is_in_results(current, results_root):
        if current.exists() and not current.is_dir():
            return True
        if current.resolve() == results_root or current == current.parent:
            return False
        current = current.parent
    return False


def _validate_summary_json_path(path: Path, *, label: str = "--summary-json path", must_exist: bool = False) -> None:
    results_root, candidate = _summary_json_path_context(path)
    if not _summary_json_path_is_in_current_results(path, results_root=results_root, candidate=candidate):
        raise ValueError(f"{label} must be under the current repo benchmarks/results for retained validation evidence")
    if ".." in path.parts:
        raise ValueError(f"{label} must not contain parent traversal for retained validation evidence")
    if path.suffix != ".json":
        raise ValueError(f"{label} must end with .json for retained validation evidence")
    if path.is_symlink():
        raise ValueError(f"{label} must be a regular .json file, not a symlink, for retained validation evidence")
    if _summary_json_path_has_symlink_parent(path, results_root=results_root, candidate=candidate):
        raise ValueError(f"{label} parent directories must not be symlinks for retained validation evidence")
    if _summary_json_path_has_non_directory_parent(path, results_root=results_root, candidate=candidate):
        raise ValueError(f"{label} parent directories must be directories for retained validation evidence")
    if path.exists() and path.is_dir():
        raise ValueError(f"{label} must be a .json file, not a directory, for retained validation evidence")
    if path.exists() and not path.is_file():
        raise ValueError(f"{label} must be a regular .json file for retained validation evidence")
    if must_exist and not path.exists():
        raise ValueError(f"{label} must exist as a .json file for retained validation evidence")


def _validate_validation_summary_output_path(path: Path, summary: Mapping[str, Any], *, label: str = "--summary-json path") -> None:
    _validate_summary_json_path(path, label=label)
    mode = summary.get("mode")
    if mode not in {"artifact_schema", "rollup_evidence"}:
        return
    source_artifact_path = summary.get("source_artifact_path")
    stem_source = source_artifact_path if isinstance(source_artifact_path, str) and source_artifact_path else summary.get("artifact_json")
    if not isinstance(stem_source, str) or not stem_source:
        return
    normalized_source = stem_source.replace("\\", "/")
    source_dir, source_name = normalized_source.rsplit("/", 1) if "/" in normalized_source else ("", normalized_source)
    artifact_stem = source_name.removesuffix(".json")
    suffix = "rollup-check" if mode == "rollup_evidence" else "schema-check"
    expected_name = f"{artifact_stem}-{suffix}.json"
    expected_path = f"{source_dir}/{expected_name}" if source_dir else expected_name
    actual_path = _benchmark_results_relative_path(str(path)).replace("\\", "/")
    if actual_path != expected_path:
        raise ValueError(f"{label} must be {expected_path} for {mode}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Qwen3.5 c>N diagnostic/retained benchmark artifacts")
    parser.add_argument("artifact_json", type=Path, help="Artifact JSON to validate")
    parser.add_argument(
        "--rollup-evidence",
        action="store_true",
        help="Also require benchmark_rollup metadata and live README/CHANGELOG links for promotion",
    )
    parser.add_argument(
        "--validation-summary",
        action="store_true",
        help="Treat artifact_json as a retained validation summary artifact and validate its schema",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional JSON file recording pass/fail status for automation evidence",
    )
    args = parser.parse_args(argv)
    if args.validation_summary and args.rollup_evidence:
        parser.error("--validation-summary cannot be combined with --rollup-evidence")
    if args.validation_summary and args.summary_json is not None:
        parser.error("--validation-summary cannot be combined with --summary-json")

    if args.validation_summary:
        try:
            _validate_summary_json_path(args.artifact_json, label="--validation-summary path", must_exist=True)
            summary = _load_payload(args.artifact_json)
            validate_cn_diagnostic_validation_summary(summary)
            _validate_validation_summary_output_path(args.artifact_json, summary, label="--validation-summary path")
        except Exception as exc:
            print(f"invalid c>N diagnostic artifact: {exc}", file=sys.stderr)
            return 1
        print("OK")
        return 0

    mode = "rollup_evidence" if args.rollup_evidence else "artifact_schema"
    payload: Mapping[str, Any] | None = None
    if args.summary_json is not None:
        try:
            _validate_summary_json_path(args.summary_json)
        except ValueError as exc:
            print(f"invalid c>N diagnostic artifact: {exc}", file=sys.stderr)
            return 1
    try:
        payload = _load_payload(args.artifact_json)
        if args.rollup_evidence:
            validate_cn_diagnostic_rollup_evidence(payload)
        else:
            validate_cn_diagnostic_artifact_payload(payload)
    except Exception as exc:
        summary = _validation_summary(
            artifact_json=args.artifact_json,
            mode=mode,
            passed=False,
            payload=payload,
            error=str(exc),
        )
        if args.summary_json is not None:
            try:
                _write_validation_summary(args.summary_json, summary)
            except ValueError as write_exc:
                print(f"invalid c>N diagnostic artifact: {write_exc}", file=sys.stderr)
                return 1
        print(f"invalid c>N diagnostic artifact: {exc}", file=sys.stderr)
        return 1
    summary = _validation_summary(artifact_json=args.artifact_json, mode=mode, passed=True, payload=payload)
    try:
        validate_cn_diagnostic_validation_summary(summary)
    except ValueError as exc:
        print(f"invalid c>N diagnostic artifact: {exc}", file=sys.stderr)
        return 1
    if args.summary_json is not None:
        try:
            _write_validation_summary(args.summary_json, summary)
        except ValueError as exc:
            print(f"invalid c>N diagnostic artifact: {exc}", file=sys.stderr)
            return 1
    print("OK")
    return 0


__all__ = [
    "DISALLOWED_ACCEPTED_DIAGNOSTIC_COMMAND_FRAGMENTS",
    "DISALLOWED_ACCEPTED_DIAGNOSTIC_EVIDENCE_FRAGMENTS",
    "DISALLOWED_ACCEPTED_DIAGNOSTIC_TRACE_FIELD_FRAGMENTS",
    "DISALLOWED_ACCEPTED_DIAGNOSTIC_TRACE_FIELD_NAMES",
    "main",
    "validate_cn_diagnostic_artifact_payload",
    "validate_cn_diagnostic_rollup_evidence",
    "validate_cn_diagnostic_validation_summary",
]


if __name__ == "__main__":
    raise SystemExit(main())
