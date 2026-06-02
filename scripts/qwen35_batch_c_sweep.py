#!/usr/bin/env python3
"""Run or plan the Qwen/PARO c=1/2/4/8 concurrency diagnostic sweep.

The sweep is intentionally an orchestration wrapper: it records exactly which
primitive, scheduler-serial, and native diagnostic commands would run (or did
run), where each artifact is written, and whether the repository was dirty at
launch.  Use ``--dry-run`` for CI/unit tests and command review without touching
ROCm.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.qwen35_batch_artifact_schema import _load_json_value
from scripts.qwen35_batch_constants import (
    PROFILER_DISALLOWED_DIAGNOSTIC_KERNEL_NAME_FRAGMENTS,
    RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_NUMPY_MAX_ABS_LIMIT,
    RETAINED_ARTIFACT_PROFILER_TRACE_DURATION_COLUMNS,
    RETAINED_ARTIFACT_PROFILER_TRACE_END_COLUMNS,
    RETAINED_ARTIFACT_PROFILER_TRACE_KERNEL_NAME_COLUMNS,
    RETAINED_ARTIFACT_PROFILER_TRACE_START_COLUMNS,
    RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_SCRIPT,
    RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_UNIQUE_FLAGS,
    RETAINED_ARTIFACT_PROFILER_TRACE_SYNTHESIZED_FIELDS,
    RETAINED_ARTIFACT_ROCPROF_COMMAND_FLAGS,
    RETAINED_ARTIFACT_ROCPROF_EXECUTABLE,
    RETAINED_ARTIFACT_ROCPROF_OUTPUT_FORMAT,
    RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SCHEMA,
    RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SEED,
    RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS,
    RETAINED_ARTIFACT_REQUIRED_PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES,
    RETAINED_ARTIFACT_SERIAL_BRIDGE_SCRIPT,
    RETAINED_ARTIFACT_REQUIRED_PROFILER_KERNEL_DURATION_CATEGORIES,
    RETAINED_ARTIFACT_RETAINED_BENCH_UNIQUE_FLAGS,
    RETAINED_ARTIFACT_RETAINED_CONDITION_STATUS_LABELS,
    RETAINED_ARTIFACT_RETAINED_GATE_FLAGS,
    RETAINED_ARTIFACT_GGUF_DIAGNOSTIC_SCRIPT,
    RETAINED_ARTIFACT_INT8_DIAGNOSTIC_SCRIPT,
    RETAINED_ARTIFACT_LEGACY_NATIVE_BENCH_SCRIPT,
    RETAINED_ARTIFACT_RETAINED_BENCH_SCRIPT,
    RETAINED_ARTIFACT_RETAINED_GATE_LABELS,
    RETAINED_ARTIFACT_RETAINED_POSTCONDITION_KINDS,
    RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_DISALLOWED_FLAGS,
    RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_UNIQUE_FLAGS,
    RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_VALUE_FLAGS,
    RETAINED_ARTIFACT_RETAINED_PRECONDITION_KINDS,
    RETAINED_ARTIFACT_SWEEP_COMMAND_CATEGORIES,
    RETAINED_ARTIFACT_SWEEP_COMMAND_STATUS_LABELS,
    RETAINED_ARTIFACT_UNUSABLE_SCALING_BASELINE_STATUSES,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/"
    "snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"
)
DEFAULT_FIXTURE = "fixtures/qwen35_paro/parent_512_32_seed1234.json"
DEFAULT_BATCH_SIZES = (1, 2, 4, 8)
_OUTPUT_TAIL_MAX_CHARS = 4000
_DISALLOWED_PROFILER_KERNEL_NAME_FRAGMENTS = PROFILER_DISALLOWED_DIAGNOSTIC_KERNEL_NAME_FRAGMENTS
_UNUSABLE_SCALING_REFERENCE_STATUSES = RETAINED_ARTIFACT_UNUSABLE_SCALING_BASELINE_STATUSES
_PROFILER_KERNEL_DURATION_CATEGORIES = RETAINED_ARTIFACT_REQUIRED_PROFILER_KERNEL_DURATION_CATEGORIES
_PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES = RETAINED_ARTIFACT_REQUIRED_PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES
_PROFILER_TRACE_KERNEL_NAME_COLUMNS = RETAINED_ARTIFACT_PROFILER_TRACE_KERNEL_NAME_COLUMNS
_PROFILER_TRACE_START_COLUMNS = RETAINED_ARTIFACT_PROFILER_TRACE_START_COLUMNS
_PROFILER_TRACE_END_COLUMNS = RETAINED_ARTIFACT_PROFILER_TRACE_END_COLUMNS
_PROFILER_TRACE_DURATION_COLUMNS = RETAINED_ARTIFACT_PROFILER_TRACE_DURATION_COLUMNS
_ROCPROF_COMMAND_FLAGS = RETAINED_ARTIFACT_ROCPROF_COMMAND_FLAGS
_ROCPROF_EXECUTABLE = RETAINED_ARTIFACT_ROCPROF_EXECUTABLE
_ROCPROF_OUTPUT_FORMAT = RETAINED_ARTIFACT_ROCPROF_OUTPUT_FORMAT
_PRIMITIVE_CORRECTNESS_SCRIPT = RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_SCRIPT
_REQUIRED_PRIMITIVE_CORRECTNESS_SCHEMA = RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SCHEMA
_REQUIRED_PRIMITIVE_CORRECTNESS_SEED = RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SEED
_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS = RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS
_PRIMITIVE_CORRECTNESS_NUMPY_MAX_ABS_LIMIT = RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_NUMPY_MAX_ABS_LIMIT


def _load_json_path(path: Path) -> Any:
    return _load_json_value(path)


def _is_python_executable(executable: str) -> bool:
    name = Path(executable).name
    if name == "python":
        return True
    if not name.startswith("python"):
        return False
    suffix = name[len("python") :]
    version_suffix = suffix.rstrip("dmt")
    return bool(version_suffix) and all(part.isdecimal() for part in version_suffix.split("."))


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


_COMMAND_ENV_KEYS = ("HIP_VISIBLE_DEVICES",)


def _command_env_prefix_parts() -> tuple[str, ...]:
    assignments = tuple(
        f"{key}={value}"
        for key in _COMMAND_ENV_KEYS
        if (value := os.environ.get(key)) is not None and value.strip()
    )
    return ("env", *assignments) if assignments else ()


def _python_command_prefix() -> tuple[str, ...]:
    return (*_command_env_prefix_parts(), sys.executable)


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


def _profiled_command_argv(command: str) -> list[str] | None:
    try:
        argv = shlex.split(command)
    except ValueError:
        return None
    if "--" not in argv:
        return None
    return argv[argv.index("--") + 1 :]


def _required_primitive_context_lens(rows: int) -> list[int]:
    max_context_len = _REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS["max_context_len"]
    return [(idx % max_context_len) + 1 for idx in range(rows)]


def _primitive_context_lens_matches(value: Any, rows: int) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        and value == _required_primitive_context_lens(rows)
    )


def _primitive_device_metadata_blockers(device: Any) -> list[str]:
    if type(device) is not dict:
        return ["device metadata is missing or not a plain object"]
    blockers: list[str] = []
    expected_device_keys = {
        "env",
        "hipGetDeviceCount_error",
        "visible_device_count",
        "hipGetDevice_error",
        "current_device",
        "hipDeviceGetName_error",
        "device_name",
    }
    if set(device) - expected_device_keys:
        blockers.append("device metadata contains unknown keys")
    env = device.get("env")
    if type(env) is not dict:
        blockers.append("device.env is missing or not a plain object")
    else:
        known_env_keys = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
        if set(env) - set(known_env_keys):
            blockers.append("device.env contains unknown keys")
        for key in known_env_keys:
            value = env.get(key)
            if value is not None and (not isinstance(value, str) or not value):
                blockers.append(f"device.env.{key} is not a non-empty string when present")
            elif isinstance(value, str) and not value.strip():
                blockers.append(f"device.env.{key} is not a non-blank string when present")
    for field in ("hipGetDeviceCount_error", "hipGetDevice_error", "hipDeviceGetName_error"):
        value = device.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value != 0:
            blockers.append(f"device.{field} is missing or not integer zero")
    visible_count = device.get("visible_device_count")
    if not isinstance(visible_count, int) or isinstance(visible_count, bool) or visible_count <= 0:
        blockers.append("device.visible_device_count is missing or not a positive int")
    current_device = device.get("current_device")
    if not isinstance(current_device, int) or isinstance(current_device, bool) or current_device < 0:
        blockers.append("device.current_device is missing or not a non-negative int")
    device_name = device.get("device_name")
    if not isinstance(device_name, str) or not device_name:
        blockers.append("device.device_name is missing or not a non-empty string")
    elif not device_name.strip():
        blockers.append("device.device_name is not a non-blank string")
    return blockers


_PROFILER_SYNTHESIZED_FIELDS = RETAINED_ARTIFACT_PROFILER_TRACE_SYNTHESIZED_FIELDS
_SERIAL_BRIDGE_SCRIPT = RETAINED_ARTIFACT_SERIAL_BRIDGE_SCRIPT
_LEGACY_NATIVE_BENCH_SCRIPT = RETAINED_ARTIFACT_LEGACY_NATIVE_BENCH_SCRIPT
_INT8_DIAGNOSTIC_SCRIPT = RETAINED_ARTIFACT_INT8_DIAGNOSTIC_SCRIPT
_GGUF_DIAGNOSTIC_SCRIPT = RETAINED_ARTIFACT_GGUF_DIAGNOSTIC_SCRIPT
_RETAINED_BENCH_SCRIPT = RETAINED_ARTIFACT_RETAINED_BENCH_SCRIPT
_RETAINED_BENCH_UNIQUE_FLAGS = RETAINED_ARTIFACT_RETAINED_BENCH_UNIQUE_FLAGS
_RETAINED_PROFILED_COMMAND_DISALLOWED_FLAGS = RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_DISALLOWED_FLAGS
_RETAINED_PROFILED_COMMAND_UNIQUE_FLAGS = RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_UNIQUE_FLAGS
_RETAINED_PROFILED_COMMAND_VALUE_FLAGS = RETAINED_ARTIFACT_RETAINED_PROFILED_COMMAND_VALUE_FLAGS
_PRIMITIVE_CORRECTNESS_UNIQUE_FLAGS = RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_UNIQUE_FLAGS
_RETAINED_PRECONDITION_KINDS = RETAINED_ARTIFACT_RETAINED_PRECONDITION_KINDS
_RETAINED_POSTCONDITION_KINDS = RETAINED_ARTIFACT_RETAINED_POSTCONDITION_KINDS
_RETAINED_CONDITION_STATUS_LABELS = RETAINED_ARTIFACT_RETAINED_CONDITION_STATUS_LABELS
_RETAINED_GATE_FLAGS = RETAINED_ARTIFACT_RETAINED_GATE_FLAGS
_RETAINED_GATE_LABELS = RETAINED_ARTIFACT_RETAINED_GATE_LABELS
_SWEEP_COMMAND_CATEGORIES = RETAINED_ARTIFACT_SWEEP_COMMAND_CATEGORIES
_SWEEP_COMMAND_STATUS_LABELS = RETAINED_ARTIFACT_SWEEP_COMMAND_STATUS_LABELS
(
    _PRIMITIVE_COMMAND_CATEGORY,
    _SERIAL_BRIDGE_COMMAND_CATEGORY,
    _NATIVE_DIAGNOSTIC_COMMAND_CATEGORY,
    _INT8_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY,
    _GGUF_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY,
) = _SWEEP_COMMAND_CATEGORIES
(
    _PLANNED_COMMAND_STATUS,
    _PASSED_COMMAND_STATUS,
    _SKIPPED_COMMAND_STATUS,
    _FAILED_COMMAND_STATUS,
) = _SWEEP_COMMAND_STATUS_LABELS
_INT8_DIAGNOSTIC_UNIQUE_FLAGS = ("--model", "--fixture", "--rows", "--future-json", "--primitive-cpu-json", "--primitive-hip-json")
_GGUF_DIAGNOSTIC_BACKEND = "hip_gfx1100"
_GGUF_DIAGNOSTIC_MAX_NEW_TOKENS = 4
_GGUF_DIAGNOSTIC_UNIQUE_FLAGS = ("--fixture", "--rows", "--backend", "--quant", "--max-new-tokens")
_BATCH_SAMPLE_UNIQUE_FLAGS = (
    "--batch-sample-mode",
    "--batch-sample-eq-ok",
    "--batch-sample-eq-artifact",
    "--batch-sample-eq-rows",
)
_SWEEP_COMMAND_KNOWN_FLAGS = tuple(
    dict.fromkeys(
        _RETAINED_PROFILED_COMMAND_UNIQUE_FLAGS
        + _BATCH_SAMPLE_UNIQUE_FLAGS
        + _PRIMITIVE_CORRECTNESS_UNIQUE_FLAGS
        + _INT8_DIAGNOSTIC_UNIQUE_FLAGS
        + _GGUF_DIAGNOSTIC_UNIQUE_FLAGS
    )
)
_GGUF_DIAGNOSTIC_FIXTURE = "tests/fixtures/gguf/qwen35_0_8b_q4_k_m_e2e.json"
_GGUF_DIAGNOSTIC_QUANTS = ("gguf_q4_k_m", "gguf_q5_k_m", "gguf_q6_k", "gguf_q8_0")


@dataclass(frozen=True, slots=True)
class SweepCommand:
    category: str
    batch_size: int
    artifact_path: Path
    argv: tuple[str, ...]

    @property
    def command(self) -> str:
        return shlex.join(self.argv)


def parse_batch_sizes(text: str) -> tuple[int, ...]:
    if not isinstance(text, str) or not text:
        raise argparse.ArgumentTypeError("batch sizes must be a non-empty string")
    parts = text.split(",")
    if any(not item.strip() for item in parts):
        raise argparse.ArgumentTypeError("batch sizes must not contain empty entries")
    try:
        values = tuple(int(item.strip()) for item in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch sizes must be integers") from exc
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("batch sizes must be unique")
    return values


def parse_cli_path(text: str) -> Path:
    if not isinstance(text, str) or not text.strip():
        raise argparse.ArgumentTypeError("path must be non-empty")
    return Path(text)


def _format_batch_template(value: str | Path, *, batch_size: int, option: str) -> str:
    text = str(value)
    try:
        return text.format(batch_size=batch_size, c=batch_size)
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(f"{option} supports only {{batch_size}}/{{c}} placeholders") from exc


def build_sweep_commands(args: argparse.Namespace) -> tuple[SweepCommand, ...]:
    output_dir = Path(args.output_dir)
    commands: list[SweepCommand] = []
    for c in args.batch_sizes:
        primitive_json = output_dir / f"primitive-c{c}.json"
        commands.append(
            SweepCommand(
                category=_PRIMITIVE_COMMAND_CATEGORY,
                batch_size=c,
                artifact_path=primitive_json,
                argv=(
                    *_python_command_prefix(),
                    _PRIMITIVE_CORRECTNESS_SCRIPT,
                    "--rows",
                    str(c),
                    "--seed",
                    str(args.seed),
                    "--json",
                    str(primitive_json),
                ),
            )
        )

        serial_json = output_dir / f"serial-bridge-c{c}.json"
        commands.append(
            SweepCommand(
                category=_SERIAL_BRIDGE_COMMAND_CATEGORY,
                batch_size=c,
                artifact_path=serial_json,
                argv=tuple(
                    _batch_bench_argv(
                        _SERIAL_BRIDGE_SCRIPT,
                        args,
                        batch_size=c,
                        artifact_path=serial_json,
                    )
                ),
            )
        )

        if c == 1:
            native_json = output_dir / "native-baseline-c1.json"
            native_argv = [
                *_python_command_prefix(),
                _LEGACY_NATIVE_BENCH_SCRIPT,
                "--model",
                str(args.model),
                "--prompt-length",
                str(args.prompt_length),
                "--decode-tokens",
                str(args.decode_tokens),
                "--warmup-decode-tokens",
                str(args.warmup_decode_tokens),
                "--max-layers",
                str(args.max_layers),
                "--json",
                str(native_json),
            ]
            if args.compiler_version_file is not None:
                native_argv.extend(["--compiler-version-file", str(args.compiler_version_file)])
            if args.require_cached_build:
                native_argv.append("--require-cached-build")
            commands.append(
                SweepCommand(
                    category=_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY,
                    batch_size=c,
                    artifact_path=native_json,
                    argv=tuple(native_argv),
                )
            )
            continue

        native_json = output_dir / f"native-diagnostic-c{c}.json"
        native_argv = _batch_bench_argv(
            _RETAINED_BENCH_SCRIPT,
            args,
            batch_size=c,
            artifact_path=native_json,
        )
        native_argv.extend(
            [
                _RETAINED_GATE_FLAGS[0],
                str(output_dir / "native-baseline-c1.json"),
                _RETAINED_GATE_FLAGS[1],
                str(serial_json),
                _RETAINED_GATE_FLAGS[2],
                str(primitive_json),
                _RETAINED_GATE_FLAGS[3],
                str(output_dir / f"profiler-c{c}.json"),
            ]
        )
        commands.append(
            SweepCommand(
                category=_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY,
                batch_size=c,
                artifact_path=native_json,
                argv=tuple(native_argv),
            )
        )
        if getattr(args, "include_int8", False):
            int8_json = output_dir / f"int8-native-diagnostic-c{c}.json"
            int8_future_json = output_dir / f"int8-native-retained-future-c{c}.json"
            int8_primitive_cpu_json = output_dir / f"int8-primitive-cpu-c{c}.json"
            int8_primitive_hip_json = output_dir / f"int8-primitive-hip-c{c}.json"
            commands.append(
                SweepCommand(
                    category=_INT8_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY,
                    batch_size=c,
                    artifact_path=int8_json,
                    argv=(
                        *_python_command_prefix(),
                        _INT8_DIAGNOSTIC_SCRIPT,
                        "--model",
                        str(args.model),
                        "--fixture",
                        str(args.fixture),
                        "--prompt-length",
                        str(args.prompt_length),
                        "--rows",
                        str(c),
                        "--decode-tokens",
                        str(args.decode_tokens),
                        "--warmup-decode-tokens",
                        str(args.warmup_decode_tokens),
                        "--max-layers",
                        str(args.max_layers),
                        "--future-json",
                        str(int8_future_json),
                        "--primitive-cpu-json",
                        str(int8_primitive_cpu_json),
                        "--primitive-hip-json",
                        str(int8_primitive_hip_json),
                        "--json",
                        str(int8_json),
                    ),
                )
            )
        if getattr(args, "include_gguf", False):
            for quant in _GGUF_DIAGNOSTIC_QUANTS:
                gguf_json = output_dir / f"gguf-native-diagnostic-c{c}-{quant}.json"
                commands.append(
                    SweepCommand(
                        category=_GGUF_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY,
                        batch_size=c,
                        artifact_path=gguf_json,
                        argv=(
                            *_python_command_prefix(),
                            _GGUF_DIAGNOSTIC_SCRIPT,
                            "--fixture",
                            _GGUF_DIAGNOSTIC_FIXTURE,
                            "--rows",
                            str(c),
                            "--backend",
                            _GGUF_DIAGNOSTIC_BACKEND,
                            "--quant",
                            quant,
                            "--max-new-tokens",
                            str(_GGUF_DIAGNOSTIC_MAX_NEW_TOKENS),
                            "--json",
                            str(gguf_json),
                        ),
                    )
                )
    return tuple(commands)


def _batch_bench_argv(
    script: str,
    args: argparse.Namespace,
    *,
    batch_size: int,
    artifact_path: Path,
) -> list[str]:
    argv = [
        *_python_command_prefix(),
        script,
        "--model",
        str(args.model),
        "--fixture",
        str(args.fixture),
        "--prompt-length",
        str(args.prompt_length),
        "--batch-size",
        str(batch_size),
        "--decode-tokens",
        str(args.decode_tokens),
        "--warmup-decode-tokens",
        str(args.warmup_decode_tokens),
        "--max-layers",
        str(args.max_layers),
        "--json",
        str(artifact_path),
    ]
    projection_dispatch_artifact = getattr(args, "projection_dispatch_artifact", None)
    if script == _RETAINED_BENCH_SCRIPT and projection_dispatch_artifact is not None:
        argv.extend(["--projection-dispatch-artifact", str(projection_dispatch_artifact)])
    if script == _RETAINED_BENCH_SCRIPT:
        batch_sample_mode = getattr(args, "batch_sample_mode", None)
        if batch_sample_mode is not None:
            argv.extend(["--batch-sample-mode", str(batch_sample_mode)])
        if getattr(args, "batch_sample_eq_ok", False):
            argv.append("--batch-sample-eq-ok")
        batch_sample_eq_artifact_template = getattr(args, "batch_sample_eq_artifact_template", None)
        if batch_sample_eq_artifact_template is not None:
            argv.extend(
                [
                    "--batch-sample-eq-artifact",
                    _format_batch_template(
                        batch_sample_eq_artifact_template,
                        batch_size=batch_size,
                        option="--batch-sample-eq-artifact-template",
                    ),
                ]
            )
        batch_sample_eq_rows = getattr(args, "batch_sample_eq_rows", None)
        if batch_sample_eq_rows is not None:
            argv.extend(
                [
                    "--batch-sample-eq-rows",
                    _format_batch_template(
                        batch_sample_eq_rows,
                        batch_size=batch_size,
                        option="--batch-sample-eq-rows",
                    ),
                ]
            )
        elif getattr(args, "batch_sample_eq_ok", False):
            argv.extend(["--batch-sample-eq-rows", str(batch_size)])
    if args.compiler_version_file is not None:
        argv.extend(["--compiler-version-file", str(args.compiler_version_file)])
    if args.require_cached_build:
        argv.append("--require-cached-build")
    return argv


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_stripped_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _is_positive_finite_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value)) and float(value) > 0.0


def _is_nonnegative_finite_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value)) and float(value) >= 0.0


def _is_zero_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


def _is_exact_zero_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value)) and float(value) == 0.0


def _is_bounded_primitive_numpy_oracle(value: Any) -> bool:
    return (
        _is_number(value)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= _PRIMITIVE_CORRECTNESS_NUMPY_MAX_ABS_LIMIT
    )


def _command_arg_path(command: SweepCommand, flag: str, *, kind: str) -> tuple[Path | None, dict[str, Any] | None]:
    argv = list(command.argv)
    try:
        idx = argv.index(flag)
        return Path(argv[idx + 1]), None
    except (ValueError, IndexError):
        return None, {
            "kind": kind,
            "artifact_path": None,
            "passed": False,
            "reason": f"retained native diagnostic is missing {flag}",
        }


def _primitive_correctness_precondition(command: SweepCommand) -> dict[str, Any]:
    primitive_path, error = _command_arg_path(
        command,
        _RETAINED_GATE_FLAGS[2],
        kind=_RETAINED_PRECONDITION_KINDS[0],
    )
    if error is not None:
        return error
    assert primitive_path is not None
    if not primitive_path.exists():
        return {
            "kind": _RETAINED_PRECONDITION_KINDS[0],
            "artifact_path": str(primitive_path),
            "passed": False,
            "reason": "primitive correctness artifact does not exist",
        }
    try:
        payload = _load_json_path(primitive_path)
    except Exception as exc:
        return {
            "kind": _RETAINED_PRECONDITION_KINDS[0],
            "artifact_path": str(primitive_path),
            "passed": False,
            "reason": f"primitive correctness artifact is invalid JSON: {type(exc).__name__}: {exc}",
        }
    reasons: list[str] = []
    if not isinstance(payload, dict):
        reasons.append("primitive correctness artifact root is not an object")
    else:
        primitive_schema = payload.get("schema")
        if (
            not isinstance(primitive_schema, int)
            or isinstance(primitive_schema, bool)
            or primitive_schema != _REQUIRED_PRIMITIVE_CORRECTNESS_SCHEMA
        ):
            reasons.append("schema is missing or not 1")
        primitive_artifact_path = payload.get("artifact_path")
        if not isinstance(primitive_artifact_path, str) or not primitive_artifact_path:
            reasons.append("artifact_path is missing or not a non-empty string")
        elif primitive_artifact_path != str(primitive_path):
            reasons.append("artifact_path does not match primitive correctness artifact path")
        primitive_seed = payload.get("seed")
        if (
            not isinstance(primitive_seed, int)
            or isinstance(primitive_seed, bool)
            or primitive_seed != _REQUIRED_PRIMITIVE_CORRECTNESS_SEED
        ):
            reasons.append("seed is missing or not 1234")
        for field, expected_value in _REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS.items():
            value = payload.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value != expected_value:
                reasons.append(f"{field} is missing or not {expected_value}")
        context_lens = payload.get("context_lens")
        if not _primitive_context_lens_matches(context_lens, command.batch_size):
            reasons.append("context_lens is missing or does not match fixture coverage")
        primitive_rows = payload.get("rows")
        if not isinstance(primitive_rows, int) or isinstance(primitive_rows, bool) or primitive_rows != command.batch_size:
            reasons.append(f"rows={primitive_rows!r} is missing or does not match batch_size={command.batch_size}")
        if payload.get("passed") is not True:
            reasons.append("passed is not true")
        for field in ("append_key_mismatch", "append_value_mismatch"):
            if not _is_zero_int(payload.get(field)):
                reasons.append(f"{field} is missing or not integer zero")
        attn_vs_c1 = payload.get("attn_batch_vs_c1_max_abs")
        if not _is_exact_zero_number(attn_vs_c1):
            reasons.append("attn_batch_vs_c1_max_abs is missing or not 0.0")
        attn_vs_numpy = payload.get("attn_batch_vs_numpy_max_abs")
        if not _is_bounded_primitive_numpy_oracle(attn_vs_numpy):
            reasons.append("attn_batch_vs_numpy_max_abs is missing, non-finite, negative, or above 2e-5")
        if not reasons:
            for field in ("append_batch_aa_key_mismatch", "append_batch_aa_value_mismatch"):
                if not _is_zero_int(payload.get(field)):
                    reasons.append(f"{field} is missing or not integer zero")
            if not _is_exact_zero_number(payload.get("attn_batch_aa_max_abs")):
                reasons.append("attn_batch_aa_max_abs is missing or not 0.0")
            if payload.get("aa_passed") is not True:
                reasons.append("aa_passed is not true")
        if not reasons:
            reasons.extend(_primitive_device_metadata_blockers(payload.get("device")))
    result: dict[str, Any] = {
        "kind": _RETAINED_PRECONDITION_KINDS[0],
        "artifact_path": str(primitive_path),
        "passed": not reasons,
        "reason": None if not reasons else "; ".join(reasons),
    }
    if not reasons and isinstance(payload, dict):
        result.update(
            {
                "primitive_schema": int(payload["schema"]),
                "primitive_artifact_path": str(payload["artifact_path"]),
                "primitive_seed": int(payload["seed"]),
                **{
                    f"primitive_{field}": int(payload[field])
                    for field in _REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS
                },
                "primitive_context_lens": list(payload["context_lens"]),
                "primitive_rows": int(payload["rows"]),
                "primitive_device": dict(payload["device"]),
                "append_key_mismatch": int(payload["append_key_mismatch"]),
                "append_value_mismatch": int(payload["append_value_mismatch"]),
                "append_batch_aa_key_mismatch": int(payload["append_batch_aa_key_mismatch"]),
                "append_batch_aa_value_mismatch": int(payload["append_batch_aa_value_mismatch"]),
                "attn_batch_aa_max_abs": float(payload["attn_batch_aa_max_abs"]),
                "aa_passed": bool(payload["aa_passed"]),
                "attn_batch_vs_c1_max_abs": float(payload["attn_batch_vs_c1_max_abs"]),
                "attn_batch_vs_numpy_max_abs": float(payload["attn_batch_vs_numpy_max_abs"]),
            }
        )
    return result


def _extract_decode_rates(payload: dict[str, Any]) -> tuple[float | None, float | None]:
    measurements = payload.get("measurements")
    aggregate = None
    per_request = None
    if isinstance(measurements, dict):
        if _is_number(measurements.get("decode_tok_s_aggregate")):
            aggregate = float(measurements["decode_tok_s_aggregate"])
        if _is_number(measurements.get("decode_tok_s_per_request")):
            per_request = float(measurements["decode_tok_s_per_request"])
    throughput = payload.get("throughput")
    if isinstance(throughput, dict) and _is_number(throughput.get("warmed_decode_tok_s")):
        aggregate = float(throughput["warmed_decode_tok_s"])
        per_request = float(throughput["warmed_decode_tok_s"])
    workload = payload.get("workload")
    if aggregate is not None and per_request is None and isinstance(workload, dict):
        concurrency = workload.get("concurrency")
        if isinstance(concurrency, int) and not isinstance(concurrency, bool) and concurrency > 0:
            per_request = aggregate / concurrency
    return aggregate, per_request


def _argv_value(argv: Sequence[str], flag: str) -> str | None:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _flag_token_matches(token: str, flag: str) -> bool:
    return token == flag or token.startswith(f"{flag}=")


def _duplicate_flags(argv: Sequence[str], flags: Sequence[str]) -> list[str]:
    return [flag for flag in flags if sum(1 for token in argv if _flag_token_matches(token, flag)) > 1]


def _empty_inline_flag_values(argv: Sequence[str], flags: Sequence[str]) -> list[str]:
    return [flag for flag in flags if f"{flag}=" in argv]


def _command_arg_value(command: SweepCommand, flag: str) -> str | None:
    return _argv_value(list(command.argv), flag)


def _command_arg_int(command: SweepCommand, flag: str) -> int | None:
    value = _command_arg_value(command, flag)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _command_text_has_flag(command_text: str, flag: str) -> bool:
    try:
        argv = shlex.split(command_text)
    except ValueError:
        return False
    return flag in argv


def _command_text_arg(command_text: str, flag: str) -> str | None:
    try:
        argv = shlex.split(command_text)
    except ValueError:
        return None
    for index, value in enumerate(argv):
        if value == flag:
            try:
                return argv[index + 1]
            except IndexError:
                return None
        prefix = f"{flag}="
        if value.startswith(prefix):
            return value[len(prefix) :]
    return None


def _reference_label(payload: dict[str, Any], *keys: str) -> Any:
    workload = payload.get("workload")
    if isinstance(workload, dict):
        for key in keys:
            value = workload.get(key)
            if value is not None:
                return value
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _profiler_command_label(profiler: dict[str, Any], payload: dict[str, Any] | None) -> str | None:
    for source in (profiler, payload):
        if not isinstance(source, dict):
            continue
        for key in ("command", "profiler_command"):
            value = source.get(key)
            if isinstance(value, str) and value:
                return value
        commands = source.get("commands")
        if isinstance(commands, dict):
            value = commands.get("profiler")
            if isinstance(value, str) and value:
                return value
    return None


def _has_disallowed_profiler_kernel_fragment(name: str) -> bool:
    lowered = name.lower()
    return any(fragment in lowered for fragment in _DISALLOWED_PROFILER_KERNEL_NAME_FRAGMENTS)


def _is_kernel_trace_csv_path(trace_file: str) -> bool:
    name = Path(trace_file).name.lower()
    return Path(trace_file).suffix.lower() == ".csv" and "kernel" in name and "trace" in name


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


def _path_has_parent_directory_component(path: str | Path) -> bool:
    return ".." in Path(path).parts


def _resolve_profiler_trace_file(trace_file: str, *, profiler_path: Path) -> Path:
    path = Path(trace_file)
    if path.is_absolute():
        return path
    parent_relative = profiler_path.parent / path
    if parent_relative.exists():
        return parent_relative
    return path


def _profiler_trace_row_kernel_name(row: dict[str, Any]) -> str:
    for column in _PROFILER_TRACE_KERNEL_NAME_COLUMNS:
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _profiler_trace_row_duration_ns(row: dict[str, Any]) -> float | None:
    for column in _PROFILER_TRACE_DURATION_COLUMNS:
        value = row.get(column)
        if value in (None, ""):
            continue
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if duration > 0.0 and math.isfinite(duration):
            return duration
    start = None
    end = None
    for column in _PROFILER_TRACE_START_COLUMNS:
        value = row.get(column)
        if value in (None, ""):
            continue
        try:
            start = float(value)
            break
        except (TypeError, ValueError):
            continue
    for column in _PROFILER_TRACE_END_COLUMNS:
        value = row.get(column)
        if value in (None, ""):
            continue
        try:
            end = float(value)
            break
        except (TypeError, ValueError):
            continue
    if start is None or end is None:
        return None
    duration = end - start
    return duration if duration > 0.0 and math.isfinite(duration) else None


def _read_profiler_trace_kernel_names(trace_file: Path) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    try:
        with trace_file.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = _profiler_trace_row_kernel_name(row)
                if name and name not in seen:
                    names.append(name)
                    seen.add(name)
    except OSError:
        return []
    return names


def _read_profiler_trace_kernel_durations(trace_file: Path) -> dict[str, float]:
    durations: dict[str, float] = {}
    try:
        with trace_file.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = _profiler_trace_row_kernel_name(row)
                duration_ns = _profiler_trace_row_duration_ns(row)
                if name and duration_ns is not None:
                    durations[name] = durations.get(name, 0.0) + duration_ns
    except OSError:
        return {}
    return durations


def _synthesized_profiler_trace_kernel_names(profiler: dict[str, Any], *, profiler_path: Path) -> list[str] | None:
    trace_files = profiler.get("trace_files")
    if not isinstance(trace_files, list) or not trace_files:
        return None
    names: list[str] = []
    seen: set[str] = set()
    for trace_file in trace_files:
        if not isinstance(trace_file, str) or not trace_file:
            continue
        for kernel_name in _read_profiler_trace_kernel_names(_resolve_profiler_trace_file(trace_file, profiler_path=profiler_path)):
            if kernel_name not in seen:
                names.append(kernel_name)
                seen.add(kernel_name)
    return names or None


def _synthesized_profiler_kernel_durations_from_traces(profiler: dict[str, Any], *, profiler_path: Path) -> dict[str, float] | None:
    trace_files = profiler.get("trace_files")
    if not isinstance(trace_files, list) or not trace_files:
        return None
    durations: dict[str, float] = {}
    for trace_file in trace_files:
        if not isinstance(trace_file, str) or not trace_file:
            continue
        for kernel_name, duration_ns in _read_profiler_trace_kernel_durations(
            _resolve_profiler_trace_file(trace_file, profiler_path=profiler_path)
        ).items():
            durations[kernel_name] = durations.get(kernel_name, 0.0) + duration_ns
    return durations or None


def _synthesize_profiler_trace_fields(profiler: dict[str, Any], *, profiler_path: Path) -> list[str]:
    synthesized_fields: list[str] = []
    if "trace_kernel_names" not in profiler:
        trace_kernel_names = _synthesized_profiler_trace_kernel_names(profiler, profiler_path=profiler_path)
        if trace_kernel_names is not None:
            profiler["trace_kernel_names"] = trace_kernel_names
            synthesized_fields.append("trace_kernel_names")
    synthesized_durations_from_trace = False
    if "kernel_durations_ns" not in profiler:
        kernel_durations = _synthesized_profiler_kernel_durations_from_traces(profiler, profiler_path=profiler_path)
        if kernel_durations is not None:
            profiler["kernel_durations_ns"] = kernel_durations
            synthesized_fields.append("kernel_durations_ns")
            synthesized_durations_from_trace = True
    kernel_durations = profiler.get("kernel_durations_ns")
    if not isinstance(kernel_durations, dict) or not kernel_durations or not synthesized_durations_from_trace:
        return synthesized_fields
    if "total_kernel_duration_ns" not in profiler:
        total = sum(
            float(duration_ns)
            for duration_ns in kernel_durations.values()
            if _is_number(duration_ns) and float(duration_ns) > 0.0
        )
        if total > 0.0:
            profiler["total_kernel_duration_ns"] = total
            synthesized_fields.append("total_kernel_duration_ns")
    total_duration = profiler.get("total_kernel_duration_ns")
    if "kernel_duration_shares" not in profiler and _is_number(total_duration) and float(total_duration) > 0.0:
        profiler["kernel_duration_shares"] = {
            str(kernel_name): float(duration_ns) / float(total_duration)
            for kernel_name, duration_ns in kernel_durations.items()
            if isinstance(kernel_name, str) and kernel_name and _is_number(duration_ns) and float(duration_ns) > 0.0
        }
        synthesized_fields.append("kernel_duration_shares")
    if "kernel_duration_categories_ns" not in profiler:
        profiler["kernel_duration_categories_ns"] = _profiler_kernel_duration_category_sums(kernel_durations)
        synthesized_fields.append("kernel_duration_categories_ns")
    duration_categories = profiler.get("kernel_duration_categories_ns")
    if (
        "kernel_duration_category_shares" not in profiler
        and isinstance(duration_categories, dict)
        and _is_number(total_duration)
        and float(total_duration) > 0.0
    ):
        profiler["kernel_duration_category_shares"] = {
            category: float(duration_categories.get(category, 0.0)) / float(total_duration)
            for category in _PROFILER_KERNEL_DURATION_CATEGORIES
        }
        synthesized_fields.append("kernel_duration_category_shares")
    return synthesized_fields


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


def _profiler_kernel_duration_category_sums(kernel_durations: dict[Any, Any]) -> dict[str, float]:
    categories = dict.fromkeys(_PROFILER_KERNEL_DURATION_CATEGORIES, 0.0)
    for kernel_name, duration_ns in kernel_durations.items():
        if not isinstance(kernel_name, str) or not kernel_name:
            continue
        if not _is_number(duration_ns) or float(duration_ns) <= 0.0:
            continue
        categories[_profiler_kernel_duration_category(kernel_name)] += float(duration_ns)
    return categories


def _validate_profiler_kernel_durations(profiler: dict[str, Any], reasons: list[str]) -> None:
    kernel_durations = profiler.get("kernel_durations_ns")
    if not isinstance(kernel_durations, dict) or not kernel_durations:
        return
    total_duration = profiler.get("total_kernel_duration_ns")
    duration_shares = profiler.get("kernel_duration_shares")
    if not _is_positive_finite_number(total_duration):
        reasons.append("total_kernel_duration_ns is missing or non-positive finite numeric")
        return
    if not isinstance(duration_shares, dict) or not duration_shares:
        reasons.append("kernel_duration_shares is missing or empty")
        return
    if any(not _is_stripped_non_empty_string(key) for key in kernel_durations):
        reasons.append("kernel_durations_ns keys must be non-empty strings")
    if any(not _is_stripped_non_empty_string(key) for key in duration_shares):
        reasons.append("kernel_duration_shares keys must be non-empty strings")
    duration_keys = {key for key in kernel_durations if _is_stripped_non_empty_string(key)}
    share_keys = {key for key in duration_shares if _is_stripped_non_empty_string(key)}
    if duration_keys != share_keys:
        reasons.append("kernel_duration_shares keys do not match kernel_durations_ns")
    if any(_has_disallowed_profiler_kernel_fragment(key) for key in share_keys):
        reasons.append("kernel_duration_shares contains a serial/per-row/fallback kernel")

    duration_sum = 0.0
    share_sum = 0.0
    for kernel_name in sorted(duration_keys):
        duration_ns = kernel_durations.get(kernel_name)
        duration_share = duration_shares.get(kernel_name)
        if not _is_positive_finite_number(duration_ns):
            reasons.append(f"kernel_durations_ns.{kernel_name} is missing or non-positive finite numeric")
            continue
        duration_sum += float(duration_ns)
        if not _is_positive_finite_number(duration_share):
            reasons.append(f"kernel_duration_shares.{kernel_name} is missing or non-positive finite numeric")
            continue
        share_sum += float(duration_share)
        expected_share = float(duration_ns) / float(total_duration)
        if abs(float(duration_share) - expected_share) > 1e-6:
            reasons.append(f"kernel_duration_shares.{kernel_name} does not match kernel duration share")
    tolerance = max(1.0, duration_sum * 1e-6)
    if duration_sum > 0.0 and abs(float(total_duration) - duration_sum) > tolerance:
        reasons.append("total_kernel_duration_ns does not match sum(kernel_durations_ns)")
    if share_sum > 0.0 and abs(share_sum - 1.0) > 1e-6:
        reasons.append("kernel_duration_shares does not sum to 1.0")


def _validate_profiler_kernel_duration_categories(profiler: dict[str, Any], reasons: list[str]) -> None:
    total_duration = profiler.get("total_kernel_duration_ns")
    if not _is_positive_finite_number(total_duration):
        return
    duration_categories = profiler.get("kernel_duration_categories_ns")
    category_shares = profiler.get("kernel_duration_category_shares")
    if not isinstance(duration_categories, dict) or not duration_categories:
        reasons.append("kernel_duration_categories_ns is missing or empty")
        return
    if not isinstance(category_shares, dict) or not category_shares:
        reasons.append("kernel_duration_category_shares is missing or empty")
        return
    expected_keys = set(_PROFILER_KERNEL_DURATION_CATEGORIES)
    duration_keys = set(duration_categories)
    share_keys = set(category_shares)
    if duration_keys != expected_keys:
        reasons.append("kernel_duration_categories_ns keys do not match required categories")
    if share_keys != expected_keys:
        reasons.append("kernel_duration_category_shares keys do not match required categories")
    kernel_durations = profiler.get("kernel_durations_ns")
    if isinstance(kernel_durations, dict) and duration_keys == expected_keys:
        expected_categories = _profiler_kernel_duration_category_sums(kernel_durations)
        if any(
            _is_number(duration_categories.get(category))
            and abs(float(duration_categories[category]) - expected_duration) > max(1.0, expected_duration * 1e-6)
            for category, expected_duration in expected_categories.items()
        ):
            reasons.append("kernel_duration_categories_ns does not match categorized kernel_durations_ns")

    duration_sum = 0.0
    share_sum = 0.0
    category_value_error = False
    for category in _PROFILER_KERNEL_DURATION_CATEGORIES:
        duration_ns = duration_categories.get(category)
        duration_share = category_shares.get(category)
        if not _is_nonnegative_finite_number(duration_ns):
            reasons.append(f"kernel_duration_categories_ns.{category} is missing or negative/non-finite numeric")
            category_value_error = True
            continue
        duration_sum += float(duration_ns)
        if not _is_nonnegative_finite_number(duration_share):
            reasons.append(f"kernel_duration_category_shares.{category} is missing or negative/non-finite numeric")
            category_value_error = True
            continue
        share_sum += float(duration_share)
        expected_share = float(duration_ns) / float(total_duration)
        if abs(float(duration_share) - expected_share) > 1e-6:
            reasons.append(f"kernel_duration_category_shares.{category} does not match kernel category duration share")
    if category_value_error:
        return
    tolerance = max(1.0, float(total_duration) * 1e-6)
    if abs(duration_sum - float(total_duration)) > tolerance:
        reasons.append("kernel_duration_categories_ns does not sum to total_kernel_duration_ns")
    if abs(share_sum - 1.0) > 1e-6:
        reasons.append("kernel_duration_category_shares does not sum to 1.0")


def _validate_profiler_cpu_side_bottlenecks(profiler: dict[str, Any], reasons: list[str]) -> None:
    cpu_total = profiler.get("cpu_side_total_seconds")
    durations = profiler.get("cpu_side_bottlenecks_seconds")
    shares = profiler.get("cpu_side_bottleneck_shares")
    if not _is_positive_finite_number(cpu_total):
        reasons.append("cpu_side_total_seconds is missing or non-positive finite numeric")
        return
    if not isinstance(durations, dict) or not durations:
        reasons.append("cpu_side_bottlenecks_seconds is missing or empty")
        return
    if not isinstance(shares, dict) or not shares:
        reasons.append("cpu_side_bottleneck_shares is missing or empty")
        return
    expected_keys = set(_PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES)
    duration_keys = set(durations)
    share_keys = set(shares)
    if duration_keys != expected_keys:
        reasons.append("cpu_side_bottlenecks_seconds keys do not match required categories")
    if share_keys != expected_keys:
        reasons.append("cpu_side_bottleneck_shares keys do not match required categories")

    duration_sum = 0.0
    share_sum = 0.0
    cpu_value_error = False
    for category in _PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES:
        duration_seconds = durations.get(category)
        duration_share = shares.get(category)
        if not _is_nonnegative_finite_number(duration_seconds):
            reasons.append(f"cpu_side_bottlenecks_seconds.{category} is missing or negative/non-finite numeric")
            cpu_value_error = True
            continue
        duration_sum += float(duration_seconds)
        if not _is_nonnegative_finite_number(duration_share):
            reasons.append(f"cpu_side_bottleneck_shares.{category} is missing or negative/non-finite numeric")
            cpu_value_error = True
            continue
        share_sum += float(duration_share)
        expected_share = float(duration_seconds) / float(cpu_total)
        if abs(float(duration_share) - expected_share) > 1e-6:
            reasons.append(f"cpu_side_bottleneck_shares.{category} does not match cpu-side duration share")
    if cpu_value_error:
        return
    tolerance = max(1e-9, float(cpu_total) * 1e-6)
    if abs(duration_sum - float(cpu_total)) > tolerance:
        reasons.append("cpu_side_bottlenecks_seconds does not sum to cpu_side_total_seconds")
    if abs(share_sum - 1.0) > 1e-6:
        reasons.append("cpu_side_bottleneck_shares does not sum to 1.0")


def _visible_device_env_assignments(payload: Mapping[str, Any]) -> tuple[dict[str, str], list[str]]:
    hardware = payload.get("hardware")
    visible_device = hardware.get("visible_device") if isinstance(hardware, Mapping) else None
    env = visible_device.get("env") if isinstance(visible_device, Mapping) else None
    assignments: dict[str, str] = {}
    reasons: list[str] = []
    gpu_name = hardware.get("gpu") if isinstance(hardware, Mapping) else None
    device_name = visible_device.get("device_name") if isinstance(visible_device, Mapping) else None
    if isinstance(gpu_name, str) and gpu_name and isinstance(device_name, str) and device_name and gpu_name != device_name:
        reasons.append("hardware.gpu does not match hardware.visible_device.device_name")
    if not isinstance(env, Mapping):
        return assignments, reasons
    for key in _COMMAND_ENV_KEYS:
        value = env.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            reasons.append(f"hardware.visible_device.env.{key} is not a non-empty string when present")
        elif not value.strip():
            reasons.append(f"hardware.visible_device.env.{key} is not a non-blank string when present")
        else:
            assignments[key] = value
    return assignments, reasons


def _scaling_reference_software_reasons(payload: Mapping[str, Any]) -> list[str]:
    software = payload.get("software")
    if not isinstance(software, Mapping):
        return ["software provenance is missing for device-stamped scaling reference"]
    reasons: list[str] = []
    for field in ("python", "hipcc_version", "hipengine_commit"):
        value = software.get(field)
        if not isinstance(value, str) or not value:
            reasons.append(f"software.{field} is missing or not a non-empty string")
    dirty = software.get("hipengine_dirty")
    if not isinstance(dirty, bool):
        reasons.append("software.hipengine_dirty is missing or not a bool")
    return reasons


def _scaling_reference_command_env_assignments(
    payload: Mapping[str, Any],
    *,
    required_env: Mapping[str, str] | None = None,
    require_command_label: bool = False,
    expected_command_script: str | None = None,
    expected_inputs: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    commands = payload.get("commands")
    command = commands.get("benchmark") if isinstance(commands, Mapping) else payload.get("command")
    if command is None:
        reasons = [
            f"commands.benchmark is missing while retained command env sets {key}"
            for key, value in (required_env or {}).items()
            if require_command_label and value
        ]
        return {}, reasons
    if not isinstance(command, str) or not command:
        return {}, ["commands.benchmark is not a non-empty string when present"]
    try:
        argv = shlex.split(command)
    except ValueError:
        return {}, ["commands.benchmark is not shell-parseable"]
    raw_assignments = _command_device_env_assignments(argv)
    assignments: dict[str, str] = {}
    reasons: list[str] = []
    for key, value in raw_assignments.items():
        if not value.strip():
            reasons.append(f"commands.benchmark device env {key} is not a non-blank string when present")
        else:
            assignments[key] = value
    reasons.extend(
        f"commands.benchmark device env {key} is missing while retained command env sets it"
        for key, value in (required_env or {}).items()
        if value and key not in raw_assignments
    )
    if expected_command_script is not None:
        launch = _strip_command_env_prefix(argv)
        if len(launch) < 2 or not _is_python_executable(launch[0]) or launch[1] != expected_command_script:
            reasons.append(f"commands.benchmark must launch {expected_command_script}")
    for label, flag in (("model", "--model"), ("fixture", "--fixture")):
        expected_value = (expected_inputs or {}).get(label)
        if not expected_value:
            continue
        command_value = _command_text_arg(command, flag)
        if command_value is None:
            reasons.append(f"commands.benchmark must include {flag} matching retained {label}")
        elif command_value != expected_value:
            reasons.append(f"commands.benchmark {flag} does not match retained {label}")
    return assignments, reasons


def _scaling_reference_expected_command_script(kind: str) -> str | None:
    if kind == _RETAINED_PRECONDITION_KINDS[1]:
        return _LEGACY_NATIVE_BENCH_SCRIPT
    if kind == _RETAINED_PRECONDITION_KINDS[2]:
        return _SERIAL_BRIDGE_SCRIPT
    return None


def _scaling_reference_expected_inputs(command: SweepCommand, kind: str) -> dict[str, str]:
    expected_model = _command_arg_value(command, "--model")
    expected_fixture = _command_arg_value(command, "--fixture")
    result = {"model": expected_model} if expected_model else {}
    if kind != _RETAINED_PRECONDITION_KINDS[1] and expected_fixture:
        result["fixture"] = expected_fixture
    return result


def _scaling_reference_precondition(
    command: SweepCommand,
    *,
    flag: str,
    kind: str,
    expected_concurrency: int | None = None,
) -> dict[str, Any]:
    path, error = _command_arg_path(command, flag, kind=kind)
    if error is not None:
        return error
    assert path is not None
    if not path.exists():
        return {
            "kind": kind,
            "artifact_path": str(path),
            "passed": False,
            "reason": "scaling reference artifact does not exist",
        }
    try:
        payload = _load_json_path(path)
    except Exception as exc:
        return {
            "kind": kind,
            "artifact_path": str(path),
            "passed": False,
            "reason": f"scaling reference artifact is invalid JSON: {type(exc).__name__}: {exc}",
        }
    reasons: list[str] = []
    status: str | None = None
    reference_reason: Any = None
    aggregate: float | None = None
    per_request: float | None = None
    concurrency: int | None = None
    prompt_tokens: int | None = None
    gen_tokens: int | None = None
    source_artifact_path: str | None = None
    benchmark_model: str | None = None
    benchmark_fixture: str | None = None
    if not isinstance(payload, dict):
        reasons.append("scaling reference artifact root is not an object")
    else:
        raw_artifact_path = payload.get("artifact_path")
        if not isinstance(raw_artifact_path, str) or not raw_artifact_path:
            reasons.append("artifact_path is missing or not a non-empty string")
        elif raw_artifact_path != str(path):
            source_artifact_path = raw_artifact_path
            reasons.append("artifact_path does not match scaling reference artifact path")
        else:
            source_artifact_path = raw_artifact_path
        reference_device_env, reference_device_env_reasons = _visible_device_env_assignments(payload)
        reasons.extend(reference_device_env_reasons)
        retained_device_env = _command_device_env_assignments(command.argv)
        if reference_device_env:
            for key, value in reference_device_env.items():
                if retained_device_env.get(key) != value:
                    reasons.append(f"hardware.visible_device.env.{key} does not match retained command env")
        commands = payload.get("commands")
        benchmark_command = commands.get("benchmark") if isinstance(commands, Mapping) else payload.get("command")
        if isinstance(benchmark_command, str):
            benchmark_model = _command_text_arg(benchmark_command, "--model")
            benchmark_fixture = _command_text_arg(benchmark_command, "--fixture")
        reference_command_env, reference_command_env_reasons = _scaling_reference_command_env_assignments(
            payload,
            required_env=retained_device_env,
            require_command_label=bool(reference_device_env),
            expected_command_script=_scaling_reference_expected_command_script(kind),
            expected_inputs=_scaling_reference_expected_inputs(command, kind),
        )
        reasons.extend(reference_command_env_reasons)
        if reference_command_env:
            for key, value in reference_command_env.items():
                if retained_device_env.get(key) != value:
                    reasons.append(f"commands.benchmark device env {key} does not match retained command env")
            if not reasons:
                for key, value in reference_command_env.items():
                    if retained_device_env.get(key) == value and key not in reference_device_env:
                        reasons.append(f"hardware.visible_device.env.{key} is missing while benchmark command env sets it")
        if reference_device_env and not reasons:
            reasons.extend(_scaling_reference_software_reasons(payload))
        raw_status = payload.get("status")
        status = str(raw_status) if raw_status else "loaded"
        if status in _UNUSABLE_SCALING_REFERENCE_STATUSES:
            reasons.append(f"status={status!r} is not usable as a scaling reference")
        reference_reason = payload.get("reason")
        if reference_reason is not None:
            reasons.append(f"scaling reference reason is non-null: {reference_reason}")
        aggregate, per_request = _extract_decode_rates(payload)
        if aggregate is None or per_request is None:
            reasons.append("decode throughput fields are missing")
        elif not _is_positive_finite_number(aggregate) or not _is_positive_finite_number(per_request):
            reasons.append("decode throughput fields must be positive finite numbers")
        workload = payload.get("workload")
        raw_concurrency = workload.get("concurrency") if isinstance(workload, dict) else None
        if isinstance(raw_concurrency, int) and not isinstance(raw_concurrency, bool):
            concurrency = raw_concurrency
        if kind == _RETAINED_PRECONDITION_KINDS[1] and concurrency is None:
            concurrency = 1
        if expected_concurrency is not None and concurrency != expected_concurrency:
            reasons.append(f"workload.concurrency={concurrency!r} does not match batch_size={expected_concurrency}")
        if (
            aggregate is not None
            and per_request is not None
            and _is_positive_finite_number(aggregate)
            and _is_positive_finite_number(per_request)
            and concurrency is not None
        ):
            expected_aggregate = float(per_request) * int(concurrency)
            if abs(float(aggregate) - expected_aggregate) > max(1e-9, expected_aggregate * 1e-6):
                reasons.append("decode aggregate rate does not match per-request rate times concurrency")
        expected_prompt_length = _command_arg_int(command, "--prompt-length")
        raw_prompt_tokens = _reference_label(payload, "prompt_tokens_per_request", "prompt_length")
        if not isinstance(raw_prompt_tokens, int) or isinstance(raw_prompt_tokens, bool):
            reasons.append("prompt token count label is missing")
        else:
            prompt_tokens = raw_prompt_tokens
            if expected_prompt_length is not None and prompt_tokens != expected_prompt_length:
                reasons.append(f"prompt_tokens_per_request={prompt_tokens!r} does not match prompt_length={expected_prompt_length}")
        expected_decode_tokens = _command_arg_int(command, "--decode-tokens")
        raw_gen_tokens = _reference_label(payload, "gen_tokens_per_request", "decode_tokens")
        if not isinstance(raw_gen_tokens, int) or isinstance(raw_gen_tokens, bool):
            reasons.append("decode token count label is missing")
        else:
            gen_tokens = raw_gen_tokens
            if expected_decode_tokens is not None and gen_tokens != expected_decode_tokens:
                reasons.append(f"gen_tokens_per_request={gen_tokens!r} does not match decode_tokens={expected_decode_tokens}")
    return {
        "kind": kind,
        "artifact_path": str(path),
        "passed": not reasons,
        "reason": None if not reasons else "; ".join(reasons),
        "reference_artifact_path": source_artifact_path,
        "reference_status": status,
        "reference_reason": reference_reason,
        "benchmark_model": benchmark_model,
        "benchmark_fixture": benchmark_fixture,
        "workload_concurrency": concurrency,
        "prompt_tokens_per_request": prompt_tokens,
        "gen_tokens_per_request": gen_tokens,
        "decode_tok_s_aggregate": aggregate,
        "decode_tok_s_per_request": per_request,
    }


def _profiler_summary_precondition(command: SweepCommand) -> dict[str, Any]:
    profiler_path, error = _command_arg_path(
        command,
        _RETAINED_GATE_FLAGS[3],
        kind=_RETAINED_PRECONDITION_KINDS[3],
    )
    if error is not None:
        return error
    assert profiler_path is not None
    if not profiler_path.exists():
        return {
            "kind": _RETAINED_PRECONDITION_KINDS[3],
            "artifact_path": str(profiler_path),
            "passed": False,
            "reason": "profiler summary artifact does not exist",
        }
    try:
        payload = _load_json_path(profiler_path)
    except Exception as exc:
        return {
            "kind": _RETAINED_PRECONDITION_KINDS[3],
            "artifact_path": str(profiler_path),
            "passed": False,
            "reason": f"profiler summary artifact is invalid JSON: {type(exc).__name__}: {exc}",
        }
    profiler = (
        payload.get("profiler")
        if isinstance(payload, dict) and isinstance(payload.get("profiler"), dict)
        else payload
    )
    reasons: list[str] = []
    profiler_command: str | None = None
    profiler_output_format: str | None = None
    profiler_trace_dir: str | None = None
    profiler_trace_files: list[str] = []
    profiler_trace_kernel_names: list[str] = []
    profiler_trace_kernel_names_from_csv: list[str] = []
    profiler_trace_kernel_durations_from_csv: dict[str, float] = {}
    profiler_trace_synthesized_fields: list[str] = []
    profiler_source_artifact_path: str | None = None
    trace_kernel_names_valid = False
    if not isinstance(profiler, dict):
        reasons.append("profiler summary root is not an object")
    else:
        raw_profiler_artifact_path = profiler.get("artifact_path")
        if raw_profiler_artifact_path != str(profiler_path):
            reasons.append("artifact_path does not match --profiler-json path")
        elif isinstance(raw_profiler_artifact_path, str):
            profiler_source_artifact_path = raw_profiler_artifact_path
        raw_rows = profiler.get("rows")
        if raw_rows is None:
            workload = profiler.get("workload")
            if not isinstance(workload, dict) and isinstance(payload, dict):
                workload = payload.get("workload")
            if isinstance(workload, dict):
                raw_rows = workload.get("concurrency")
        if raw_rows != command.batch_size:
            reasons.append(f"rows={raw_rows!r} does not match batch_size={command.batch_size}")
        raw_prompt_tokens = _reference_label(profiler, "prompt_tokens_per_request", "prompt_length")
        if raw_prompt_tokens is None and isinstance(payload, dict):
            raw_prompt_tokens = _reference_label(payload, "prompt_tokens_per_request", "prompt_length")
        expected_prompt_length = _command_arg_int(command, "--prompt-length")
        if not isinstance(raw_prompt_tokens, int) or isinstance(raw_prompt_tokens, bool):
            reasons.append("prompt token count label is missing")
        elif expected_prompt_length is not None and raw_prompt_tokens != expected_prompt_length:
            reasons.append(f"prompt_tokens_per_request={raw_prompt_tokens!r} does not match prompt_length={expected_prompt_length}")
        raw_gen_tokens = _reference_label(profiler, "gen_tokens_per_request", "decode_tokens")
        if raw_gen_tokens is None and isinstance(payload, dict):
            raw_gen_tokens = _reference_label(payload, "gen_tokens_per_request", "decode_tokens")
        expected_decode_tokens = _command_arg_int(command, "--decode-tokens")
        if not isinstance(raw_gen_tokens, int) or isinstance(raw_gen_tokens, bool):
            reasons.append("decode token count label is missing")
        elif expected_decode_tokens is not None and raw_gen_tokens != expected_decode_tokens:
            reasons.append(f"gen_tokens_per_request={raw_gen_tokens!r} does not match decode_tokens={expected_decode_tokens}")
        raw_trace_dir = profiler.get("trace_dir")
        if isinstance(raw_trace_dir, str) and raw_trace_dir:
            profiler_trace_dir = raw_trace_dir
            trace_dir_check_path = Path(raw_trace_dir)
            if not trace_dir_check_path.is_absolute():
                trace_dir_check_path = REPO_ROOT / trace_dir_check_path
            trace_dir_has_path_error = False
            if _path_has_parent_directory_component(raw_trace_dir):
                reasons.append("profiler.trace_dir contains parent-directory components")
                trace_dir_has_path_error = True
            if trace_dir_check_path.is_symlink():
                reasons.append("profiler.trace_dir is a symlink")
                trace_dir_has_path_error = True
            if _path_has_symlink_parent(trace_dir_check_path):
                reasons.append("profiler.trace_dir parent directories contain symlinks")
                trace_dir_has_path_error = True
            if _path_has_non_directory_parent(trace_dir_check_path):
                reasons.append("profiler.trace_dir parent directories contain non-directories")
                trace_dir_has_path_error = True
            if not trace_dir_has_path_error:
                if not trace_dir_check_path.exists():
                    reasons.append("profiler.trace_dir does not exist")
                elif not trace_dir_check_path.is_dir():
                    reasons.append("profiler.trace_dir is not a directory")
        raw_trace_files = profiler.get("trace_files")
        if not isinstance(raw_trace_files, list) or not raw_trace_files:
            reasons.append("profiler.trace_files is missing or empty")
        elif not all(isinstance(trace_file, str) and trace_file for trace_file in raw_trace_files):
            reasons.append("profiler.trace_files contains a non-string entry")
        else:
            profiler_trace_files = list(raw_trace_files)
            trace_files_have_path_error = False
            if len(set(profiler_trace_files)) != len(profiler_trace_files):
                reasons.append("profiler.trace_files contains duplicates")
                trace_files_have_path_error = True
            if any(Path(trace_file).suffix.lower() != ".csv" for trace_file in profiler_trace_files):
                reasons.append("profiler.trace_files contains a non-CSV trace file")
                trace_files_have_path_error = True
            if not trace_files_have_path_error:
                kernel_trace_csv_count = sum(1 for trace_file in profiler_trace_files if _is_kernel_trace_csv_path(trace_file))
                if kernel_trace_csv_count == 0:
                    reasons.append("profiler.trace_files does not include a kernel-trace CSV")
                    trace_files_have_path_error = True
                elif kernel_trace_csv_count > 1:
                    reasons.append("profiler.trace_files must include exactly one kernel-trace CSV")
                    trace_files_have_path_error = True
            trace_file_check_paths = [Path(trace_file) for trace_file in profiler_trace_files]
            trace_file_check_paths = [
                trace_file_path if trace_file_path.is_absolute() else REPO_ROOT / trace_file_path
                for trace_file_path in trace_file_check_paths
            ]
            if any(trace_file_path.is_symlink() for trace_file_path in trace_file_check_paths):
                reasons.append("profiler.trace_files contains a symlink")
                trace_files_have_path_error = True
            if any(_path_has_symlink_parent(trace_file_path) for trace_file_path in trace_file_check_paths):
                reasons.append("profiler.trace_files parent directories contain symlinks")
                trace_files_have_path_error = True
            if any(_path_has_non_directory_parent(trace_file_path) for trace_file_path in trace_file_check_paths):
                reasons.append("profiler.trace_files parent directories contain non-directories")
                trace_files_have_path_error = True
            if any(_path_has_parent_directory_component(trace_file) for trace_file in profiler_trace_files):
                reasons.append("profiler.trace_files contains parent-directory components")
                trace_files_have_path_error = True
            elif profiler_trace_dir is not None:
                for trace_file in profiler_trace_files:
                    if not _is_resolved_path_relative_to(trace_file, profiler_trace_dir):
                        reasons.append("profiler.trace_files contains a path outside profiler.trace_dir")
                        trace_files_have_path_error = True
                        break
            if not trace_files_have_path_error:
                if any(not trace_file_path.exists() for trace_file_path in trace_file_check_paths):
                    reasons.append("profiler.trace_files contains a missing file")
                elif any(not trace_file_path.is_file() for trace_file_path in trace_file_check_paths):
                    reasons.append("profiler.trace_files contains a non-file path")
                else:
                    kernel_trace_file_paths = [
                        trace_file_path
                        for trace_file, trace_file_path in zip(profiler_trace_files, trace_file_check_paths)
                        if _is_kernel_trace_csv_path(trace_file)
                    ]
                    kernel_trace_names: list[str] = []
                    seen_kernel_trace_names: set[str] = set()
                    for trace_file_path in kernel_trace_file_paths:
                        for kernel_name in _read_profiler_trace_kernel_names(trace_file_path):
                            if kernel_name not in seen_kernel_trace_names:
                                kernel_trace_names.append(kernel_name)
                                seen_kernel_trace_names.add(kernel_name)
                    if not kernel_trace_names:
                        reasons.append("profiler.trace_files contain no readable kernel trace rows")
                    else:
                        profiler_trace_kernel_names_from_csv = kernel_trace_names
                        for trace_file_path in kernel_trace_file_paths:
                            for kernel_name, duration_ns in _read_profiler_trace_kernel_durations(trace_file_path).items():
                                profiler_trace_kernel_durations_from_csv[kernel_name] = (
                                    profiler_trace_kernel_durations_from_csv.get(kernel_name, 0.0) + duration_ns
                                )
        profiler_trace_synthesized_fields = _synthesize_profiler_trace_fields(profiler, profiler_path=profiler_path)
        raw_trace_kernel_names = profiler.get("trace_kernel_names")
        if not isinstance(raw_trace_kernel_names, list) or not raw_trace_kernel_names:
            reasons.append("profiler.trace_kernel_names is missing or empty")
        elif not all(_is_stripped_non_empty_string(kernel_name) for kernel_name in raw_trace_kernel_names):
            reasons.append("profiler.trace_kernel_names contains a non-string entry")
        else:
            profiler_trace_kernel_names = list(raw_trace_kernel_names)
            trace_kernel_names_valid = True
            if len(set(profiler_trace_kernel_names)) != len(profiler_trace_kernel_names):
                reasons.append("profiler.trace_kernel_names contains duplicates")
            if not any("batch" in kernel_name.lower() for kernel_name in profiler_trace_kernel_names):
                reasons.append("profiler.trace_kernel_names does not include a native batch kernel")
            if any(_has_disallowed_profiler_kernel_fragment(kernel_name) for kernel_name in profiler_trace_kernel_names):
                reasons.append("profiler.trace_kernel_names contains a serial/per-row/fallback kernel")
            if profiler_trace_kernel_names_from_csv and set(profiler_trace_kernel_names) != set(profiler_trace_kernel_names_from_csv):
                reasons.append("profiler.trace_kernel_names must match kernel-trace CSV rows")
        profiler_command = _profiler_command_label(profiler, payload if isinstance(payload, dict) else None)
        if profiler_command is None:
            reasons.append("profiler command is missing")
        else:
            if _ROCPROF_EXECUTABLE not in profiler_command or _ROCPROF_COMMAND_FLAGS[0] not in profiler_command:
                reasons.append("profiler command does not include rocprofv3 --kernel-trace")
            command_output_format = _command_text_arg(profiler_command, _ROCPROF_COMMAND_FLAGS[1])
            if command_output_format != _ROCPROF_OUTPUT_FORMAT:
                reasons.append(f"profiler command output-format={command_output_format!r} does not match {_ROCPROF_OUTPUT_FORMAT!r}")
            command_trace_dir = _command_text_arg(profiler_command, _ROCPROF_COMMAND_FLAGS[2])
            if command_trace_dir is None:
                reasons.append("profiler command is missing -d <trace_dir>")
            elif profiler_trace_dir is not None and command_trace_dir != profiler_trace_dir:
                reasons.append(f"profiler command trace-dir={command_trace_dir!r} does not match profiler.trace_dir={profiler_trace_dir}")
            if _RETAINED_BENCH_SCRIPT not in profiler_command:
                reasons.append("profiler command does not target qwen35_batch_retained_bench.py")
            retained_env = _command_device_env_assignments(command.argv)
            profiler_profiled_argv = _profiled_command_argv(profiler_command)
            profiler_env = _command_device_env_assignments(profiler_profiled_argv or ())
            if any(not value.strip() for value in (*retained_env.values(), *profiler_env.values())):
                reasons.append("profiler command device env prefix values must be non-blank")
            elif retained_env != profiler_env:
                reasons.append("profiler command device env prefix does not match retained command")
            for flag in _RETAINED_PROFILED_COMMAND_DISALLOWED_FLAGS:
                if _command_text_has_flag(profiler_command, flag):
                    reasons.append(f"profiler command must not include {flag}")
            expected_model = _command_arg_value(command, "--model")
            command_model = _command_text_arg(profiler_command, "--model")
            if expected_model is not None and command_model != expected_model:
                reasons.append(f"profiler command model={command_model!r} does not match model={expected_model}")
            expected_fixture = _command_arg_value(command, "--fixture")
            command_fixture = _command_text_arg(profiler_command, "--fixture")
            if expected_fixture is not None and command_fixture != expected_fixture:
                reasons.append(f"profiler command fixture={command_fixture!r} does not match fixture={expected_fixture}")
            command_output_path = _command_text_arg(profiler_command, "--json")
            if command_output_path != str(command.artifact_path):
                reasons.append("profiler command --json path does not match retained artifact_path")
            command_profiler_path = _command_text_arg(profiler_command, _RETAINED_GATE_FLAGS[3])
            if command_profiler_path != str(profiler_path):
                reasons.append("profiler command --profiler-json path does not match artifact_path")
            for flag, label in zip(_RETAINED_GATE_FLAGS[:3], _RETAINED_GATE_LABELS[:3]):
                expected_reference_path = _command_arg_value(command, flag)
                command_reference_path = _command_text_arg(profiler_command, flag)
                if expected_reference_path is not None and command_reference_path != expected_reference_path:
                    reasons.append(
                        f"profiler command {flag}={command_reference_path!r} does not match {label}={expected_reference_path}"
                    )
            command_batch_size = _command_text_arg(profiler_command, "--batch-size")
            if command_batch_size != str(command.batch_size):
                reasons.append(f"profiler command batch-size={command_batch_size!r} does not match batch_size={command.batch_size}")
            command_prompt_length = _command_text_arg(profiler_command, "--prompt-length")
            if expected_prompt_length is not None and command_prompt_length != str(expected_prompt_length):
                reasons.append(
                    f"profiler command prompt-length={command_prompt_length!r} does not match prompt_length={expected_prompt_length}"
                )
            command_decode_tokens = _command_text_arg(profiler_command, "--decode-tokens")
            if expected_decode_tokens is not None and command_decode_tokens != str(expected_decode_tokens):
                reasons.append(
                    f"profiler command decode-tokens={command_decode_tokens!r} does not match decode_tokens={expected_decode_tokens}"
                )
            expected_warmup_decode_tokens = _command_arg_int(command, "--warmup-decode-tokens")
            command_warmup_decode_tokens = _command_text_arg(profiler_command, "--warmup-decode-tokens")
            if expected_warmup_decode_tokens is not None and command_warmup_decode_tokens != str(expected_warmup_decode_tokens):
                reasons.append(
                    "profiler command warmup-decode-tokens="
                    f"{command_warmup_decode_tokens!r} does not match warmup_decode_tokens={expected_warmup_decode_tokens}"
                )
            expected_max_layers = _command_arg_int(command, "--max-layers")
            command_max_layers = _command_text_arg(profiler_command, "--max-layers")
            if expected_max_layers is not None and command_max_layers != str(expected_max_layers):
                reasons.append(f"profiler command max-layers={command_max_layers!r} does not match max_layers={expected_max_layers}")
            expected_compiler_version_file = _command_arg_value(command, "--compiler-version-file")
            if expected_compiler_version_file is not None:
                command_compiler_version_file = _command_text_arg(profiler_command, "--compiler-version-file")
                if command_compiler_version_file != expected_compiler_version_file:
                    reasons.append(
                        "profiler command compiler-version-file="
                        f"{command_compiler_version_file!r} does not match compiler_version_file={expected_compiler_version_file}"
                    )
            if "--require-cached-build" in command.argv and not _command_text_has_flag(
                profiler_command,
                "--require-cached-build",
            ):
                reasons.append("profiler command is missing --require-cached-build")
            expected_sample_mode = _command_arg_value(command, "--batch-sample-mode")
            if expected_sample_mode is not None:
                command_sample_mode = _command_text_arg(profiler_command, "--batch-sample-mode")
                if command_sample_mode != expected_sample_mode:
                    reasons.append("profiler command --batch-sample-mode must match retained command")
                if expected_sample_mode == "batched_lm_head":
                    if "--batch-sample-eq-ok" in command.argv and not _command_text_has_flag(profiler_command, "--batch-sample-eq-ok"):
                        reasons.append("profiler command is missing --batch-sample-eq-ok")
                    expected_sample_artifact = _command_arg_value(command, "--batch-sample-eq-artifact")
                    command_sample_artifact = _command_text_arg(profiler_command, "--batch-sample-eq-artifact")
                    if expected_sample_artifact is not None and command_sample_artifact != expected_sample_artifact:
                        reasons.append("profiler command --batch-sample-eq-artifact must match retained command")
                    expected_sample_rows = _command_arg_value(command, "--batch-sample-eq-rows")
                    command_sample_rows = _command_text_arg(profiler_command, "--batch-sample-eq-rows")
                    if expected_sample_rows is not None and command_sample_rows != expected_sample_rows:
                        reasons.append("profiler command --batch-sample-eq-rows must match retained command")
        if profiler.get("status") != "captured":
            reasons.append("status is not 'captured'")
        raw_output_format = profiler.get("output_format")
        if isinstance(raw_output_format, str):
            profiler_output_format = raw_output_format
        if profiler_output_format != _ROCPROF_OUTPUT_FORMAT:
            reasons.append(f"profiler.output_format={profiler_output_format!r} does not match {_ROCPROF_OUTPUT_FORMAT!r}")
        if profiler_trace_dir is None:
            reasons.append("profiler.trace_dir is missing")
        if profiler.get("expected_kernels_present") is not True:
            reasons.append("expected_kernels_present is not true")
        expected_kernel_names = profiler.get("expected_kernel_names")
        if not isinstance(expected_kernel_names, list) or not expected_kernel_names:
            reasons.append("expected_kernel_names is missing or empty")
        elif not all(_is_stripped_non_empty_string(name) for name in expected_kernel_names):
            reasons.append("expected_kernel_names contains a non-string entry")
        elif len(set(expected_kernel_names)) != len(expected_kernel_names):
            reasons.append("expected_kernel_names contains duplicates")
        elif not any("batch" in name.lower() for name in expected_kernel_names):
            reasons.append("expected_kernel_names does not include a native batch kernel")
        elif any(_has_disallowed_profiler_kernel_fragment(name) for name in expected_kernel_names):
            reasons.append("expected_kernel_names contains a serial/per-row/fallback kernel")
        kernel_durations = profiler.get("kernel_durations_ns")
        if not isinstance(kernel_durations, dict) or not kernel_durations:
            reasons.append("kernel_durations_ns is missing or empty")
        else:
            if any(
                isinstance(kernel_name, str) and _has_disallowed_profiler_kernel_fragment(kernel_name)
                for kernel_name in kernel_durations
            ):
                reasons.append("kernel_durations_ns contains a serial/per-row/fallback kernel")
            if isinstance(expected_kernel_names, list):
                for kernel_name in expected_kernel_names:
                    duration_ns = kernel_durations.get(kernel_name)
                    if _is_stripped_non_empty_string(kernel_name) and (
                        not _is_number(duration_ns) or float(duration_ns) <= 0.0
                    ):
                        reasons.append(f"kernel_durations_ns.{kernel_name} is missing or non-positive numeric")
                        break
            if trace_kernel_names_valid:
                missing_duration_names = sorted(
                    kernel_name
                    for kernel_name in kernel_durations
                    if _is_stripped_non_empty_string(kernel_name)
                    and not _has_disallowed_profiler_kernel_fragment(kernel_name)
                    and kernel_name not in profiler_trace_kernel_names
                )
                if missing_duration_names:
                    reasons.append("profiler.trace_kernel_names must include kernel_durations_ns keys")
                unmeasured_trace_names = sorted(
                    kernel_name
                    for kernel_name in profiler_trace_kernel_names
                    if _is_stripped_non_empty_string(kernel_name)
                    and not _has_disallowed_profiler_kernel_fragment(kernel_name)
                    and kernel_name not in kernel_durations
                )
                if unmeasured_trace_names:
                    reasons.append("profiler.kernel_durations_ns keys must include trace_kernel_names")
                duration_key_set = {
                    kernel_name
                    for kernel_name, duration_ns in kernel_durations.items()
                    if _is_stripped_non_empty_string(kernel_name) and _is_positive_finite_number(duration_ns)
                }
                if profiler_trace_kernel_durations_from_csv and duration_key_set == set(profiler_trace_kernel_durations_from_csv):
                    for kernel_name, trace_duration_ns in profiler_trace_kernel_durations_from_csv.items():
                        duration_ns = float(kernel_durations[kernel_name])
                        tolerance = max(1.0, abs(trace_duration_ns) * 1e-6)
                        if abs(duration_ns - trace_duration_ns) > tolerance:
                            reasons.append("profiler.kernel_durations_ns must match kernel-trace CSV durations")
                            break
        _validate_profiler_kernel_durations(profiler, reasons)
        _validate_profiler_kernel_duration_categories(profiler, reasons)
        _validate_profiler_cpu_side_bottlenecks(profiler, reasons)
    result: dict[str, Any] = {
        "kind": _RETAINED_PRECONDITION_KINDS[3],
        "artifact_path": str(profiler_path),
        "passed": not reasons,
        "reason": None if not reasons else "; ".join(reasons),
    }
    if not reasons and isinstance(profiler, dict):
        kernel_durations = profiler["kernel_durations_ns"]
        result.update(
            {
                "profiler_status": str(profiler["status"]),
                "profiler_source_artifact_path": profiler_source_artifact_path,
                "profiler_command": profiler_command,
                "profiler_output_format": str(profiler["output_format"]),
                "profiler_trace_dir": str(profiler["trace_dir"]),
                "profiler_trace_files": list(profiler_trace_files),
                "profiler_trace_kernel_names": list(profiler_trace_kernel_names),
                "profiler_trace_synthesized_fields": list(profiler_trace_synthesized_fields),
                "retained_artifact_path": str(command.artifact_path),
                "c1_baseline_artifact_path": _command_arg_value(command, _RETAINED_GATE_FLAGS[0]),
                "serial_bridge_artifact_path": _command_arg_value(command, _RETAINED_GATE_FLAGS[1]),
                "primitive_correctness_artifact_path": _command_arg_value(command, _RETAINED_GATE_FLAGS[2]),
                "profiler_compiler_version_file": _command_text_arg(profiler_command, "--compiler-version-file"),
                "profiler_require_cached_build": _command_text_has_flag(profiler_command, "--require-cached-build"),
                "profiler_model": _command_text_arg(profiler_command, "--model"),
                "profiler_fixture": _command_text_arg(profiler_command, "--fixture"),
                "profiler_warmup_decode_tokens": int(_command_text_arg(profiler_command, "--warmup-decode-tokens")),
                "profiler_max_layers": int(_command_text_arg(profiler_command, "--max-layers")),
                "workload_concurrency": int(raw_rows),
                "prompt_tokens_per_request": int(raw_prompt_tokens),
                "gen_tokens_per_request": int(raw_gen_tokens),
                "expected_kernel_names": list(profiler["expected_kernel_names"]),
                "kernel_durations_ns": {
                    kernel_name: float(duration_ns)
                    for kernel_name, duration_ns in kernel_durations.items()
                    if isinstance(kernel_name, str) and kernel_name
                },
                "total_kernel_duration_ns": float(profiler["total_kernel_duration_ns"]),
                "kernel_duration_shares": {
                    kernel_name: float(profiler["kernel_duration_shares"][kernel_name])
                    for kernel_name in kernel_durations
                    if isinstance(kernel_name, str) and kernel_name
                },
                "kernel_duration_categories_ns": {
                    category: float(profiler["kernel_duration_categories_ns"][category])
                    for category in _PROFILER_KERNEL_DURATION_CATEGORIES
                },
                "kernel_duration_category_shares": {
                    category: float(profiler["kernel_duration_category_shares"][category])
                    for category in _PROFILER_KERNEL_DURATION_CATEGORIES
                },
                "cpu_side_total_seconds": float(profiler["cpu_side_total_seconds"]),
                "cpu_side_bottlenecks_seconds": {
                    category: float(profiler["cpu_side_bottlenecks_seconds"][category])
                    for category in _PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES
                },
                "cpu_side_bottleneck_shares": {
                    category: float(profiler["cpu_side_bottleneck_shares"][category])
                    for category in _PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES
                },
            }
        )
    return result


def _native_retained_preconditions(command: SweepCommand) -> tuple[dict[str, Any], ...] | None:
    if command.category != _NATIVE_DIAGNOSTIC_COMMAND_CATEGORY or command.batch_size <= 1:
        return None
    return (
        _primitive_correctness_precondition(command),
        _scaling_reference_precondition(
            command,
            flag=_RETAINED_GATE_FLAGS[0],
            kind=_RETAINED_PRECONDITION_KINDS[1],
            expected_concurrency=1,
        ),
        _scaling_reference_precondition(
            command,
            flag=_RETAINED_GATE_FLAGS[1],
            kind=_RETAINED_PRECONDITION_KINDS[2],
            expected_concurrency=command.batch_size,
        ),
        _profiler_summary_precondition(command),
    )


def _first_failed_precondition(preconditions: Sequence[dict[str, Any]] | None) -> dict[str, Any] | None:
    if preconditions is None:
        return None
    for precondition in preconditions:
        if not precondition["passed"]:
            return precondition
    return None


def _profiler_summary_precondition_record(preconditions: Sequence[dict[str, Any]] | None) -> dict[str, Any] | None:
    if preconditions is None:
        return None
    for precondition in preconditions:
        if precondition.get("kind") == _RETAINED_PRECONDITION_KINDS[3] and precondition.get("passed") is True:
            return precondition
    return None


def _retained_profiler_synthesis_postcondition(
    command: SweepCommand,
    preconditions: Sequence[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    profiler_precondition = _profiler_summary_precondition_record(preconditions)
    if command.category != _NATIVE_DIAGNOSTIC_COMMAND_CATEGORY or profiler_precondition is None:
        return None
    expected_source_artifact_path = profiler_precondition.get("profiler_source_artifact_path")
    result: dict[str, Any] = {
        "kind": _RETAINED_POSTCONDITION_KINDS[0],
        "artifact_path": str(command.artifact_path),
        "profiler_precondition_artifact_path": profiler_precondition.get("artifact_path"),
        "passed": False,
        "reason": None,
    }
    if isinstance(expected_source_artifact_path, str) and expected_source_artifact_path:
        result["profiler_precondition_source_artifact_path"] = expected_source_artifact_path
    if not command.artifact_path.exists():
        result["reason"] = "retained artifact was not written for profiler provenance cross-check"
        return result
    expected_fields = profiler_precondition.get("profiler_trace_synthesized_fields")
    if not isinstance(expected_fields, list) or not all(isinstance(field, str) for field in expected_fields):
        result["reason"] = "profiler precondition synthesized fields are missing or malformed"
        return result
    try:
        payload = _load_json_path(command.artifact_path)
    except Exception as exc:
        result["reason"] = f"retained artifact is invalid JSON: {type(exc).__name__}: {exc}"
        return result
    profiler = payload.get("profiler") if isinstance(payload, dict) else None
    if not isinstance(profiler, dict):
        result["reason"] = "retained artifact profiler object is missing"
        return result
    actual_fields = profiler.get("synthesized_fields")
    if not isinstance(actual_fields, list) or not all(isinstance(field, str) for field in actual_fields):
        result["reason"] = "retained artifact profiler.synthesized_fields is missing or malformed"
        return result
    if isinstance(expected_source_artifact_path, str) and expected_source_artifact_path:
        actual_source_artifact_path = profiler.get("source_artifact_path")
        if not isinstance(actual_source_artifact_path, str) or not actual_source_artifact_path:
            result["reason"] = "retained artifact profiler.source_artifact_path is missing or malformed"
            return result
        result["profiler_source_artifact_path"] = actual_source_artifact_path
        if actual_source_artifact_path != expected_source_artifact_path:
            result["reason"] = "retained artifact profiler.source_artifact_path does not match profiler precondition source path"
            return result
    result["profiler_synthesized_fields"] = list(actual_fields)
    result["profiler_precondition_synthesized_fields"] = list(expected_fields)
    if list(actual_fields) != list(expected_fields):
        result["reason"] = "retained artifact profiler.synthesized_fields does not match profiler precondition synthesized fields"
        return result
    result["passed"] = True
    return result


def _summary_json(summary: Mapping[str, Any]) -> str:
    return json.dumps(summary, indent=2, allow_nan=False)


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    _validate_run_options(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = build_sweep_commands(args)
    git = _git_state()
    entries: list[dict[str, Any]] = []
    for command in commands:
        entry: dict[str, Any] = {
            "category": command.category,
            "batch_size": command.batch_size,
            "command": command.command,
            "argv": list(command.argv),
            "artifact_path": str(command.artifact_path),
            "git_dirty": git["dirty"],
        }
        if args.dry_run:
            entry.update({"status": _PLANNED_COMMAND_STATUS, "returncode": None, "duration_seconds": 0.0})
        else:
            preconditions = _native_retained_preconditions(command)
            if preconditions is not None:
                entry["preconditions"] = list(preconditions)
            precondition = _first_failed_precondition(preconditions)
            if precondition is not None:
                entry.update(
                    {
                        "status": _SKIPPED_COMMAND_STATUS,
                        "returncode": None,
                        "duration_seconds": 0.0,
                        "precondition": precondition,
                        "output_tail": precondition["reason"],
                    }
                )
                entries.append(entry)
                if args.stop_on_failure:
                    break
                continue
            start = time.perf_counter()
            proc = subprocess.run(
                list(command.argv),
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            entry.update(
                {
                    "status": _PASSED_COMMAND_STATUS if proc.returncode == 0 else _FAILED_COMMAND_STATUS,
                    "returncode": proc.returncode,
                    "duration_seconds": time.perf_counter() - start,
                    "output_tail": proc.stdout[-_OUTPUT_TAIL_MAX_CHARS:],
                }
            )
            if entry["status"] == _PASSED_COMMAND_STATUS:
                postcondition = _retained_profiler_synthesis_postcondition(command, preconditions)
                if postcondition is not None:
                    entry["postconditions"] = [postcondition]
                    if postcondition["passed"] is not True:
                        entry["status"] = _FAILED_COMMAND_STATUS
                        entry["postcondition"] = postcondition
                        entry["output_tail"] = str(postcondition["reason"])
        entries.append(entry)
        if entry["status"] == _FAILED_COMMAND_STATUS and args.stop_on_failure:
            break

    summary = {
        "schema": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "batch_sizes": list(args.batch_sizes),
        "output_dir": str(output_dir),
        "options": _summary_options(args),
        "command_count": len(commands),
        "completed_command_count": len(entries),
        "git": git,
        "commands": entries,
        "status_counts": _status_counts(entries),
        "category_status_counts": _category_status_counts(entries),
        "retained_precondition_counts": _retained_precondition_counts(entries),
        "retained_postcondition_counts": _retained_postcondition_counts(entries),
        "skipped_preconditions": _skipped_preconditions(entries),
        "failed_postconditions": _failed_postconditions(entries),
        "status": _summary_status(entries),
    }
    validate_sweep_summary(summary)
    if args.summary_json is not None:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_summary_json(summary) + "\n")
    return summary


def validate_sweep_summary(summary: Mapping[str, Any]) -> None:
    if not isinstance(summary, Mapping):
        raise ValueError("invalid c-sweep summary: summary must be an object")
    errors: list[str] = []
    expected_summary_keys = {
        "schema",
        "timestamp",
        "dry_run",
        "batch_sizes",
        "output_dir",
        "options",
        "command_count",
        "completed_command_count",
        "git",
        "commands",
        "status_counts",
        "category_status_counts",
        "retained_precondition_counts",
        "retained_postcondition_counts",
        "skipped_preconditions",
        "failed_postconditions",
        "status",
    }
    if set(summary) != expected_summary_keys:
        errors.append("summary must contain exactly the c-sweep schema keys")
    schema = summary.get("schema")
    if not isinstance(schema, int) or isinstance(schema, bool) or schema != 1:
        errors.append("schema must be typed int 1")
    timestamp = summary.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        errors.append("timestamp must be a non-empty string")
    else:
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp)
        except ValueError:
            errors.append("timestamp must be ISO-8601 parseable")
        else:
            if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() is None:
                errors.append("timestamp must include timezone")
    status = summary.get("status")
    if not isinstance(status, str) or not status.strip():
        errors.append("status must be a non-empty string")
    elif status not in {"planned", "passed", "blocked", "failed"}:
        errors.append("status must be planned, passed, blocked, or failed")
    if not isinstance(summary.get("dry_run"), bool):
        errors.append("dry_run must be a bool")
    batch_sizes = summary.get("batch_sizes")
    if (
        not isinstance(batch_sizes, list)
        or not batch_sizes
        or not all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in batch_sizes)
        or len(set(batch_sizes)) != len(batch_sizes)
    ):
        errors.append("batch_sizes must be a non-empty unique positive-int list")
        batch_sizes = []
    output_dir_text = summary.get("output_dir")
    if not isinstance(output_dir_text, str) or not output_dir_text.strip():
        errors.append("output_dir must be a non-empty string")
    else:
        output_dir_path = Path(output_dir_text)
        output_dir_check_path = output_dir_path if output_dir_path.is_absolute() else REPO_ROOT / output_dir_path
        if _path_has_parent_directory_component(output_dir_text):
            errors.append("output_dir must not contain parent-directory components")
        elif output_dir_check_path.is_symlink():
            errors.append("output_dir must not be a symlink")
        elif _path_has_symlink_parent(output_dir_check_path):
            errors.append("output_dir parent directories must not be symlinks")
        elif _path_has_non_directory_parent(output_dir_check_path):
            errors.append("output_dir parent directories must be directories")
    options = summary.get("options")
    option_model: str | None = None
    option_fixture: str | None = None
    option_seed: int | None = None
    option_projection_dispatch_artifact: str | None = None
    option_batch_sample_mode: str | None = None
    option_batch_sample_eq_ok = False
    option_batch_sample_eq_artifact_template: str | None = None
    option_batch_sample_eq_rows: str | None = None
    option_shape_values: dict[str, int] = {}
    if not isinstance(options, Mapping):
        errors.append("options must be an object")
    else:
        expected_option_keys = {
            "model",
            "fixture",
            "prompt_length",
            "decode_tokens",
            "warmup_decode_tokens",
            "max_layers",
            "seed",
            "stop_on_failure",
            "include_int8",
            "include_gguf",
            "require_cached_build",
            "compiler_version_file",
            "projection_dispatch_artifact",
            "batch_sample_mode",
            "batch_sample_eq_ok",
            "batch_sample_eq_artifact_template",
            "batch_sample_eq_rows",
        }
        if set(options) != expected_option_keys:
            errors.append("options must contain exactly the c-sweep schema keys")
        for option in ("stop_on_failure", "include_int8", "include_gguf", "require_cached_build", "batch_sample_eq_ok"):
            if not isinstance(options.get(option), bool):
                errors.append(f"options.{option} must be a bool")
        if isinstance(options.get("batch_sample_eq_ok"), bool):
            option_batch_sample_eq_ok = bool(options.get("batch_sample_eq_ok"))
        for option in ("model", "fixture"):
            value = options.get(option)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"options.{option} must be a non-empty string")
            elif option == "model":
                option_model = value
            else:
                option_fixture = value
        seed_value = options.get("seed")
        if not isinstance(seed_value, int) or isinstance(seed_value, bool):
            errors.append("options.seed must be an int")
        elif seed_value != _REQUIRED_PRIMITIVE_CORRECTNESS_SEED:
            errors.append("options.seed must match required primitive correctness seed")
        else:
            option_seed = seed_value
        for option in ("prompt_length", "decode_tokens", "warmup_decode_tokens", "max_layers"):
            value = options.get(option)
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"options.{option} must be an int")
            elif option == "warmup_decode_tokens" and value < 0:
                errors.append("options.warmup_decode_tokens must be non-negative")
            elif option != "warmup_decode_tokens" and value <= 0:
                errors.append(f"options.{option} must be positive")
            else:
                option_shape_values[option] = value
        projection_dispatch_artifact = options.get("projection_dispatch_artifact")
        if projection_dispatch_artifact is not None and not isinstance(projection_dispatch_artifact, str):
            errors.append("options.projection_dispatch_artifact must be a string or null")
        elif isinstance(projection_dispatch_artifact, str):
            if not projection_dispatch_artifact.strip():
                errors.append("options.projection_dispatch_artifact must be a non-empty string or null")
            else:
                projection_path = Path(projection_dispatch_artifact)
                if projection_path.is_absolute() or len(projection_path.parts) < 3 or projection_path.parts[:2] != ("benchmarks", "results") or _path_has_parent_directory_component(projection_dispatch_artifact):
                    errors.append("options.projection_dispatch_artifact must be a relative path under benchmarks/results")
                else:
                    option_projection_dispatch_artifact = projection_dispatch_artifact
        batch_sample_mode = options.get("batch_sample_mode")
        if batch_sample_mode is not None and not isinstance(batch_sample_mode, str):
            errors.append("options.batch_sample_mode must be a string or null")
        elif isinstance(batch_sample_mode, str):
            if batch_sample_mode not in {"serial_lm_head", "batched_lm_head"}:
                errors.append("options.batch_sample_mode must be serial_lm_head, batched_lm_head, or null")
            else:
                option_batch_sample_mode = batch_sample_mode
        batch_sample_eq_artifact_template = options.get("batch_sample_eq_artifact_template")
        if batch_sample_eq_artifact_template is not None and not isinstance(batch_sample_eq_artifact_template, str):
            errors.append("options.batch_sample_eq_artifact_template must be a string or null")
        elif isinstance(batch_sample_eq_artifact_template, str):
            if not batch_sample_eq_artifact_template.strip():
                errors.append("options.batch_sample_eq_artifact_template must be a non-empty string or null")
            else:
                option_batch_sample_eq_artifact_template = batch_sample_eq_artifact_template
        batch_sample_eq_rows = options.get("batch_sample_eq_rows")
        if batch_sample_eq_rows is not None and not isinstance(batch_sample_eq_rows, str):
            errors.append("options.batch_sample_eq_rows must be a string or null")
        elif isinstance(batch_sample_eq_rows, str):
            if not batch_sample_eq_rows.strip():
                errors.append("options.batch_sample_eq_rows must be a non-empty string or null")
            else:
                option_batch_sample_eq_rows = batch_sample_eq_rows
        compiler_version_file = options.get("compiler_version_file")
        if compiler_version_file is not None and not isinstance(compiler_version_file, str):
            errors.append("options.compiler_version_file must be a string or null")
        elif isinstance(compiler_version_file, str):
            if not compiler_version_file.strip():
                errors.append("options.compiler_version_file must be a non-empty string or null")
            else:
                compiler_version_path = Path(compiler_version_file)
                compiler_version_check_path = compiler_version_path if compiler_version_path.is_absolute() else REPO_ROOT / compiler_version_path
                if _path_has_parent_directory_component(compiler_version_file):
                    errors.append("options.compiler_version_file must not contain parent-directory components")
                elif compiler_version_check_path.is_symlink():
                    errors.append("options.compiler_version_file must not be a symlink")
                elif _path_has_symlink_parent(compiler_version_check_path):
                    errors.append("options.compiler_version_file parent directories must not be symlinks")
                elif _path_has_non_directory_parent(compiler_version_check_path):
                    errors.append("options.compiler_version_file parent directories must be directories")
    commands = summary.get("commands")
    if not isinstance(commands, list):
        errors.append("commands must be a list")
        commands = []
    entries: list[dict[str, Any]] = []
    for command in commands:
        if isinstance(command, dict):
            entries.append(command)
        else:
            errors.append("commands entries must be objects")
            break
    if not entries:
        errors.append("commands must be a non-empty list")
    git = summary.get("git")
    git_dirty: bool | None = None
    if not isinstance(git, Mapping):
        errors.append("git must be an object")
    else:
        if set(git) != {"commit", "dirty", "status_short"}:
            errors.append("git must contain exactly the c-sweep provenance keys")
        if not isinstance(git.get("commit"), str) or not git.get("commit").strip():
            errors.append("git.commit must be a non-empty string")
        if not isinstance(git.get("dirty"), bool):
            errors.append("git.dirty must be a bool")
        else:
            git_dirty = bool(git["dirty"])
        status_short = git.get("status_short")
        if not isinstance(status_short, list) or not all(isinstance(item, str) and item.strip() for item in status_short):
            errors.append("git.status_short must be a non-empty string list")
        elif git_dirty is not None and git_dirty is not bool(status_short):
            errors.append("git.dirty must match bool(git.status_short)")
    expected_skipped_precondition_keys = {
        "category",
        "batch_size",
        "artifact_path",
        "kind",
        "precondition_artifact_path",
        "reason",
    }
    expected_failed_postcondition_keys = {
        "category",
        "batch_size",
        "artifact_path",
        "kind",
        "profiler_precondition_artifact_path",
        "reason",
    }
    skipped_preconditions = summary.get("skipped_preconditions")
    if not isinstance(skipped_preconditions, list):
        errors.append("skipped_preconditions must be a list")
    else:
        for skipped_precondition in skipped_preconditions:
            if not isinstance(skipped_precondition, Mapping) or set(skipped_precondition) != expected_skipped_precondition_keys:
                errors.append("skipped_preconditions[] must contain exactly skipped precondition rollup keys")
                break
    expected_skipped_preconditions = _skipped_preconditions(entries)
    if not _typed_json_like_matches(skipped_preconditions, expected_skipped_preconditions):
        errors.append("skipped_preconditions must match commands.preconditions")
    summary_failed_postconditions = summary.get("failed_postconditions")
    if not isinstance(summary_failed_postconditions, list):
        errors.append("failed_postconditions must be a list")
    else:
        for failed_postcondition in summary_failed_postconditions:
            if not isinstance(failed_postcondition, Mapping) or set(failed_postcondition) != expected_failed_postcondition_keys:
                errors.append("failed_postconditions[] must contain exactly failed postcondition rollup keys")
                break
    expected_command_keys = {
        "category",
        "batch_size",
        "command",
        "argv",
        "artifact_path",
        "git_dirty",
        "status",
        "returncode",
        "duration_seconds",
        "output_tail",
        "preconditions",
        "precondition",
        "postconditions",
        "postcondition",
    }
    expected_planned_command_keys = {
        "category",
        "batch_size",
        "command",
        "argv",
        "artifact_path",
        "git_dirty",
        "status",
        "returncode",
        "duration_seconds",
    }
    expected_skipped_command_keys = {
        "category",
        "batch_size",
        "command",
        "argv",
        "artifact_path",
        "git_dirty",
        "status",
        "returncode",
        "duration_seconds",
        "output_tail",
        "preconditions",
        "precondition",
    }
    expected_simple_executed_command_keys = {
        "category",
        "batch_size",
        "command",
        "argv",
        "artifact_path",
        "git_dirty",
        "status",
        "returncode",
        "duration_seconds",
        "output_tail",
    }
    expected_command_categories = {
        _PRIMITIVE_COMMAND_CATEGORY,
        _SERIAL_BRIDGE_COMMAND_CATEGORY,
        _NATIVE_DIAGNOSTIC_COMMAND_CATEGORY,
        _INT8_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY,
        _GGUF_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY,
    }
    expected_scripts_by_category = {
        _PRIMITIVE_COMMAND_CATEGORY: {_PRIMITIVE_CORRECTNESS_SCRIPT},
        _SERIAL_BRIDGE_COMMAND_CATEGORY: {_SERIAL_BRIDGE_SCRIPT},
        _NATIVE_DIAGNOSTIC_COMMAND_CATEGORY: {_LEGACY_NATIVE_BENCH_SCRIPT, _RETAINED_BENCH_SCRIPT},
        _INT8_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY: {_INT8_DIAGNOSTIC_SCRIPT},
        _GGUF_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY: {_GGUF_DIAGNOSTIC_SCRIPT},
    }
    expected_minimal_failed_condition_keys = {
        "kind",
        "artifact_path",
        "passed",
        "reason",
    }
    expected_passed_retained_postcondition_keys = {
        "kind",
        "artifact_path",
        "profiler_precondition_artifact_path",
        "passed",
        "reason",
        "profiler_precondition_source_artifact_path",
        "profiler_source_artifact_path",
        "profiler_synthesized_fields",
        "profiler_precondition_synthesized_fields",
    }
    expected_passed_primitive_precondition_keys = {
        "kind",
        "artifact_path",
        "passed",
        "reason",
        "primitive_schema",
        "primitive_artifact_path",
        "primitive_seed",
        "primitive_block_size",
        "primitive_max_context_len",
        "primitive_num_q_heads",
        "primitive_num_kv_heads",
        "primitive_head_dim",
        "primitive_context_lens",
        "primitive_rows",
        "primitive_device",
        "append_key_mismatch",
        "append_value_mismatch",
        "append_batch_aa_key_mismatch",
        "append_batch_aa_value_mismatch",
        "attn_batch_aa_max_abs",
        "aa_passed",
        "attn_batch_vs_c1_max_abs",
        "attn_batch_vs_numpy_max_abs",
    }
    expected_passed_primitive_source_keys = {
        "schema",
        "artifact_path",
        "seed",
        "block_size",
        "max_context_len",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "context_lens",
        "rows",
        "device",
        "append_key_mismatch",
        "append_value_mismatch",
        "append_batch_aa_key_mismatch",
        "append_batch_aa_value_mismatch",
        "attn_batch_aa_max_abs",
        "aa_passed",
        "attn_batch_vs_c1_max_abs",
        "attn_batch_vs_numpy_max_abs",
        "passed",
    }
    expected_passed_scaling_precondition_keys = {
        "kind",
        "artifact_path",
        "passed",
        "reason",
        "reference_artifact_path",
        "reference_status",
        "reference_reason",
        "benchmark_model",
        "benchmark_fixture",
        "workload_concurrency",
        "prompt_tokens_per_request",
        "gen_tokens_per_request",
        "decode_tok_s_aggregate",
        "decode_tok_s_per_request",
    }
    expected_passed_profiler_precondition_keys = {
        "kind",
        "artifact_path",
        "passed",
        "reason",
        "profiler_status",
        "profiler_source_artifact_path",
        "profiler_command",
        "profiler_output_format",
        "profiler_trace_dir",
        "profiler_trace_files",
        "profiler_trace_kernel_names",
        "profiler_trace_synthesized_fields",
        "retained_artifact_path",
        "c1_baseline_artifact_path",
        "serial_bridge_artifact_path",
        "primitive_correctness_artifact_path",
        "profiler_compiler_version_file",
        "profiler_require_cached_build",
        "profiler_model",
        "profiler_fixture",
        "profiler_warmup_decode_tokens",
        "profiler_max_layers",
        "workload_concurrency",
        "prompt_tokens_per_request",
        "gen_tokens_per_request",
        "expected_kernel_names",
        "kernel_durations_ns",
        "total_kernel_duration_ns",
        "kernel_duration_shares",
        "kernel_duration_categories_ns",
        "kernel_duration_category_shares",
        "cpu_side_total_seconds",
        "cpu_side_bottlenecks_seconds",
        "cpu_side_bottleneck_shares",
    }
    expected_condition_keys = {
        "kind",
        "artifact_path",
        "passed",
        "reason",
        "primitive_schema",
        "primitive_artifact_path",
        "primitive_seed",
        "primitive_block_size",
        "primitive_max_context_len",
        "primitive_num_q_heads",
        "primitive_num_kv_heads",
        "primitive_head_dim",
        "primitive_context_lens",
        "primitive_rows",
        "primitive_device",
        "append_key_mismatch",
        "append_value_mismatch",
        "append_batch_aa_key_mismatch",
        "append_batch_aa_value_mismatch",
        "attn_batch_aa_max_abs",
        "aa_passed",
        "attn_batch_vs_c1_max_abs",
        "attn_batch_vs_numpy_max_abs",
        "reference_artifact_path",
        "reference_status",
        "reference_reason",
        "benchmark_model",
        "benchmark_fixture",
        "workload_concurrency",
        "prompt_tokens_per_request",
        "gen_tokens_per_request",
        "decode_tok_s_aggregate",
        "decode_tok_s_per_request",
        "profiler_status",
        "profiler_source_artifact_path",
        "profiler_command",
        "profiler_output_format",
        "profiler_trace_dir",
        "profiler_trace_files",
        "profiler_trace_kernel_names",
        "profiler_trace_synthesized_fields",
        "retained_artifact_path",
        "c1_baseline_artifact_path",
        "serial_bridge_artifact_path",
        "primitive_correctness_artifact_path",
        "profiler_compiler_version_file",
        "profiler_require_cached_build",
        "profiler_model",
        "profiler_fixture",
        "profiler_warmup_decode_tokens",
        "profiler_max_layers",
        "expected_kernel_names",
        "kernel_durations_ns",
        "total_kernel_duration_ns",
        "kernel_duration_shares",
        "kernel_duration_categories_ns",
        "kernel_duration_category_shares",
        "cpu_side_total_seconds",
        "cpu_side_bottlenecks_seconds",
        "cpu_side_bottleneck_shares",
        "profiler_precondition_artifact_path",
        "profiler_precondition_source_artifact_path",
        "profiler_synthesized_fields",
        "profiler_precondition_synthesized_fields",
    }
    expected_command_device_env: dict[str, str] | None = None
    if entries:
        for entry in entries:
            if not set(entry).issubset(expected_command_keys):
                errors.append("commands[] must contain only c-sweep schema keys")
                break
            if not isinstance(entry.get("category"), str) or not entry.get("category").strip():
                errors.append("commands[].category must be a non-empty string")
                break
            if entry.get("category") not in expected_command_categories:
                errors.append("commands[].category must be a known c-sweep command category")
                break
            if not isinstance(entry.get("batch_size"), int) or isinstance(entry.get("batch_size"), bool) or entry.get("batch_size") <= 0:
                errors.append("commands[].batch_size must be a positive int")
                break
            if batch_sizes and entry.get("batch_size") not in batch_sizes:
                errors.append("commands[].batch_size must be listed in batch_sizes")
                break
            if not isinstance(entry.get("artifact_path"), str) or not entry.get("artifact_path").strip():
                errors.append("commands[].artifact_path must be a non-empty string")
                break
            if _path_has_parent_directory_component(entry["artifact_path"]):
                errors.append("commands[].artifact_path must not contain parent-directory components")
                break
            artifact_path_for_symlink_check = Path(entry["artifact_path"])
            artifact_path_for_symlink_check = (
                artifact_path_for_symlink_check if artifact_path_for_symlink_check.is_absolute() else REPO_ROOT / artifact_path_for_symlink_check
            )
            if artifact_path_for_symlink_check.is_symlink():
                errors.append("commands[].artifact_path must be a regular file, not a symlink")
                break
            if _path_has_symlink_parent(artifact_path_for_symlink_check):
                errors.append("commands[].artifact_path parent directories must not be symlinks")
                break
            if _path_has_non_directory_parent(artifact_path_for_symlink_check):
                errors.append("commands[].artifact_path parent directories must be directories")
                break
            if artifact_path_for_symlink_check.exists() and not artifact_path_for_symlink_check.is_file():
                errors.append("commands[].artifact_path must be a regular file when it already exists")
                break
            command_text = entry.get("command")
            if not isinstance(command_text, str) or not command_text.strip():
                errors.append("commands[].command must be a non-empty string")
                break
            argv = entry.get("argv")
            if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item.strip() for item in argv):
                errors.append("commands[].argv must be a non-empty string list")
                break
            if _empty_inline_flag_values(argv, _SWEEP_COMMAND_KNOWN_FLAGS):
                errors.append("commands[].argv flag values must be non-empty")
                break
            if command_text != shlex.join(argv):
                errors.append("commands[].command must match shlex.join(commands[].argv)")
                break
            command_device_env = _command_device_env_assignments(argv)
            if any(not value.strip() for value in command_device_env.values()):
                errors.append("commands[].argv device env prefix values must be non-blank")
                break
            if expected_command_device_env is None:
                expected_command_device_env = dict(command_device_env)
            elif command_device_env != expected_command_device_env:
                errors.append("commands[].argv device env prefix must match the first command")
                break
            launch_argv = _strip_command_env_prefix(argv)
            if len(launch_argv) < 2 or not _is_python_executable(launch_argv[0]):
                errors.append("commands[].argv must start with a python executable")
                break
            command_category = entry.get("category")
            expected_scripts = expected_scripts_by_category.get(command_category)
            if expected_scripts is not None and launch_argv[1] not in expected_scripts:
                errors.append("commands[].category must match commands[].argv script")
                break
            if command_category in {
                _SERIAL_BRIDGE_COMMAND_CATEGORY,
                _NATIVE_DIAGNOSTIC_COMMAND_CATEGORY,
                _INT8_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY,
            } and option_model is not None:
                if _argv_value(argv, "--model") != option_model:
                    errors.append("commands[].argv --model must match options.model")
                    break
            fixture_required = command_category in {_SERIAL_BRIDGE_COMMAND_CATEGORY, _INT8_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY} or (
                command_category == _NATIVE_DIAGNOSTIC_COMMAND_CATEGORY and entry.get("batch_size") != 1
            )
            if fixture_required and option_fixture is not None and _argv_value(argv, "--fixture") != option_fixture:
                errors.append("commands[].argv --fixture must match options.fixture")
                break
            if command_category == _GGUF_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY:
                if _argv_value(argv, "--fixture") != _GGUF_DIAGNOSTIC_FIXTURE:
                    errors.append("commands[].argv GGUF --fixture must match the template fixture")
                    break
                if _argv_value(argv, "--backend") != _GGUF_DIAGNOSTIC_BACKEND:
                    errors.append("commands[].argv GGUF --backend must match the template backend")
                    break
                if _argv_value(argv, "--quant") not in _GGUF_DIAGNOSTIC_QUANTS:
                    errors.append("commands[].argv GGUF --quant must be one of the template quants")
                    break
                if _argv_value(argv, "--max-new-tokens") != str(_GGUF_DIAGNOSTIC_MAX_NEW_TOKENS):
                    errors.append("commands[].argv GGUF --max-new-tokens must match the template decode length")
                    break
            if command_category == _INT8_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY and _duplicate_flags(argv, _INT8_DIAGNOSTIC_UNIQUE_FLAGS):
                errors.append("commands[].argv must not repeat INT8 diagnostic flags")
                break
            if command_category == _GGUF_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY and _duplicate_flags(argv, _GGUF_DIAGNOSTIC_UNIQUE_FLAGS):
                errors.append("commands[].argv must not repeat GGUF diagnostic flags")
                break
            duplicated_retained_flags = _duplicate_flags(
                argv,
                tuple(dict.fromkeys(_RETAINED_PROFILED_COMMAND_UNIQUE_FLAGS + _BATCH_SAMPLE_UNIQUE_FLAGS)),
            )
            if duplicated_retained_flags:
                errors.append("commands[].argv must not repeat retained benchmark flags")
                break
            projection_dispatch_arg = _argv_value(argv, "--projection-dispatch-artifact")
            projection_dispatch_flag_present = any(
                _flag_token_matches(token, "--projection-dispatch-artifact") for token in argv
            )
            if entry.get("category") == _NATIVE_DIAGNOSTIC_COMMAND_CATEGORY:
                if entry.get("batch_size") == 1:
                    if projection_dispatch_flag_present:
                        errors.append("commands[].argv c=1 native baseline must not include --projection-dispatch-artifact")
                        break
                elif option_projection_dispatch_artifact is None:
                    if projection_dispatch_flag_present:
                        errors.append("commands[].argv --projection-dispatch-artifact requires options.projection_dispatch_artifact")
                        break
                elif projection_dispatch_arg != option_projection_dispatch_artifact:
                    errors.append("commands[].argv --projection-dispatch-artifact must match options.projection_dispatch_artifact")
                    break
            elif projection_dispatch_flag_present:
                errors.append("commands[].argv --projection-dispatch-artifact is only valid for c>N retained native commands")
                break
            sample_flags_present = any(
                _flag_token_matches(token, flag) for token in argv for flag in _BATCH_SAMPLE_UNIQUE_FLAGS
            )
            sample_mode_arg = _argv_value(argv, "--batch-sample-mode")
            sample_eq_ok_present = any(_flag_token_matches(token, "--batch-sample-eq-ok") for token in argv)
            sample_eq_artifact_arg = _argv_value(argv, "--batch-sample-eq-artifact")
            sample_eq_rows_arg = _argv_value(argv, "--batch-sample-eq-rows")
            if entry.get("category") == _NATIVE_DIAGNOSTIC_COMMAND_CATEGORY:
                if entry.get("batch_size") == 1:
                    if sample_flags_present:
                        errors.append("commands[].argv c=1 native baseline must not include batch sampler evidence flags")
                        break
                elif option_batch_sample_mode is None:
                    if sample_flags_present:
                        errors.append("commands[].argv batch sampler evidence flags require options.batch_sample_mode")
                        break
                else:
                    if sample_mode_arg != option_batch_sample_mode:
                        errors.append("commands[].argv --batch-sample-mode must match options.batch_sample_mode")
                        break
                    if sample_eq_ok_present != option_batch_sample_eq_ok:
                        errors.append("commands[].argv --batch-sample-eq-ok must match options.batch_sample_eq_ok")
                        break
                    if option_batch_sample_eq_artifact_template is None:
                        if sample_eq_artifact_arg is not None:
                            errors.append("commands[].argv --batch-sample-eq-artifact requires options.batch_sample_eq_artifact_template")
                            break
                    else:
                        try:
                            expected_sample_eq_artifact = _format_batch_template(
                                option_batch_sample_eq_artifact_template,
                                batch_size=int(entry["batch_size"]),
                                option="options.batch_sample_eq_artifact_template",
                            )
                        except ValueError as exc:
                            errors.append(str(exc))
                            break
                        sample_eq_artifact_path = Path(expected_sample_eq_artifact)
                        if (
                            sample_eq_artifact_path.is_absolute()
                            or len(sample_eq_artifact_path.parts) < 3
                            or sample_eq_artifact_path.parts[:2] != ("benchmarks", "results")
                            or _path_has_parent_directory_component(expected_sample_eq_artifact)
                        ):
                            errors.append("options.batch_sample_eq_artifact_template must expand to a relative path under benchmarks/results")
                            break
                        if sample_eq_artifact_arg != expected_sample_eq_artifact:
                            errors.append("commands[].argv --batch-sample-eq-artifact must match options.batch_sample_eq_artifact_template")
                            break
                    if option_batch_sample_eq_rows is None:
                        expected_sample_eq_rows = str(entry["batch_size"]) if option_batch_sample_eq_ok else None
                    else:
                        try:
                            expected_sample_eq_rows = _format_batch_template(
                                option_batch_sample_eq_rows,
                                batch_size=int(entry["batch_size"]),
                                option="options.batch_sample_eq_rows",
                            )
                        except ValueError as exc:
                            errors.append(str(exc))
                            break
                    if expected_sample_eq_rows is None:
                        if sample_eq_rows_arg is not None:
                            errors.append("commands[].argv --batch-sample-eq-rows requires options.batch_sample_eq_rows or options.batch_sample_eq_ok")
                            break
                    elif sample_eq_rows_arg != expected_sample_eq_rows:
                        errors.append("commands[].argv --batch-sample-eq-rows must match options.batch_sample_eq_rows/current batch size")
                        break
                    if sample_eq_rows_arg is not None:
                        try:
                            sample_eq_rows_int = int(sample_eq_rows_arg, 10)
                        except ValueError:
                            errors.append("commands[].argv --batch-sample-eq-rows must be an integer")
                            break
                        if sample_eq_rows_int <= 0:
                            errors.append("commands[].argv --batch-sample-eq-rows must be positive")
                            break
            elif sample_flags_present:
                errors.append("commands[].argv batch sampler evidence flags are only valid for c>N retained native commands")
                break
            try:
                json_path = argv[argv.index("--json") + 1]
            except (ValueError, IndexError):
                errors.append("commands[].argv must include --json <artifact_path>")
                break
            if json_path != entry.get("artifact_path"):
                errors.append("commands[].artifact_path must match commands[].argv --json")
                break
            output_dir_text = summary.get("output_dir")
            if isinstance(output_dir_text, str) and output_dir_text:
                artifact_path = Path(entry["artifact_path"])
                output_dir_path = Path(output_dir_text)
                artifact_abs = (artifact_path if artifact_path.is_absolute() else REPO_ROOT / artifact_path).resolve()
                output_dir_abs = (output_dir_path if output_dir_path.is_absolute() else REPO_ROOT / output_dir_path).resolve()
                if not artifact_abs.is_relative_to(output_dir_abs):
                    errors.append("commands[].artifact_path must be under output_dir")
                    break
                category = entry.get("category")
                batch_size = entry.get("batch_size")
                expected_artifact_name = None
                if category == _PRIMITIVE_COMMAND_CATEGORY:
                    expected_artifact_name = f"primitive-c{batch_size}.json"
                elif category == _SERIAL_BRIDGE_COMMAND_CATEGORY:
                    expected_artifact_name = f"serial-bridge-c{batch_size}.json"
                elif category == _NATIVE_DIAGNOSTIC_COMMAND_CATEGORY:
                    expected_artifact_name = "native-baseline-c1.json" if batch_size == 1 else f"native-diagnostic-c{batch_size}.json"
                elif category == _INT8_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY:
                    expected_artifact_name = f"int8-native-diagnostic-c{batch_size}.json"
                elif category == _GGUF_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY:
                    quant = _argv_value(argv, "--quant")
                    if quant in _GGUF_DIAGNOSTIC_QUANTS:
                        expected_artifact_name = f"gguf-native-diagnostic-c{batch_size}-{quant}.json"
                if expected_artifact_name is not None and artifact_abs != (output_dir_abs / expected_artifact_name).resolve():
                    errors.append("commands[].artifact_path must match category/batch-size filename")
                    break
                if category == _INT8_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY:
                    expected_int8_paths = {
                        "--future-json": output_dir_path / f"int8-native-retained-future-c{batch_size}.json",
                        "--primitive-cpu-json": output_dir_path / f"int8-primitive-cpu-c{batch_size}.json",
                        "--primitive-hip-json": output_dir_path / f"int8-primitive-hip-c{batch_size}.json",
                    }
                    int8_path_error = False
                    for flag, expected_path in expected_int8_paths.items():
                        if _argv_value(argv, flag) != str(expected_path):
                            errors.append(f"commands[].argv INT8 {flag} must match the c-specific output_dir artifact path")
                            int8_path_error = True
                            break
                    if int8_path_error:
                        break
            declared_batch_size: int | None = None
            batch_arg_error = False
            for batch_flag in ("--batch-size", "--rows"):
                if batch_flag not in argv:
                    continue
                try:
                    declared_batch_size = int(argv[argv.index(batch_flag) + 1])
                except (IndexError, ValueError):
                    errors.append(f"commands[].argv {batch_flag} must have an int value")
                    batch_arg_error = True
                    break
                break
            if batch_arg_error:
                break
            baseline_c1_native = (
                entry.get("category") == _NATIVE_DIAGNOSTIC_COMMAND_CATEGORY and entry.get("batch_size") == 1
            )
            if baseline_c1_native and declared_batch_size is not None:
                errors.append("commands[].argv c=1 native baseline must not include --batch-size or --rows")
                break
            if declared_batch_size is None and not baseline_c1_native:
                errors.append("commands[].argv must include --batch-size or --rows")
                break
            if declared_batch_size is not None and declared_batch_size != entry.get("batch_size"):
                errors.append("commands[].batch_size must match commands[].argv --batch-size/--rows")
                break
            if entry.get("category") in {
                _SERIAL_BRIDGE_COMMAND_CATEGORY,
                _NATIVE_DIAGNOSTIC_COMMAND_CATEGORY,
                _INT8_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY,
            }:
                shape_arg_error = False
                for shape_flag in ("--prompt-length", "--decode-tokens", "--warmup-decode-tokens", "--max-layers"):
                    try:
                        shape_value = int(argv[argv.index(shape_flag) + 1])
                    except (IndexError, ValueError):
                        errors.append(f"commands[].argv {shape_flag} must have an int value")
                        shape_arg_error = True
                        break
                    if shape_flag == "--warmup-decode-tokens":
                        if shape_value < 0:
                            errors.append(f"commands[].argv {shape_flag} must be non-negative")
                            shape_arg_error = True
                            break
                    elif shape_value <= 0:
                        errors.append(f"commands[].argv {shape_flag} must be positive")
                        shape_arg_error = True
                        break
                if shape_arg_error:
                    break
                shape_option_error = False
                for option, flag in (
                    ("prompt_length", "--prompt-length"),
                    ("decode_tokens", "--decode-tokens"),
                    ("warmup_decode_tokens", "--warmup-decode-tokens"),
                    ("max_layers", "--max-layers"),
                ):
                    expected_shape_value = option_shape_values.get(option)
                    if expected_shape_value is not None and _argv_value(argv, flag) != str(expected_shape_value):
                        errors.append(f"commands[].argv {flag} must match options.{option}")
                        shape_option_error = True
                        break
                if shape_option_error:
                    break
            if entry.get("category") == _PRIMITIVE_COMMAND_CATEGORY:
                duplicated_primitive_flags = _duplicate_flags(argv, _PRIMITIVE_CORRECTNESS_UNIQUE_FLAGS)
                if duplicated_primitive_flags:
                    errors.append("commands[].argv must not repeat primitive correctness flags")
                    break
                try:
                    primitive_rows = int(argv[argv.index("--rows") + 1])
                except (IndexError, ValueError):
                    errors.append("commands[].argv --rows must have an int value")
                    break
                if primitive_rows != entry.get("batch_size"):
                    errors.append("commands[].batch_size must match primitive --rows")
                    break
                try:
                    primitive_seed = int(argv[argv.index("--seed") + 1])
                except (IndexError, ValueError):
                    errors.append("commands[].argv --seed must have an int value")
                    break
                if primitive_seed != _REQUIRED_PRIMITIVE_CORRECTNESS_SEED:
                    errors.append("commands[].argv --seed must match required primitive correctness seed")
                    break
                if option_seed is not None and primitive_seed != option_seed:
                    errors.append("commands[].argv --seed must match options.seed")
                    break
            if entry.get("category") == _NATIVE_DIAGNOSTIC_COMMAND_CATEGORY and entry.get("batch_size") != 1:
                retained_gate_flags = _RETAINED_GATE_FLAGS
                if any(not isinstance(_argv_value(argv, flag), str) or not _argv_value(argv, flag) for flag in retained_gate_flags):
                    errors.append("commands[].argv must include retained native gate artifact flags")
                    break
                output_dir_text = summary.get("output_dir")
                if isinstance(output_dir_text, str) and output_dir_text:
                    output_dir_path = Path(output_dir_text)
                    output_dir_abs = (output_dir_path if output_dir_path.is_absolute() else REPO_ROOT / output_dir_path).resolve()
                    retained_gate_names = {
                        _RETAINED_GATE_FLAGS[0]: "native-baseline-c1.json",
                        _RETAINED_GATE_FLAGS[1]: f"serial-bridge-c{entry.get('batch_size')}.json",
                        _RETAINED_GATE_FLAGS[2]: f"primitive-c{entry.get('batch_size')}.json",
                        _RETAINED_GATE_FLAGS[3]: f"profiler-c{entry.get('batch_size')}.json",
                    }
                    retained_gate_path_error = False
                    for flag, expected_name in retained_gate_names.items():
                        gate_text = _argv_value(argv, flag)
                        if not isinstance(gate_text, str):
                            errors.append("commands[].argv must include retained native gate artifact flags")
                            retained_gate_path_error = True
                            break
                        if _path_has_parent_directory_component(gate_text):
                            errors.append("commands[].argv retained native gate artifact paths must not contain parent-directory components")
                            retained_gate_path_error = True
                            break
                        gate_path = Path(gate_text)
                        gate_check_path = gate_path if gate_path.is_absolute() else REPO_ROOT / gate_path
                        if gate_check_path.is_symlink():
                            errors.append("commands[].argv retained native gate artifact paths must not be symlinks")
                            retained_gate_path_error = True
                            break
                        if _path_has_symlink_parent(gate_check_path):
                            errors.append("commands[].argv retained native gate artifact path parent directories must not be symlinks")
                            retained_gate_path_error = True
                            break
                        if _path_has_non_directory_parent(gate_check_path):
                            errors.append("commands[].argv retained native gate artifact path parent directories must be directories")
                            retained_gate_path_error = True
                            break
                        gate_abs = gate_check_path.resolve()
                        if gate_abs != (output_dir_abs / expected_name).resolve():
                            errors.append("commands[].argv retained native gate artifact paths must match output_dir filenames")
                            retained_gate_path_error = True
                            break
                    if retained_gate_path_error:
                        break
            compiler_version_file = _argv_value(argv, "--compiler-version-file")
            if compiler_version_file is not None:
                if _path_has_parent_directory_component(compiler_version_file):
                    errors.append("commands[].argv compiler-version-file must not contain parent-directory components")
                    break
                compiler_version_path = Path(compiler_version_file)
                compiler_version_check_path = compiler_version_path if compiler_version_path.is_absolute() else REPO_ROOT / compiler_version_path
                if compiler_version_check_path.is_symlink():
                    errors.append("commands[].argv compiler-version-file must not be a symlink")
                    break
                if _path_has_symlink_parent(compiler_version_check_path):
                    errors.append("commands[].argv compiler-version-file parent directories must not be symlinks")
                    break
                if _path_has_non_directory_parent(compiler_version_check_path):
                    errors.append("commands[].argv compiler-version-file parent directories must be directories")
                    break
            if entry.get("category") in {_SERIAL_BRIDGE_COMMAND_CATEGORY, _NATIVE_DIAGNOSTIC_COMMAND_CATEGORY} and isinstance(options, Mapping):
                option_compiler_version_file = options.get("compiler_version_file")
                if option_compiler_version_file is None or isinstance(option_compiler_version_file, str):
                    if compiler_version_file != option_compiler_version_file:
                        errors.append("commands[].argv compiler-version-file must match options.compiler_version_file")
                        break
                option_require_cached_build = options.get("require_cached_build")
                if isinstance(option_require_cached_build, bool) and ("--require-cached-build" in argv) != option_require_cached_build:
                    errors.append("commands[].argv require-cached-build must match options.require_cached_build")
                    break
            status = entry.get("status")
            if status not in set(_SWEEP_COMMAND_STATUS_LABELS):
                errors.append("commands[].status must be planned, passed, skipped, or failed")
                break
            dry_run = summary.get("dry_run")
            if isinstance(dry_run, bool):
                if dry_run and status != _PLANNED_COMMAND_STATUS:
                    errors.append("commands[].status must be planned for dry-run summaries")
                    break
                if not dry_run and status == _PLANNED_COMMAND_STATUS:
                    errors.append("commands[].status cannot be planned for executed summaries")
                    break
            returncode = entry.get("returncode")
            if status in {_PLANNED_COMMAND_STATUS, _SKIPPED_COMMAND_STATUS}:
                if returncode is not None:
                    errors.append("commands[].returncode must be null for planned/skipped rows")
                    break
            elif not isinstance(returncode, int) or isinstance(returncode, bool):
                errors.append("commands[].returncode must be an int for passed/failed rows")
                break
            duration_seconds = entry.get("duration_seconds")
            if (
                not isinstance(duration_seconds, (int, float))
                or isinstance(duration_seconds, bool)
                or float(duration_seconds) < 0.0
            ):
                errors.append("commands[].duration_seconds must be a non-negative number")
                break
            if not math.isfinite(float(duration_seconds)):
                errors.append("commands[].duration_seconds must be finite")
                break
            if status == _PLANNED_COMMAND_STATUS and float(duration_seconds) != 0.0:
                errors.append("commands[].duration_seconds must be zero for planned rows")
                break
            if status == _SKIPPED_COMMAND_STATUS and float(duration_seconds) != 0.0:
                errors.append("commands[].duration_seconds must be zero for skipped rows")
                break
            if status == _PLANNED_COMMAND_STATUS and "output_tail" in entry:
                errors.append("commands[].output_tail must be absent for planned rows")
                break
            if status == _PLANNED_COMMAND_STATUS and any(
                field in entry for field in ("preconditions", "precondition", "postconditions", "postcondition")
            ):
                errors.append("commands[].conditions must be absent for planned rows")
                break
            if status == _PLANNED_COMMAND_STATUS and set(entry) != expected_planned_command_keys:
                errors.append("commands[] planned rows must contain exactly planned command keys")
                break
            if status != _PLANNED_COMMAND_STATUS and not isinstance(entry.get("output_tail"), str):
                errors.append("commands[].output_tail must be a string for non-planned rows")
                break
            if isinstance(entry.get("output_tail"), str) and len(entry["output_tail"]) > _OUTPUT_TAIL_MAX_CHARS:
                errors.append("commands[].output_tail must be no longer than 4000 characters")
                break
            condition_schema_error = False
            for condition_field in ("preconditions", "postconditions"):
                if condition_field not in entry:
                    continue
                conditions = entry[condition_field]
                if not isinstance(conditions, list):
                    errors.append(f"commands[].{condition_field} must be a list")
                    condition_schema_error = True
                    break
                for condition in conditions:
                    if not isinstance(condition, dict):
                        errors.append(f"commands[].{condition_field}[] must be an object")
                        condition_schema_error = True
                        break
                    if not set(condition).issubset(expected_condition_keys):
                        errors.append(f"commands[].{condition_field}[] must contain only c-sweep condition schema keys")
                        condition_schema_error = True
                        break
                    if not isinstance(condition.get("kind"), str) or not condition.get("kind").strip():
                        errors.append(f"commands[].{condition_field}[].kind must be a non-empty string")
                        condition_schema_error = True
                        break
                    if not isinstance(condition.get("passed"), bool):
                        errors.append(f"commands[].{condition_field}[].passed must be a bool")
                        condition_schema_error = True
                        break
                    if condition_field == "preconditions":
                        reason = condition.get("reason")
                        if condition.get("passed") is True and reason is not None:
                            errors.append("commands[].preconditions[].reason must be null when passed")
                            condition_schema_error = True
                            break
                        if condition.get("passed") is False and (not isinstance(reason, str) or not reason.strip()):
                            errors.append("commands[].preconditions[].reason must be a non-empty string when failed")
                            condition_schema_error = True
                            break
                        if condition.get("passed") is False and isinstance(reason, str):
                            has_minimal_failure_reason = (
                                reason.startswith("retained native diagnostic is missing ")
                                or reason.endswith(" artifact does not exist")
                                or " artifact is invalid JSON: " in reason
                            )
                            if has_minimal_failure_reason and set(condition) != expected_minimal_failed_condition_keys:
                                errors.append("commands[].preconditions[] minimal failed conditions must contain exactly generic failure keys")
                                condition_schema_error = True
                                break
                if condition_schema_error:
                    break
            if condition_schema_error:
                break
            if "preconditions" in entry and (entry.get("category") != _NATIVE_DIAGNOSTIC_COMMAND_CATEGORY or entry.get("batch_size") == 1):
                errors.append("commands[].preconditions are only valid for retained native diagnostic rows")
                break
            if status == _FAILED_COMMAND_STATUS and isinstance(returncode, int) and returncode != 0 and any(
                field in entry for field in ("postconditions", "postcondition")
            ):
                errors.append("commands[].postconditions must be absent for failed rows with nonzero returncode")
                break
            if status == _SKIPPED_COMMAND_STATUS and any(field in entry for field in ("postconditions", "postcondition")):
                errors.append("commands[].postconditions must be absent for skipped rows")
                break
            if "postconditions" in entry and (entry.get("category") != _NATIVE_DIAGNOSTIC_COMMAND_CATEGORY or entry.get("batch_size") == 1):
                errors.append("commands[].postconditions are only valid for retained native diagnostic rows")
                break
            if (
                entry.get("category") == _NATIVE_DIAGNOSTIC_COMMAND_CATEGORY
                and entry.get("batch_size") != 1
                and status == _PASSED_COMMAND_STATUS
                and "postconditions" not in entry
            ):
                errors.append("commands[].postconditions must include retained native postconditions for passed retained rows")
                break
            if isinstance(entry.get("postconditions"), list) and [
                condition.get("kind") for condition in entry["postconditions"]
            ] != list(_RETAINED_POSTCONDITION_KINDS):
                errors.append("commands[].postconditions must include retained native postcondition kinds")
                break
            if entry.get("category") == _NATIVE_DIAGNOSTIC_COMMAND_CATEGORY and entry.get("batch_size") != 1 and status != _PLANNED_COMMAND_STATUS:
                preconditions = entry.get("preconditions")
                expected_retained_kinds = list(_RETAINED_PRECONDITION_KINDS)
                if not isinstance(preconditions, list) or [condition.get("kind") for condition in preconditions] != expected_retained_kinds:
                    errors.append("commands[].preconditions must include retained native gate kinds")
                    break
                expected_retained_precondition_paths = [
                    _argv_value(argv, _RETAINED_GATE_FLAGS[2]),
                    _argv_value(argv, _RETAINED_GATE_FLAGS[0]),
                    _argv_value(argv, _RETAINED_GATE_FLAGS[1]),
                    _argv_value(argv, _RETAINED_GATE_FLAGS[3]),
                ]
                precondition_artifact_path_error = False
                for condition in preconditions:
                    precondition_artifact_path = condition.get("artifact_path")
                    if not isinstance(precondition_artifact_path, str) or not precondition_artifact_path:
                        errors.append("commands[].preconditions[].artifact_path must be a non-empty string")
                        precondition_artifact_path_error = True
                        break
                    if _path_has_parent_directory_component(precondition_artifact_path):
                        errors.append("commands[].preconditions[].artifact_path must not contain parent-directory components")
                        precondition_artifact_path_error = True
                        break
                    precondition_path = Path(precondition_artifact_path)
                    precondition_check_path = precondition_path if precondition_path.is_absolute() else REPO_ROOT / precondition_path
                    if precondition_check_path.is_symlink():
                        errors.append("commands[].preconditions[].artifact_path must not be a symlink")
                        precondition_artifact_path_error = True
                        break
                    if _path_has_symlink_parent(precondition_check_path):
                        errors.append("commands[].preconditions[].artifact_path parent directories must not be symlinks")
                        precondition_artifact_path_error = True
                        break
                    if _path_has_non_directory_parent(precondition_check_path):
                        errors.append("commands[].preconditions[].artifact_path parent directories must be directories")
                        precondition_artifact_path_error = True
                        break
                if precondition_artifact_path_error:
                    break
                if [condition.get("artifact_path") for condition in preconditions] != expected_retained_precondition_paths:
                    errors.append("commands[].preconditions[].artifact_path must match retained native gate argv")
                    break
                primitive_precondition = preconditions[0]
                if primitive_precondition.get("passed") is True:
                    primitive_schema = primitive_precondition.get("primitive_schema")
                    if (
                        not isinstance(primitive_schema, int)
                        or isinstance(primitive_schema, bool)
                        or primitive_schema != _REQUIRED_PRIMITIVE_CORRECTNESS_SCHEMA
                    ):
                        errors.append("commands[].preconditions[].primitive_schema must be typed int 1 when primitive passed")
                        break
                    primitive_artifact_path = primitive_precondition.get("primitive_artifact_path")
                    if not isinstance(primitive_artifact_path, str) or not primitive_artifact_path:
                        errors.append("commands[].preconditions[].primitive_artifact_path must be a non-empty string when primitive passed")
                        break
                    if not primitive_artifact_path.strip():
                        errors.append("commands[].preconditions[].primitive_artifact_path must be a non-blank string when primitive passed")
                        break
                    primitive_alias_path = Path(primitive_artifact_path)
                    primitive_alias_check_path = primitive_alias_path if primitive_alias_path.is_absolute() else REPO_ROOT / primitive_alias_path
                    if _path_has_parent_directory_component(primitive_artifact_path):
                        errors.append("commands[].preconditions[].primitive_artifact_path must not contain parent-directory components when primitive passed")
                        break
                    if primitive_alias_check_path.is_symlink():
                        errors.append("commands[].preconditions[].primitive_artifact_path must be a regular file, not a symlink, when primitive passed")
                        break
                    if _path_has_symlink_parent(primitive_alias_check_path):
                        errors.append("commands[].preconditions[].primitive_artifact_path parent directories must not be symlinks when primitive passed")
                        break
                    if _path_has_non_directory_parent(primitive_alias_check_path):
                        errors.append("commands[].preconditions[].primitive_artifact_path parent directories must be directories when primitive passed")
                        break
                    if primitive_alias_check_path.exists() and not primitive_alias_check_path.is_file():
                        errors.append("commands[].preconditions[].primitive_artifact_path must be a regular file when it already exists")
                        break
                    if primitive_artifact_path != primitive_precondition.get("artifact_path"):
                        errors.append("commands[].preconditions[].primitive_artifact_path must match primitive artifact_path when primitive passed")
                        break
                    primitive_source_payload: dict[str, Any] | None = None
                    if primitive_alias_check_path.exists() and primitive_alias_check_path.is_file():
                        try:
                            raw_primitive_source_payload = _load_json_path(primitive_alias_check_path)
                        except Exception:
                            errors.append("commands[].preconditions[].primitive_artifact_path must be valid JSON when primitive passed")
                            break
                        if not isinstance(raw_primitive_source_payload, dict):
                            errors.append("commands[].preconditions[].primitive_artifact_path must contain an object when primitive passed")
                            break
                        primitive_source_payload = raw_primitive_source_payload
                        primitive_source_keys = set(primitive_source_payload)
                        if primitive_source_keys not in (
                            expected_passed_primitive_source_keys,
                            expected_passed_primitive_source_keys | {"source_artifact_path"},
                        ):
                            errors.append("commands[].preconditions[].primitive_artifact_path JSON must contain exactly primitive correctness source keys when primitive passed")
                            break
                        primitive_source_artifact_path_field = primitive_source_payload.get("artifact_path")
                        if not isinstance(primitive_source_artifact_path_field, str) or not primitive_source_artifact_path_field:
                            errors.append("commands[].preconditions[].primitive_artifact_path JSON artifact_path must be a non-empty string when primitive passed")
                            break
                        if not primitive_source_artifact_path_field.strip():
                            errors.append("commands[].preconditions[].primitive_artifact_path JSON artifact_path must be a non-blank string when primitive passed")
                            break
                        if primitive_source_artifact_path_field != primitive_artifact_path:
                            errors.append("commands[].preconditions[].primitive_artifact_path JSON artifact_path must match when primitive passed")
                            break
                        if "source_artifact_path" in primitive_source_payload:
                            primitive_source_artifact_path = primitive_source_payload.get("source_artifact_path")
                            if not isinstance(primitive_source_artifact_path, str) or not primitive_source_artifact_path:
                                errors.append("commands[].preconditions[].primitive_artifact_path JSON source_artifact_path must be a non-empty string when primitive passed")
                                break
                            if not primitive_source_artifact_path.strip():
                                errors.append("commands[].preconditions[].primitive_artifact_path JSON source_artifact_path must be a non-blank string when primitive passed")
                                break
                            if primitive_source_artifact_path != primitive_artifact_path:
                                errors.append("commands[].preconditions[].primitive_artifact_path JSON source_artifact_path must match when primitive passed")
                                break
                    primitive_seed = primitive_precondition.get("primitive_seed")
                    if (
                        not isinstance(primitive_seed, int)
                        or isinstance(primitive_seed, bool)
                        or primitive_seed != _REQUIRED_PRIMITIVE_CORRECTNESS_SEED
                    ):
                        errors.append("commands[].preconditions[].primitive_seed must be typed int 1234 when primitive passed")
                        break
                    primitive_shape_error = False
                    for field, expected_value in _REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS.items():
                        value = primitive_precondition.get(f"primitive_{field}")
                        if not isinstance(value, int) or isinstance(value, bool) or value != expected_value:
                            errors.append(f"commands[].preconditions[].primitive_{field} must be a typed int matching fixture shape when primitive passed")
                            primitive_shape_error = True
                            break
                    if primitive_shape_error:
                        break
                    if not _primitive_context_lens_matches(
                        primitive_precondition.get("primitive_context_lens"),
                        int(entry.get("batch_size")),
                    ):
                        errors.append("commands[].preconditions[].primitive_context_lens must match fixture coverage when primitive passed")
                        break
                    primitive_rows = primitive_precondition.get("primitive_rows")
                    batch_size = entry.get("batch_size")
                    if not isinstance(primitive_rows, int) or isinstance(primitive_rows, bool) or primitive_rows != batch_size:
                        errors.append("commands[].preconditions[].primitive_rows must be a typed int matching retained batch_size")
                        break
                    primitive_device_blockers = _primitive_device_metadata_blockers(primitive_precondition.get("primitive_device"))
                    if primitive_device_blockers:
                        errors.append(
                            "commands[].preconditions[].primitive_device must contain valid device metadata when primitive passed: "
                            + "; ".join(primitive_device_blockers)
                        )
                        break
                    if not _is_zero_int(primitive_precondition.get("append_key_mismatch")) or not _is_zero_int(primitive_precondition.get("append_value_mismatch")):
                        errors.append("commands[].preconditions[].primitive append mismatches must be typed integer zeros when passed")
                        break
                    if not _is_zero_int(primitive_precondition.get("append_batch_aa_key_mismatch")) or not _is_zero_int(primitive_precondition.get("append_batch_aa_value_mismatch")):
                        errors.append("commands[].preconditions[].primitive append A/A mismatches must be typed integer zeros when passed")
                        break
                    if not _is_exact_zero_number(primitive_precondition.get("attn_batch_aa_max_abs")):
                        errors.append("commands[].preconditions[].attn_batch_aa_max_abs must be exactly 0.0 when primitive passed")
                        break
                    if primitive_precondition.get("aa_passed") is not True:
                        errors.append("commands[].preconditions[].aa_passed must be true when primitive passed")
                        break
                    attn_vs_c1 = primitive_precondition.get("attn_batch_vs_c1_max_abs")
                    if not _is_exact_zero_number(attn_vs_c1):
                        errors.append("commands[].preconditions[].attn_batch_vs_c1_max_abs must be exactly 0.0 when primitive passed")
                        break
                    attn_vs_numpy = primitive_precondition.get("attn_batch_vs_numpy_max_abs")
                    if not _is_bounded_primitive_numpy_oracle(attn_vs_numpy):
                        errors.append("commands[].preconditions[].attn_batch_vs_numpy_max_abs must be finite between 0.0 and 2e-5 when primitive passed")
                        break
                    if primitive_source_payload is not None:
                        primitive_source_integer_fields = (
                            "schema",
                            "rows",
                            "seed",
                            *_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS,
                        )
                        if any(
                            not isinstance(primitive_source_payload.get(field), int)
                            or isinstance(primitive_source_payload.get(field), bool)
                            for field in primitive_source_integer_fields
                        ):
                            errors.append("commands[].preconditions[].primitive_artifact_path JSON integer labels must be typed ints when primitive passed")
                            break
                        if primitive_source_payload.get("schema") != primitive_precondition.get("primitive_schema"):
                            errors.append("commands[].preconditions[].primitive_artifact_path JSON schema must match when primitive passed")
                            break
                        if primitive_source_payload.get("rows") != primitive_precondition.get("primitive_rows"):
                            errors.append("commands[].preconditions[].primitive_artifact_path JSON rows must match when primitive passed")
                            break
                        if primitive_source_payload.get("seed") != primitive_precondition.get("primitive_seed"):
                            errors.append("commands[].preconditions[].primitive_artifact_path JSON seed must match when primitive passed")
                            break
                        if any(primitive_source_payload.get(field) != primitive_precondition.get(f"primitive_{field}") for field in _REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS):
                            errors.append("commands[].preconditions[].primitive_artifact_path JSON shape fields must match when primitive passed")
                            break
                        primitive_source_context_lens = primitive_source_payload.get("context_lens")
                        if not isinstance(primitive_source_context_lens, list) or any(
                            not isinstance(item, int) or isinstance(item, bool) for item in primitive_source_context_lens
                        ):
                            errors.append("commands[].preconditions[].primitive_artifact_path JSON context_lens must be a typed int list when primitive passed")
                            break
                        if primitive_source_context_lens != primitive_precondition.get("primitive_context_lens"):
                            errors.append("commands[].preconditions[].primitive_artifact_path JSON context_lens must match when primitive passed")
                            break
                        if any(not isinstance(primitive_source_payload.get(field), bool) for field in ("passed", "aa_passed")):
                            errors.append("commands[].preconditions[].primitive_artifact_path JSON boolean labels must be typed bools when primitive passed")
                            break
                        if primitive_source_payload.get("passed") != primitive_precondition.get("passed"):
                            errors.append("commands[].preconditions[].primitive_artifact_path JSON passed must match when primitive passed")
                            break
                        primitive_source_device_blockers = _primitive_device_metadata_blockers(primitive_source_payload.get("device"))
                        if primitive_source_device_blockers:
                            errors.append(
                                "commands[].preconditions[].primitive_artifact_path JSON device must contain valid device metadata when primitive passed: "
                                + "; ".join(primitive_source_device_blockers)
                            )
                            break
                        if primitive_source_payload.get("device") != primitive_precondition.get("primitive_device"):
                            errors.append("commands[].preconditions[].primitive_artifact_path JSON device must match when primitive passed")
                            break
                        primitive_source_append_fields = (
                            "append_key_mismatch",
                            "append_value_mismatch",
                            "append_batch_aa_key_mismatch",
                            "append_batch_aa_value_mismatch",
                        )
                        if any(
                            not isinstance(primitive_source_payload.get(field), int)
                            or isinstance(primitive_source_payload.get(field), bool)
                            for field in primitive_source_append_fields
                        ):
                            errors.append("commands[].preconditions[].primitive_artifact_path JSON append mismatches must be typed ints when primitive passed")
                            break
                        if any(primitive_source_payload.get(field) != primitive_precondition.get(field) for field in primitive_source_append_fields):
                            errors.append("commands[].preconditions[].primitive_artifact_path JSON append mismatches must match when primitive passed")
                            break
                        primitive_source_attention_fields = (
                            "attn_batch_aa_max_abs",
                            "attn_batch_vs_c1_max_abs",
                            "attn_batch_vs_numpy_max_abs",
                        )
                        if any(
                            not isinstance(primitive_source_payload.get(field), float)
                            or not math.isfinite(primitive_source_payload.get(field))
                            for field in primitive_source_attention_fields
                        ):
                            errors.append("commands[].preconditions[].primitive_artifact_path JSON attention metrics must be finite typed floats when primitive passed")
                            break
                        if any(
                            primitive_source_payload.get(field) != primitive_precondition.get(field)
                            for field in (*primitive_source_attention_fields, "aa_passed")
                        ):
                            errors.append("commands[].preconditions[].primitive_artifact_path JSON attention metrics must match when primitive passed")
                            break
                    if set(primitive_precondition) != expected_passed_primitive_precondition_keys:
                        errors.append("commands[].preconditions[] passed primitive_correctness must contain exactly primitive precondition keys")
                        break
                expected_prompt_tokens = int(_argv_value(argv, "--prompt-length"))
                expected_decode_tokens = int(_argv_value(argv, "--decode-tokens"))
                scaling_preconditions = (
                    (preconditions[1], 1),
                    (preconditions[2], entry.get("batch_size")),
                )
                scaling_precondition_error = False
                for scaling_precondition, expected_concurrency in scaling_preconditions:
                    if scaling_precondition.get("passed") is not True:
                        continue
                    workload_concurrency = scaling_precondition.get("workload_concurrency")
                    if (
                        not isinstance(workload_concurrency, int)
                        or isinstance(workload_concurrency, bool)
                        or workload_concurrency != expected_concurrency
                    ):
                        errors.append("commands[].preconditions[].workload_concurrency must be a typed int matching retained scaling gate")
                        scaling_precondition_error = True
                        break
                    prompt_tokens_per_request = scaling_precondition.get("prompt_tokens_per_request")
                    if (
                        not isinstance(prompt_tokens_per_request, int)
                        or isinstance(prompt_tokens_per_request, bool)
                        or prompt_tokens_per_request != expected_prompt_tokens
                    ):
                        errors.append("commands[].preconditions[].prompt_tokens_per_request must be a typed int matching retained command shape")
                        scaling_precondition_error = True
                        break
                    reference_artifact_path = scaling_precondition.get("reference_artifact_path")
                    if not isinstance(reference_artifact_path, str) or not reference_artifact_path:
                        errors.append("commands[].preconditions[].reference_artifact_path must be a non-empty string when scaling reference passed")
                        scaling_precondition_error = True
                        break
                    if not reference_artifact_path.strip():
                        errors.append("commands[].preconditions[].reference_artifact_path must be a non-blank string when scaling reference passed")
                        scaling_precondition_error = True
                        break
                    reference_path = Path(reference_artifact_path)
                    reference_check_path = reference_path if reference_path.is_absolute() else REPO_ROOT / reference_path
                    if _path_has_parent_directory_component(reference_artifact_path):
                        errors.append("commands[].preconditions[].reference_artifact_path must not contain parent-directory components when scaling reference passed")
                        scaling_precondition_error = True
                        break
                    if reference_check_path.is_symlink():
                        errors.append("commands[].preconditions[].reference_artifact_path must be a regular file, not a symlink, when scaling reference passed")
                        scaling_precondition_error = True
                        break
                    if _path_has_symlink_parent(reference_check_path):
                        errors.append("commands[].preconditions[].reference_artifact_path parent directories must not be symlinks when scaling reference passed")
                        scaling_precondition_error = True
                        break
                    if _path_has_non_directory_parent(reference_check_path):
                        errors.append("commands[].preconditions[].reference_artifact_path parent directories must be directories when scaling reference passed")
                        scaling_precondition_error = True
                        break
                    if reference_check_path.exists() and not reference_check_path.is_file():
                        errors.append("commands[].preconditions[].reference_artifact_path must be a regular file when it already exists")
                        scaling_precondition_error = True
                        break
                    if reference_artifact_path != scaling_precondition.get("artifact_path"):
                        errors.append("commands[].preconditions[].reference_artifact_path must match scaling reference artifact_path when passed")
                        scaling_precondition_error = True
                        break
                    benchmark_model = scaling_precondition.get("benchmark_model")
                    benchmark_fixture = scaling_precondition.get("benchmark_fixture")
                    if any(
                        value is not None and (not isinstance(value, str) or not value.strip())
                        for value in (benchmark_model, benchmark_fixture)
                    ):
                        errors.append("commands[].preconditions[].benchmark input labels must be null or non-empty strings when passed")
                        scaling_precondition_error = True
                        break
                    if benchmark_model is not None and benchmark_model != _argv_value(argv, "--model"):
                        errors.append("commands[].preconditions[].benchmark_model must match retained command model when present")
                        scaling_precondition_error = True
                        break
                    if (
                        benchmark_fixture is not None
                        and scaling_precondition.get("kind") == _RETAINED_PRECONDITION_KINDS[2]
                        and benchmark_fixture != _argv_value(argv, "--fixture")
                    ):
                        errors.append("commands[].preconditions[].benchmark_fixture must match retained command fixture when present")
                        scaling_precondition_error = True
                        break
                    gen_tokens_per_request = scaling_precondition.get("gen_tokens_per_request")
                    if (
                        not isinstance(gen_tokens_per_request, int)
                        or isinstance(gen_tokens_per_request, bool)
                        or gen_tokens_per_request != expected_decode_tokens
                    ):
                        errors.append("commands[].preconditions[].gen_tokens_per_request must be a typed int matching retained command shape")
                        scaling_precondition_error = True
                        break
                    for rate_field in ("decode_tok_s_aggregate", "decode_tok_s_per_request"):
                        if not _is_positive_finite_number(scaling_precondition.get(rate_field)):
                            errors.append("commands[].preconditions[].decode rates must be positive finite numbers when passed")
                            scaling_precondition_error = True
                            break
                    if scaling_precondition_error:
                        break
                    expected_aggregate_rate = float(scaling_precondition["decode_tok_s_per_request"]) * int(expected_concurrency)
                    if abs(float(scaling_precondition["decode_tok_s_aggregate"]) - expected_aggregate_rate) > max(1e-9, expected_aggregate_rate * 1e-6):
                        errors.append("commands[].preconditions[].decode aggregate rate must match per-request rate times concurrency when passed")
                        scaling_precondition_error = True
                        break
                    if set(scaling_precondition) != expected_passed_scaling_precondition_keys:
                        errors.append("commands[].preconditions[] passed scaling reference must contain exactly scaling precondition keys")
                        scaling_precondition_error = True
                        break
                if scaling_precondition_error:
                    break
                profiler_precondition = preconditions[3]
                if profiler_precondition.get("passed") is True:
                    workload_concurrency = profiler_precondition.get("workload_concurrency")
                    if (
                        not isinstance(workload_concurrency, int)
                        or isinstance(workload_concurrency, bool)
                        or workload_concurrency != entry.get("batch_size")
                    ):
                        errors.append("commands[].preconditions[].profiler workload_concurrency must be a typed int matching retained batch_size")
                        break
                    prompt_tokens_per_request = profiler_precondition.get("prompt_tokens_per_request")
                    if (
                        not isinstance(prompt_tokens_per_request, int)
                        or isinstance(prompt_tokens_per_request, bool)
                        or prompt_tokens_per_request != expected_prompt_tokens
                    ):
                        errors.append("commands[].preconditions[].profiler prompt_tokens_per_request must be a typed int matching retained command shape")
                        break
                    gen_tokens_per_request = profiler_precondition.get("gen_tokens_per_request")
                    if (
                        not isinstance(gen_tokens_per_request, int)
                        or isinstance(gen_tokens_per_request, bool)
                        or gen_tokens_per_request != expected_decode_tokens
                    ):
                        errors.append("commands[].preconditions[].profiler gen_tokens_per_request must be a typed int matching retained command shape")
                        break
                    profiler_warmup_decode_tokens = profiler_precondition.get("profiler_warmup_decode_tokens")
                    if (
                        not isinstance(profiler_warmup_decode_tokens, int)
                        or isinstance(profiler_warmup_decode_tokens, bool)
                        or profiler_warmup_decode_tokens != int(_argv_value(argv, "--warmup-decode-tokens"))
                    ):
                        errors.append("commands[].preconditions[].profiler_warmup_decode_tokens must be a typed int matching retained command shape")
                        break
                    profiler_max_layers = profiler_precondition.get("profiler_max_layers")
                    if (
                        not isinstance(profiler_max_layers, int)
                        or isinstance(profiler_max_layers, bool)
                        or profiler_max_layers != int(_argv_value(argv, "--max-layers"))
                    ):
                        errors.append("commands[].preconditions[].profiler_max_layers must be a typed int matching retained command shape")
                        break
                    profiler_command = profiler_precondition.get("profiler_command")
                    if (
                        not isinstance(profiler_command, str)
                        or _ROCPROF_EXECUTABLE not in profiler_command
                        or _ROCPROF_COMMAND_FLAGS[0] not in profiler_command
                        or _RETAINED_BENCH_SCRIPT not in profiler_command
                    ):
                        errors.append("commands[].preconditions[].profiler_command must include rocprofv3 kernel trace retained bench when passed")
                        break
                    try:
                        profiler_command_argv = shlex.split(profiler_command)
                    except ValueError:
                        profiler_command_argv = []
                    if not profiler_command_argv or Path(profiler_command_argv[0]).name != _ROCPROF_EXECUTABLE:
                        errors.append("commands[].preconditions[].profiler_command must start with rocprofv3 when passed")
                        break
                    if profiler_command_argv.count("--") != 1:
                        errors.append("commands[].preconditions[].profiler_command must include exactly one rocprof command separator when passed")
                        break
                    separator_index = profiler_command_argv.index("--")
                    rocprof_command_argv = profiler_command_argv[:separator_index]
                    profiled_command_argv = profiler_command_argv[separator_index + 1 :]
                    rocprof_command_flags = _ROCPROF_COMMAND_FLAGS
                    if _duplicate_flags(rocprof_command_argv, rocprof_command_flags):
                        errors.append("commands[].preconditions[].profiler_command rocprof options must be unique")
                        break
                    if _empty_inline_flag_values(rocprof_command_argv, rocprof_command_flags):
                        errors.append("commands[].preconditions[].profiler_command rocprof option values must be non-empty")
                        break
                    if _ROCPROF_COMMAND_FLAGS[0] not in rocprof_command_argv:
                        errors.append("commands[].preconditions[].profiler_command must include --kernel-trace flag before rocprof separator when passed")
                        break
                    if _argv_value(rocprof_command_argv, _ROCPROF_COMMAND_FLAGS[1]) != _ROCPROF_OUTPUT_FORMAT:
                        errors.append("commands[].preconditions[].profiler_command must include --output-format csv before rocprof separator when passed")
                        break
                    profiled_launch_argv = _strip_command_env_prefix(profiled_command_argv)
                    if (
                        len(profiled_launch_argv) < 2
                        or not _is_python_executable(profiled_launch_argv[0])
                        or profiled_launch_argv[1] != _RETAINED_BENCH_SCRIPT
                    ):
                        errors.append("commands[].preconditions[].profiler_command must launch retained bench after rocprof separator when passed")
                        break
                    profiler_retained_unique_flags = tuple(
                        dict.fromkeys(_RETAINED_PROFILED_COMMAND_UNIQUE_FLAGS + _BATCH_SAMPLE_UNIQUE_FLAGS)
                    )
                    if _duplicate_flags(profiled_command_argv, profiler_retained_unique_flags):
                        errors.append("commands[].preconditions[].profiler profiled command flags must be unique")
                        break
                    if any(_flag_token_matches(token, flag) for token in profiled_command_argv for flag in _RETAINED_PROFILED_COMMAND_DISALLOWED_FLAGS):
                        errors.append("commands[].preconditions[].profiler profiled command must not skip generated equality")
                        break
                    if _empty_inline_flag_values(profiled_command_argv, profiler_retained_unique_flags):
                        errors.append("commands[].preconditions[].profiler profiled command flag values must be non-empty")
                        break
                    if _command_device_env_assignments(profiled_command_argv) != _command_device_env_assignments(argv):
                        errors.append("commands[].preconditions[].profiler_command device env prefix must match retained command")
                        break
                    if _command_text_arg(profiler_command, "--model") != _argv_value(argv, "--model") or profiler_precondition.get("profiler_model") != _argv_value(argv, "--model"):
                        errors.append("commands[].preconditions[].profiler model must match retained command")
                        break
                    if _command_text_arg(profiler_command, "--fixture") != _argv_value(argv, "--fixture") or profiler_precondition.get("profiler_fixture") != _argv_value(argv, "--fixture"):
                        errors.append("commands[].preconditions[].profiler fixture must match retained command")
                        break
                    if _command_text_arg(profiler_command, "--json") != entry.get("artifact_path"):
                        errors.append("commands[].preconditions[].profiler command --json must match retained artifact")
                        break
                    if _command_text_arg(profiler_command, _RETAINED_GATE_FLAGS[3]) != profiler_precondition.get("artifact_path"):
                        errors.append("commands[].preconditions[].profiler command --profiler-json must match profiler precondition artifact")
                        break
                    profiler_source_artifact_path = profiler_precondition.get("profiler_source_artifact_path")
                    if not isinstance(profiler_source_artifact_path, str):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path must be a non-empty string when profiler passed")
                        break
                    if not profiler_source_artifact_path.strip():
                        errors.append("commands[].preconditions[].profiler_source_artifact_path must be a non-empty string when profiler passed")
                        break
                    if _path_has_parent_directory_component(profiler_source_artifact_path):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path must not contain parent-directory components when profiler passed")
                        break
                    profiler_source_check_path = Path(profiler_source_artifact_path)
                    if not profiler_source_check_path.is_absolute():
                        profiler_source_check_path = REPO_ROOT / profiler_source_check_path
                    if profiler_source_check_path.is_symlink():
                        errors.append("commands[].preconditions[].profiler_source_artifact_path must not be a symlink when profiler passed")
                        break
                    if _path_has_symlink_parent(profiler_source_check_path):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path parent directories must not be symlinks when profiler passed")
                        break
                    if _path_has_non_directory_parent(profiler_source_check_path):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path parent directories must be directories when profiler passed")
                        break
                    if profiler_source_artifact_path != profiler_precondition.get("artifact_path"):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path must match profiler artifact_path when profiler passed")
                        break
                    if not profiler_source_check_path.exists():
                        errors.append("commands[].preconditions[].profiler_source_artifact_path must exist when profiler passed")
                        break
                    if not profiler_source_check_path.is_file():
                        errors.append("commands[].preconditions[].profiler_source_artifact_path must be a regular file when profiler passed")
                        break
                    try:
                        profiler_source_payload = _load_json_path(profiler_source_check_path)
                    except Exception:
                        errors.append("commands[].preconditions[].profiler_source_artifact_path must be valid JSON when profiler passed")
                        break
                    profiler_source_object = (
                        profiler_source_payload.get("profiler")
                        if isinstance(profiler_source_payload, dict) and isinstance(profiler_source_payload.get("profiler"), dict)
                        else profiler_source_payload
                    )
                    if not isinstance(profiler_source_object, dict):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path must contain a profiler object when profiler passed")
                        break
                    if profiler_source_object.get("artifact_path") != profiler_source_artifact_path:
                        errors.append("commands[].preconditions[].profiler_source_artifact_path JSON artifact_path must match when profiler passed")
                        break
                    if (
                        isinstance(profiler_source_payload, dict)
                        and "source_artifact_path" in profiler_source_payload
                        and profiler_source_payload.get("source_artifact_path") != profiler_source_artifact_path
                    ):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path JSON root source_artifact_path must match when profiler passed")
                        break
                    if "source_artifact_path" in profiler_source_object and profiler_source_object.get("source_artifact_path") != profiler_source_artifact_path:
                        errors.append("commands[].preconditions[].profiler_source_artifact_path JSON source_artifact_path must match when profiler passed")
                        break
                    profiler_source_command = _profiler_command_label(
                        profiler_source_object,
                        profiler_source_payload if isinstance(profiler_source_payload, dict) else None,
                    )
                    profiler_source_synthesized_fields = _synthesize_profiler_trace_fields(
                        profiler_source_object,
                        profiler_path=profiler_source_check_path,
                    )
                    if any(
                        _command_text_arg(profiler_command, flag) != _argv_value(argv, flag)
                        for flag in _RETAINED_GATE_FLAGS[:3]
                    ):
                        errors.append("commands[].preconditions[].profiler command gate paths must match retained command")
                        break
                    if profiler_precondition.get("retained_artifact_path") != entry.get("artifact_path"):
                        errors.append("commands[].preconditions[].profiler retained_artifact_path must match retained artifact")
                        break
                    profiler_reference_paths = {
                        "c1_baseline_artifact_path": _argv_value(argv, _RETAINED_GATE_FLAGS[0]),
                        "serial_bridge_artifact_path": _argv_value(argv, _RETAINED_GATE_FLAGS[1]),
                        "primitive_correctness_artifact_path": _argv_value(argv, _RETAINED_GATE_FLAGS[2]),
                    }
                    if any(profiler_precondition.get(field) != expected_path for field, expected_path in profiler_reference_paths.items()):
                        errors.append("commands[].preconditions[].profiler gate artifact paths must match retained command")
                        break
                    if profiler_precondition.get("profiler_compiler_version_file") != _argv_value(argv, "--compiler-version-file"):
                        errors.append("commands[].preconditions[].profiler_compiler_version_file must match retained command")
                        break
                    if profiler_precondition.get("profiler_require_cached_build") is not ("--require-cached-build" in argv):
                        errors.append("commands[].preconditions[].profiler_require_cached_build must match retained command")
                        break
                    if any(_argv_value(profiled_command_argv, flag) != _argv_value(argv, flag) for flag in _RETAINED_PROFILED_COMMAND_VALUE_FLAGS) or (
                        "--require-cached-build" in profiled_command_argv
                    ) != ("--require-cached-build" in argv):
                        errors.append("commands[].preconditions[].profiler profiled command flags must match retained command")
                        break
                    expected_sample_mode = _argv_value(argv, "--batch-sample-mode")
                    if expected_sample_mode is not None:
                        if _argv_value(profiled_command_argv, "--batch-sample-mode") != expected_sample_mode:
                            errors.append("commands[].preconditions[].profiler sampler flags must match retained command")
                            break
                        if ("--batch-sample-eq-ok" in profiled_command_argv) != ("--batch-sample-eq-ok" in argv):
                            errors.append("commands[].preconditions[].profiler sampler flags must match retained command")
                            break
                        sampler_flag_mismatch = any(
                            _argv_value(profiled_command_argv, sample_flag) != _argv_value(argv, sample_flag)
                            for sample_flag in ("--batch-sample-eq-artifact", "--batch-sample-eq-rows")
                        )
                        if sampler_flag_mismatch:
                            errors.append("commands[].preconditions[].profiler sampler flags must match retained command")
                            break
                    if any(_flag_token_matches(token, flag) for token in rocprof_command_argv for flag in profiler_retained_unique_flags):
                        errors.append("commands[].preconditions[].profiler_command retained bench flags must appear after rocprof separator when passed")
                        break
                    if any(_flag_token_matches(token, flag) for token in profiled_command_argv for flag in rocprof_command_flags):
                        errors.append("commands[].preconditions[].profiler_command rocprof options must appear before rocprof separator when passed")
                        break
                    profiler_synthesized_fields = profiler_precondition.get("profiler_trace_synthesized_fields")
                    if not isinstance(profiler_synthesized_fields, list) or not all(isinstance(field, str) for field in profiler_synthesized_fields):
                        errors.append("commands[].preconditions[].profiler_trace_synthesized_fields must be a string list when profiler passed")
                        break
                    if any(field not in _PROFILER_SYNTHESIZED_FIELDS for field in profiler_synthesized_fields):
                        errors.append("commands[].preconditions[].profiler_trace_synthesized_fields must contain only known trace-derived fields")
                        break
                    if len(set(profiler_synthesized_fields)) != len(profiler_synthesized_fields):
                        errors.append("commands[].preconditions[].profiler_trace_synthesized_fields must be unique when profiler passed")
                        break
                    if profiler_source_synthesized_fields != profiler_synthesized_fields:
                        errors.append("commands[].preconditions[].profiler_source_artifact_path synthesized fields must match when profiler passed")
                        break
                    profiler_source_workload = (
                        profiler_source_payload.get("workload")
                        if isinstance(profiler_source_payload, dict) and isinstance(profiler_source_payload.get("workload"), dict)
                        else profiler_source_object.get("workload")
                    )
                    if isinstance(profiler_source_workload, dict):
                        if profiler_source_workload.get("concurrency") != profiler_precondition.get("workload_concurrency"):
                            errors.append("commands[].preconditions[].profiler_source_artifact_path JSON workload.concurrency must match when profiler passed")
                            break
                        if profiler_source_workload.get("prompt_tokens_per_request") != profiler_precondition.get("prompt_tokens_per_request"):
                            errors.append("commands[].preconditions[].profiler_source_artifact_path JSON workload.prompt_tokens_per_request must match when profiler passed")
                            break
                        if profiler_source_workload.get("gen_tokens_per_request") != profiler_precondition.get("gen_tokens_per_request"):
                            errors.append("commands[].preconditions[].profiler_source_artifact_path JSON workload.gen_tokens_per_request must match when profiler passed")
                            break
                    if profiler_source_object.get("rows") is not None and profiler_source_object.get("rows") != profiler_precondition.get("workload_concurrency"):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path JSON rows must match when profiler passed")
                        break
                    if profiler_precondition.get("profiler_status") != "captured":
                        errors.append("commands[].preconditions[].profiler_status must be captured when passed")
                        break
                    if profiler_precondition.get("profiler_output_format") != _ROCPROF_OUTPUT_FORMAT:
                        errors.append("commands[].preconditions[].profiler_output_format must be csv when passed")
                        break
                    profiler_trace_dir = profiler_precondition.get("profiler_trace_dir")
                    if not isinstance(profiler_trace_dir, str) or not profiler_trace_dir:
                        errors.append("commands[].preconditions[].profiler_trace_dir must be a non-empty string when passed")
                        break
                    if _argv_value(rocprof_command_argv, _ROCPROF_COMMAND_FLAGS[2]) != profiler_trace_dir:
                        errors.append("commands[].preconditions[].profiler_trace_dir must match profiler command -d")
                        break
                    profiler_trace_dir_path = Path(profiler_trace_dir)
                    profiler_trace_dir_check_path = (
                        profiler_trace_dir_path if profiler_trace_dir_path.is_absolute() else REPO_ROOT / profiler_trace_dir_path
                    )
                    if profiler_trace_dir_check_path.is_symlink():
                        errors.append("commands[].preconditions[].profiler_trace_dir must not be a symlink when passed")
                        break
                    if _path_has_symlink_parent(profiler_trace_dir_check_path):
                        errors.append("commands[].preconditions[].profiler_trace_dir parent directories must not be symlinks when passed")
                        break
                    if _path_has_non_directory_parent(profiler_trace_dir_check_path):
                        errors.append("commands[].preconditions[].profiler_trace_dir parent directories must be directories when passed")
                        break
                    output_dir_text = summary.get("output_dir")
                    if isinstance(output_dir_text, str) and output_dir_text:
                        trace_dir_path = Path(profiler_trace_dir)
                        output_dir_path = Path(output_dir_text)
                        trace_dir_abs = (trace_dir_path if trace_dir_path.is_absolute() else REPO_ROOT / trace_dir_path).resolve()
                        output_dir_abs = (output_dir_path if output_dir_path.is_absolute() else REPO_ROOT / output_dir_path).resolve()
                        if not trace_dir_abs.is_relative_to(output_dir_abs):
                            errors.append("commands[].preconditions[].profiler_trace_dir must be under output_dir when passed")
                            break
                    if _path_has_parent_directory_component(profiler_trace_dir):
                        errors.append("commands[].preconditions[].profiler_trace_dir must not contain parent-directory components when passed")
                        break
                    if not profiler_trace_dir_check_path.exists():
                        errors.append("commands[].preconditions[].profiler_trace_dir must exist when passed")
                        break
                    if not profiler_trace_dir_check_path.is_dir():
                        errors.append("commands[].preconditions[].profiler_trace_dir must be a directory when passed")
                        break
                    profiler_trace_files = profiler_precondition.get("profiler_trace_files")
                    if (
                        not isinstance(profiler_trace_files, list)
                        or not profiler_trace_files
                        or not all(isinstance(trace_file, str) and trace_file for trace_file in profiler_trace_files)
                        or not any(_is_kernel_trace_csv_path(trace_file) for trace_file in profiler_trace_files)
                    ):
                        errors.append("commands[].preconditions[].profiler_trace_files must include a kernel-trace CSV when passed")
                        break
                    if any(Path(trace_file).suffix.lower() != ".csv" for trace_file in profiler_trace_files):
                        errors.append("commands[].preconditions[].profiler_trace_files must contain only CSV files when passed")
                        break
                    if len(set(profiler_trace_files)) != len(profiler_trace_files):
                        errors.append("commands[].preconditions[].profiler_trace_files must be unique when passed")
                        break
                    if sum(1 for trace_file in profiler_trace_files if _is_kernel_trace_csv_path(trace_file)) != 1:
                        errors.append("commands[].preconditions[].profiler_trace_files must include exactly one kernel-trace CSV when passed")
                        break
                    trace_file_check_paths = [
                        trace_file_path if trace_file_path.is_absolute() else REPO_ROOT / trace_file_path
                        for trace_file_path in (Path(trace_file) for trace_file in profiler_trace_files)
                    ]
                    kernel_trace_file_paths = [
                        trace_file_path
                        for trace_file, trace_file_path in zip(profiler_trace_files, trace_file_check_paths)
                        if _is_kernel_trace_csv_path(trace_file)
                    ]
                    if any(trace_file_path.is_symlink() for trace_file_path in trace_file_check_paths):
                        errors.append("commands[].preconditions[].profiler_trace_files must not be symlinks when passed")
                        break
                    if any(_path_has_symlink_parent(trace_file_path) for trace_file_path in trace_file_check_paths):
                        errors.append("commands[].preconditions[].profiler_trace_files parent directories must not be symlinks when passed")
                        break
                    if any(_path_has_non_directory_parent(trace_file_path) for trace_file_path in trace_file_check_paths):
                        errors.append("commands[].preconditions[].profiler_trace_files parent directories must be directories when passed")
                        break
                    if any(not _is_resolved_path_relative_to(trace_file, profiler_trace_dir) for trace_file in profiler_trace_files):
                        errors.append("commands[].preconditions[].profiler_trace_files must be under profiler_trace_dir when passed")
                        break
                    if any(_path_has_parent_directory_component(trace_file) for trace_file in profiler_trace_files):
                        errors.append("commands[].preconditions[].profiler_trace_files must not contain parent-directory components when passed")
                        break
                    if any(not trace_file_path.exists() for trace_file_path in trace_file_check_paths):
                        errors.append("commands[].preconditions[].profiler_trace_files must exist when passed")
                        break
                    if any(not trace_file_path.is_file() for trace_file_path in trace_file_check_paths):
                        errors.append("commands[].preconditions[].profiler_trace_files must be files when passed")
                        break
                    profiler_kernel_names = profiler_precondition.get("profiler_trace_kernel_names")
                    if (
                        not isinstance(profiler_kernel_names, list)
                        or not profiler_kernel_names
                        or not all(_is_stripped_non_empty_string(kernel_name) for kernel_name in profiler_kernel_names)
                        or not any("batch" in kernel_name.lower() for kernel_name in profiler_kernel_names)
                        or any(_has_disallowed_profiler_kernel_fragment(kernel_name) for kernel_name in profiler_kernel_names)
                    ):
                        errors.append("commands[].preconditions[].profiler_trace_kernel_names must include native batch kernels only when passed")
                        break
                    if len(set(profiler_kernel_names)) != len(profiler_kernel_names):
                        errors.append("commands[].preconditions[].profiler_trace_kernel_names must be unique when passed")
                        break
                    expected_kernel_names = profiler_precondition.get("expected_kernel_names")
                    if (
                        not isinstance(expected_kernel_names, list)
                        or not expected_kernel_names
                        or not all(_is_stripped_non_empty_string(kernel_name) for kernel_name in expected_kernel_names)
                        or not any("batch" in kernel_name.lower() for kernel_name in expected_kernel_names)
                        or any(_has_disallowed_profiler_kernel_fragment(kernel_name) for kernel_name in expected_kernel_names)
                    ):
                        errors.append("commands[].preconditions[].expected_kernel_names must include native batch kernels only when profiler passed")
                        break
                    if len(set(expected_kernel_names)) != len(expected_kernel_names):
                        errors.append("commands[].preconditions[].expected_kernel_names must be unique when profiler passed")
                        break
                    if any(kernel_name not in profiler_kernel_names for kernel_name in expected_kernel_names):
                        errors.append("commands[].preconditions[].expected_kernel_names must be present in profiler_trace_kernel_names")
                        break
                    kernel_durations = profiler_precondition.get("kernel_durations_ns")
                    if not isinstance(kernel_durations, dict) or not kernel_durations:
                        errors.append("commands[].preconditions[].kernel_durations_ns must contain positive kernel durations when profiler passed")
                        break
                    invalid_kernel_duration = False
                    for kernel_name, duration_ns in kernel_durations.items():
                        if (
                            not _is_stripped_non_empty_string(kernel_name)
                            or _has_disallowed_profiler_kernel_fragment(kernel_name)
                            or not _is_positive_finite_number(duration_ns)
                        ):
                            invalid_kernel_duration = True
                            break
                    if invalid_kernel_duration:
                        errors.append("commands[].preconditions[].kernel_durations_ns must contain positive kernel durations when profiler passed")
                        break
                    missing_expected_kernel_duration = [
                        kernel_name
                        for kernel_name in expected_kernel_names
                        if kernel_name not in kernel_durations
                    ]
                    if missing_expected_kernel_duration:
                        errors.append("commands[].preconditions[].kernel_durations_ns must include expected profiler kernels")
                        break
                    if any(kernel_name not in profiler_kernel_names for kernel_name in kernel_durations):
                        errors.append("commands[].preconditions[].kernel_durations_ns keys must be present in profiler_trace_kernel_names")
                        break
                    if any(kernel_name not in kernel_durations for kernel_name in profiler_kernel_names):
                        errors.append("commands[].preconditions[].profiler_trace_kernel_names must be present in kernel_durations_ns")
                        break
                    if not _is_positive_finite_number(profiler_precondition.get("total_kernel_duration_ns")):
                        errors.append("commands[].preconditions[].total_kernel_duration_ns must be positive when profiler passed")
                        break
                    total_kernel_duration = float(profiler_precondition["total_kernel_duration_ns"])
                    kernel_duration_sum = sum(float(duration_ns) for duration_ns in kernel_durations.values())
                    if abs(total_kernel_duration - kernel_duration_sum) > max(1.0, kernel_duration_sum * 1e-6):
                        errors.append("commands[].preconditions[].total_kernel_duration_ns must match sum(kernel_durations_ns) when profiler passed")
                        break
                    kernel_duration_shares = profiler_precondition.get("kernel_duration_shares")
                    if (
                        not isinstance(kernel_duration_shares, dict)
                        or set(kernel_duration_shares) != set(kernel_durations)
                        or any(not _is_positive_finite_number(kernel_duration_shares.get(kernel_name)) for kernel_name in kernel_durations)
                    ):
                        errors.append("commands[].preconditions[].kernel_duration_shares must match kernel_durations_ns keys with positive shares when profiler passed")
                        break
                    kernel_share_sum = 0.0
                    kernel_share_error = False
                    for kernel_name, duration_ns in kernel_durations.items():
                        kernel_share = float(kernel_duration_shares[kernel_name])
                        kernel_share_sum += kernel_share
                        if abs(kernel_share - (float(duration_ns) / total_kernel_duration)) > 1e-6:
                            errors.append("commands[].preconditions[].kernel_duration_shares must match kernel duration ratios when profiler passed")
                            kernel_share_error = True
                            break
                    if kernel_share_error:
                        break
                    if abs(kernel_share_sum - 1.0) > 1e-6:
                        errors.append("commands[].preconditions[].kernel_duration_shares must sum to 1.0 when profiler passed")
                        break
                    kernel_categories = profiler_precondition.get("kernel_duration_categories_ns")
                    kernel_category_shares = profiler_precondition.get("kernel_duration_category_shares")
                    if (
                        not isinstance(kernel_categories, dict)
                        or set(kernel_categories) != set(_PROFILER_KERNEL_DURATION_CATEGORIES)
                        or any(not _is_nonnegative_finite_number(kernel_categories.get(category)) for category in _PROFILER_KERNEL_DURATION_CATEGORIES)
                        or not isinstance(kernel_category_shares, dict)
                        or set(kernel_category_shares) != set(_PROFILER_KERNEL_DURATION_CATEGORIES)
                        or any(not _is_nonnegative_finite_number(kernel_category_shares.get(category)) for category in _PROFILER_KERNEL_DURATION_CATEGORIES)
                    ):
                        errors.append("commands[].preconditions[].kernel duration categories must include required non-negative categories when profiler passed")
                        break
                    expected_kernel_categories = _profiler_kernel_duration_category_sums(kernel_durations)
                    if any(
                        abs(float(kernel_categories[category]) - expected_duration) > max(1.0, expected_duration * 1e-6)
                        for category, expected_duration in expected_kernel_categories.items()
                    ):
                        errors.append("commands[].preconditions[].kernel_duration_categories_ns must match categorized kernel_durations_ns when profiler passed")
                        break
                    category_duration_sum = 0.0
                    category_share_sum = 0.0
                    category_share_error = False
                    for category in _PROFILER_KERNEL_DURATION_CATEGORIES:
                        category_duration = float(kernel_categories[category])
                        category_share = float(kernel_category_shares[category])
                        category_duration_sum += category_duration
                        category_share_sum += category_share
                        if abs(category_share - (category_duration / total_kernel_duration)) > 1e-6:
                            errors.append("commands[].preconditions[].kernel_duration_category_shares must match category duration ratios when profiler passed")
                            category_share_error = True
                            break
                    if category_share_error:
                        break
                    if abs(category_duration_sum - total_kernel_duration) > max(1.0, total_kernel_duration * 1e-6):
                        errors.append("commands[].preconditions[].kernel_duration_categories_ns must sum to total_kernel_duration_ns when profiler passed")
                        break
                    if abs(category_share_sum - 1.0) > 1e-6:
                        errors.append("commands[].preconditions[].kernel_duration_category_shares must sum to 1.0 when profiler passed")
                        break
                    trace_kernel_names_from_csv: list[str] = []
                    seen_trace_kernel_names: set[str] = set()
                    trace_kernel_durations_from_csv: dict[str, float] = {}
                    for trace_file_path in kernel_trace_file_paths:
                        for kernel_name in _read_profiler_trace_kernel_names(trace_file_path):
                            if kernel_name not in seen_trace_kernel_names:
                                trace_kernel_names_from_csv.append(kernel_name)
                                seen_trace_kernel_names.add(kernel_name)
                        for kernel_name, duration_ns in _read_profiler_trace_kernel_durations(trace_file_path).items():
                            trace_kernel_durations_from_csv[kernel_name] = trace_kernel_durations_from_csv.get(kernel_name, 0.0) + duration_ns
                    if not trace_kernel_names_from_csv:
                        errors.append("commands[].preconditions[].profiler_trace_files must contain readable kernel rows when passed")
                        break
                    if set(trace_kernel_names_from_csv) != set(profiler_kernel_names):
                        errors.append("commands[].preconditions[].profiler_trace_kernel_names must match kernel-trace CSV rows when passed")
                        break
                    if set(trace_kernel_durations_from_csv) != set(kernel_durations):
                        errors.append("commands[].preconditions[].kernel_durations_ns keys must match kernel-trace CSV rows when profiler passed")
                        break
                    trace_duration_error = False
                    for kernel_name, trace_duration_ns in trace_kernel_durations_from_csv.items():
                        duration_ns = float(kernel_durations[kernel_name])
                        tolerance = max(1.0, abs(trace_duration_ns) * 1e-6)
                        if abs(duration_ns - trace_duration_ns) > tolerance:
                            errors.append("commands[].preconditions[].kernel_durations_ns must match kernel-trace CSV durations when profiler passed")
                            trace_duration_error = True
                            break
                    if trace_duration_error:
                        break
                    if not _is_positive_finite_number(profiler_precondition.get("cpu_side_total_seconds")):
                        errors.append("commands[].preconditions[].cpu_side_total_seconds must be positive when profiler passed")
                        break
                    cpu_durations = profiler_precondition.get("cpu_side_bottlenecks_seconds")
                    cpu_shares = profiler_precondition.get("cpu_side_bottleneck_shares")
                    if (
                        not isinstance(cpu_durations, dict)
                        or set(cpu_durations) != set(_PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES)
                        or any(not _is_nonnegative_finite_number(cpu_durations.get(category)) for category in _PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES)
                        or not isinstance(cpu_shares, dict)
                        or set(cpu_shares) != set(_PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES)
                        or any(not _is_nonnegative_finite_number(cpu_shares.get(category)) for category in _PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES)
                    ):
                        errors.append("commands[].preconditions[].cpu-side bottlenecks must include required non-negative categories when profiler passed")
                        break
                    cpu_side_total = float(profiler_precondition["cpu_side_total_seconds"])
                    cpu_duration_sum = sum(float(cpu_durations[category]) for category in _PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES)
                    if abs(cpu_duration_sum - cpu_side_total) > max(1e-9, cpu_side_total * 1e-6):
                        errors.append("commands[].preconditions[].cpu_side_bottlenecks_seconds must sum to cpu_side_total_seconds when profiler passed")
                        break
                    cpu_share_sum = 0.0
                    cpu_share_error = False
                    for category in _PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES:
                        cpu_share = float(cpu_shares[category])
                        cpu_share_sum += cpu_share
                        if abs(cpu_share - (float(cpu_durations[category]) / cpu_side_total)) > 1e-6:
                            errors.append("commands[].preconditions[].cpu_side_bottleneck_shares must match CPU duration ratios when profiler passed")
                            cpu_share_error = True
                            break
                    if cpu_share_error:
                        break
                    if abs(cpu_share_sum - 1.0) > 1e-6:
                        errors.append("commands[].preconditions[].cpu_side_bottleneck_shares must sum to 1.0 when profiler passed")
                        break
                    if profiler_source_object.get("status") != profiler_precondition.get("profiler_status"):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path JSON status must match when profiler passed")
                        break
                    if profiler_source_object.get("output_format") != profiler_precondition.get("profiler_output_format"):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path JSON output_format must match when profiler passed")
                        break
                    if profiler_source_object.get("trace_dir") != profiler_precondition.get("profiler_trace_dir"):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path JSON trace_dir must match when profiler passed")
                        break
                    if profiler_source_object.get("trace_files") != profiler_precondition.get("profiler_trace_files"):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path JSON trace_files must match when profiler passed")
                        break
                    if profiler_source_object.get("trace_kernel_names") != profiler_precondition.get("profiler_trace_kernel_names"):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path JSON trace_kernel_names must match when profiler passed")
                        break
                    if profiler_source_object.get("expected_kernel_names") != profiler_precondition.get("expected_kernel_names"):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path JSON expected_kernel_names must match when profiler passed")
                        break
                    if profiler_source_object.get("kernel_durations_ns") != profiler_precondition.get("kernel_durations_ns"):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path JSON kernel_durations_ns must match when profiler passed")
                        break
                    if profiler_source_object.get("total_kernel_duration_ns") != profiler_precondition.get("total_kernel_duration_ns"):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path JSON total_kernel_duration_ns must match when profiler passed")
                        break
                    if profiler_source_object.get("kernel_duration_shares") != profiler_precondition.get("kernel_duration_shares"):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path JSON kernel_duration_shares must match when profiler passed")
                        break
                    if profiler_source_object.get("kernel_duration_categories_ns") != profiler_precondition.get("kernel_duration_categories_ns"):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path JSON kernel_duration_categories_ns must match when profiler passed")
                        break
                    if profiler_source_object.get("kernel_duration_category_shares") != profiler_precondition.get("kernel_duration_category_shares"):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path JSON kernel_duration_category_shares must match when profiler passed")
                        break
                    if profiler_source_object.get("cpu_side_total_seconds") != profiler_precondition.get("cpu_side_total_seconds"):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path JSON cpu_side_total_seconds must match when profiler passed")
                        break
                    if profiler_source_object.get("cpu_side_bottlenecks_seconds") != profiler_precondition.get("cpu_side_bottlenecks_seconds"):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path JSON cpu_side_bottlenecks_seconds must match when profiler passed")
                        break
                    if profiler_source_object.get("cpu_side_bottleneck_shares") != profiler_precondition.get("cpu_side_bottleneck_shares"):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path JSON cpu_side_bottleneck_shares must match when profiler passed")
                        break
                    if profiler_source_command != profiler_precondition.get("profiler_command"):
                        errors.append("commands[].preconditions[].profiler_source_artifact_path JSON command must match when profiler passed")
                        break
                    if set(profiler_precondition) != expected_passed_profiler_precondition_keys:
                        errors.append("commands[].preconditions[] passed profiler_summary must contain exactly profiler precondition keys")
                        break
            postconditions = entry.get("postconditions")
            preconditions = entry.get("preconditions")
            if isinstance(postconditions, list):
                postcondition = postconditions[0]
                if postcondition.get("artifact_path") != entry.get("artifact_path"):
                    errors.append("commands[].postconditions[].artifact_path must match commands[].artifact_path")
                    break
                profiler_precondition = next(
                    (
                        condition
                        for condition in preconditions
                        if isinstance(condition, dict) and condition.get("kind") == "profiler_summary"
                    ),
                    None,
                ) if isinstance(preconditions, list) else None
                if (
                    isinstance(profiler_precondition, dict)
                    and postcondition.get("profiler_precondition_artifact_path") != profiler_precondition.get("artifact_path")
                ):
                    errors.append("commands[].postconditions[].profiler_precondition_artifact_path must match profiler_summary precondition")
                    break
                profiler_precondition_source = (
                    profiler_precondition.get("profiler_source_artifact_path")
                    if isinstance(profiler_precondition, dict)
                    else None
                )
                if isinstance(profiler_precondition, dict) and postcondition.get("passed") is True:
                    if (
                        not isinstance(profiler_precondition_source, str)
                        or not profiler_precondition_source
                        or postcondition.get("profiler_precondition_source_artifact_path") != profiler_precondition_source
                    ):
                        errors.append("commands[].postconditions[].profiler_precondition_source_artifact_path must match profiler_summary precondition")
                        break
                    if postcondition.get("profiler_source_artifact_path") != profiler_precondition_source:
                        errors.append("commands[].postconditions[].profiler_source_artifact_path must match profiler_summary precondition")
                        break
                reason = postcondition.get("reason")
                source_malformed_reason = "retained artifact profiler.source_artifact_path is missing or malformed"
                source_mismatch_reason = "retained artifact profiler.source_artifact_path does not match profiler precondition source path"
                if (
                    isinstance(profiler_precondition, dict)
                    and postcondition.get("passed") is not True
                    and "profiler_precondition_source_artifact_path" in postcondition
                    and (
                        not isinstance(profiler_precondition_source, str)
                        or not profiler_precondition_source
                        or postcondition.get("profiler_precondition_source_artifact_path") != profiler_precondition_source
                    )
                ):
                    errors.append("commands[].postconditions[].profiler_precondition_source_artifact_path must match profiler_summary precondition when present")
                    break
                source_artifact_path = postcondition.get("profiler_source_artifact_path")
                if (
                    isinstance(profiler_precondition, dict)
                    and postcondition.get("passed") is not True
                    and reason == source_malformed_reason
                    and any(
                        key in postcondition
                        for key in (
                            "profiler_source_artifact_path",
                            "profiler_synthesized_fields",
                            "profiler_precondition_synthesized_fields",
                        )
                    )
                ):
                    errors.append("commands[].postconditions[] malformed source failures must not include source or synthesized-field evidence")
                    break
                if (
                    isinstance(profiler_precondition, dict)
                    and postcondition.get("passed") is not True
                    and reason == source_mismatch_reason
                    and (
                        not isinstance(profiler_precondition_source, str)
                        or not profiler_precondition_source
                        or not isinstance(source_artifact_path, str)
                        or not source_artifact_path
                        or _path_has_parent_directory_component(source_artifact_path)
                        or source_artifact_path == profiler_precondition_source
                    )
                ):
                    errors.append("commands[].postconditions[].profiler_source_artifact_path must document source mismatch when source check failed")
                    break
                output_dir_text = summary.get("output_dir")
                if (
                    isinstance(profiler_precondition, dict)
                    and postcondition.get("passed") is not True
                    and reason == source_mismatch_reason
                    and isinstance(source_artifact_path, str)
                    and source_artifact_path
                    and not _path_has_parent_directory_component(source_artifact_path)
                    and isinstance(output_dir_text, str)
                    and output_dir_text
                ):
                    source_path = Path(source_artifact_path)
                    source_check_path = source_path if source_path.is_absolute() else REPO_ROOT / source_path
                    if source_check_path.is_symlink():
                        errors.append("commands[].postconditions[].profiler_source_artifact_path must not be a symlink when source check failed")
                        break
                    if _path_has_symlink_parent(source_check_path):
                        errors.append("commands[].postconditions[].profiler_source_artifact_path parent directories must not be symlinks when source check failed")
                        break
                    if _path_has_non_directory_parent(source_check_path):
                        errors.append("commands[].postconditions[].profiler_source_artifact_path parent directories must be directories when source check failed")
                        break
                    if source_check_path.suffix.lower() != ".json":
                        errors.append("commands[].postconditions[].profiler_source_artifact_path must be a JSON artifact path when source check failed")
                        break
                    output_dir_abs = (Path(output_dir_text) if Path(output_dir_text).is_absolute() else REPO_ROOT / output_dir_text).resolve()
                    source_abs = source_check_path.resolve()
                    if not source_abs.is_relative_to(output_dir_abs):
                        errors.append("commands[].postconditions[].profiler_source_artifact_path must be under output_dir when source check failed")
                        break
                if (
                    isinstance(profiler_precondition, dict)
                    and postcondition.get("passed") is not True
                    and reason != source_mismatch_reason
                    and "profiler_source_artifact_path" in postcondition
                    and postcondition.get("profiler_source_artifact_path") != profiler_precondition_source
                ):
                    errors.append("commands[].postconditions[].profiler_source_artifact_path must match profiler_summary precondition when not the failed source check")
                    break
                if postcondition.get("passed") is True and reason is not None:
                    errors.append("commands[].postconditions[].reason must be null when passed")
                    break
                if postcondition.get("passed") is False and (not isinstance(reason, str) or not reason.strip()):
                    errors.append("commands[].postconditions[].reason must be a non-empty string when failed")
                    break
                if postcondition.get("passed") is False and not set(postcondition).issubset(expected_passed_retained_postcondition_keys):
                    errors.append("commands[].postconditions[] failed retained profiler synthesis must contain only retained postcondition keys")
                    break
                if postcondition.get("passed") is True and set(postcondition) != expected_passed_retained_postcondition_keys:
                    errors.append("commands[].postconditions[] passed retained profiler synthesis must contain exactly retained postcondition keys")
                    break
                profiler_fields = postcondition.get("profiler_synthesized_fields")
                precondition_fields = postcondition.get("profiler_precondition_synthesized_fields")
                has_profiler_field_evidence = "profiler_synthesized_fields" in postcondition
                has_precondition_field_evidence = "profiler_precondition_synthesized_fields" in postcondition
                has_synthesized_field_evidence = has_profiler_field_evidence or has_precondition_field_evidence
                profiler_precondition_fields = (
                    profiler_precondition.get("profiler_trace_synthesized_fields")
                    if isinstance(profiler_precondition, dict)
                    else None
                )
                if postcondition.get("passed") is not True and has_profiler_field_evidence != has_precondition_field_evidence:
                    errors.append("commands[].postconditions[].profiler synthesized fields must be paired when present")
                    break
                if postcondition.get("passed") is not True and has_synthesized_field_evidence and (
                    not isinstance(profiler_fields, list)
                    or not all(isinstance(field, str) for field in profiler_fields)
                    or not isinstance(precondition_fields, list)
                    or not all(isinstance(field, str) for field in precondition_fields)
                ):
                    errors.append("commands[].postconditions[].profiler synthesized fields must be string lists when present")
                    break
                if postcondition.get("passed") is not True and has_synthesized_field_evidence and (
                    not isinstance(profiler_precondition_fields, list)
                    or not all(isinstance(field, str) for field in profiler_precondition_fields)
                ):
                    errors.append("commands[].preconditions[].profiler_trace_synthesized_fields must be a string list when retained postcondition has field evidence")
                    break
                if postcondition.get("passed") is not True and has_synthesized_field_evidence and list(precondition_fields) != list(profiler_precondition_fields):
                    errors.append("commands[].postconditions[].profiler_precondition_synthesized_fields must match profiler_summary precondition when present")
                    break
                if postcondition.get("passed") is not True and has_synthesized_field_evidence and any(field not in _PROFILER_SYNTHESIZED_FIELDS for field in profiler_fields):
                    errors.append("commands[].postconditions[].profiler synthesized fields must be known trace-derived fields when present")
                    break
                if postcondition.get("passed") is not True and has_synthesized_field_evidence and len(set(profiler_fields)) != len(profiler_fields):
                    errors.append("commands[].postconditions[].profiler synthesized fields must be unique when present")
                    break
                if postcondition.get("passed") is True and (
                    not isinstance(profiler_fields, list)
                    or not all(isinstance(field, str) for field in profiler_fields)
                    or not isinstance(precondition_fields, list)
                    or not all(isinstance(field, str) for field in precondition_fields)
                ):
                    errors.append("commands[].postconditions[].profiler synthesized fields must be string lists when passed")
                    break
                if postcondition.get("passed") is True and list(profiler_fields) != list(precondition_fields):
                    errors.append("commands[].postconditions[].profiler synthesized fields must match when passed")
                    break
                if postcondition.get("passed") is True and (
                    not isinstance(profiler_precondition_fields, list)
                    or not all(isinstance(field, str) for field in profiler_precondition_fields)
                ):
                    errors.append("commands[].preconditions[].profiler_trace_synthesized_fields must be a string list when retained postcondition passed")
                    break
                if postcondition.get("passed") is True and list(precondition_fields) != list(profiler_precondition_fields):
                    errors.append("commands[].postconditions[].profiler_precondition_synthesized_fields must match profiler_summary precondition")
                    break
                if postcondition.get("passed") is True and any(field not in _PROFILER_SYNTHESIZED_FIELDS for field in profiler_precondition_fields):
                    errors.append("commands[].postconditions[].profiler synthesized fields must be known trace-derived fields")
                    break
                if postcondition.get("passed") is True and len(set(profiler_precondition_fields)) != len(profiler_precondition_fields):
                    errors.append("commands[].postconditions[].profiler synthesized fields must be unique")
                    break
            failed_postconditions = [
                postcondition
                for postcondition in postconditions
                if isinstance(postcondition, dict) and postcondition.get("passed") is not True
            ] if isinstance(postconditions, list) else []
            failed_preconditions = [
                precondition
                for precondition in preconditions
                if isinstance(precondition, dict) and precondition.get("passed") is not True
            ] if isinstance(preconditions, list) else []
            if status == _PASSED_COMMAND_STATUS and failed_preconditions:
                errors.append("commands[].status passed cannot include failed preconditions")
                break
            if status == _FAILED_COMMAND_STATUS and failed_preconditions:
                errors.append("commands[].status failed cannot include failed preconditions")
                break
            if status == _PASSED_COMMAND_STATUS and returncode != 0:
                errors.append("commands[].status passed requires returncode 0")
                break
            if status == _PASSED_COMMAND_STATUS and failed_postconditions:
                errors.append("commands[].status passed cannot include failed postconditions")
                break
            if status == _PASSED_COMMAND_STATUS and summary.get("status") == "passed":
                artifact_path = Path(entry["artifact_path"])
                if artifact_path.is_symlink():
                    errors.append("commands[].artifact_path must be a regular file, not a symlink, for passed summary rows")
                    break
                if not artifact_path.is_file():
                    errors.append("commands[].artifact_path must exist for passed summary rows")
                    break
            if status == _FAILED_COMMAND_STATUS and returncode == 0 and isinstance(postconditions, list) and postconditions and not failed_postconditions:
                errors.append("commands[].status failed with returncode 0 cannot include only passed postconditions")
                break
            if status == _FAILED_COMMAND_STATUS and returncode == 0 and not failed_postconditions:
                errors.append("commands[].status failed with returncode 0 requires a failed postcondition")
                break
            if status == _SKIPPED_COMMAND_STATUS:
                if "postconditions" in entry or "postcondition" in entry:
                    errors.append("commands[].postconditions must be absent for skipped rows")
                    break
                preconditions = entry.get("preconditions")
                if not isinstance(preconditions, list):
                    errors.append("commands[].preconditions must be a list for skipped rows")
                    break
                failed_preconditions = [
                    precondition
                    for precondition in preconditions
                    if isinstance(precondition, dict) and precondition.get("passed") is not True
                ]
                if not failed_preconditions:
                    errors.append("commands[].precondition must identify a failed precondition for skipped rows")
                    break
                if not _typed_json_like_matches(entry.get("precondition"), failed_preconditions[0]):
                    errors.append("commands[].precondition must match the first failed precondition")
                    break
                if entry.get("output_tail") != failed_preconditions[0].get("reason"):
                    errors.append("commands[].output_tail must match skipped precondition reason")
                    break
                if set(entry) != expected_skipped_command_keys:
                    errors.append("commands[] skipped rows must contain exactly skipped command keys")
                    break
            if status in {_PASSED_COMMAND_STATUS, _FAILED_COMMAND_STATUS} and not any(
                field in entry for field in ("preconditions", "precondition", "postconditions", "postcondition")
            ) and set(entry) != expected_simple_executed_command_keys:
                errors.append("commands[] simple executed rows must contain exactly executed command keys")
                break
            entry_git_dirty = entry.get("git_dirty")
            if not isinstance(entry_git_dirty, bool):
                errors.append("commands[].git_dirty must be a bool")
                break
            if git_dirty is not None and entry_git_dirty is not git_dirty:
                errors.append("commands[].git_dirty must match git.dirty")
                break
    completed_command_count = summary.get("completed_command_count")
    if (
        not isinstance(completed_command_count, int)
        or isinstance(completed_command_count, bool)
        or completed_command_count != len(entries)
    ):
        errors.append("completed_command_count must be a typed int matching len(commands)")
    command_count = summary.get("command_count")
    command_count_shape_ok = False
    if not isinstance(command_count, int) or isinstance(command_count, bool) or command_count < len(entries):
        errors.append("command_count must be an int greater than or equal to completed_command_count")
    else:
        command_count_shape_ok = True
        if summary.get("dry_run") is True and command_count != len(entries):
            errors.append("dry-run summaries must include all planned commands")
        elif isinstance(options, Mapping) and options.get("stop_on_failure") is False and command_count != len(entries):
            errors.append("non-stop summaries must include all planned commands")
    if command_count_shape_ok and isinstance(options, Mapping) and isinstance(options.get("include_int8"), bool) and isinstance(options.get("include_gguf"), bool) and batch_sizes:
        expected_plan: list[tuple[str, int]] = []
        for c in batch_sizes:
            expected_plan.extend([
                (_PRIMITIVE_COMMAND_CATEGORY, c),
                (_SERIAL_BRIDGE_COMMAND_CATEGORY, c),
                (_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY, c),
            ])
            if options["include_int8"] and c != 1:
                expected_plan.append((_INT8_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY, c))
            if options["include_gguf"] and c != 1:
                expected_plan.extend((_GGUF_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY, c) for _ in _GGUF_DIAGNOSTIC_QUANTS)
        if command_count != len(expected_plan):
            errors.append("command_count must match batch_sizes/options.include_int8/include_gguf")
        elif [(entry.get("category"), entry.get("batch_size")) for entry in entries] != expected_plan[: len(entries)]:
            errors.append("commands[] category/batch_size order must match batch_sizes/options.include_int8/include_gguf")
        elif options["include_gguf"]:
            expected_gguf_quant_plan: list[tuple[int, str]] = []
            for c in batch_sizes:
                if c != 1:
                    expected_gguf_quant_plan.extend((c, quant) for quant in _GGUF_DIAGNOSTIC_QUANTS)
            actual_gguf_quant_plan = [
                (int(entry["batch_size"]), str(_argv_value(entry["argv"], "--quant")))
                for entry in entries
                if entry.get("category") == _GGUF_NATIVE_DIAGNOSTIC_COMMAND_CATEGORY
                and isinstance(entry.get("batch_size"), int)
                and not isinstance(entry.get("batch_size"), bool)
                and isinstance(entry.get("argv"), list)
            ]
            if actual_gguf_quant_plan != expected_gguf_quant_plan[: len(actual_gguf_quant_plan)]:
                errors.append("commands[] GGUF quant order must match the template quant set for each c>1")
    if isinstance(options, Mapping) and options.get("stop_on_failure") is True:
        for index, entry in enumerate(entries[:-1]):
            if entry.get("status") in {_FAILED_COMMAND_STATUS, _SKIPPED_COMMAND_STATUS}:
                errors.append("commands[] failed/skipped row must be final when stop_on_failure is true")
                break
    status_counts = summary.get("status_counts")
    if not isinstance(status_counts, Mapping):
        errors.append("status_counts must be an object")
    if not _count_leaf_labels_within(status_counts, set(_SWEEP_COMMAND_STATUS_LABELS)):
        errors.append("status_counts must contain only known command status labels")
    if not _count_leaf_values_are_nonnegative_ints(status_counts):
        errors.append("status_counts must contain only non-negative integer count values")
    expected_status_counts = _status_counts(entries)
    if not _typed_count_mapping_matches(status_counts, expected_status_counts):
        errors.append("status_counts must match commands")
    category_status_counts = summary.get("category_status_counts")
    if not isinstance(category_status_counts, Mapping):
        errors.append("category_status_counts must be an object")
    if not _count_top_labels_within(category_status_counts, set(_SWEEP_COMMAND_CATEGORIES)):
        errors.append("category_status_counts must contain only known command category labels")
    if not _count_leaf_labels_within(category_status_counts, set(_SWEEP_COMMAND_STATUS_LABELS)):
        errors.append("category_status_counts must contain only known command status labels")
    if not _count_leaf_values_are_nonnegative_ints(category_status_counts):
        errors.append("category_status_counts must contain only non-negative integer count values")
    expected_category_status_counts = _category_status_counts(entries)
    if not _typed_count_mapping_matches(category_status_counts, expected_category_status_counts):
        errors.append("category_status_counts must match commands")
    expected_status = _summary_status(entries)
    if summary.get("status") != expected_status:
        errors.append("status must match commands")
    retained_precondition_counts = summary.get("retained_precondition_counts")
    if not isinstance(retained_precondition_counts, Mapping):
        errors.append("retained_precondition_counts must be an object")
    if not _count_top_labels_within(retained_precondition_counts, set(_RETAINED_PRECONDITION_KINDS)):
        errors.append("retained_precondition_counts must contain only known retained precondition labels")
    if not _count_leaf_labels_within(retained_precondition_counts, set(_RETAINED_CONDITION_STATUS_LABELS)):
        errors.append("retained_precondition_counts must contain only passed/failed status labels")
    if not _count_leaf_values_are_nonnegative_ints(retained_precondition_counts):
        errors.append("retained_precondition_counts must contain only non-negative integer count values")
    expected_precondition_counts = _retained_precondition_counts(entries)
    if not _typed_count_mapping_matches(retained_precondition_counts, expected_precondition_counts):
        errors.append("retained_precondition_counts must match commands.preconditions")
    retained_postcondition_counts = summary.get("retained_postcondition_counts")
    if not isinstance(retained_postcondition_counts, Mapping):
        errors.append("retained_postcondition_counts must be an object")
    if not _count_top_labels_within(retained_postcondition_counts, set(_RETAINED_POSTCONDITION_KINDS)):
        errors.append("retained_postcondition_counts must contain only known retained postcondition labels")
    if not _count_leaf_labels_within(retained_postcondition_counts, set(_RETAINED_CONDITION_STATUS_LABELS)):
        errors.append("retained_postcondition_counts must contain only passed/failed status labels")
    if not _count_leaf_values_are_nonnegative_ints(retained_postcondition_counts):
        errors.append("retained_postcondition_counts must contain only non-negative integer count values")
    expected_postcondition_counts = _retained_postcondition_counts(entries)
    if not _typed_count_mapping_matches(retained_postcondition_counts, expected_postcondition_counts):
        errors.append("retained_postcondition_counts must match commands.postconditions")
    expected_failed_postconditions = _failed_postconditions(entries)
    if not _typed_json_like_matches(summary_failed_postconditions, expected_failed_postconditions):
        errors.append("failed_postconditions must match commands.postconditions")
    for entry in entries:
        preconditions = entry.get("preconditions")
        if isinstance(preconditions, list):
            failed_preconditions = [
                precondition
                for precondition in preconditions
                if isinstance(precondition, dict) and precondition.get("passed") is not True
            ]
            if failed_preconditions and not _typed_json_like_matches(entry.get("precondition"), failed_preconditions[0]):
                errors.append("commands[].precondition must match the first failed precondition")
                break
            if not failed_preconditions and "precondition" in entry:
                errors.append("commands[].precondition must be absent unless a precondition failed")
                break
        elif "precondition" in entry:
            errors.append("commands[].precondition must be absent unless preconditions include a failure")
            break
        postconditions = entry.get("postconditions")
        if not isinstance(postconditions, list):
            if "postcondition" in entry:
                errors.append("commands[].postcondition must be absent unless postconditions include a failure")
                break
            continue
        failed_postconditions = [
            postcondition
            for postcondition in postconditions
            if isinstance(postcondition, dict) and postcondition.get("passed") is not True
        ]
        if failed_postconditions and not _typed_json_like_matches(entry.get("postcondition"), failed_postconditions[0]):
            errors.append("commands[].postcondition must match the first failed postcondition")
            break
        if failed_postconditions and entry.get("output_tail") != str(failed_postconditions[0].get("reason")):
            errors.append("commands[].output_tail must match failed postcondition reason")
            break
        if not failed_postconditions and "postcondition" in entry:
            errors.append("commands[].postcondition must be absent unless a postcondition failed")
            break
    if errors:
        raise ValueError("invalid c-sweep summary: " + "; ".join(errors))


def _validate_cli_path_option(flag: str, path: str | Path) -> None:
    if not isinstance(path, (str, Path)):
        raise ValueError(f"{flag} must be a typed path")
    if isinstance(path, str) and not path.strip():
        raise ValueError(f"{flag} must be a non-empty path")
    path = Path(path)
    if _path_has_parent_directory_component(path):
        raise ValueError(f"{flag} must not contain parent-directory components")
    check_path = path if path.is_absolute() else REPO_ROOT / path
    if check_path.is_symlink():
        raise ValueError(f"{flag} must not be a symlink")
    if _path_has_symlink_parent(check_path):
        raise ValueError(f"{flag} parent directories must not be symlinks")
    if _path_has_non_directory_parent(check_path):
        raise ValueError(f"{flag} parent directories must be directories")


def _validate_run_options(args: argparse.Namespace) -> None:
    for option in ("model", "fixture"):
        value = getattr(args, option, None)
        if not isinstance(value, str) or not value.strip():
            flag = "--" + option.replace("_", "-")
            raise ValueError(f"{flag} must be a non-empty string")
    for option in ("dry_run", "stop_on_failure", "include_int8", "include_gguf", "require_cached_build", "batch_sample_eq_ok"):
        if not isinstance(getattr(args, option, None), bool):
            flag = "--" + option.replace("_", "-")
            raise ValueError(f"{flag} must be a typed bool")
    batch_sizes = getattr(args, "batch_sizes", None)
    if (
        not isinstance(batch_sizes, (list, tuple))
        or not batch_sizes
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in batch_sizes)
        or len(set(batch_sizes)) != len(batch_sizes)
    ):
        raise ValueError("--batch-sizes must be a non-empty unique positive-int list")
    if not isinstance(args.seed, int) or isinstance(args.seed, bool) or args.seed != _REQUIRED_PRIMITIVE_CORRECTNESS_SEED:
        raise ValueError("--seed must be typed int 1234")
    for option, minimum in (
        ("prompt_length", 1),
        ("decode_tokens", 1),
        ("warmup_decode_tokens", 0),
        ("max_layers", 1),
    ):
        value = getattr(args, option)
        flag = "--" + option.replace("_", "-")
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{flag} must be a typed integer")
        if value < minimum:
            qualifier = "non-negative" if minimum == 0 else "positive"
            raise ValueError(f"{flag} must be {qualifier}")
    _validate_cli_path_option("--output-dir", args.output_dir)
    if args.compiler_version_file is not None:
        _validate_cli_path_option("--compiler-version-file", args.compiler_version_file)
    if args.summary_json is not None:
        _validate_cli_path_option("--summary-json", args.summary_json)
    batch_sample_mode = getattr(args, "batch_sample_mode", None)
    if batch_sample_mode is not None and batch_sample_mode not in {"serial_lm_head", "batched_lm_head"}:
        raise ValueError("--batch-sample-mode must be serial_lm_head or batched_lm_head")
    for option in ("batch_sample_eq_artifact_template", "batch_sample_eq_rows"):
        value = getattr(args, option, None)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            flag = "--" + option.replace("_", "-")
            raise ValueError(f"{flag} must be a non-empty string")


def _summary_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": str(args.model),
        "fixture": str(args.fixture),
        "prompt_length": int(args.prompt_length),
        "decode_tokens": int(args.decode_tokens),
        "warmup_decode_tokens": int(args.warmup_decode_tokens),
        "max_layers": int(args.max_layers),
        "seed": int(args.seed),
        "stop_on_failure": bool(args.stop_on_failure),
        "include_int8": bool(getattr(args, "include_int8", False)),
        "include_gguf": bool(getattr(args, "include_gguf", False)),
        "require_cached_build": bool(args.require_cached_build),
        "compiler_version_file": None if args.compiler_version_file is None else str(args.compiler_version_file),
        "projection_dispatch_artifact": None
        if getattr(args, "projection_dispatch_artifact", None) is None
        else str(args.projection_dispatch_artifact),
        "batch_sample_mode": getattr(args, "batch_sample_mode", None),
        "batch_sample_eq_ok": bool(getattr(args, "batch_sample_eq_ok", False)),
        "batch_sample_eq_artifact_template": getattr(args, "batch_sample_eq_artifact_template", None),
        "batch_sample_eq_rows": getattr(args, "batch_sample_eq_rows", None),
    }


def _typed_json_like_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual.keys()) != set(expected.keys()):
            return False
        return all(_typed_json_like_matches(actual.get(key), value) for key, value in expected.items())
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        return all(_typed_json_like_matches(actual_value, expected_value) for actual_value, expected_value in zip(actual, expected))
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if isinstance(expected, float):
        return isinstance(actual, float) and actual == expected
    if expected is None:
        return actual is None
    return type(actual) is type(expected) and actual == expected


def _typed_count_mapping_matches(actual: Any, expected: Mapping[str, Any]) -> bool:
    if not isinstance(actual, Mapping) or set(actual.keys()) != set(expected.keys()):
        return False
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, Mapping):
            if not _typed_count_mapping_matches(actual_value, expected_value):
                return False
            continue
        if not isinstance(actual_value, int) or isinstance(actual_value, bool) or actual_value != expected_value:
            return False
    return True


def _count_top_labels_within(actual: Any, allowed: set[str]) -> bool:
    if not isinstance(actual, Mapping):
        return True
    return all(isinstance(key, str) and key in allowed for key in actual)


def _count_leaf_labels_within(actual: Any, allowed: set[str]) -> bool:
    if not isinstance(actual, Mapping):
        return True
    if not actual:
        return True
    if all(isinstance(value, Mapping) for value in actual.values()):
        return all(_count_leaf_labels_within(value, allowed) for value in actual.values())
    return all(isinstance(key, str) and key in allowed for key in actual)


def _count_leaf_values_are_nonnegative_ints(actual: Any) -> bool:
    if not isinstance(actual, Mapping):
        return True
    if not actual:
        return True
    if all(isinstance(value, Mapping) for value in actual.values()):
        return all(_count_leaf_values_are_nonnegative_ints(value) for value in actual.values())
    return all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in actual.values())


def _status_counts(entries: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        status = str(entry.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _category_status_counts(entries: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for entry in entries:
        category = str(entry.get("category") or "unknown")
        status = str(entry.get("status") or "unknown")
        category_counts = counts.setdefault(category, {})
        category_counts[status] = category_counts.get(status, 0) + 1
    return counts


def _retained_precondition_counts(entries: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for entry in entries:
        preconditions = entry.get("preconditions")
        if not isinstance(preconditions, list):
            continue
        for precondition in preconditions:
            if not isinstance(precondition, dict):
                continue
            kind = str(precondition.get("kind") or "unknown")
            status = (
                _RETAINED_CONDITION_STATUS_LABELS[0]
                if precondition.get("passed") is True
                else _RETAINED_CONDITION_STATUS_LABELS[1]
            )
            kind_counts = counts.setdefault(kind, {})
            kind_counts[status] = kind_counts.get(status, 0) + 1
    return counts


def _retained_postcondition_counts(entries: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for entry in entries:
        postconditions = entry.get("postconditions")
        if not isinstance(postconditions, list):
            continue
        for postcondition in postconditions:
            if not isinstance(postcondition, dict):
                continue
            kind = str(postcondition.get("kind") or "unknown")
            status = (
                _RETAINED_CONDITION_STATUS_LABELS[0]
                if postcondition.get("passed") is True
                else _RETAINED_CONDITION_STATUS_LABELS[1]
            )
            kind_counts = counts.setdefault(kind, {})
            kind_counts[status] = kind_counts.get(status, 0) + 1
    return counts


def _failed_postconditions(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for entry in entries:
        postconditions = entry.get("postconditions")
        if not isinstance(postconditions, list):
            continue
        for postcondition in postconditions:
            if not isinstance(postcondition, dict) or postcondition.get("passed") is True:
                continue
            failed.append(
                {
                    "category": entry.get("category"),
                    "batch_size": entry.get("batch_size"),
                    "artifact_path": entry.get("artifact_path"),
                    "kind": postcondition.get("kind"),
                    "profiler_precondition_artifact_path": postcondition.get("profiler_precondition_artifact_path"),
                    "reason": postcondition.get("reason"),
                }
            )
    return failed


def _skipped_preconditions(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    skipped: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("status") != _SKIPPED_COMMAND_STATUS:
            continue
        precondition = entry.get("precondition")
        if not isinstance(precondition, dict):
            continue
        skipped.append(
            {
                "category": entry.get("category"),
                "batch_size": entry.get("batch_size"),
                "artifact_path": entry.get("artifact_path"),
                "kind": precondition.get("kind"),
                "precondition_artifact_path": precondition.get("artifact_path"),
                "reason": precondition.get("reason"),
            }
        )
    return skipped


def _summary_status(entries: Sequence[dict[str, Any]]) -> str:
    statuses = [entry.get("status") for entry in entries]
    if any(status == _FAILED_COMMAND_STATUS for status in statuses):
        return "failed"
    if any(status == _SKIPPED_COMMAND_STATUS for status in statuses):
        return "blocked"
    if statuses and all(status == _PLANNED_COMMAND_STATUS for status in statuses):
        return "planned"
    return "passed"


def _git_state() -> dict[str, Any]:
    commit = _capture(["git", "rev-parse", "--short", "HEAD"])
    status = _capture(["git", "status", "--short"])
    return {
        "commit": commit.strip(),
        "dirty": bool(status.strip()),
        "status_short": status.splitlines(),
    }


def _capture(argv: Sequence[str]) -> str:
    proc = subprocess.run(
        list(argv),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.stdout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup-decode-tokens", type=int, default=8)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--compiler-version-file", type=parse_cli_path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument(
        "--projection-dispatch-artifact",
        type=parse_cli_path,
        help="Optional benchmarks/results JSON with projection_dispatch_candidates passed to retained native commands",
    )
    parser.add_argument(
        "--batch-sample-mode",
        choices=("serial_lm_head", "batched_lm_head"),
        help="Optional sampler/LM-head path passed to retained native c>N commands",
    )
    parser.add_argument(
        "--batch-sample-eq-ok",
        action="store_true",
        help="Pass --batch-sample-eq-ok to retained native c>N commands",
    )
    parser.add_argument(
        "--batch-sample-eq-artifact-template",
        help="Template for retained native --batch-sample-eq-artifact; supports {batch_size} or {c} placeholders",
    )
    parser.add_argument(
        "--batch-sample-eq-rows",
        help="Template/value for retained native --batch-sample-eq-rows; defaults to each c when --batch-sample-eq-ok is set",
    )
    parser.add_argument("--include-int8", action="store_true", help="Plan blocked INT8 KV c>N diagnostics for c>1 rows")
    parser.add_argument("--include-gguf", action="store_true", help="Plan blocked GGUF c>N diagnostics for c>1 rows and all template quants")
    parser.add_argument("--output-dir", type=parse_cli_path, default=Path("/tmp/hipengine-batch-c-sweep"))
    parser.add_argument("--summary-json", type=parse_cli_path)
    parser.add_argument("--validate-summary-json", type=parse_cli_path, help="Validate an existing c-sweep summary JSON and exit")
    parser.add_argument("--dry-run", action="store_true", help="Write the command summary without executing commands")
    parser.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.validate_summary_json is not None:
        try:
            _validate_cli_path_option("--validate-summary-json", args.validate_summary_json)
            summary = _load_json_path(Path(args.validate_summary_json))
            validate_sweep_summary(summary)
        except Exception as exc:
            print(f"invalid c-sweep summary: {exc}", file=sys.stderr)
            return 1
        print("OK")
        return 0
    try:
        summary = run_sweep(args)
    except ValueError as exc:
        print(f"invalid c-sweep run: {exc}", file=sys.stderr)
        return 1
    print(_summary_json(summary))
    return 1 if summary["status"] in {"failed", "blocked"} else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
