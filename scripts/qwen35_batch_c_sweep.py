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
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/"
    "snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"
)
DEFAULT_FIXTURE = "fixtures/qwen35_paro/parent_512_32_seed1234.json"
DEFAULT_BATCH_SIZES = (1, 2, 4, 8)
_OUTPUT_TAIL_MAX_CHARS = 4000
_DISALLOWED_PROFILER_KERNEL_NAME_FRAGMENTS = ("serial", "fallback", "per_row", "per-row")
_PROFILER_KERNEL_DURATION_CATEGORIES = (
    "attention",
    "moe",
    "projection",
    "sampling",
    "graph_replay",
    "other",
)
_PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES = (
    "load",
    "prefill",
    "warmup_decode",
    "decode",
    "validation",
    "other",
)
_PROFILER_TRACE_KERNEL_NAME_COLUMNS = ("Kernel_Name", "KernelName", "Name")
_PROFILER_TRACE_START_COLUMNS = ("Start_Timestamp", "StartTimestamp", "StartNs", "Start")
_PROFILER_TRACE_END_COLUMNS = ("End_Timestamp", "EndTimestamp", "EndNs", "End")
_PROFILER_TRACE_DURATION_COLUMNS = ("DurationNs", "Duration_NS", "Duration_Ns", "Duration")


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
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one batch size is required")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("batch sizes must be unique")
    return values


def build_sweep_commands(args: argparse.Namespace) -> tuple[SweepCommand, ...]:
    output_dir = Path(args.output_dir)
    commands: list[SweepCommand] = []
    for c in args.batch_sizes:
        primitive_json = output_dir / f"primitive-c{c}.json"
        commands.append(
            SweepCommand(
                category="primitive",
                batch_size=c,
                artifact_path=primitive_json,
                argv=(
                    sys.executable,
                    "scripts/qwen35_batch_correctness.py",
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
                category="serial_bridge",
                batch_size=c,
                artifact_path=serial_json,
                argv=tuple(
                    _batch_bench_argv(
                        "scripts/qwen35_batch_serial_bench.py",
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
                sys.executable,
                "scripts/qwen35_paro_bench.py",
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
                    category="native_diagnostic",
                    batch_size=c,
                    artifact_path=native_json,
                    argv=tuple(native_argv),
                )
            )
            continue

        native_json = output_dir / f"native-diagnostic-c{c}.json"
        native_argv = _batch_bench_argv(
            "scripts/qwen35_batch_retained_bench.py",
            args,
            batch_size=c,
            artifact_path=native_json,
        )
        native_argv.extend(
            [
                "--c1-baseline-json",
                str(output_dir / "native-baseline-c1.json"),
                "--serial-bridge-json",
                str(serial_json),
                "--primitive-correctness-json",
                str(primitive_json),
                "--profiler-json",
                str(output_dir / f"profiler-c{c}.json"),
            ]
        )
        commands.append(
            SweepCommand(
                category="native_diagnostic",
                batch_size=c,
                artifact_path=native_json,
                argv=tuple(native_argv),
            )
        )
        if getattr(args, "include_int8", False):
            int8_json = output_dir / f"int8-native-diagnostic-c{c}.json"
            commands.append(
                SweepCommand(
                    category="int8_native_diagnostic",
                    batch_size=c,
                    artifact_path=int8_json,
                    argv=(
                        sys.executable,
                        "scripts/qwen35_batch_int8_diagnostic.py",
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
                        "--json",
                        str(int8_json),
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
        sys.executable,
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
    if args.compiler_version_file is not None:
        argv.extend(["--compiler-version-file", str(args.compiler_version_file)])
    if args.require_cached_build:
        argv.append("--require-cached-build")
    return argv


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
        "--primitive-correctness-json",
        kind="primitive_correctness",
    )
    if error is not None:
        return error
    assert primitive_path is not None
    if not primitive_path.exists():
        return {
            "kind": "primitive_correctness",
            "artifact_path": str(primitive_path),
            "passed": False,
            "reason": "primitive correctness artifact does not exist",
        }
    try:
        payload = json.loads(primitive_path.read_text())
    except Exception as exc:
        return {
            "kind": "primitive_correctness",
            "artifact_path": str(primitive_path),
            "passed": False,
            "reason": f"primitive correctness artifact is invalid JSON: {type(exc).__name__}: {exc}",
        }
    reasons: list[str] = []
    if not isinstance(payload, dict):
        reasons.append("primitive correctness artifact root is not an object")
    else:
        if payload.get("rows") != command.batch_size:
            reasons.append(f"rows={payload.get('rows')!r} does not match batch_size={command.batch_size}")
        if payload.get("passed") is not True:
            reasons.append("passed is not true")
        if payload.get("append_key_mismatch") != 0:
            reasons.append("append_key_mismatch is non-zero")
        if payload.get("append_value_mismatch") != 0:
            reasons.append("append_value_mismatch is non-zero")
        attn_vs_c1 = payload.get("attn_batch_vs_c1_max_abs")
        if not _is_number(attn_vs_c1) or float(attn_vs_c1) > 1e-6:
            reasons.append("attn_batch_vs_c1_max_abs is missing or above 1e-6")
    return {
        "kind": "primitive_correctness",
        "artifact_path": str(primitive_path),
        "passed": not reasons,
        "reason": None if not reasons else "; ".join(reasons),
    }


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
    if not _is_number(total_duration) or float(total_duration) <= 0.0:
        reasons.append("total_kernel_duration_ns is missing or non-positive numeric")
        return
    if not isinstance(duration_shares, dict) or not duration_shares:
        reasons.append("kernel_duration_shares is missing or empty")
        return
    duration_keys = {key for key in kernel_durations if isinstance(key, str) and key}
    share_keys = {key for key in duration_shares if isinstance(key, str) and key}
    if duration_keys != share_keys:
        reasons.append("kernel_duration_shares keys do not match kernel_durations_ns")
    if any(_has_disallowed_profiler_kernel_fragment(key) for key in share_keys):
        reasons.append("kernel_duration_shares contains a serial/per-row/fallback kernel")

    duration_sum = 0.0
    share_sum = 0.0
    for kernel_name in sorted(duration_keys):
        duration_ns = kernel_durations.get(kernel_name)
        duration_share = duration_shares.get(kernel_name)
        if not _is_number(duration_ns) or float(duration_ns) <= 0.0:
            continue
        duration_sum += float(duration_ns)
        if not _is_number(duration_share) or float(duration_share) <= 0.0:
            reasons.append(f"kernel_duration_shares.{kernel_name} is missing or non-positive numeric")
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
    if not _is_number(total_duration) or float(total_duration) <= 0.0:
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
    duration_keys = {key for key in duration_categories if isinstance(key, str)}
    share_keys = {key for key in category_shares if isinstance(key, str)}
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
    for category in _PROFILER_KERNEL_DURATION_CATEGORIES:
        duration_ns = duration_categories.get(category)
        duration_share = category_shares.get(category)
        if not _is_number(duration_ns) or float(duration_ns) < 0.0:
            reasons.append(f"kernel_duration_categories_ns.{category} is missing or negative numeric")
            continue
        duration_sum += float(duration_ns)
        if not _is_number(duration_share) or float(duration_share) < 0.0:
            reasons.append(f"kernel_duration_category_shares.{category} is missing or negative numeric")
            continue
        share_sum += float(duration_share)
        expected_share = float(duration_ns) / float(total_duration)
        if abs(float(duration_share) - expected_share) > 1e-6:
            reasons.append(f"kernel_duration_category_shares.{category} does not match kernel category duration share")
    tolerance = max(1.0, float(total_duration) * 1e-6)
    if abs(duration_sum - float(total_duration)) > tolerance:
        reasons.append("kernel_duration_categories_ns does not sum to total_kernel_duration_ns")
    if abs(share_sum - 1.0) > 1e-6:
        reasons.append("kernel_duration_category_shares does not sum to 1.0")


def _validate_profiler_cpu_side_bottlenecks(profiler: dict[str, Any], reasons: list[str]) -> None:
    cpu_total = profiler.get("cpu_side_total_seconds")
    durations = profiler.get("cpu_side_bottlenecks_seconds")
    shares = profiler.get("cpu_side_bottleneck_shares")
    if not _is_number(cpu_total) or float(cpu_total) <= 0.0:
        reasons.append("cpu_side_total_seconds is missing or non-positive numeric")
        return
    if not isinstance(durations, dict) or not durations:
        reasons.append("cpu_side_bottlenecks_seconds is missing or empty")
        return
    if not isinstance(shares, dict) or not shares:
        reasons.append("cpu_side_bottleneck_shares is missing or empty")
        return
    expected_keys = set(_PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES)
    duration_keys = {key for key in durations if isinstance(key, str)}
    share_keys = {key for key in shares if isinstance(key, str)}
    if duration_keys != expected_keys:
        reasons.append("cpu_side_bottlenecks_seconds keys do not match required categories")
    if share_keys != expected_keys:
        reasons.append("cpu_side_bottleneck_shares keys do not match required categories")

    duration_sum = 0.0
    share_sum = 0.0
    for category in _PROFILER_CPU_SIDE_BOTTLENECK_CATEGORIES:
        duration_seconds = durations.get(category)
        duration_share = shares.get(category)
        if not _is_number(duration_seconds) or float(duration_seconds) < 0.0:
            reasons.append(f"cpu_side_bottlenecks_seconds.{category} is missing or negative numeric")
            continue
        duration_sum += float(duration_seconds)
        if not _is_number(duration_share) or float(duration_share) < 0.0:
            reasons.append(f"cpu_side_bottleneck_shares.{category} is missing or negative numeric")
            continue
        share_sum += float(duration_share)
        expected_share = float(duration_seconds) / float(cpu_total)
        if abs(float(duration_share) - expected_share) > 1e-6:
            reasons.append(f"cpu_side_bottleneck_shares.{category} does not match cpu-side duration share")
    tolerance = max(1e-9, float(cpu_total) * 1e-6)
    if abs(duration_sum - float(cpu_total)) > tolerance:
        reasons.append("cpu_side_bottlenecks_seconds does not sum to cpu_side_total_seconds")
    if abs(share_sum - 1.0) > 1e-6:
        reasons.append("cpu_side_bottleneck_shares does not sum to 1.0")


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
        payload = json.loads(path.read_text())
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
    if not isinstance(payload, dict):
        reasons.append("scaling reference artifact root is not an object")
    else:
        raw_status = payload.get("status")
        status = str(raw_status) if raw_status else "loaded"
        if status in {"failed", "rejected", "rejected_correctness", "missing", "invalid_json"}:
            reasons.append(f"status={status!r} is not usable as a scaling reference")
        reference_reason = payload.get("reason")
        if reference_reason is not None:
            reasons.append(f"scaling reference reason is non-null: {reference_reason}")
        aggregate, per_request = _extract_decode_rates(payload)
        if aggregate is None or per_request is None:
            reasons.append("decode throughput fields are missing")
        workload = payload.get("workload")
        raw_concurrency = workload.get("concurrency") if isinstance(workload, dict) else None
        if isinstance(raw_concurrency, int) and not isinstance(raw_concurrency, bool):
            concurrency = raw_concurrency
        if kind == "c1_baseline" and concurrency is None:
            concurrency = 1
        if expected_concurrency is not None and concurrency != expected_concurrency:
            reasons.append(f"workload.concurrency={concurrency!r} does not match batch_size={expected_concurrency}")
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
        "reference_status": status,
        "reference_reason": reference_reason,
        "workload_concurrency": concurrency,
        "prompt_tokens_per_request": prompt_tokens,
        "gen_tokens_per_request": gen_tokens,
        "decode_tok_s_aggregate": aggregate,
        "decode_tok_s_per_request": per_request,
    }


def _profiler_summary_precondition(command: SweepCommand) -> dict[str, Any]:
    profiler_path, error = _command_arg_path(
        command,
        "--profiler-json",
        kind="profiler_summary",
    )
    if error is not None:
        return error
    assert profiler_path is not None
    if not profiler_path.exists():
        return {
            "kind": "profiler_summary",
            "artifact_path": str(profiler_path),
            "passed": False,
            "reason": "profiler summary artifact does not exist",
        }
    try:
        payload = json.loads(profiler_path.read_text())
    except Exception as exc:
        return {
            "kind": "profiler_summary",
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
    profiler_trace_synthesized_fields: list[str] = []
    trace_kernel_names_valid = False
    if not isinstance(profiler, dict):
        reasons.append("profiler summary root is not an object")
    else:
        if profiler.get("artifact_path") != str(profiler_path):
            reasons.append("artifact_path does not match --profiler-json path")
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
        raw_trace_files = profiler.get("trace_files")
        if not isinstance(raw_trace_files, list) or not raw_trace_files:
            reasons.append("profiler.trace_files is missing or empty")
        elif not all(isinstance(trace_file, str) and trace_file for trace_file in raw_trace_files):
            reasons.append("profiler.trace_files contains a non-string entry")
        else:
            profiler_trace_files = list(raw_trace_files)
            if any(Path(trace_file).suffix.lower() != ".csv" for trace_file in profiler_trace_files):
                reasons.append("profiler.trace_files contains a non-CSV trace file")
            if not any(_is_kernel_trace_csv_path(trace_file) for trace_file in profiler_trace_files):
                reasons.append("profiler.trace_files does not include a kernel-trace CSV")
            if profiler_trace_dir is not None:
                trace_dir_path = Path(profiler_trace_dir)
                for trace_file in profiler_trace_files:
                    try:
                        Path(trace_file).relative_to(trace_dir_path)
                    except ValueError:
                        reasons.append("profiler.trace_files contains a path outside profiler.trace_dir")
                        break
        profiler_trace_synthesized_fields = _synthesize_profiler_trace_fields(profiler, profiler_path=profiler_path)
        raw_trace_kernel_names = profiler.get("trace_kernel_names")
        if not isinstance(raw_trace_kernel_names, list) or not raw_trace_kernel_names:
            reasons.append("profiler.trace_kernel_names is missing or empty")
        elif not all(isinstance(kernel_name, str) and kernel_name for kernel_name in raw_trace_kernel_names):
            reasons.append("profiler.trace_kernel_names contains a non-string entry")
        else:
            profiler_trace_kernel_names = list(raw_trace_kernel_names)
            trace_kernel_names_valid = True
            if not any("batch" in kernel_name.lower() for kernel_name in profiler_trace_kernel_names):
                reasons.append("profiler.trace_kernel_names does not include a native batch kernel")
            if any(_has_disallowed_profiler_kernel_fragment(kernel_name) for kernel_name in profiler_trace_kernel_names):
                reasons.append("profiler.trace_kernel_names contains a serial/per-row/fallback kernel")
        profiler_command = _profiler_command_label(profiler, payload if isinstance(payload, dict) else None)
        if profiler_command is None:
            reasons.append("profiler command is missing")
        else:
            if "rocprofv3" not in profiler_command or "--kernel-trace" not in profiler_command:
                reasons.append("profiler command does not include rocprofv3 --kernel-trace")
            command_output_format = _command_text_arg(profiler_command, "--output-format")
            if command_output_format != "csv":
                reasons.append(f"profiler command output-format={command_output_format!r} does not match 'csv'")
            command_trace_dir = _command_text_arg(profiler_command, "-d")
            if command_trace_dir is None:
                reasons.append("profiler command is missing -d <trace_dir>")
            elif profiler_trace_dir is not None and command_trace_dir != profiler_trace_dir:
                reasons.append(f"profiler command trace-dir={command_trace_dir!r} does not match profiler.trace_dir={profiler_trace_dir}")
            if "scripts/qwen35_batch_retained_bench.py" not in profiler_command:
                reasons.append("profiler command does not target qwen35_batch_retained_bench.py")
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
            command_profiler_path = _command_text_arg(profiler_command, "--profiler-json")
            if command_profiler_path != str(profiler_path):
                reasons.append("profiler command --profiler-json path does not match artifact_path")
            for flag, label in (
                ("--c1-baseline-json", "c1_baseline_json"),
                ("--serial-bridge-json", "serial_bridge_json"),
                ("--primitive-correctness-json", "primitive_correctness_json"),
            ):
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
        if profiler.get("status") != "captured":
            reasons.append("status is not 'captured'")
        raw_output_format = profiler.get("output_format")
        if isinstance(raw_output_format, str):
            profiler_output_format = raw_output_format
        if profiler_output_format != "csv":
            reasons.append(f"profiler.output_format={profiler_output_format!r} does not match 'csv'")
        if profiler_trace_dir is None:
            reasons.append("profiler.trace_dir is missing")
        if profiler.get("expected_kernels_present") is not True:
            reasons.append("expected_kernels_present is not true")
        expected_kernel_names = profiler.get("expected_kernel_names")
        if not isinstance(expected_kernel_names, list) or not expected_kernel_names:
            reasons.append("expected_kernel_names is missing or empty")
        elif not all(isinstance(name, str) and name for name in expected_kernel_names):
            reasons.append("expected_kernel_names contains a non-string entry")
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
                    if isinstance(kernel_name, str) and kernel_name and (
                        not _is_number(duration_ns) or float(duration_ns) <= 0.0
                    ):
                        reasons.append(f"kernel_durations_ns.{kernel_name} is missing or non-positive numeric")
                        break
            if trace_kernel_names_valid:
                missing_duration_names = sorted(
                    kernel_name
                    for kernel_name in kernel_durations
                    if isinstance(kernel_name, str)
                    and kernel_name
                    and not _has_disallowed_profiler_kernel_fragment(kernel_name)
                    and kernel_name not in profiler_trace_kernel_names
                )
                if missing_duration_names:
                    reasons.append("profiler.trace_kernel_names must include kernel_durations_ns keys")
        _validate_profiler_kernel_durations(profiler, reasons)
        _validate_profiler_kernel_duration_categories(profiler, reasons)
        _validate_profiler_cpu_side_bottlenecks(profiler, reasons)
    result: dict[str, Any] = {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": not reasons,
        "reason": None if not reasons else "; ".join(reasons),
    }
    if not reasons and isinstance(profiler, dict):
        kernel_durations = profiler["kernel_durations_ns"]
        result.update(
            {
                "profiler_status": str(profiler["status"]),
                "profiler_command": profiler_command,
                "profiler_output_format": str(profiler["output_format"]),
                "profiler_trace_dir": str(profiler["trace_dir"]),
                "profiler_trace_files": list(profiler_trace_files),
                "profiler_trace_kernel_names": list(profiler_trace_kernel_names),
                "profiler_trace_synthesized_fields": list(profiler_trace_synthesized_fields),
                "retained_artifact_path": str(command.artifact_path),
                "c1_baseline_artifact_path": _command_arg_value(command, "--c1-baseline-json"),
                "serial_bridge_artifact_path": _command_arg_value(command, "--serial-bridge-json"),
                "primitive_correctness_artifact_path": _command_arg_value(command, "--primitive-correctness-json"),
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
    if command.category != "native_diagnostic" or command.batch_size <= 1:
        return None
    return (
        _primitive_correctness_precondition(command),
        _scaling_reference_precondition(
            command,
            flag="--c1-baseline-json",
            kind="c1_baseline",
            expected_concurrency=1,
        ),
        _scaling_reference_precondition(
            command,
            flag="--serial-bridge-json",
            kind="serial_bridge",
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
        if precondition.get("kind") == "profiler_summary" and precondition.get("passed") is True:
            return precondition
    return None


def _retained_profiler_synthesis_postcondition(
    command: SweepCommand,
    preconditions: Sequence[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    profiler_precondition = _profiler_summary_precondition_record(preconditions)
    if command.category != "native_diagnostic" or profiler_precondition is None:
        return None
    result: dict[str, Any] = {
        "kind": "retained_profiler_synthesis",
        "artifact_path": str(command.artifact_path),
        "profiler_precondition_artifact_path": profiler_precondition.get("artifact_path"),
        "passed": False,
        "reason": None,
    }
    if not command.artifact_path.exists():
        result["reason"] = "retained artifact was not written for profiler provenance cross-check"
        return result
    expected_fields = profiler_precondition.get("profiler_trace_synthesized_fields")
    if not isinstance(expected_fields, list) or not all(isinstance(field, str) for field in expected_fields):
        result["reason"] = "profiler precondition synthesized fields are missing or malformed"
        return result
    try:
        payload = json.loads(command.artifact_path.read_text())
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
    result["profiler_synthesized_fields"] = list(actual_fields)
    result["profiler_precondition_synthesized_fields"] = list(expected_fields)
    if list(actual_fields) != list(expected_fields):
        result["reason"] = "retained artifact profiler.synthesized_fields does not match profiler precondition synthesized fields"
        return result
    result["passed"] = True
    return result


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
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
            entry.update({"status": "planned", "returncode": None, "duration_seconds": 0.0})
        else:
            preconditions = _native_retained_preconditions(command)
            if preconditions is not None:
                entry["preconditions"] = list(preconditions)
            precondition = _first_failed_precondition(preconditions)
            if precondition is not None:
                entry.update(
                    {
                        "status": "skipped",
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
                    "status": "passed" if proc.returncode == 0 else "failed",
                    "returncode": proc.returncode,
                    "duration_seconds": time.perf_counter() - start,
                    "output_tail": proc.stdout[-_OUTPUT_TAIL_MAX_CHARS:],
                }
            )
            if entry["status"] == "passed":
                postcondition = _retained_profiler_synthesis_postcondition(command, preconditions)
                if postcondition is not None:
                    entry["postconditions"] = [postcondition]
                    if postcondition["passed"] is not True:
                        entry["status"] = "failed"
                        entry["postcondition"] = postcondition
                        entry["output_tail"] = str(postcondition["reason"])
        entries.append(entry)
        if entry["status"] == "failed" and args.stop_on_failure:
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
        path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def validate_sweep_summary(summary: Mapping[str, Any]) -> None:
    errors: list[str] = []
    if summary.get("schema") != 1:
        errors.append("schema must be 1")
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
    if not isinstance(summary.get("output_dir"), str) or not summary.get("output_dir"):
        errors.append("output_dir must be a non-empty string")
    options = summary.get("options")
    if not isinstance(options, Mapping):
        errors.append("options must be an object")
    else:
        for option in ("stop_on_failure", "include_int8", "require_cached_build"):
            if not isinstance(options.get(option), bool):
                errors.append(f"options.{option} must be a bool")
        compiler_version_file = options.get("compiler_version_file")
        if compiler_version_file is not None and not isinstance(compiler_version_file, str):
            errors.append("options.compiler_version_file must be a string or null")
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
    git = summary.get("git")
    git_dirty: bool | None = None
    if not isinstance(git, Mapping):
        errors.append("git must be an object")
    else:
        if not isinstance(git.get("commit"), str) or not git.get("commit"):
            errors.append("git.commit must be a non-empty string")
        if not isinstance(git.get("dirty"), bool):
            errors.append("git.dirty must be a bool")
        else:
            git_dirty = bool(git["dirty"])
        status_short = git.get("status_short")
        if not isinstance(status_short, list) or not all(isinstance(item, str) for item in status_short):
            errors.append("git.status_short must be a string list")
        elif git_dirty is not None and git_dirty is not bool(status_short):
            errors.append("git.dirty must match bool(git.status_short)")
    expected_skipped_preconditions = _skipped_preconditions(entries)
    if summary.get("skipped_preconditions") != expected_skipped_preconditions:
        errors.append("skipped_preconditions must match commands.preconditions")
    if entries:
        for entry in entries:
            if not isinstance(entry.get("category"), str) or not entry.get("category"):
                errors.append("commands[].category must be a non-empty string")
                break
            if not isinstance(entry.get("batch_size"), int) or isinstance(entry.get("batch_size"), bool) or entry.get("batch_size") <= 0:
                errors.append("commands[].batch_size must be a positive int")
                break
            if batch_sizes and entry.get("batch_size") not in batch_sizes:
                errors.append("commands[].batch_size must be listed in batch_sizes")
                break
            if not isinstance(entry.get("artifact_path"), str) or not entry.get("artifact_path"):
                errors.append("commands[].artifact_path must be a non-empty string")
                break
            command_text = entry.get("command")
            if not isinstance(command_text, str) or not command_text:
                errors.append("commands[].command must be a non-empty string")
                break
            argv = entry.get("argv")
            if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
                errors.append("commands[].argv must be a non-empty string list")
                break
            if command_text != shlex.join(argv):
                errors.append("commands[].command must match shlex.join(commands[].argv)")
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
                if category == "primitive":
                    expected_artifact_name = f"primitive-c{batch_size}.json"
                elif category == "serial_bridge":
                    expected_artifact_name = f"serial-bridge-c{batch_size}.json"
                elif category == "native_diagnostic":
                    expected_artifact_name = "native-baseline-c1.json" if batch_size == 1 else f"native-diagnostic-c{batch_size}.json"
                elif category == "int8_native_diagnostic":
                    expected_artifact_name = f"int8-native-diagnostic-c{batch_size}.json"
                if expected_artifact_name is not None and artifact_abs != (output_dir_abs / expected_artifact_name).resolve():
                    errors.append("commands[].artifact_path must match category/batch-size filename")
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
            if declared_batch_size is not None and declared_batch_size != entry.get("batch_size"):
                errors.append("commands[].batch_size must match commands[].argv --batch-size/--rows")
                break
            if entry.get("category") in {"serial_bridge", "native_diagnostic", "int8_native_diagnostic"}:
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
            expected_scripts_by_category = {
                "primitive": {"scripts/qwen35_batch_correctness.py"},
                "serial_bridge": {"scripts/qwen35_batch_serial_bench.py"},
                "native_diagnostic": {"scripts/qwen35_paro_bench.py", "scripts/qwen35_batch_retained_bench.py"},
                "int8_native_diagnostic": {"scripts/qwen35_batch_int8_diagnostic.py"},
            }
            expected_scripts = expected_scripts_by_category.get(entry.get("category"))
            if expected_scripts is not None and (len(argv) < 2 or argv[1] not in expected_scripts):
                errors.append("commands[].category must match commands[].argv script")
                break
            if entry.get("category") == "native_diagnostic" and entry.get("batch_size") != 1:
                retained_gate_flags = (
                    "--c1-baseline-json",
                    "--serial-bridge-json",
                    "--primitive-correctness-json",
                    "--profiler-json",
                )
                if any(not isinstance(_argv_value(argv, flag), str) or not _argv_value(argv, flag) for flag in retained_gate_flags):
                    errors.append("commands[].argv must include retained native gate artifact flags")
                    break
            status = entry.get("status")
            if status not in {"planned", "passed", "skipped", "failed"}:
                errors.append("commands[].status must be planned, passed, skipped, or failed")
                break
            dry_run = summary.get("dry_run")
            if isinstance(dry_run, bool):
                if dry_run and status != "planned":
                    errors.append("commands[].status must be planned for dry-run summaries")
                    break
                if not dry_run and status == "planned":
                    errors.append("commands[].status cannot be planned for executed summaries")
                    break
            returncode = entry.get("returncode")
            if status in {"planned", "skipped"}:
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
            if status == "planned" and float(duration_seconds) != 0.0:
                errors.append("commands[].duration_seconds must be zero for planned rows")
                break
            if status == "skipped" and float(duration_seconds) != 0.0:
                errors.append("commands[].duration_seconds must be zero for skipped rows")
                break
            if status == "planned" and "output_tail" in entry:
                errors.append("commands[].output_tail must be absent for planned rows")
                break
            if status == "planned" and any(
                field in entry for field in ("preconditions", "precondition", "postconditions", "postcondition")
            ):
                errors.append("commands[].conditions must be absent for planned rows")
                break
            if status != "planned" and not isinstance(entry.get("output_tail"), str):
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
                    if not isinstance(condition.get("kind"), str) or not condition.get("kind"):
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
                        if condition.get("passed") is False and (not isinstance(reason, str) or not reason):
                            errors.append("commands[].preconditions[].reason must be a non-empty string when failed")
                            condition_schema_error = True
                            break
                if condition_schema_error:
                    break
            if condition_schema_error:
                break
            if "preconditions" in entry and (entry.get("category") != "native_diagnostic" or entry.get("batch_size") == 1):
                errors.append("commands[].preconditions are only valid for retained native diagnostic rows")
                break
            if status == "failed" and isinstance(returncode, int) and returncode != 0 and any(
                field in entry for field in ("postconditions", "postcondition")
            ):
                errors.append("commands[].postconditions must be absent for failed rows with nonzero returncode")
                break
            if status == "skipped" and any(field in entry for field in ("postconditions", "postcondition")):
                errors.append("commands[].postconditions must be absent for skipped rows")
                break
            if "postconditions" in entry and (entry.get("category") != "native_diagnostic" or entry.get("batch_size") == 1):
                errors.append("commands[].postconditions are only valid for retained native diagnostic rows")
                break
            if (
                entry.get("category") == "native_diagnostic"
                and entry.get("batch_size") != 1
                and status == "passed"
                and "postconditions" not in entry
            ):
                errors.append("commands[].postconditions must include retained native postconditions for passed retained rows")
                break
            if isinstance(entry.get("postconditions"), list) and [
                condition.get("kind") for condition in entry["postconditions"]
            ] != ["retained_profiler_synthesis"]:
                errors.append("commands[].postconditions must include retained native postcondition kinds")
                break
            if entry.get("category") == "native_diagnostic" and entry.get("batch_size") != 1 and status != "planned":
                preconditions = entry.get("preconditions")
                expected_retained_kinds = ["primitive_correctness", "c1_baseline", "serial_bridge", "profiler_summary"]
                if not isinstance(preconditions, list) or [condition.get("kind") for condition in preconditions] != expected_retained_kinds:
                    errors.append("commands[].preconditions must include retained native gate kinds")
                    break
                expected_retained_precondition_paths = [
                    _argv_value(argv, "--primitive-correctness-json"),
                    _argv_value(argv, "--c1-baseline-json"),
                    _argv_value(argv, "--serial-bridge-json"),
                    _argv_value(argv, "--profiler-json"),
                ]
                if [condition.get("artifact_path") for condition in preconditions] != expected_retained_precondition_paths:
                    errors.append("commands[].preconditions[].artifact_path must match retained native gate argv")
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
                reason = postcondition.get("reason")
                if postcondition.get("passed") is True and reason is not None:
                    errors.append("commands[].postconditions[].reason must be null when passed")
                    break
                if postcondition.get("passed") is False and (not isinstance(reason, str) or not reason):
                    errors.append("commands[].postconditions[].reason must be a non-empty string when failed")
                    break
                profiler_fields = postcondition.get("profiler_synthesized_fields")
                precondition_fields = postcondition.get("profiler_precondition_synthesized_fields")
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
                profiler_precondition_fields = (
                    profiler_precondition.get("profiler_trace_synthesized_fields")
                    if isinstance(profiler_precondition, dict)
                    else None
                )
                if postcondition.get("passed") is True and (
                    not isinstance(profiler_precondition_fields, list)
                    or not all(isinstance(field, str) for field in profiler_precondition_fields)
                ):
                    errors.append("commands[].preconditions[].profiler_trace_synthesized_fields must be a string list when retained postcondition passed")
                    break
                if postcondition.get("passed") is True and list(precondition_fields) != list(profiler_precondition_fields):
                    errors.append("commands[].postconditions[].profiler_precondition_synthesized_fields must match profiler_summary precondition")
                    break
            failed_postconditions = [
                postcondition
                for postcondition in postconditions
                if isinstance(postcondition, dict) and postcondition.get("passed") is not True
            ] if isinstance(postconditions, list) else []
            if status == "passed" and returncode != 0:
                errors.append("commands[].status passed requires returncode 0")
                break
            if status == "passed" and failed_postconditions:
                errors.append("commands[].status passed cannot include failed postconditions")
                break
            if status == "passed" and summary.get("status") == "passed" and not Path(entry["artifact_path"]).is_file():
                errors.append("commands[].artifact_path must exist for passed summary rows")
                break
            if status == "failed" and returncode == 0 and not failed_postconditions:
                errors.append("commands[].status failed with returncode 0 requires a failed postcondition")
                break
            if status == "skipped":
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
                if entry.get("precondition") != failed_preconditions[0]:
                    errors.append("commands[].precondition must match the first failed precondition")
                    break
                if entry.get("output_tail") != failed_preconditions[0].get("reason"):
                    errors.append("commands[].output_tail must match skipped precondition reason")
                    break
            if git_dirty is not None and entry.get("git_dirty") is not git_dirty:
                errors.append("commands[].git_dirty must match git.dirty")
                break
    if summary.get("completed_command_count") != len(entries):
        errors.append("completed_command_count must match len(commands)")
    command_count = summary.get("command_count")
    if not isinstance(command_count, int) or isinstance(command_count, bool) or command_count < len(entries):
        errors.append("command_count must be an int greater than or equal to completed_command_count")
    elif isinstance(options, Mapping) and isinstance(options.get("include_int8"), bool) and batch_sizes:
        expected_plan: list[tuple[str, int]] = []
        for c in batch_sizes:
            expected_plan.extend([("primitive", c), ("serial_bridge", c), ("native_diagnostic", c)])
            if options["include_int8"] and c != 1:
                expected_plan.append(("int8_native_diagnostic", c))
        if command_count != len(expected_plan):
            errors.append("command_count must match batch_sizes/options.include_int8")
        elif [(entry.get("category"), entry.get("batch_size")) for entry in entries] != expected_plan[: len(entries)]:
            errors.append("commands[] category/batch_size order must match batch_sizes/options.include_int8")
    if isinstance(options, Mapping) and options.get("stop_on_failure") is True:
        for index, entry in enumerate(entries[:-1]):
            if entry.get("status") in {"failed", "skipped"}:
                errors.append("commands[] failed/skipped row must be final when stop_on_failure is true")
                break
    expected_status_counts = _status_counts(entries)
    if summary.get("status_counts") != expected_status_counts:
        errors.append("status_counts must match commands")
    expected_category_status_counts = _category_status_counts(entries)
    if summary.get("category_status_counts") != expected_category_status_counts:
        errors.append("category_status_counts must match commands")
    expected_status = _summary_status(entries)
    if summary.get("status") != expected_status:
        errors.append("status must match commands")
    expected_precondition_counts = _retained_precondition_counts(entries)
    if summary.get("retained_precondition_counts") != expected_precondition_counts:
        errors.append("retained_precondition_counts must match commands.preconditions")
    expected_postcondition_counts = _retained_postcondition_counts(entries)
    if summary.get("retained_postcondition_counts") != expected_postcondition_counts:
        errors.append("retained_postcondition_counts must match commands.postconditions")
    expected_failed_postconditions = _failed_postconditions(entries)
    if summary.get("failed_postconditions") != expected_failed_postconditions:
        errors.append("failed_postconditions must match commands.postconditions")
    for entry in entries:
        preconditions = entry.get("preconditions")
        if isinstance(preconditions, list):
            failed_preconditions = [
                precondition
                for precondition in preconditions
                if isinstance(precondition, dict) and precondition.get("passed") is not True
            ]
            if failed_preconditions and entry.get("precondition") != failed_preconditions[0]:
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
        if failed_postconditions and entry.get("postcondition") != failed_postconditions[0]:
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


def _summary_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "stop_on_failure": bool(args.stop_on_failure),
        "include_int8": bool(getattr(args, "include_int8", False)),
        "require_cached_build": bool(args.require_cached_build),
        "compiler_version_file": None if args.compiler_version_file is None else str(args.compiler_version_file),
    }


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
            status = "passed" if precondition.get("passed") is True else "failed"
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
            status = "passed" if postcondition.get("passed") is True else "failed"
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
        if entry.get("status") != "skipped":
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
    if any(entry["status"] == "failed" for entry in entries):
        return "failed"
    if any(entry["status"] == "skipped" for entry in entries):
        return "blocked"
    if all(entry["status"] == "planned" for entry in entries):
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
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--include-int8", action="store_true", help="Plan blocked INT8 KV c>N diagnostics for c>1 rows")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/hipengine-batch-c-sweep"))
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--validate-summary-json", type=Path, help="Validate an existing c-sweep summary JSON and exit")
    parser.add_argument("--dry-run", action="store_true", help="Write the command summary without executing commands")
    parser.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.validate_summary_json is not None:
        try:
            summary = json.loads(Path(args.validate_summary_json).read_text())
            if not isinstance(summary, Mapping):
                raise ValueError("summary root must be an object")
            validate_sweep_summary(summary)
        except Exception as exc:
            print(f"invalid c-sweep summary: {exc}", file=sys.stderr)
            return 1
        print("OK")
        return 0
    summary = run_sweep(args)
    print(json.dumps(summary, indent=2))
    return 1 if summary["status"] in {"failed", "blocked"} else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
