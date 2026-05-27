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
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/"
    "snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"
)
DEFAULT_FIXTURE = "fixtures/qwen35_paro/parent_512_32_seed1234.json"
DEFAULT_BATCH_SIZES = (1, 2, 4, 8)
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


def _command_arg_value(command: SweepCommand, flag: str) -> str | None:
    argv = list(command.argv)
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


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
        profiler_command = _profiler_command_label(profiler, payload if isinstance(payload, dict) else None)
        if profiler_command is None:
            reasons.append("profiler command is missing")
        else:
            if "rocprofv3" not in profiler_command or "--kernel-trace" not in profiler_command:
                reasons.append("profiler command does not include rocprofv3 --kernel-trace")
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
            command_profiler_path = _command_text_arg(profiler_command, "--profiler-json")
            if command_profiler_path != str(profiler_path):
                reasons.append("profiler command --profiler-json path does not match artifact_path")
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
        if profiler.get("expected_kernels_present") is not True:
            reasons.append("expected_kernels_present is not true")
        expected_kernel_names = profiler.get("expected_kernel_names")
        if not isinstance(expected_kernel_names, list) or not expected_kernel_names:
            reasons.append("expected_kernel_names is missing or empty")
        elif not all(isinstance(name, str) and name for name in expected_kernel_names):
            reasons.append("expected_kernel_names contains a non-string entry")
        elif not any("batch" in name.lower() for name in expected_kernel_names):
            reasons.append("expected_kernel_names does not include a native batch kernel")
        kernel_durations = profiler.get("kernel_durations_ns")
        if not isinstance(kernel_durations, dict) or not kernel_durations:
            reasons.append("kernel_durations_ns is missing or empty")
        elif isinstance(expected_kernel_names, list):
            for kernel_name in expected_kernel_names:
                duration_ns = kernel_durations.get(kernel_name)
                if isinstance(kernel_name, str) and kernel_name and (
                    not _is_number(duration_ns) or float(duration_ns) <= 0.0
                ):
                    reasons.append(f"kernel_durations_ns.{kernel_name} is missing or non-positive numeric")
                    break
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
                    "output_tail": proc.stdout[-4000:],
                }
            )
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
        "skipped_preconditions": _skipped_preconditions(entries),
        "status": _summary_status(entries),
    }
    if args.summary_json is not None:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


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
    parser.add_argument("--dry-run", action="store_true", help="Write the command summary without executing commands")
    parser.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_sweep(args)
    print(json.dumps(summary, indent=2))
    return 1 if summary["status"] in {"failed", "blocked"} else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
