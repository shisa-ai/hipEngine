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
    if not isinstance(payload, dict):
        reasons.append("scaling reference artifact root is not an object")
    else:
        status = payload.get("status")
        if status in {"failed", "rejected", "rejected_correctness"}:
            reasons.append(f"status={status!r} is not usable as a scaling reference")
        aggregate, per_request = _extract_decode_rates(payload)
        if aggregate is None or per_request is None:
            reasons.append("decode throughput fields are missing")
        workload = payload.get("workload")
        if expected_concurrency is not None:
            concurrency = workload.get("concurrency") if isinstance(workload, dict) else None
            if concurrency != expected_concurrency:
                reasons.append(f"workload.concurrency={concurrency!r} does not match batch_size={expected_concurrency}")
    return {
        "kind": kind,
        "artifact_path": str(path),
        "passed": not reasons,
        "reason": None if not reasons else "; ".join(reasons),
    }


def _native_retained_precondition(command: SweepCommand) -> dict[str, Any] | None:
    if command.category != "native_diagnostic" or command.batch_size <= 1:
        return None
    for precondition in (
        _primitive_correctness_precondition(command),
        _scaling_reference_precondition(
            command,
            flag="--c1-baseline-json",
            kind="c1_baseline",
        ),
        _scaling_reference_precondition(
            command,
            flag="--serial-bridge-json",
            kind="serial_bridge",
            expected_concurrency=command.batch_size,
        ),
    ):
        if not precondition["passed"]:
            return precondition
    return precondition


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
            precondition = _native_retained_precondition(command)
            if precondition is not None and not precondition["passed"]:
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
