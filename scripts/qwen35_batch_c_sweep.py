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
        commands.append(
            SweepCommand(
                category="native_diagnostic",
                batch_size=c,
                artifact_path=native_json,
                argv=tuple(
                    _batch_bench_argv(
                        "scripts/qwen35_batch_retained_bench.py",
                        args,
                        batch_size=c,
                        artifact_path=native_json,
                    )
                ),
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
        "git": git,
        "commands": entries,
        "status": _summary_status(entries),
    }
    if args.summary_json is not None:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _summary_status(entries: Sequence[dict[str, Any]]) -> str:
    if any(entry["status"] == "failed" for entry in entries):
        return "failed"
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
    return 1 if summary["status"] == "failed" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
