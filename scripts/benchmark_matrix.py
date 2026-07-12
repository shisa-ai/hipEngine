#!/usr/bin/env python3
"""Run and assemble canonical direct/server PARO/GGUF benchmark matrices."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from hipengine.benchmark.matrix import (
    MatrixError,
    build_benchmark_matrix,
    validate_benchmark_matrix,
)
from hipengine.benchmark.provenance import collect_artifact_provenance


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MatrixError(f"{label} must contain a JSON object")
    return payload


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _run_manifest_commands(manifest: Mapping[str, Any], *, base_dir: Path) -> None:
    workdir_raw = manifest.get("workdir", ".")
    if not isinstance(workdir_raw, str) or not workdir_raw.strip():
        raise MatrixError("manifest.workdir must be a non-empty string when provided")
    workdir = _resolve(base_dir, workdir_raw)
    if not workdir.is_dir():
        raise MatrixError(f"manifest.workdir does not exist: {workdir}")
    raw_rows = manifest.get("rows")
    if not isinstance(raw_rows, list):
        raise MatrixError("manifest.rows must be a list")
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, Mapping):
            raise MatrixError(f"manifest.rows[{index}] must be an object")
        command = raw_row.get("command")
        if command is None:
            continue
        if not isinstance(command, list) or not command or not all(
            isinstance(part, str) and part for part in command
        ):
            raise MatrixError(f"manifest.rows[{index}].command must be a non-empty string list")
        raw_environment = raw_row.get("environment", {})
        if not isinstance(raw_environment, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in raw_environment.items()
        ):
            raise MatrixError(
                f"manifest.rows[{index}].environment must map strings to strings"
            )
        row_id = str(raw_row.get("id") or index)
        print(f"[matrix] run {row_id}: {' '.join(command)}", flush=True)
        completed = subprocess.run(
            command,
            cwd=workdir,
            env={**os.environ, **{str(key): str(value) for key, value in raw_environment.items()}},
            check=False,
        )
        if completed.returncode != 0:
            raise MatrixError(
                f"manifest row {row_id!r} command failed with exit {completed.returncode}"
            )


def _common(values: Sequence[Any], fallback: Any = None) -> Any:
    present = [value for value in values if value is not None]
    if present and all(value == present[0] for value in present):
        return present[0]
    return fallback


def _source_provenance(manifest: Mapping[str, Any], *, base_dir: Path) -> list[dict[str, Any]]:
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        return []
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("artifact"), str):
            continue
        artifact = _load_object(
            _resolve(base_dir, str(row["artifact"])),
            label=f"manifest.rows[{index}].artifact",
        )
        provenance = artifact.get("provenance")
        if isinstance(provenance, dict):
            result.append(provenance)
    return result


def _report_provenance(
    manifest: Mapping[str, Any],
    *,
    base_dir: Path,
    command: Sequence[str],
) -> dict[str, Any]:
    rows = _source_provenance(manifest, base_dir=base_dir)
    configured_backend = _common(
        [row.get("configured_backend") for row in rows], fallback="mixed"
    )
    resolved_backend = _common(
        [row.get("resolved_backend") for row in rows], fallback="mixed"
    )
    target_arch = _common([row.get("target_arch") for row in rows])
    device_name = _common([row.get("device_name") for row in rows])
    kv_dtype = _common([row.get("kv_dtype") for row in rows], fallback="mixed")
    return collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(configured_backend),
        resolved_backend=str(resolved_backend),
        target_arch=None if target_arch is None else str(target_arch),
        device_name=None if device_name is None else str(device_name),
        quant="mixed",
        kv_dtype=None if kv_dtype is None else str(kv_dtype),
        command=command,
        timing_protocol="exact_token_matrix_report_v1",
        warmups=0,
        repetitions=1,
        profiler={"enabled": False, "kind": None, "command": None},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    build = subparsers.add_parser("build", help="assemble a matrix from a manifest")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--json", type=Path, required=True)
    build.add_argument(
        "--run-commands",
        action="store_true",
        help="run each optional row command before loading its artifact",
    )
    build.add_argument(
        "--allow-ineligible",
        action="store_true",
        help="write and return success for a diagnostic matrix with failed eligibility gates",
    )

    validate = subparsers.add_parser("validate", help="validate an existing matrix artifact")
    validate.add_argument("--json", type=Path, required=True)
    return parser


def _build(args: argparse.Namespace, argv: Sequence[str]) -> int:
    manifest_path = args.manifest.resolve()
    manifest = _load_object(manifest_path, label="manifest")
    base_dir = manifest_path.parent
    if args.run_commands:
        _run_manifest_commands(manifest, base_dir=base_dir)
    report_provenance = _report_provenance(
        manifest,
        base_dir=base_dir,
        command=(str(Path(__file__)), *argv),
    )
    matrix = build_benchmark_matrix(
        manifest,
        base_dir=base_dir,
        report_provenance=report_provenance,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.json}")
    if not matrix["eligibility"]["passed"] and not args.allow_ineligible:
        for blocker in matrix["eligibility"]["blockers"]:
            print(f"blocked: {blocker}", file=sys.stderr)
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    try:
        if args.action == "validate":
            validate_benchmark_matrix(_load_object(args.json, label="matrix artifact"))
            print(f"valid matrix: {args.json}")
            return 0
        return _build(args, raw_argv)
    except MatrixError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
