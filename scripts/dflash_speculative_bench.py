#!/usr/bin/env python3
"""Build hipEngine DFlash/MTP speculative benchmark artifacts.

This is the torch-free benchmark-contract driver for the native DFlash lane.  It
ports the metric shape from the parent amd-gpu-tuning DFlash/MTP harnesses while
leaving PyTorch/HF hot loops behind.  Future native runners can emit row JSON and
this script normalizes it into the standard schema-2 artifact under
``benchmarks/results/``.

The script can also emit a synthetic schema fixture (clearly marked as not a
performance claim) so CI can validate the JSON contract before the native
verifier exists.
"""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.prompts import (  # noqa: E402
    DEFAULT_STABLE_PROMPT_FIXTURE,
    file_sha256,
    load_prompt_records,
    validate_prompt_records,
)
from hipengine.benchmark.speculative import (
    DEFAULT_DFLASH_DRAFTER,
    DEFAULT_TARGET_MODEL,
    SpeculativeBenchmarkModels,
    build_speculative_artifact,
    schema_fixture_row,
)


def _load_rows_from_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, dict):
        if isinstance(payload.get("rows"), list):
            return [dict(row) for row in payload["rows"]]
        measurements = payload.get("measurements")
        if isinstance(measurements, dict) and isinstance(measurements.get("rows"), list):
            return [dict(row) for row in measurements["rows"]]
    raise ValueError(f"{path} must contain a list of rows, {{'rows': [...]}} or {{'measurements': {{'rows': [...]}}}}")


def _load_rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(value)
    return rows


def _run_quiet(cmd: list[str], *, cwd: Path = REPO_ROOT) -> str | None:
    try:
        proc = subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True, timeout=15)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _git_context() -> dict[str, Any]:
    commit = _run_quiet(["git", "rev-parse", "HEAD"])
    branch = _run_quiet(["git", "branch", "--show-current"])
    porcelain = _run_quiet(["git", "status", "--porcelain"])
    return {
        "hipengine_commit": commit,
        "hipengine_branch": branch,
        "hipengine_dirty": bool(porcelain),
        "hipengine_status_porcelain": porcelain,
    }


def _read_optional_text(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.read_text()


def _parse_key_values(items: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in items:
        key, sep, value = raw.partition("=")
        if not sep:
            raise ValueError(f"expected KEY=VALUE, got {raw!r}")
        out[key] = value
    return out


def _default_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in ["python3", "scripts/dflash_speculative_bench.py", *argv])


def _prompt_suite_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, int] = {}
    categories: dict[str, int] = {}
    for record in records:
        groups[str(record.get("benchmark_group"))] = groups.get(str(record.get("benchmark_group")), 0) + 1
        categories[str(record.get("category"))] = categories.get(str(record.get("category")), 0) + 1
    return {
        "rows": len(records),
        "benchmark_groups": dict(sorted(groups.items())),
        "categories": dict(sorted(categories.items())),
        "prompt_tokens_min": min(int(row["prompt_tokens"]) for row in records) if records else 0,
        "prompt_tokens_max": max(int(row["prompt_tokens"]) for row in records) if records else 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-json", type=Path, help="JSON file containing rows or a prior artifact with measurements.rows")
    parser.add_argument("--rows-jsonl", type=Path, help="JSONL file containing one raw speculative row per line")
    parser.add_argument(
        "--emit-schema-fixture",
        action="store_true",
        help="Emit one synthetic row that exercises the complete DFlash metric contract (not a perf claim)",
    )
    parser.add_argument("--json", type=Path, required=True, help="Output artifact path under benchmarks/results/")
    parser.add_argument("--run-tag", default="dflash-speculative-benchmark-contract")
    parser.add_argument(
        "--summary",
        default="DFlash speculative benchmark contract for gfx1151 packed-target work",
    )
    parser.add_argument(
        "--status",
        default="diagnostic",
        choices=("accepted", "diagnostic", "diagnostic_retained", "blocked", "rejected_correctness", "rejected_variance"),
    )
    parser.add_argument("--target-name", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--target-path", default=None)
    parser.add_argument("--target-revision", default=None)
    parser.add_argument("--target-quant", default="w4_paro_packed")
    parser.add_argument("--drafter-name", default=DEFAULT_DFLASH_DRAFTER)
    parser.add_argument("--drafter-path", default=None)
    parser.add_argument("--drafter-revision", default=None)
    parser.add_argument("--drafter-dtype", default="bf16")
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--target-arch", default="gfx1151")
    parser.add_argument("--gpu", default="AMD RYZEN AI MAX+ 395 w/ Radeon 8060S")
    parser.add_argument("--prompt-suite", default=str(DEFAULT_STABLE_PROMPT_FIXTURE))
    parser.add_argument("--skip-prompt-suite-validation", action="store_true")
    parser.add_argument("--prompt-suite-sha256", default=None)
    parser.add_argument("--decode-tokens", type=int, default=0)
    parser.add_argument("--draft-budgets", default="2,4,8")
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--hardware-note", default=None)
    parser.add_argument("--software-note", default=None)
    parser.add_argument("--decision-reason", default=None)
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument(
        "--extra-workload",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional workload field to store in the artifact. Repeatable.",
    )
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    if args.rows_json is not None:
        rows.extend(_load_rows_from_json(args.rows_json))
    if args.rows_jsonl is not None:
        rows.extend(_load_rows_from_jsonl(args.rows_jsonl))
    if args.emit_schema_fixture:
        rows.append(schema_fixture_row())
    if not rows and args.status == "accepted":
        raise ValueError("--status accepted requires at least one row")

    models = SpeculativeBenchmarkModels(
        target_name=args.target_name,
        target_path=args.target_path,
        target_revision=args.target_revision,
        target_quant=args.target_quant,
        drafter_name=args.drafter_name,
        drafter_path=args.drafter_path,
        drafter_revision=args.drafter_revision,
        drafter_dtype=args.drafter_dtype,
    )
    hardware = {
        "gpu": args.gpu,
        "arch": args.target_arch,
        "backend": args.backend,
        "note": args.hardware_note,
    }
    software = {
        **_git_context(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hipcc_version": _read_optional_text(args.compiler_version_file),
        "note": args.software_note,
    }
    prompt_suite_summary = None
    prompt_suite_sha256 = args.prompt_suite_sha256
    if args.prompt_suite and prompt_suite_sha256 is None and Path(args.prompt_suite).exists():
        prompt_suite_sha256 = file_sha256(args.prompt_suite)
    if args.prompt_suite and not args.skip_prompt_suite_validation:
        prompt_records = load_prompt_records(args.prompt_suite)
        prompt_errors = validate_prompt_records(prompt_records)
        if prompt_errors:
            raise ValueError(f"prompt suite validation failed: {prompt_errors[:4]}")
        prompt_suite_summary = _prompt_suite_summary(prompt_records)

    workload = {
        "shape": "speculative_decode",
        "provider": "dflash",
        "verify_modes": ["verify_chain"],
        "prompt_suite": args.prompt_suite,
        "prompt_suite_sha256": prompt_suite_sha256,
        "prompt_suite_summary": prompt_suite_summary,
        "decode_tokens": args.decode_tokens or None,
        "draft_budgets": [int(x) for x in args.draft_budgets.split(",") if x.strip()],
        "same_session_ar_required": True,
        "speed_promotion_gate": ">1.10x AR and exact/finite correctness",
        **_parse_key_values(args.extra_workload),
    }
    commands = {
        "benchmark_contract": _default_command(sys.argv[1:] if argv is None else argv),
        "row_sources": {
            "rows_json": str(args.rows_json) if args.rows_json is not None else None,
            "rows_jsonl": str(args.rows_jsonl) if args.rows_jsonl is not None else None,
            "schema_fixture": bool(args.emit_schema_fixture),
        },
    }
    notes = list(args.note)
    if not rows:
        notes.append("No native DFlash rows supplied yet; artifact records the benchmark contract only.")

    artifact = build_speculative_artifact(
        run_tag=args.run_tag,
        summary=args.summary,
        rows=rows,
        models=models,
        status=args.status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        hardware=hardware,
        software=software,
        workload=workload,
        commands=commands,
        notes=notes,
        synthetic_schema_fixture=bool(args.emit_schema_fixture),
        decision_reason=args.decision_reason,
    )
    text = json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
