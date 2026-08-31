#!/usr/bin/env python3
"""Run the canonical Qwen4Exp prompt-chunk ladder as isolated clean children."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "benchmarks" / "fixtures" / "qwen4exp_canonical_ar_p512_p1024_p4096.json"


def _valid_chunks(prompt_tokens: int, chunks: Sequence[int]) -> list[int]:
    prompt = int(prompt_tokens)
    return [int(chunk) for chunk in chunks if 0 < int(chunk) <= prompt]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--case-id", nargs="+")
    parser.add_argument("--chunk-size", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--child-root", type=Path, default=Path("/tmp/qwen4exp-chunk-sweep"))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    if not args.model_root.is_dir() or not args.fixture.is_file():
        raise ValueError("model root and fixture must exist")
    fixture = json.loads(args.fixture.read_text())
    selected = set(args.case_id or [row["id"] for row in fixture["cases"]])
    cases = [row for row in fixture["cases"] if row["id"] in selected]
    if {row["id"] for row in cases} != selected:
        raise ValueError("unknown case id in selection")
    chunks = tuple(dict.fromkeys(int(value) for value in args.chunk_size))
    if any(value <= 0 for value in chunks):
        raise ValueError("chunk sizes must be positive")
    args.child_root.mkdir(parents=True, exist_ok=True)
    children: list[dict[str, Any]] = []
    for chunk in chunks:
        eligible = [row for row in cases if chunk in _valid_chunks(int(row["prompt_tokens"]), chunks)]
        if not eligible:
            children.append({"chunk_size": chunk, "status": "not_applicable", "reason": "chunk exceeds every selected prompt"})
            continue
        output_path = args.child_root / f"run-chunk{chunk}.json"
        child_command = [
            sys.executable, str(ROOT / "scripts" / "qwen4exp_canonical_ar_bench.py"),
            "hipengine", "--model-root", str(args.model_root), "--fixture", str(args.fixture),
            "--case-id", *[str(row["id"]) for row in eligible],
            "--prefill-chunk-size", str(chunk), "--warmups", str(args.warmups),
            "--repetitions", str(args.repetitions), "--output", str(output_path),
        ]
        if args.compiler_version_file is not None:
            child_command += ["--compiler-version-file", str(args.compiler_version_file)]
        if args.require_cached_build:
            child_command += ["--require-cached-build"]
        result = subprocess.run(child_command, cwd=ROOT, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"chunk {chunk} child failed: {result.stderr[-2000:]}")
        payload = json.loads(output_path.read_text())
        children.append({
            "chunk_size": chunk, "status": payload["status"], "command": child_command,
            "fixture": str(args.fixture), "output": str(output_path), "source": payload["source"],
            "host": payload["host"], "profile": payload["profile"], "protocol": payload["protocol"],
            "summary": payload["summary"], "memory_before_close": payload["memory_before_close"],
            "memory_after_close": payload["memory_after_close"],
        })
        print(f"chunk={chunk} cases={len(eligible)} deterministic={payload['summary']['all_cases_deterministic']}", flush=True)
    completed = [row for row in children if row["status"] == "completed"]
    passed = bool(completed and all(row["summary"]["all_cases_deterministic"] and row["memory_after_close"]["current_allocated_bytes"] == 0 and row["source"]["tracked_clean"] for row in completed))
    return {"schema": 1, "kind": "qwen4exp_canonical_chunk_sweep", "status": "passed" if passed else "failed", "command": list(command), "model_root": str(args.model_root), "fixture": str(args.fixture), "selected_cases": [row["id"] for row in cases], "chunk_sizes": list(chunks), "children": children, "decision": {"passed": passed, "next": "select chunk by shape with correctness/memory controls"}}


def main() -> int:
    args = build_parser().parse_args(); payload = run(args, command=[str(Path(sys.argv[0]).name), *sys.argv[1:]])
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(args.output)})); return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__": raise SystemExit(main())
