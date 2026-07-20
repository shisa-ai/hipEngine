#!/usr/bin/env python3
"""Validate and roll up normalized coding-agent server turn records.

A0 is deliberately model-free. A later live-Uvicorn collector writes the
``hipengine_agentic_coding_records`` input consumed here; this command rejects
ambiguous token, timing, tool, batch-owner, or final-ownership data before an
artifact can support a performance comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.agentic import (  # noqa: E402
    DEFAULT_AGENTIC_WORKLOADS,
    AgenticBenchmarkError,
    build_agentic_benchmark_artifact,
    load_agentic_workload_suite,
)


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AgenticBenchmarkError(f"{path} must contain a JSON object")
    return payload


def build_artifact_from_paths(
    workloads_path: str | Path,
    records_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build an A0 artifact from checked-in workloads and normalized records."""

    suite = load_agentic_workload_suite(workloads_path)
    records = _load_json_object(records_path)
    artifact = build_agentic_benchmark_artifact(suite, records)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate normalized coding-agent turn records and emit an A0 artifact."
    )
    parser.add_argument(
        "--workloads",
        type=Path,
        default=DEFAULT_AGENTIC_WORKLOADS,
        help=f"Committed workload suite (default: {DEFAULT_AGENTIC_WORKLOADS})",
    )
    parser.add_argument("--records", type=Path, required=True, help="Normalized turn-record JSON")
    parser.add_argument("--json", type=Path, required=True, help="Output artifact path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        artifact = build_artifact_from_paths(args.workloads, args.records, args.json)
    except (AgenticBenchmarkError, json.JSONDecodeError, OSError) as exc:
        print(f"agentic benchmark rejected: {exc}", file=sys.stderr)
        return 2
    coverage = artifact["coverage"]
    print(
        "agentic benchmark A0 passed: "
        f"{coverage['agents']} agents, {coverage['turns']} turns, "
        f"{coverage['generated_tokens']} exact generated tokens -> {args.json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
