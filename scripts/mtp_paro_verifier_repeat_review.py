#!/usr/bin/env python3
"""Aggregate repeated PARO verifier captures without inflating quality rows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from scripts.mtp_paro_verifier_review_aggregate import aggregate


def aggregate_repeats(
    paths: Sequence[Path],
    *,
    expected_repeats: int = 3,
) -> dict[str, Any]:
    if expected_repeats <= 0:
        raise ValueError("expected_repeats must be positive")
    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        grouped[str(payload["prompt"]["name"])].append((path, payload))
    if not grouped:
        raise ValueError("at least one repeated capture is required")

    canonical_paths: list[Path] = []
    repeat_summary: dict[str, Any] = {}
    repeat_failures: list[str] = []
    for prompt in sorted(grouped):
        entries = sorted(grouped[prompt], key=lambda item: str(item[0]))
        if len(entries) != expected_repeats:
            raise ValueError(
                f"prompt {prompt!r} has {len(entries)} repeats, expected {expected_repeats}"
            )
        hashes = [str(payload["capture_sha256"]) for _path, payload in entries]
        unique = sorted(set(hashes))
        deterministic = len(unique) == 1
        if not deterministic:
            repeat_failures.append(prompt)
        canonical_paths.append(entries[0][0])
        repeat_summary[prompt] = {
            "captures": [str(path) for path, _payload in entries],
            "capture_sha256": hashes,
            "unique_capture_sha256": unique,
            "deterministic": deterministic,
        }

    result = aggregate(canonical_paths)
    result["schema"] = "hipengine.paro_mtp_verifier_full_repeat_review.v1"
    result["coverage"]["repeats_per_prompt"] = expected_repeats
    result["coverage"]["capture_files"] = len(paths)
    result["repeat_determinism"] = {
        "passed": not repeat_failures,
        "failed_prompts": repeat_failures,
        "prompts": repeat_summary,
    }
    result["checks"]["repeat_determinism"] = not repeat_failures
    numerical_keys = (
        "finite",
        "mean_kl",
        "p95_kl",
        "p99_kl",
        "max_kl",
        "top1",
        "per_scope",
        "repeat_determinism",
    )
    numerical_repeat_pass = all(bool(result["checks"][key]) for key in numerical_keys)
    result["review"]["numerical_and_repeat_gates_passed"] = numerical_repeat_pass
    result["status"] = (
        "numerical_repeat_pass_task_review_pending"
        if numerical_repeat_pass
        else "numerical_or_repeat_gate_failed"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate_repeats(args.inputs, expected_repeats=args.expected_repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                **result["aggregate"],
                "scope_failures": result["scope_failures"],
                "repeat_failures": result["repeat_determinism"]["failed_prompts"],
                "task_decision_mismatches": len(
                    result["review"]["task_decision_mismatches"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] != "numerical_or_repeat_gate_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
