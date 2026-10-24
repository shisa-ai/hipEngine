#!/usr/bin/env python3
"""Review repeat determinism without promoting numerical samples to policy gates.

One canonical trajectory per prompt supplies numerical observations. Every
repeat contributes execution provenance and must have the same model, backend
and manifests. Independent execution or repeatability failures fail the review;
otherwise status and policy_decision remain inconclusive, at any sample size.
"""

from __future__ import annotations

import argparse
import json
import re
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
    seen_paths: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen_paths:
            raise ValueError(f"duplicate capture path: {path}")
        seen_paths.add(resolved)
        payload = json.loads(path.read_text(encoding="utf-8"))
        digest = payload.get("capture_sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError(f"capture needs a usable SHA-256 hash: {path}")
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
        hashes = [payload["capture_sha256"].lower() for _path, payload in entries]
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
    result["schema"] = "hipengine.paro_mtp_verifier_full_repeat_review.v2"
    result["coverage"]["repeats_per_prompt"] = expected_repeats
    result["coverage"]["capture_files"] = len(paths)
    result["repeat_determinism"] = {
        "passed": not repeat_failures,
        "failed_prompts": repeat_failures,
        "prompts": repeat_summary,
    }
    result["checks"]["repeat_determinism"] = not repeat_failures
    repeat_reports = [aggregate([path]) for path in paths]
    for repeat in repeat_reports:
        if any(repeat[key] != result[key] for key in ("model", "backend", "manifests")):
            raise ValueError("all repeats must share model, backend and manifests")
    sources = [source for repeat in repeat_reports
               for source in repeat["provenance"]["source_captures"]]
    result["provenance"]["all_repeat_captures"] = sources
    for summary, field in (
        ("route_flags_attested_all", "route_flags_attested"),
        ("manifests_stable_all", "manifests_stable_during_capture"),
        ("execution_failures_known_all", "execution_failures_known"),
    ):
        result["provenance"][summary] = all(source[field] is True for source in sources)
    failures = sorted({(failure["prompt"], failure["check"])
                       for repeat in repeat_reports
                       for failure in repeat["review"]["source_execution_failures"]})
    result["review"]["source_execution_failures"] = [
        {"prompt": prompt, "check": check} for prompt, check in failures
    ]
    result["qualification"]["execution_failure_count"] = len(failures)
    result["review"]["aggregation_failures"] = sorted({
        failure for repeat in repeat_reports for failure in repeat["review"]["aggregation_failures"]
    })
    result["qualification"]["aggregation_failures"] = result["review"]["aggregation_failures"]
    result["execution_checks"] = {
        "canonical_aggregation": result["status"] != "failed",
        "repeat_determinism": not repeat_failures,
        "all_repeat_executions": all(repeat["status"] != "failed" for repeat in repeat_reports),
    }
    result["execution_failures"] = [
        name for name, passed in result["execution_checks"].items() if not passed
    ]
    result["status"] = "failed" if result["execution_failures"] else "inconclusive"
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
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
