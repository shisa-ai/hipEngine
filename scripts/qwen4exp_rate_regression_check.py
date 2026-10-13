#!/usr/bin/env python3
"""Check a Qwen4Exp canonical AR run against a committed baseline run.

Both inputs are artifacts written by ``scripts/qwen4exp_canonical_ar_bench.py
hipengine`` over the same fixture on the same host. For every prompt length
the candidate's weighted prefill and decode rates must stay within a
tolerance of the baseline's, every case must be deterministic, and, when
``--require-identical-output`` is given, every case must produce the
baseline's output token ids. The exit status is 0 when every check passes.

Rates from two different machines are not comparable; the check refuses
artifacts whose machine ids differ.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


def _machine_id(artifact: Mapping[str, Any]) -> str | None:
    host = artifact.get("host")
    return None if not isinstance(host, Mapping) else host.get("machine_id")


def compare_runs(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    prefill_tolerance: float = 0.03,
    decode_tolerance: float = 0.03,
    require_identical_output: bool = False,
) -> dict[str, Any]:
    """Return per-shape deltas and the list of failed checks."""

    failures: list[str] = []
    if baseline.get("status") != "completed" or candidate.get("status") != "completed":
        failures.append("both runs must have status 'completed'")
    if baseline.get("fixture_sha256") != candidate.get("fixture_sha256"):
        failures.append("runs used different fixtures")
    base_machine, cand_machine = _machine_id(baseline), _machine_id(candidate)
    if base_machine is None or base_machine != cand_machine:
        failures.append("runs are not from the same physical host")
    base_shapes = baseline.get("summary", {}).get("shapes", {})
    cand_shapes = candidate.get("summary", {}).get("shapes", {})
    if set(base_shapes) != set(cand_shapes) or not base_shapes:
        failures.append("runs cover different prompt lengths")
    shapes: dict[str, Any] = {}
    for shape in sorted(set(base_shapes) & set(cand_shapes), key=int):
        row: dict[str, Any] = {}
        for metric, tolerance in (
            ("prefill_tok_s_weighted", prefill_tolerance),
            ("decode_tok_s_weighted", decode_tolerance),
        ):
            before = float(base_shapes[shape][metric])
            after = float(cand_shapes[shape][metric])
            delta = after / before - 1.0
            row[metric] = {"baseline": before, "candidate": after, "delta": delta}
            if delta < -tolerance:
                failures.append(
                    f"p{shape} {metric}: {before:.2f} -> {after:.2f} "
                    f"({delta * 100.0:+.2f}%, limit -{tolerance * 100.0:.1f}%)"
                )
        shapes[shape] = row
    base_cases = baseline.get("summary", {}).get("cases", {})
    cand_cases = candidate.get("summary", {}).get("cases", {})
    if set(base_cases) != set(cand_cases):
        failures.append("runs cover different cases")
    changed_output: list[str] = []
    for case in sorted(set(base_cases) & set(cand_cases)):
        if not cand_cases[case].get("deterministic", False):
            failures.append(f"{case}: candidate output is not deterministic")
        if (
            base_cases[case].get("output_token_ids_sha256")
            != cand_cases[case].get("output_token_ids_sha256")
        ):
            changed_output.append(case)
    if require_identical_output and changed_output:
        failures.append("output token ids changed: " + ", ".join(changed_output))
    return {
        "passed": not failures,
        "failures": failures,
        "shapes": shapes,
        "changed_output_cases": changed_output,
        "prefill_tolerance": prefill_tolerance,
        "decode_tolerance": decode_tolerance,
        "require_identical_output": require_identical_output,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--prefill-tolerance", type=float, default=0.03)
    parser.add_argument("--decode-tolerance", type=float, default=0.03)
    parser.add_argument("--require-identical-output", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = compare_runs(
        json.loads(args.baseline.read_text()),
        json.loads(args.candidate.read_text()),
        prefill_tolerance=args.prefill_tolerance,
        decode_tolerance=args.decode_tolerance,
        require_identical_output=args.require_identical_output,
    )
    result["baseline"] = str(args.baseline)
    result["candidate"] = str(args.candidate)
    for shape, row in result["shapes"].items():
        prefill = row["prefill_tok_s_weighted"]
        decode = row["decode_tok_s_weighted"]
        print(
            f"p{shape}: prefill {prefill['baseline']:.1f} -> {prefill['candidate']:.1f} "
            f"({prefill['delta'] * 100.0:+.2f}%), decode {decode['baseline']:.2f} -> "
            f"{decode['candidate']:.2f} ({decode['delta'] * 100.0:+.2f}%)"
        )
    if result["changed_output_cases"]:
        print("output token ids changed: " + ", ".join(result["changed_output_cases"]))
    for failure in result["failures"]:
        print("FAIL: " + failure)
    print("PASS" if result["passed"] else "FAILED")
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
