#!/usr/bin/env python3
"""Offline replay audit for MTP adaptive budget policies.

This does not predict real adaptive performance. It checks whether a policy can
be replayed exactly from fixed-budget prompt-suite traces by requiring evidence
for the chosen budget at the same generated-token offset.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


def _offset_map(summary: dict[str, Any]) -> dict[int, dict[str, int]]:
    accepted = [int(x) for x in (summary.get("accepted_lengths_by_run") or [[]])[0]]
    active = [int(x) for x in (summary.get("active_budgets_by_run") or [[]])[0]]
    offset = 0
    out: dict[int, dict[str, int]] = {}
    for accepted_len, active_budget in zip(accepted, active):
        out[int(offset)] = {
            "accepted": int(accepted_len),
            "active_budget": int(active_budget),
        }
        offset += 1 + int(accepted_len)
    return out


def _next_budget(current: int, *, direction: int, min_budget: int, max_budget: int) -> int:
    if direction > 0:
        return min(int(max_budget), int(current) + 1)
    return max(int(min_budget), int(current) - 1)


def _simulate_policy(
    result: dict[str, Any],
    *,
    decode_tokens: int,
    start_budget: int,
    promote_after_full: int,
    demote_after_zero: int,
    partial_action: str,
    min_budget: int,
    max_budget: int,
) -> dict[str, Any]:
    by_budget = result.get("by_budget") or {}
    maps = {int(b): _offset_map(by_budget[str(b)]) for b in range(min_budget, max_budget + 1)}

    offset = 0
    policy_budget = int(start_budget)
    full_streak = 0
    zero_streak = 0
    cycles: list[dict[str, int]] = []
    while offset < int(decode_tokens):
        remaining = int(decode_tokens) - int(offset)
        if remaining <= 1:
            return {
                "valid": True,
                "cycles": cycles,
                "terminal_ar_tokens": int(remaining),
            }
        active_budget = min(int(policy_budget), int(remaining) - 1)
        evidence = maps.get(int(active_budget), {}).get(int(offset))
        if evidence is None:
            return {
                "valid": False,
                "missing": {
                    "offset": int(offset),
                    "policy_budget": int(policy_budget),
                    "active_budget": int(active_budget),
                    "lookup_budget": int(active_budget),
                },
                "cycles": cycles,
            }

        accepted = min(int(evidence["accepted"]), int(active_budget))
        old_budget = int(policy_budget)
        reason = "keep"
        if accepted >= active_budget:
            full_streak += 1
            zero_streak = 0
            if active_budget == policy_budget and full_streak >= int(promote_after_full):
                policy_budget = _next_budget(policy_budget, direction=1, min_budget=min_budget, max_budget=max_budget)
                full_streak = 0
                reason = "promote_full_accept"
        elif accepted == 0:
            zero_streak += 1
            full_streak = 0
            if zero_streak >= int(demote_after_zero):
                policy_budget = _next_budget(policy_budget, direction=-1, min_budget=min_budget, max_budget=max_budget)
                zero_streak = 0
                reason = "demote_zero_accept"
        else:
            full_streak = 0
            zero_streak = 0
            if partial_action == "demote_one":
                policy_budget = _next_budget(policy_budget, direction=-1, min_budget=min_budget, max_budget=max_budget)
                reason = "demote_partial"
            elif partial_action == "min":
                policy_budget = int(min_budget)
                reason = "demote_partial_min"

        cycles.append(
            {
                "offset": int(offset),
                "policy_budget": int(old_budget),
                "active_budget": int(active_budget),
                "accepted": int(accepted),
                "next_policy_budget": int(policy_budget),
                "transition": reason != "keep",
            }
        )
        offset += 1 + int(accepted)

    return {"valid": True, "cycles": cycles, "terminal_ar_tokens": 0}


def _policy_grid(min_budget: int, max_budget: int) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for start_budget in range(int(min_budget), int(max_budget) + 1):
        for promote_after_full in (1, 2, 3):
            for demote_after_zero in (1, 2):
                for partial_action in ("keep", "demote_one", "min"):
                    policies.append(
                        {
                            "start_budget": int(start_budget),
                            "promote_after_full": int(promote_after_full),
                            "demote_after_zero": int(demote_after_zero),
                            "partial_action": partial_action,
                        }
                    )
    return policies


def replay(data: dict[str, Any], *, min_budget: int = 1, max_budget: int = 3) -> dict[str, Any]:
    results = data.get("results") or []
    decode_tokens = int(data.get("decode_tokens") or 0)
    policy_results: list[dict[str, Any]] = []
    for policy in _policy_grid(min_budget, max_budget):
        invalid: list[dict[str, Any]] = []
        for result in results:
            sim = _simulate_policy(
                result,
                decode_tokens=decode_tokens,
                min_budget=min_budget,
                max_budget=max_budget,
                **policy,
            )
            prompt_result = {
                "prompt": str(result.get("name")),
                "valid": bool(sim.get("valid")),
                "cycles_before_missing": len(sim.get("cycles") or []),
            }
            if not sim.get("valid"):
                prompt_result["missing"] = sim.get("missing")
                invalid.append(prompt_result)
        policy_results.append(
            {
                "policy": policy,
                "valid_all_prompts": not invalid,
                "invalid_prompt_count": len(invalid),
                "first_invalid": invalid[0] if invalid else None,
            }
        )

    valid = [p for p in policy_results if p["valid_all_prompts"]]
    invalid_counts: dict[str, int] = {}
    for policy in policy_results:
        if policy["valid_all_prompts"]:
            continue
        first = policy.get("first_invalid") or {}
        prompt = str(first.get("prompt"))
        invalid_counts[prompt] = invalid_counts.get(prompt, 0) + 1

    return {
        "schema": 1,
        "date": date.today().isoformat(),
        "status": "diagnostic_no_hold",
        "performance_claim": False,
        "purpose": "Check whether simple online MTP adaptive-budget policies can be exactly replayed from fixed-budget prompt-suite traces.",
        "source_artifact": str(data.get("_source_artifact", "")),
        "decode_tokens": int(decode_tokens),
        "budgets": list(range(int(min_budget), int(max_budget) + 1)),
        "fixed_budget_prompt_mean_speedup": {
            str(k): (data.get("aggregate_by_budget") or {}).get(str(k), {}).get("actual_decode_speedup_vs_ar_mean_across_prompts_mean")
            for k in range(int(min_budget), int(max_budget) + 1)
        },
        "policy_count": len(policy_results),
        "valid_policy_count": len(valid),
        "invalid_policy_count": len(policy_results) - len(valid),
        "invalid_first_prompt_counts": invalid_counts,
        "decision": "no_hold",
        "decision_reason": "All tested simple ladder policies land on at least one generated-token offset where the existing fixed-budget traces do not provide evidence for the chosen budget. The fixed-budget artifact is insufficient for an exact offline adaptive-policy promotion gate.",
        "policies": policy_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Prompt-suite economics JSON containing B=1/B=2/B=3 fixed-budget traces.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-budget", type=int, default=1)
    parser.add_argument("--max-budget", type=int, default=3)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    data["_source_artifact"] = str(args.input)
    artifact = replay(data, min_budget=int(args.min_budget), max_budget=int(args.max_budget))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "policy_count": artifact["policy_count"],
                "valid_policy_count": artifact["valid_policy_count"],
                "invalid_policy_count": artifact["invalid_policy_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
