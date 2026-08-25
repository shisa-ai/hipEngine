#!/usr/bin/env python3
"""Validate and aggregate the independent gfx1100 SPECDEC2 performance bridge.

The module is intentionally lightweight: P1 runtime children emit one row per
request/arm, while this owner enforces common timing, denominator, provenance,
and counterbalance contracts before any speed ratio is retained.  Runtime lane
execution is added behind the same row contract; model logic remains in the
existing dense and PARO harnesses.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence


CANONICAL_PROMPTS: tuple[tuple[str, str, str], ...] = (
    ("code_merge_intervals", "code", "train"),
    ("code_topological_sort", "code", "train"),
    ("code_lru_cache", "code", "train"),
    ("code_markdown_table", "code", "heldout"),
    ("general_en_plan", "general_en", "train"),
    ("general_en_explain", "general_en", "heldout"),
    ("general_ja_plan", "general_ja", "train"),
    ("general_ja_explain", "general_ja", "heldout"),
    ("mixed_ja_en_translate", "mixed_ja_en", "train"),
    ("mixed_ja_en_review", "mixed_ja_en", "heldout"),
)
ARMS = ("true_ar", "direct", "staged")
LANES = ("gguf", "paro")
PROFILES = ("strict", "production")
REQUIRED_TOP_LEVEL_STAGES = (
    "tokenize",
    "admission",
    "claims_reserve",
    "target_prefill",
    "provider_prompt_prime",
    "provider_open",
    "resident_owner_transition",
    "cycle_total",
    "output_publish",
    "claims_release",
    "terminal_reclaim",
)
_HEX_RE = re.compile(r"^[0-9a-f]+$")


def _tuple_text(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values)
    if not result or any(not value for value in result):
        raise ValueError(f"{name} must contain non-empty values")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def build_execution_plan(
    *,
    lane: str,
    profiles: Sequence[str],
    candidate_budgets: Sequence[int],
    runs: int,
    prompt_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Return content-agnostic prompt/run ordering for all three bridge arms."""

    selected_lane = str(lane)
    if selected_lane not in LANES:
        raise ValueError(f"lane must be one of {LANES!r}")
    selected_profiles = _tuple_text(profiles, name="profiles")
    if any(profile not in PROFILES for profile in selected_profiles):
        raise ValueError(f"profiles must be a subset of {PROFILES!r}")
    budgets = tuple(int(value) for value in candidate_budgets)
    if not budgets or any(value <= 0 for value in budgets):
        raise ValueError("candidate_budgets must be positive")
    if len(set(budgets)) != len(budgets):
        raise ValueError("candidate_budgets must not contain duplicates")
    repeat_count = int(runs)
    if repeat_count <= 0:
        raise ValueError("runs must be positive")
    prompts = _tuple_text(prompt_ids, name="prompt_ids")

    plan: list[dict[str, Any]] = []
    for run_index in range(repeat_count):
        for prompt_index, prompt_id in enumerate(prompts):
            parity = (run_index + prompt_index) % 2
            arm_order = ARMS if parity == 0 else tuple(reversed(ARMS))
            plan.append(
                {
                    "lane": selected_lane,
                    "run_index": run_index,
                    "prompt_index": prompt_index,
                    "prompt_id": prompt_id,
                    "profiles": selected_profiles,
                    "candidate_budgets": budgets,
                    "arm_order": arm_order,
                    "counterbalance_parity": parity,
                }
            )
    return plan


def _is_hash(value: Any, *, lengths: tuple[int, ...]) -> bool:
    text = str(value)
    return len(text) in lengths and _HEX_RE.fullmatch(text) is not None


def _finite_nonnegative(value: Any, *, label: str, positive: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or (positive and number <= 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be finite and {qualifier}")
    return number


def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["lane"]),
        str(row["profile"]),
        int(row["concurrency"]),
        int(row["candidate_budget"]),
        int(row["run_index"]),
        str(row["prompt_id"]),
        str(row["arm"]),
    )


def validate_bridge_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    lane: str,
    profiles: Sequence[str],
    candidate_budgets: Sequence[int],
    runs: int,
    max_tokens: int,
    require_full_suite: bool = True,
    strict_generated_ids: bool = True,
) -> list[dict[str, Any]]:
    """Validate bridge rows and return detached JSON-ready mappings.

    Ratios are legal only when every `(profile,C,K,run,prompt)` group contains a
    separate true-AR row, direct control, and staged row under one clean source.
    """

    selected_lane = str(lane)
    if selected_lane not in LANES:
        raise ValueError(f"lane must be one of {LANES!r}")
    selected_profiles = _tuple_text(profiles, name="profiles")
    budgets = tuple(int(value) for value in candidate_budgets)
    repeat_count = int(runs)
    output_tokens = int(max_tokens)
    if not budgets or any(value <= 0 for value in budgets):
        raise ValueError("candidate_budgets must be positive")
    if repeat_count <= 0 or output_tokens <= 1:
        raise ValueError("runs must be positive and max_tokens must exceed one")
    detached = [json.loads(json.dumps(dict(row), allow_nan=False)) for row in rows]
    if not detached:
        raise ValueError("bridge rows are empty")

    prompt_contract = {prompt_id: (category, split) for prompt_id, category, split in CANONICAL_PROMPTS}
    observed_prompts = {str(row.get("prompt_id")) for row in detached}
    if require_full_suite and observed_prompts != set(prompt_contract):
        raise ValueError("bridge rows do not cover the canonical prompt suite")
    if not observed_prompts <= set(prompt_contract):
        raise ValueError("bridge rows contain an unknown canonical prompt id")

    seen_keys: set[tuple[Any, ...]] = set()
    timing_owner_ids: set[str] = set()
    grouped: dict[tuple[str, int, int, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in detached:
        if int(row.get("schema", 0)) != 1:
            raise ValueError("bridge row schema must be 1")
        if str(row.get("lane")) != selected_lane:
            raise ValueError("bridge row lane does not match the selected lane")
        profile = str(row.get("profile"))
        if profile not in selected_profiles:
            raise ValueError("bridge row profile is outside the selected profiles")
        arm = str(row.get("arm"))
        if arm not in ARMS:
            raise ValueError(f"bridge arm must be one of {ARMS!r}")
        prompt_id = str(row.get("prompt_id"))
        expected_category, expected_split = prompt_contract[prompt_id]
        if (str(row.get("category")), str(row.get("split"))) != (
            expected_category,
            expected_split,
        ):
            raise ValueError("bridge prompt category/split identity is invalid")
        run_index = int(row.get("run_index", -1))
        order_index = int(row.get("order_index", -1))
        concurrency = int(row.get("concurrency", 0))
        budget = int(row.get("candidate_budget", 0))
        if run_index not in range(repeat_count) or order_index not in range(len(ARMS)):
            raise ValueError("bridge run/order index is out of range")
        if concurrency <= 0 or budget not in budgets:
            raise ValueError("bridge C/K cell is outside the selected packet")
        if int(row.get("max_tokens", 0)) != output_tokens:
            raise ValueError("bridge row max_tokens does not match the packet")
        ids = row.get("generated_token_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(token, int) for token in ids):
            raise ValueError("bridge rows require exact generated_token_ids")

        timing = row.get("timing")
        if not isinstance(timing, dict):
            raise ValueError("bridge rows require timing ownership")
        if timing.get("timing_owner") is not True:
            raise ValueError("each request/arm requires exactly one timing owner")
        owner_id = str(timing.get("timing_owner_id") or "")
        if not owner_id:
            raise ValueError("each request/arm requires exactly one timing owner")
        if owner_id in timing_owner_ids:
            raise ValueError(f"duplicate timing owner: {owner_id}")
        timing_owner_ids.add(owner_id)
        if str(timing.get("timing_scope")) != "request":
            raise ValueError("P1 C1 bridge timing scope must be request")
        complete = _finite_nonnegative(
            timing.get("complete_request_seconds"),
            label="complete_request_seconds",
            positive=True,
        )
        decode = _finite_nonnegative(
            timing.get("decode_only_seconds"),
            label="decode_only_seconds",
            positive=True,
        )
        if decode > complete + max(1e-9, complete * 1e-6):
            raise ValueError("decode-only timing exceeds complete request wall")
        stages = timing.get("top_level_stage_seconds")
        if not isinstance(stages, dict) or set(stages) != set(REQUIRED_TOP_LEVEL_STAGES):
            raise ValueError("bridge row is missing required top-level timing stages")
        stage_total = sum(
            _finite_nonnegative(value, label=f"stage {name}")
            for name, value in stages.items()
        )
        residual = _finite_nonnegative(
            timing.get("unattributed_seconds"), label="unattributed_seconds"
        )
        tolerance = max(1e-9, complete * 1e-6)
        if abs(stage_total + residual - complete) > tolerance:
            raise ValueError("top-level timing stages do not reconcile with complete wall")

        route = row.get("route")
        if not isinstance(route, dict) or not str(route.get("realized") or ""):
            raise ValueError("bridge row requires a realized route")
        if arm == "true_ar":
            if route.get("true_autoregressive_path") is not True:
                raise ValueError("speed ratios require a separate true AR denominator")
            if route.get("staged_generation2") or route.get("direct_control"):
                raise ValueError("true AR denominator cannot be a speculative control")
            if int(row.get("realized_candidate_budget", -1)) != 0:
                raise ValueError("true AR denominator must realize K0")
        elif arm == "direct" and route.get("direct_control") is not True:
            raise ValueError("direct arm did not realize the direct control")
        elif arm == "staged" and route.get("staged_generation2") is not True:
            raise ValueError("staged arm did not realize Generation-2")

        manifests = row.get("manifests")
        if not isinstance(manifests, dict) or not all(
            _is_hash(manifests.get(name), lengths=(64,))
            for name in ("selected_sha256", "strict_sha256")
        ):
            raise ValueError("bridge row has malformed selected/strict manifest hashes")
        provenance = row.get("provenance")
        if not isinstance(provenance, dict) or not _is_hash(
            provenance.get("commit"), lengths=(40, 64)
        ):
            raise ValueError("bridge row has malformed source provenance")
        if any(
            bool(provenance.get(name))
            for name in ("staged_dirty", "unstaged_dirty", "untracked_dirty")
        ):
            raise ValueError("retained bridge rows require clean, not dirty provenance")

        key = _row_key(row)
        if key in seen_keys:
            raise ValueError(f"duplicate bridge row: {key!r}")
        seen_keys.add(key)
        group_key = (profile, concurrency, budget, run_index, prompt_id)
        grouped[group_key][arm] = row

    expected_groups = {
        (profile, 1, budget, run_index, prompt_id)
        for profile in selected_profiles
        for budget in budgets
        for run_index in range(repeat_count)
        for prompt_id in observed_prompts
    }
    if set(grouped) != expected_groups:
        raise ValueError("bridge rows do not cover every requested profile/C/K/run/prompt cell")
    prompt_positions = {
        prompt_id: index
        for index, (prompt_id, _category, _split) in enumerate(CANONICAL_PROMPTS)
    }
    for (profile, concurrency, budget, run_index, prompt_id), by_arm in grouped.items():
        if set(by_arm) != set(ARMS):
            raise ValueError("each bridge cell requires true_ar, direct, and staged arms")
        expected_order = (
            ARMS
            if (run_index + prompt_positions[prompt_id]) % 2 == 0
            else tuple(reversed(ARMS))
        )
        realized_order = tuple(
            arm for arm, _row in sorted(by_arm.items(), key=lambda item: int(item[1]["order_index"]))
        )
        if realized_order != expected_order:
            raise ValueError("bridge arm order violates the content-agnostic counterbalance")
        ar_ids = by_arm["true_ar"]["generated_token_ids"]
        if strict_generated_ids and any(by_arm[arm]["generated_token_ids"] != ar_ids for arm in ("direct", "staged")):
            raise ValueError("strict bridge generated IDs differ from true AR")
        commits = {str(by_arm[arm]["provenance"]["commit"]) for arm in ARMS}
        if len(commits) != 1:
            raise ValueError("bridge arms do not share one source commit")
    return detached


def _aggregate_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = sum(float(row["timing"]["complete_request_seconds"]) for row in rows)
    decode = sum(float(row["timing"]["decode_only_seconds"]) for row in rows)
    tokens = sum(len(row["generated_token_ids"]) for row in rows)
    stages: dict[str, float] = defaultdict(float)
    categories: dict[str, dict[str, float]] = defaultdict(
        lambda: {"complete_request_seconds": 0.0, "decode_only_seconds": 0.0, "generated_tokens": 0.0}
    )
    for row in rows:
        for name, value in row["timing"]["top_level_stage_seconds"].items():
            stages[str(name)] += float(value)
        category = categories[str(row["category"])]
        category["complete_request_seconds"] += float(row["timing"]["complete_request_seconds"])
        category["decode_only_seconds"] += float(row["timing"]["decode_only_seconds"])
        category["generated_tokens"] += len(row["generated_token_ids"])
    return {
        "rows": len(rows),
        "generated_tokens": tokens,
        "complete_request_seconds": complete,
        "decode_only_seconds": decode,
        "complete_tokens_per_second": tokens / complete,
        "decode_tokens_per_second": tokens / decode,
        "top_level_stage_seconds": dict(sorted(stages.items())),
        "categories": dict(sorted(categories.items())),
        "samples": [
            {
                "run_index": int(row["run_index"]),
                "prompt_id": str(row["prompt_id"]),
                "order_index": int(row["order_index"]),
                "complete_request_seconds": float(row["timing"]["complete_request_seconds"]),
                "decode_only_seconds": float(row["timing"]["decode_only_seconds"]),
            }
            for row in rows
        ],
    }


def aggregate_bridge_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate already-validated rows without inventing an AR denominator."""

    cells: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            f"{row['lane']}:{row['profile']}:c{int(row['concurrency'])}:"
            f"k{int(row['candidate_budget'])}"
        )
        cells[key].append(row)
    payload: dict[str, Any] = {}
    for key, cell_rows in sorted(cells.items()):
        by_arm = {
            arm: _aggregate_arm([row for row in cell_rows if row["arm"] == arm])
            for arm in ARMS
        }
        ar = by_arm["true_ar"]
        speedups = {
            arm: (
                by_arm[arm]["complete_tokens_per_second"]
                / ar["complete_tokens_per_second"]
            )
            for arm in ("direct", "staged")
        }
        payload[key] = {
            "arms": by_arm,
            "complete_speedup_vs_true_ar": speedups,
            "staged_speedup_vs_direct": (
                by_arm["staged"]["complete_tokens_per_second"]
                / by_arm["direct"]["complete_tokens_per_second"]
            ),
        }
    return {
        "schema": 1,
        "kind": "specdec2_perf_gfx1100_bridge_aggregate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cells": payload,
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one complete checkpoint next to its destination, then replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_text(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _csv_int(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _csv_text(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="write a content-agnostic arm plan")
    plan.add_argument("--lane", choices=LANES, required=True)
    plan.add_argument("--profiles", type=_csv_text, required=True)
    plan.add_argument("--candidate-budgets", type=_csv_int, required=True)
    plan.add_argument("--runs", type=int, default=3)
    plan.add_argument("--prompt-ids", type=_csv_text, default=tuple(item[0] for item in CANONICAL_PROMPTS))
    plan.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="validate and aggregate emitted rows")
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument("--lane", choices=LANES, required=True)
    validate.add_argument("--profiles", type=_csv_text, required=True)
    validate.add_argument("--candidate-budgets", type=_csv_int, required=True)
    validate.add_argument("--runs", type=int, default=3)
    validate.add_argument("--max-tokens", type=int, default=25)
    validate.add_argument("--allow-partial-suite", action="store_true")
    validate.add_argument("--production-id-drift", action="store_true")
    validate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        payload = {
            "schema": 1,
            "kind": "specdec2_perf_gfx1100_bridge_plan",
            "plan": build_execution_plan(
                lane=args.lane,
                profiles=args.profiles,
                candidate_budgets=args.candidate_budgets,
                runs=args.runs,
                prompt_ids=args.prompt_ids,
            ),
        }
    else:
        source = json.loads(args.input.read_text(encoding="utf-8"))
        source_rows = source.get("rows") if isinstance(source, dict) else source
        if not isinstance(source_rows, list):
            raise ValueError("bridge input must be a row list or an object containing rows")
        validated = validate_bridge_rows(
            source_rows,
            lane=args.lane,
            profiles=args.profiles,
            candidate_budgets=args.candidate_budgets,
            runs=args.runs,
            max_tokens=args.max_tokens,
            require_full_suite=not args.allow_partial_suite,
            strict_generated_ids=not args.production_id_drift,
        )
        payload = aggregate_bridge_rows(validated)
        payload["rows"] = validated
    atomic_write_json(args.output, payload)
    print(json.dumps({"status": "passed", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
