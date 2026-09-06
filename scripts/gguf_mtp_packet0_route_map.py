#!/usr/bin/env python3
"""Packet 0 route map + cell inventory for the Qwen3.8 gfx1100 better-MTP campaign.

Digests one or more `gguf_mtp_c1c8_server_bench.py` output JSONs into a
per-(C,N,K) route map (admitted K, logical/padded frontier rows, proposal
groups, target passes, execution routes, graph activity, fallbacks, drain)
and a 24-cell C1-C8 x K1-K3 inventory status.

Usage:
    python scripts/gguf_mtp_packet0_route_map.py \
        --output campaign-artifacts/packet0/route-map.json \
        campaign-artifacts/packet0/p0-*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cell_width(cell: Mapping[str, Any]) -> int:
    return int(cell.get("width") or 0)


def _mtp_arm(cell: Mapping[str, Any]) -> Mapping[str, Any]:
    arm = cell.get("mtp")
    return arm if isinstance(arm, Mapping) else {}


def _routes(arm: Mapping[str, Any]) -> Mapping[str, Any]:
    ro = arm.get("resident_observability")
    if not isinstance(ro, Mapping):
        return {}
    routes = ro.get("routes")
    return routes if isinstance(routes, Mapping) else {}


def _adapter_contract(arm: Mapping[str, Any]) -> Mapping[str, Any]:
    ro = arm.get("resident_observability")
    if not isinstance(ro, Mapping):
        return {}
    adapter = ro.get("mtp2_adapter")
    return adapter if isinstance(adapter, Mapping) else {}


def _request_rows(arm: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    completed = _routes(arm).get("recent_completed")
    if not isinstance(completed, Sequence) or isinstance(completed, (str, bytes)):
        return []
    return [row for row in completed if isinstance(row, Mapping)]


def _route_map_for_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    arm = _mtp_arm(cell)
    width = _cell_width(cell)
    requests = _request_rows(arm)
    used = [row for row in requests if row.get("specdec2_mtp2_used")]
    candidate_counts: list[int] = []
    accepted_counts: list[int] = []
    target_rows: list[int] = []
    proposal_rows: list[int] = []
    proposal_batch_calls = 0
    target_batch_calls = 0
    commit_batch_calls = 0
    device_accept_calls = 0
    failures = 0
    catchups = 0
    routes_seen: set[str] = set()
    for row in used:
        for key, sink in (
            ("specdec2_mtp2_candidate_counts", candidate_counts),
            ("specdec2_mtp2_accepted_counts", accepted_counts),
            ("specdec2_mtp2_target_physical_rows", target_rows),
            ("specdec2_mtp2_proposal_physical_rows", proposal_rows),
        ):
            values = row.get(key)
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                sink.extend(int(value) for value in values)
        proposal_batch_calls += int(row.get("specdec2_mtp2_proposal_batch_calls") or 0)
        target_batch_calls += int(row.get("specdec2_mtp2_target_batch_calls") or 0)
        commit_batch_calls += int(
            row.get("specdec2_mtp2_selected_commit_batch_calls") or 0
        )
        device_accept_calls += int(row.get("specdec2_mtp2_device_accept_calls") or 0)
        failures += int(row.get("specdec2_mtp2_recoverable_failures") or 0)
        catchups += int(row.get("specdec2_mtp2_k0_catchups") or 0)
        values = row.get("specdec2_mtp2_execution_routes")
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            routes_seen.update(str(value) for value in values)
    adapter = _adapter_contract(arm)
    physical_width = adapter.get("physical_width")
    physical_width = physical_width if isinstance(physical_width, Mapping) else {}
    routes = _routes(arm)
    counts = routes.get("counts")
    counts = counts if isinstance(counts, Mapping) else {}
    graph = None
    ro = arm.get("resident_observability")
    if isinstance(ro, Mapping):
        buckets = ro.get("graph_buckets")
        if isinstance(buckets, Mapping):
            graph = {
                "captures": buckets.get("captures_total"),
                "replays": buckets.get("replays_total"),
                "invalidations": buckets.get("invalidations_total"),
            }
    admitted_k = sorted(set(candidate_counts)) if candidate_counts else []
    logical_r = width * (max(admitted_k) + 1) if admitted_k else None
    return {
        "width": width,
        "requests_with_mtp": len(used),
        "engaged": bool(used) and bool(cell.get("mtp_engaged")),
        "budget_conformed": bool(cell.get("mtp_budget_conformed")),
        "admitted_k": admitted_k,
        "logical_frontier_rows": logical_r,
        "padded_frontier_rows_max": max(target_rows) if target_rows else None,
        "proposal_physical_rows_max": max(proposal_rows) if proposal_rows else None,
        "proposal_batch_calls": proposal_batch_calls,
        "target_batch_calls": target_batch_calls,
        "device_accept_calls": device_accept_calls,
        "selected_commit_batch_calls": commit_batch_calls,
        "execution_routes": sorted(routes_seen),
        "recoverable_failures": failures,
        "k0_catchups": catchups,
        "accept_capacity_rows": physical_width.get("physical_accept_max_rows"),
        "resolved_partition": (physical_width.get("last_partition") or {}).get(
            "resolved_max_requests"
        )
        if isinstance(physical_width.get("last_partition"), Mapping)
        else None,
        "screening_cell": (physical_width.get("last_screening_cell") or None),
        "cycle_workspace_shape": (adapter.get("cycle_workspace") or {}).get("shape")
        if isinstance(adapter.get("cycle_workspace"), Mapping)
        else None,
        "graph": graph,
        "native_packed_decode_steps": counts.get("native_packed_decode_steps"),
        "physical_width_decode_steps": routes.get("physical_width_decode_steps"),
        "fallback_reasons": routes.get("fallback_reasons") or {},
        "tracked_memory_delta": arm.get("tracked_memory_delta"),
    }


def _cell_rate_row(cell: Mapping[str, Any]) -> dict[str, Any]:
    ar = cell.get("ar") or {}
    mtp = cell.get("mtp") or {}
    ar_rate = _num(ar.get("tok_s"))
    mtp_rate = _num(mtp.get("tok_s"))
    ratio = mtp_rate / ar_rate if ar_rate and mtp_rate else None
    return {
        "prompt_id": cell.get("prompt_id"),
        "category": cell.get("category"),
        "heldout": bool(cell.get("heldout")),
        "width": _cell_width(cell),
        "order": cell.get("order"),
        "ar_tok_s": ar_rate,
        "mtp_tok_s": mtp_rate,
        "mtp_vs_ar": ratio,
        "exact": bool(cell.get("exact")),
        "engaged": bool(cell.get("mtp_engaged")),
        "budget_conformed": bool(cell.get("mtp_budget_conformed")),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    files: dict[str, Any] = {}
    cell_routes: list[dict[str, Any]] = []
    rate_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for path in args.inputs:
        payload = json.loads(path.read_text())
        files[path.name] = {
            "status": payload.get("status"),
            "passed": payload.get("passed"),
            "model": payload.get("model"),
            "hardware": payload.get("hardware"),
            "runtime_profile": payload.get("runtime_profile"),
            "failure_reasons": payload.get("failure_reasons"),
        }
        summaries.append(
            {
                "file": path.name,
                "status": payload.get("status"),
                "passed": payload.get("passed"),
                "failure_reasons": payload.get("failure_reasons"),
            }
        )
        for cell in payload.get("cells") or []:
            if not isinstance(cell, Mapping):
                continue
            rate_rows.append(_cell_rate_row(cell))
            if cell.get("mtp_engaged") or _request_rows(_mtp_arm(cell)):
                route = _route_map_for_cell(cell)
                route["source"] = path.name
                route["prompt_id"] = cell.get("prompt_id")
                cell_routes.append(route)

    # Aggregate route map per (width, admitted-K) across prompts and files.
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for route in cell_routes:
        for k in route["admitted_k"] or [0]:
            grouped[(int(route["width"]), int(k))].append(route)
    route_map = []
    for (width, k), rows in sorted(grouped.items()):
        target_rows = [
            row["padded_frontier_rows_max"] for row in rows
            if row["padded_frontier_rows_max"] is not None
        ]
        route_map.append(
            {
                "C": width,
                "K": k,
                "cells": len(rows),
                "logical_R": width * (k + 1),
                "padded_P_observed": sorted(set(target_rows)),
                "proposal_batch_calls_total": sum(
                    row["proposal_batch_calls"] for row in rows
                ),
                "target_batch_calls_total": sum(
                    row["target_batch_calls"] for row in rows
                ),
                "accept_capacity_rows": rows[0]["accept_capacity_rows"],
                "execution_routes": sorted(
                    {r for row in rows for r in row["execution_routes"]}
                ),
                "recoverable_failures_total": sum(
                    row["recoverable_failures"] for row in rows
                ),
                "k0_catchups_total": sum(row["k0_catchups"] for row in rows),
                "screening_cells": [
                    row["screening_cell"] for row in rows if row["screening_cell"]
                ],
                "cycle_workspace_shape": rows[0]["cycle_workspace_shape"],
                "graph": rows[0]["graph"],
            }
        )

    # 24-cell inventory: engaged / rejected-before-mutation / unmeasured.
    measured: dict[tuple[int, int], dict[str, Any]] = {}
    for route in cell_routes:
        for k in route["admitted_k"] or [0]:
            key = (int(route["width"]), int(k))
            measured.setdefault(
                key,
                {"C": key[0], "K": key[1], "status": "engaged", "prompts": 0},
            )["prompts"] += 1
    # Median per-cell MTP/AR ratios from the rate rows, keyed by width.
    import statistics as _statistics

    ratios_by_width: dict[int, list[float]] = defaultdict(list)
    for row in rate_rows:
        if row["engaged"] and row["mtp_vs_ar"] is not None:
            ratios_by_width[int(row["width"])].append(float(row["mtp_vs_ar"]))
    inventory = []
    for width in range(1, 9):
        for k in range(1, 4):
            key = (width, k)
            if key in measured:
                entry = measured[key]
                ratios = ratios_by_width.get(width, [])
                entry["median_mtp_vs_ar_all_depths"] = (
                    round(_statistics.median(ratios), 4) if ratios else None
                )
                inventory.append(entry)
            elif width == 1:
                inventory.append(
                    {
                        "C": width,
                        "K": k,
                        "status": "rejected_before_mutation",
                        "reason": "partition_max_requests forces width-1 groups to K0; native physical C1 is Packet 2",
                    }
                )
            elif (width, k) == (2, 3):
                inventory.append(
                    {
                        "C": width,
                        "K": k,
                        "status": "rejected_before_mutation",
                        "reason": "no serving-evidence row covers K3 at resident capacity 2 (C2/K2 row caps at K2; C8/K3 row requires capacity 8)",
                    }
                )
            else:
                inventory.append({"C": width, "K": k, "status": "unmeasured"})

    result = {
        "schema": 1,
        "kind": "gfx1100-better-mtp-packet0-route-map",
        "inputs": summaries,
        "route_map": route_map,
        "inventory": inventory,
        "rate_rows": rate_rows,
    }
    args.output.write_text(json.dumps(result, indent=1, default=str))
    print(f"wrote {args.output} ({len(route_map)} route rows, {len(inventory)} inventory cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
