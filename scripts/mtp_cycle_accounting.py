#!/usr/bin/env python3
"""Extract per-cycle MTP2 accounting from a gguf_mtp_c1c8_server_bench raw run.

Scaling-campaign M0 instrumentation owner (docs/QWEN38-GFX1151-SCALING-CAMPAIGN.md).
Reads the raw AR/MTP suite JSON (resident_observability per-request
specdec2_mtp2_* records) and emits the per-width per-cycle accounting table:

- physical target passes per cycle and pass shapes (subgroup decomposition),
- ms per target row (exact per-pass samples),
- accepted draft tokens, committed output tokens per target pass,
- operation-complete ms per committed output token (complete wall / tokens),
- matched AR ms per decode output row at the same width.

Pass reconstruction: per tick t, every request r reports its subgroup's total
target rows (specdec2_mtp2_target_physical_rows[t]) and the same wall time of
that subgroup pass (specdec2_mtp2_target_pass_ms[t]). Requests are therefore
partitioned per tick by the exact (rows, sample) pair; each bucket is valid
only when sum over members of (candidate_counts[t] + 1) equals the reported
rows. No shape is hardcoded, so tail-shrink and width-4 partition cycles are
both reconstructed exactly.

Consistency invariants (violation = hard error):
- len(target_pass_ms) == target_batch_calls and sum(samples) == target_ms
- len(accept_pass_ms) == cycles and sum == accept_ms where present
- per request: sum(accepted_counts) + cycles == generated tokens

This script re-attributes recorded telemetry; it makes no new perf claim.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _request_records(cell: dict[str, Any]) -> list[dict[str, Any]]:
    ro = cell["mtp"].get("resident_observability") or {}
    return [
        r
        for r in ro.get("routes", {}).get("recent_completed", [])
        if "specdec2_mtp2_cycles" in r
    ]


def _passes_for_tick(records: list[dict[str, Any]], t: int) -> list[tuple[int, float]]:
    """Return [(target_rows, pass_ms), ...] for tick t, exactly reconstructed."""
    buckets: dict[tuple[int, float], list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        rows = int(rec["specdec2_mtp2_target_physical_rows"][t])
        ms = float(rec["specdec2_mtp2_target_pass_ms"][t])
        candidates = int(rec["specdec2_mtp2_candidate_counts"][t])
        buckets[(rows, ms)].append(i)
    passes: list[tuple[int, float]] = []
    for (rows, ms), members in buckets.items():
        member_rows = sum(
            int(records[i]["specdec2_mtp2_candidate_counts"][t]) + 1
            for i in members
        )
        if member_rows != rows:
            raise AssertionError(
                f"tick {t} bucket rows={rows} ms={ms}: member rows sum {member_rows} "
                "!= reported rows (impossible bucket: either telemetry drift or two "
                "passes shared byte-identical samples)"
            )
        passes.append((rows, ms))
    return passes


def _analyze_cell(cell: dict[str, Any]) -> dict[str, Any]:
    records = _request_records(cell)
    width = int(cell["width"])
    assert len(records) == width, f"{cell['prompt_id']} w{width}: {len(records)} records"
    cycles_set = {int(r["specdec2_mtp2_cycles"]) for r in records}
    assert len(cycles_set) == 1, f"ragged cycles {cycles_set}"
    cycles = cycles_set.pop()
    generated = int(cell["mtp"]["generated_tokens"])
    expected_committed = 0
    for rec in records:
        calls = int(rec["specdec2_mtp2_target_batch_calls"])
        rows = list(rec["specdec2_mtp2_target_physical_rows"])
        t_ms = list(rec.get("specdec2_mtp2_target_pass_ms", []))
        assert len(rows) == calls and len(t_ms) == calls
        assert abs(sum(t_ms) - float(rec["specdec2_mtp2_target_ms"])) < 1e-6
        a_ms = list(rec.get("specdec2_mtp2_accept_pass_ms", []))
        if a_ms:
            assert len(a_ms) == int(rec["specdec2_mtp2_cycles"])
            assert abs(sum(a_ms) - float(rec["specdec2_mtp2_accept_ms"])) < 1e-6
        acc = sum(int(x) for x in rec["specdec2_mtp2_accepted_counts"])
        assert len(rec["specdec2_mtp2_candidate_counts"]) == cycles
        # per-request committed = 1 bootstrap token + accepted + one visible
        # token per cycle; final-cycle overshoot is truncated at max_tokens
        expected_committed += 1 + acc + cycles
    residual = generated - expected_committed
    if abs(residual) > width:
        raise AssertionError(
            f"{cell['prompt_id']} w{width}: committed identity residual "
            f"{residual} exceeds width"
        )
    passes: list[tuple[int, float]] = []
    for t in range(cycles):
        passes.extend(_passes_for_tick(records, t))
    rows_total = sum(r for r, _ in passes)
    ms_total = sum(m for _, m in passes)
    accepted = sum(
        int(sum(int(x) for x in rec["specdec2_mtp2_accepted_counts"])) for rec in records
    )
    shapes: dict[int, int] = defaultdict(int)
    for r, _ in passes:
        shapes[r] += 1
    proposal_calls = sum(int(r["specdec2_mtp2_proposal_batch_calls"]) for r in records)
    draft_tokens = sum(
        int(sum(int(x) for x in r["specdec2_mtp2_candidate_counts"])) for r in records
    )
    proposal_ms = sum(float(r["specdec2_mtp2_proposal_ms"]) for r in records)
    accept_ms = sum(float(r["specdec2_mtp2_accept_ms"]) for r in records)
    provider_ms = sum(float(r["specdec2_mtp2_provider_update_ms"]) for r in records)
    commit_ms = sum(float(r["specdec2_mtp2_selected_commit_ms"]) for r in records)
    streaming = any(bool(r["specdec2_mtp2_prompt_streaming"]) for r in records)
    wall_ms = float(cell["mtp"]["wall_seconds"]) * 1000.0
    ar_wall_ms = float(cell["ar"]["wall_seconds"]) * 1000.0
    ar_generated = int(cell["ar"]["generated_tokens"])
    return {
        "prompt_id": cell["prompt_id"],
        "width": width,
        "cycles": cycles,
        "physical_target_passes": len(passes),
        "target_pass_shapes": {str(k): shapes[k] for k in sorted(shapes)},
        "target_rows_total": rows_total,
        "target_ms_total": ms_total,
        "proposal_ms_member_sum": proposal_ms,
        "accept_ms_member_sum": accept_ms,
        "provider_update_ms_member_sum": provider_ms,
        "selected_commit_ms_member_sum": commit_ms,
        "proposal_batch_calls_member_sum": proposal_calls,
        "prompt_streaming_engaged": streaming,
        "accepted_draft_tokens": accepted,
        "draft_tokens_generated": draft_tokens,
        "committed_output_tokens": generated,
        "complete_wall_ms": wall_ms,
        "ar_wall_ms": ar_wall_ms,
        "ar_generated_tokens": ar_generated,
        "exact": bool(cell["exact"]),
        "committed_identity_residual_tokens": generated - expected_committed,
        "mtp_engaged": bool(cell["mtp_engaged"]),
        "budget_conformed": bool(cell["mtp_budget_conformed"]),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, Any] = {
        "cells": len(rows),
        "prompts_exact": sum(1 for r in rows if r["exact"]),
        "prompts_engaged": sum(1 for r in rows if r["mtp_engaged"]),
        "prompts_budget_conformed": sum(1 for r in rows if r["budget_conformed"]),
    }
    for key in (
        "cycles",
        "physical_target_passes",
        "target_rows_total",
        "target_ms_total",
        "proposal_ms_member_sum",
        "accept_ms_member_sum",
        "provider_update_ms_member_sum",
        "selected_commit_ms_member_sum",
        "proposal_batch_calls_member_sum",
        "accepted_draft_tokens",
        "draft_tokens_generated",
        "committed_output_tokens",
        "complete_wall_ms",
        "ar_wall_ms",
        "ar_generated_tokens",
    ):
        agg[key] = sum(float(r[key]) for r in rows)
    cycles = agg["cycles"]
    passes = agg["physical_target_passes"]
    agg["target_passes_per_cycle"] = passes / cycles
    agg["ms_per_target_pass_mean"] = agg["target_ms_total"] / passes
    agg["ms_per_target_row"] = agg["target_ms_total"] / agg["target_rows_total"]
    agg["committed_tokens_per_target_pass"] = agg["committed_output_tokens"] / passes
    agg["committed_tokens_per_cycle"] = agg["committed_output_tokens"] / cycles
    agg["complete_ms_per_committed_token"] = (
        agg["complete_wall_ms"] / agg["committed_output_tokens"]
    )
    agg["ar_ms_per_output_row"] = agg["ar_wall_ms"] / agg["ar_generated_tokens"]
    agg["acceptance_rate"] = (
        agg["accepted_draft_tokens"] / agg["draft_tokens_generated"]
        if agg["draft_tokens_generated"]
        else None
    )
    shapes: dict[int, int] = defaultdict(int)
    for r in rows:
        for k, v in r["target_pass_shapes"].items():
            shapes[int(k)] += v
    agg["target_pass_shapes"] = {str(k): shapes[k] for k in sorted(shapes)}
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar-mtp-raw", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    raw = json.loads(args.ar_mtp_raw.read_text())
    cells = raw["cells"]
    per_width: dict[str, Any] = {}
    for width in sorted({int(c["width"]) for c in cells}):
        ws = [c for c in cells if int(c["width"]) == width]
        rows = [_analyze_cell(c) for c in ws]
        per_width[str(width)] = _aggregate(rows)
    payload = {
        "kind": "mtp_cycle_accounting",
        "source_commit": raw["source"]["commit"],
        "source_dirty": raw["source"]["dirty"],
        "protocol": raw["protocol"],
        "runtime_profile": raw["runtime_profile"],
        "model": raw["model"],
        "hardware": raw.get("hardware"),
        "elapsed_seconds": raw.get("elapsed_seconds"),
        "per_width": per_width,
    }
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    for w, a in sorted(per_width.items(), key=lambda kv: int(kv[0])):
        print(
            f"C{w}: cycles={a['cycles']:.0f} passes/cycle="
            f"{a['target_passes_per_cycle']:.2f} shapes={a['target_pass_shapes']} "
            f"ms/row={a['ms_per_target_row']:.3f} ms/token="
            f"{a['complete_ms_per_committed_token']:.3f} "
            f"AR ms/row={a['ar_ms_per_output_row']:.3f}"
        )
    print(json.dumps({"ok": True, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
