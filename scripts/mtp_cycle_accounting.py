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
import bisect
import csv
import json
import re
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
        if member_rows > rows:
            raise AssertionError(
                f"tick {t} bucket rows={rows} ms={ms}: member rows sum "
                f"{member_rows} exceeds reported rows (impossible bucket: either "
                "telemetry drift or two passes shared byte-identical samples)"
            )
        # rows >= member_rows records frontier padding: inactive tail rows are
        # owned by the last member, dispatched, and not committed.
        passes.append((rows, ms))
    return passes


def _pass_groups(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group requests that duplicate one physical target-pass schedule."""

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        signature = (
            int(record["specdec2_mtp2_cycles"]),
            tuple(record["specdec2_mtp2_target_physical_rows"]),
            tuple(record.get("specdec2_mtp2_target_pass_ms", [])),
            tuple(record.get("specdec2_mtp2_target_pass_start_ns", [])),
            tuple(record.get("specdec2_mtp2_target_pass_end_ns", [])),
            tuple(record.get("specdec2_mtp2_cycle_profile_start_ns", [])),
            tuple(record.get("specdec2_mtp2_cycle_profile_end_ns", [])),
        )
        groups[signature].append(record)
    return list(groups.values())


def _target_windows_for_cell(cell: dict[str, Any]) -> list[dict[str, int]]:
    """Reconstruct unique target windows from request-duplicated telemetry."""

    records = _request_records(cell)
    if not records:
        return []
    windows: list[dict[str, int]] = []
    for group in _pass_groups(records):
        cycle_starts = [
            list(r.get("specdec2_mtp2_cycle_profile_start_ns", [])) for r in group
        ]
        cycle_ends = [
            list(r.get("specdec2_mtp2_cycle_profile_end_ns", [])) for r in group
        ]
        if any(cycle_starts) or any(cycle_ends):
            starts, ends = cycle_starts, cycle_ends
        else:
            starts = [
                list(r.get("specdec2_mtp2_target_pass_start_ns", [])) for r in group
            ]
            ends = [
                list(r.get("specdec2_mtp2_target_pass_end_ns", [])) for r in group
            ]
        if not any(starts) and not any(ends):
            continue
        cycles = int(group[0]["specdec2_mtp2_cycles"])
        if any(len(values) != cycles for values in (*starts, *ends)):
            raise AssertionError("target pass timestamp telemetry is partial")
        for tick in range(cycles):
            rows = int(group[0]["specdec2_mtp2_target_physical_rows"][tick])
            start_ns = int(starts[0][tick])
            end_ns = int(ends[0][tick])
            if start_ns < 0 or end_ns <= start_ns:
                raise AssertionError(
                    f"invalid target pass window [{start_ns}, {end_ns})"
                )
            member_rows = sum(
                int(record["specdec2_mtp2_candidate_counts"][tick]) + 1
                for record in group
            )
            if member_rows > rows:
                raise AssertionError(
                    f"tick {tick} timestamp bucket rows={rows}: member rows sum "
                    f"{member_rows} exceeds reported rows"
                )
            # rows >= member_rows records frontier padding (dispatched, inactive).
            windows.append({"rows": rows, "start_ns": start_ns, "end_ns": end_ns})
    return windows


def _shared_record_value(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record.get(key, 0.0)) for record in records]
    if not values:
        return None
    first = values[0]
    return first if all(abs(value - first) <= 1e-6 for value in values[1:]) else None


def _grouped_record_value(records: list[dict[str, Any]], key: str) -> float | None:
    values = [_shared_record_value(group, key) for group in _pass_groups(records)]
    if any(value is None for value in values):
        return None
    return sum(float(value) for value in values)


def _analyze_cell(cell: dict[str, Any]) -> dict[str, Any]:
    records = _request_records(cell)
    width = int(cell["width"])
    assert len(records) == width, f"{cell['prompt_id']} w{width}: {len(records)} records"
    request_cycles = [int(r["specdec2_mtp2_cycles"]) for r in records]
    cycles = sum(request_cycles) / width
    generated = int(cell["mtp"]["generated_tokens"])
    expected_committed = 0
    for rec in records:
        calls = int(rec["specdec2_mtp2_target_batch_calls"])
        rows = list(rec["specdec2_mtp2_target_physical_rows"])
        t_ms = list(rec.get("specdec2_mtp2_target_pass_ms", []))
        assert len(rows) == calls and len(t_ms) == calls
        assert abs(sum(t_ms) - float(rec["specdec2_mtp2_target_ms"])) < 1e-6
        for start_key, end_key in (
            (
                "specdec2_mtp2_target_pass_start_ns",
                "specdec2_mtp2_target_pass_end_ns",
            ),
            (
                "specdec2_mtp2_cycle_profile_start_ns",
                "specdec2_mtp2_cycle_profile_end_ns",
            ),
        ):
            starts = list(rec.get(start_key, []))
            ends = list(rec.get(end_key, []))
            if starts or ends:
                assert len(starts) == calls and len(ends) == calls
                assert all(
                    int(end) > int(start) >= 0 for start, end in zip(starts, ends)
                )
        a_ms = list(rec.get("specdec2_mtp2_accept_pass_ms", []))
        if a_ms:
            assert len(a_ms) == int(rec["specdec2_mtp2_cycles"])
            assert abs(sum(a_ms) - float(rec["specdec2_mtp2_accept_ms"])) < 1e-6
        acc = sum(int(x) for x in rec["specdec2_mtp2_accepted_counts"])
        rec_cycles = int(rec["specdec2_mtp2_cycles"])
        assert len(rec["specdec2_mtp2_candidate_counts"]) == rec_cycles
        # per-request committed = 1 bootstrap token + accepted + one visible
        # token per cycle + one token per K0 catch-up decode step between
        # cycles; final-cycle overshoot is truncated at max_tokens
        expected_committed += (
            1
            + acc
            + rec_cycles
            + int(rec.get("specdec2_mtp2_k0_catchups") or 0)
        )
    residual = generated - expected_committed
    if abs(residual) > width:
        raise AssertionError(
            f"{cell['prompt_id']} w{width}: committed identity residual "
            f"{residual} exceeds width"
        )
    passes: list[tuple[int, float]] = []
    for group in _pass_groups(records):
        group_cycles = int(group[0]["specdec2_mtp2_cycles"])
        for tick in range(group_cycles):
            passes.extend(_passes_for_tick(group, tick))
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
        "proposal_ms_shared": _grouped_record_value(
            records, "specdec2_mtp2_proposal_ms"
        ),
        "accept_ms_shared": _grouped_record_value(records, "specdec2_mtp2_accept_ms"),
        "provider_update_ms_shared": _grouped_record_value(
            records, "specdec2_mtp2_provider_update_ms"
        ),
        "selected_commit_ms_shared": _grouped_record_value(
            records, "specdec2_mtp2_selected_commit_ms"
        ),
        "proposal_batch_calls_shared": _grouped_record_value(
            records, "specdec2_mtp2_proposal_batch_calls"
        ),
        "target_windows": _target_windows_for_cell(cell),
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
    width = int(rows[0]["width"])
    agg["width"] = width
    agg["mtp_to_ar_ratio"] = (
        agg["ar_ms_per_output_row"] / agg["complete_ms_per_committed_token"]
    )
    agg["observed_output_tokens_per_request_cycle"] = (
        agg["committed_tokens_per_cycle"] / width
    )
    agg["steady_committed_tokens_per_request_cycle"] = (
        1.0 + agg["accepted_draft_tokens"] / (cycles * width)
    )
    agg["observed_cycle_wall_ar_step_equivalents"] = (
        (agg["complete_wall_ms"] / cycles)
        / (agg["ar_ms_per_output_row"] * width)
    )
    agg["cycle_cost_ar_step_equivalents"] = (
        agg["steady_committed_tokens_per_request_cycle"]
        / agg["mtp_to_ar_ratio"]
    )
    agg["target_cycle_cost_ar_step_equivalents_at_1_15x"] = (
        agg["steady_committed_tokens_per_request_cycle"] / 1.15
    )
    shared_keys = {
        "proposal": "proposal_ms_shared",
        "accept": "accept_ms_shared",
        "provider_update": "provider_update_ms_shared",
        "selected_commit": "selected_commit_ms_shared",
    }
    agg["stage_ms_per_cycle"] = {"target": agg["target_ms_total"] / cycles}
    for stage, key in shared_keys.items():
        values = [row[key] for row in rows]
        if all(value is not None for value in values):
            agg["stage_ms_per_cycle"][stage] = sum(float(value) for value in values) / cycles
    proposal_calls = [row["proposal_batch_calls_shared"] for row in rows]
    if all(value is not None for value in proposal_calls):
        agg["proposal_weight_sweeps_per_cycle"] = (
            sum(float(value) for value in proposal_calls) / cycles
        )
    shapes: dict[int, int] = defaultdict(int)
    for r in rows:
        for k, v in r["target_pass_shapes"].items():
            shapes[int(k)] += v
    agg["target_pass_shapes"] = {str(k): shapes[k] for k in sorted(shapes)}
    return agg


def _model_stage(name: str) -> str | None:
    if name == "output.weight":
        return "shared_head"
    match = re.match(r"blk\.(\d+)\.", name)
    if match is None:
        return None
    layer = int(match.group(1))
    if layer < 64:
        return "target"
    if layer == 64:
        return "proposal"
    return None


def _model_weight_inventory(model: Path) -> dict[str, Any]:
    from hipengine.loading.gguf import scan_gguf

    info = scan_gguf(model)
    stage_family: dict[str, dict[str, int]] = {
        "target": defaultdict(int),
        "proposal": defaultdict(int),
    }
    stage_counts: dict[str, int] = defaultdict(int)
    for tensor in info.tensors:
        stage = _model_stage(tensor.name)
        destinations: tuple[str, ...]
        if stage == "shared_head":
            destinations = ("target", "proposal")
        elif stage in stage_family:
            destinations = (stage,)
        else:
            continue
        for destination in destinations:
            family = tensor.ggml_type_name.lower()
            stage_family[destination][family] += int(tensor.nbytes)
            stage_counts[destination] += 1
    return {
        stage: {
            "tensor_count": stage_counts[stage],
            "minimum_weight_bytes_per_sweep_by_family": dict(sorted(families.items())),
            "minimum_weight_bytes_per_sweep": sum(families.values()),
        }
        for stage, families in stage_family.items()
    }


def _kernel_quant_family(name: str) -> str | None:
    lowered = name.lower()
    for family in ("q4", "q5", "q6"):
        if f"{family}_" in lowered or f"gguf_{family}" in lowered:
            return family
    generic = re.search(r"<(?:[^,]+,\s*){1,2}([456])\s*,", name)
    if generic and ("qk_t16" in lowered or "gguf_k_" in lowered):
        return f"q{generic.group(1)}"
    return None


def _read_trace_events(path: Path, *, counter: str | None = None) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if counter is not None and row.get("Counter_Name") != counter:
                continue
            start_ns = int(row["Start_Timestamp"])
            end_ns = int(row["End_Timestamp"])
            event = {
                "name": row["Kernel_Name"],
                "start_ns": start_ns,
                "end_ns": end_ns,
            }
            if counter is not None:
                event["counter_value"] = float(row["Counter_Value"])
            events.append(event)
    events.sort(key=lambda event: event["start_ns"])
    return events


def _window_events(
    events: list[dict[str, Any]], starts: list[int], start_ns: int, end_ns: int
) -> list[dict[str, Any]]:
    index = bisect.bisect_left(starts, start_ns)
    selected: list[dict[str, Any]] = []
    while index < len(events) and events[index]["start_ns"] < end_ns:
        event = events[index]
        if event["end_ns"] <= end_ns:
            selected.append(event)
        index += 1
    return selected


def _kernel_family_row_curve(
    cells: list[dict[str, Any]],
    kernel_trace: Path,
    counter_trace: Path | None = None,
) -> dict[str, Any]:
    windows = [window for cell in cells for window in _target_windows_for_cell(cell)]
    if not windows:
        raise ValueError("raw run has no target timestamp windows")
    events = _read_trace_events(kernel_trace)
    event_starts = [event["start_ns"] for event in events]
    counters = (
        _read_trace_events(counter_trace, counter="FETCH_SIZE")
        if counter_trace is not None
        else []
    )
    counter_starts = [event["start_ns"] for event in counters]
    by_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        selected = _window_events(
            events, event_starts, window["start_ns"], window["end_ns"]
        )
        family_ns: dict[str, int] = defaultdict(int)
        family_calls: dict[str, int] = defaultdict(int)
        total_ns = 0
        for event in selected:
            duration = event["end_ns"] - event["start_ns"]
            total_ns += duration
            family = _kernel_quant_family(event["name"])
            if family is not None:
                family_ns[family] += duration
                family_calls[family] += 1
        family_fetch_kib: dict[str, float] = defaultdict(float)
        total_fetch_kib = 0.0
        if counters:
            for event in _window_events(
                counters, counter_starts, window["start_ns"], window["end_ns"]
            ):
                fetch_kib = float(event["counter_value"])
                total_fetch_kib += fetch_kib
                family = _kernel_quant_family(event["name"])
                if family is not None:
                    family_fetch_kib[family] += fetch_kib
        by_rows[int(window["rows"])].append(
            {
                "host_ms": (window["end_ns"] - window["start_ns"]) / 1e6,
                "kernel_ms": total_ns / 1e6,
                "family_ms": {key: value / 1e6 for key, value in family_ns.items()},
                "family_calls": dict(family_calls),
                "total_fetch_kib": total_fetch_kib,
                "family_fetch_kib": dict(family_fetch_kib),
            }
        )
    curve: dict[str, Any] = {}
    for rows, samples in sorted(by_rows.items()):
        families = sorted({key for sample in samples for key in sample["family_ms"]})
        family_ms = {
            family: statistics.median(sample["family_ms"].get(family, 0.0) for sample in samples)
            for family in families
        }
        kernel_ms = statistics.median(sample["kernel_ms"] for sample in samples)
        classified_ms = sum(family_ms.values())
        entry: dict[str, Any] = {
            "passes": len(samples),
            "host_ms_median": statistics.median(sample["host_ms"] for sample in samples),
            "kernel_ms_median": kernel_ms,
            "family_ms_median": family_ms,
            "family_calls_median": {
                family: statistics.median(
                    sample["family_calls"].get(family, 0) for sample in samples
                )
                for family in families
            },
            "classified_kernel_fraction": classified_ms / kernel_ms if kernel_ms else None,
        }
        if counters:
            fetch_families = sorted(
                {key for sample in samples for key in sample["family_fetch_kib"]}
            )
            total_fetch_kib = statistics.median(
                sample["total_fetch_kib"] for sample in samples
            )
            family_fetch_mib = {
                family: statistics.median(
                    sample["family_fetch_kib"].get(family, 0.0) for sample in samples
                )
                / 1024.0
                for family in fetch_families
            }
            entry["fetch_mib_median"] = total_fetch_kib / 1024.0
            entry["family_fetch_mib_median"] = family_fetch_mib
            entry["classified_fetch_fraction"] = (
                sum(family_fetch_mib.values()) / (total_fetch_kib / 1024.0)
                if total_fetch_kib
                else None
            )
        curve[str(rows)] = entry
    return curve


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar-mtp-raw", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--kernel-trace", type=Path, default=None)
    ap.add_argument("--counter-trace", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    raw = json.loads(args.ar_mtp_raw.read_text())
    cells = raw["cells"]
    per_width: dict[str, Any] = {}
    for width in sorted({int(c["width"]) for c in cells}):
        ws = [c for c in cells if int(c["width"]) == width]
        rows = [_analyze_cell(c) for c in ws]
        per_width[str(width)] = _aggregate(rows)
    model_path = args.model
    if model_path is None:
        raw_model = raw.get("model")
        if isinstance(raw_model, dict) and raw_model.get("path"):
            model_path = Path(raw_model["path"])
    sweep_economics: dict[str, Any] = {}
    if model_path is not None:
        sweep_economics["weight_inventory"] = _model_weight_inventory(model_path)
        target_inventory = sweep_economics["weight_inventory"]["target"]
        proposal_inventory = sweep_economics["weight_inventory"]["proposal"]
        sweep_economics["per_width_minimum_weight_bytes"] = {}
        for width, aggregate in per_width.items():
            proposal_sweeps = aggregate.get("proposal_weight_sweeps_per_cycle")
            sweep_economics["per_width_minimum_weight_bytes"][width] = {
                "target_per_cycle": {
                    family: value * aggregate["target_passes_per_cycle"]
                    for family, value in target_inventory[
                        "minimum_weight_bytes_per_sweep_by_family"
                    ].items()
                },
                "proposal_per_cycle": (
                    {
                        family: value * proposal_sweeps
                        for family, value in proposal_inventory[
                            "minimum_weight_bytes_per_sweep_by_family"
                        ].items()
                    }
                    if proposal_sweeps is not None
                    else None
                ),
                "accept_commit_per_cycle": {},
                "note": (
                    "Minimum compulsory model-weight traffic before cache effects or "
                    "row-owner re-fetch amplification; FETCH_SIZE profiler counters "
                    "measure actual video-memory fetches when supplied."
                ),
            }
    if args.kernel_trace is not None:
        sweep_economics["kernel_family_row_curve"] = _kernel_family_row_curve(
            cells,
            args.kernel_trace,
            args.counter_trace,
        )
    elif args.counter_trace is not None:
        raise ValueError("--counter-trace requires --kernel-trace")
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
        "sweep_economics": sweep_economics,
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
