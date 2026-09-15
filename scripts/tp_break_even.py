#!/usr/bin/env python3
"""TP2 break-even projection from a measured synchronized group time.

This is the Packet 0 go/no-go input for Packet 3. It does not measure an engine;
it answers one question with measured numbers on both sides:

    how much per-token collective time can TP2 afford before it stops winning?

Model
-----

A TP2 decode step is one *synchronized group*: at every dependency boundary the
group waits for its slowest rank. With the two-term model

    token_time = fixed + weights / bandwidth

that gives

    T2 = max_rank(fixed_ms) + max_rank(rank_weight_ms) + collective_ms

and the comparison is against the *faster* matched TP1 arm, because a TP2 group
is one result, not one result per rank:

    baseline = min_rank(T1_ms)
    speedup  = baseline / T2
    C*       = baseline - max_rank(fixed_ms) - max_rank(rank_weight_ms)

``fixed`` covers attention over the KV cache, the GDN recurrence, launch
overhead and everything else that does not shrink when weights are halved. Only
its *share* of the TP1 token time is assumed, and the projection is reported
across a range of shares, so no single guess decides the verdict. ``bandwidth``
is not assumed either: it is implied by the TP1 row itself, dividing rather than
multiplying by the share of the token that is weight traffic
(``bandwidth = weights / ((1 - fixed_share) * T1)``).

Guards
------

Two conditions decide whether a projection may be called certified, and both are
recorded in the artifact:

- **Matched protocols.** Every TP1 arm must declare its workload shape, and the
  arms must agree. Combining a 512-prompt INT8-KV row with an 8192-token BF16 row
  is not one protocol, and the previous report did exactly that.
- **Pre-Packet-3 gate.** The packet plan requires local shard-shaped kernel
  measurements before Packet 3. Pass ``--shard-kernel-evidence`` to record that
  they exist; without it the projection is reported as uncertified.

The collective input must come from a measurement, not a default: pass
``--dependent-chain-artifact`` to read the per-reduction marginal out of a
dependent chain artifact, or state ``--marginal-us`` explicitly. Both are
recorded with their source. ``--dependent-chain-mode`` selects which structure
supplies the marginal, and the artifact's own dependency check is enforced, so a
structure whose reductions collapsed instead of chaining is refused rather than
projected.

Usage:
    python3 scripts/tp_break_even.py \
        --tp1 "W7900=27.9:15.652:8.646:512/128/int8-kv" \
        --tp1 "XTX=29.82:15.652:7.009:8192/8/bf16-kv" \
        --dependent-chain-artifact benchmarks/results/2026-09-14-w7900-tp2-dependent-reduction-chain.json \
        --dependent-chain-mode staged_exchange_host_sync \
        --reduction-points 128 \
        --json benchmarks/results/tp2_break_even_staged_exchange_host_sync.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_FIXED_SHARES = (0.0, 0.1, 0.2, 0.3)
#: The planning aspiration. The design accepts any qualified net improvement, so
#: missing this is reported as an unmet aspiration rather than as a failure.
TARGET_SPEEDUP = 1.3

#: The gate that actually decides whether a TP2 group is worth building: it must
#: be faster than the faster matched TP1 arm. A row at or below this is a genuine
#: blocker, because the group would not pay for itself at that fixed share.
BEATS_TP1_SPEEDUP = 1.0


def parse_tp1(spec: str) -> dict[str, Any]:
    """Parse ``NAME=TOK_S:TP1_GIB:RANK_GIB:PROTOCOL``.

    ``TP1_GIB`` is the whole-model weight bytes one GPU reads per token at TP1
    (what TP2 halves), and ``RANK_GIB`` is the rank-local shard this GPU would
    read at TP2. ``PROTOCOL`` names the workload shape the measurement used
    (prompt/decode/KV configuration); it is required because two arms measured
    under different shapes are not a matched pair.
    """

    if "=" not in spec:
        raise ValueError(f"expected NAME=TOK_S:TP1_GIB:RANK_GIB:PROTOCOL, got {spec!r}")
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"expected NAME=TOK_S:TP1_GIB:RANK_GIB:PROTOCOL, got {spec!r} "
            "(the workload shape is required: arms measured under different "
            "shapes are not a matched pair)"
        )
    tok_s_text, tp1_text, rank_text, protocol = parts
    name = name.strip()
    protocol = protocol.strip()
    if not name:
        raise ValueError(f"missing device name in {spec!r}")
    if not protocol:
        raise ValueError(f"missing workload protocol in {spec!r}")
    tok_s = float(tok_s_text)
    tp1_gib = float(tp1_text)
    rank_gib = float(rank_text)
    if tok_s <= 0 or tp1_gib <= 0 or rank_gib <= 0:
        raise ValueError(f"values must be positive in {spec!r}")
    if rank_gib > tp1_gib:
        raise ValueError(f"rank shard {rank_gib} exceeds TP1 weights {tp1_gib} in {spec!r}")
    return {
        "name": name,
        "tok_s": tok_s,
        "tp1_gib": tp1_gib,
        "rank_gib": rank_gib,
        "protocol": protocol,
    }


def rank_projection(device: dict[str, Any], *, fixed_share: float) -> dict[str, Any]:
    """One rank's contribution to a synchronized TP2 group."""

    tp1_ms = 1000.0 / float(device["tok_s"])
    weight_share = 1.0 - float(fixed_share)
    weight_ms = tp1_ms * weight_share
    # bandwidth = weights / weight_time: the share divides, it does not multiply.
    bandwidth_gbs = float(device["tp1_gib"]) / (weight_ms / 1000.0) if weight_ms > 0 else None
    fixed_ms = tp1_ms * float(fixed_share)
    rank_weight_ms = float(device["rank_gib"]) / bandwidth_gbs * 1000.0 if bandwidth_gbs else 0.0
    return {
        "device": device["name"],
        "protocol": device["protocol"],
        "tp1_ms_per_token": tp1_ms,
        "implied_bandwidth_gbs": bandwidth_gbs,
        "fixed_ms_per_token": fixed_ms,
        "rank_weight_ms_per_token": rank_weight_ms,
        "rank_share_of_token": rank_weight_ms / tp1_ms if tp1_ms else None,
    }


def project_group(
    devices: list[dict[str, Any]],
    *,
    collective_ms: float,
    fixed_share: float,
) -> dict[str, Any]:
    """Project one TP2 group time against the faster matched TP1 arm.

    The group waits for its slowest rank at each dependency boundary, so the
    fixed and rank-weight terms take the maximum across ranks, and the result is
    compared with the *fastest* TP1 arm. Reporting a separate TP2 throughput per
    rank would double-count one group as two independent results.
    """

    per_rank = [rank_projection(device, fixed_share=fixed_share) for device in devices]
    fixed_group_ms = max(entry["fixed_ms_per_token"] for entry in per_rank)
    rank_weight_group_ms = max(entry["rank_weight_ms_per_token"] for entry in per_rank)
    baseline_ms = min(entry["tp1_ms_per_token"] for entry in per_rank)
    tp2_group_ms = fixed_group_ms + rank_weight_group_ms + float(collective_ms)
    break_even_ms = baseline_ms - fixed_group_ms - rank_weight_group_ms
    return {
        "fixed_share": float(fixed_share),
        "collective_ms_per_token": float(collective_ms),
        "baseline_tp1_ms_per_token": baseline_ms,
        "baseline_device": min(per_rank, key=lambda entry: entry["tp1_ms_per_token"])["device"],
        "fixed_group_ms_per_token": fixed_group_ms,
        "rank_weight_group_ms_per_token": rank_weight_group_ms,
        "tp2_group_ms_per_token": tp2_group_ms,
        "projected_speedup": baseline_ms / tp2_group_ms,
        "tp2_tok_s": 1000.0 / tp2_group_ms,
        "break_even_collective_ms": break_even_ms,
        "headroom_factor": break_even_ms / float(collective_ms) if collective_ms > 0 else None,
        "required_improvement_factor": (
            float(collective_ms) / break_even_ms if break_even_ms > 0 else None
        ),
        # The plan's own stop rule: a projection that cannot beat the faster TP1
        # arm even when rank-local weight reads cost nothing is a lower bound no
        # shard kernel can rescue, so it decides the question without waiting for
        # per-kernel measurements.
        "speedup_if_rank_weights_were_free": baseline_ms / float(collective_ms)
        if collective_ms > 0
        else None,
        "limiting_rank_fixed": max(per_rank, key=lambda entry: entry["fixed_ms_per_token"])["device"],
        "limiting_rank_weight": max(
            per_rank, key=lambda entry: entry["rank_weight_ms_per_token"]
        )["device"],
        "per_rank": per_rank,
    }


#: Structures whose per-reduction cost is usable as a per-layer cost, because the
#: dependent-chain value check proves every reduction consumed its predecessor.
DEPENDENT_CHAIN_MODES = (
    "per_step",
    "per_step_alternating",
    "per_step_alternating_graph",
    "staged_exchange_host_sync",
    "staged_exchange_batched_return_wait",
    "staged_exchange_batched",
    # The same batched protocol driven from a native loop instead of Python. Its
    # marginal comes from the A/B driver, not from the chain artifact, so
    # ``read_dependent_chain_marginal`` rejects it and
    # ``read_native_ab_marginal`` reads it.
    "native_staged_exchange_batched",
)

#: Modes whose marginal is read from the native A/B artifact instead.
NATIVE_AB_MODES = ("native_staged_exchange_batched",)


def read_native_ab_marginal(path: Path) -> dict[str, Any]:
    """Read the native arm's per-step marginal out of the A/B artifact.

    The native runner runs the chain through one ``step`` implementation for both
    its timed and its verified passes, checks the whole vector on both ranks,
    rejects nonfinite values, and exits non-zero when a check fails. A depth is
    accepted here when the bounded recurrence verified exactly and the timed
    recurrence either verified exactly or is honestly reported as saturating fp32
    at that depth. The comparison against the Python arm is reported as
    provisional when the two arms did not match on payload, depths, protocol,
    devices and balanced repetitions, and that flag is carried into the source
    record rather than dropped.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") != "tp2-staged-exchange-native-ab":
        raise ValueError(f"{path}: not a native A/B artifact")
    arms = payload.get("arms") or {}
    native_arm = arms.get("native") or {}
    marginal = (native_arm.get("marginal") or {}).get("overall_us_per_step")
    if marginal is None:
        raise ValueError(f"{path}: the native arm has no overall marginal")
    depths = native_arm.get("depths") or {}
    if not depths:
        raise ValueError(f"{path}: the native arm has no depths")
    unverified = []
    for depth, entry in depths.items():
        verification = entry.get("verification") or {}
        bounded = verification.get("bounded") or {}
        summed = verification.get("sum") or {}
        if not (bounded.get("exact") and bounded.get("finite")):
            unverified.append(depth)
            continue
        timed_ok = (
            bool(summed.get("finite") and summed.get("exact"))
            if summed.get("informative")
            else bool(summed.get("saturation_expected"))
        )
        if not timed_ok:
            unverified.append(depth)
    if unverified:
        raise ValueError(
            f"{path}: depths {sorted(unverified, key=int)} did not verify their chain, "
            "so the marginal is not a per-layer cost"
        )
    comparison = payload.get("comparison") or {}
    return {
        "marginal_us_per_step": float(marginal),
        "source": str(path),
        "mode": "native_staged_exchange_batched",
        "depends_on_every_step": True,
        "payload_bytes": payload.get("payload_bytes"),
        "world_size": 2,
        "group_boundary": payload.get("protocol"),
        "python_arm_us_per_step": comparison.get("python_us_per_step"),
        "python_over_native": comparison.get("python_over_native"),
        "comparison_provisional": bool(comparison.get("provisional", True)),
        "provenance_match": payload.get("provenance_match"),
        "git_commit": (payload.get("provenance") or {}).get("git_commit"),
    }


def read_dependent_chain_marginal(
    path: Path,
    *,
    case_key: str,
    mode: str = "per_step",
) -> dict[str, Any]:
    """Read the measured per-step marginal out of a dependent chain artifact.

    The marginal must come from a mode whose arithmetic check proves each
    reduction consumed its predecessor. ``per_step`` is the original structure,
    ``per_step_alternating`` removes the artificial device copy between
    reductions, ``per_step_alternating_graph`` replays the same structure from a
    captured graph, and ``staged_exchange_host_sync`` replaces the collective
    with a page-locked host exchange (``staged_exchange_host_sync`` one rank at a
    time, ``staged_exchange_batched`` both ranks submitted before either is
    awaited). A ``per_chain`` or single-group marginal describes
    a group in which the collectives are deferred or collapsed, so it cannot
    carry a layer dependency; the recorded ``depends_on_every_step`` flag is
    checked here rather than trusted from the mode name.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = payload.get("collective", {}).get("cases", {})
    if case_key not in cases:
        raise ValueError(
            f"{path}: no case {case_key!r}; available: {sorted(cases)}"
        )
    chain = cases[case_key].get("dependent_chain")
    if not isinstance(chain, dict) or "modes" not in chain:
        raise ValueError(f"{path}: case {case_key!r} has no dependent chain report")
    if mode in NATIVE_AB_MODES:
        raise ValueError(
            f"{mode!r} is measured by the native A/B driver; pass its artifact to "
            "--native-ab-artifact instead"
        )
    if mode not in chain["modes"]:
        raise ValueError(f"{path}: case {case_key!r} has no {mode!r} mode")
    mode_report = chain["modes"][mode]
    marginal = mode_report.get("marginal", {}).get("overall_us_per_step")
    if marginal is None:
        raise ValueError(f"{path}: case {case_key!r} {mode!r} has no overall marginal")
    if not mode_report.get("depends_on_every_step"):
        raise ValueError(
            f"{path}: case {case_key!r} {mode!r} failed its dependency check, so its "
            "marginal is not a per-layer cost"
        )
    return {
        "marginal_us_per_step": float(marginal),
        "source": str(path),
        "case": case_key,
        "mode": mode,
        "depends_on_every_step": True,
        "payload_bytes": chain.get("payload_bytes"),
        "world_size": chain.get("world_size"),
        "group_boundary": mode_report.get("group_boundary"),
    }


#: Keys a kernel record may carry its name under, and the duration fields it
#: may carry its runtime under. A record counts as kernel execution only when
#: both are present and the duration is positive.
_KERNEL_NAME_KEYS = ("Kernel_Name", "kernel_name", "name", "kernel")
_KERNEL_DURATION_KEYS = ("DurationNs", "duration_ns", "duration_ns_us", "duration")
_KERNEL_TIMESTAMP_KEYS = (
    ("End_Timestamp", "Start_Timestamp"),
    ("End_Timestamp", "Begin_Timestamp"),
)


def _default_results_root() -> Path:
    """The repo's benchmark-artifact directory, resolved from this script."""

    return Path(__file__).resolve().parent.parent / "benchmarks" / "results"


def _kernel_records_from_json(node: Any) -> list[tuple[str, int]]:
    """Collect (kernel name, duration ns) pairs from arbitrary JSON."""

    records: list[tuple[str, int]] = []
    if isinstance(node, dict):
        name = next((node[k] for k in _KERNEL_NAME_KEYS if k in node), None)
        duration = next((node[k] for k in _KERNEL_DURATION_KEYS if k in node), None)
        if duration is None:
            for end_key, start_key in _KERNEL_TIMESTAMP_KEYS:
                if end_key in node and start_key in node:
                    try:
                        duration = int(node[end_key]) - int(node[start_key])
                    except (TypeError, ValueError):
                        duration = None
                    break
        if (
            isinstance(name, str)
            and name.strip()
            and duration is not None
            and not isinstance(duration, bool)
        ):
            try:
                duration_value = int(duration)
            except (TypeError, ValueError):
                duration_value = 0
            if duration_value > 0:
                records.append((name.strip(), duration_value))
        for value in node.values():
            records.extend(_kernel_records_from_json(value))
    elif isinstance(node, list):
        for item in node:
            records.extend(_kernel_records_from_json(item))
    return records


def _kernel_records_from_csv(text: str) -> list[tuple[str, int]]:
    """Collect (kernel name, duration ns) pairs from a rocprofv3 CSV trace."""

    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return []
    header = rows[0].keys()
    if not ("Kernel_Name" in header and "DurationNs" in header):
        return []
    records: list[tuple[str, int]] = []
    for row in rows:
        name = (row.get("Kernel_Name") or "").strip()
        try:
            duration = int(row.get("DurationNs") or 0)
        except ValueError:
            continue
        if name and duration > 0:
            records.append((name, duration))
    return records


def _validate_shard_kernel_evidence(
    paths: list[str] | None,
    *,
    results_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Validate each shard-kernel evidence artifact's contents, not its existence.

    The pre-Packet-3 gate used to be satisfied by any nonempty path list, so a
    dangling path or a file with no kernel record in it certified the projection.
    Evidence here means a readable artifact under ``benchmarks/results/`` that
    contains at least one kernel record with a name and a positive duration - the
    same shape ``rocprofv3 --kernel-trace`` writes and the same quantity the
    kernel gate requires. Anything else is reported as the reason it does not
    qualify, and the verdict stays uncertified.
    """

    root = (results_root or _default_results_root()).resolve()
    records: list[dict[str, Any]] = []
    for raw in paths or []:
        path_text = str(raw)
        entry: dict[str, Any] = {"path": path_text, "kernel_records": 0}
        path = Path(path_text)
        try:
            resolved = path.resolve()
            under_results = root in resolved.parents or resolved.parent == root
            if not under_results:
                entry["error"] = "not under benchmarks/results/"
            elif not path.is_file():
                entry["error"] = "not a file"
            elif path.suffix.lower() == ".json":
                parsed = json.loads(path.read_text())
                found = _kernel_records_from_json(parsed)
                entry["kernel_records"] = len(found)
                if found:
                    entry["example"] = {"kernel": found[0][0], "duration_ns": found[0][1]}
                else:
                    entry["error"] = "no kernel record with a name and a positive duration"
            elif path.suffix.lower() == ".csv":
                found = _kernel_records_from_csv(path.read_text())
                entry["kernel_records"] = len(found)
                if found:
                    entry["example"] = {"kernel": found[0][0], "duration_ns": found[0][1]}
                else:
                    entry["error"] = (
                        "no Kernel_Name/DurationNs columns or no positive-duration row"
                    )
            else:
                entry["error"] = "unsupported artifact type; expected .json or .csv"
        except (OSError, json.JSONDecodeError, ValueError) as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        records.append(entry)
    return records


def build_report(
    devices: list[dict[str, Any]],
    *,
    collective_ms: tuple[float, ...],
    fixed_shares: tuple[float, ...] = DEFAULT_FIXED_SHARES,
    reduction_points: int | None = None,
    collective_source: dict[str, Any] | None = None,
    shard_kernel_evidence: list[str] | None = None,
    results_root: Path | None = None,
) -> dict[str, Any]:
    protocols = sorted({str(device["protocol"]) for device in devices})
    evidence_records = _validate_shard_kernel_evidence(
        shard_kernel_evidence, results_root=results_root
    )
    evidence_ok = bool(evidence_records) and all("error" not in r for r in evidence_records)
    report: dict[str, Any] = {
        "kind": "tp2_break_even",
        "model": (
            "one synchronized group time: T2 = max_rank(fixed) + max_rank(rank weights) + "
            "collective, compared against the faster matched TP1 arm"
        ),
        "assumption": (
            "fixed cost (attention, GDN recurrence, launch overhead) does not shrink with "
            "tensor parallelism; only its share of the TP1 token time is assumed and the "
            "projection is reported across a range of shares"
        ),
        "collective_ms_per_token": list(collective_ms),
        "collective_source": collective_source,
        "baselines": [
            {
                "device": device["name"],
                "tp1_tok_s": device["tok_s"],
                "tp1_weights_gib": device["tp1_gib"],
                "tp2_rank_weights_gib": device["rank_gib"],
                "protocol": device["protocol"],
            }
            for device in devices
        ],
        "protocol_match": {
            "matched": len(protocols) == 1,
            "protocols": protocols,
        },
        "pre_packet3_gate": {
            "satisfied": evidence_ok,
            "evidence": evidence_records,
            "requirement": (
                "the packet plan requires local shard-shaped kernel measurements before "
                "Packet 3; pass --shard-kernel-evidence with rocprofv3 kernel-trace "
                "artifacts (.json or .csv) under benchmarks/results/ - each must contain "
                "at least one kernel record with a name and a positive duration"
            ),
        },
        "rows": [],
        "errors": [],
    }
    if reduction_points is not None:
        report["reduction_points_per_token"] = int(reduction_points)
    if collective_source is not None and reduction_points is not None:
        marginal = float(collective_source["marginal_us_per_step"])
        expected = int(reduction_points) * marginal / 1000.0
        report["collective_ms_expected_from_count"] = round(expected, 3)
        for value in collective_ms:
            if abs(value - expected) > max(0.02 * expected, 0.02):
                report["errors"].append(
                    f"collective budget {value} ms does not match {int(reduction_points)} "
                    f"reduction points x {marginal} us = {expected:.3f} ms"
                )
    for share in fixed_shares:
        for collective in collective_ms:
            report["rows"].append(
                project_group(devices, collective_ms=collective, fixed_share=share)
            )
    speedups = [row["projected_speedup"] for row in report["rows"]]
    headroom = [
        row["headroom_factor"] for row in report["rows"] if row["headroom_factor"] is not None
    ]
    withheld: list[str] = []
    if not report["protocol_match"]["matched"]:
        withheld.append(
            f"baseline arms were measured under different protocols: {protocols}"
        )
    if not report["pre_packet3_gate"]["satisfied"]:
        # Name the actual defect so a fabricated path cannot be confused with a
        # missing one and a wrong-content artifact cannot pass as evidence.
        for record in evidence_records:
            if "error" in record:
                withheld.append(
                    f"shard-kernel evidence {record['path']}: {record['error']}"
                )
        if not evidence_records:
            withheld.append("the pre-Packet-3 shard-kernel gate has no recorded evidence")
    # Two thresholds, deliberately not conflated:
    #   beats_faster_tp1_arm   the gate. Every row must beat the faster TP1 arm.
    #   meets_planning_aspiration   the 1.3x target. Missing it is recorded, not
    #                               withheld: the design accepts smaller wins.
    beats = bool(speedups) and min(speedups) > BEATS_TP1_SPEEDUP
    if not beats:
        withheld.append(
            f"a row projects at or below {BEATS_TP1_SPEEDUP}x, so the group would not "
            "beat the faster TP1 arm at that fixed share"
        )
    meets_aspiration = bool(speedups) and min(speedups) >= TARGET_SPEEDUP
    optimistic = [
        row["speedup_if_rank_weights_were_free"]
        for row in report["rows"]
        if row["speedup_if_rank_weights_were_free"] is not None
    ]
    optimistic_bound = max(optimistic) if optimistic else None
    # A structurally unreachable *gate* is a blocker; an unreachable aspiration
    # only bounds how far the win can go.
    optimistic_beats = (
        None if optimistic_bound is None else optimistic_bound > BEATS_TP1_SPEEDUP
    )
    optimistic_meets_aspiration = (
        None if optimistic_bound is None else optimistic_bound >= TARGET_SPEEDUP
    )
    if optimistic_beats is False:
        withheld.append(
            f"the optimistic bound ({optimistic_bound:.3f}x with free rank-weight reads) is "
            f"at or below {BEATS_TP1_SPEEDUP}x, so no shard kernel can make the group "
            "faster than the faster TP1 arm"
        )
    aspiration_notes: list[str] = []
    if not meets_aspiration:
        aspiration_notes.append(
            f"the worst row projects {min(speedups):.3f}x, below the {TARGET_SPEEDUP}x "
            "planning aspiration; a qualified net improvement is still retained"
        )
    if optimistic_meets_aspiration is False and optimistic_beats:
        aspiration_notes.append(
            f"the optimistic bound ({optimistic_bound:.3f}x with free rank-weight reads) is "
            f"below the {TARGET_SPEEDUP}x aspiration, so the win is bounded below it"
        )
    report["verdict"] = {
        "target_speedup": TARGET_SPEEDUP,
        "aspiration_target_speedup": TARGET_SPEEDUP,
        "beats_faster_tp1_arm": beats,
        "meets_planning_aspiration": meets_aspiration,
        "aspiration_notes": aspiration_notes,
        "passes_target_in_every_row": meets_aspiration,
        "certified": beats and not withheld,
        "withheld_reasons": withheld,
        "optimistic_bound_speedup": optimistic_bound,
        "optimistic_bound_beats_tp1": optimistic_beats,
        "optimistic_bound_clears_target": optimistic_meets_aspiration,
        "minimum_headroom_factor": min(headroom) if headroom else None,
        "maximum_required_improvement_factor": max(
            (
                row["required_improvement_factor"]
                for row in report["rows"]
                if row["required_improvement_factor"] is not None
            ),
            default=None,
        ),
        "worst_case_speedup": min(speedups) if speedups else None,
        "best_case_speedup": max(speedups) if speedups else None,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tp1",
        action="append",
        required=True,
        metavar="NAME=TOK_S:TP1_GIB:RANK_GIB:PROTOCOL",
        help="same-host TP1 measurement, its TP2 rank shard, and the workload shape; repeatable",
    )
    parser.add_argument(
        "--collective-ms",
        default=None,
        help=(
            "measured per-token collective cost, comma list; defaults to "
            "reduction-points x marginal when a dependent-chain artifact is given"
        ),
    )
    parser.add_argument(
        "--fixed-share",
        default=",".join(str(share) for share in DEFAULT_FIXED_SHARES),
        help="assumed fixed-cost share of the TP1 token time, comma list",
    )
    parser.add_argument(
        "--reduction-points",
        type=int,
        default=None,
        help="per-token cross-rank reductions, from the shard manifest (row-split tensor count)",
    )
    parser.add_argument(
        "--dependent-chain-artifact",
        type=Path,
        default=None,
        help="artifact from scripts/tp_collective_bench.py --dependent-depths to read the marginal from",
    )
    parser.add_argument(
        "--dependent-chain-case",
        default="all_reduce:rows1:fp32",
        help="case key inside the dependent-chain artifact (default: the decode shape)",
    )
    parser.add_argument(
        "--dependent-chain-mode",
        default="per_step",
        choices=DEPENDENT_CHAIN_MODES,
        help=(
            "mode to read the marginal from; the choices are the structures whose "
            "arithmetic check proves each reduction consumed its predecessor "
            "(per_chain and the single-group modes fail that check on this host)"
        ),
    )
    parser.add_argument(
        "--native-ab-artifact",
        type=Path,
        default=None,
        help=(
            "artifact from scripts/tp_staged_exchange_native_ab.py; selects the native "
            "arm of the same batched protocol as the per-reduction cost"
        ),
    )
    parser.add_argument(
        "--marginal-us",
        default=None,
        help="measured per-collective cost in microseconds when no artifact is given",
    )
    parser.add_argument(
        "--shard-kernel-evidence",
        action="append",
        default=None,
        help="path or identifier of a local shard-shaped kernel measurement; repeatable",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the JSON artifact here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    devices = [parse_tp1(spec) for spec in args.tp1]
    fixed_shares = tuple(float(chunk) for chunk in args.fixed_share.split(",") if chunk.strip())

    collective_source: dict[str, Any] | None = None
    if args.native_ab_artifact is not None:
        collective_source = read_native_ab_marginal(args.native_ab_artifact)
    elif args.dependent_chain_artifact is not None:
        collective_source = read_dependent_chain_marginal(
            args.dependent_chain_artifact,
            case_key=str(args.dependent_chain_case),
            mode=str(args.dependent_chain_mode),
        )
    elif args.marginal_us is None:
        raise SystemExit(
            "provide --dependent-chain-artifact or --marginal-us: the collective input must "
            "come from a measurement, and its source is recorded"
        )
    else:
        collective_source = {
            "marginal_us_per_step": float(args.marginal_us),
            "source": "command line",
            "mode": "unspecified",
            "depends_on_every_step": None,
        }

    if args.collective_ms is not None:
        collective_ms = tuple(
            float(chunk) for chunk in str(args.collective_ms).split(",") if chunk.strip()
        )
    elif args.reduction_points is not None:
        value = (
            int(args.reduction_points) * float(collective_source["marginal_us_per_step"]) / 1000.0
        )
        collective_ms = (round(value, 3),)
    else:
        raise SystemExit("provide --collective-ms or --reduction-points")

    report = build_report(
        devices,
        collective_ms=collective_ms,
        fixed_shares=fixed_shares,
        reduction_points=args.reduction_points,
        collective_source=collective_source,
        shard_kernel_evidence=args.shard_kernel_evidence,
    )
    for error in report["errors"]:
        print(f"error: {error}", file=sys.stderr)

    if not args.quiet:
        print(
            f"{'fixed':>6s} {'coll_ms':>8s} {'base_ms':>8s} {'tp2_ms':>8s} {'speedup':>8s} "
            f"{'break_even':>10s} {'headroom':>9s}"
        )
        for row in report["rows"]:
            print(
                f"{row['fixed_share'] * 100:5.0f}% {row['collective_ms_per_token']:8.2f} "
                f"{row['baseline_tp1_ms_per_token']:8.2f} {row['tp2_group_ms_per_token']:8.2f} "
                f"{row['projected_speedup']:8.3f} {row['break_even_collective_ms']:10.2f} "
                f"{(row['headroom_factor'] or float('nan')):8.2f}x"
            )
        verdict = report["verdict"]
        print(
            f"verdict: beats the faster TP1 arm in every row = "
            f"{verdict['beats_faster_tp1_arm']}; meets the "
            f"{verdict['aspiration_target_speedup']}x aspiration = "
            f"{verdict['meets_planning_aspiration']}; certified = {verdict['certified']}"
        )
        for reason in verdict["withheld_reasons"]:
            print(f"  withheld: {reason}", file=sys.stderr)
        for note in verdict["aspiration_notes"]:
            print(f"  aspiration: {note}", file=sys.stderr)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
