#!/usr/bin/env python3
"""Per-component prefill attribution from a delimited rocprofv3 trace.

The comparator harness can be asked to leave an idle gap between requests
(``--inter-request-gap-ms``). A gap wider than the profiler's burst threshold
splits the trace so each request is its own burst, which is what makes the
warmup prefill separable from the measured ones. Without that gap the two run
back to back and a burst covers both, which is why an earlier version of this
analysis reported kernel-sum over measured-prompt near 1.7: the window held two
prefills.

``qwen4exp_comparator_role_map.read_trace`` already splits bursts and can scope
to one of them. What it cannot do is decide *which* burst belongs to *which*
request, and that decision is the one that silently went wrong before. This
script makes it explicitly: it matches each measured request to the burst whose
kernel sum is a plausible fraction of that request's reported ``prompt_ms``, and
refuses to attribute a request it cannot match. A match that only works because
two prefills share a burst fails the ratio check rather than being reported.

Buckets come from the role map, so both engines classify a kernel the same way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen4exp_comparator_role_map as role_map  # noqa: E402

# A burst is accepted for a request when its kernel sum lands inside this band
# of the request's reported prompt_ms. The remainder is host-side launch and
# synchronisation time, which is a real and bounded part of the window. The
# upper bound is the important one: a burst holding two prefills reads near 2.0
# and is rejected.
MATCH_LOW = 0.70
MATCH_HIGH = 1.05

# Midpoint of the accepted band, used to pick the closest candidate when more
# than one burst is plausible.
MATCH_IDEAL = 0.93


def _requests(report: dict) -> list[dict]:
    """Every request the harness made, in the order it made them."""
    out = []
    for case in report["cases"]:
        for _ in range(int(report.get("warmups") or 0)):
            out.append({"case_id": case["id"], "rep": None, "prompt_ms": None})
        for rep in case["repetitions"]:
            out.append({
                "case_id": case["id"],
                "rep": rep["rep"],
                "prompt_ms": float(rep["prompt_ms"]),
            })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True,
                        help="rocprofv3 *_kernel_trace.csv for one server run")
    parser.add_argument("--report", type=Path, required=True,
                        help="the harness JSON whose requests produced the trace")
    parser.add_argument("--dialect", choices=sorted(role_map.MAPPERS), default="llamacpp")
    parser.add_argument("--gap-ms", type=float, default=250.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    mapper = role_map.MAPPERS[args.dialect]

    # One pass over the whole trace gives every burst's kernel sum, which is
    # what the match is made on.
    whole = role_map.read_trace(args.trace, mapper, burst_gap_ms=args.gap_ms)
    bursts = whole["bursts_in_trace"]
    substantial = [b for b in bursts if b["kernel_ms"] > 100.0]

    requests = _requests(report)
    measured = [r for r in requests if r["prompt_ms"] is not None]

    records = []
    used: set[int] = set()
    for request in measured:
        best = None
        for burst in substantial:
            if burst["index"] in used:
                continue
            ratio = burst["kernel_ms"] / request["prompt_ms"]
            if MATCH_LOW <= ratio <= MATCH_HIGH:
                score = abs(ratio - MATCH_IDEAL)
                if best is None or score < best[0]:
                    best = (score, burst, ratio)
        if best is None:
            records.append({
                **request,
                "matched": False,
                "reason": (
                    "no burst whose kernel sum is a plausible fraction of "
                    "prompt_ms; the trace is not delimited into per-request bursts"
                ),
            })
            continue
        _, burst, ratio = best
        used.add(burst["index"])
        scoped = role_map.read_trace(
            args.trace, mapper, burst_gap_ms=args.gap_ms,
            delimit=f"burst:{burst['index']}",
        )
        records.append({
            **request,
            "matched": True,
            "burst_index": burst["index"],
            "burst_dispatch_count": burst["dispatches"],
            "kernel_sum_ms": burst["kernel_ms"],
            "kernel_sum_over_prompt_ms": round(ratio, 4),
            "unaccounted_ms": round(request["prompt_ms"] - burst["kernel_ms"], 3),
            "by_family_ms": scoped["by_family_ms"],
            "by_family_share_pct": scoped["by_family_share_pct"],
            "kernels": scoped["kernels"],
            "unmapped_top": scoped["unmapped_top"],
        })

    matched = [r for r in records if r["matched"]]
    summary: dict[str, dict[str, float]] = {}
    if matched:
        families = sorted({f for r in matched for f in r["by_family_ms"]})
        for family in families:
            values = [r["by_family_ms"].get(family, 0.0) for r in matched]
            summary[family] = {
                "median_ms": round(sorted(values)[len(values) // 2], 2),
                "min_ms": round(min(values), 2),
                "max_ms": round(max(values), 2),
                "median_share_pct": round(
                    sorted(r["by_family_share_pct"].get(family, 0.0) for r in matched)[
                        len(matched) // 2
                    ],
                    2,
                ),
            }

    artifact = {
        "schema": 1,
        "kind": "qwen4exp_delimited_prefill_attribution",
        "performance_claim": False,
        "numerics_evaluated": False,
        "why": (
            "Absolute per-component milliseconds require the profiled window to "
            "hold exactly one prefill. The harness is asked for an idle gap "
            "between requests so each becomes its own burst; this script matches "
            "bursts to requests by duration and refuses a burst that does not "
            "look like a single prefill."
        ),
        "trace": str(args.trace),
        "report": str(args.report),
        "label": report.get("label"),
        "measurement_class": report.get("measurement_class"),
        "dialect": args.dialect,
        "gap_ms": args.gap_ms,
        "bursts_in_trace": bursts,
        "requests_examined": len(measured),
        "requests_matched": len(matched),
        "all_measured_requests_matched": len(matched) == len(measured),
        "matched_kernel_sum_over_prompt_ms": [
            r["kernel_sum_over_prompt_ms"] for r in matched
        ],
        "per_family_median_ms": summary,
        "records": records,
    }
    args.output.write_text(json.dumps(artifact, indent=1) + "\n")
    print(json.dumps({
        "label": artifact["label"],
        "bursts": len(bursts),
        "matched": f"{len(matched)}/{len(measured)}",
        "kernel_sum_over_prompt_ms": artifact["matched_kernel_sum_over_prompt_ms"],
        "per_family_median_ms": summary,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
