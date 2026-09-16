#!/usr/bin/env python3
"""Side-by-side per-component prefill comparison between two delimited traces.

Takes two outputs of ``qwen4exp_delimited_prefill_attribution.py`` and reports
what each kernel family costs in each engine, the difference, and each family's
share of the total. Every number is read from those artifacts, so the table
cannot drift from the traces it came from.

The comparison is only meaningful when both artifacts matched every measured
request, which is asserted rather than assumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not payload.get("all_measured_requests_matched"):
        raise SystemExit(
            f"{path}: not every measured request matched a burst "
            f"({payload.get('requests_matched')}/{payload.get('requests_examined')}); "
            "per-component milliseconds from this trace are not delimited"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = _load(args.base)
    candidate = _load(args.candidate)

    base_families = base["per_family_median_ms"]
    candidate_families = candidate["per_family_median_ms"]
    families = sorted(
        set(base_families) | set(candidate_families),
        key=lambda f: -base_families.get(f, {}).get("median_ms", 0.0),
    )
    base_total = sum(base_families.get(f, {}).get("median_ms", 0.0) for f in families)
    candidate_total = sum(
        candidate_families.get(f, {}).get("median_ms", 0.0) for f in families
    )

    rows = []
    for family in families:
        before = base_families.get(family, {}).get("median_ms", 0.0)
        after = candidate_families.get(family, {}).get("median_ms", 0.0)
        rows.append({
            "family": family,
            "base_ms": before,
            "candidate_ms": after,
            "delta_ms": round(after - before, 2),
            "delta_pct": round(100.0 * (after - before) / before, 2) if before else None,
            "base_share_pct": round(100.0 * before / base_total, 2) if base_total else None,
            "candidate_share_pct": (
                round(100.0 * after / candidate_total, 2) if candidate_total else None
            ),
            "base_kernels": None,
        })

    def _measured_ms(artifact: dict) -> float:
        values = [r["prompt_ms"] for r in artifact["records"] if r["matched"]]
        return sorted(values)[len(values) // 2]

    base_prompt = _measured_ms(base)
    candidate_prompt = _measured_ms(candidate)

    artifact = {
        "schema": 1,
        "kind": "qwen4exp_prefill_component_gap",
        "performance_claim": False,
        "numerics_evaluated": False,
        "question": (
            "Held to one case on one host, which kernel families account for the "
            "prefill-time difference between these two engines?"
        ),
        "base": {
            "label": base["label"],
            "trace": base["trace"],
            "report": base["report"],
            "measured_prompt_ms": base_prompt,
            "kernel_sum_ms": base_total,
            "kernel_sum_over_prompt_ms": base["matched_kernel_sum_over_prompt_ms"],
        },
        "candidate": {
            "label": candidate["label"],
            "trace": candidate["trace"],
            "report": candidate["report"],
            "measured_prompt_ms": candidate_prompt,
            "kernel_sum_ms": candidate_total,
            "kernel_sum_over_prompt_ms": candidate["matched_kernel_sum_over_prompt_ms"],
        },
        "measured_prompt_delta_ms": round(candidate_prompt - base_prompt, 2),
        "measured_prompt_ratio": round(candidate_prompt / base_prompt, 4),
        "kernel_sum_delta_ms": round(candidate_total - base_total, 2),
        "kernel_sum_ratio": round(candidate_total / base_total, 4),
        "components": rows,
        "largest_deltas_ms": sorted(rows, key=lambda r: r["delta_ms"])[:6],
    }
    args.output.write_text(json.dumps(artifact, indent=1) + "\n")

    print(f"{'family':<22}{'base ms':>10}{'cand ms':>10}{'delta ms':>10}{'delta %':>9}")
    for row in rows:
        delta_pct = f"{row['delta_pct']:>8.1f}%" if row["delta_pct"] is not None else "     n/a"
        print(
            f"{row['family']:<22}{row['base_ms']:>10.1f}{row['candidate_ms']:>10.1f}"
            f"{row['delta_ms']:>10.1f}{delta_pct}"
        )
    print(
        f"{'KERNEL TOTAL':<22}{base_total:>10.1f}{candidate_total:>10.1f}"
        f"{candidate_total - base_total:>10.1f}"
    )
    print(
        f"{'measured prompt_ms':<22}{base_prompt:>10.1f}{candidate_prompt:>10.1f}"
        f"{candidate_prompt - base_prompt:>10.1f}"
        f"{100.0 * (candidate_prompt - base_prompt) / base_prompt:>8.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
