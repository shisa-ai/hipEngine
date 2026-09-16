#!/usr/bin/env python3
"""Put hipEngine and a comparator on one shared-family prefill cost table.

The two engines do not name things the same way. hipEngine carries the tensor
role in a ROCTX range or the launch-census role stack, so a symbol can serve
several roles; the llama.cpp family names the family in the symbol, and in the
MMB dialect it names it directly. Comparing role tables from the two therefore
compares two taxonomies, which is not a comparison.

``qwen4exp_comparator_role_map`` already exists to express both in one
vocabulary, and it already classified the comparators. It gained a
``hipengine`` mapper for this script, so all three sides are classified from
kernel symbols by one module with one family list.

Inputs:

* ``--hipengine-role-analysis``: a role-marked ``rocprofv3`` analysis
  (``qwen4exp_role_analyze.py``). Its ``exact_role_kernels`` already carries
  per-(role, kernel) milliseconds, and the role is not used here -- the symbol
  is. That keeps the hipEngine column independent of the role taxonomy.
* ``--comparator``: ``label=path`` for a ``qwen4exp_delimited_prefill_attribution``
  artifact. Repeatable. Its ``per_family_median_ms`` is the comparator side.

Both sides must be the same case and the same host, and they are not the same
arithmetic. The table answers "what does each operation cost each engine on this
case"; it is not an efficiency verdict, and no row here is a like-for-like
kernel comparison.

Example:
    python3 scripts/qwen4exp_shared_family_comparison.py \\
        --hipengine-role-analysis <dir>/role-analysis.json \\
        --comparator halobox-base=<dir>/attrib-base-69946438a-code4096.json \\
        --comparator halobox-pr63=<dir>/attrib-pr63-c4aa30229-code4096.json \\
        --output <dir>/shared-family-comparison.json \\
        --markdown <dir>/shared-family-comparison.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen4exp_comparator_role_map as role_map  # noqa: E402

# Families that hipEngine's mapper folds into another family. Reported as a
# separate total so the fold cannot hide the machinery.
REPAIR_TOKENS = ("sparse_exact_repair", "selected_sparse_repair")


def load_hipengine(path: Path, unmapped_floor_ms: float) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    rows = payload.get("exact_role_kernels")
    if not rows:
        raise SystemExit(f"{path}: no exact_role_kernels")

    by_family: Counter[str] = Counter()
    by_family_kernels: Counter[str] = Counter()
    risk_or_repair_ms = 0.0
    unmapped: Counter[str] = Counter()
    total = 0.0

    for row in rows:
        name = row["kernel"]
        ms = float(row["ms"])
        total += ms
        family = role_map.map_hipengine(name, ("1", "1", "1"))
        by_family[family] += ms
        by_family_kernels[family] += 1
        if family == "other":
            unmapped[name] += ms
        if any(token in name for token in REPAIR_TOKENS):
            risk_or_repair_ms += ms

    over_floor = [
        {"kernel": name, "ms": round(ms, 3)}
        for name, ms in unmapped.most_common()
        if ms >= unmapped_floor_ms
    ]
    return {
        "label": payload.get("label"),
        "window_ms": payload.get("window_ms"),
        "attributed_ms": payload.get("attributed_ms"),
        "attributed_time_pct": payload.get("attributed_time_pct"),
        "total_ms": total,
        "kernel_count": len(rows),
        "by_family_ms": {k: round(v, 1) for k, v in by_family.items()},
        "by_family_kernel_count": dict(by_family_kernels),
        "risk_or_repair_ms": round(risk_or_repair_ms, 1),
        "unmapped_over_floor": over_floor,
    }


def load_comparator(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    per_family = payload.get("per_family_median_ms")
    if not per_family:
        raise SystemExit(f"{path}: no per_family_median_ms")
    by_family = {k: round(float(v["median_ms"]), 1) for k, v in per_family.items()}
    return {
        "label": payload.get("label"),
        "dialect": payload.get("dialect"),
        "total_ms": round(sum(by_family.values()), 1),
        "by_family_ms": by_family,
    }


def build_table(
    hipengine: dict[str, Any],
    comparators: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    families = list(role_map.FAMILIES)
    for comp in comparators:
        for family in comp["by_family_ms"]:
            if family not in families:
                families.append(family)

    ours_total = hipengine["total_ms"]
    rows: list[dict[str, Any]] = []
    for family in families:
        ours = hipengine["by_family_ms"].get(family, 0.0)
        row: dict[str, Any] = {
            "family": family,
            "hipengine_ms": ours,
            "hipengine_share_pct": round(100.0 * ours / ours_total, 1) if ours_total else None,
        }
        for comp in comparators:
            theirs = comp["by_family_ms"].get(family, 0.0)
            row[f"{comp['label']}_ms"] = theirs
            row[f"{comp['label']}_ratio"] = (
                round(ours / theirs, 2) if theirs else None
            )
        rows.append(row)
    rows.sort(key=lambda r: -r["hipengine_ms"])
    return rows


def render_markdown(
    payload: dict[str, Any],
    comparators: list[dict[str, Any]],
) -> str:
    hipengine = payload["hipengine"]
    rows = payload["table"]
    labels = [c["label"] for c in comparators]

    header = ["Family", "ours ms", "ours %"]
    for label in labels:
        header += [f"{label} ms", "ratio"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| --- " + " | ---: " * (len(header) - 1) + "|",
    ]
    for row in rows:
        cells = [
            f"`{row['family']}`",
            f"{row['hipengine_ms']:.1f}",
            f"{row['hipengine_share_pct']:.1f}",
        ]
        for label in labels:
            theirs = row.get(f"{label}_ms", 0.0)
            ratio = row.get(f"{label}_ratio")
            cells += [
                f"{theirs:.1f}" if theirs else "0.0",
                f"{ratio:.2f}x" if ratio else "—",
            ]
        lines.append("| " + " | ".join(cells) + " |")

    totals = ["**total**", f"**{hipengine['total_ms']:.1f}**", "**100.0**"]
    for comp in comparators:
        ratio = hipengine["total_ms"] / comp["total_ms"] if comp["total_ms"] else None
        totals += [f"**{comp['total_ms']:.1f}**", f"**{ratio:.2f}x**" if ratio else "—"]
    lines.append("| " + " | ".join(totals) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hipengine-role-analysis", type=Path, required=True)
    parser.add_argument("--hipengine-label", default="hipengine")
    parser.add_argument(
        "--comparator", action="append", default=[], metavar="LABEL=PATH",
        help="Repeatable. Comparator attribution artifact.",
    )
    parser.add_argument("--case-id", default="code-p4096")
    parser.add_argument("--unmapped-floor-ms", type=float, default=1.0)
    parser.add_argument(
        "--strict", action="store_true",
        help="Fail when any hipEngine kernel above the floor is unmapped",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    hipengine = load_hipengine(args.hipengine_role_analysis, args.unmapped_floor_ms)
    hipengine["label"] = args.hipengine_label

    comparators = []
    for spec in args.comparator:
        if "=" not in spec:
            raise SystemExit(f"--comparator wants LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        comp = load_comparator(Path(path))
        comp["label"] = label
        comparators.append(comp)

    if args.strict and hipengine["unmapped_over_floor"]:
        raise SystemExit(
            "unmapped hipEngine kernels above the floor: "
            + json.dumps(hipengine["unmapped_over_floor"])
        )

    payload = {
        "schema": 1,
        "kind": "qwen4exp_shared_family_comparison",
        "performance_claim": False,
        "numerics_evaluated": False,
        "case_id": args.case_id,
        "question": (
            "On one case on one host, what does each operation cost each engine, "
            "in one taxonomy?"
        ),
        "taxonomy": {
            "source": "scripts/qwen4exp_comparator_role_map.py",
            "families": list(role_map.FAMILIES),
            "hipengine_mapper": "map_hipengine",
        },
        "caveats": [
            "Both sides are the same case on the same host, but not the same arithmetic.",
            "hipEngine's iu8 exact-repair passes are folded into the matmul family they correct; the folded total is reported as risk_or_repair_ms.",
            "Milliseconds are kernel time, not wall. Neither side is a throughput claim.",
        ],
        "hipengine": hipengine,
        "comparators": comparators,
        "table": build_table(hipengine, comparators),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {args.output}")

    markdown = render_markdown(payload, comparators)
    if args.markdown:
        args.markdown.write_text(markdown + "\n")
        print(f"wrote {args.markdown}")
    print()
    print(markdown)

    if hipengine["unmapped_over_floor"]:
        print()
        print("unmapped above floor:")
        for entry in hipengine["unmapped_over_floor"]:
            print(f"  {entry['ms']:8.2f}  {entry['kernel'][:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
