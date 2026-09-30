"""Group a prefill kernel trace by GDN-family membership.

GD-4's premise is that the GDN *family* is worth optimizing - its reference
fuses the chunk scan with conv/SiLU, q/k norm and gated RMS norm, so it targets
the stages that sit outside the recurrence.  GD-3's screen bounded the
recurrence alone at ~1.1% of prefill; this answers the different question of how
much prefill the family as a whole occupies, which is the gate GD-4 actually
has to pass.

Classification is by kernel-name substring, which is deliberately crude and
therefore auditable: every bucket is printed with its member kernels so a
mis-filing is visible rather than hidden inside a total.

    python3 scripts/qwen4exp_family_share.py --trace-dir <dir> [--json out.json]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

# Ordered: the first pattern that matches a kernel name wins, so put the
# specific families before the generic ones.
FAMILIES: tuple[tuple[str, str], ...] = (
    ("gdn", r"gdn"),
    ("conv", r"conv"),
    ("qsa", r"qsa|sparse|block_pool|selection"),
    ("ple", r"ple"),
    ("gr", r"gr_|gated_mean|four_branch|coltile"),
    ("moe", r"expert|selected_|router|topk|top_k|shexp"),
    ("norm", r"rmsnorm|rms_norm|norm_"),
    ("linear", r"linear|gemm|gemv|mmq|wmma|dp4a|matmul"),
    ("attention", r"attn|attention|rope|rotate|paged"),
    ("kv", r"kv_|cache_|paged_kv"),
    ("mtp", r"mtp|nextn|draft"),
    ("embed", r"embed|lm_head|logits|argmax|sample"),
    ("cast", r"cast|quant|dequant|repack|pack"),
)

_DURATION_COLUMNS = (
    "durationns",
    "duration_ns",
    "end_timestamp",
    "start_timestamp",
)


def _columns(header: list[str]) -> dict[str, str]:
    return {name.lower(): name for name in header}


def _pick(keys: dict[str, str], *candidates: str) -> str | None:
    for candidate in candidates:
        if candidate in keys:
            return keys[candidate]
    return None


def load_kernels(csv_path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        keys = _columns(reader.fieldnames or [])
        name_col = _pick(keys, "kernel_name", "kernel name", "kernname", "kernel")
        if name_col is None:
            raise ValueError(f"no kernel-name column in {csv_path}: {reader.fieldnames}")
        dur_col = _pick(keys, "durationns", "duration_ns")
        start_col = _pick(keys, "start_timestamp", "start")
        end_col = _pick(keys, "end_timestamp", "end")
        for row in reader:
            name = row.get(name_col) or ""
            if not name:
                continue
            if dur_col is not None and row.get(dur_col):
                duration = float(row[dur_col])
            elif start_col and end_col and row.get(start_col) and row.get(end_col):
                duration = float(row[end_col]) - float(row[start_col])
            else:
                continue
            rows.append((name, duration))
    return rows


def classify(name: str) -> str:
    lowered = name.lower()
    for label, pattern in FAMILIES:
        if re.search(pattern, lowered):
            return label
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    candidates = sorted(args.trace_dir.rglob("*_kernel_trace.csv"))
    if not candidates:
        print(f"no *_kernel_trace.csv under {args.trace_dir}", file=sys.stderr)
        return 2
    csv_path = candidates[0]
    rows = load_kernels(csv_path)
    if not rows:
        print(f"no kernel rows parsed from {csv_path}", file=sys.stderr)
        return 2

    total = sum(d for _, d in rows)
    by_family: dict[str, list[float]] = collections.defaultdict(list)
    by_kernel: dict[str, list[float]] = collections.defaultdict(list)
    for name, duration in rows:
        by_family[classify(name)].append(duration)
        by_kernel[name].append(duration)

    print(f"trace: {csv_path}")
    print(f"kernel rows: {len(rows)}   total kernel time: {total / 1e6:.3f} ms")
    print()
    print(f"{'family':<12}{'calls':>8}{'total ms':>12}{'share':>9}  example kernel")
    print("-" * 92)
    for label, durations in sorted(by_family.items(), key=lambda kv: -sum(kv[1])):
        subtotal = sum(durations)
        example = max(
            (n for n in by_kernel if classify(n) == label),
            key=lambda n: sum(by_kernel[n]),
        )
        print(
            f"{label:<12}{len(durations):>8}{subtotal / 1e6:>12.3f}"
            f"{100.0 * subtotal / total:>8.2f}%  {example[:52]}"
        )

    print()
    print(f"top {args.top} individual kernels:")
    print(f"{'calls':>7}{'total ms':>12}{'share':>9}  kernel")
    print("-" * 92)
    for name, durations in sorted(by_kernel.items(), key=lambda kv: -sum(kv[1]))[: args.top]:
        subtotal = sum(durations)
        print(
            f"{len(durations):>7}{subtotal / 1e6:>12.3f}"
            f"{100.0 * subtotal / total:>8.2f}%  {name[:62]}"
        )

    if args.json:
        payload = {
            "schema": 1,
            "kind": "qwen4exp-prefill-family-share",
            "trace": str(csv_path),
            "kernel_rows": len(rows),
            "total_kernel_ns": total,
            "families": {
                label: {
                    "calls": len(durations),
                    "total_ns": sum(durations),
                    "share": sum(durations) / total,
                }
                for label, durations in sorted(by_family.items(), key=lambda kv: -sum(kv[1]))
            },
            "kernels": {
                name: {"calls": len(d), "total_ns": sum(d), "share": sum(d) / total}
                for name, d in sorted(by_kernel.items(), key=lambda kv: -sum(kv[1]))
            },
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
