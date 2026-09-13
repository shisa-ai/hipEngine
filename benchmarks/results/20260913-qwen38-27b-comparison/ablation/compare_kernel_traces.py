#!/usr/bin/env python3
"""Compare two rocprofv3 kernel traces by kernel family.

Usage: compare_kernel_traces.py <upstream_csv> <candidate_csv> [--top N]
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict

FAMILY_RULES = (
    ("MMQ (quantized mat-mul, prefill)", re.compile(r"mul_mat_q|mmq")),
    ("MMVQ/MMV (quantized mat-vec, decode)", re.compile(r"mul_mat_vec|mmvq|dequantize_mul_mat_vec")),
    ("Flash attention", re.compile(r"fattn|flash_attn")),
    ("Gated DeltaNet / linear attn", re.compile(r"gated_delta|delta_net|gdn|ssm|conv")),
    ("Attention (non-FA)", re.compile(r"attn|soft_max|rope")),
    ("Norm / RMSNorm", re.compile(r"norm")),
    ("Quantize / dequant", re.compile(r"quantize|dequant")),
    ("Elementwise / unary / activation", re.compile(r"unary|silu|gelu|clamp|add|mul|scale|cont")),
    ("Copy / memset", re.compile(r"copy|memset|__amd_rocclr")),
    ("Top-k / argmax / sort", re.compile(r"top_k|argmax|sort|topk")),
    ("MoE routing / mul_mat_id", re.compile(r"mul_mat_id|mmid|expert|route")),
)


def load(path: str) -> tuple[dict[str, list[int]], dict[str, int]]:
    per_kernel: dict[str, list[int]] = defaultdict(list)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Kind"] != "KERNEL_DISPATCH":
                continue
            start, end = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
            per_kernel[row["Kernel_Name"]].append(end - start)
    return per_kernel, {k: sum(v) for k, v in per_kernel.items()}


def family(name: str) -> str:
    for label, pattern in FAMILY_RULES:
        if pattern.search(name):
            return label
    return "other"


def family_totals(totals: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for name, ns in totals.items():
        out[family(name)] += ns
    return dict(out)


def main() -> int:
    top = 20
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])
    base_kernels, base_totals = load(args[0])
    cand_kernels, cand_totals = load(args[1])
    base_sum, cand_sum = sum(base_totals.values()), sum(cand_totals.values())

    print(f"total kernel time: upstream {base_sum/1e9:.3f} s | candidate {cand_sum/1e9:.3f} s "
          f"| candidate/upstream {cand_sum/base_sum:.4f}")

    print("\n== By family (candidate vs upstream) ==")
    print(f"{'family':38s} {'upstream s':>10s} {'cand s':>10s} {'delta s':>9s} {'ratio':>7s}")
    families = sorted(set(family_totals(base_totals)) | set(family_totals(cand_totals)),
                      key=lambda f: -family_totals(base_totals).get(f, 0))
    for fam in families:
        b = family_totals(base_totals).get(fam, 0) / 1e9
        c = family_totals(cand_totals).get(fam, 0) / 1e9
        if b + c < 0.05:
            continue
        print(f"{fam:38s} {b:10.3f} {c:10.3f} {c-b:+9.3f} {c/b if b else float('nan'):7.3f}")

    print(f"\n== Top {top} kernels by upstream time ==")
    print(f"{'kernel':58s} {'calls':>6s} {'up s':>8s} {'cand s':>8s} {'ratio':>6s}")
    for name, _ in sorted(base_totals.items(), key=lambda kv: -kv[1])[:top]:
        b = base_totals[name] / 1e9
        c = cand_totals.get(name, 0) / 1e9
        print(f"{name[:58]:58s} {len(base_kernels[name]):6d} {b:8.3f} {c:8.3f} "
              f"{c/b if b else float('nan'):6.3f}")

    only_cand = sorted(((n, cand_totals[n]) for n in cand_totals if n not in base_totals),
                       key=lambda kv: -kv[1])[:top]
    if only_cand:
        print(f"\n== Kernels present only in the candidate (top {len(only_cand)}) ==")
        for name, ns in only_cand:
            print(f"{name[:58]:58s} {len(cand_kernels[name]):6d} {ns/1e9:8.3f}")
    only_base = sorted(((n, base_totals[n]) for n in base_totals if n not in cand_totals),
                       key=lambda kv: -kv[1])[:top]
    if only_base:
        print(f"\n== Kernels present only in upstream (top {len(only_base)}) ==")
        for name, ns in only_base:
            print(f"{name[:58]:58s} {len(base_kernels[name]):6d} {ns/1e9:8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
