#!/usr/bin/env python3
"""Group kernel time by functional area across several rocprofv3 traces.

Usage: group_breakdown.py <label>=<csv> [<label>=<csv> ...]
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

GROUPS = [
    ("Q4_K MMQ (mul_mat_q type 12)", lambda n: "mul_mat_q<(ggml_type)12" in n),
    ("Q5_K MMQ (mul_mat_q type 13)", lambda n: "mul_mat_q<(ggml_type)13" in n),
    ("Q6_K MMQ (mul_mat_q type 14)", lambda n: "mul_mat_q<(ggml_type)14" in n),
    ("GDN + concat + ssm_conv", lambda n: any(k in n for k in
        ("gated_delta", "concat", "ssm_conv", "dequantize_mul_mat_vec"))),
    ("quantize Q8_1 for MMQ", lambda n: "quantize_mmq_q8_1" in n),
    ("flash attention", lambda n: "flash_attn" in n),
    ("norm", lambda n: "norm" in n),
    ("convert / unary / elementwise", lambda n: any(k in n for k in
        ("convert_unary", "convert_f32", "convert_f16", "unary", "k_bin_bcast", "cpy_scalar",
         "dequantize_block"))),
    ("hipBLASLt GEMM (Cijk_*)", lambda n: n.startswith("Cijk_")),
]


def load(path: str) -> tuple[dict[str, int], dict[str, int]]:
    total: dict[str, int] = defaultdict(int)
    calls: dict[str, int] = defaultdict(int)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Kind"] != "KERNEL_DISPATCH":
                continue
            total[row["Kernel_Name"]] += int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
            calls[row["Kernel_Name"]] += 1
    return total, calls


def by_group(total: dict[str, int], calls: dict[str, int]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {name: [0, 0] for name, _ in GROUPS}
    out["everything else"] = [0, 0]
    for name, ns in total.items():
        for label, predicate in GROUPS:
            if predicate(name):
                out[label][0] += ns
                out[label][1] += calls[name]
                break
        else:
            out["everything else"][0] += ns
            out["everything else"][1] += calls[name]
    return out


labels, data = [], []
for spec in sys.argv[1:]:
    label, path = spec.split("=", 1)
    labels.append(label)
    data.append(by_group(*load(path)))

order = [name for name, _ in GROUPS] + ["everything else"]
header = f"{'group':34s} {'calls':>6s} " + " ".join(f"{l:>12s}" for l in labels)
if len(labels) > 1:
    header += f" {'last-first':>11s}"
print(header)
for name in order:
    row = f"{name:34s} {data[0][name][1]:6d} " + " ".join(f"{d[name][0]/1e9:12.3f}" for d in data)
    if len(labels) > 1:
        row += f" {(data[-1][name][0]-data[0][name][0])/1e9:+11.3f}"
    print(row)
total = f"{'TOTAL':34s} {sum(v[1] for v in data[0].values()):6d} " + \
        " ".join(f"{sum(v[0] for v in d.values())/1e9:12.3f}" for d in data)
if len(labels) > 1:
    total += f" {(sum(v[0] for v in data[-1].values())-sum(v[0] for v in data[0].values()))/1e9:+11.3f}"
print(total)
