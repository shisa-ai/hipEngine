#!/usr/bin/env python3
"""Attribute one prefill to kernel families for each engine, on one case.

The engine taxonomies do not line up: hipEngine reports owners
(``linear``/``moe``/``gr_read``/``qsa``/``gdn``), and the llama.cpp family
reports flat kernel names. This reads the raw ``rocprofv3`` kernel traces and
buckets both sides into a small shared set of *work* categories, so the two can
be put side by side per 4096-token prefill.

The normalisation matters more than the bucketing. A llama.cpp window contains
the warmup request plus the measured one, so its kernel total covers two
prefills while hipEngine's covers one. Every row here is divided by the number
of prefills the window actually contains, which is passed in and recorded
rather than guessed.

Bucketing is by kernel name and is deliberately explicit: a name that matches
nothing lands in ``unattributed`` and is printed, so a new kernel family cannot
quietly inflate a category.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

# Order matters: the first match wins. Two orderings are load-bearing.
#
# * ``moe_gate_up`` precedes ``moe_down`` because ``mmb_routed_glu`` contains
#   ``mmb_routed``.
# * ``quantize_pack`` precedes ``dense_matmul`` because ``quantize_mmq_q8_1``
#   contains ``mmq``. Ordering these the other way counted an activation
#   packer as a matmul.
#
# Patterns are matched against a lowercased name, so they must be lowercase:
# ``Cijk`` never matched and the rocBLAS/Tensile GEMM fell into
# ``unattributed`` for every llama.cpp-family engine.
BUCKETS: tuple[tuple[str, str], ...] = (
    # MoE expert gate/up (hipEngine iu8 risk+repair pair; fork fused GLU).
    ("moe_gate_up",
     r"q4_k_selected_dual_wmma_iu8_risk|q4_k_selected_dual_sparse_exact_repair"
     r"|mmb_routed_glu|mul_mat_q_routed|expert_gate|routed_glu"
     # Plain llama.cpp routes the Q4_K experts through the un-suffixed MMQ
     # kernel; ggml_type 12 is Q4_K and every dense tensor in this file is
     # Q8_0/Q5_1/Q6_K, so this name is the expert gate/up there.
     r"|mul_mat_q<\(ggml_type\)12"),
    # MoE expert down and its reduction.
    ("moe_down",
     r"q5_1_selected_wmma_iu8_risk|q5_1_selected_sparse_exact_repair"
     r"|q8_0_selected_sparse_repair|mmb_routed_kernel|moe_weighted_reduction"
     r"|grouped_down|expert_down"),
    # Hyper-connection / GR projections and their reduce-combine tails.
    ("gr_read", r"gr_up|gr_down|gr_write|gr_read|hc_|hyper"),
    # QSA and paged attention. `attn` alone catches names like `qsa3_attn_kernel`
    # that neither `qsa_` nor `\battn` matched.
    ("attention", r"qsa|flash_attn|fattn|paged_full_attn|attn|softmax"),
    # Gated DeltaNet / SSM and its convolution.
    ("gdn", r"gated_delta|gdn_prefill|gdn_|ssm_|mamba|conv"),
    # Expert routing and index bookkeeping.
    ("router_index", r"router_logits|ids_helper|topk|top_k|argsort|\bsort|index"),
    # Operand staging, packing and dtype conversion. Ahead of the matmul row so
    # an activation quantizer is not counted as matrix arithmetic.
    ("quantize_pack",
     r"quantize|dequantize|im2col|\bpack|concat|transpose|permute|convert"),
    # Dense quantized matmuls, whatever the kernel is called.
    ("dense_matmul",
     r"gguf_k_prefill_out|mul_mat|mmb_dense|mmb_f32split|gemm|mmq|wmma|cijk"
     r"|rocblas|tensile|dense_gemv|grouped_wmma"),
    # Normalisation and elementwise tails.
    ("elementwise_norm",
     r"norm|silu|gelu|sigmoid|\badd|mul_f32|scale|cpy|sqrt|tanh|clamp|rope"
     r"|bin_bcast|unary|activation|weighted_lanes"),
)


def bucket(name: str) -> str:
    """Classify one kernel name. Unmatched names return ``unattributed``."""
    lowered = name.lower()
    for label, pattern in BUCKETS:
        if re.search(pattern, lowered):
            return label
    return "unattributed"


def _read_trace(trace_dir: Path) -> tuple[dict[str, float], dict[str, float], int]:
    csvs = sorted(trace_dir.rglob("*kernel_trace*.csv"))
    if not csvs:
        raise SystemExit(f"no kernel trace under {trace_dir}")
    buckets: dict[str, float] = defaultdict(float)
    kernels: dict[str, float] = defaultdict(float)
    count = 0
    with csvs[-1].open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row.get("Name") or row.get("Kernel_Name") or ""
            duration = row.get("Duration") or row.get("Duration_ns")
            if duration is None and name and row.get("Start_Timestamp") and row.get("End_Timestamp"):
                duration = int(float(row["End_Timestamp"])) - int(float(row["Start_Timestamp"]))
            if not name or not duration:
                continue
            ms = int(float(duration)) / 1e6
            buckets[bucket(name)] += ms
            kernels[name] += ms
            count += 1
    return dict(buckets), dict(kernels), count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", action="append", required=True,
        metavar="LABEL=TRACE_DIR:PREFILLS",
        help=(
            "One row. TRACE_DIR holds the rocprofv3 output; PREFILLS is how many "
            "prefills the window contains. A llama.cpp window with one warmup and "
            "one measured request is 2, and so is a hipEngine profiled capture, "
            "whose warmup prefill sits in the same trace; the marker-delimited "
            "owner total is the independent check that the divisor is right."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for spec in args.engine:
        label, _, rest = spec.partition("=")
        trace_dir, _, prefills = rest.rpartition(":")
        if not label or not trace_dir or not prefills:
            raise SystemExit(f"bad --engine spec: {spec!r}")
        prefills_n = int(prefills)
        buckets, kernels, count = _read_trace(Path(trace_dir))
        total = sum(buckets.values())
        rows.append({
            "label": label,
            "trace_dir": trace_dir,
            "prefills_in_window": prefills_n,
            "kernels_in_window": count,
            "window_device_ms": round(total, 1),
            "per_prefill_device_ms": round(total / prefills_n, 1),
            "per_prefill_ms": {
                k: round(v / prefills_n, 1)
                for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])
            },
            "top_kernels_per_prefill_ms": {
                k: round(v / prefills_n, 1)
                for k, v in sorted(kernels.items(), key=lambda kv: -kv[1])[:15]
            },
        })

    shared = sorted({k for row in rows for k in row["per_prefill_ms"]})
    header = f"{'bucket':18s}" + "".join(f"{row['label'][:16]:>18s}" for row in rows)
    lines = [header]
    for bucket in shared:
        lines.append(
            f"{bucket:18s}"
            + "".join(f"{row['per_prefill_ms'].get(bucket, 0.0):18.1f}" for row in rows)
        )
    lines.append(
        f"{'TOTAL':18s}"
        + "".join(f"{row['per_prefill_device_ms']:18.1f}" for row in rows)
    )
    table = "\n".join(lines)
    print(table)

    args.output.write_text(
        json.dumps({"schema": 1, "kind": "per_prefill_kernel_attribution", "rows": rows},
                   indent=1) + "\n"
    )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
