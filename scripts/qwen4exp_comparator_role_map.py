#!/usr/bin/env python3
"""Map comparator kernel launches onto the shared tensor-role taxonomy.

The gap attribution needs hipEngine and the llama.cpp-family comparators
expressed in one vocabulary. This reads a comparator's ``rocprofv3`` kernel
trace and reports time per shared family.

The families are:

    dense_projection    attention/shared-expert/hyper-connection projections
    expert_gate_up      routed MoE gate and up projections
    expert_down         routed MoE down projection
    moe_reduce          routed-expert weighting and reduction
    hyper_connection    hyper-connection (GR) combine/mix
    gdn                 gated delta net
    qsa_attention       QSA / flash attention
    ple                 per-layer embedding
    indexer             QSA indexer
    elementwise_norm    norms, broadcasts, dtype conversion
    quantize_pack       activation quantization feeding a quantized matmul
    other               everything not otherwise mapped

How the rules were derived, so they can be re-checked rather than trusted:

* **ggml quant type identifies the expert families in the llama.cpp family.**
  ``GGML_TYPE_Q4_K`` is 12, ``Q5_1`` is 7, ``Q8_0`` is 8 (read from
  ``ggml/include/ggml.h`` in the comparator source). The model's
  ``ffn_gate_exps``/``ffn_up_exps`` are the only Q4_K tensors and
  ``ffn_down_exps`` is Q5_1 in 43 of 48 layers, so those two types map to
  expert gate/up and expert down unambiguously.
* **Grid shape separates the remaining expert-down layers.** Four layers carry
  Q8_0 expert-down, which would otherwise hide among the dense Q8_0 dispatches.
  They are identifiable by their grid Y, which equals the routed token count
  (1704/1708) rather than a weight-derived dimension.
* **The MMB comparators name the family directly** (``routed_glu`` vs
  ``routed`` vs ``dense``), so those rules are name matches.

An unmapped kernel above ``--unmapped-floor-ms`` is reported and, with
``--strict``, is an error. That keeps the mapping from rotting silently when a
comparator changes.

Example:
    python3 scripts/qwen4exp_comparator_role_map.py \
        --trace <dir>/gfx1151/*_kernel_trace.csv \
        --label halobox-69946438a-bf16 \
        --output benchmarks/results/<dir>/halobox-roles.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

FAMILIES = (
    "dense_projection",
    "expert_gate_up",
    "expert_down",
    "moe_reduce",
    "hyper_connection",
    "gdn",
    "qsa_attention",
    "ple",
    "indexer",
    "elementwise_norm",
    "quantize_pack",
    "other",
)

# ggml_type values, from ggml/include/ggml.h in the comparator source.
GGML_Q5_1 = 7
GGML_Q8_0 = 8
GGML_Q4_K = 12

# The routed token count appears as grid Y on the expert dispatches. Anything
# else in the Y slot means the dispatch is a dense projection.
ROUTED_GRID_Y = {1704, 1708}


def _ggml_type(name: str) -> int | None:
    m = re.search(r"ggml_type\)(\d+)", name)
    return int(m.group(1)) if m else None


def map_llamacpp(name: str, grid: tuple[str, str, str]) -> str:
    """llama.cpp / ggml kernels."""
    if "gated_delta_net" in name or "gdn_conv" in name or "ssm_conv" in name:
        return "gdn"
    if ("flash_attn_ext" in name or "qsa3_attn" in name or "rope_multi" in name
            or "qsa3_rows" in name or "qsa3_merge" in name or "flash_attn_mask" in name):
        return "qsa_attention"
    if "quantize_mmq_q8_1" in name or "quantize_row_q8" in name:
        return "quantize_pack"
    if ("moe_weighted_reduction" in name or "moe_weighted_sum" in name
            or "topk_moe" in name):
        return "moe_reduce"
    if name.startswith("hc_") or "hc_combine" in name or "hc_mix" in name:
        return "hyper_connection"
    if "mm_ids_helper" in name:
        return "indexer"
    if "top_k" in name or "nary_search" in name:
        return "indexer"
    if "mul_mat_q" in name or name.startswith("Cijk_") or "mul_mat_vec" in name:
        ggml = _ggml_type(name)
        if "routed" in name or "_moe" in name or ggml == GGML_Q4_K:
            return "expert_gate_up" if "down" not in name else "expert_down"
        if ggml == GGML_Q5_1:
            return "expert_down"
        # Q8_0 and rocBLAS GEMM: dense unless the grid marks a routed dispatch.
        if grid[1] in {str(y) for y in ROUTED_GRID_Y}:
            return "expert_down"
        return "dense_projection"
    if "rms_norm" in name or "norm_f32" in name:
        return "elementwise_norm"
    if "k_bin_bcast" in name or "k_fill" in name or "fillBuffer" in name:
        return "elementwise_norm"
    if "concat" in name or "cpy_" in name or "dup_" in name or "cont_" in name:
        return "elementwise_norm"
    if "set_rows" in name or "get_rows" in name:
        return "elementwise_norm"
    if ("unary_gated_op" in name or "scale_f32" in name or "convert_unary" in name
            or "unary_op_kernel" in name or "scale_unary" in name):
        return "elementwise_norm"
    if "copyBuffer" in name or "trampoline_kernel" in name:
        return "elementwise_norm"
    return "other"


def map_mmb(name: str, grid: tuple[str, str, str]) -> str:
    """pwilkin's MMB kernels, which name the family directly."""
    if "gated_delta_net" in name or "gdn_conv" in name or "ssm_conv" in name:
        return "gdn"
    if ("flash_attn_ext" in name or "qsa3_attn" in name or "qsa_expand" in name
            or "qsa3_rows" in name or "qsa3_merge" in name):
        return "qsa_attention"
    if "rope_multi" in name:
        return "qsa_attention"
    if "mmb_routed_glu" in name:
        return "expert_gate_up"
    if "mmb_routed" in name or ("mul_mat_vec_q_moe" in name and _ggml_type(name) == GGML_Q5_1):
        return "expert_down"
    if "mul_mat_vec_q_moe" in name:
        return "expert_gate_up"
    if "mmb_dense" in name or "mmb_f32split" in name or "mul_mat_vec" in name:
        return "dense_projection"
    if "mm_ids_helper" in name or name.startswith("mm_ids"):
        return "indexer"
    if "top_k" in name or "nary_search" in name:
        return "indexer"
    if "moe_weighted_reduction" in name or "topk_moe" in name:
        return "moe_reduce"
    if "hc_combine" in name or "hc_gate_mix" in name or name.startswith("hc_"):
        return "hyper_connection"
    if "mmb_cvt_" in name:
        return "elementwise_norm"
    if "rms_norm" in name or "rms_rows" in name or "norm_f32" in name:
        return "elementwise_norm"
    if "k_bin_bcast" in name or "fillBuffer" in name or "copyBuffer" in name:
        return "elementwise_norm"
    if "unary_gated_op" in name or "scale_f32" in name:
        return "elementwise_norm"
    if "cpy_scalar" in name or "cpy_" in name:
        return "elementwise_norm"
    return "other"


MAPPERS: dict[str, Callable[[str, tuple[str, str, str]], str]] = {
    "llamacpp": map_llamacpp,
    "mmb": map_mmb,
}


def read_trace(path: Path, mapper: Callable[[str, tuple[str, str, str]], str]) -> dict[str, Any]:
    by_family: Counter[str] = Counter()
    by_family_count: Counter[str] = Counter()
    by_kernel: dict[tuple[str, str], list[float]] = {}
    unmapped: Counter[str] = Counter()
    total = 0.0

    starts: list[int] = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("Kind") != "KERNEL_DISPATCH":
                continue
            name = row["Kernel_Name"]
            grid = (row["Grid_Size_X"], row["Grid_Size_Y"], row["Grid_Size_Z"])
            start = int(row["Start_Timestamp"])
            starts.append(start)
            ms = (int(row["End_Timestamp"]) - start) / 1e6
            family = mapper(name, grid)
            total += ms
            by_family[family] += ms
            by_family_count[family] += 1
            entry = by_kernel.setdefault((family, name), [0.0, 0.0])
            entry[0] += ms
            entry[1] += 1
            if family == "other":
                unmapped[name] += ms

    # A trace is only usable for a gap comparison if it is delimited to the
    # measured window. Comparators profiled as a whole server lifetime carry
    # warmup prefills, which inflates every family total.
    span_ms = (max(starts) - min(starts)) / 1e6 if len(starts) > 1 else 0.0
    active_ms = total
    return {
        "total_ms": round(total, 1),
        "trace_span_ms": round(span_ms, 1),
        "kernel_duty_cycle_pct": round(100.0 * active_ms / span_ms, 2) if span_ms else None,
        "dispatches": sum(by_family_count.values()),
        "by_family_ms": {f: round(by_family.get(f, 0.0), 1) for f in FAMILIES},
        "by_family_dispatches": {f: by_family_count.get(f, 0) for f in FAMILIES},
        "by_family_share_pct": {
            f: round(100.0 * by_family.get(f, 0.0) / total, 2) if total else 0.0
            for f in FAMILIES
        },
        "kernels": sorted(
            (
                {"family": f, "kernel": n, "ms": round(v[0], 2), "dispatches": int(v[1])}
                for (f, n), v in by_kernel.items()
            ),
            key=lambda r: -r["ms"],
        ),
        "unmapped_top": [
            {"kernel": n, "ms": round(ms, 2)}
            for n, ms in unmapped.most_common(15)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True, help="rocprofv3 kernel_trace.csv")
    parser.add_argument("--label", required=True)
    parser.add_argument("--dialect", choices=sorted(MAPPERS), default="llamacpp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unmapped-floor-ms", type=float, default=5.0)
    parser.add_argument(
        "--measured-prompt-ms",
        type=float,
        default=None,
        help="The harness's measured prefill wall time for the same case. The "
             "kernel sum should be close to it; a large excess means the trace "
             "covers warmup prefills as well and the family split is not the "
             "measured prefill's.",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    result = read_trace(args.trace, MAPPERS[args.dialect])
    result["label"] = args.label
    result["measured_prompt_ms"] = args.measured_prompt_ms
    result["delimited"] = (
        None
        if args.measured_prompt_ms is None
        else abs(result["total_ms"] - args.measured_prompt_ms)
        <= 0.25 * args.measured_prompt_ms
    )
    result["dialect"] = args.dialect
    result["trace"] = str(args.trace)

    print(f"{args.label}  ({args.dialect})  total {result['total_ms']:.1f} ms, "
          f"{result['dispatches']} dispatches")
    print(f"{'family':20s} {'ms':>10s} {'share':>7s} {'disp':>8s}")
    for family in FAMILIES:
        ms = result["by_family_ms"][family]
        if ms <= 0:
            continue
        print(f"{family:20s} {ms:10.1f} {result['by_family_share_pct'][family]:6.1f}% "
              f"{result['by_family_dispatches'][family]:8d}")

    print(f"\ntrace span {result['trace_span_ms']:.0f} ms, kernel time "
          f"{result['total_ms']:.0f} ms, duty cycle {result['kernel_duty_cycle_pct']}%")
    if args.measured_prompt_ms is not None:
        ratio = result["total_ms"] / args.measured_prompt_ms
        verdict = "delimited" if result["delimited"] else "NOT DELIMITED"
        print(f"measured prompt {args.measured_prompt_ms:.0f} ms -> kernel sum is "
              f"{ratio:.2f}x it: {verdict}")
        if not result["delimited"]:
            print("  the family split below sums over warmup prefills too and must "
                  "not be used for a per-component gap")

    big_unmapped = [u for u in result["unmapped_top"] if u["ms"] >= args.unmapped_floor_ms]
    if big_unmapped:
        print(f"\nunmapped kernels at or above {args.unmapped_floor_ms} ms:")
        for u in big_unmapped:
            print(f"  {u['ms']:9.2f} ms  {u['kernel'][:74]}")
        if args.strict:
            print("\n--strict: refusing to write with kernels unmapped", flush=True)
            return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema": 1,
        "kind": "qwen4exp_comparator_role_map",
        "performance_claim": False,
        "status": "diagnostic",
        "label": args.label,
        "dialect": args.dialect,
        "trace": str(args.trace),
        "families": list(FAMILIES),
        "measured_prompt_ms": args.measured_prompt_ms,
        **result,
        "notes": [
            "Shared taxonomy for the hipEngine/comparator gap attribution.",
            "llama.cpp expert families are identified by ggml quant type; the Q8_0 expert-down layers are separated by routed grid Y.",
            "Time is rocprofv3 kernel duration, not wall time; it excludes host gaps between dispatches.",
        ],
    }, indent=1) + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
