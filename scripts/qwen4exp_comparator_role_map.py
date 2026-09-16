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
    """llama.cpp / ggml kernels.

    Also handles the ``mmb_*`` family: halo-box PR #63 ports the Strix Halo
    optimization line's MMB kernels into this tree, so a current llama.cpp-family
    build emits both naming schemes. Dialect is no longer a reliable
    discriminator and the two mappers delegate to each other.
    """
    if "mmb_" in name or "mm_ids" in name or "hc_gate_mix" in name:
        return map_mmb(name, grid)
    if "gated_delta_net" in name or "gdn_conv" in name or "ssm_conv" in name:
        return "gdn"
    if ("flash_attn_ext" in name or "qsa3_attn" in name or "rope_multi" in name
            or "qsa3_rows" in name or "qsa3_merge" in name or "flash_attn_mask" in name
            or "qsa_expand" in name):
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
    if "rms_norm" in name or "rms_rows" in name or "norm_f32" in name:
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
    """pwilkin's MMB kernels, which name the family directly.

    Delegates to the llama.cpp rules for names this dialect does not own, so one
    mapper serves a build that contains both schemes.
    """
    if not ("mmb_" in name or "mm_ids" in name or "hc_gate_mix" in name):
        return map_llamacpp(name, grid)
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


def _hipengine_bare(name: str) -> str:
    """Strip the template-argument and anonymous-namespace noise from a symbol."""
    name = re.sub(r"^\(anonymous namespace\)::", "", name)
    name = re.sub(r"^void\s+", "", name)
    return name.strip()


def map_hipengine(name: str, grid: tuple[str, str, str]) -> str:
    """hipEngine's Qwen4Exp prefill kernels, in the shared taxonomy.

    hipEngine carries the tensor role in a separate mechanism -- a ROCTX range,
    or the launch-census role stack -- and not in the symbol, so one symbol can
    serve several roles. These rules therefore key on the operation the symbol
    performs. The two genuinely ambiguous groups are resolved by quant type,
    the same way the llama.cpp rules resolve the expert families:

    * ``selected_dual_wmma_iu8_risk_prefill`` is the routed gate+up pair, and
      this model's gate/up expert tensors are Q4_K/Q5_K.
    * ``q5_1_selected_wmma_iu8_risk_prefill`` and
      ``q8_0_selected_grouped_wmma_prefill`` are the routed down projection,
      which is Q5_1 in 43 of 48 layers and Q8_0 in the remaining five.

    Two deliberate folds, both reported separately in the comparison output so
    they cannot hide:

    * The iu8 exact-repair passes are folded into the family of the matmul they
      correct, because that is the operation whose cost they are. hipEngine has
      no equivalent of this machinery on the dense path and the comparator has
      none at all, so the folded milliseconds are also totalled as
      ``risk_or_repair_ms``.
    * GLU-style activations (``silu_mul``, ``scaled_silu``, ``gated_mean_sigmoid``)
      go to ``elementwise_norm``, matching the comparator rules, where
      ``unary_gated_op`` is elementwise and not part of the expert matmul.

    ``grid`` is accepted for signature compatibility and is not used: no rule
    here needs it, unlike the llama.cpp rules where grid Y separates a routed
    dispatch from a dense one.
    """
    bare = _hipengine_bare(name)

    # Routed MoE projections and their exact-repair passes.
    if "selected_dual_wmma_iu8_risk_prefill" in bare:
        return "expert_gate_up"
    if "selected_dual_sparse_exact_repair" in bare:
        return "expert_gate_up"
    if "selected_wmma_iu8_risk_prefill" in bare:
        return "expert_down"
    if "selected_grouped_wmma_prefill" in bare:
        return "expert_down"
    if "selected_sparse_exact_repair" in bare or "selected_sparse_repair" in bare:
        return "expert_down"

    # Routing, scatter and reduction around the routed experts.
    for token in (
        "router_logits", "router_select", "moe_group_scatter_gather",
        "moe_wmma_tile_map", "moe_group_prefix", "moe_group_count",
        "weighted_lanes_sum", "weighted_lanes_inverse",
    ):
        if token in bare:
            return "moe_reduce"

    # Hyper-connection read/up and write-back. The projections that consume
    # these tensors are dense_projection, as in the comparator taxonomy.
    if "gr_up" in bare or bare.startswith("gr_write"):
        return "hyper_connection"

    if "gdn_prefill" in bare or "linear_attn_conv" in bare:
        return "gdn"

    for token in (
        "paged_full_attn", "qsa_sparse_attention", "qsa_score",
        "qsa_split_norm_rope", "qsa_norm_rope", "qsa_gate_context",
        "qsa_pool_norm_rope",
    ):
        if token in bare:
            return "qsa_attention"

    # Selection and gather for the sparse-attention index, analogous to the
    # comparator's top_k / nary_search / mm_ids_helper rules.
    if "qsa_topk_expand" in bare or "qsa_scatter_index_keys" in bare:
        return "indexer"

    if "ple_" in bare:
        return "ple"

    for token in (
        "rmsnorm", "silu_mul", "scaled_silu", "gated_mean_sigmoid",
        "f32_to_bf16", "bf16_to_f32", "shared_gate_combine",
        "repeat_bf16_branches", "fillBuffer", "copyBuffer",
    ):
        if token in bare:
            return "elementwise_norm"

    # The prompt K/V write into the paged cache. The comparator writes the same
    # bytes with a ``set_rows``/``cpy`` op, which the llama.cpp rules send to
    # elementwise_norm, so this goes there too rather than to ``other``.
    if "write_paged_kv" in bare:
        return "elementwise_norm"

    for token in (
        "dense_gemv", "gguf_k_prefill_out_coltile_rowbatch",
        "gguf_k_pack8_prefill_out",
    ):
        if token in bare:
            return "dense_projection"

    return "other"


MAPPERS: dict[str, Callable[[str, tuple[str, str, str]], str]] = {
    "llamacpp": map_llamacpp,
    "mmb": map_mmb,
    "hipengine": map_hipengine,
}


def read_trace(
    path: Path,
    mapper: Callable[[str, tuple[str, str, str]], str],
    burst_gap_ms: float = 250.0,
    delimit: str = "none",
) -> dict[str, Any]:
    dispatches: list[tuple[int, int, str, tuple[str, str, str]]] = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("Kind") != "KERNEL_DISPATCH":
                continue
            dispatches.append((
                int(row["Start_Timestamp"]),
                int(row["End_Timestamp"]),
                row["Kernel_Name"],
                (row["Grid_Size_X"], row["Grid_Size_Y"], row["Grid_Size_Z"]),
            ))
    if not dispatches:
        raise SystemExit(f"no KERNEL_DISPATCH rows in {path}")
    dispatches.sort(key=lambda d: d[0])

    # Split on idle gaps. A prefill request is a dense run of dispatches; the
    # gaps between requests, and the long tail of graph building at startup,
    # are where the boundaries are. This is what delimits a comparator capture
    # whose trace covers the whole server lifetime.
    gap_ns = int(burst_gap_ms * 1e6)
    bursts: list[list[tuple[int, int, str, tuple[str, str, str]]]] = [[dispatches[0]]]
    for prev, cur in zip(dispatches, dispatches[1:]):
        if cur[0] - prev[1] > gap_ns:
            bursts.append([])
        bursts[-1].append(cur)

    burst_report = []
    for index, burst in enumerate(bursts):
        span = (burst[-1][1] - burst[0][0]) / 1e6
        busy = sum(e - s for s, e, _, _ in burst) / 1e6
        burst_report.append({
            "index": index,
            "dispatches": len(burst),
            "span_ms": round(span, 1),
            "kernel_ms": round(busy, 1),
            "duty_pct": round(100.0 * busy / span, 1) if span else None,
        })

    selected = dispatches
    selected_bursts: list[int] | None = None
    if delimit != "none":
        if delimit == "last-burst":
            chosen = [len(bursts) - 1]
        elif delimit == "last-two":
            chosen = list(range(max(0, len(bursts) - 2), len(bursts)))
        elif delimit.startswith("burst:"):
            chosen = [int(delimit.split(":", 1)[1])]
        else:
            raise SystemExit(f"unknown --delimit {delimit!r}")
        for index in chosen:
            if index >= len(bursts):
                raise SystemExit(
                    f"--delimit {delimit} asks for burst {index} but the trace "
                    f"has {len(bursts)}"
                )
        selected_bursts = chosen
        selected = [d for index in chosen for d in bursts[index]]

    by_family: Counter[str] = Counter()
    by_family_count: Counter[str] = Counter()
    by_kernel: dict[tuple[str, str], list[float]] = {}
    unmapped: Counter[str] = Counter()
    total = 0.0
    starts = [d[0] for d in selected]

    for start, end, name, grid in selected:
        ms = (end - start) / 1e6
        family = mapper(name, grid)
        total += ms
        by_family[family] += ms
        by_family_count[family] += 1
        entry = by_kernel.setdefault((family, name), [0.0, 0.0])
        entry[0] += ms
        entry[1] += 1
        if family == "other":
            unmapped[name] += ms

    span_ms = (max(starts) - min(starts)) / 1e6 if len(starts) > 1 else 0.0
    active_ms = total
    return {
        "total_ms": round(total, 1),
        "trace_span_ms": round(span_ms, 1),
        "kernel_duty_cycle_pct": round(100.0 * active_ms / span_ms, 2) if span_ms else None,
        "delimit": delimit,
        "delimit_bursts": selected_bursts,
        "bursts_in_trace": burst_report,
        "burst_gap_ms": burst_gap_ms,
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
        "--delimit",
        default="none",
        help=(
            "Restrict the family split to part of the trace: 'none', "
            "'last-burst', 'last-two', or 'burst:N'. Bursts are the dense runs "
            "of dispatches separated by idle gaps, so this is what recovers a "
            "single measured prefill from a whole-server capture."
        ),
    )
    parser.add_argument(
        "--burst-gap-ms",
        type=float,
        default=250.0,
        help="Idle gap that separates two bursts (default 250 ms).",
    )
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

    result = read_trace(
        args.trace, MAPPERS[args.dialect],
        burst_gap_ms=args.burst_gap_ms, delimit=args.delimit,
    )
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
    print(f"bursts in trace (gap > {result['burst_gap_ms']:.0f} ms), "
          f"delimit={result['delimit']}")
    for burst in result["bursts_in_trace"]:
        mark = ""
        if result["delimit_bursts"] and burst["index"] in result["delimit_bursts"]:
            mark = "  <- selected"
        print(f"  burst {burst['index']:3d}: {burst['dispatches']:6d} disp  "
              f"span {burst['span_ms']:9.1f} ms  kernel {burst['kernel_ms']:9.1f} ms  "
              f"duty {burst['duty_pct']}%{mark}")
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
