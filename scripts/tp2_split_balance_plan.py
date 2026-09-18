#!/usr/bin/env python3
"""Plan an uneven TP2 shard split from measured per-rank rates.

The TP2 route splits every sharded tensor evenly. The two cards are not
equal - on this host the RX 7900 XTX runs the same kernels 15-21% faster than
the W7900 - so an even split makes the slower card the pacer while the faster
card idles inside the exchange spin. This script turns measured per-rank
service rates into a concrete, block-aligned per-rank split.

Scope is deliberately narrow: only ``blk.*.ffn_(gate|up|down).weight`` is
eligible. The attention and GDN tensors are head-structured, and
``partition_groups`` refuses uneven splits for them on purpose - query-head
ownership and KV-head loading have to correspond, and an uneven split there
would be silently wrong attention rather than an error.

Inputs come from two measurements, both recorded in the repository:

* per-rank bytes, from the shard manifest (this script reads it directly);
* per-rank service rates and fixed costs, from the decode kernel inventory
  artifact, or overridden on the command line.

The output is a plan, not a change: it prints the split and the predicted
per-rank and per-step times. ``--json`` writes the artifact form.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MLP_PREFIXES = ("ffn_gate", "ffn_up", "ffn_down")

# A kernel is bandwidth-bound when it streams weights: the GGUF GEMV family
# plus the narrow pair routes. Everything else is fixed work that both ranks
# pay for equal bytes.
BANDWIDTH_MARKERS = ("gemv", "planar", "narrow")

# The exchange spin kernel's duration is mostly waiting for the peer, not
# work. Counting it as cost would hide the imbalance this script exists to
# measure.
SPIN_MARKER = "spin_add"

# The LM head is ``owner``-kind in the manifest (rank 0 loads the whole
# tensor) but the runtime splits its vocabulary rows across the group, so its
# streaming bytes count per rank as half the tensor.
HEAD_TENSOR = "output.weight"


@dataclass(frozen=True)
class TensorPlan:
    name: str
    blocks: int
    block_values: int
    nbytes_per_block: int

    def blocks_for(self, rank: int, counts: list[int]) -> int:
        return counts[rank]


def eligible(name: str) -> bool:
    if not name.endswith(".weight"):
        return False
    leaf = name.split(".", 2)[-1].removesuffix(".weight")
    return leaf in MLP_PREFIXES


def split_blocks(blocks: int, fractions: list[float], block_values: int) -> list[int]:
    """Whole blocks per rank, preserving the total and every rank's minimum.

    Fractions are resolved largest-remainder so the split stays as close to
    the requested ratio as whole blocks allow, and no rank is given zero
    blocks for a tensor it must own.
    """

    total = sum(fractions)
    if total <= 0:
        raise ValueError("fractions must be positive")
    if blocks < len(fractions):
        raise ValueError(f"{blocks} blocks cannot cover {len(fractions)} ranks")
    exact = [blocks * f / total for f in fractions]
    counts = [int(value) for value in exact]
    remainder = blocks - sum(counts)
    order = sorted(range(len(counts)), key=lambda i: exact[i] - counts[i], reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    for index in range(len(counts)):
        if counts[index] < 1:
            donor = max(range(len(counts)), key=lambda i: counts[i])
            if counts[donor] <= 1:
                raise ValueError("cannot leave every rank at least one block")
            counts[donor] -= 1
            counts[index] = 1
    if sum(counts) != blocks:
        raise ValueError("block counts do not sum to the tensor's block count")
    return counts


def balance_fractions(
    fixed_ms: list[float], other_ms: list[float], per_ms: list[float]
) -> list[float]:
    """Fraction of the movable pool that equalizes two ranks' predicted times.

    Rank ``i`` costs ``fixed_ms[i] + other_ms[i] + f_i * per_ms[i]``, where
    ``per_ms[i]`` is the time rank ``i`` would need for the *whole* movable
    pool at its own rate. The fractions sum to one.
    """

    if len(fixed_ms) != 2 or len(other_ms) != 2 or len(per_ms) != 2:
        raise ValueError("the balance solve is written for two ranks")
    slope = per_ms[0] + per_ms[1]
    if slope <= 0:
        raise ValueError("the movable pool must have a positive service time")
    intercept = (fixed_ms[1] + other_ms[1] + per_ms[1]) - (fixed_ms[0] + other_ms[0])
    fraction0 = intercept / slope
    if not 0.0 < fraction0 < 1.0:
        raise ValueError(
            f"the solve puts fraction {fraction0:.4f} of the movable pool on rank 0; "
            "the inputs do not describe an imbalance a split can fix"
        )
    return [fraction0, 1.0 - fraction0]


def derive_rank_costs(
    per_rank: dict[str, dict], streamed_gib: list[float]
) -> tuple[list[float], list[float], list[float]]:
    """Per-rank (rate GiB/s, fixed ms, spin ms) from a kernel inventory.

    The spin kernel's duration is mostly waiting for the peer, so it is
    reported separately and never counted as cost; counting it would hide the
    imbalance this script exists to measure.
    """

    rates: list[float] = []
    fixed: list[float] = []
    spins: list[float] = []
    for rank in range(len(streamed_gib)):
        entry = per_rank[str(rank)]
        streamed_ms = sum(
            kernel["ms_per_step"]
            for kernel in entry["kernels"]
            if any(marker in kernel["kernel"] for marker in BANDWIDTH_MARKERS)
        )
        spin_ms = sum(
            kernel["ms_per_step"]
            for kernel in entry["kernels"]
            if SPIN_MARKER in kernel["kernel"]
        )
        if streamed_ms <= 0:
            raise ValueError(f"rank {rank} has no bandwidth-bound kernel time")
        rates.append(streamed_gib[rank] / (streamed_ms / 1000.0))
        fixed.append(entry["kernel_ms_per_step"] - streamed_ms - spin_ms)
        spins.append(spin_ms)
    return rates, fixed, spins


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    parser.add_argument(
        "--inventory",
        default="benchmarks/results/2026-09-18-w7900-tp2-decode-kernel-inventory.json",
        help="decode kernel inventory artifact supplying per-rank rates",
    )
    parser.add_argument(
        "--rate",
        action="append",
        type=float,
        default=None,
        help="per-rank bandwidth-bound service rate in GiB/s (repeat per rank)",
    )
    parser.add_argument(
        "--fixed-ms",
        action="append",
        type=float,
        default=None,
        help="per-rank non-splittable time per step in ms (repeat per rank)",
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    from hipengine.loading.gguf import scan_gguf
    from hipengine.loading.qwen35_gguf_admission import build_qwen35_gguf_role_manifest
    from hipengine.loading.qwen35_gguf_materialize import build_qwen35_gguf_tensor_map
    from hipengine.loading.qwen35_gguf_shards import build_shard_manifest

    info = scan_gguf(args.model)
    model_map = build_qwen35_gguf_tensor_map(info)
    fingerprint = build_qwen35_gguf_role_manifest(model_map).fingerprint
    manifest = build_shard_manifest(
        info, world_size=args.world_size, model_hash=fingerprint
    )

    plans: list[TensorPlan] = []
    for plan in manifest.tensors:
        if not eligible(plan.name) or plan.replicated:
            continue
        blocks = int(plan.source_shape[0 if plan.kind == "column" else 1]) // int(
            plan.block_size
        )
        plans.append(
            TensorPlan(
                name=plan.name,
                blocks=blocks,
                block_values=int(plan.block_size),
                nbytes_per_block=int(plan.source_nbytes) // blocks,
            )
        )
    if not plans:
        raise SystemExit("no eligible MLP tensors in the manifest")

    splittable_bytes = sum(p.blocks * p.nbytes_per_block for p in plans)
    rank_bytes_even = [
        manifest.rank_bytes(rank) for rank in range(args.world_size)
    ]

    # Streamed bytes exclude ``owner`` tensors: rank 0 holds the whole LM head
    # and the whole token embedding, but the head's vocabulary rows are split
    # at runtime and the embedding is read one row per token. Counting either
    # as rank 0's streamed bytes would hide the imbalance.
    head_bytes = sum(
        plan.source_nbytes for plan in manifest.tensors if plan.name == HEAD_TENSOR
    )
    streamed_bytes = [
        sum(
            plan.slice_for(rank).local_nbytes
            for plan in manifest.tensors
            if not plan.replicated and plan.kind != "owner"
        )
        + head_bytes // args.world_size
        for rank in range(args.world_size)
    ]
    eligible_bytes = [0] * args.world_size
    for plan in manifest.tensors:
        if not eligible(plan.name) or plan.replicated:
            continue
        for rank in range(args.world_size):
            eligible_bytes[rank] += plan.slice_for(rank).local_nbytes
    other_streamed = [
        streamed_bytes[rank] - eligible_bytes[rank]
        for rank in range(args.world_size)
    ]

    rates = args.rate
    fixed_ms = args.fixed_ms
    spin_ms = [0.0] * args.world_size
    if rates is None or fixed_ms is None:
        inventory = json.loads((REPO / args.inventory).read_text())
        derived_rate, derived_fixed, spin_ms = derive_rank_costs(
            inventory["per_rank"],
            [streamed_bytes[rank] / 2**30 for rank in range(args.world_size)],
        )
        if rates is None:
            rates = derived_rate
        if fixed_ms is None:
            fixed_ms = derived_fixed

    # Solve for the eligible byte fraction that equalizes the two ranks. Only
    # the eligible pool moves; the rest of each rank's streamed bytes stay put.
    # ``rates`` are GiB/s, so every byte-to-time conversion scales by 1000.
    other_ms = [
        other_streamed[rank] / 2**30 / rates[rank] * 1000.0
        for rank in range(args.world_size)
    ]
    total_eligible = sum(eligible_bytes) / 2**30
    per_ms = [total_eligible / rates[rank] * 1000.0 for rank in range(args.world_size)]
    try:
        fractions = balance_fractions(fixed_ms, other_ms, per_ms)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    balanced_ms = fixed_ms[0] + other_ms[0] + fractions[0] * per_ms[0]

    per_tensor = []
    uneven_rank_bytes = list(rank_bytes_even)
    uneven_eligible = [0] * args.world_size
    for plan in plans:
        counts = split_blocks(plan.blocks, fractions, plan.block_values)
        even = [plan.blocks // args.world_size] * args.world_size
        for rank in range(args.world_size):
            uneven_rank_bytes[rank] += (counts[rank] - even[rank]) * plan.nbytes_per_block
            uneven_eligible[rank] += counts[rank] * plan.nbytes_per_block
        per_tensor.append(
            {
                "name": plan.name,
                "blocks": plan.blocks,
                "blocks_even": even,
                "blocks_uneven": counts,
                "mib_uneven": [round(c * plan.nbytes_per_block / 2**20, 3) for c in counts],
            }
        )

    even_eligible = [
        sum(eligible_bytes) / args.world_size / 2**30 for _ in range(args.world_size)
    ]
    even_ms = [
        fixed_ms[rank]
        + other_ms[rank]
        + even_eligible[rank] / rates[rank] * 1000.0
        for rank in range(args.world_size)
    ]
    uneven_ms = [
        fixed_ms[rank]
        + other_ms[rank]
        + uneven_eligible[rank] / 2**30 / rates[rank] * 1000.0
        for rank in range(args.world_size)
    ]
    pacer_even = max(even_ms)
    pacer_uneven = max(uneven_ms)

    report = {
        "schema": 1,
        "kind": "tp2-split-balance-plan",
        "model": args.model,
        "world_size": args.world_size,
        "eligible": "blk.*.ffn_(gate|up|down).weight",
        "splittable_gib": round(splittable_bytes / 2**30, 4),
        "rank_rates_gib_s": [round(r, 4) for r in rates],
        "rank_fixed_ms": [round(m, 4) for m in fixed_ms],
        "rank_spin_ms": [round(m, 4) for m in spin_ms],
        "rank_streamed_gib": [round(b / 2**30, 4) for b in streamed_bytes],
        "rank_eligible_gib": [round(b / 2**30, 4) for b in eligible_bytes],
        "predicted_even_ms": [round(m, 4) for m in even_ms],
        "predicted_uneven_ms": [round(m, 4) for m in uneven_ms],
        "byte_fractions": [round(f, 6) for f in fractions],
        "rank_bytes_even_gib": [round(b / 2**30, 4) for b in rank_bytes_even],
        "rank_bytes_uneven_gib": [round(b / 2**30, 4) for b in uneven_rank_bytes],
        "predicted_balanced_ms_per_step": round(balanced_ms, 4),
        "predicted_pacer_ms_even": round(pacer_even, 4),
        "predicted_pacer_ms_uneven": round(pacer_uneven, 4),
        "predicted_saving_ms_per_step": round(pacer_even - pacer_uneven, 4),
        "tensors": per_tensor,
    }

    print(f"eligible tensors: {len(plans)}  splittable {splittable_bytes / 2**30:.4f} GiB")
    print("per-rank rates GiB/s:", report["rank_rates_gib_s"])
    print("per-rank fixed ms   :", report["rank_fixed_ms"])
    print("per-rank spin ms    :", report["rank_spin_ms"], "(excluded from cost)")
    print("per-rank streamed GiB:", report["rank_streamed_gib"], "eligible:", report["rank_eligible_gib"])
    print("byte fractions      :", report["byte_fractions"])
    print("resident even  GiB  :", report["rank_bytes_even_gib"])
    print("resident uneven GiB :", report["rank_bytes_uneven_gib"])
    print(
        f"pacer {report['predicted_pacer_ms_even']:.3f} -> "
        f"{report['predicted_pacer_ms_uneven']:.3f} ms/step "
        f"(saving {report['predicted_saving_ms_per_step']:.3f} ms/step)"
    )
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
