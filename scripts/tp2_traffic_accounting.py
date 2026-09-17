#!/usr/bin/env python3
"""Per-token weight-traffic accounting for the TP2 shard plan (derived, not measured).

This script answers a design question, not a performance question: for a given
sharded-tensor set, how many bytes must each rank read per decode token, and
what does that imply for the decode wall given a *measured* achieved bandwidth?
Every input is either a file on disk (the GGUF index, the shard plan report) or
a number quoted from a named benchmark artifact, so the derivation is
reproducible and no rate is claimed that was not measured somewhere else.

The measured inputs are passed on the command line and recorded in the output:

    --tp1-p50-ms / --tp1-gib   the matched resident TP1 control's decode p50 and
                               the bytes it reads per token (the full AR set)
    --tp2-p50-ms               the TP2 default route's decode p50

Derived outputs:

* per-role bytes in the AR tensor set (MLP, attention, GDN/SSM, head, embedding,
  norms) from the GGUF index;
* per-rank per-token bytes for the *current* TP2 route (MLP + head sharded,
  everything else replicated) and for the *full* degree-2 shard plan;
* the traffic ratio and the implied wall if the current route achieved the
  resident route's measured bandwidth per byte;
* the reduction-point count each route pays (one cross-rank sum per row-split
  tensor per token).

The implied walls are arithmetic on a measured bandwidth, not measurements.
They bound what a traffic reduction can buy; they do not predict a wall,
because the wall is also set by the exchange, the replicated non-bandwidth-bound
work (GDN recurrence) and device-side gaps.
"""

from __future__ import annotations

import argparse
import collections
import json
import platform
import re
import time
from pathlib import Path
from typing import Any

from hipengine.loading.gguf import scan_gguf

#: Role classification by tensor-name suffix. The MTP/NextN block is not part of
#: the autoregressive shard set (the shard plan excludes its 15 tensors too), so
#: the accounting drops every ``blk.<n>.*`` at or above the AR block count.
ROLE_GROUPS = (
    ("mlp", ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")),
    (
        "attention",
        (
            "attn_qkv.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_gate.weight",
        ),
    ),
    (
        "gdn_ssm",
        (
            "ssm_out.weight",
            "ssm_alpha.weight",
            "ssm_beta.weight",
            "ssm_conv1d.weight",
            "ssm_a",
            "ssm_dt.bias",
            "ssm_norm.weight",
        ),
    ),
    ("head", ("output.weight",)),
    ("embedding", ("token_embd.weight",)),
)

#: The tensor suffixes the current TP2 route splits (row/column sharded) and the
#: ones whose cross-rank reduction it therefore pays.
CURRENT_SHARDED_SUFFIXES = ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")
CURRENT_REDUCED_SUFFIXES = ("ffn_down.weight",)


def _suffix(name: str) -> str:
    match = re.match(r"^blk\.\d+\.(.+)$", name)
    return match.group(1) if match else name


def _role_of(suffix: str) -> str:
    for role, suffixes in ROLE_GROUPS:
        if suffix in suffixes:
            return role
    return "norms_other"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    parser.add_argument(
        "--shard-plan",
        type=Path,
        default=Path("benchmarks/results/tp2_shard_plan_report.json"),
    )
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument(
        "--tp1-p50-ms",
        type=float,
        required=True,
        help="the faster matched resident TP1 arm's decode p50 (the headline control)",
    )
    parser.add_argument(
        "--tp1-slow-rank-p50-ms",
        type=float,
        default=None,
        help=(
            "the slowest resident TP1 rank's decode p50. A two-rank wall is paced "
            "by its slowest rank, so this is the binding reference for the "
            "implied TP2 walls; defaults to --tp1-p50-ms"
        ),
    )
    parser.add_argument(
        "--tp1-gib",
        type=float,
        default=None,
        help=(
            "the resident control's per-token weight bytes. Defaults to the "
            "AR tensor set this script derives from the same index (the MTP "
            "block is not read by an autoregressive token); override only to "
            "declare a different route"
        ),
    )
    parser.add_argument("--tp2-p50-ms", type=float, required=True)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    info = scan_gguf(args.model)
    metadata = info.metadata or {}
    block_count = int(metadata.get("qwen35.block_count", 0))
    ar_blocks = block_count - int(metadata.get("qwen35.nextn_predict_layers", 0))
    role_bytes: collections.Counter[str] = collections.Counter()
    suffix_bytes: collections.Counter[str] = collections.Counter()
    layer_counts: collections.Counter[str] = collections.Counter()
    mtp_bytes = 0
    for tensor in info.tensors:
        suffix = _suffix(tensor.name)
        block = re.match(r"^blk\.(\d+)\.", tensor.name)
        if tensor.name.startswith("nextn.") or (
            block is not None and ar_blocks and int(block.group(1)) >= ar_blocks
        ):
            mtp_bytes += tensor.nbytes
            continue
        role_bytes[_role_of(suffix)] += tensor.nbytes
        suffix_bytes[suffix] += tensor.nbytes
        if block is not None:
            layer_counts[suffix] += 1
    current_reductions = layer_counts["ffn_down.weight"]
    plan_reductions_derived = (
        layer_counts["ffn_down.weight"]
        + layer_counts["ssm_out.weight"]
        + layer_counts["attn_output.weight"]
    )

    ar_bytes = sum(role_bytes.values())
    gib = float(2**30)

    sharded_bytes = sum(suffix_bytes[s] for s in CURRENT_SHARDED_SUFFIXES)
    replicated_bytes = ar_bytes - sharded_bytes
    # The head is split by vocabulary rows and needs no reduction (each rank
    # owns disjoint rows, so the argmax is taken over the concatenation).
    head_bytes = role_bytes["head"]
    embedding_bytes = role_bytes["embedding"]

    # Per-token per-rank bytes. The embedding is a single-row gather, so its
    # table size is not read per token; the head is read per token.
    per_token_current = (
        (sharded_bytes - 0) / args.degree
        + (replicated_bytes - head_bytes - embedding_bytes)
        + head_bytes / args.degree
    )

    plan = json.loads(args.shard_plan.read_text())
    degree = plan["degrees"][str(args.degree)]
    plan_rank_bytes = degree["rank_bytes"]
    plan_reductions = degree["reduction_points"]
    plan_by_suffix = degree["reduction_points_by_suffix"]

    tp1_gib = args.tp1_gib if args.tp1_gib is not None else ar_bytes / gib
    # Bandwidths are decimal (bytes per second); keep every rate in bytes and
    # convert to GiB only for display, so the binary/decimal factor cannot leak
    # into an implied wall.
    tp1_bytes = tp1_gib * gib
    slow_ms = (
        args.tp1_slow_rank_p50_ms
        if args.tp1_slow_rank_p50_ms is not None
        else args.tp1_p50_ms
    )
    resident_bps = tp1_bytes / (args.tp1_p50_ms / 1e3)
    slow_rank_bps = tp1_bytes / (slow_ms / 1e3)
    # The wall is paced by the slowest rank, so the binding comparison uses that
    # rank's own measured resident bandwidth.
    implied_current_ms = per_token_current / slow_rank_bps * 1e3
    implied_full_ms = max(plan_rank_bytes) / slow_rank_bps * 1e3
    measured_bps = per_token_current / (args.tp2_p50_ms / 1e3)

    record: dict[str, Any] = {
        "schema": 1,
        "kind": "tp2-per-token-traffic-accounting",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "model": args.model,
        "derived_not_measured": True,
        "model_bytes": ar_bytes,
        "model_gib": round(ar_bytes / gib, 4),
        "mtp_bytes_excluded": mtp_bytes,
        "role_gib": {k: round(v / gib, 4) for k, v in sorted(role_bytes.items())},
        "role_share": {
            k: round(v / ar_bytes, 4) for k, v in sorted(role_bytes.items())
        },
        "ar_blocks": ar_blocks,
        "layer_tensor_counts": dict(sorted(layer_counts.items())),
        "current_route": {
            "sharded_suffixes": list(CURRENT_SHARDED_SUFFIXES),
            "sharded_gib": round(sharded_bytes / gib, 4),
            "replicated_gib": round(replicated_bytes / gib, 4),
            "per_rank_per_token_gib": round(per_token_current / gib, 4),
            "reduction_points": current_reductions,
            "reduction_points_note": (
                "one cross-rank sum per row-split tensor per token: ffn_down is "
                "row-split at every AR block and needs one; ffn_gate/ffn_up are "
                "column-split and need none; the head's vocabulary split needs "
                "none (disjoint rows)"
            ),
        },
        "full_plan": {
            "degree": args.degree,
            "per_rank_per_token_gib": [
                round(b / gib, 4) for b in plan_rank_bytes
            ],
            "reduction_points": plan_reductions,
            "reduction_points_by_suffix": plan_by_suffix,
            "reduction_points_derived_from_index": plan_reductions_derived,
        },
        "measured_inputs": {
            "tp1_resident_decode_p50_ms": args.tp1_p50_ms,
            "tp1_resident_gib_per_token": round(tp1_gib, 4),
            "tp1_gib_source": (
                "declared on the command line"
                if args.tp1_gib is not None
                else "derived from the AR tensor set in the same index"
            ),
            "tp1_resident_achieved_gb_s": round(resident_bps / 1e9, 1),
            "tp1_slow_rank_p50_ms": slow_ms,
            "tp1_slow_rank_achieved_gb_s": round(slow_rank_bps / 1e9, 1),
            "tp2_default_decode_p50_ms": args.tp2_p50_ms,
            "tp2_current_achieved_gb_s": round(measured_bps / 1e9, 1),
        },
        "implied": {
            "current_route_ms_at_slow_rank_bandwidth": round(implied_current_ms, 3),
            "full_plan_ms_at_slow_rank_bandwidth": round(implied_full_ms, 3),
            "measured_wall_minus_implied_ms": round(
                args.tp2_p50_ms - implied_current_ms, 3
            ),
            "traffic_ratio_current_over_full_plan": round(
                per_token_current / min(plan_rank_bytes), 4
            ),
            "note": (
                "arithmetic on the measured resident bandwidth, not measurements; "
                "the wall is also set by the exchange, replicated non-bandwidth-"
                "bound GDN recurrence, and device-side gaps"
            ),
        },
    }
    text = json.dumps(record, indent=1, sort_keys=True) + "\n"
    if args.json is not None:
        args.json.write_text(text)
        print(f"wrote {args.json}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
