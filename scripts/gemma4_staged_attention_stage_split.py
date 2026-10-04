#!/usr/bin/env python3
"""Split the staged prefill attention into its score / softmax / P*V stages.

The Python-level kernel census cannot see inside this path: the staged launcher
issues all three phases from one C++ entry point, so the census prices it as a
single `attention[gemma4_staged]` row. The three phases are separate `__global__`
kernels with their own names, though, so a `rocprofv3` kernel trace prices them
without touching the kernel.

The rates are derived rather than assumed: the (query, key, head) triple count
comes from the model's own geometry (per-layer head counts, head dims, and the
sliding window, read from the GGUF metadata without loading weights), so the
achieved TFLOP/s and the read amplification are arithmetic on measurements.

Usage:
    .venv/bin/python scripts/gemma4_staged_attention_stage_split.py \
        --trace /tmp/gemma4-prof/pf32768_kernel_trace.csv \
        --artifact /models/gguf/.../gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
        --tokens 32768 --prefills 2 --out benchmarks/results/....json
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import time
from pathlib import Path

STAGES = (
    ("score", "gemma4_attention_staged_score_kernel"),
    ("softmax", "gemma4_attention_staged_softmax_kernel"),
    ("pv", "gemma4_attention_staged_pv_kernel"),
)


def geometry(artifact: Path) -> dict:
    from hipengine.loading.gguf import scan_gguf
    from hipengine.loading.gemma4_gguf import gemma4_gguf_config_from_metadata

    info = scan_gguf(str(artifact))
    config = gemma4_gguf_config_from_metadata(info)
    layers = []
    for index in range(config.block_count):
        layers.append(
            {
                "sliding": bool(config.is_sliding(index)),
                "heads": int(config.head_count(index)),
                "kv_heads": int(config.head_count_kv_for(index)),
                "head_dim": int(config.head_dim(index)),
            }
        )
    return {
        "block_count": int(config.block_count),
        "sliding_window": int(config.sliding_window),
        "layers": layers,
    }


def key_pairs(tokens: int, window: int | None) -> float:
    """(query, key) pairs for one layer: causal, and window-bounded when sliding."""

    if window is None:
        return tokens * (tokens + 1) / 2
    ramp = min(window, tokens)
    return ramp * (ramp + 1) / 2 + max(0, tokens - ramp) * window


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--prefills", type=int, default=2, help="full prefills inside the trace")
    ap.add_argument("--wall-s", type=float, default=0.0, help="reported prefill seconds, for reference")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.trace)))
    totals: collections.Counter[str] = collections.Counter()
    calls: collections.Counter[str] = collections.Counter()
    shapes: dict[str, dict] = {}
    for row in rows:
        name = row["Kernel_Name"]
        try:
            elapsed = float(row["End_Timestamp"]) - float(row["Start_Timestamp"])
        except (TypeError, ValueError):
            continue
        totals[name] += elapsed
        calls[name] += 1
        shapes.setdefault(
            name,
            {
                "workgroup": [row["Workgroup_Size_X"], row["Workgroup_Size_Y"], row["Workgroup_Size_Z"]],
                "grid": [row["Grid_Size_X"], row["Grid_Size_Y"], row["Grid_Size_Z"]],
                "vgpr": row["VGPR_Count"],
                "lds_bytes": row["LDS_Block_Size"],
            },
        )

    total_s = sum(totals.values()) / 1e9
    stages = []
    for label, needle in STAGES:
        names = [n for n in totals if needle in n]
        seconds = sum(totals[n] for n in names) / 1e9
        stages.append(
            {
                "stage": label,
                "seconds": round(seconds, 3),
                "seconds_per_prefill": round(seconds / args.prefills, 4),
                "launches": sum(calls[n] for n in names),
                "variants": sorted(n.split("::")[-1] for n in names),
                "launch_shape": shapes.get(names[0]) if names else None,
            }
        )
    attention_s = sum(s["seconds"] for s in stages)

    geo = geometry(Path(args.artifact))
    tokens = int(args.tokens)
    pairs = 0.0
    macs = 0.0
    qk_bytes = 0.0
    for layer in geo["layers"]:
        window = geo["sliding_window"] if layer["sliding"] else None
        layer_pairs = key_pairs(tokens, window) * layer["heads"]
        pairs += layer_pairs
        macs += layer_pairs * layer["head_dim"]
        # One Q row and one K row read per (query, key, head) triple when nothing
        # is staged: this is the amplification the LDS figure is about.
        qk_bytes += layer_pairs * layer["head_dim"] * 2 * 2

    flops = 2.0 * macs
    score_s = next(s["seconds_per_prefill"] for s in stages if s["stage"] == "score")
    pv_s = next(s["seconds_per_prefill"] for s in stages if s["stage"] == "pv")
    qk_flops = 2.0 * macs
    attention_s_per_prefill = attention_s / args.prefills
    payload = {
        "kind": "gemma4_staged_attention_stage_split",
        "performance_claim": False,
        "created": time.strftime("%Y-%m-%d"),
        "command": " ".join(sys.argv),
        "trace": str(args.trace),
        "model": "Gemma 4 26B-A4B-it",
        "quant": "UD-Q4_K_XL",
        "physical_host": "zbook",
        "hardware": "AMD Radeon 8060S Graphics (gfx1151)",
        "kv_dtype": "bf16",
        "prompt_tokens": tokens,
        "prefills_in_trace": args.prefills,
        "geometry": {
            "block_count": geo["block_count"],
            "sliding_window": geo["sliding_window"],
            "full_layers": sum(1 for x in geo["layers"] if not x["sliding"]),
            "sliding_layers": sum(1 for x in geo["layers"] if x["sliding"]),
        },
        "kernel_time_s": round(total_s, 3),
        "reported_prefill_s": args.wall_s,
        "attention_stages": stages,
        "attention_share_of_kernel_time": round(attention_s / total_s, 4) if total_s else None,
        "derived": {
            "query_key_head_triples_per_prefill": round(pairs),
            "qk_flops_per_prefill": round(qk_flops),
            "pv_flops_per_prefill": round(qk_flops),
            "attention_flops_per_prefill": round(2.0 * qk_flops),
            "attention_tflops": round(2.0 * qk_flops / attention_s_per_prefill / 1e12, 4),
            "score_tflops": round(qk_flops / score_s / 1e12, 4),
            "pv_tflops": round(qk_flops / pv_s / 1e12, 4),
            "score_read_bytes_per_prefill_if_unstaged": round(qk_bytes),
            "score_effective_gbps_if_unstaged": round(qk_bytes / score_s / 1e9, 1),
            "note": (
                "QK and P*V each cost one multiply-accumulate per (query, key, head) "
                "element of the head dimension, so both are reported at the same "
                "FLOP count. The effective read rate is what the kernel would move "
                "if it staged nothing; it exceeds the part's DRAM roofline, so a "
                "large part of it is served from cache. The point is the "
                "amplification, not the rate."
            ),
        },
        "top_kernels": [
            {"name": n, "seconds": round(totals[n] / 1e9, 3), "launches": calls[n]}
            for n in sorted(totals, key=lambda n: -totals[n])[:12]
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {out}")
    for s in stages:
        print(f"  {s['stage']:8} {s['seconds_per_prefill']:9.4f} s/prefill  {s['seconds']:8.3f}s total")
    print(f"  attention {attention_s/args.prefills:.3f} s/prefill  ({attention_s/total_s*100:.1f}% of kernel time)")
    print(f"  score {payload['derived']['score_tflops']:.3f} TFLOP/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
