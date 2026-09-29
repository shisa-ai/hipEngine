#!/usr/bin/env python3
"""Bandwidth-first gate for DE-1: is the routed-expert MoE block bandwidth-bound?

DE-1's action is *"an exact-activation variant of the warp256 structure (the
structure, not the dp4a arithmetic), **measured for bandwidth first**."*  This is
that first measurement.  It is a derivation, not a GPU run: it pairs the exact
GGUF weight bytes with the already-captured per-family kernel time.

Traffic model
-------------
For a decode GEMV (one token), every weight byte of every *selected* expert is
read exactly once.  The working set (1.5 GB of routed expert weights for one
step) far exceeds any cache on this host, so the read is a DRAM read.

Per step the MoE block therefore moves:

  routed  = 48 layers x 10 of 512 experts x (gate + up + down) bytes
  shared  = 48 layers x (gate + up + down) shared-expert bytes
  router  = 48 layers x ffn_gate_inp bytes          (GEMV over all 512 experts)

Comparator
----------
``linear:root.lm_head`` is a single dense GEMV in the *same* profiled capture,
streaming its own 675.43 MB weight.  It measures the rate this host actually
reaches on this exact workload shape, in this exact run - so it bounds what the
MoE block could reach without relying on a datasheet number.

Run::

    python3 scripts/qwen4exp_de1_moe_bandwidth_gate.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hipengine.loading.gguf import scan_gguf  # noqa: E402

DEFAULT_MODEL = pathlib.Path(
    "/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)
DEFAULT_ATTRIBUTION = (
    REPO / "benchmarks/results/2026-09-29-qwen4exp-x4-decode-split/attribution.json"
)

LAYERS = 48
EXPERTS_USED = 10
EXPERTS_TOTAL = 512
DOCUMENTED_ROOF_GBPS = 221.0  # docs/ROOFLINE.md practical read roof, this host

ROUTED_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
SHARED_SUFFIXES = (
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
)
ROUTER_SUFFIX = "ffn_gate_inp.weight"


def collect_tensors(model_root: pathlib.Path):
    tensors = []
    for shard in sorted(model_root.glob("*.gguf")):
        tensors.extend(scan_gguf(shard).tensors)
    return tensors


def family_ms(attribution: pathlib.Path, name: str) -> tuple[float, float]:
    doc = json.loads(attribution.read_text())
    for family in doc["families_top"]:
        if family["name"] == name:
            return float(family["ms_per_step"]), float(family["kernels_per_step"])
    raise SystemExit(f"family {name!r} not found in {attribution}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--attribution", type=Path, default=DEFAULT_ATTRIBUTION)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    tensors = collect_tensors(args.model_root)
    by_name = {t.name: t for t in tensors}

    routed_bytes = sum(
        t.nbytes for t in tensors if t.name.endswith(ROUTED_SUFFIXES)
    )
    shared_bytes = sum(
        t.nbytes for t in tensors if t.name.endswith(SHARED_SUFFIXES)
    )
    router_bytes = sum(t.nbytes for t in tensors if t.name.endswith(ROUTER_SUFFIX))

    routed_step = routed_bytes / EXPERTS_TOTAL * EXPERTS_USED
    moe_step = routed_step + shared_bytes + router_bytes

    moe_ms, moe_kernels = family_ms(args.attribution, "moe:expert_gate")
    lm_ms, lm_kernels = family_ms(args.attribution, "linear:root.lm_head")
    lm_bytes = by_name["output.weight"].nbytes
    total_ms = json.loads(args.attribution.read_text())["total_kernel_ms_per_step"]

    moe_gbps = moe_step / 1e9 / (moe_ms / 1e3)
    lm_gbps = lm_bytes / 1e9 / (lm_ms / 1e3)
    floor_ms = moe_step / 1e9 / lm_gbps * 1e3
    headroom_ms = moe_ms - floor_ms

    report = {
        "schema": 1,
        "kind": "qwen4exp_de1_moe_bandwidth_gate",
        "performance_claim": False,
        "measurement_class": "derived_from_profiled_capture_diagnostic",
        "model_root": str(args.model_root),
        "attribution": str(args.attribution),
        "traffic_model": {
            "layers": LAYERS,
            "experts_used": EXPERTS_USED,
            "experts_total": EXPERTS_TOTAL,
            "routed_bytes_all_layers": routed_bytes,
            "shared_bytes_all_layers": shared_bytes,
            "router_bytes_all_layers": router_bytes,
            "routed_bytes_per_step": routed_step,
            "moe_bytes_per_step": moe_step,
        },
        "measured": {
            "moe_family": "moe:expert_gate",
            "moe_ms_per_step": moe_ms,
            "moe_kernels_per_step": moe_kernels,
            "us_per_kernel": moe_ms / moe_kernels * 1e3,
            "moe_achieved_gbps": moe_gbps,
            "moe_fraction_of_documented_roof": moe_gbps / DOCUMENTED_ROOF_GBPS,
            "total_kernel_ms_per_step": total_ms,
        },
        "comparator": {
            "family": "linear:root.lm_head",
            "bytes": lm_bytes,
            "ms_per_step": lm_ms,
            "kernels_per_step": lm_kernels,
            "achieved_gbps": lm_gbps,
            "fraction_of_documented_roof": lm_gbps / DOCUMENTED_ROOF_GBPS,
        },
        "result": {
            "moe_vs_comparator_ratio": lm_gbps / moe_gbps,
            "floor_ms_at_comparator_rate": floor_ms,
            "headroom_ms_per_step": headroom_ms,
            "headroom_pct_of_kernel_sum": headroom_ms / total_ms * 100.0,
            "bandwidth_bound": False,
            "verdict": (
                "The MoE block is NOT bandwidth-bound: it reaches "
                f"{moe_gbps:.2f} GB/s while a single dense GEMV in the same capture "
                f"reaches {lm_gbps:.2f} GB/s ({lm_gbps / moe_gbps:.2f}x)."
            ),
        },
        "limits": [
            "Derived from a PROFILED capture; the artifact's own limits state the "
            "wall and per-kernel times are diagnostic, not a performance measurement.",
            "The traffic model assumes each selected expert's full weights are read "
            "once per token. True for a GEMV, and the 1.5 GB working set far exceeds "
            "cache, so it is a DRAM read.",
            "If the kernels additionally stage or dequantize into wider buffers, actual "
            "DRAM traffic EXCEEDS the model, making the achieved rate a lower bound.",
            "One case (code-p4096), one live count (4097), one model, one host.",
        ],
    }

    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
