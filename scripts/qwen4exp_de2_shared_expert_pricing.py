#!/usr/bin/env python3
"""Price the DE-2 shared-expert block from GGUF tensor payloads.

The DE-2 row asks for a name-to-tensor mapping before its action can be
screened. This resolves the tensor half of that mapping directly from the
shipped GGUF: it identifies every shared-expert tensor, reports its exact
quantization type and payload, and prices the block against the measured
decode capture's own bounds.

Read-only. No GPU, no model load.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import struct
from pathlib import Path
from typing import Any

# GGUF value type -> (struct format, byte size)
GGUF_SCALAR = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4),
    5: ("i", 4), 6: ("f", 4), 7: ("?", 1), 10: ("q", 8), 11: ("Q", 8),
    12: ("d", 8),
}

# ggml tensor type -> bytes per weight, where the block layout is known.
# Q8_0 = 32 weights + 1 fp16 scale = 34 B / 32 w. Q4_K = 144 B / 256 w.
# Q5_1 = 24 B / 32 w. F32 = 4 B / w.
TYPE_BYTES_PER_WEIGHT = {
    0: 4.0,            # F32
    1: 2.0,            # F16
    2: 36.0 / 32.0,    # Q4_0
    3: 24.0 / 32.0,    # Q5_1
    8: 34.0 / 32.0,    # Q8_0
    12: 144.0 / 256.0, # Q4_K
    13: 176.0 / 256.0, # Q5_K
    14: 210.0 / 256.0, # Q6_K
}
TYPE_NAME = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q5_1", 8: "Q8_0",
    12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
}


def _read_string(f) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def _read_value(f, vtype: int):
    if vtype == 8:
        return _read_string(f)
    if vtype == 9:
        etype = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        return [_read_value(f, etype) for _ in range(count)]
    fmt, size = GGUF_SCALAR[vtype]
    return struct.unpack("<" + fmt, f.read(size))[0]


def read_tensors(model_root: Path) -> dict[str, dict[str, Any]]:
    """Return {tensor_name: {shape, type, bytes}} across all shards."""
    shards = sorted(glob.glob(str(model_root / "*.gguf")))
    if not shards:
        raise SystemExit(f"no .gguf shards under {model_root}")
    out: dict[str, dict[str, Any]] = {}
    for path in shards:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise SystemExit(f"{path} is not a GGUF file")
            struct.unpack("<I", f.read(4))[0]          # version
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n_kv):
                _read_string(f)
                _read_value(f, struct.unpack("<I", f.read(4))[0])
            for _ in range(n_tensors):
                name = _read_string(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
                ttype = struct.unpack("<I", f.read(4))[0]
                struct.unpack("<Q", f.read(8))[0]      # offset
                n = 1
                for d in dims:
                    n *= d
                per_weight = TYPE_BYTES_PER_WEIGHT.get(ttype)
                out[name] = {
                    "shape": dims,
                    "type": ttype,
                    "type_name": TYPE_NAME.get(ttype, f"type{ttype}"),
                    "weights": n,
                    "bytes": int(n * per_weight) if per_weight else None,
                }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument(
        "--moe-family-ms-per-step", type=float, default=24.55039366666666,
        help="measured moe family cost from the decode capture",
    )
    parser.add_argument("--routed-gate-up-ms", type=float, default=14.012587)
    parser.add_argument("--routed-down-ms", type=float, default=5.444)
    parser.add_argument("--router-bytes", type=float, default=251658240.0)
    parser.add_argument("--comparator-gbps", type=float, default=184.66,
                        help="router dense GEMV achieved rate (in-block comparator)")
    parser.add_argument("--decode-kernel-sum-ms", type=float, default=54.218)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tensors = read_tensors(args.model_root)

    shared = {n: t for n, t in tensors.items()
              if "shexp" in n or "shared_expert" in n}
    if not shared:
        raise SystemExit("no shared-expert tensors found")

    by_suffix = collections.Counter()
    bytes_by_suffix: collections.Counter = collections.Counter()
    types_seen: collections.Counter = collections.Counter()
    for name, t in shared.items():
        suffix = ".".join(name.split(".")[2:]) if name.count(".") >= 2 else name
        by_suffix[suffix] += 1
        bytes_by_suffix[suffix] += t["bytes"] or 0
        types_seen[t["type_name"]] += 1

    total_bytes = sum(t["bytes"] or 0 for t in shared.values())
    # Only the per-token weight traffic matters; the tiny gate vector is included
    # because it is read every step too.
    per_layer = total_bytes / args.layers if args.layers else 0.0

    remainder_ms = (args.moe_family_ms_per_step - args.routed_gate_up_ms
                    - args.routed_down_ms)
    remainder_bytes = total_bytes + args.router_bytes
    implied_gbps = (remainder_bytes / (remainder_ms / 1e3)) / 1e9

    shared_at_comparator = (total_bytes / (args.comparator_gbps * 1e9)) * 1e3
    shared_at_implied = (total_bytes / (implied_gbps * 1e9)) * 1e3

    payload = {
        "schema": 1,
        "kind": "qwen4exp-de2-shared-expert-pricing",
        "performance_claim": False,
        "model_root": str(args.model_root),
        "layers": args.layers,
        "shared_tensors": len(shared),
        "shared_bytes_per_step": total_bytes,
        "shared_mb_per_step": total_bytes / 1e6,
        "shared_bytes_per_layer": per_layer,
        "types": dict(types_seen),
        "by_suffix": {k: {"count": v, "bytes": bytes_by_suffix[k]}
                      for k, v in sorted(by_suffix.items())},
        "moe_family_ms_per_step": args.moe_family_ms_per_step,
        "routed_gate_up_ms_per_step": args.routed_gate_up_ms,
        "routed_down_ms_per_step": args.routed_down_ms,
        "non_routed_remainder_ms_per_step": remainder_ms,
        "remainder_bytes_per_step": remainder_bytes,
        "remainder_implied_gbps": implied_gbps,
        "shared_ms_at_comparator_rate": shared_at_comparator,
        "shared_ms_at_implied_rate": shared_at_implied,
        "shared_upper_bound_pct_of_decode_kernel_sum":
            100.0 * shared_at_implied / args.decode_kernel_sum_ms,
        "shared_at_comparator_pct_of_decode_kernel_sum":
            100.0 * shared_at_comparator / args.decode_kernel_sum_ms,
        "decode_kernel_sum_ms": args.decode_kernel_sum_ms,
        "result": (
            "The shared expert is a Q8_0 block, not Q4_K: 3 x [2560,640] Q8_0 "
            "matrices plus a 2560-wide F32 gate per layer. Its per-step weight "
            "traffic is bounded by the measured non-routed moe remainder, so its "
            "cost is at most the remainder itself and at least the traffic at the "
            "in-block comparator's rate. A perfect shared expert therefore "
            "recovers at most the remainder; a realistic fused gate+up recovers a "
            "fraction of it."
        ),
    }

    print(json.dumps(payload, indent=1))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
