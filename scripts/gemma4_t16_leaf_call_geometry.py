#!/usr/bin/env python3
"""Record the argument geometry of every T16 gate/up leaf call in a real prefill.

The counter measurement says the leaf reads 277.3 MB of DRAM per call, which is
below the 316.5 MB floor implied by "one pass over its own weights plus
activations". Before that observation is explained, the call's actual work has to
be known: how many compact rows, how many experts, and how many calls there
really are. This wraps the leaf launcher and records what it is handed.

Diagnostic only. No production module is edited.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--prompt", type=int, default=4096)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    from scripts.gemma4_campaign_bench import _resolve_generator, resolve_artifact

    model = args.model or str(resolve_artifact())
    llm, _runner, info = _resolve_generator(Path(model), 4096)
    print(f"load_s={info['load_s']:.1f} resolution={info['resolution']}")

    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts as ge

    original = ge._gemma4_project_experts_gate_up_wmma_t16
    calls: list[dict[str, int]] = []

    def wrapper(weight, x_ptr, out_ptr, expert_start, compact_rows, num_experts,
                in_features, intermediate, **kwargs):
        result = original(
            weight, x_ptr, out_ptr, expert_start, compact_rows, num_experts,
            in_features, intermediate, **kwargs,
        )
        if result:
            calls.append({
                "compact_rows": int(compact_rows),
                "num_experts": int(num_experts),
                "in_features": int(in_features),
                "intermediate": int(intermediate),
            })
        return result

    ge._gemma4_project_experts_gate_up_wmma_t16 = wrapper

    from hipengine.llm import SamplingParams

    tokens = [int(v) for v in range(1, args.prompt + 1)]
    for _ in range(2):
        calls.clear()
        llm.generate(tokens, SamplingParams(max_tokens=1, temperature=0.0))

    shapes = collections.Counter(
        (c["compact_rows"], c["num_experts"], c["in_features"], c["intermediate"])
        for c in calls
    )
    per_shape = collections.defaultdict(list)
    for c in calls:
        key = (c["compact_rows"], c["num_experts"], c["in_features"], c["intermediate"])
        per_shape[key].append(c)

    rows_total = sum(c["compact_rows"] for c in calls)
    out = {
        "kind": "gemma4_t16_leaf_call_geometry",
        "prompt": args.prompt,
        "model": model,
        "t16_calls": len(calls),
        "distinct_geometries": len(shapes),
        "geometries": [
            {
                "compact_rows": key[0],
                "num_experts": key[1],
                "in_features": key[2],
                "intermediate": key[3],
                "calls": count,
                "row_tiles_uniform_padding": (
                    ((key[0] + key[1] - 1) // key[1] + 15) // 16 * key[1]
                ),
            }
            for key, count in shapes.most_common()
        ],
        "compact_rows_total": rows_total,
        "compact_rows_per_call_mean": round(rows_total / len(calls), 1) if calls else 0.0,
        "note": (
            "row_tiles_uniform_padding is what the standalone probe fixture would "
            "use for the same (compact_rows, num_experts): rows spread evenly over "
            "experts, each expert padded up to a multiple of 16 rows. If the real "
            "per-call compact_rows is not num_experts x 32, the fixture's 4096 x 128 "
            "geometry does not describe the in-situ call."
        ),
    }
    print(json.dumps(out, indent=1))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
