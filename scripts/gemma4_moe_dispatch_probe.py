#!/usr/bin/env python3
"""Record which kernels the Gemma 4 MoE expert projections actually run.

The campaign's MoE numbers have been carried forward from microbenches whose
weights were synthetic substitutes. This probe answers the prior question
directly: on a real ``LLM.generate()``, what quant key, shape, expert count and
resolved kernel does each expert projection dispatch to at decode?

The expert path resolves through ``registry.resolve`` with the selected variant
rather than through ``resolve_gguf_linear_dispatch``, so patching the GGUF linear
dispatch alone reports the dense projections and no experts at all. This patches
the expert entry points themselves, which carry the shape, the expert count and
the row count.

Usage:
    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=1 PYTHONPATH=. \
        .venv/bin/python scripts/gemma4_moe_dispatch_probe.py \
        --prompt 8 --output 4
"""

from __future__ import annotations

import argparse
import collections
import json
import sys


def main() -> int:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.gemma4_campaign_bench import resolve_artifact

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(resolve_artifact()))
    ap.add_argument("--prompt", type=int, default=8)
    ap.add_argument("--output", type=int, default=4)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts as ex

    seen: dict[tuple, int] = collections.Counter()
    results: dict[tuple, list] = {}

    orig_selected = ex.gemma4_project_experts_selected
    orig_one = ex.gemma4_project_expert

    def spy_selected(weight, x_ptr, selected_ptr, out_ptr, x_rows, rows, num_experts,
                     in_features, out_features, **kw):
        out = orig_selected(weight, x_ptr, selected_ptr, out_ptr, x_rows, rows,
                            num_experts, in_features, out_features, **kw)
        try:
            quant = weight.spec.quant_key if not isinstance(weight, int) else "<bf16 ptr>"
        except Exception:
            quant = "<unreadable>"
        key = ("selected", quant, int(in_features), int(out_features),
               int(num_experts), int(rows), bool(out))
        seen[key] += 1
        results[key] = [quant]
        return out

    def spy_one(weight, expert, x_ptr, out_ptr, rows, in_features, out_features, **kw):
        out = orig_one(weight, expert, x_ptr, out_ptr, rows, in_features,
                       out_features, **kw)
        try:
            quant = weight.spec.quant_key if not isinstance(weight, int) else "<bf16 ptr>"
        except Exception:
            quant = "<unreadable>"
        key = ("per-expert", quant, int(in_features), int(out_features), -1,
               int(rows), True)
        seen[key] += 1
        results[key] = [quant]
        return out

    ex.gemma4_project_experts_selected = spy_selected
    ex.gemma4_project_expert = spy_one
    # The runtime imports these by name in places, so patch there too.
    try:
        from hipengine.runtime import gemma4 as rt
        if hasattr(rt, "gemma4_project_experts_selected"):
            rt.gemma4_project_experts_selected = spy_selected
        if hasattr(rt, "gemma4_project_expert"):
            rt.gemma4_project_expert = spy_one
    except Exception:
        pass

    import hipengine
    from hipengine.llm import SamplingParams

    llm = hipengine.LLM(args.model)
    text = llm.generate(
        "The capital of France is",
        SamplingParams(max_tokens=args.output),
    )
    print(f"generated: {text!r}", file=sys.stderr)

    rows = [
        {"path": k[0], "quant_key": k[1], "in_features": k[2], "out_features": k[3],
         "num_experts": k[4], "rows": k[5], "selected_served": k[6], "launches": n}
        for k, n in sorted(seen.items(), key=lambda kv: -kv[1])
    ]
    print(f"\n{'path':11s} {'quant_key':14s} {'in':>5s} {'out':>5s} {'exp':>4s} "
          f"{'rows':>4s} {'n':>6s}  served")
    for r in rows:
        print(f"{r['path']:11s} {r['quant_key']:14s} {r['in_features']:5d} "
              f"{r['out_features']:5d} {r['num_experts']:4d} {r['rows']:4d} "
              f"{r['launches']:6d}  {r['selected_served']}")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
