#!/usr/bin/env python3
"""Count which raw GGUF K-quant GEMV symbols the Gemma 4 decode path launches.

``launch_gguf_linear`` applies ``_pack8_decode_dispatch`` before resolving the
kernel, so the *contract* variant (``gemv_bf16_bf16_out``) is not necessarily
the kernel that runs. This probe counts the actual launches by wrapping the one
choke point every Q8_0/Q5_K/Q6_K GEMV goes through.

Usage:
    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0 PYTHONPATH=. \
        python3 scripts/gemma4_gemv_route_probe.py --prompt 8 --output 4
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

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--prompt", type=int, default=8)
    ap.add_argument("--output", type=int, default=4)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as gk

    counts: collections.Counter = collections.Counter()
    shapes: collections.Counter = collections.Counter()
    orig_launch = gk._launch

    def spy_launch(quant, symbol, *a, **kw):
        counts[(str(quant), str(symbol))] += 1
        if len(a) >= 5:
            shapes[(str(symbol), int(a[3]), int(a[4]))] += 1
        return orig_launch(quant, symbol, *a, **kw)

    gk._launch = spy_launch

    from scripts.gemma4_campaign_bench import _resolve_generator, resolve_artifact

    model = args.model or str(resolve_artifact())
    llm, runner, _ = _resolve_generator(Path(model), 4096)

    # Prefill once, then count only the decode forwards.
    from hipengine.llm import SamplingParams

    prompt_ids = list(range(1000, 1000 + args.prompt))
    llm.generate_detailed(prompt_ids, SamplingParams(max_tokens=1, temperature=0.0))
    counts.clear()
    shapes.clear()
    output = llm.generate_detailed(
        prompt_ids, SamplingParams(max_tokens=args.output, temperature=0.0)
    )[0]
    tokens = len(output.generated_token_ids or ())
    print(f"decode forwards counted over {tokens} generated tokens")
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts as _ex

    print("--- expert route counts ---")
    route_counts = _ex.gemma4_moe_expert_route_counts()
    for k, v in sorted(route_counts.items(), key=lambda kv: -kv[1]):
        print(f"{v:8d}  {k}")
    print("--- grouped variant counts ---")
    grouped_counts = _ex.gemma4_moe_grouped_variant_counts()
    for k, v in sorted(grouped_counts.items(), key=lambda kv: -kv[1]):
        print(f"{v:8d}  {k}")

    rows = sorted(
        (
            {"quant": q, "symbol": s, "launches": n}
            for (q, s), n in counts.items()
        ),
        key=lambda r: -r["launches"],
    )
    for r in rows:
        print(f"{r['launches']:8d}  {r['quant']:10s} {r['symbol']}")
    print("--- by shape ---")
    for (sym, in_f, out_f), n in sorted(shapes.items(), key=lambda kv: -kv[1]):
        print(f"{n:8d}  {sym:50s} in={in_f:6d} out={out_f:6d}")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "rows": rows,
                    "shapes": [
                        {"symbol": s, "in_features": i, "out_features": o,
                         "launches": n}
                        for (s, i, o), n in sorted(shapes.items(), key=lambda kv: -kv[1])
                    ],
                    "prompt": args.prompt,
                    "output": args.output,
                    "generated_tokens": tokens,
                    "expert_route_counts": route_counts,
                    "grouped_variant_counts": grouped_counts,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
