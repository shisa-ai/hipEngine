#!/usr/bin/env python3
"""Why did the DE-3 gate report zero KL?  Probe the route, do not assume it.

The first DE-3 KL-envelope run returned ``kl_mean = 0.0`` with
``strict_logits_sha256 == candidate_logits_sha256`` on all 18 prompts, i.e. the
two arms were bit-identical and the candidate route never engaged. A green
quality status in that situation is vacuous, so the gate result was discarded.

This probe answers the only question that matters before spending another gate
run: does the candidate route fire at all in the model the gate uses?  It
answers it two ways, independently:

* it wraps ``_pack8_gemv_decode_threads`` and counts every decision, recording
  the ``(rows, out_features, variant)`` triples that reached it and how many of
  them resolved to the wider block; and
* it runs the same decode with ``HIPENGINE_KERNEL_CENSUS`` on and reports which
  quant/symbol/shape triples actually launched.

If the counter shows no eligible shape, the lever is simply not exercised by
this model and a KL gate on it is meaningless.  If the counter shows eligible
shapes but the census shows no ``pack8_gemv_*`` launch at those shapes, the
predicate is running but the dispatch never selects that family, which is a
different bug.

Usage:
    python3 scripts/qwen4exp_de3_route_probe.py [--decode-steps 8]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

DEFAULT_MODEL = pathlib.Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--decode-steps", type=int, default=8)
    ap.add_argument("--prompt", default="Write a short function that reverses a list.")
    ap.add_argument("--json-out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    os.environ["HIPENGINE_KERNEL_CENSUS"] = "1"
    os.environ["HIPENGINE_GGUF_PACK8_GEMV_WIDE"] = "1"

    from hipengine.kernels import launch_census
    from hipengine.runtime import gguf_linear

    launch_census.reset()

    decisions: collections.Counter = collections.Counter()
    eligible: collections.Counter = collections.Counter()
    real = gguf_linear._pack8_gemv_decode_threads

    def counting(**kwargs):
        value = real(**kwargs)
        decisions[(kwargs.get("rows"), kwargs.get("out_features"), kwargs.get("variant"), value)] += 1
        if kwargs.get("rows") == 1:
            eligible[(kwargs.get("out_features"), kwargs.get("variant"))] += 1
        return value

    gguf_linear._pack8_gemv_decode_threads = counting

    from hipengine.loading.gguf import scan_gguf
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
    from scripts.gguf_mtp_bench import build_chat_prompt

    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
    tokens = build_chat_prompt(tokenizer, args.prompt)
    print(f"prompt tokens: {len(tokens)}", flush=True)

    with Qwen35GGUFResidentSession(
        args.model,
        backend=str(args.backend),
        max_sequence_length=len(tokens) + args.decode_steps + 8,
        prefill_config=PrefillConfig(attn_aotriton_min_tokens=0),
    ) as session:
        if session.runner is None:
            print("session closed during setup")
            return 2
        print(f"backend: {session.runner.backend}", flush=True)
        session.reset()
        current = session.prefill(
            [int(t) for t in tokens],
            use_bulk=True,
            bulk_attention_mode="bulk",
            return_logits=True,
            capture_hidden_seed_fp32=False,
        )
        current = int(current.token_id)
        for _ in range(int(args.decode_steps)):
            result = session.step(current, return_logits=True)
            current = int(result.token_id)

    census = launch_census.snapshot()
    rows = census.get("rows", census) if isinstance(census, dict) else census

    print("\n=== predicate decisions (rows, out, variant, value) ===")
    if not decisions:
        print("  NEVER CALLED")
    for (r, o, v, val), n in decisions.most_common(20):
        print(f"  rows={r} out={o:>7} value={val:>4} x{n:<6} {v}")

    print("\n=== eligible rows==1 shapes seen by the predicate ===")
    if not eligible:
        print("  NONE")
    for (o, v), n in eligible.most_common(20):
        print(f"  out={o:>7} x{n:<6} {v}")

    wide = {k: n for k, n in decisions.items() if k[3]}
    print(f"\npredicate returned a wider block {sum(wide.values())} times")

    print("\n=== census: pack8_gemv launches ===")
    n_pack8 = 0
    for row in rows if isinstance(rows, list) else []:
        sym = str(row.get("symbol", ""))
        if "pack8_gemv" in sym or "pack8_gemv" in str(row.get("quant", "")):
            n_pack8 += 1
            print(f"  {json.dumps(row)[:220]}")
    if n_pack8 == 0:
        print("  NO pack8_gemv launches recorded")
        print(f"  (census total rows: {len(rows) if isinstance(rows, list) else 'n/a'})")
        for row in (rows if isinstance(rows, list) else [])[:15]:
            print(f"    {json.dumps(row)[:200]}")

    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                {
                    "decisions": {str(k): v for k, v in decisions.items()},
                    "eligible": {str(k): v for k, v in eligible.items()},
                    "census": census,
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
