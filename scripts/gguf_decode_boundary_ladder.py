#!/usr/bin/env python3
"""R0-R4 one-token-boundary decode ladder (roadmap P5, C1).

Measures the marginal decode-token cost at each stack layer with the same
model, quant, KV policy, prompt token IDs, sampler, and eager (non-graph)
execution mode:

- R0: raw ``Qwen35GGUFResidentSession.prefill`` + ``step`` loop with the
  shipping low-level selectors (direct compute reference).
- R1: the same PRIVATE session class running ``prefill_batch_native`` +
  ``step`` - the packed slot-local ENTRY the server's prefill takes, but
  WITHOUT the server's pool binding, slot views, or scheduler. It isolates
  the executor-entry cost only; it is NOT the roadmap's R1 (server
  pool/scheduler execution), which the server allocation probe and
  engine-comparison harnesses approximate at the HTTP boundary.
- R3: the public ``LLM.generate`` token-ID path (EngineService, commands,
  collectors). Whole-request wall only: prefill and decode are not
  separated inside this arm, so decode-only figures must be derived
  against R0's step median with that caveat stated.

Protocol notes:
- One process, arms run sequentially, each with a fresh session/LLM.
- Per-token walls are taken around each ``step`` call (R0/R1) so the
  median/p95 isolate the token boundary; R3 reports the aggregate wall
  and tokens, plus the service's own per-token timing when exposed.
- Greedy sampling everywhere; deterministic fixture (rng 20260909, the
  parity manifest's generator) so every arm sees identical inputs.

Output: JSON artifact with per-arm per-token statistics and stage deltas.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    ap.add_argument("--prompt-rows", type=int, default=2048)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--max-sequence-length", type=int, default=16384)
    ap.add_argument("--vocab-span", type=int, default=32000)
    ap.add_argument(
        "--arms", default="r0_raw_session,r1_packed_entry,r3_public_llm"
    )
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    import numpy as np

    rng = np.random.default_rng(20260909)
    prompt = [
        int(t)
        for t in rng.integers(1000, args.vocab_span, size=args.prompt_rows)
    ]

    out: dict[str, object] = {
        "kind": "gguf_decode_boundary_ladder",
        "model": args.model,
        "prompt_rows": len(prompt),
        "prompt_kind": "deterministic_varied_rng20260909",
        "max_sequence_length": int(args.max_sequence_length),
        "decode_tokens": int(args.decode_tokens),
        "execution_mode": "eager (no graph capture)",
        "sampler": "greedy",
        "kv": "int8_per_token_head + fp32 scales",
        "arms": {},
    }

    def step_loop(session, first_token, count):
        walls = []
        nxt = first_token
        for _ in range(count - 1):
            t0 = time.perf_counter()
            step = session.step(nxt, return_logits=False)
            walls.append(time.perf_counter() - t0)
            nxt = int(step.token_id)
        return nxt, walls

    def percentiles(walls):
        if not walls:
            return {}
        ordered = sorted(walls)
        p95 = ordered[max(0, int(round(0.95 * len(ordered))) - 1)]
        return {
            "count": len(walls),
            "median_ms": round(statistics.median(ordered) * 1e3, 3),
            "p95_ms": round(p95 * 1e3, 3),
            "min_ms": round(ordered[0] * 1e3, 3),
            "max_ms": round(ordered[-1] * 1e3, 3),
        }

    selected = [a.strip() for a in str(args.arms).split(",") if a.strip()]

    if "r0_raw_session" in selected:
        from hipengine.core.hip import get_hip_runtime
        from hipengine.kvcache import resolve_kv_policy
        from hipengine.runtime.prefill import PrefillConfig
        from hipengine.runtime.qwen35_gguf_runner import (
            Qwen35GGUFResidentSession,
        )

        runtime = get_hip_runtime()
        policy = resolve_kv_policy("int8_per_token_head", scale_dtype="fp32")
        with Qwen35GGUFResidentSession(
            args.model,
            runtime=runtime,
            max_sequence_length=int(args.max_sequence_length),
            prefill_config=PrefillConfig(),
            kv_policy=policy.create_policy(),
            kv_scale_dtype="fp32",
            kv_scale_granularity=policy.scale_granularity,
            use_wmma_prefill=True,
            use_gemv_decode=True,
        ) as session:
            t0 = time.perf_counter()
            res = session.prefill(prompt, use_bulk=True, return_logits=False)
            prefill_wall = time.perf_counter() - t0
            nxt, walls = step_loop(session, int(res.token_id), int(args.decode_tokens))
            out["arms"]["r0_raw_session"] = {
                "prefill_wall_seconds": round(prefill_wall, 3),
                "prefill_tok_s": round(len(prompt) / prefill_wall, 2),
                "step_walls": percentiles(walls),
                "generated_head": [int(res.token_id)] + [nxt],
            }
            print(
                f"[r0_raw_session] prefill {len(prompt)/prefill_wall:.1f} tok/s, "
                f"step median {out['arms']['r0_raw_session']['step_walls']['median_ms']} ms",
                flush=True,
            )

    if "r1_packed_entry" in selected:
        from hipengine.core.hip import get_hip_runtime
        from hipengine.kvcache import resolve_kv_policy
        from hipengine.runtime.prefill import PrefillConfig
        from hipengine.runtime.qwen35_gguf_runner import (
            Qwen35GGUFResidentSession,
        )

        runtime = get_hip_runtime()
        policy = resolve_kv_policy("int8_per_token_head", scale_dtype="fp32")
        with Qwen35GGUFResidentSession(
            args.model,
            runtime=runtime,
            max_sequence_length=int(args.max_sequence_length),
            prefill_config=PrefillConfig(),
            kv_policy=policy.create_policy(),
            kv_scale_dtype="fp32",
            kv_scale_granularity=policy.scale_granularity,
            use_wmma_prefill=True,
            use_gemv_decode=True,
        ) as session:
            t0 = time.perf_counter()
            res = session.prefill_batch_native(
                [prompt], sessions=[session], return_logits=False
            )[0]
            prefill_wall = time.perf_counter() - t0
            nxt, walls = step_loop(session, int(res.token_id), int(args.decode_tokens))
            out["arms"]["r1_packed_entry"] = {
                "prefill_wall_seconds": round(prefill_wall, 3),
                "prefill_tok_s": round(len(prompt) / prefill_wall, 2),
                "executor_mode": (session.last_packed_prefill_plan or {}).get(
                    "executor_mode", "single_slab"
                ),
                "step_walls": percentiles(walls),
                "generated_head": [int(res.token_id)] + [nxt],
            }
            print(
                f"[r1_packed_entry] prefill {len(prompt)/prefill_wall:.1f} tok/s, "
                f"step median {out['arms']['r1_packed_entry']['step_walls']['median_ms']} ms",
                flush=True,
            )

    if "r3_public_llm" in selected:
        from hipengine.llm import LLM, SamplingParams

        llm = LLM(
            args.model,
            backend="hip_gfx1100",
            quant="gguf_q4_k_m",
            max_active_requests=1,
            max_sequence_length=int(args.max_sequence_length),
            kv_storage="int8_per_token_head",
            kv_scale_dtype="fp32",
            kv_scale_granularity="per_token_head",
            speculative_mtp_serving="off",
            prefix_cache="off",
        )
        # Grow the resident sessions to the serving context before the first
        # request (the server's startup path does the same).
        llm.prepare()
        t0 = time.perf_counter()
        texts = llm.generate(
            [prompt],
            SamplingParams(
                max_tokens=int(args.decode_tokens), temperature=0.0
            ),
        )
        wall = time.perf_counter() - t0
        out["arms"]["r3_public_llm"] = {
            "wall_seconds": round(wall, 3),
            "tokens_per_s_aggregate": round(
                int(args.decode_tokens) / wall, 3
            ),
            "generated_head": texts[0][:16],
        }
        print(
            f"[r3_public_llm] {args.decode_tokens} tokens in {wall:.2f} s "
            f"({args.decode_tokens/wall:.2f} tok/s aggregate incl. prefill)",
            flush=True,
        )

    r0 = out["arms"].get("r0_raw_session")
    r1 = out["arms"].get("r1_packed_entry")
    if r0 and r1:
        m0 = r0["step_walls"].get("median_ms")
        m1 = r1["step_walls"].get("median_ms")
        if m0 and m1:
            out["r1_over_r0_step_median_ratio"] = round(m1 / m0, 3)
            out["r1_minus_r0_step_median_ms"] = round(m1 - m0, 3)

    payload = json.dumps(out, indent=2, default=str)
    print(payload)
    if args.json:
        args.json.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
