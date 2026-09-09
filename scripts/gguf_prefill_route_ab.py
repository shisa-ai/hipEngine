#!/usr/bin/env python3
"""Speed + agreement A/B of the two GGUF prefill entry points.

Arm ``scalar_bulk``       -- ``session.prefill(use_bulk=True)``, the c1 parent.
Arm ``packed_slot_local`` -- ``session.prefill_batch_native([ids])``, the route
                             every server request takes (``int8_direct`` forces
                             it unconditionally, see
                             ``_gguf_single_row_block_table_prefill_required``).

Both arms run in one process on fresh identical sessions with a deterministic
varied prompt, and report prefill tok/s, tracked peak, greedy continuation IDs
and final-prefill logit agreement.

The two arms are NOT expected to agree with each other on the ``int8_direct``
route: they use different GDN state-capture arithmetic by design
(``hipengine/generation/qwen35_gguf.py`` ``_prefill_native_row``). Use this to
compare one arm across a code or flag change (before vs after), which is the
binding oracle, and to keep the other arm as an untouched control.

Example - gate the slot-local AOTriton admission flag:

    for mode in off on; do
      [ $mode = on ] && export HIPENGINE_GGUF_INT8_PREFILL_SLOT_LOCAL_AOTRITON=1 \
                     || unset HIPENGINE_GGUF_INT8_PREFILL_SLOT_LOCAL_AOTRITON
      HIP_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=1 \
      HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG=1 \
      HIPENGINE_GGUF_INT8_KV_BF16_FULL_LAYERS=none \
      python scripts/gguf_prefill_route_ab.py --prompt-length 8192 \
        --json /tmp/route-ab-$mode.json
    done
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    ap.add_argument("--prompt-length", type=int, default=4096)
    ap.add_argument("--max-sequence-length", type=int, default=16384)
    ap.add_argument("--decode-tokens", type=int, default=8)
    ap.add_argument("--vocab-span", type=int, default=32000)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    import numpy as np
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.kvcache import resolve_kv_policy
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    rng = np.random.default_rng(20260909)
    prompt = [int(t) for t in rng.integers(1000, args.vocab_span, size=args.prompt_length)]
    runtime = get_hip_runtime()
    policy = resolve_kv_policy("int8_per_token_head", scale_dtype="fp32")

    out = {
        "kind": "prefill_route_parity",
        "model": args.model,
        "prompt_length": len(prompt),
        "prompt_kind": "deterministic_varied_rng20260909",
        "max_sequence_length": int(args.max_sequence_length),
        "decode_tokens": int(args.decode_tokens),
        "env": {k: os.environ.get(k) for k in (
            "HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG",
            "HIPENGINE_GGUF_INT8_KV_BF16_FULL_LAYERS",
        )},
        "arms": {},
    }

    def run(arm: str):
        with Qwen35GGUFResidentSession(
            args.model, runtime=runtime,
            max_sequence_length=int(args.max_sequence_length),
            prefill_config=PrefillConfig(),
            kv_policy=policy.create_policy(),
            kv_scale_dtype="fp32",
            kv_scale_granularity=str(policy.scale_granularity),
        ) as session:
            t0 = time.perf_counter()
            if arm == "scalar_bulk":
                res = session.prefill(prompt, use_bulk=True, return_logits=True)
            else:
                res = session.prefill_batch_native(
                    [prompt], sessions=[session],
                    full_prompt_lengths=[len(prompt)], return_logits=True,
                )[0]
            elapsed = time.perf_counter() - t0
            logits = np.asarray(res.logits, dtype=np.float32).reshape(-1)
            ids = [int(res.token_id)]
            nxt = ids[0]
            for _ in range(int(args.decode_tokens) - 1):
                step = session.step(nxt, return_logits=False)
                nxt = int(step.token_id)
                ids.append(nxt)
            return {
                "kv_attention_source": getattr(session, "kv_attention_source", None),
                "lifetime_mode": getattr(
                    getattr(session, "_int8_prefill_lifetime_plan", None), "mode", None),
                "prefill_wall_seconds": round(elapsed, 3),
                "prefill_tok_s": round(len(prompt) / elapsed, 2),
                "generated_ids": ids,
                "logits": logits,
                "tracked_peak_gib": round(
                    int(memory_stats().get("peak_allocated_bytes", 0)) / 2**30, 4),
            }

    for arm in ("scalar_bulk", "packed_slot_local"):
        r = run(arm)
        logits = r.pop("logits")
        r["logits_sha_head"] = float(logits[:1][0])
        out["arms"][arm] = r
        out.setdefault("_logits", {})[arm] = logits
        print(f"[{arm}] {r['prefill_wall_seconds']} s {r['prefill_tok_s']} tok/s "
              f"ids={r['generated_ids'][:4]}", flush=True)

    la = out["_logits"]["scalar_bulk"]
    lb = out["_logits"]["packed_slot_local"]
    import numpy as np
    n = min(la.size, lb.size)
    diff = np.abs(la[:n] - lb[:n])
    out["parity"] = {
        "logit_max_abs_diff": float(diff.max()),
        "logit_mean_abs_diff": float(diff.mean()),
        "bitwise_identical_logits": bool(np.array_equal(la[:n], lb[:n])),
        "top1_match": int(np.argmax(la[:n])) == int(np.argmax(lb[:n])),
        "generated_ids_match": (
            out["arms"]["scalar_bulk"]["generated_ids"]
            == out["arms"]["packed_slot_local"]["generated_ids"]
        ),
    }
    out.pop("_logits")
    a = out["arms"]["scalar_bulk"]["prefill_tok_s"]
    b = out["arms"]["packed_slot_local"]["prefill_tok_s"]
    out["speed_ratio_scalar_over_packed"] = round(a / b, 3) if b else None
    payload = json.dumps(out, indent=2, default=str)
    print(payload)
    if args.json:
        args.json.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
