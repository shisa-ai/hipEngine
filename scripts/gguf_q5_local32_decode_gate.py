"""Teacher-forced decode gate for the Q5 T16 local32 decode owner.

The Q5 route lives in GGUF_T16_C1_VARIANTS_BY_QUANT_SHAPE (the rows==1
shape-keyed variant override), not the IQ decode policy, so the
IQ-policy probe cannot gate it. This probe swaps the C1 table instead:
the candidate is the shipped table, the incumbent is the shipped table
minus the Q5 entries. Natural prompts (self-generated under the
incumbent) AND the 18 tokenized category fixtures are both teacher-
forced onto the incumbent trajectory, with the campaign predicate on
the pooled per-position KL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--compiler-version-file", type=Path, default=None)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file)

    import hipengine.kernels.hip_gfx1100 as be
    from hipengine.benchmark.correctness import evaluate_logits
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
    from hipengine.loading.gguf import scan_gguf
    from scripts.gguf_mtp_bench import build_chat_prompt

    compiler_version = (
        Path(args.compiler_version_file).read_text()
        if args.compiler_version_file else None)

    shipped = {
        q: dict(entries)
        for q, entries in be.GGUF_T16_C1_VARIANTS_BY_QUANT_SHAPE.items()
    }
    incumbent = {
        q: {s: v for s, v in entries.items() if q != "gguf_q5_k_t16_v1"}
        for q, entries in shipped.items()
    }

    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
    rng = np.random.default_rng(args.seed)
    n = int(args.decode_tokens)

    prompt_rows = []
    for path in (
            REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl",
            REPO_ROOT / "benchmarks/prompts/gdn-prefill-category-heldouts.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                prompt_rows.append(json.loads(line))

    with Qwen35GGUFResidentSession(
            args.model,
            compiler_version=compiler_version,
            max_sequence_length=args.prompt_tokens + n + 64,
            use_wmma_prefill=True,
            use_gemv_decode=True,
    ) as session:
        def _arm(c1_table, prompt_ids, forced=None):
            be.GGUF_T16_C1_VARIANTS_BY_QUANT_SHAPE = c1_table
            logits_rows = []
            cur = session.prefill(
                prompt_ids, use_bulk=True, bulk_attention_mode="bulk",
                return_logits=True)
            logits_rows.append(np.asarray(
                cur.logits, dtype=np.float32).reshape(-1))
            for i in range(n):
                feed = (int(cur.token_id) if forced is None
                        else int(forced[i]))
                cur = session.step(feed, return_logits=True)
                logits_rows.append(np.asarray(
                    cur.logits, dtype=np.float32).reshape(-1))
            return np.vstack(logits_rows)

        def _run_pair(ids):
            ref = _arm(incumbent, ids)
            ref_tokens = [int(x) for x in np.argmax(ref, -1)]
            session.reset()
            cand = _arm(shipped, ids, forced=ref_tokens[:-1])
            session.reset()
            return ref, cand

        # natural half: self-generated under the incumbent
        seed_ids = [int(t) for t in rng.integers(1000, 50000, size=8)]
        be.GGUF_T16_C1_VARIANTS_BY_QUANT_SHAPE = incumbent
        cur = session.prefill(seed_ids, use_bulk=True,
                              bulk_attention_mode="bulk",
                              return_logits=True)
        ids = list(seed_ids)
        while len(ids) < args.prompt_tokens:
            t = int(cur.token_id)
            ids.append(t)
            cur = session.step(t, return_logits=True)
        session.reset()
        ref_n, cand_n = _run_pair(ids)

        # category half: tokenized fixture prompts, incumbent-extended
        cat_refs, cat_cands = [], []
        for row in prompt_rows:
            content = row["messages"][0]["content"]
            base = build_chat_prompt(tokenizer, content)
            be.GGUF_T16_C1_VARIANTS_BY_QUANT_SHAPE = incumbent
            ids = list(base)
            cur = session.prefill(ids, use_bulk=True,
                                  bulk_attention_mode="bulk",
                                  return_logits=True)
            while len(ids) < args.prompt_tokens:
                t = int(cur.token_id)
                ids.append(t)
                cur = session.step(t, return_logits=True)
            session.reset()
            ref, cand = _run_pair(ids)
            cat_refs.append(ref)
            cat_cands.append(cand)

    be.GGUF_T16_C1_VARIANTS_BY_QUANT_SHAPE = shipped

    def _kl(ref_m, cand_m):
        ref_lse = ref_m - ref_m.max(-1, keepdims=True)
        ref_p = np.exp(ref_lse)
        ref_p /= ref_p.sum(-1, keepdims=True)
        cand_lse = cand_m - cand_m.max(-1, keepdims=True)
        cand_p = np.exp(cand_lse)
        cand_p /= cand_p.sum(-1, keepdims=True)
        return np.sum(ref_p * (np.log(ref_p + 1e-12)
                               - np.log(cand_p + 1e-12)), axis=-1)

    nat_dec_kl = _kl(ref_n[1:], cand_n[1:])
    cat_dec_kl = _kl(np.vstack(cat_refs)[1:], np.vstack(cat_cands)[1:])
    # pool: natural decode positions + category decode positions
    pooled = np.concatenate([nat_dec_kl, cat_dec_kl])
    top1 = float(np.mean(
        np.argmax(np.vstack(cat_refs + [ref_n])[1:], -1)
        == np.argmax(np.vstack(cat_cands + [cand_n])[1:], -1)))
    finite = bool(np.isfinite(cand_n).all()
                  and np.isfinite(np.vstack(cat_cands)).all())
    pre = _kl(ref_n[:1], cand_n[:1])
    metrics = {
        "model": str(args.model), "seed": args.seed,
        "prompt_tokens": args.prompt_tokens,
        "natural": {
            "positions": int(nat_dec_kl.size),
            "kl_mean": float(nat_dec_kl.mean()),
            "kl_max": float(nat_dec_kl.max()),
            "top1": float(np.mean(np.argmax(ref_n[1:], -1)
                                  == np.argmax(cand_n[1:], -1))),
        },
        "category": {
            "positions": int(cat_dec_kl.size),
            "kl_mean": float(cat_dec_kl.mean()),
            "kl_p95": float(np.percentile(cat_dec_kl, 95)),
            "kl_p99": float(np.percentile(cat_dec_kl, 99)),
            "kl_max": float(cat_dec_kl.max()),
        },
        "pooled": {
            "positions": int(pooled.size),
            "kl_mean": float(pooled.mean()),
            "kl_p95": float(np.percentile(pooled, 95)),
            "kl_p99": float(np.percentile(pooled, 99)),
            "kl_max": float(pooled.max()),
        },
        "top1_agreement": top1,
        "candidate_finite": finite,
        "prefill_kl_sanity": float(pre[0]),
    }
    gate = (
        metrics["pooled"]["kl_mean"] <= 1e-3
        and metrics["pooled"]["kl_p95"] <= 5e-3
        and metrics["pooled"]["kl_p99"] <= 2e-2
        and metrics["pooled"]["kl_max"] <= 5e-2
        and top1 >= 0.99
        and finite
        and metrics["prefill_kl_sanity"] <= 1e-3
    )
    metrics["gate_pass"] = bool(gate)
    print(f"natural : mean {metrics['natural']['kl_mean']:.3e} "
          f"max {metrics['natural']['kl_max']:.3e} "
          f"top1 {metrics['natural']['top1']:.4f}")
    print(f"category: mean {metrics['category']['kl_mean']:.3e} "
          f"p99 {metrics['category']['kl_p99']:.3e} "
          f"max {metrics['category']['kl_max']:.3e}")
    print(f"pooled  : mean {metrics['pooled']['kl_mean']:.3e} "
          f"p99 {metrics['pooled']['kl_p99']:.3e} "
          f"max {metrics['pooled']['kl_max']:.3e} top1 {top1:.4f}")
    print(f"PROBE GATE (mean<=1e-3 & p95<=5e-3 & p99<=2e-2 & max<=5e-2 "
          f"& top1>=0.99 & finite & prefill-clean): "
          f"{'PASS' if gate else 'FAIL'}")
    if args.json is not None:
        Path(args.json).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
