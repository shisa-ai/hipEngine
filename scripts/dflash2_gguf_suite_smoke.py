#!/usr/bin/env python3
"""Multi-prompt DFlash2 native-cycle smoke across mtpbench categories.

Loads the Qwen3.8-27B GGUF target + DFlash2 drafter once, then runs the native
cycle (forward + selector + sequential greedy verify) on one prompt from each
of the four mtpbench categories. Reports acceptance + recall per prompt so the
D4 acceptance blocker can be checked for prompt-specificity.

Usage:
  python scripts/dflash2_gguf_suite_smoke.py [--max-new-tokens 40]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.loading.safetensors import load_weight_index
from hipengine.loading.dflash import validate_dflash_drafter_metadata
from hipengine.speculative.dflash2_drafter import DFlash2NumpyDrafter, load_dflash2_numpy_weights
from hipengine.speculative.dflash2_native import DFlash2NativeDrafter, _to_bf16_bits

from dflash2_gguf_cycle import (
    _capture_taps_host,
    _load_target_arrays,
    _run_dflash2_cycle_batch,
    _run_dflash2_cycle_native,
    _run_ar,
    DFLASH2_TAP_DEPTHS,
    DFlash2HiddenCaptureTargets,
    load_and_build_drafter,
)
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession


def _prompt_text(messages) -> str:
    return "".join(m["content"] for m in messages if m.get("content"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    parser.add_argument(
        "--drafter",
        default="/home/lhl/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
    )
    parser.add_argument("--prompts-file", default="benchmarks/prompts/mtpbench-code-general-ja.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--limit", type=int, default=4, help="prompts to run (default all 4 categories)")
    parser.add_argument("--batch-verify", action="store_true", help="use the B7 batched chain verifier")
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(args.prompts_file, encoding="utf-8")]
    cats = ["code", "general_en", "general_ja", "mixed_ja_en"]
    FULL = [
        "code_merge_intervals", "code_topological_sort", "code_lru_cache",
        "code_markdown_table", "general_en_plan", "general_en_explain",
        "general_ja_plan", "general_ja_explain", "mixed_ja_en_translate",
        "mixed_ja_en_review",
    ]
    chosen = []
    if args.limit is not None and args.limit == 10:
        for pid in FULL:
            for r in rows:
                if r.get("id") == pid:
                    chosen.append(r)
                    break
    else:
        for c in cats:
            for r in rows:
                if r.get("category") == c and r not in chosen:
                    chosen.append(r)
                    break
        chosen = chosen[: args.limit]

    tokenizer, token_embd, head = _load_target_arrays(args.model)
    drafter, numpy_weights = load_and_build_drafter(args.drafter)
    block_size = int(drafter.config.block_size)
    runtime = get_hip_runtime()
    max_seq = 4096

    results = {}
    with Qwen35GGUFResidentSession(
        args.model, backend=args.backend, compiler_version=None,
        require_cached_build=False, max_sequence_length=max_seq, max_batch_size=1,
    ) as session:
        # AR reference once (deterministic per prompt) for correctness.
        for r in chosen:
            prompt_text = _prompt_text(r["messages"])
            prompt_ids = tokenizer.encode(prompt_text)
            seq = len(prompt_ids) + args.max_new_tokens + block_size + 2
            native = DFlash2NativeDrafter(drafter.config, numpy_weights, max_context_len=seq)
            try:
                ar_out, ar_s = _run_ar(session, prompt_ids=prompt_ids, max_new_tokens=args.max_new_tokens, runtime=runtime)
                if args.batch_verify:
                    df2 = _run_dflash2_cycle_batch(
                        session, native, numpy_weights, token_embd, head,
                        prompt_ids=prompt_ids, max_new_tokens=args.max_new_tokens,
                        block_size=block_size, runtime=runtime,
                    )
                else:
                    df2 = _run_dflash2_cycle_native(
                        session, native, numpy_weights, token_embd, head,
                        prompt_ids=prompt_ids, max_new_tokens=args.max_new_tokens,
                        block_size=block_size, runtime=runtime,
                    )
            finally:
                native.close()
            ids = df2["output_ids"]
            common = min(len(ids), len(ar_out))
            agree = sum(1 for a, b in zip(ids[:common], ar_out[:common]) if a == b)
            entry = {
                "category": r["category"],
                "id": r["id"],
                "prompt_len": len(prompt_ids),
                "mean_acceptance": df2["mean_acceptance"],
                "accepted_per_draft": df2["accepted_per_draft"],
                "recall_at1": df2["recall_at1"],
                "recall_at16": df2["recall_at16"],
                "recall_unary_argmax": df2["recall_unary_argmax"],
                "tokens_per_s": df2["tokens_per_s"],
                "ar_tokens_per_s": args.max_new_tokens / ar_s,
                "ar_agreement": agree / common if common else None,
            }
            results[r["id"]] = entry
            print(f"[{r['category']:10s} {r['id']:24s}] plen={entry['prompt_len']:3d} "
                  f"acc={entry['mean_acceptance']:.2f} acc/draft={entry['accepted_per_draft']:.3f} "
                  f"r1={entry['recall_at1']:.3f} r16={entry['recall_at16']:.3f} unary={entry['recall_unary_argmax']:.3f} "
                  f"tok/s={entry['tokens_per_s']:.2f} ar_agree={entry['ar_agreement']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
