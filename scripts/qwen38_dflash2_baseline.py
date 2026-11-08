#!/usr/bin/env python3
"""Qwen3.8-27B DFlash2 relative baseline (zbook — internal, NOT official publishing hw).

Full mtpbench category suite, N reps, native batched chain verify at block B.
Reuses the dflash2_gguf_cycle helpers (same path as dflash2_gguf_suite_smoke).
The per-prompt AR reference is deterministic and run once for correctness +
ar tok/s; DFlash2 decode tok/s is the median across reps. One discarded warmup
prompt precedes measurement. Writes a compact JSON artifact.

  PYTHONPATH=. .venv/bin/python scripts/qwen38_dflash2_baseline.py \
      --block-size 3 --reps 3 --max-new-tokens 25 --output OUT.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from hipengine.core.hip import get_hip_runtime
from hipengine.speculative.dflash2_native import DFlash2NativeDrafter
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from dflash2_gguf_cycle import (
    _load_target_arrays,
    _run_dflash2_cycle_batch,
    _run_ar,
    load_and_build_drafter,
)

FULL = [
    "code_merge_intervals", "code_topological_sort", "code_lru_cache",
    "code_markdown_table", "general_en_plan", "general_en_explain",
    "general_ja_plan", "general_ja_explain", "mixed_ja_en_translate",
    "mixed_ja_en_review",
]


def _prompt_text(messages) -> str:
    return "".join(m["content"] for m in messages if m.get("content"))


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    ap.add_argument("--drafter", default="/home/lhl/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4")
    ap.add_argument("--prompts-file", default="benchmarks/prompts/mtpbench-code-general-ja.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=25)
    ap.add_argument("--backend", default="hip_gfx1151")
    ap.add_argument("--block-size", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = {r["id"]: r for r in (json.loads(l) for l in open(args.prompts_file, encoding="utf-8"))}
    chosen = [rows[pid] for pid in FULL if pid in rows]
    tokenizer, token_embd, head = _load_target_arrays(args.model)
    drafter, numpy_weights = load_and_build_drafter(args.drafter)
    block_size = min(max(2, args.block_size), int(drafter.config.block_size))
    runtime = get_hip_runtime()
    max_seq = 4096

    def run_prompt(r, measure: bool):
        prompt_ids = tokenizer.encode(_prompt_text(r["messages"]))
        seq = len(prompt_ids) + args.max_new_tokens + block_size + 2
        native = DFlash2NativeDrafter(drafter.config, numpy_weights, max_context_len=seq)
        try:
            ar_out, ar_s = _run_ar(session, prompt_ids=prompt_ids,
                                   max_new_tokens=args.max_new_tokens, runtime=runtime)
            df2 = _run_dflash2_cycle_batch(
                session, native, numpy_weights, token_embd, head,
                prompt_ids=prompt_ids, max_new_tokens=args.max_new_tokens,
                block_size=block_size, runtime=runtime)
        finally:
            native.close()
        ids = df2["output_ids"]
        common = min(len(ids), len(ar_out))
        agree = sum(1 for a, b in zip(ids[:common], ar_out[:common]) if a == b)
        return {
            "id": r["id"], "category": r["category"], "prompt_len": len(prompt_ids),
            "dflash2_tok_s": df2["tokens_per_s"], "ar_tok_s": args.max_new_tokens / ar_s if ar_s else 0.0,
            "mean_acceptance": df2["mean_acceptance"], "accepted_per_draft": df2["accepted_per_draft"],
            "ar_agreement": agree / common if common else None,
            "exact": agree == common and common > 0,
        }

    with Qwen35GGUFResidentSession(
        args.model, backend=args.backend, compiler_version=None,
        require_cached_build=False, max_sequence_length=max_seq, max_batch_size=1,
    ) as session:
        # discarded warmup
        run_prompt(chosen[0], measure=False)
        per_prompt = {r["id"]: [] for r in chosen}
        meta = {}
        for rep in range(args.reps):
            for r in chosen:
                e = run_prompt(r, measure=True)
                per_prompt[r["id"]].append(e)
                meta[r["id"]] = {k: e[k] for k in ("category", "prompt_len", "ar_tok_s", "mean_acceptance", "accepted_per_draft")}
            print(f"rep {rep+1}/{args.reps} done", flush=True)

    results = []
    for pid in FULL:
        if pid not in per_prompt:
            continue
        reps = per_prompt[pid]
        med = statistics.median(e["dflash2_tok_s"] for e in reps)
        ar_med = statistics.median(e["ar_tok_s"] for e in reps)
        results.append({
            **meta[pid], "id": pid,
            "dflash2_tok_s_median": med, "ar_tok_s_median": ar_med,
            "dflash2_tok_s_reps": [e["dflash2_tok_s"] for e in reps],
            "ar_agreement": reps[0]["ar_agreement"],
            "exact_all_reps": all(e["exact"] for e in reps),
            "ratio_vs_ar": med / ar_med if ar_med else None,
        })
    df_med = statistics.median(r["dflash2_tok_s_median"] for r in results)
    ar_med = statistics.median(r["ar_tok_s_median"] for r in results)
    artifact = {
        "schema": 1, "kind": "qwen38_27b_dflash2_relative_baseline",
        "status": "relative_baseline_internal", "performance_claim": False,
        "publishable": False,
        "note": "zbook laptop relative baseline; NOT the official gfx1151 publishing lane. Thermally throttled; for internal before/after comparison only.",
        "provenance": {
            "hipengine_commit": _git("rev-parse", "HEAD"), "git_branch": _git("branch", "--show-current"),
            "model": args.model, "drafter": args.drafter, "backend": args.backend,
            "command": " ".join(sys.argv), "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "protocol": {
            "prompts_file": args.prompts_file, "suite": "mtpbench-code-general-ja (10 prompts)",
            "max_new_tokens": args.max_new_tokens, "block_size": block_size, "reps": args.reps,
            "verify": "native batched chain", "warmup": "1 discarded prompt",
            "aggregation": "median DFlash2 decode tok/s across reps; AR reference deterministic per prompt",
        },
        "summary": {
            "dflash2_tok_s_median_of_prompts": df_med, "ar_tok_s_median_of_prompts": ar_med,
            "ratio_vs_ar": df_med / ar_med if ar_med else None,
            "exact_all": all(r["exact_all_reps"] for r in results),
            "mean_acceptance": statistics.mean(r["mean_acceptance"] for r in results),
        },
        "rows": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2))
    print(f"DFlash2 B{block_size}: {df_med:.2f} tok/s vs AR {ar_med:.2f} "
          f"({df_med/ar_med:.2f}x) exact={artifact['summary']['exact_all']} "
          f"acc={artifact['summary']['mean_acceptance']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())