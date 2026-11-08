#!/usr/bin/env python3
"""Qwen3.8-27B DSpine relative baseline (internal, NOT official publishing hw).

Full mtpbench category suite, N reps, native DSpine block verify. Reuses the
dspine_gguf_cycle helpers (same resident-session surface as the DFlash2 baseline).
The per-prompt AR reference is deterministic and run once for correctness + ar
tok/s; DSpine decode tok/s is the median across reps. One discarded warmup prompt
precedes measurement. Writes a compact JSON artifact.

  PYTHONPATH=. .venv/bin/python scripts/qwen38_dspine_baseline.py \
      --reps 3 --max-new-tokens 25 --output OUT.json
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
from hipengine.kernels.cpu_reference.dspine import DSpineNumpyDrafter
from hipengine.runtime.dspine import DSpineNativeDrafter
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from dflash2_gguf_cycle import _load_target_arrays, _run_ar
from dspine_gguf_cycle import load_dspine_weights, run_dspine_cycle

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
    ap.add_argument("--drafter", default="/home/lhl/.cache/huggingface/hub/models--tencent--Qwen3.8-27B-DSpine-b16/snapshots/ea909c0f5fab8865f9e14600334f55f7ce47308f")
    ap.add_argument("--prompts-file", default="benchmarks/prompts/mtpbench-code-general-ja.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=25)
    ap.add_argument("--backend", default="hip_gfx1151")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = {r["id"]: r for r in (json.loads(l) for l in open(args.prompts_file, encoding="utf-8"))}
    chosen = [rows[pid] for pid in FULL if pid in rows]
    tokenizer, token_embd, head = _load_target_arrays(args.model)
    config, weights = load_dspine_weights(args.drafter)
    nd = DSpineNumpyDrafter(config, weights)
    block_size = int(config["block_size"])
    runtime = get_hip_runtime()
    max_seq = 4096

    def run_prompt(r):
        prompt_ids = tokenizer.encode(_prompt_text(r["messages"]))
        seq = len(prompt_ids) + args.max_new_tokens + block_size + 2
        native = DSpineNativeDrafter(config, weights, runtime=runtime, max_context_len=seq)
        try:
            ar_out, ar_s = _run_ar(session, prompt_ids=prompt_ids,
                                   max_new_tokens=args.max_new_tokens, runtime=runtime)
            ds = run_dspine_cycle(session, native, nd, token_embd, head,
                                  prompt_ids=prompt_ids, max_new_tokens=args.max_new_tokens,
                                  runtime=runtime)
        finally:
            native.close()
        ids = ds["output_ids"]
        common = min(len(ids), len(ar_out))
        agree = sum(1 for a, b in zip(ids[:common], ar_out[:common]) if a == b)
        return {
            "id": r["id"], "category": r["category"], "prompt_len": len(prompt_ids),
            "dspine_tok_s": ds["tokens_per_s"], "ar_tok_s": args.max_new_tokens / ar_s if ar_s else 0.0,
            "mean_acceptance": ds["mean_acceptance"], "accepted_per_draft": ds["accepted_per_draft"],
            "ar_agreement": agree / common if common else None,
            "exact": agree == common and common > 0,
        }

    with Qwen35GGUFResidentSession(
        args.model, backend=args.backend, compiler_version=None,
        require_cached_build=False, max_sequence_length=max_seq, max_batch_size=1,
    ) as session:
        run_prompt(chosen[0])  # discarded warmup
        per_prompt = {r["id"]: [] for r in chosen}
        meta = {}
        for rep in range(args.reps):
            for r in chosen:
                e = run_prompt(r)
                per_prompt[r["id"]].append(e)
                meta[r["id"]] = {k: e[k] for k in ("category", "prompt_len", "ar_tok_s", "mean_acceptance", "accepted_per_draft")}
            print(f"rep {rep+1}/{args.reps} done", flush=True)

    results = []
    for pid in FULL:
        if pid not in per_prompt:
            continue
        reps = per_prompt[pid]
        med = statistics.median(e["dspine_tok_s"] for e in reps)
        ar_med = statistics.median(e["ar_tok_s"] for e in reps)
        results.append({
            **meta[pid], "id": pid,
            "dspine_tok_s_median": med, "ar_tok_s_median": ar_med,
            "dspine_tok_s_reps": [e["dspine_tok_s"] for e in reps],
            "ar_agreement": reps[0]["ar_agreement"],
            "exact_all_reps": all(e["exact"] for e in reps),
            "ratio_vs_ar": med / ar_med if ar_med else None,
        })
    ds_med = statistics.median(r["dspine_tok_s_median"] for r in results)
    ar_med = statistics.median(r["ar_tok_s_median"] for r in results)
    artifact = {
        "schema": 1, "kind": "qwen38_27b_dspine_relative_baseline",
        "status": "relative_baseline_internal", "performance_claim": False,
        "publishable": False,
        "note": "Internal relative baseline; native DSpine drafter with CPU transition-cache refine. Speed is pre-optimization (task #18).",
        "provenance": {
            "hipengine_commit": _git("rev-parse", "HEAD"), "git_branch": _git("branch", "--show-current"),
            "model": args.model, "drafter": args.drafter, "backend": args.backend,
            "command": " ".join(sys.argv), "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "protocol": {
            "prompts_file": args.prompts_file, "suite": "mtpbench-code-general-ja (10 prompts)",
            "max_new_tokens": args.max_new_tokens, "block_size": block_size, "reps": args.reps,
            "verify": "native chain (verify_target_block)", "warmup": "1 discarded prompt",
            "aggregation": "median DSpine decode tok/s across reps; AR reference deterministic per prompt",
        },
        "summary": {
            "dspine_tok_s_median_of_prompts": ds_med, "ar_tok_s_median_of_prompts": ar_med,
            "ratio_vs_ar": ds_med / ar_med if ar_med else None,
            "exact_all": all(r["exact_all_reps"] for r in results),
            "mean_acceptance": statistics.mean(r["mean_acceptance"] for r in results),
        },
        "rows": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2))
    print(f"DSpine B{block_size}: {ds_med:.2f} tok/s vs AR {ar_med:.2f} "
          f"({ds_med/ar_med:.2f}x) exact={artifact['summary']['exact_all']} "
          f"acc={artifact['summary']['mean_acceptance']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())