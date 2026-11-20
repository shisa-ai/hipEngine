#!/usr/bin/env python3
"""Qwen3.8-27B DFlash2 relative baseline (zbook — internal, NOT official publishing hw).

Full mtpbench category suite, N reps, native batched chain verify at block B.
Reuses the dflash2_gguf_cycle helpers (same path as dflash2_gguf_suite_smoke).
AR and speculative runs are paired for every prompt and repetition. Primary
throughput pools generated-token counts and elapsed times; per-prompt medians
are diagnostic. One discarded warmup prompt precedes measurement. The legacy
post-prefill cycle timing is not a public streaming latency measurement.

  PYTHONPATH=. .venv/bin/python scripts/qwen38_dflash2_baseline.py \
      --block-size 3 --reps 3 --max-new-tokens 25 --output OUT.json
"""
from __future__ import annotations

import argparse
import json
import platform
import shlex
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
from speculative_baseline_metrics import AGGREGATION, TIMING_SCOPE, aggregate, measurement
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
    if args.reps < 1 or args.max_new_tokens < 2:
        ap.error("require reps >= 1 and max-new-tokens >= 2")

    with open(args.prompts_file, encoding="utf-8") as handle:
        rows = {r["id"]: r for r in map(json.loads, handle)}
    chosen = [rows[pid] for pid in FULL]
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
        return measurement(
            method="dflash2", prompt=r, prompt_len=len(prompt_ids),
            output_ids=df2["output_ids"], ar_ids=ar_out, expected_tokens=args.max_new_tokens,
            seconds=df2["decode_s"], ar_seconds=ar_s,
            mean_acceptance=df2["mean_acceptance"], accepted_per_draft=df2["accepted_per_draft"],
        )

    with Qwen35GGUFResidentSession(
        args.model, backend=args.backend, compiler_version=None,
        require_cached_build=False, max_sequence_length=max_seq, max_batch_size=1,
    ) as session:
        # discarded warmup
        run_prompt(chosen[0], measure=False)
        per_prompt = {r["id"]: [] for r in chosen}
        for rep in range(args.reps):
            for r in chosen:
                per_prompt[r["id"]].append(run_prompt(r, measure=True))
            print(f"rep {rep+1}/{args.reps} done", flush=True)

    results, summary = aggregate(per_prompt, method="dflash2")
    artifact = {
        "schema": 2, "kind": "qwen38_27b_dflash2_relative_baseline",
        "status": "relative_baseline_internal", "performance_claim": False,
        "publishable": False,
        "note": "zbook laptop relative baseline; NOT the official gfx1151 publishing lane. Thermally throttled; for internal before/after comparison only.",
        "provenance": {
            "hipengine_commit": _git("rev-parse", "HEAD"), "git_branch": _git("branch", "--show-current"),
            "model": args.model, "drafter": args.drafter, "backend": args.backend,
            "command": shlex.join([sys.executable, *sys.argv]), "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "host": platform.node(),
        },
        "protocol": {
            "prompts_file": args.prompts_file, "suite": "mtpbench-code-general-ja (10 prompts)",
            "max_new_tokens": args.max_new_tokens, "block_size": block_size, "reps": args.reps,
            "verify": "native batched chain", "warmup": "1 discarded prompt",
            "aggregation": AGGREGATION, "timing_scope": TIMING_SCOPE,
        },
        "summary": summary,
        "rows": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2))
    print(f"DFlash2 B{block_size}: {summary['dflash2_tok_s_weighted']:.2f} tok/s weighted "
          f"vs AR {summary['ar_tok_s_weighted']:.2f} ({summary['ratio_vs_ar']:.2f}x) "
          f"exact={summary['exact_all']} acc={summary['mean_acceptance']:.3f}", flush=True)
    return 0 if summary["exact_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())