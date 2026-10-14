#!/usr/bin/env python3
"""In-process Qwen4Exp MTP economics on the full mtp-bench category suite.

One resident target, one process: for every prompt of
``benchmarks/prompts/mtpbench-code-general-ja.jsonl`` (all four categories,
heldouts included) a true autoregressive arm (target prefill + greedy decode
steps, no MTP) and an MTP arm (``Qwen4ExpMTPTextProvider``) generate the same
number of tokens from the same chat-formatted prompt. Reports request wall,
tok/s, MTP/AR ratio by category and overall, draft acceptance, verify phase
cost, and per-prompt token agreement with the AR arm (a diagnostic: the MTP
verifier runs prefill-shaped kernels, so its greedy path may legitimately
diverge from serial decode under production arithmetic).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.qwen4exp_canonical_ar_bench import _git_metadata, _host_metadata, _write_json
from scripts.qwen4exp_mtp_phase_census import DEFAULT_MODEL, DEFAULT_PROMPTS, DEFAULT_SIDECAR


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--execution-profile", default="production")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_mtp import Qwen4ExpMTPTextProvider
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_BACKEND, QWEN4_EXP_MODEL, QWEN4_EXP_QUANTS,
        register_qwen4_exp_gfx1151_profiles,
    )
    from hipengine.generation.registry import GenerationRequest
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models import resolve_model
    from hipengine.speculative.registry import SpeculativeProviderConfig

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    index = load_gguf_index(discover_gguf_files(args.model_root)[0])
    profile = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND, quant=QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile(args.execution_profile))
    generator = profile.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=args.model_root, weight_index=index,
        model_plugin=resolve_model(index.architecture or ""), backend=QWEN4_EXP_BACKEND,
        max_sequence_length=1024, prefill_chunk_size=1024))
    provider = Qwen4ExpMTPTextProvider(
        target_generator=generator,
        config=SpeculativeProviderConfig(
            provider="qwen4_exp_mtp", draft_model=args.sidecar,
            candidate_budget=args.budget))
    runner = generator.runner
    rows = []
    try:
        for row in _load(args.prompts):
            text = "\n".join(str(m["content"]) for m in row["messages"])
            prompt = [int(t) for t in build_chat_prompt(generator.tokenizer, text)]
            # Warm both arms on this prompt once (JIT, graphs), then measure.
            for measured in (False, True):
                started = time.perf_counter()
                runner.reset()
                result = runner.prefill(prompt)
                ar_tokens = [int(result.token_id)]
                while len(ar_tokens) < args.max_tokens:
                    result = runner.step(ar_tokens[-1])
                    ar_tokens.append(int(result.token_id))
                ar_seconds = time.perf_counter() - started
                request = GenerationRequest(
                    prompts=(tuple(prompt),), max_tokens=args.max_tokens,
                    temperature=0.0, top_p=1.0, ignore_eos=True, top_k=1)
                started = time.perf_counter()
                output = provider.generate_detailed(request)[0]
                mtp_seconds = time.perf_counter() - started
            diagnostics = output.telemetry.diagnostics
            mtp_tokens = [int(t) for t in output.generated_token_ids]
            agree = 0
            for a, b in zip(ar_tokens, mtp_tokens):
                if a != b:
                    break
                agree += 1
            census = diagnostics.get("phase_census", {})
            rows.append({
                "id": row["id"], "category": row["category"],
                "prompt_tokens": len(prompt),
                "ar_seconds": ar_seconds, "mtp_seconds": mtp_seconds,
                "ar_tok_s": len(ar_tokens) / ar_seconds,
                "mtp_tok_s": len(mtp_tokens) / mtp_seconds,
                "mtp_over_ar": ar_seconds / mtp_seconds,
                "proposed": diagnostics["proposed_draft_tokens"],
                "accepted": diagnostics["accepted_draft_tokens"],
                "acceptance": diagnostics["draft_acceptance"],
                "cycles": census.get("cycles"),
                "verify_ms": census.get("target_verify", {}).get("ms"),
                "proposal_ms": census.get("proposal", {}).get("ms"),
                "agree_prefix_with_ar": agree,
                "identical_to_ar": ar_tokens == mtp_tokens,
                "target_verify": diagnostics.get("target_verify"),
            })
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                              for k, v in rows[-1].items()}), flush=True)
    finally:
        provider.close()
        generator.close()

    groups = defaultdict(list)
    for r in rows:
        groups["full"].append(r)
        groups["category:" + r["category"]].append(r)
    summary = {}
    for name, group in groups.items():
        ar = sum(r["ar_seconds"] for r in group)
        mtp = sum(r["mtp_seconds"] for r in group)
        tokens = args.max_tokens * len(group)
        proposed = sum(r["proposed"] for r in group)
        summary[name] = {
            "prompts": len(group), "ar_tok_s": tokens / ar, "mtp_tok_s": tokens / mtp,
            "mtp_over_ar": ar / mtp,
            "acceptance": sum(r["accepted"] for r in group) / proposed if proposed else 0.0,
            "identical_to_ar": sum(int(r["identical_to_ar"]) for r in group),
        }
    result = {
        "schema": 1, "kind": "qwen4exp_mtp_inprocess_bench",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "performance_claim": True,
        "source": _git_metadata(ROOT), "host": _host_metadata(),
        "command": [Path(sys.argv[0]).name, *(argv or sys.argv[1:])],
        "protocol": {
            "prompts": str(args.prompts), "max_tokens": args.max_tokens,
            "budget": args.budget, "execution_profile": args.execution_profile,
            "warmup": "one unmeasured AR and MTP run per prompt",
            "batched_verify_env": os.environ.get("HIPENGINE_QWEN4_EXP_MTP_BATCHED_VERIFY"),
        },
        "rows": rows, "summary": summary,
    }
    _write_json(args.output, result)
    print(json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                          for kk, vv in v.items()} for k, v in summary.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
