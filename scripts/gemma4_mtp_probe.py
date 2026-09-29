"""Compare Gemma 4 greedy AR and MTP through the public LLM API.

Rates count emitted tokens over the whole call, including prefill. This is not
an isolated decode benchmark. Exit nonzero for missing token IDs, unequal
outputs, invalid lengths, or runtime errors. A single prompt is diagnostic;
use category suites and heldouts before making a general speedup claim.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import hipengine
from hipengine.llm import SamplingParams

MODEL_DIR = "/models/gguf/gemma-4-26B-A4B-it-GGUF"
DEFAULT_ARTIFACT = f"{MODEL_DIR}/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
DEFAULT_DRAFT = f"{MODEL_DIR}/mtp-gemma-4-26B-A4B-it-Q8_0.gguf"
DEFAULT_PROMPT = "Explain in a few sentences why the sky is blue."


def compare_outputs(plain, spec, *, requested_tokens: int, plain_s: float, spec_s: float) -> dict:
    """Validate a paired result before publishing its rate ratio."""
    ids = [getattr(value, "generated_token_ids", None) for value in (plain, spec)]
    reasons = [getattr(getattr(value, "finish_details", None), "reason", None)
               for value in (plain, spec)]
    errors = []
    counts = [None if tokens is None else len(tokens) for tokens in ids]
    for name, count, reason in zip(("plain", "spec"), counts, reasons):
        if count is None:
            errors.append(f"{name}: missing generated_token_ids")
        elif count <= 0 or count > requested_tokens:
            errors.append(f"{name}: invalid emitted count {count} for limit {requested_tokens}")
        elif count < requested_tokens and reason not in ("eos", "stop"):
            errors.append(f"{name}: short output without EOS or stop")
    tokens_equal = all(tokens is not None for tokens in ids) and tuple(ids[0]) == tuple(ids[1])
    text_equal = plain.text == spec.text
    if not tokens_equal:
        errors.append("generated token IDs differ or are missing")
    if not text_equal:
        errors.append("decoded text differs")
    if reasons[0] != reasons[1]:
        errors.append("finish reasons differ")
    valid_time = all(math.isfinite(t) and t > 0 for t in (plain_s, spec_s))
    if not valid_time:
        errors.append("wall times must be finite and positive")
    return {
        "timing_scope": "whole_public_call_including_prefill",
        "requested_tokens": requested_tokens,
        "plain_tokens": counts[0], "spec_tokens": counts[1],
        "plain_token_ids": None if ids[0] is None else list(ids[0]),
        "spec_token_ids": None if ids[1] is None else list(ids[1]),
        "plain_finish_reason": reasons[0], "spec_finish_reason": reasons[1],
        "plain_s": plain_s, "spec_s": spec_s,
        "plain_tok_s": counts[0] / plain_s if valid_time and counts[0] is not None else None,
        "spec_tok_s": counts[1] / spec_s if valid_time and counts[1] is not None else None,
        "tokens_identical": tokens_equal, "text_identical": text_equal,
        "fixed_length_complete": counts == [requested_tokens, requested_tokens],
        "speedup": plain_s / spec_s if not errors else None,
        "passed": not errors, "errors": errors,
    }


def load_cases(paths: list[Path], prompt: str) -> list[dict]:
    if not paths:
        return [{"id": "single_prompt", "category": "diagnostic", "prompt": prompt}]
    cases = []
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            case["source"] = str(path)
            if not case.get("id") or not case.get("category"):
                raise ValueError(f"{path}: each case needs id and category")
            if not case.get("messages") and not case.get("prompt"):
                raise ValueError(f"{path}: {case['id']} needs messages or prompt")
            cases.append(case)
    if not cases:
        raise ValueError("prompt suites are empty")
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    ap.add_argument("--draft", default=DEFAULT_DRAFT)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--suite", type=Path, action="append", default=[],
                    help="JSONL prompts with id/category and messages or prompt; repeat for heldouts")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--budget", type=int, default=2)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--provider", default="gemma4_mtp")
    ap.add_argument("--json", type=Path, help="write paired results and full model provenance")
    args = ap.parse_args()
    if args.tokens <= 0 or args.budget <= 0 or args.samples <= 0:
        ap.error("tokens, budget, and samples must be positive")
    if not os.path.exists(args.draft):
        ap.error(f"assistant sidecar missing: {args.draft}")
    cases = load_cases(args.suite, args.prompt)
    llm = hipengine.LLM(model=args.artifact, speculative_provider=args.provider,
                        draft_model=args.draft, speculative_candidate_budget=args.budget)
    generator = llm._get_text_generator()
    print(f"provider: {args.provider}; candidate_budget: {args.budget}")
    print("timing scope: whole public call including prefill; emitted-token denominator")
    results = []
    for case in cases:
        prompt = (generator.render_chat_prompt(case["messages"])
                  if "messages" in case else case["prompt"])
        params = SamplingParams(max_tokens=args.tokens, temperature=0.0)
        warmup = SamplingParams(max_tokens=min(8, args.tokens), temperature=0.0)
        try:
            # Plain first deliberately leaves state for MTP to reset. Do not
            # reset the private runner here: that would hide request leakage.
            llm.generate([prompt], warmup)
            llm.generate_speculative_mtp_detailed([prompt], warmup)
            for sample in range(args.samples):
                started = time.perf_counter()
                plain = llm.generate_detailed([prompt], params)
                plain_s = time.perf_counter() - started
                started = time.perf_counter()
                spec = llm.generate_speculative_mtp_detailed([prompt], params)
                spec_s = time.perf_counter() - started
                if len(plain) != 1 or len(spec) != 1:
                    raise ValueError("one prompt must produce exactly one output per arm")
                row = compare_outputs(plain[0], spec[0], requested_tokens=args.tokens,
                                      plain_s=plain_s, spec_s=spec_s)
                row.update(id=case["id"], category=case["category"], sample=sample,
                           source=case.get("source"), prompt=prompt)
                results.append(row)
                print(json.dumps({k: v for k, v in row.items()
                                  if k not in ("plain_token_ids", "spec_token_ids", "prompt")}), flush=True)
        except Exception as exc:
            results.append({"id": case["id"], "category": case["category"],
                            "passed": False, "errors": [f"{type(exc).__name__}: {exc}"]})
            print(f"{case['id']}: failed: {type(exc).__name__}: {exc}", flush=True)
    passed = bool(results) and all(row["passed"] for row in results)
    report = {"schema": "hipengine.gemma4.mtp_probe.v1", "performance_claim": False,
              "passed": passed, "candidate_budget": args.budget, "results": results,
              "limitations": ["Whole-call timing is not isolated decode throughput.",
                              "Token equality is a greedy self-consistency check, not a task-quality gate.",
                              "Plain-first ordering checks inherited state but is not order-balanced timing."]}
    if args.json:
        from hipengine.benchmark.provenance import collect_artifact_provenance, collect_model_identity
        report["provenance"] = collect_artifact_provenance(
            repo_root=Path(__file__).resolve().parents[1], model_path=args.artifact,
            quant=None, kv_dtype="bf16", command=[sys.executable, *sys.argv],
            timing_protocol="plain-first whole public call including prefill; emitted tokens",
            warmups=1, repetitions=args.samples)
        report["draft_provenance"] = collect_model_identity(args.draft)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
