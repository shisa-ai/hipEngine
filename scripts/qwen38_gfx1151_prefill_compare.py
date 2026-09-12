#!/usr/bin/env python3
"""Same-session prefill comparison; no runtime defaults are changed."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hipengine.kernels import hip_gfx1151
from hipengine.kernels.policy import QWEN35_DENSE_H5120_GEOMETRY
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.runtime import gguf_linear
from scripts.gguf_gdn_semantic_gate import DEFAULT_PROMPTS, _load_suites
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.execution_profile_gguf_fp16_state_gate import _run_logits_trajectory
from scripts.qwen38_production_ar_gate import profile_session

MODES = ("shipped", "unequal_pair", "no_retiles")
CAPABILITY = "GGUF_Q4_T16_UNEQUAL_PAIR_PREFILL_POLICIES"
TRACKED = (
    ("linear", "dense_unequal_dual_wmma_prefill_bf16_bf16_out"),
    ("linear_pair_silu", "dense_dual_wmma_prefill_row64_bf16_bf16_out"),
    ("linear_pair_silu", "dense_dual_wmma_prefill_row128_bf16_bf16_out"),
)


@contextmanager
def candidate_scope(mode):
    if mode not in MODES:
        raise ValueError(f"unknown candidate {mode}")
    present = hasattr(hip_gfx1151, CAPABILITY)
    previous = getattr(hip_gfx1151, CAPABILITY, None)
    retile = gguf_linear._Q4_T16_DUAL_SILU_RETILE_RESOLVED
    try:
        if mode == "unequal_pair":
            setattr(hip_gfx1151, CAPABILITY, {
                (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): True,
            })
        if mode == "no_retiles":
            gguf_linear._Q4_T16_DUAL_SILU_RETILE_RESOLVED = False
        yield
    finally:
        gguf_linear._Q4_T16_DUAL_SILU_RETILE_RESOLVED = retile
        if present:
            setattr(hip_gfx1151, CAPABILITY, previous)
        elif hasattr(hip_gfx1151, CAPABILITY):
            delattr(hip_gfx1151, CAPABILITY)


@contextmanager
def dispatch_counts():
    originals, counts = {}, {}
    try:
        for layer, variant in TRACKED:
            key = KernelKey("hip_gfx1151", layer, "gguf_q4_k_t16_v1", variant)
            inner = resolve(backend=key.backend, layer=key.layer,
                            quant=key.quant, variant=key.variant)
            originals[key] = inner
            counts[variant] = 0

            def wrapper(*args, _inner=inner, _variant=variant, **kwargs):
                counts[_variant] += 1
                return _inner(*args, **kwargs)

            register(key, wrapper, replace=True)
        yield counts
    finally:
        for key, inner in originals.items():
            register(key, inner, replace=True)


def trajectory_digest(trajectory):
    digest = hashlib.sha256()
    for step in trajectory:
        values = np.ascontiguousarray(step["logits"], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite trajectory")
        digest.update(values.tobytes())
        digest.update(int(step["token_id"]).to_bytes(8, "little"))
    return digest.hexdigest()


def run(args):
    from hipengine.benchmark.provenance import collect_artifact_provenance
    from hipengine.loading.gguf import scan_gguf
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
    prompts = _load_suites(DEFAULT_PROMPTS)
    modes = (args.baseline, args.candidate)
    quality, timings = [], []
    with profile_session(args, None) as (session, profile):
        for prompt in prompts:
            tokens = build_chat_prompt(tokenizer, str(prompt["prompt"]))
            captures = {}
            for mode in modes:
                with candidate_scope(mode), dispatch_counts() as counts:
                    trajectory = _run_logits_trajectory(
                        session, prompt_ids=tokens, decode_steps=args.decode_steps,
                        bulk_attention_mode="bulk")
                captures[mode] = {
                    "logits_and_tokens_sha256": trajectory_digest(trajectory),
                    "generated_ids": [step["token_id"] for step in trajectory],
                    "dispatch_counts": dict(counts),
                }
            equal = (captures[modes[0]]["logits_and_tokens_sha256"]
                     == captures[modes[1]]["logits_and_tokens_sha256"])
            quality.append(dict(prompt_id=prompt["id"], category=prompt["category"],
                                suite=prompt["suite"], exact=equal, arms=captures))
            print(f"{prompt['id']}: full-logit/token exact={equal}", flush=True)

        for rows in args.rows:
            tokens = [9707] * rows
            samples = {mode: [] for mode in modes}
            hashes = {mode: [] for mode in modes}
            for repeat in range(args.warmups + args.repetitions):
                order = modes if repeat % 2 == 0 else tuple(reversed(modes))
                for mode in order:
                    with candidate_scope(mode):
                        session.reset()
                        session.runtime.device_synchronize()
                        start = time.perf_counter()
                        result = session.prefill(tokens, use_bulk=True, return_logits=False)
                        session.runtime.device_synchronize()
                        elapsed = time.perf_counter() - start
                    if repeat >= args.warmups:
                        samples[mode].append(elapsed)
                        hashes[mode].append(int(result.token_id))
            medians = {mode: statistics.median(values) for mode, values in samples.items()}
            timings.append({
                "rows": rows, "seconds": samples, "median_seconds": medians,
                "prefill_tokens_per_second": {mode: rows / value for mode, value in medians.items()},
                "candidate_speedup": medians[modes[0]] / medians[modes[1]],
                "selected_tokens": hashes,
                "tokens_equal": len({v for values in hashes.values() for v in values}) == 1,
            })
            print(f"rows={rows}: speedup={timings[-1]['candidate_speedup']:.5f}", flush=True)

    exact = all(row["exact"] for row in quality) and all(row["tokens_equal"] for row in timings)
    counts = {
        mode: {variant: sum(row["arms"][mode]["dispatch_counts"][variant]
                            for row in quality) for _, variant in TRACKED}
        for mode in modes
    }
    if "unequal_pair" in modes:
        engaged = counts["unequal_pair"][TRACKED[0][1]] > 0
    else:
        engaged = all(counts["shipped"][variant] > 0
                      and counts["no_retiles"][variant] == 0
                      for _, variant in TRACKED[1:])
    provenance = collect_artifact_provenance(
        repo_root=ROOT, configured_backend="hip_gfx1151", resolved_backend="hip_gfx1151",
        target_arch="gfx1151", model_path=args.model, quant="gguf_q4_k_m",
        kv_dtype="bf16", command=[sys.executable, *sys.argv],
        environment={}, build_profile="prefill_candidate_comparison",
        timing_protocol="counterbalanced_same_session_synchronized_prefill_wall",
        warmups=args.warmups, repetitions=args.repetitions, profiler={"enabled": False})
    return {
        "kind": "qwen38_gfx1151_prefill_candidate_comparison", "schema_version": 1,
        "baseline": args.baseline, "candidate": args.candidate,
        "profile": profile, "quality": quality, "timings": timings,
        "exact_parent_gate_passed": exact,
        "candidate_effect_observed": engaged, "dispatch_totals": counts,
        "production_numerical_review_required": not exact,
        "performance_claim": False,
        "limitation": "Parent-parity screen, not complete serving/default qualification. Nonexact candidates require the full production review, not automatic rejection.",
        "provenance": provenance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"))
    parser.add_argument("--baseline", choices=MODES, default="shipped")
    parser.add_argument("--candidate", choices=MODES, default="unequal_pair")
    parser.add_argument("--rows", type=int, nargs="+", default=[64, 128, 512, 1024, 4096])
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--max-sequence-length", type=int, default=8192)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.baseline == args.candidate or min(args.rows) < 1
            or max(args.rows) >= args.max_sequence_length
            or args.repetitions < 2 or args.warmups < 1 or args.decode_steps < 32):
        parser.error("requires distinct arms, valid contexts, warmup, >=2 repeats and >=32 decode steps")
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(args.output)
    return 0 if (payload["exact_parent_gate_passed"]
                 and payload["candidate_effect_observed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
