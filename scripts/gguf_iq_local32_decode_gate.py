"""Teacher-forced decode gate for the local32 IQ4_XS decode owner.

Runs the incumbent (local32 route disabled) eagerly to fix a token trajectory,
then teacher-forces the candidate (route enabled) onto the same tokens so
every position shares an identical context, and evaluates per-position
logit-KL / top-1 with the campaign's evaluate_logits. Modeled on
scripts/gguf_fused_moe_ffn_teacher_forced_kl.py.

The campaign's production-referenced gate scores candidate production against
the incumbent production default under the calibrated section-6.1 envelope;
this probe measures exactly that arm pair at the actual owner change.
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
    ap.add_argument("--model", type=Path,
                     default=Path("/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf"))
    ap.add_argument("--compiler-version-file", type=Path, default=None)
    ap.add_argument("--require-cached-build", action="store_true")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    from hipengine.benchmark.correctness import evaluate_logits
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    import hipengine.kernels.hip_gfx1100 as be

    compiler_version = (Path(args.compiler_version_file).read_text()
                        if args.compiler_version_file else None)
    rng = np.random.default_rng(args.seed)
    if os.environ.get("HIPENGINE_LOCAL32_GATE_NATURAL") == "1":
        # Self-generated text (the Q3_K gate lesson, 2026-09-10): uniform
        # random token ids measure a prompt-sensitivity floor at ~1e-3 that
        # masks route differences; the model's own greedy output is the
        # realistic token distribution the campaign's real prompts
        # approximate.
        from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession as _S
        with _S(
            args.model, compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
            max_sequence_length=args.prompt_tokens + args.decode_tokens + 64,
            use_wmma_prefill=True, use_gemv_decode=True,
        ) as gen:
            seed_ids = [int(t) for t in rng.integers(1000, 50000, size=8)]
            cur = gen.prefill(seed_ids, use_bulk=True, bulk_attention_mode="bulk",
                              return_logits=True)
            prompt = list(seed_ids)
            while len(prompt) < args.prompt_tokens:
                cur = gen.step(int(cur.token_id), return_logits=True)
                prompt.append(int(cur.token_id))
            prompt = prompt[:args.prompt_tokens]
    else:
        prompt = [int(t) for t in rng.integers(0, 150000, size=int(args.prompt_tokens))]
    n = int(args.decode_tokens)

    # The shipped policy is empty by decision; the candidate is explicit so
    # this probe stays meaningful regardless of the shipped default.
    base_policy = dict(be.GGUF_IQ_DENSE_DECODE_POLICY)
    candidate_policy = {"gguf_iq4_xs": {"variant": "local32_gemv_bf16_bf16_out"}}

    def run(local32: bool, forced_tokens=None):
        be.GGUF_IQ_DENSE_DECODE_POLICY = candidate_policy if local32 else base_policy
        logits_rows = []
        tokens = []
        first = session.prefill(prompt, use_bulk=True, bulk_attention_mode="bulk",
                                 return_logits=True)
        logits_rows.append(np.asarray(first.logits, dtype=np.float32).reshape(-1))
        tokens.append(int(first.token_id))
        cur = first
        for i in range(n):
            feed = int(cur.token_id) if forced_tokens is None else int(forced_tokens[i])
            cur = session.step(feed, return_logits=True)
            logits_rows.append(np.asarray(cur.logits, dtype=np.float32).reshape(-1))
            tokens.append(int(cur.token_id))
        return np.vstack(logits_rows), tokens

    try:
        with Qwen35GGUFResidentSession(
            args.model,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
            max_sequence_length=len(prompt) + n + 2,
            use_wmma_prefill=True,
            use_gemv_decode=True,
        ) as session:
            # Incumbent: local32 disabled, eager -> fixes the trajectory.
            ref_logits, ref_tokens = run(local32=False)
            session.reset()
            # Candidate: local32 enabled, teacher-forced on the same tokens.
            cand_logits, _ = run(local32=True, forced_tokens=ref_tokens[:-1])
    finally:
        be.GGUF_IQ_DENSE_DECODE_POLICY = base_policy

    ref_dec, cand_dec = ref_logits[1:], cand_logits[1:]
    metrics = evaluate_logits(ref_dec, cand_dec)
    top1 = float(np.mean(np.argmax(ref_dec, -1) == np.argmax(cand_dec, -1)))
    finite = bool(np.all(np.isfinite(cand_logits)))
    pre = evaluate_logits(ref_logits[:1], cand_logits[:1])
    print(f"teacher-forced decode positions: {ref_dec.shape[0]}  (prompt={len(prompt)})")
    print(f"prefill position KL (route-independent sanity): {pre.kl_mean:.3e}")
    print(f"DECODE per-position KL:  mean={metrics.kl_mean:.4e}  max={metrics.kl_max:.4e}")
    print(f"DECODE per-position top1 agreement: {top1:.4f}")
    print(f"candidate logits finite: {finite}")
    # Section 6.1 envelope (mean/p95/p99/max/top-1) is the binding gate for the
    # production run; this probe's single-context screen uses its mean and top-1.
    gate = metrics.kl_mean <= 1e-3 and top1 >= 0.99
    print(f"PROBE GATE (mean KL<=1e-3 & top1>=0.99): {'PASS' if gate else 'FAIL'}")
    if args.json is not None:
        args.json.write_text(json.dumps({
            "decode_positions": int(ref_dec.shape[0]),
            "prompt_tokens": len(prompt),
            "kl_mean": metrics.kl_mean, "kl_max": metrics.kl_max,
            "top1_agreement": top1, "candidate_finite": finite,
            "prefill_kl_sanity": pre.kl_mean, "probe_gate_pass": gate,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
