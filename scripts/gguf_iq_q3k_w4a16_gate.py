"""Teacher-forced probe for routing Q3_K over W4A16 (the mean-gate candidate).

Incumbent = the shipped four-quant W4A16 policy; candidate = the same policy
plus gguf_q3_k (K_M carries no IQ2_S/IQ2_XS, so the delta is Q3_K alone).
The zbook measured the candidate at mean 0.001061 (6.1% over the calibrated
1e-3); this probe re-establishes the pair on this host with the same
teacher-forced method as the local32 decode probe.
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
    if os.environ.get("HIPENGINE_Q3K_GATE_NATURAL") == "1":
        # Self-generated text: greedily extend a short random seed, then use
        # the extension as the probe prompt. Uniform-random token ids measure
        # a prompt-sensitivity floor at ~1e-3 that masks route differences;
        # the model's own output is the realistic token distribution the
        # campaign's real prompts approximate.
        with Qwen35GGUFResidentSession(
            args.model, compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
            max_sequence_length=args.prompt_tokens + 64,
            use_wmma_prefill=True, use_gemv_decode=True,
        ) as gen:
            seed_ids = [int(t) for t in rng.integers(1000, 50000, size=8)]
            first = gen.prefill(seed_ids, use_bulk=True, bulk_attention_mode="bulk",
                                return_logits=True)
            prompt = list(seed_ids)
            cur = first
            while len(prompt) < args.prompt_tokens:
                cur = gen.step(int(cur.token_id), return_logits=True)
                prompt.append(int(cur.token_id))
            prompt = prompt[:args.prompt_tokens]
    else:
        prompt = [int(t) for t in rng.integers(0, 150000, size=int(args.prompt_tokens))]
    n = int(args.decode_tokens)

    # The incumbent is pinned explicitly to the original four-quant W4A16 set
    # (the pre-enable shipped default): the shipped policy now routes Q3_K,
    # so copying the module default would make the candidate arm a no-op and
    # falsely certify regressions on rerun. The owner assertion below fails
    # loudly if the arms ever resolve to the same owner set again.
    base_policy = {
        quant: {"min_rows": 8, "max_rows": 131072,
                "variant": "dense_wmma_w4a16_prefill_bf16_bf16_out"}
        for quant in ("gguf_iq4_xs", "gguf_iq3_xxs", "gguf_iq3_s", "gguf_iq4_nl")
    }
    cand_policy = {**base_policy, "gguf_q3_k": dict(base_policy["gguf_iq4_xs"])}

    def _assert_owners_differ():
        from hipengine.kernels.registry import KernelKey, is_registered
        for quant in ("gguf_q3_k",):
            key = KernelKey(
                "hip_gfx1100", "linear", quant,
                "dense_wmma_w4a16_prefill_bf16_bf16_out")
            if not is_registered(key):
                raise RuntimeError(f"candidate owner {key} is not registered")
        if set(base_policy) == set(cand_policy):
            raise RuntimeError(
                "gate arms route the same quant set; the incumbent pin has "
                "gone stale"
            )

    _assert_owners_differ()
    # Optional third arm: the all-strict reference, to measure the incumbent's
    # own noise level on the same prompts (HIPENGINE_Q3K_GATE_STRICT_ARM=1).
    import os as _os
    strict_arm = _os.environ.get("HIPENGINE_Q3K_GATE_STRICT_ARM") == "1"
    ref_policy = {} if strict_arm else base_policy

    def run(policy, forced_tokens=None):
        be.GGUF_IQ_DENSE_PREFILL_POLICY = policy
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
            ref_logits, ref_tokens = run(ref_policy)
            session.reset()
            cand_logits, _ = run(cand_policy, forced_tokens=ref_tokens[:-1])
    finally:
        be.GGUF_IQ_DENSE_PREFILL_POLICY = base_policy

    # All positions count: the route changes prefill AND every decode position's
    # teacher-forced context through the changed prefill hidden states.
    metrics = evaluate_logits(ref_logits, cand_logits)
    top1 = float(np.mean(np.argmax(ref_logits, -1) == np.argmax(cand_logits, -1)))
    finite = bool(np.all(np.isfinite(cand_logits)))
    print(f"positions: {ref_logits.shape[0]}  (prompt={len(prompt)})")
    label = "all-strict ref" if strict_arm else "4-quant incumbent"
    print(f"KL vs {label}:  mean={metrics.kl_mean:.4e}  max={metrics.kl_max:.4e}")
    print(f"top1 agreement: {top1:.4f}  finite: {finite}")
    # Complete section-6.1 screen: calibrated mean, absolute per-position
    # ceiling, top-1, and finite candidate logits all bind. (The zbook's
    # recorded 7-quant arm delta was +0.000234 mean, 0.000827 -> 0.001061.)
    gate = (
        metrics.kl_mean <= 1e-3
        and metrics.kl_max <= 5e-2
        and top1 >= 0.99
        and finite
    )
    print(
        "PROBE GATE (mean<=1e-3 & max<=5e-2 & top1>=0.99 & finite): "
        f"{'PASS' if gate else 'FAIL'}"
    )
    if args.json is not None:
        args.json.write_text(json.dumps({
            "positions": int(ref_logits.shape[0]),
            "kl_mean": metrics.kl_mean, "kl_max": metrics.kl_max,
            "top1_agreement": top1, "candidate_finite": finite,
            "probe_gate_pass": gate,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
