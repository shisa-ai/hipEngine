"""Combined-stack admission gate for the shipped UD route set (gfx1100).

Compares the declared production incumbent (the session-start shipped state:
four-quant W4A16 prefill policy, empty decode policy - strict per-row GEMV)
against the current shipped production stack (seven-quant prefill incl. the
Q3_K hi+lo split, local32 IQ4_XS decode) on one artifact, teacher-forced, on
natural self-generated prompts. Both arms' policies are pinned explicitly and
asserted to differ - copying the module defaults would compare the candidate
against itself now that the routes are enabled.

The complete section-6.1 screen binds: mean KL <= 1e-3, per-position max
<= 5e-2, top-1 >= 0.99, finite candidate logits, and a zero-KL prefill-only
sanity arm (decode policy must not perturb prefill).
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

_W4A16_ENTRY = {"min_rows": 8, "max_rows": 131072,
                "variant": "dense_wmma_w4a16_prefill_bf16_bf16_out"}
INCUMBENT_PREFILL = {
    quant: dict(_W4A16_ENTRY)
    for quant in ("gguf_iq4_xs", "gguf_iq3_xxs", "gguf_iq3_s", "gguf_iq4_nl")
}
SHIPPED_PREFILL = {
    **{quant: dict(_W4A16_ENTRY)
       for quant in ("gguf_iq4_xs", "gguf_iq3_xxs", "gguf_iq3_s", "gguf_iq4_nl",
                     "gguf_q3_k", "gguf_iq2_s", "gguf_iq2_xs")},
}
INCUMBENT_DECODE: dict = {}
SHIPPED_DECODE = {"gguf_iq4_xs": {"variant": "local32_gemv_bf16_bf16_out"}}


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

    # Arm-pin assertions: the policies must route different quant sets.
    assert set(INCUMBENT_PREFILL) != set(SHIPPED_PREFILL), (
        "prefill arms route the same quant set; the incumbent pin is stale")
    assert INCUMBENT_DECODE != SHIPPED_DECODE, (
        "decode arms route the same policy; the incumbent pin is stale")

    rng = np.random.default_rng(args.seed)
    n = int(args.decode_tokens)

    def natural_prompt(gen):
        seed_ids = [int(t) for t in rng.integers(1000, 50000, size=8)]
        cur = gen.prefill(seed_ids, use_bulk=True, bulk_attention_mode="bulk",
                          return_logits=True)
        prompt = list(seed_ids)
        while len(prompt) < args.prompt_tokens:
            cur = gen.step(int(cur.token_id), return_logits=True)
            prompt.append(int(cur.token_id))
        return prompt[:args.prompt_tokens]

    def run(prefill_policy, decode_policy, forced_tokens=None):
        be.GGUF_IQ_DENSE_PREFILL_POLICY = prefill_policy
        be.GGUF_IQ_DENSE_DECODE_POLICY = decode_policy
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

    prompt = None
    try:
        with Qwen35GGUFResidentSession(
            args.model,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
            max_sequence_length=args.prompt_tokens + n + 64,
            use_wmma_prefill=True,
            use_gemv_decode=True,
        ) as session:
            prompt = natural_prompt(session)
            session.reset()
            # Incumbent: pinned four-quant prefill + strict decode, eager.
            ref_logits, ref_tokens = run(INCUMBENT_PREFILL, INCUMBENT_DECODE)
            session.reset()
            # Candidate: shipped prefill + decode, teacher-forced.
            cand_logits, _ = run(SHIPPED_PREFILL, SHIPPED_DECODE,
                                 forced_tokens=ref_tokens[:-1])
    finally:
        be.GGUF_IQ_DENSE_PREFILL_POLICY = SHIPPED_PREFILL
        be.GGUF_IQ_DENSE_DECODE_POLICY = SHIPPED_DECODE

    metrics = evaluate_logits(ref_logits, cand_logits)
    top1 = float(np.mean(np.argmax(ref_logits, -1) == np.argmax(cand_logits, -1)))
    finite = bool(np.all(np.isfinite(cand_logits)))
    print(f"model: {args.model.name}  seed: {args.seed}")
    print(f"positions: {ref_logits.shape[0]}  (natural prompt={len(prompt)})")
    print(f"KL vs pinned incumbent:  mean={metrics.kl_mean:.4e}  max={metrics.kl_max:.4e}")
    print(f"top1 agreement: {top1:.4f}  finite: {finite}")
    gate = (
        metrics.kl_mean <= 1e-3
        and metrics.kl_max <= 5e-2
        and top1 >= 0.99
        and finite
    )
    print(f"COMBINED GATE (mean<=1e-3 & max<=5e-2 & top1>=0.99 & finite): "
          f"{'PASS' if gate else 'FAIL'}")
    if args.json is not None:
        args.json.write_text(json.dumps({
            "model": str(args.model), "seed": args.seed,
            "positions": int(ref_logits.shape[0]),
            "kl_mean": metrics.kl_mean, "kl_max": metrics.kl_max,
            "top1_agreement": top1, "candidate_finite": finite,
            "gate_pass": gate,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
