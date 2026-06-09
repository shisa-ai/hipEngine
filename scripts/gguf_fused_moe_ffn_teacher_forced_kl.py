"""B2 E2E correctness certification: teacher-forced per-position logit-KL for the
fused MoE FFN megakernel vs the unfused chain on the real GGUF model.

Eager decode makes the fused/unfused greedy paths diverge (benign argmax ties),
so eager KL is meaningless past the first divergence. Instead this runs the
reference (fused OFF) eagerly to fix a token trajectory, then teacher-forces the
candidate (fused ON) onto the SAME tokens so every position shares an identical
context. evaluate_logits then gives a true per-position KL / top-1 agreement.

Raw Q4_K mode (no decode-repack) is used because the fused kernel only applies
there. Prefill is unaffected by the flag (rows>1 path), so position 0 KL is ~0.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf"))
    ap.add_argument("--compiler-version-file", type=Path, default=None)
    ap.add_argument("--require-cached-build", action="store_true")
    ap.add_argument("--prompt-tokens", type=int, default=48)
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    from hipengine.benchmark.correctness import evaluate_logits
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    # Deterministic, spread-out prompt token ids (content-agnostic numerical test).
    prompt = [int((i * 7919 + 13) % 150000) for i in range(int(args.prompt_tokens))]
    n = int(args.decode_tokens)
    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = Path(args.compiler_version_file).read_text()

    # Raw Q4_K decode mode: no decode-repack so expert weights stay raw gguf_q4_k.
    os.environ.pop("HIPENGINE_GGUF_DECODE_REPACK", None)

    def run(fused: bool, forced_tokens=None):
        os.environ["HIPENGINE_GGUF_FUSED_MOE_FFN"] = "1" if fused else "0"
        logits_rows = []
        tokens = []
        first = session.prefill(prompt, use_bulk=True, bulk_attention_mode="bulk", return_logits=True)
        logits_rows.append(np.asarray(first.logits, dtype=np.float32).reshape(-1))
        tokens.append(int(first.token_id))
        cur = first
        for i in range(n):
            feed = int(cur.token_id) if forced_tokens is None else int(forced_tokens[i])
            cur = session.step(feed, return_logits=True)
            logits_rows.append(np.asarray(cur.logits, dtype=np.float32).reshape(-1))
            tokens.append(int(cur.token_id))
        return np.vstack(logits_rows), tokens

    with Qwen35GGUFResidentSession(
        args.model,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        max_sequence_length=len(prompt) + n + 2,
        use_wmma_prefill=None,
        use_gemv_decode=False,
    ) as session:
        # Reference: fused OFF, eager -> fixes the teacher-forcing trajectory.
        ref_logits, ref_tokens = run(fused=False)
        # Candidate: fused ON, teacher-forced on the reference tokens.
        session.reset()
        cand_logits, cand_tokens = run(fused=True, forced_tokens=ref_tokens[:-1])

    # Position 0 is prefill (flag-independent). Certify the decode positions.
    ref_dec = ref_logits[1:]
    cand_dec = cand_logits[1:]
    metrics = evaluate_logits(ref_dec, cand_dec)
    # Per-position top-1 agreement on the teacher-forced contexts.
    top1 = float(np.mean(np.argmax(ref_dec, -1) == np.argmax(cand_dec, -1)))
    finite = bool(np.all(np.isfinite(cand_logits)))
    print(f"teacher-forced decode positions: {ref_dec.shape[0]}  (prompt={len(prompt)})")
    print(f"prefill position KL (flag-independent sanity): "
          f"{evaluate_logits(ref_logits[:1], cand_logits[:1]).kl_mean:.3e}")
    print(f"DECODE per-position KL:  mean={metrics.kl_mean:.4e}  max={metrics.kl_max:.4e}")
    print(f"DECODE per-position top1 agreement: {top1:.4f}")
    print(f"candidate logits finite: {finite}")
    gate = metrics.kl_mean <= 0.05 and top1 >= 0.90
    print(f"GATE (KL<=0.05 & top1>=0.90): {'PASS' if gate else 'FAIL'}")
    if args.json is not None:
        import json
        args.json.write_text(json.dumps({
            "decode_positions": int(ref_dec.shape[0]),
            "prompt_tokens": len(prompt),
            "kl_mean": metrics.kl_mean, "kl_max": metrics.kl_max,
            "top1_agreement": top1, "candidate_finite": finite, "gate_pass": gate,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
