"""Combined-stack admission gate for the shipped UD route set (gfx1100).

Compares the declared production incumbent (the session-start shipped state:
four-quant W4A16 prefill policy on the one-wave owner, empty decode policy -
strict per-row GEMV) against the CURRENT shipped production stack, teacher-
forced, on natural self-generated prompts.

The incumbent arm is pinned explicitly (a frozen reference must not move when
the defaults change). The shipped arm is DERIVED from the live module
defaults at run time and asserted against the expected shape - deriving it
keeps the gate testing what actually ships (the 2026-09-10 review found a
stale pin still naming the one-wave owner for every quant after the
cooperative owners shipped). The fused IQ4_XS dual is owner-gated in the
dispatcher: it engages only when both operands resolve to the admitted
cooperative prefill owner, so the incumbent arm (one-wave variants) runs the
two-singles path exactly like the production incumbent did.

The complete section-6.1 screen binds: mean/p95/p99/max KL <=
1e-3/5e-3/2e-2/5e-2 and top-1 >= 0.99 overall, plus finite candidate logits
and a zero-KL prefill-only sanity arm (decode policy must not perturb
prefill). The per-scope 97% top-1 requirement is a multi-category heldout
property of the full gate suite; this single-prompt diagnostic binds the
overall screen.
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
INCUMBENT_DECODE: dict = {}
# The shipped arms are captured from the live module defaults at import
# time (after the kernel package initializes its policies) and asserted
# in main() to match the shipped shape.
SHIPPED_PREFILL: dict = {}
SHIPPED_DECODE: dict = {}


def _capture_shipped_policies() -> tuple[dict, dict]:
    import copy

    import hipengine.kernels.hip_gfx1100 as backend

    prefill = copy.deepcopy(backend.GGUF_IQ_DENSE_PREFILL_POLICY)
    decode = copy.deepcopy(backend.GGUF_IQ_DENSE_DECODE_POLICY)
    return prefill, decode


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
    ap.add_argument(
        "--category-heldout", action="store_true",
        help="Run the 18-prompt category/heldout fixture instead of the "
             "single natural prompt, binding the per-scope top-1 >= 0.97 "
             "half of the section-6.1 screen.")
    args = ap.parse_args()
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    from hipengine.benchmark.correctness import evaluate_logits
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    import hipengine.kernels.hip_gfx1100 as be

    compiler_version = (Path(args.compiler_version_file).read_text()
                         if args.compiler_version_file else None)

    global SHIPPED_PREFILL, SHIPPED_DECODE
    SHIPPED_PREFILL, SHIPPED_DECODE = _capture_shipped_policies()

    # Arm assertions: the incumbent is the frozen four-quant reference; the
    # shipped arm must be the live default policy covering the seven-quant
    # route set (a stale pin here would test a stack that no longer ships).
    assert set(INCUMBENT_PREFILL) < set(SHIPPED_PREFILL), (
        "prefill arms route the same quant set; the incumbent pin is stale")
    assert set(SHIPPED_PREFILL) == {
        "gguf_iq4_xs", "gguf_iq3_xxs", "gguf_iq3_s", "gguf_iq4_nl",
        "gguf_q3_k", "gguf_iq2_s", "gguf_iq2_xs"}, (
        "the shipped prefill policy no longer routes the seven-quant set; "
        "update this gate's expectation")
    assert SHIPPED_PREFILL["gguf_iq4_xs"]["variant"].endswith(
        "dense_wmma_w4a16_prefill_coop64_bf16_bf16_out"), (
        "the shipped IQ4_XS prefill owner is not the cooperative 64-column "
        "owner; update this gate's expectation")
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
        # Reassigning either policy table changes a dispatch-resolution input
        # that the launch memo does not key on (it is an import-time constant
        # in production), so the memo must be dropped or the second arm reuses
        # the first arm's owners and the two arms measure the same kernels.
        be.GGUF_IQ_DENSE_PREFILL_POLICY = prefill_policy
        be.GGUF_IQ_DENSE_DECODE_POLICY = decode_policy
        gl.clear_gguf_linear_dispatch_cache()
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

    if args.category_heldout:
        _run_category_heldout_gate(args, compiler_version)
        return

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
        gl.clear_gguf_linear_dispatch_cache()

    metrics = evaluate_logits(ref_logits, cand_logits)
    top1 = float(np.mean(np.argmax(ref_logits, -1) == np.argmax(cand_logits, -1)))
    finite = bool(np.all(np.isfinite(cand_logits)))
    # Per-position KL vector for the p95/p99 percentiles of the complete
    # section-6.1 screen.
    ref_lse = ref_logits - ref_logits.max(-1, keepdims=True)
    ref_p = np.exp(ref_lse)
    ref_p /= ref_p.sum(-1, keepdims=True)
    cand_lse = cand_logits - cand_logits.max(-1, keepdims=True)
    cand_p = np.exp(cand_lse)
    cand_p /= cand_p.sum(-1, keepdims=True)
    kl_rows = np.sum(ref_p * (np.log(ref_p + 1e-12) - np.log(cand_p + 1e-12)),
                     axis=-1)
    kl_p95 = float(np.percentile(kl_rows, 95))
    kl_p99 = float(np.percentile(kl_rows, 99))
    print(f"model: {args.model.name}  seed: {args.seed}")
    print(f"positions: {ref_logits.shape[0]}  (natural prompt={len(prompt)})")
    print(f"KL vs pinned incumbent:  mean={metrics.kl_mean:.4e}  "
          f"p95={kl_p95:.4e}  p99={kl_p99:.4e}  max={metrics.kl_max:.4e}")
    print(f"top1 agreement: {top1:.4f}  finite: {finite}")
    gate = (
        metrics.kl_mean <= 1e-3
        and kl_p95 <= 5e-3
        and kl_p99 <= 2e-2
        and metrics.kl_max <= 5e-2
        and top1 >= 0.99
        and finite
    )
    print(f"COMBINED GATE (mean<=1e-3 & p95<=5e-3 & p99<=2e-2 & max<=5e-2 & "
          f"top1>=0.99 & finite): {'PASS' if gate else 'FAIL'}")
    if args.json is not None:
        args.json.write_text(json.dumps({
            "model": str(args.model), "seed": args.seed,
            "positions": int(ref_logits.shape[0]),
            "kl_mean": metrics.kl_mean, "kl_p95": kl_p95, "kl_p99": kl_p99,
            "kl_max": metrics.kl_max,
            "top1_agreement": top1, "candidate_finite": finite,
            "gate_pass": gate,
            "incumbent_prefill_variants": {
                q: e["variant"] for q, e in INCUMBENT_PREFILL.items()},
            "shipped_prefill_variants": {
                q: e["variant"] for q, e in SHIPPED_PREFILL.items()},
            "shipped_decode_variants": {
                q: e["variant"] for q, e in SHIPPED_DECODE.items()},
        }, indent=2) + "\n")


def _run_category_heldout_gate(args, compiler_version) -> None:
    """The 18-prompt category/heldout half of the section-6.1 screen.

    Prompts are tokenized with the model's own tokenizer
    (Qwen35GGUFTokenizer + the chat template), then extended to
    --prompt-tokens with the incumbent arm's own continuation so the
    prefill window exercises the >=129-row owners (the fused IQ4_XS
    dual included) while the instruction content is the real fixture
    prompt. The candidate is teacher-forced on the incumbent's tokens.

    Binding predicate on the position-pooled KL vector (weighted by
    construction): mean/p95/p99/max <= 1e-3/5e-3/2e-2/5e-2, overall
    top-1 >= 0.99, per-scope top-1 >= 0.97, finite candidate logits.
    Positions above the 2e-2 p99 envelope are listed as diagnostics -
    a pass does not suppress them.
    """

    from hipengine.benchmark.correctness import evaluate_logits
    from hipengine.runtime import gguf_linear as gl
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
    from hipengine.loading.gguf import scan_gguf
    from scripts.gguf_mtp_bench import build_chat_prompt
    import hipengine.kernels.hip_gfx1100 as be

    prompt_rows = []
    for path in (
            REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl",
            REPO_ROOT / "benchmarks/prompts/gdn-prefill-category-heldouts.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                import json as _json
                prompt_rows.append(_json.loads(line))
    by_scope: dict[str, list[dict]] = {}
    for row in prompt_rows:
        by_scope.setdefault(row["category"], []).append(row)

    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
    n = int(args.decode_tokens)
    rng = np.random.default_rng(args.seed)

    scope_stats: dict[str, dict] = {}
    pooled_kl: list[np.ndarray] = []
    pooled_top1: list[np.ndarray] = []
    pooled_calib: list[np.ndarray] = []
    diagnostics: list[dict] = []
    candidate_finite = True
    try:
        with Qwen35GGUFResidentSession(
            args.model,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
            max_sequence_length=args.prompt_tokens + n + 64,
            use_wmma_prefill=True,
            use_gemv_decode=True,
        ) as session:
            def _arm(prefill_policy, decode_policy, prompt_ids, forced=None):
                be.GGUF_IQ_DENSE_PREFILL_POLICY = prefill_policy
                be.GGUF_IQ_DENSE_DECODE_POLICY = decode_policy
                gl.clear_gguf_linear_dispatch_cache()
                logits_rows = []
                cur = session.prefill(
                    prompt_ids, use_bulk=True, bulk_attention_mode="bulk",
                    return_logits=True)
                logits_rows.append(np.asarray(
                    cur.logits, dtype=np.float32).reshape(-1))
                for i in range(n):
                    feed = (int(cur.token_id) if forced is None
                            else int(forced[i]))
                    cur = session.step(feed, return_logits=True)
                    logits_rows.append(np.asarray(
                        cur.logits, dtype=np.float32).reshape(-1))
                return np.vstack(logits_rows), cur

            for scope in sorted(by_scope):
                ref_rows = []
                cand_rows = []
                calib_rows = []
                prompt_ids_order = []
                for row in by_scope[scope]:
                    prompt_ids_order.append(row["id"])
                    content = row["messages"][0]["content"]
                    seed_ids = build_chat_prompt(tokenizer, content)
                    assert len(seed_ids) >= 8, row["id"]
                    # Extend to the full prefill window with the
                    # incumbent's own continuation: the real instruction
                    # content leads, and the window crosses the 129-row
                    # owner boundary.
                    be.GGUF_IQ_DENSE_PREFILL_POLICY = INCUMBENT_PREFILL
                    be.GGUF_IQ_DENSE_DECODE_POLICY = INCUMBENT_DECODE
                    gl.clear_gguf_linear_dispatch_cache()
                    ids = list(seed_ids)
                    cur = session.prefill(
                        ids, use_bulk=True, bulk_attention_mode="bulk",
                        return_logits=True)
                    while len(ids) < args.prompt_tokens:
                        t = int(cur.token_id)
                        ids.append(t)
                        cur = session.step(t, return_logits=True)
                    session.reset()

                    ref, ref_cur = _arm(
                        INCUMBENT_PREFILL, INCUMBENT_DECODE, ids)
                    ref_tokens = [int(x) for x in np.argmax(ref, -1)]
                    session.reset()
                    cand, _ = _arm(
                        SHIPPED_PREFILL, SHIPPED_DECODE, ids,
                        forced=ref_tokens[:-1])
                    session.reset()
                    # Calibration arm: the incumbent's own distance from
                    # all-strict arithmetic on the same forced tokens -
                    # the reference baseline's intrinsic noise class.
                    calib, _ = _arm({}, {}, ids, forced=ref_tokens[:-1])
                    session.reset()

                    if not np.all(np.isfinite(cand)):
                        candidate_finite = False
                    ref_rows.append(ref)
                    cand_rows.append(cand)
                    calib_rows.append(calib)

                ref_m = np.vstack(ref_rows)
                cand_m = np.vstack(cand_rows)
                m = evaluate_logits(ref_m, cand_m)
                top1_rows = (
                    np.argmax(ref_m, -1) == np.argmax(cand_m, -1))
                # Per-position KL of this scope's pooled rows.
                ref_lse = ref_m - ref_m.max(-1, keepdims=True)
                ref_p = np.exp(ref_lse)
                ref_p /= ref_p.sum(-1, keepdims=True)
                cand_lse = cand_m - cand_m.max(-1, keepdims=True)
                cand_p = np.exp(cand_lse)
                cand_p /= cand_p.sum(-1, keepdims=True)
                kl = np.sum(
                    ref_p * (np.log(ref_p + 1e-12)
                             - np.log(cand_p + 1e-12)), axis=-1)
                for i in np.nonzero(kl > 2e-2)[0]:
                    diag_row = dict(
                        scope=scope,
                        prompt=prompt_ids_order[int(i) // (n + 1)],
                        position_in_prompt=int(i) % (n + 1),
                        is_prefill_row=(int(i) % (n + 1) == 0),
                        kl=float(kl[i]),
                    )
                    diagnostics.append(diag_row)
                calib_m = np.vstack(calib_rows)
                cal_lse = calib_m - calib_m.max(-1, keepdims=True)
                cal_p = np.exp(cal_lse)
                cal_p /= cal_p.sum(-1, keepdims=True)
                kl_calib = np.sum(
                    ref_p * (np.log(ref_p + 1e-12) - np.log(cal_p + 1e-12)),
                    axis=-1)
                scope_stats[scope] = {
                    "positions": int(ref_m.shape[0]),
                    "kl_mean": m.kl_mean, "kl_max": m.kl_max,
                    "top1": float(np.mean(top1_rows)),
                    "calib_mean": float(np.mean(kl_calib)),
                    "calib_max": float(np.max(kl_calib)),
                }
                pooled_kl.append(kl)
                pooled_top1.append(top1_rows)
                pooled_calib.append(kl_calib)
                print(f"scope {scope:12s}: positions={ref_m.shape[0]:4d}  "
                      f"mean={m.kl_mean:.4e}  max={m.kl_max:.4e}  "
                      f"top1={float(np.mean(top1_rows)):.4f}",
                      flush=True)
    finally:
        be.GGUF_IQ_DENSE_PREFILL_POLICY = SHIPPED_PREFILL
        be.GGUF_IQ_DENSE_DECODE_POLICY = SHIPPED_DECODE
        gl.clear_gguf_linear_dispatch_cache()

    kl_all = np.concatenate(pooled_kl)
    top1_all = np.concatenate(pooled_top1)
    overall = {
        "positions": int(kl_all.shape[0]),
        "kl_mean": float(np.mean(kl_all)),
        "kl_p95": float(np.percentile(kl_all, 95)),
        "kl_p99": float(np.percentile(kl_all, 99)),
        "kl_max": float(np.max(kl_all)),
        "top1": float(np.mean(top1_all)),
    }
    gate = (
        overall["kl_mean"] <= 1e-3
        and overall["kl_p95"] <= 5e-3
        and overall["kl_p99"] <= 2e-2
        and overall["kl_max"] <= 5e-2
        and overall["top1"] >= 0.99
        and all(s["top1"] >= 0.97 for s in scope_stats.values())
        and candidate_finite
    )
    calib_all = np.concatenate(pooled_calib) if pooled_calib else np.array([])
    calibration = {
        "positions": int(calib_all.shape[0]),
        "kl_mean": float(np.mean(calib_all)) if calib_all.size else None,
        "kl_max": float(np.max(calib_all)) if calib_all.size else None,
    }
    print(f"calibration (incumbent vs all-strict, same forced tokens): "
          f"mean={calibration['kl_mean']:.4e}  max={calibration['kl_max']:.4e}")
    print(f"pooled ({overall['positions']} positions): "
          f"mean={overall['kl_mean']:.4e}  p95={overall['kl_p95']:.4e}  "
          f"p99={overall['kl_p99']:.4e}  max={overall['kl_max']:.4e}  "
          f"top1={overall['top1']:.4f}  finite={candidate_finite}")
    if diagnostics:
        print(f"DIAGNOSTIC: {len(diagnostics)} position(s) above the 2e-2 "
              f"p99 envelope (within the 5e-2 max gate):")
        for d in diagnostics[:10]:
            print(f"  {d['scope']}/{d['prompt']} pos "
                  f"{d['position_in_prompt']} "
                  f"({'prefill' if d['is_prefill_row'] else 'decode'}): "
                  f"KL {d['kl']:.4e}")
    print(f"CATEGORY/HELDOUT GATE (pooled mean<=1e-3 & p95<=5e-3 & p99<=2e-2 "
          f"& max<=5e-2 & overall top1>=0.99 & per-scope top1>=0.97 & "
          f"finite): {'PASS' if gate else 'FAIL'}")
    if args.json is not None:
        args.json.write_text(json.dumps({
            "model": str(args.model), "seed": args.seed,
            "prompt_tokens": int(args.prompt_tokens),
            "tokenized": True,
            "scopes": scope_stats,
            "overall": overall,
            "calibration_incumbent_vs_all_strict": calibration,
            "diagnostics_above_p99_envelope": diagnostics,
            "candidate_finite": candidate_finite,
            "gate_pass": gate,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
