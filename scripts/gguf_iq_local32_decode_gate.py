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
    ap.add_argument(
        "--gate-quants", type=str,
        default="gguf_iq3_s,gguf_iq3_xxs,gguf_iq2_s,gguf_iq2_xs",
        help="quants under test: the candidate is the shipped policy, the "
             "incumbent is the shipped policy minus these")
    ap.add_argument(
        "--backend", type=str, default="hip_gfx1100",
        help="backend package whose decode policy is gated. The default keeps "
             "the original gfx1100 arm pair; pass hip_gfx1151 to gate that "
             "backend's declaration against its own all-strict incumbent.")
    args = ap.parse_args()
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    from hipengine.benchmark.correctness import evaluate_logits
    from hipengine.runtime import gguf_linear as gl
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    backend = args.backend
    be = __import__(f"hipengine.kernels.{backend}", fromlist=["_"])

    compiler_version = (Path(args.compiler_version_file).read_text()
                        if args.compiler_version_file else None)
    rng = np.random.default_rng(args.seed)
    # The candidate is the shipped policy; the incumbent is the shipped
    # policy minus the quants under test. The natural prompt is generated
    # under the INCUMBENT so the natural text - and the probe itself - is
    # stable regardless of which routes the candidate adds.
    ap_gated = [q.strip() for q in args.gate_quants.split(",") if q.strip()]
    candidate_policy = {
        q: dict(entry) for q, entry in be.GGUF_IQ_DENSE_DECODE_POLICY.items()}
    base_policy = {q: dict(entry) for q, entry in candidate_policy.items()
                  if q not in ap_gated}
    be.GGUF_IQ_DENSE_DECODE_POLICY = base_policy
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
            use_wmma_prefill=True, use_gemv_decode=True, backend=backend,
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

    # The arms derive from the shipped policy (2026-09-11): the candidate is
    # the shipped decode policy itself; the incumbent is the shipped policy
    # minus the quants under test. Copying a stale hardcoded pair instead
    # compared all-strict against an XS-only candidate - arms that no longer
    # match any production boundary (and whose all-strict incumbent has a
    # known NaN fragility on rare self-generated prompts). The owner
    # assertion fails loudly if the two arms ever resolve to the same
    # kernel for a gated quant.
    def _assert_owners_differ():
        from hipengine.kernels.backends import load_backend_kernel_package
        from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq
        from hipengine.runtime.gguf_linear import (
            GGUFLinearDispatch, _iq_dense_decode_dispatch)
        from hipengine.kernels.registry import KernelKey
        load_backend_kernel_package(backend)
        with iq_mmq.iq_dense_mmq_session(True):
            saved = dict(be.GGUF_IQ_DENSE_DECODE_POLICY)
            try:
                for gated in ap_gated:
                    base = GGUFLinearDispatch(
                        KernelKey(backend, "linear", gated,
                                 "gemv_bf16_bf16_out"), "raw")
                    be.GGUF_IQ_DENSE_DECODE_POLICY = base_policy
                    incumbent_owner = _iq_dense_decode_dispatch(
                        base, rows=1, out_features=17408).key.variant
                    be.GGUF_IQ_DENSE_DECODE_POLICY = candidate_policy
                    candidate_owner = _iq_dense_decode_dispatch(
                        base, rows=1, out_features=17408).key.variant
                    if incumbent_owner == candidate_owner:
                        raise RuntimeError(
                            "gate arms resolve to the same decode owner "
                            f"({incumbent_owner}) for {gated}; the incumbent "
                            "pin has gone stale")
            finally:
                be.GGUF_IQ_DENSE_DECODE_POLICY = saved

    _assert_owners_differ()

    def run(local32: bool, forced_tokens=None):
        # Flipping the policy table changes a resolution input the dispatch
        # memo does not key on (it is an import-time constant in production),
        # so the memo must be dropped or the second arm reuses the first
        # arm's owner and the two arms measure the same kernels. The spy below
        # then proves at the real launch site that they did not.
        be.GGUF_IQ_DENSE_DECODE_POLICY = candidate_policy if local32 else base_policy
        gl.clear_gguf_linear_dispatch_cache()
        resolved: set[str] = set()
        real_resolve = gl.resolve

        def _spy(*, backend, layer, quant, variant):
            if str(quant).startswith("gguf_iq"):
                resolved.add(str(variant))
            return real_resolve(backend=backend, layer=layer, quant=quant,
                                variant=variant)

        gl.resolve = _spy
        logits_rows = []
        tokens = []
        try:
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
        finally:
            gl.resolve = real_resolve
        return np.vstack(logits_rows), tokens, frozenset(resolved)

    try:
        with Qwen35GGUFResidentSession(
            args.model,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
            max_sequence_length=len(prompt) + n + 2,
            use_wmma_prefill=True,
            use_gemv_decode=True,
            backend=backend,
        ) as session:
            # Incumbent: local32 disabled, eager -> fixes the trajectory.
            ref_logits, ref_tokens, incumbent_owners = run(local32=False)
            session.reset()
            # Candidate: local32 enabled, teacher-forced on the same tokens.
            cand_logits, _, candidate_owners = run(
                local32=True, forced_tokens=ref_tokens[:-1])
    finally:
        be.GGUF_IQ_DENSE_DECODE_POLICY = base_policy
        gl.clear_gguf_linear_dispatch_cache()

    # The arms must have launched different kernels. Without this the probe
    # reports a clean PASS whenever a dispatch-memo hit collapses the two arms
    # into one, which is exactly what an un-cleared cache does.
    if incumbent_owners == candidate_owners:
        raise SystemExit(
            "gate arms launched the same dense-IQ owners "
            f"({sorted(candidate_owners)}); the probe measured one arm twice")
    expected = {"local32_gemv_bf16_bf16_out"}
    if not expected <= candidate_owners:
        raise SystemExit(
            f"candidate arm did not launch {sorted(expected)}; it launched "
            f"{sorted(candidate_owners)}")

    ref_dec, cand_dec = ref_logits[1:], cand_logits[1:]
    metrics = evaluate_logits(ref_dec, cand_dec)
    top1 = float(np.mean(np.argmax(ref_dec, -1) == np.argmax(cand_dec, -1)))
    finite = bool(np.all(np.isfinite(cand_logits)))
    pre = evaluate_logits(ref_logits[:1], cand_logits[:1])
    print(f"teacher-forced decode positions: {ref_dec.shape[0]}  (prompt={len(prompt)})")
    print(f"incumbent dense-IQ owners: {sorted(incumbent_owners)}")
    print(f"candidate dense-IQ owners: {sorted(candidate_owners)}")
    print(f"decode logits bit-identical across arms: "
          f"{bool(np.array_equal(ref_dec, cand_dec))}")
    print(f"prefill position KL (route-independent sanity): {pre.kl_mean:.3e}")
    print(f"DECODE per-position KL:  mean={metrics.kl_mean:.4e}  max={metrics.kl_max:.4e}")
    print(f"DECODE per-position top1 agreement: {top1:.4f}")
    print(f"candidate logits finite: {finite}")
    # The complete section-6.1 screen: the calibrated mean (<= 1e-3), the
    # absolute per-position ceiling (max <= 5e-2), top-1 (>= 0.99), finite
    # candidate logits, and the prefill sanity check (the decode policy must
    # not perturb the prefill position: its KL is identically zero by
    # construction, so any nonzero value means the arms are contaminated).
    prefill_clean = pre.kl_mean == 0.0
    gate = (
        metrics.kl_mean <= 1e-3
        and metrics.kl_max <= 5e-2
        and top1 >= 0.99
        and finite
        and prefill_clean
    )
    print(
        "PROBE GATE (mean<=1e-3 & max<=5e-2 & top1>=0.99 & finite & "
        f"prefill-clean): {'PASS' if gate else 'FAIL'}"
    )
    if args.json is not None:
        args.json.write_text(json.dumps({
            "backend": backend,
            "gate_quants": ap_gated,
            "incumbent_owners": sorted(incumbent_owners),
            "candidate_owners": sorted(candidate_owners),
            "decode_logits_bit_identical": bool(np.array_equal(ref_dec, cand_dec)),
            "decode_positions": int(ref_dec.shape[0]),
            "prompt_tokens": len(prompt),
            "kl_mean": metrics.kl_mean, "kl_max": metrics.kl_max,
            "top1_agreement": top1, "candidate_finite": finite,
            "prefill_kl_sanity": pre.kl_mean, "prefill_clean": prefill_clean,
            "probe_gate_pass": gate,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
