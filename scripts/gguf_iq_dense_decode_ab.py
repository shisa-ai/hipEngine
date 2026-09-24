#!/usr/bin/env python3
"""Same-window A/B for the dense raw-IQ decode policy on one backend.

The dense-IQ row regimes are backend-declared policy tables, so changing one
changes production arithmetic on the default path. Comparing a run before the
declaration against a run after it is a cross-window comparison, which cannot
separate the change from thermal and allocator drift on the same host.

This driver loads the model once and runs the published decode protocol
(``scripts/gguf_decode_graph_rocprof_driver.py``: 512-token prefill, 4 eager
warm steps, graph capture, 8 warm replays, a 0.5 s GPU idle gap, then
``--steps`` measured replays) in both arms inside one process, alternating
arms so drift lands on both. The incumbent arm is the shipped policy minus the
quants under test - with ``--gate-quants`` set to every declared quant that is
the all-strict owner the backend used before it declared a policy at all.

The decode graph bakes in whichever owner is live at capture time, so each arm
captures its own graph; the owner assertion fails loudly if the two arms
resolve to the same kernel, which is the failure mode that turns an A/B into
two measurements of one thing.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _compiler_version(explicit: Path | None) -> str | None:
    if explicit is not None:
        return explicit.read_text().strip()
    from_env = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE", "")
    return Path(from_env).read_text().strip() if from_env else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("--backend", type=str, default=None,
                    help="defaults to the detected backend")
    ap.add_argument("--gate-quants", type=str, default=None,
                    help="quants whose route is the change under test; "
                         "defaults to every quant the shipped policy declares")
    ap.add_argument("--prefill-tokens", type=int, default=512)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--repetitions", type=int, default=2,
                    help="measured replays per arm; arms alternate")
    ap.add_argument("--max-sequence-length", type=int, default=1024)
    ap.add_argument("--compiler-version-file", type=Path, default=None)
    ap.add_argument("--allow-build", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    from hipengine.kernels.backends import resolve_backend
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    backend = args.backend or resolve_backend(None, warn=False)
    be = __import__(f"hipengine.kernels.{backend}", fromlist=["_"])

    candidate_policy = {
        q: dict(entry) for q, entry in be.GGUF_IQ_DENSE_DECODE_POLICY.items()}
    gated = (
        [q.strip() for q in args.gate_quants.split(",") if q.strip()]
        if args.gate_quants
        else sorted(candidate_policy)
    )
    base_policy = {q: v for q, v in candidate_policy.items() if q not in gated}

    # The owner assertion: both arms must resolve a gated quant to different
    # kernels, or the A/B is measuring one arm twice.
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_source_mmq_prefill import (
        iq_dense_mmq_session,
    )
    from hipengine.kernels.registry import KernelKey
    from hipengine.runtime import gguf_linear as gl
    from hipengine.runtime.gguf_linear import (
        GGUFLinearDispatch,
        _iq_dense_decode_dispatch,
    )

    saved = be.GGUF_IQ_DENSE_DECODE_POLICY
    try:
        with iq_dense_mmq_session(True):
            for quant in gated:
                probe = GGUFLinearDispatch(
                    KernelKey(backend, "linear", quant, "gemv_bf16_bf16_out"), "raw")
                be.GGUF_IQ_DENSE_DECODE_POLICY = base_policy
                incumbent = _iq_dense_decode_dispatch(
                    probe, rows=1, out_features=17408).key.variant
                be.GGUF_IQ_DENSE_DECODE_POLICY = candidate_policy
                candidate = _iq_dense_decode_dispatch(
                    probe, rows=1, out_features=17408).key.variant
                if incumbent == candidate:
                    raise SystemExit(
                        f"arms resolve {quant} to the same decode owner "
                        f"({incumbent}); the A/B would measure one arm twice")
    finally:
        be.GGUF_IQ_DENSE_DECODE_POLICY = saved

    compiler_version = _compiler_version(args.compiler_version_file)
    rng = np.random.default_rng(7)
    ids = [int(t) for t in rng.integers(1000, 50000, args.prefill_tokens)]

    def measure(policy: dict, session) -> dict:
        # The policy table is not part of the dispatch-memo key (it is an
        # import-time constant in production), so dropping the memo is what
        # makes the arm real; the resolve spy below proves the kernels differ.
        be.GGUF_IQ_DENSE_DECODE_POLICY = policy
        gl.clear_gguf_linear_dispatch_cache()
        resolved: set[str] = set()
        real_resolve = gl.resolve

        def _spy(*, backend, layer, quant, variant):
            if str(quant).startswith("gguf_iq"):
                resolved.add(str(variant))
            return real_resolve(backend=backend, layer=layer, quant=quant,
                                variant=variant)

        gl.resolve = _spy
        try:
            session.reset()
            t0 = time.perf_counter()
            cur = session.prefill(ids, use_bulk=True, bulk_attention_mode="bulk")
            session.runner.runtime.stream_synchronize(0)
            prefill_s = time.perf_counter() - t0
            for _ in range(4):
                cur = session.step(int(cur.token_id))
            graph = session.capture_decode_graph(
                position=session.position,
                steps_per_replay=1,
                max_replay_steps=args.steps + 8,
                record_steps=0,
            )
            graph.replay(8)
            session.runner.runtime.stream_synchronize(0)
            time.sleep(0.5)  # the published protocol's GPU idle gap
            t1 = time.perf_counter()
            graph.replay(args.steps)
            session.runner.runtime.stream_synchronize(0)
            wall = time.perf_counter() - t1
            graph.close()
        finally:
            gl.resolve = real_resolve
        return {
            "prefill_s": prefill_s,
            "prefill_tok_s": args.prefill_tokens / prefill_s,
            "wall_s": wall,
            "ms_per_token": 1000.0 * wall / args.steps,
            "tok_s": args.steps / wall,
            "dense_iq_owners": sorted(resolved),
        }

    arms = {"incumbent": [], "candidate": []}
    with Qwen35GGUFResidentSession(
        args.model,
        compiler_version=compiler_version,
        require_cached_build=not args.allow_build,
        max_sequence_length=args.max_sequence_length,
        use_wmma_prefill=True,
        use_gemv_decode=True,
        backend=backend,
    ) as session:
        try:
            for rep in range(args.repetitions):
                for name, policy in (("incumbent", base_policy),
                                     ("candidate", candidate_policy)):
                    arms[name].append(measure(policy, session))
                    print(f"  rep{rep} {name}: "
                          f"{arms[name][-1]['tok_s']:.4f} tok/s decode, "
                          f"{arms[name][-1]['prefill_tok_s']:.3f} tok/s prefill, "
                          f"owners={arms[name][-1]['dense_iq_owners']}",
                          flush=True)
        finally:
            be.GGUF_IQ_DENSE_DECODE_POLICY = saved
            gl.clear_gguf_linear_dispatch_cache()

    def spread(rows, key):
        vals = [r[key] for r in rows]
        mean = statistics.fmean(vals)
        cv = (statistics.stdev(vals) / mean * 100.0) if len(vals) > 1 else 0.0
        return mean, cv

    out = {
        "model": str(args.model),
        "backend": backend,
        "gated_quants": gated,
        "repetitions": args.repetitions,
        "steps": args.steps,
        "prefill_tokens": args.prefill_tokens,
        "arms": arms,
    }
    # A silent dispatch-memo hit would make both arms one measurement of the
    # incumbent. Refuse to report a ratio in that case.
    inc_owners = {o for r in arms["incumbent"] for o in r["dense_iq_owners"]}
    cand_owners = {o for r in arms["candidate"] for o in r["dense_iq_owners"]}
    if inc_owners == cand_owners:
        raise SystemExit(
            "both arms launched the same dense-IQ owners "
            f"({sorted(cand_owners)}); the A/B measured one arm twice")
    if "local32_gemv_bf16_bf16_out" not in cand_owners:
        raise SystemExit(
            "candidate arm never launched the local32 owner; it launched "
            f"{sorted(cand_owners)}")
    out["incumbent_owners"] = sorted(inc_owners)
    out["candidate_owners"] = sorted(cand_owners)
    for key in ("tok_s", "prefill_tok_s"):
        inc, inc_cv = spread(arms["incumbent"], key)
        cand, cand_cv = spread(arms["candidate"], key)
        out[f"{key}_incumbent_mean"] = inc
        out[f"{key}_incumbent_cv_pct"] = inc_cv
        out[f"{key}_candidate_mean"] = cand
        out[f"{key}_candidate_cv_pct"] = cand_cv
        out[f"{key}_ratio"] = cand / inc

    print(f"\nincumbent decode  {out['tok_s_incumbent_mean']:.4f} tok/s "
          f"(CV {out['tok_s_incumbent_cv_pct']:.2f}%)")
    print(f"candidate decode  {out['tok_s_candidate_mean']:.4f} tok/s "
          f"(CV {out['tok_s_candidate_cv_pct']:.2f}%)")
    print(f"decode ratio      {out['tok_s_ratio']:.4f}x")
    print(f"incumbent prefill {out['prefill_tok_s_incumbent_mean']:.3f} tok/s")
    print(f"candidate prefill {out['prefill_tok_s_candidate_mean']:.3f} tok/s")
    print(f"prefill ratio     {out['prefill_tok_s_ratio']:.4f}x")
    if args.json is not None:
        args.json.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
