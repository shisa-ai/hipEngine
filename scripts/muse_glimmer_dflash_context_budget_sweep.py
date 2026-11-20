#!/usr/bin/env python3
"""Muse Glimmer DFlash candidate_budget sweep at context on identical KV (diagnostic).

Complements ``muse_glimmer_dflash_budget_sweep.py`` (short natural mtpbench suite)
with a long-context sweep: one coding-agent prompt per context, the prompt KV
(target + drafter) snapshotted once, then every budget decoded from the same
resident prompt KV via the prefix cache. DFlash is exact, so all budgets emit the
same IDs; ``exact`` checks them against an AR reference from the same KV.

The prompt is the repo-source corpus + a fixed instruction at every context
(``--short-natural`` switches contexts <=600 to a natural merge-intervals prompt,
matching muse_glimmer_context_profile.py; note that prompt has higher acceptance,
so do not compare its budget curve against the corpus curve as a context effect).

Re-run after changing draft/verify cost: the best budget moves with the cycle
cost and the acceptance rate. Loads each snapshot once; subsequent budgets reuse
the resident prompt KV (no per-budget disk reload).

  .venv/bin/python scripts/muse_glimmer_dflash_context_budget_sweep.py \
      --contexts 2048,32768,126000 --budgets 2,3,4,5,6,7 \
      --context-dir /tmp/glimmer-ctx --output /tmp/budget_sweep.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
MODEL_DIR = Path("/models/gguf/Muse-Glimmer-30B-GGUF")
INSTR = (
    "\n\n---\nAbove is source code from the hipEngine repository. Task: write a new "
    "Python module `hipengine/tools/kv_report.py` with a function "
    "`summarize_kv(session)` that reports per-layer KV-cache bytes, live length and "
    "SWA vs full layers for a MuseGlimmerResidentSession, plus pytest tests. Return only code."
)


def corpus() -> str:
    parts = []
    for p in sorted((REPO / "hipengine").rglob("*.py")):
        parts.append(f"\n# ===== {p.relative_to(REPO)} =====\n" + p.read_text(errors="ignore"))
    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contexts", default="2048,32768,126000")
    ap.add_argument("--budgets", default="2,3,4,5,6,7")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-context", type=int, default=131072)
    ap.add_argument("--short-natural", action="store_true",
                    help="use a natural merge-intervals prompt for contexts <=600 (higher acceptance)")
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--context-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)

    from hipengine.generation.muse_glimmer_dflash import make_muse_glimmer_dflash_text_provider
    from hipengine.generation.muse_glimmer_gguf import make_muse_glimmer_gguf_generator_gfx1151
    from hipengine.generation.registry import GenerationRequest
    from hipengine.loading.gguf import scan_gguf
    from hipengine.models.muse_glimmer import MUSE_GLIMMER_GGUF
    from hipengine.speculative.registry import SpeculativeProviderConfig

    model = a.model_dir / "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
    gen = make_muse_glimmer_gguf_generator_gfx1151(
        model_path=model, weight_index=scan_gguf(model), model_plugin=MUSE_GLIMMER_GGUF,
        max_sequence_length=a.max_context)
    prov = make_muse_glimmer_dflash_text_provider(target_generator=gen, config=SpeculativeProviderConfig(
        provider="dflash", draft_model=a.model_dir / "dflash-Muse-Glimmer-30B-Q4_K_M.gguf",
        candidate_budget=4))
    gen.attach_speculative_provider(prov)

    instr_ids = list(gen.tokenize(INSTR))[1:]
    body_ids = list(gen.tokenize(corpus()))

    def ids_for(n: int) -> tuple[int, ...]:
        if a.short_natural and n <= 600:
            return tuple(gen.tokenize(
                "Write a Python function merge_intervals(intervals) that merges "
                "overlapping closed integer intervals. Include a compact pytest-style "
                "test block. Return only code."))
        need = max(1, n - len(instr_ids))
        body = (body_ids * (need // len(body_ids) + 1))[:need]
        return tuple(body + instr_ids)

    def req(ids, mt):
        return GenerationRequest(prompts=(ids,), max_tokens=mt, temperature=0.0,
                                 top_p=1.0, ignore_eos=True)

    w = ids_for(512)
    gen.generate_detailed(req(w, 16))
    gen.generate_speculative_detailed(req(w, 16))

    a.context_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    ar_rates: dict[str, float] = {}
    for n in [int(x) for x in a.contexts.split(",")]:
        ids = ids_for(n)
        digest = hashlib.sha1(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()[:12]
        snap = a.context_dir / f"ctx{len(ids)}-{digest}"
        if not (snap / "context.json").exists():
            t = time.perf_counter()
            gen.generate_detailed(req(ids, 1))
            gen.save_context(snap)
            print(f"[{n}] built+saved snapshot in {time.perf_counter()-t:.1f}s", flush=True)
        # Load once; the resident prefix cache reuses the identical prompt KV for
        # the AR reference and every budget (no per-budget disk reload).
        gen.load_context(snap)
        out_ar = gen.generate_detailed(req(ids, a.max_tokens))[0]
        ar_ids = tuple(out_ar.generated_token_ids)
        ar_rates[str(n)] = len(ar_ids) / gen.last_decode_seconds
        print(f"[{n}] AR {ar_rates[str(n)]:.2f} tok/s ({len(ar_ids)} toks)", flush=True)
        for B in [int(x) for x in a.budgets.split(",")]:
            prov.candidate_budget = B
            prov.counters.__init__()
            t = time.perf_counter()
            out = gen.generate_speculative_detailed(req(ids, a.max_tokens))[0]
            wall = time.perf_counter() - t
            c = prov.counters
            dec = prov.last_decode_seconds
            toks = len(out.generated_token_ids)
            cyc = max(c.target_forwards - 1, 1)
            row = {"ctx": n, "prompt_tokens": len(ids), "budget": B,
                   "decode_tok_s": toks / dec, "acceptance": c.accepted / max(c.proposed, 1),
                   "tokens_per_cycle": c.committed / cyc, "ms_per_cycle": 1e3 * dec / cyc,
                   "exact": tuple(out.generated_token_ids) == ar_ids,
                   "tokens": toks, "decode_s": dec, "wall_s": wall,
                   "proposed": c.proposed, "accepted": c.accepted, "cycles": cyc}
            rows.append(row)
            a.output.write_text(json.dumps({"ar_tok_s_by_ctx": ar_rates, "rows": rows}, indent=2))
            print(f"[{n}] budget={B}: {row['decode_tok_s']:.2f} tok/s accept={row['acceptance']:.3f} "
                  f"tok/cyc={row['tokens_per_cycle']:.2f} ms/cyc={row['ms_per_cycle']:.1f} "
                  f"exact={row['exact']} wall={wall:.1f}s", flush=True)
    gen.close()
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())