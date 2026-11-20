#!/usr/bin/env python3
"""Muse Glimmer DFlash candidate_budget sweep (diagnostic).

One model load: AR reference on the mtpbench suite, then DFlash at each budget
through the public provider, reporting decode tok/s, acceptance and exactness
(IDs == AR). The provider default is budget 4; re-run after changing draft or
verify cost, since the best budget moves with the cycle cost.

  .venv/bin/python scripts/muse_glimmer_dflash_budget_sweep.py --budgets 3,4,5,6 --output OUT.json
"""
import argparse, json, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from hipengine.generation.muse_glimmer_gguf import make_muse_glimmer_gguf_generator_gfx1151
from hipengine.generation.muse_glimmer_dflash import make_muse_glimmer_dflash_text_provider
from hipengine.generation.registry import GenerationRequest
from hipengine.loading.gguf import scan_gguf
from hipengine.models.muse_glimmer import MUSE_GLIMMER_GGUF
from hipengine.speculative.registry import SpeculativeProviderConfig

TARGET = "/models/gguf/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
DRAFT = "/models/gguf/Muse-Glimmer-30B-GGUF/dflash-Muse-Glimmer-30B-Q4_K_M.gguf"
SUITE = REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
ap = argparse.ArgumentParser()
ap.add_argument("--budgets", default="2,3,4,5,6,8,10,12")
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--max-prompts", type=int, default=6)
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()
prompts = [json.loads(l) for l in SUITE.read_text().splitlines() if l.strip()][:args.max_prompts]
gen = make_muse_glimmer_gguf_generator_gfx1151(model_path=TARGET, weight_index=scan_gguf(TARGET),
    model_plugin=MUSE_GLIMMER_GGUF, max_sequence_length=2048)
provider = make_muse_glimmer_dflash_text_provider(target_generator=gen,
    config=SpeculativeProviderConfig(provider="dflash", draft_model=DRAFT, candidate_budget=15))
gen.attach_speculative_provider(provider)
def req(c): return GenerationRequest(prompts=(c,), max_tokens=args.max_tokens, temperature=0.0, top_p=1.0, ignore_eos=False)
# warmup + AR reference
gen.generate_detailed(req(prompts[0]["messages"][0]["content"]))
gen.generate_speculative_detailed(req(prompts[0]["messages"][0]["content"]))
ar_ids = {}; ar_decode = 0.0; ar_tokens = 0
for p in prompts:
    o = gen.generate_detailed(req(p["messages"][0]["content"]))[0]
    ar_ids[p["id"]] = tuple(o.generated_token_ids)
    ar_decode += gen.last_decode_seconds; ar_tokens += len(o.generated_token_ids)
ar_tps = ar_tokens/ar_decode
results = []
for budget in [int(b) for b in args.budgets.split(",")]:
    provider.candidate_budget = budget
    dec = 0.0; toks = 0; prop = 0; acc = 0; exact_all = True; t0=time.perf_counter()
    for p in prompts:
        provider.counters.__init__()
        o = gen.generate_speculative_detailed(req(p["messages"][0]["content"]))[0]
        dec += provider.last_decode_seconds; toks += len(o.generated_token_ids)
        prop += provider.counters.proposed; acc += provider.counters.accepted
        if tuple(o.generated_token_ids) != ar_ids[p["id"]]: exact_all = False
    tps = toks/dec
    results.append({"budget": budget, "decode_tok_s": tps, "acceptance": acc/max(prop,1),
                    "exact": exact_all, "tokens": toks, "decode_s": dec, "wall_s": time.perf_counter()-t0})
    print(f"budget={budget:2d}: {tps:6.2f} tok/s accept={acc/max(prop,1):.3f} exact={exact_all} ({time.perf_counter()-t0:.0f}s)", flush=True)
gen.close()
best = max((r for r in results if r["exact"]), key=lambda r: r["decode_tok_s"])
print(f"\nAR={ar_tps:.2f} tok/s | BEST exact budget={best['budget']} @ {best['decode_tok_s']:.2f} tok/s | Gate B target 21.30")
args.output.parent.mkdir(parents=True, exist_ok=True)
json.dump({"ar_tok_s": ar_tps, "max_tokens": args.max_tokens, "prompts": len(prompts), "results": results, "best": best}, open(args.output,"w"), indent=2)