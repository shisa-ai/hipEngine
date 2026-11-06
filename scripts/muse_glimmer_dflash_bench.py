#!/usr/bin/env python3
"""hipEngine Muse Glimmer DFlash vs AR mtpbench decode bench (Gate B).

Measures hipEngine DFlash speculative decode and the true no-spec AR control on
the natural-prompt mtpbench suite, under the cross-engine decode boundary
(docs/campaigns/MTP-LLAMACPP-PARITY.md): post-prefill decode wall, output
transitions counted. Reports exact greedy ID equality (DFlash == AR) per prompt
as the anti-gaming correctness gate, plus acceptance economics. The Gate B
target is the llama.cpp draft-dflash rate recorded in
benchmarks/results/2026-11-06-gfx1151-muse-glimmer-llamacpp-dflash-mtpbench-baseline.json
(21.30 tok/s). Diagnostic until a retained rollup attaches the gate.

Timing: the provider/generator record last_prefill_seconds and
last_decode_seconds (synchronized by the per-cycle logits readback). Decode
tok/s = sum(generated tokens) / sum(decode wall). A transition-normalized
variant subtracts one prompt-produced first output per request to mirror the
llama.cpp (predicted_n - 1) / predicted_ms boundary exactly.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DIR = Path("/models/gguf/Muse-Glimmer-30B-GGUF")
DEFAULT_SUITE = REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"


def _load_suite(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_DIR / "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf")
    parser.add_argument("--draft", type=Path, default=DEFAULT_DIR / "dflash-Muse-Glimmer-30B-Q4_K_M.gguf")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-prompts", type=int, default=0, help="0 = all")
    parser.add_argument("--candidate-budget", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    from hipengine.generation.muse_glimmer_dflash import make_muse_glimmer_dflash_text_provider
    from hipengine.generation.muse_glimmer_gguf import make_muse_glimmer_gguf_generator_gfx1151
    from hipengine.generation.registry import GenerationRequest
    from hipengine.loading.gguf import scan_gguf
    from hipengine.models.muse_glimmer import MUSE_GLIMMER_GGUF
    from hipengine.speculative.registry import SpeculativeProviderConfig
    from hipengine.benchmark.provenance import collect_artifact_provenance

    prompts = _load_suite(args.suite)
    if args.max_prompts > 0:
        prompts = prompts[:args.max_prompts]

    gen = make_muse_glimmer_gguf_generator_gfx1151(
        model_path=args.model, weight_index=scan_gguf(args.model),
        model_plugin=MUSE_GLIMMER_GGUF, max_sequence_length=args.max_context,
    )
    provider = make_muse_glimmer_dflash_text_provider(
        target_generator=gen,
        config=SpeculativeProviderConfig(
            provider="dflash", draft_model=args.draft, candidate_budget=args.candidate_budget,
        ),
    )
    gen.attach_speculative_provider(provider)

    def request(content: str) -> GenerationRequest:
        return GenerationRequest(
            prompts=(content,), max_tokens=args.max_tokens, temperature=0.0,
            top_p=1.0, ignore_eos=False,
        )

    # Warmup: prebuild kernels on both paths (never timed).
    warm = prompts[0]["messages"][0]["content"]
    gen.generate_detailed(request(warm))
    gen.generate_speculative_detailed(request(warm))

    rows: list[dict[str, Any]] = []
    try:
        for rep in range(args.repeats):
            for prompt in prompts:
                content = prompt["messages"][0]["content"]
                ar = gen.generate_detailed(request(content))[0]
                ar_decode = gen.last_decode_seconds
                ar_prefill = gen.last_prefill_seconds
                ar_ids = tuple(ar.generated_token_ids)

                provider.counters.__init__()
                spec = gen.generate_speculative_detailed(request(content))[0]
                spec_decode = provider.last_decode_seconds
                spec_prefill = provider.last_prefill_seconds
                spec_ids = tuple(spec.generated_token_ids)
                c = provider.counters
                exact = spec_ids == ar_ids
                rows.append({
                    "repeat": rep,
                    "id": prompt["id"],
                    "category": prompt.get("category", "uncategorized"),
                    "prompt_tokens": ar.prompt_tokens,
                    "ar_generated": len(ar_ids),
                    "spec_generated": len(spec_ids),
                    "ar_prefill_s": ar_prefill,
                    "ar_decode_s": ar_decode,
                    "spec_prefill_s": spec_prefill,
                    "spec_decode_s": spec_decode,
                    "exact": exact,
                    "proposed": c.proposed,
                    "accepted": c.accepted,
                    "committed": c.committed,
                    "target_forwards": c.target_forwards,
                    "draft_forwards": c.draft_forwards,
                    "acceptance": (c.accepted / c.proposed) if c.proposed else None,
                })
                print(f"[rep {rep}] {prompt['id']}: exact={exact} "
                      f"AR {len(ar_ids)/ar_decode:.2f} tok/s, "
                      f"DFlash {len(spec_ids)/spec_decode:.2f} tok/s, "
                      f"accept={c.accepted}/{c.proposed}", flush=True)
                if not exact:
                    print(f"  DIVERGENCE ar={ar_ids[:16]} spec={spec_ids[:16]}", flush=True)
    finally:
        gen.close()

    def agg(gen_key: str, decode_key: str) -> dict[str, Any]:
        n = sum(r[gen_key] for r in rows)
        wall = sum(r[decode_key] for r in rows)
        requests = len(rows)
        return {
            "generated_tokens": n,
            "decode_wall_s": wall,
            "decode_tok_s": (n / wall) if wall else None,
            "transition_normalized_tok_s": ((n - requests) / wall) if wall else None,
            "per_request_decode_tok_s_median": statistics.median(
                [r[gen_key] / r[decode_key] for r in rows if r[decode_key]]
            ) if rows else None,
        }

    ar_summary = agg("ar_generated", "ar_decode_s")
    spec_summary = agg("spec_generated", "spec_decode_s")
    exact_all = all(r["exact"] for r in rows)
    proposed = sum(r["proposed"] for r in rows)
    accepted = sum(r["accepted"] for r in rows)
    target_forwards = sum(r["target_forwards"] for r in rows)
    committed = sum(r["committed"] for r in rows)

    prov = collect_artifact_provenance(
        repo_root=REPO_ROOT, configured_backend="hip_gfx1151", resolved_backend="hip_gfx1151",
        target_arch="gfx1151", model_path=str(args.model), quant="gguf_q4_k_m", kv_dtype="f16/f16",
        command=[str(Path(__file__).relative_to(REPO_ROOT)), *sys.argv[1:]],
        timing_protocol="post-prefill decode wall; output transitions; per-cycle logits readback sync",
        repetitions=args.repeats,
    )
    artifact = {
        "schema": 1,
        "kind": "muse_glimmer_hipengine_dflash_mtpbench",
        "status": "diagnostic",
        "performance_claim": False,
        "date": time.strftime("%Y-%m-%d"),
        "hardware": "Radeon 8060S / gfx1151 (Ryzen AI MAX+ PRO 395, 128 GiB unified)",
        "model_path": str(args.model),
        "draft_path": str(args.draft),
        "suite": str(args.suite),
        "max_tokens": args.max_tokens,
        "candidate_budget": args.candidate_budget,
        "repeats": args.repeats,
        "provenance": prov,
        "gate_b_target_tok_s": 21.300,
        "exactness_all": exact_all,
        "ar_summary": ar_summary,
        "dflash_summary": spec_summary,
        "speedup_vs_ar": (spec_summary["decode_tok_s"] / ar_summary["decode_tok_s"])
            if ar_summary["decode_tok_s"] else None,
        "beats_gate_b": (spec_summary["transition_normalized_tok_s"] or 0) > 21.300,
        "economics": {
            "proposed": proposed, "accepted": accepted,
            "acceptance": (accepted / proposed) if proposed else None,
            "committed": committed, "target_forwards": target_forwards,
            "tokens_per_target_forward": (committed / target_forwards) if target_forwards else None,
        },
        "rows": rows,
        "timing_boundary": {
            "decode_tok_s": "sum(generated tokens) / sum(post-prefill decode wall)",
            "transition_normalized_tok_s": "(sum(generated) - requests) / sum(decode wall); mirrors llama.cpp (predicted_n-1)/predicted_ms",
            "sync": "each verify/decode cycle ends in a device-to-host logits readback",
        },
        "notes": [
            "Gate B: hipEngine DFlash must beat llama.cpp draft-dflash 21.300 tok/s "
            "(transition-normalized) with exact greedy ID equality vs true AR.",
            "Diagnostic until a retained rollup attaches the correctness gate + provenance.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nexactness_all={exact_all}")
    print(f"AR decode {ar_summary['decode_tok_s']:.3f} tok/s | "
          f"DFlash decode {spec_summary['decode_tok_s']:.3f} tok/s "
          f"(transition-norm {spec_summary['transition_normalized_tok_s']:.3f}) | "
          f"speedup vs AR {artifact['speedup_vs_ar']:.3f}x | beats Gate B: {artifact['beats_gate_b']}")
    print(f"acceptance {accepted}/{proposed} = {accepted/max(proposed,1):.1%}, "
          f"tokens/target-forward {committed/max(target_forwards,1):.2f}")
    print(f"wrote {args.output}")
    return 0 if exact_all else 1


if __name__ == "__main__":
    raise SystemExit(main())