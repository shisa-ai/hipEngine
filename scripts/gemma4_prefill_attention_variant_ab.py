#!/usr/bin/env python3
"""A/B two registered Gemma 4 prefill-attention variants at one shape.

The execution profile selects a variant as a *request*, and the profile's choice
is what ships. Comparing two of them at the same shape on the same load is
therefore the only way to price a variant that is registered but not selected —
which is the case for the WMMA prefill candidates: they are much faster at depth
and are not shipped, because saved teacher-chain KL failures keep them out of the
production arithmetic choice (`hipengine/generation/gemma4_gguf_profiles.py`).

This sets ``prefill_attention_variants`` on the generator and the runner directly
rather than editing the profile, so it measures the kernels and changes no
shipped default. After every measurement it reads
``gemma4_layer.last_prefill_attention_route()`` and records it: the request is a
request, and only the layer knows which variant the geometry admitted, so an arm
that silently fell back to the strict kernel would otherwise be reported as the
candidate's speed.

Prefill only. The variant is a prefill-attention choice; decoding is unaffected,
and a one-token output keeps the shape cheap while timing the same prefill work.

Usage:
    .venv/bin/python scripts/gemma4_prefill_attention_variant_ab.py \
        --prompt 65536 --repeats 1 --warmup 1 \
        --out /tmp/gemma4-ab/ab-65536.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# Arm name -> the variant request the profile would carry. `shipped` is the
# production selection as of 111ec0a60; `wmma` is the selection 25c76d141 shipped
# before it, which is the candidate being priced.
ARMS: dict[str, tuple[str, ...]] = {
    "staged": ("gemma4_staged",),
    "wmma": ("gemma4_wmma_flash", "gemma4_wmma_flash_full"),
}


def _merge(paths: list[Path], out_path: str) -> int:
    """Join per-shape points into one artifact, refusing a route collision."""

    points = [json.loads(p.read_text()) for p in paths]
    points.sort(key=lambda p: p["prompt_tokens"])
    rows = []
    for point in points:
        by_arm = {r["arm"]: r for r in point["rows"]}
        staged, wmma = by_arm.get("staged"), by_arm.get("wmma")
        if not staged or not wmma:
            raise SystemExit(f"{point['prompt_tokens']}: needs both arms, got {sorted(by_arm)}")
        if staged["route_selected"] == wmma["route_selected"]:
            raise SystemExit(
                f"{point['prompt_tokens']}: both arms selected {staged['route_selected']!r}; "
                "a collapsed comparison is not a comparison"
            )
        rows.append(
            {
                "prompt_tokens": point["prompt_tokens"],
                "context": point["context"],
                "staged_route": staged["route_selected"],
                "wmma_route": wmma["route_selected"],
                "staged_prefill_s": staged["prefill_best_s"],
                "wmma_prefill_s": wmma["prefill_best_s"],
                "staged_prefill_tps": staged["prefill_tps"],
                "wmma_prefill_tps": wmma["prefill_tps"],
                "speedup": round(staged["prefill_best_s"] / wmma["prefill_best_s"], 4),
                "warmup": point["warmup"],
                "repeats": staged["repeats"],
                "command": point["command"],
            }
        )
    payload = {
        "kind": "gemma4_prefill_attention_variant_ab",
        "performance_claim": False,
        "created": time.strftime("%Y-%m-%d"),
        "command": " ".join(sys.argv),
        "model": "Gemma 4 26B-A4B-it",
        "quant": "UD-Q4_K_XL",
        "physical_host": "zbook",
        "hardware": "AMD Radeon 8060S Graphics (gfx1151)",
        "kv_dtype": "bf16",
        "execution_profile": "production (both arms measured under it; only the variant request differs)",
        "arms": {
            "staged": list(ARMS["staged"]),
            "wmma": list(ARMS["wmma"]),
        },
        "why": (
            "the WMMA prefill candidates are registered but not selected, so the "
            "only way to price them is to request them directly. This changes no "
            "shipped default and is not a promotion row: the candidates fail the "
            "production numerical envelope, which is why they are not selected."
        ),
        "rows": rows,
        "provenance": points[0]["provenance"],
        "note": "diagnostic variant comparison; prefill only",
    }
    payload["findings"] = _against_ladder(rows)
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {out} ({len(rows)} points)")
    for row in rows:
        print(
            f"{row['prompt_tokens']:>7} staged {row['staged_prefill_tps']:>7.1f} tok/s | "
            f"wmma {row['wmma_prefill_tps']:>7.1f} tok/s | {row['speedup']:.3f}x"
        )
    return 0


def _against_ladder(rows: list[dict]) -> dict:
    """State what the candidate arm would do to the paired ladder's prefill column.

    The ladder artifact holds llama.cpp's measured prefill at the same shapes, so
    the consequence is arithmetic on two measured columns rather than a new claim.
    Absent the ladder artifact the block is omitted rather than guessed.
    """

    ladder = Path(__file__).resolve().parents[1] / "benchmarks/results/2026-10-04-gemma4-gfx1151-depth-ladder-paired.json"
    if not ladder.exists():
        return {"note": "paired ladder artifact not present; no comparison computed"}
    paired = json.loads(ladder.read_text())
    llama = {
        r["prompt_tokens"]: r.get("llamacpp", {}).get("prefill_tps")
        for r in paired["rows"]
    }
    table = []
    for row in rows:
        reference = llama.get(row["prompt_tokens"])
        if not reference:
            continue
        table.append(
            {
                "prompt_tokens": row["prompt_tokens"],
                "llamacpp_prefill_tps": reference,
                "shipped_staged_ratio": round(row["staged_prefill_tps"] / reference, 4),
                "candidate_wmma_ratio": round(row["wmma_prefill_tps"] / reference, 4),
            }
        )
    return {
        "prefill_ratio_vs_llamacpp": table,
        "source": ladder.name,
        "note": (
            "the candidate column is what the same-session A/B says the WMMA arm "
            "achieves; it is not a promotion, because the candidates fail the "
            "production numerical envelope"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--context", type=int, default=0)
    ap.add_argument("--arms", default="staged,wmma")
    ap.add_argument("--out", default="")
    # Combine per-shape points into one artifact. Each point carries its own
    # provenance, so the merge is a join rather than a re-derivation, and the
    # per-point commands travel with it.
    ap.add_argument("--merge", default="", help="comma-separated per-point JSONs")
    args = ap.parse_args()

    if args.merge:
        return _merge([Path(p) for p in args.merge.split(",") if p.strip()], args.out)
    if args.prompt <= 0:
        raise SystemExit("--prompt is required unless --merge is used")

    from scripts.gemma4_campaign_bench import (
        _provenance,
        _resolve_generator,
        exact_prompt_ids,
        resolve_artifact,
    )
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_layer as gl
    from hipengine.llm import SamplingParams

    context = int(args.context) or -(-(int(args.prompt) + 128) // 256) * 256
    artifact = resolve_artifact()
    llm, runner, info = _resolve_generator(Path(str(artifact)), context)
    generator = llm._get_text_generator()
    print(f"load_s={info['load_s']:.1f} context={info['context_length']} max_block={info['max_block']}")
    print(f"profile_before={generator.prefill_attention_variants!r}")

    prompt_ids = exact_prompt_ids(generator.tokenize, args.prompt)
    params = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)

    def prefill() -> float:
        started = time.perf_counter()
        llm.generate_detailed(prompt_ids, params)
        return time.perf_counter() - started

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known: {sorted(ARMS)}")

    rows = []
    for arm in arms:
        request = ARMS[arm]
        generator.prefill_attention_variants = request
        if hasattr(runner, "prefill_attention_variants"):
            runner.prefill_attention_variants = request
        for _ in range(args.warmup):
            prefill()
        gl._last_prefill_route = None
        seconds = []
        for _ in range(args.repeats):
            seconds.append(prefill())
        route = gl.last_prefill_attention_route()
        best = min(seconds)
        rows.append(
            {
                "arm": arm,
                "requested_variants": list(request),
                "route_selected": route,
                "repeats": args.repeats,
                "prefill_s": [round(s, 4) for s in seconds],
                "prefill_best_s": round(best, 4),
                "prefill_tps": round(args.prompt / best, 2),
            }
        )
        print(
            f"{arm:8} route={route!s:22} prefill {best:.3f}s "
            f"({args.prompt / best:.1f} tok/s)"
        )

    # A request that fell back to the strict kernel is not the candidate's speed,
    # so the comparison is refused rather than reported with a footnote.
    routes = {r["arm"]: r["route_selected"] for r in rows}
    distinct = {r for r in routes.values() if r is not None}
    verdict = "compared" if len(distinct) == len(rows) else "route_collision"
    if verdict != "compared":
        print(f"WARNING: arms did not select distinct routes: {routes}")

    payload = {
        "kind": "gemma4_prefill_attention_variant_ab",
        "performance_claim": False,
        "created": time.strftime("%Y-%m-%d"),
        "command": " ".join(sys.argv),
        "model": "Gemma 4 26B-A4B-it",
        "quant": "UD-Q4_K_XL",
        "physical_host": "zbook",
        "hardware": "AMD Radeon 8060S Graphics (gfx1151)",
        "kv_dtype": "bf16",
        "prompt_tokens": args.prompt,
        "context": context,
        "max_block": info["max_block"],
        "prompt_ids_sha256": hashlib.sha256(
            b"".join(int(t).to_bytes(4, "little") for t in prompt_ids)
        ).hexdigest(),
        "warmup": args.warmup,
        "verdict": verdict,
        "rows": rows,
        "provenance": _provenance(artifact),
        "note": (
            "diagnostic variant comparison; it changes no shipped default and is "
            "not a promotion row"
        ),
    }
    if len(rows) == 2 and all(r["prefill_best_s"] for r in rows):
        first, second = rows
        payload["ratio_second_over_first"] = round(
            second["prefill_best_s"] / first["prefill_best_s"], 4
        )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
