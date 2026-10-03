"""Print which MoE expert route actually ran during a short decode.

The route name is read from the engine's own counters, not inferred from a
kernel name in a trace or from the env var that was set.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/gemma4_moe_route_probe.py --output 16
  HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ=0 PYTHONPATH=. .venv/bin/python \
      scripts/gemma4_moe_route_probe.py --output 16
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=64)
    ap.add_argument("--output", type=int, default=16)
    args = ap.parse_args()

    from scripts.gemma4_campaign_bench import resolve_artifact

    import hipengine
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
        gemma4_moe_expert_route_counts,
        gemma4_moe_grouped_variant_counts,
    )

    artifact = resolve_artifact()
    llm = hipengine.LLM(model=str(artifact))
    generator = llm._get_text_generator()
    generator.context_length = 8192
    prompt_ids = [(i * 7 + 3) % 2000 + 1 for i in range(args.prompt)]

    from hipengine.llm import SamplingParams

    params = SamplingParams(
        max_tokens=int(args.output), temperature=0.0, ignore_eos=True
    )
    llm.generate_detailed(list(prompt_ids), SamplingParams(max_tokens=4, ignore_eos=True))
    started = time.perf_counter()
    llm.generate_detailed(list(prompt_ids), params)
    elapsed = time.perf_counter() - started

    print(f"env HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ="
          f"{os.environ.get('HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ', '<unset>')}")
    print(f"env HIPENGINE_GEMMA4_MOE_DOWN_MMQ="
          f"{os.environ.get('HIPENGINE_GEMMA4_MOE_DOWN_MMQ', '<unset>')}")
    print(f"prompt={args.prompt} output={args.output} wall={elapsed:.2f}s")
    print("expert routes:")
    for name, count in sorted(gemma4_moe_expert_route_counts().items()):
        print(f"  {name:<28} {count}")
    variants = gemma4_moe_grouped_variant_counts()
    if variants:
        print("grouped variants:")
        for name, count in sorted(variants.items()):
            print(f"  {name:<28} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
