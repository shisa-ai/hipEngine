#!/usr/bin/env python3
"""Per-token host cost of select_token vs history length (CPU only).

Long-context regressions in the host sampler only show at long prompts (a
history walk costs ~11 ms/token at 126K). Times greedy and sampled requests
with neutral and active penalties at each prompt length; neutral-penalty rows
should stay flat as the prompt grows.

  .venv/bin/python scripts/sampler_host_cost_probe.py --lengths 512,32768,131072
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lengths", default="512,32768,131072")
    parser.add_argument("--vocab", type=int, default=201088)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args(argv)

    from hipengine.generation.registry import GenerationRequest
    from hipengine.generation.sampling import RowSamplingState, select_token

    cases = {
        "greedy": dict(temperature=0.0),
        "greedy+rep1.1": dict(temperature=0.0, repetition_penalty=1.1),
        "t0.7/top_p0.9": dict(temperature=0.7, top_p=0.9),
    }
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(args.vocab).astype(np.float32)
    print(f"{'prompt':>8} " + " ".join(f"{name:>16}" for name in cases) + "   (ms/token)")
    for n in (int(x) for x in args.lengths.split(",")):
        ids = tuple(int(x) for x in rng.integers(0, args.vocab, n))
        cells = []
        for overrides in cases.values():
            req = GenerationRequest(prompts=(ids,), max_tokens=256, top_p=overrides.pop("top_p", 1.0),
                                    ignore_eos=True, **overrides)
            state = RowSamplingState(prompt_tokens=ids, seed=0, row_index=0)
            select_token(logits.copy(), req, state, token_text_for_id=lambda t: "")
            t0 = time.perf_counter()
            for _ in range(args.iters):
                select_token(logits.copy(), req, state, token_text_for_id=lambda t: "")
            cells.append((time.perf_counter() - t0) / args.iters * 1e3)
        print(f"{n:>8} " + " ".join(f"{c:16.2f}" for c in cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
