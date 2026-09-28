"""A/B the dense prefill route on gfx1151.

The dense projections are the largest single prefill term (189.3 ms, 28.3
percent) and sit at a 127 GB/s plateau with 16x weight re-reads
(`20260929T020000`). gfx1151 declares the raw-K rowbatch/coltile family
UNSUPPORTED by explicit policy -- "gfx1151 retains its independently admitted
Q4_K_M/T16 matrix schedules and must not inherit their rowbatch or
output-column selectors" -- so the dense path takes the WMMA T16 route instead.

That policy was decided on the W7900. This asks the narrow question the policy
does not answer: at 512 rows, on this host, is the raw-K rowbatch/coltile route
actually faster or slower than the T16 route the host ships with?

It is a measurement, not a proposal. If the disabled route wins, the policy
question is the human lead's; if it loses, the policy is confirmed for this
shape and the dense plateau is a kernel unit rather than a routing mistake.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    from scripts.gemma4_campaign_bench import _resolve_generator, resolve_artifact

    from hipengine.core.hip import get_hip_runtime
    from hipengine.llm import SamplingParams
    from hipengine.runtime.gguf_linear import (
        raw_k_prefill_rowbatch,
        raw_k_prefill_rowbatch_session,
        raw_k_prefill_variant,
        raw_k_prefill_variant_session,
    )

    model = args.model or str(resolve_artifact())
    llm, _runner, info = _resolve_generator(Path(model), 4096)
    print(f"load_s={info['load_s']:.1f} resolution={info['resolution']}", flush=True)
    runtime = get_hip_runtime()

    prompt_ids = list(range(1000, 1000 + args.prompt))
    params = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)

    def prefill() -> None:
        llm.generate_detailed(prompt_ids, params)

    def time_arm() -> float:
        for _ in range(args.warmup):
            prefill()
        samples = []
        for _ in range(args.repeats):
            runtime.device_synchronize()
            started = time.perf_counter()
            prefill()
            runtime.device_synchronize()
            samples.append(time.perf_counter() - started)
        return statistics.median(samples)

    # (label, rowbatch, variant). None means "leave the session alone".
    arms = [
        ("default (host policy)", None, None),
        ("rowbatch32 + coltile", 32, "coltile"),
        ("rowbatch32 + rowbatch", 32, "rowbatch"),
        ("rowbatch8  + coltile", 8, "coltile"),
    ]

    print(f"{'arm':26s} {'median s':>9s} {'tok/s':>8s}  session seen inside")
    for label, rowbatch, variant in arms:
        if rowbatch is None:
            median = time_arm()
            seen = f"rowbatch={raw_k_prefill_rowbatch()} variant={raw_k_prefill_variant()}"
        else:
            with raw_k_prefill_rowbatch_session(rowbatch):
                with raw_k_prefill_variant_session(variant):
                    median = time_arm()
                    seen = (
                        f"rowbatch={raw_k_prefill_rowbatch()} "
                        f"variant={raw_k_prefill_variant()}"
                    )
        print(
            f"{label:26s} {median:9.4f} {args.prompt / median:8.1f}  {seen}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
