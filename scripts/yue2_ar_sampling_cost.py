#!/usr/bin/env python3
"""What the YuE2 AR sampling path costs per token on the host.

The fixed-token AR harness measures 58.3 ms of model work per decode step, while the
product path's AR stage costs 82.1 ms per token. The difference is the per-step host
work the product loop does on a 184,704-wide logits row - the CFG combine, the
repetition-penalty distribution, top-k/top-p, the softmax and the categorical draw -
which the reference does with device ops and reduces to a single element before it
crosses to the host. This splits that cost by call so the lever is measured rather
than inferred.

CPU only; no device work, no GPU contention with a running benchmark.

    python3 scripts/yue2_ar_sampling_cost.py --history 1296 --repeats 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

VOCAB = 184704
CODEC_OFFSET = 151853
CODEC_SIZE = 16384
MUSIC_END = 151852


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=int, default=1296, help="generated tokens before this step")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    from hipengine.generation.yue2 import (
        Sampling,
        YuE2Random,
        combine_cfg,
        distribution,
        softmax_f32,
    )
    from hipengine.kernels.cpu_reference.yue2 import bf16_bits_to_f32, f32_to_bf16_bits

    # The case's own semantic sampling, and a history of codec tokens like a real run.
    sampling = Sampling(temperature=1.0, top_p=0.95, top_k=100, repetition_penalty=1.2,
                        penalty_window=50, min_tokens=200, max_tokens=9000)
    rng = np.random.default_rng(args.seed)
    history = [int(v) for v in rng.integers(CODEC_OFFSET, CODEC_OFFSET + CODEC_SIZE, args.history)]
    raw = rng.standard_normal(VOCAB).astype(np.float32)
    bits0 = f32_to_bf16_bits(raw)
    bits1 = f32_to_bf16_bits(raw * 0.9 + 0.1)
    generator = YuE2Random(args.seed)

    def timed(fn, repeats: int) -> float:
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - started) * 1000.0)
        return statistics.median(samples)

    conditional = bf16_bits_to_f32(bits0)
    unconditional = bf16_bits_to_f32(bits1)
    combined = combine_cfg(conditional, unconditional, 1.01)
    scores = distribution(combined, sampling, history, args.history, "semantic", legacy_off=True)

    parts = {
        "bf16_bits_to_f32 (per branch)": timed(lambda: bf16_bits_to_f32(bits0), args.repeats),
        "combine_cfg": timed(lambda: combine_cfg(conditional, unconditional, 1.01), args.repeats),
        "distribution (penalty + top-k + top-p)": timed(
            lambda: distribution(combined, sampling, history, args.history, "semantic", legacy_off=True),
            args.repeats,
        ),
        "softmax_f32": timed(lambda: softmax_f32(scores), args.repeats),
        "sample_categorical": timed(
            lambda: generator.sample_categorical(softmax_f32(scores)), args.repeats
        ),
    }
    parts["total"] = sum(parts.values())

    print(f"vocab {VOCAB}, history {args.history} tokens, "
          f"median of {args.repeats} runs per call")
    print()
    print("| Call | ms per token |")
    print("| --- | ---: |")
    for name, value in parts.items():
        print(f"| {name} | {value:.2f} |")
    print()
    print(f"measured model work per step (fixed-token harness): 58.30 ms")
    print(f"product AR stage per token: 82.13 ms; "
          f"host sampling here: {parts['total']:.2f} ms")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "protocol": "yue2-ar-sampling-cost-v1",
            "vocab": VOCAB,
            "history_tokens": args.history,
            "repeats": args.repeats,
            "ms_per_token": parts,
            "note": "CPU only; medians of per-call wall clock on one host thread.",
        }, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
