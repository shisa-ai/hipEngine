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
    parser.add_argument("--scaling", action="store_true",
                        help="also fit the product loop's context scaling (loads the model)")
    parser.add_argument("--case", default="mandarin-off-s1234")
    parser.add_argument("--budgets", default="300,700,1100,1450",
                        help="comma-separated token budgets for --scaling")
    args = parser.parse_args()
    args.budgets = [int(v) for v in str(args.budgets).split(",") if v]

    from hipengine.generation.yue2 import (
        Sampling,
        YuE2Random,
        combine_cfg,
        distribution,
        distribution_windowed,
        phase_window,
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
    window = phase_window("semantic")
    low, high = window
    scores = distribution(combined, sampling, history, args.history, "semantic")
    windowed = distribution_windowed(
        combined[low:high], sampling, history, args.history, "semantic", window
    )

    full = {
        "bf16_bits_to_f32 (per branch)": timed(lambda: bf16_bits_to_f32(bits0), args.repeats),
        "combine_cfg": timed(lambda: combine_cfg(conditional, unconditional, 1.01), args.repeats),
        "distribution (penalty + top-k + top-p)": timed(
            lambda: distribution(combined, sampling, history, args.history, "semantic"),
            args.repeats,
        ),
        "softmax_f32": timed(lambda: softmax_f32(scores), args.repeats),
        "sample_categorical": timed(
            lambda: generator.sample_categorical(softmax_f32(scores)), args.repeats
        ),
    }
    full["total"] = sum(full.values())

    # The product path converts the head row's window slice, combines CFG over the slice
    # and samples in window coordinates, so this is the arm that describes it.
    windowed_parts = {
        "bf16_bits_to_f32 (per branch, window slice)": timed(
            lambda: bf16_bits_to_f32(bits0[low:high]), args.repeats
        ),
        "combine_cfg (window slice)": timed(
            lambda: combine_cfg(conditional[low:high], unconditional[low:high], 1.01),
            args.repeats,
        ),
        "distribution_windowed (penalty + top-k + top-p)": timed(
            lambda: distribution_windowed(
                combined[low:high], sampling, history, args.history, "semantic", window
            ),
            args.repeats,
        ),
        "softmax_f32 (window slice)": timed(
            lambda: softmax_f32(windowed.values), args.repeats
        ),
        "sample_categorical (window slice)": timed(
            lambda: generator.sample_categorical(softmax_f32(windowed.values)), args.repeats
        ),
    }
    windowed_parts["total"] = sum(windowed_parts.values())

    print(f"vocab {VOCAB}, history {args.history} tokens, "
          f"median of {args.repeats} runs per call, semantic window {window}")
    print()
    print("| Call | full row | window |")
    print("| --- | ---: | ---: |")
    for name, value in full.items():
        match = next((v for k, v in windowed_parts.items() if k.split(" (")[0] == name.split(" (")[0]), None)
        print(f"| {name} | {value:.3f} ms | "
              f"{'%.3f ms' % match if match is not None else '-'} |")
    print()
    print(f"host sampling per token: {full['total']:.2f} ms over the full row, "
          f"{windowed_parts['total']:.2f} ms windowed")
    report = {
        "protocol": "yue2-ar-sampling-cost-v2",
        "host": {"hostname": Path("/etc/hostname").read_text().strip(),
                 "load_average": Path("/proc/loadavg").read_text().split()[:3]},
        "vocab": VOCAB,
        "history_tokens": args.history,
        "repeats": args.repeats,
        "window": list(window),
        "ms_per_token_full_row": full,
        "ms_per_token_windowed": windowed_parts,
        "note": ("CPU only, single-threaded; medians of per-call wall clock. These are host "
                 "numbers, so the load average above matters: a concurrent job inflates them. "
                 "The windowed "
                 "arm is what the product loop runs: it converts, combines and samples the "
                 "head row's window slice."),
    }


if __name__ == "__main__":
    raise SystemExit(main())
