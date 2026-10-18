#!/usr/bin/env python3
"""Per-segment timing of the YuE2 product semantic decode loop.

Replicates ``hipengine.runtime.yue2_session.generate_tokens`` step for step and times
the three segments a step is made of: both branches' output heads, the host sampler,
and the two branch forwards. A segment boundary is only attributable with a
synchronize, because otherwise the head's time absorbs whatever the device is still
doing from the previous step's forward; ``--sync`` inserts one at each boundary, and
``--depth`` warms to a product-depth context first so the numbers describe the middle
of a request rather than its start.

    python3 scripts/yue2_ar_step_profile.py --tokens 150 --depth 900 --sync --paired

``--paired`` drives ``forward_layers_paired``, which runs both CFG branches in one pass
of two-row projection launches; without it each branch gets its own ``forward_layers``
call. The two agree bit for bit, so the same flags on both runs compare only scheduling.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_CASE = "mandarin-off-s1234"


def _model_dir(explicit: str) -> Path:
    """The pinned checkpoint, from ``--model-dir``, ``YUE2_MODEL_DIR`` or the HF cache."""

    if explicit:
        return Path(explicit)
    value = os.environ.get("YUE2_MODEL_DIR")
    if value:
        return Path(value)
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    matches = sorted(cache.glob("models--m-a-p--YuE2-3B/snapshots/*"))
    if not matches:
        raise SystemExit("no YuE2-3B checkpoint found; set YUE2_MODEL_DIR")
    return matches[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, default=150, help="timed steps")
    parser.add_argument("--depth", type=int, default=0, help="untimed warm steps before timing")
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--context", type=int, default=16000)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    from hipengine.generation.yue2 import (
        CODEC_OFFSET,
        Sampling,
        SongRequest,
        YuE2Random,
        combine_cfg,
        distribution_windowed,
        negative_prefix,
        phase_window,
        softmax_f32,
    )
    from hipengine.loading.yue2 import load_yue2_weights
    from hipengine.runtime.yue2_ar import Yue2ArRuntime
    from hipengine.runtime.yue2_session import bf16_bits_to_f32
    from hipengine.tokenization.yue2 import YuE2TextTokenizer

    model_dir = _model_dir(args.model_dir)
    weights = load_yue2_weights(model_dir)
    tokenizer = YuE2TextTokenizer(Path(model_dir) / "qwen.tiktoken")
    runtime = Yue2ArRuntime(weights, branches=2, max_context=args.context)

    case = np.load(REPO / "tests/fixtures/yue2/cases" / f"{args.case}.npz")
    prefix = [int(value) for value in case["prefix"]]
    song = SongRequest(
        style="warm acoustic pop",
        lyrics="[verse]\nMorning light across the floor",
        cot="off",
        seed=1234,
        id="step-profile",
    )
    negative = [int(value) for value in negative_prefix(song, tokenizer.encode, ())]
    sampling = Sampling(
        temperature=1.0, top_p=0.95, top_k=100, repetition_penalty=1.2,
        penalty_window=50, min_tokens=0, max_tokens=args.tokens,
    )
    window = phase_window("semantic")
    low, high = window
    generator = YuE2Random(1234)

    runtime.reset()
    runtime.prefill_host_rows([runtime.embed_row(t) for t in prefix], branch=0, start_pos=0)
    runtime.prefill_host_rows([runtime.embed_row(t) for t in negative], branch=1, start_pos=0)
    runtime.runtime.device_synchronize()

    paired = args.paired
    if paired and not hasattr(runtime, "forward_layers_paired"):
        raise SystemExit("this runtime has no paired forward")
    head_seconds = sampler_seconds = forward_seconds = 0.0
    history: list[int] = []
    for step in range(args.tokens + args.depth):
        timed = step >= args.depth
        started = time.perf_counter() if timed else 0.0
        conditional = bf16_bits_to_f32(runtime.logits(0, domain=window)[low:high])
        unconditional = bf16_bits_to_f32(runtime.logits(1, domain=window)[low:high])
        logits = combine_cfg(conditional, unconditional, song.guidance)
        if args.sync:
            runtime.runtime.device_synchronize()
        head_done = time.perf_counter() if timed else 0.0
        scores = distribution_windowed(logits, sampling, history, step, "semantic", window)
        token = scores.offset + generator.sample_categorical(softmax_f32(scores.values))
        if args.sync:
            runtime.runtime.device_synchronize()
        sampler_done = time.perf_counter() if timed else 0.0
        history.append(token)
        row = runtime.embed_row(token)
        position = runtime.context_length(0)
        negative_position = runtime.context_length(1)
        runtime.push_token(row, position, branch=0)
        runtime.push_token(row, negative_position, branch=1)
        if paired:
            runtime.forward_layers_paired(position, negative_position)
        else:
            runtime.forward_layers(position, branch=0)
            runtime.forward_layers(negative_position, branch=1)
        if args.sync:
            runtime.runtime.device_synchronize()
        forward_done = time.perf_counter() if timed else 0.0
        if timed:
            head_seconds += head_done - started
            sampler_seconds += sampler_done - head_done
            forward_seconds += forward_done - sampler_done
    total = head_seconds + sampler_seconds + forward_seconds
    steps = args.tokens
    payload = {
        "case": args.case,
        "steps": steps,
        "depth": args.depth,
        "context_start": len(prefix) + args.depth,
        "paired": paired,
        "synchronized": bool(args.sync),
        "total_seconds": round(total, 3),
        "step_ms": round(total / steps * 1000.0, 3),
        "head_ms": round(head_seconds / steps * 1000.0, 3),
        "sampler_ms": round(sampler_seconds / steps * 1000.0, 3),
        "forward_ms": round(forward_seconds / steps * 1000.0, 3),
    }
    print(json.dumps(payload, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
