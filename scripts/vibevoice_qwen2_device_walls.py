#!/usr/bin/env python3
"""Time the device Qwen2 backbone prefill and decode walls (engine venv).

Protocol: warm up, then measure the mean wall of a T-token prefill and of
single-token decode steps at a fixed cache length, on the pinned checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "/models/vibevoice/VibeVoice-1.5B"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prefill-tokens", type=int, default=64)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--json", default="benchmarks/results/2026-09-15-vibevoice-qwen2-device-walls.json")
    args = parser.parse_args()

    import ctypes

    ctypes.CDLL("libamdhip64.so")
    from hipengine.core.hip import get_hip_runtime
    from hipengine.models.vibevoice_lm import VibeVoiceQwen2Device

    rng = np.random.default_rng(20260915)
    tokens = rng.integers(1000, 100000, size=args.prefill_tokens).astype(np.int64)

    dev = VibeVoiceQwen2Device.from_checkpoint(args.model, runtime=get_hip_runtime())
    try:
        # Warm up (JIT + allocator).
        dev.forward_hidden(tokens[:4])
        dev.reset()

        prefill_walls = []
        for _ in range(args.repeats):
            dev.reset()
            t0 = time.perf_counter()
            dev.forward_hidden(tokens)
            prefill_walls.append(time.perf_counter() - t0)

        # Decode: prefill once, then time single steps.
        dev.reset()
        dev.forward_hidden(tokens)
        decode_walls = []
        for i in range(args.decode_steps):
            t0 = time.perf_counter()
            dev.decode_step(int(rng.integers(1000, 100000)))
            decode_walls.append(time.perf_counter() - t0)

        report = {
            "prefill_tokens": args.prefill_tokens,
            "decode_steps": args.decode_steps,
            "repeats": args.repeats,
            "prefill_mean_ms": float(np.mean(prefill_walls) * 1e3),
            "prefill_per_token_us": float(np.mean(prefill_walls) / args.prefill_tokens * 1e6),
            "decode_mean_ms": float(np.mean(decode_walls) * 1e3),
            "decode_p50_ms": float(np.percentile(decode_walls, 50) * 1e3),
            "decode_p95_ms": float(np.percentile(decode_walls, 95) * 1e3),
        }
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        dev.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
