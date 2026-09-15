#!/usr/bin/env python3
"""Time the pinned fork's HF Qwen2 backbone walls (oracle venv, PyTorch).

Same protocol as vibevoice_qwen2_device_walls.py: T-token prefill wall and
single-token decode steps at a fixed cache length, on the W7900.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

DEFAULT_FORK = "/home/lhl/VibeVoice-community"
DEFAULT_MODEL = "/models/vibevoice/VibeVoice-1.5B"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fork", default=DEFAULT_FORK)
    parser.add_argument("--prefill-tokens", type=int, default=64)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--json", default="benchmarks/results/2026-09-15-vibevoice-qwen2-torch-walls.json")
    args = parser.parse_args()

    import sys
    import torch

    sys.path.insert(0, args.fork)
    from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration

    model = VibeVoiceForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="eager"
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        device = "cuda"
    else:
        device = "cpu"
    lm = model.model.language_model
    embed = model.get_input_embeddings().weight

    rng = np.random.default_rng(20260915)
    tokens = torch.tensor(rng.integers(1000, 100000, size=args.prefill_tokens), dtype=torch.long, device=device)[None]

    def wall(fn):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    # Warm up.
    with torch.no_grad():
        lm(tokens[:, :4])
    prefill_walls = []
    for _ in range(args.repeats):
        with torch.no_grad():
            prefill_walls.append(wall(lambda: lm(tokens)))
    prefill_walls = prefill_walls[1:]  # first repeat includes allocator growth

    decode_walls = []
    with torch.no_grad():
        past = lm(tokens)
        for _ in range(args.decode_steps):
            step = torch.tensor([[int(rng.integers(1000, 100000))]], dtype=torch.long, device=device)
            decode_walls.append(wall(lambda: lm(step, past_key_values=past.past_key_values)))

    report = {
        "device": device,
        "dtype": "float16",
        "attn_implementation": "eager",
        "prefill_tokens": args.prefill_tokens,
        "decode_steps": args.decode_steps,
        "repeats": args.repeats - 1 if device == "cuda" else args.repeats,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
