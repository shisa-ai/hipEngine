#!/usr/bin/env python3
"""Direct 35B GGUF prefill chunk-boundary numerical smoke on gfx1151."""

import argparse
import ctypes

import numpy as np

import hipengine.runtime.qwen35_gguf_runner as rm
from hipengine.runtime.prefill import PrefillConfig

ctypes.CDLL("libamdhip64.so")

MODEL = "/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
TOKEN_ID = 9707


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--length", type=int, default=2048)
    args = parser.parse_args()

    tokens = [TOKEN_ID] * args.length
    original_override = rm._gguf_prefill_chunk_sizes_for
    rm._gguf_prefill_chunk_sizes_for = lambda *unused_args, **unused_kwargs: None
    try:
        logits: dict[int, np.ndarray] = {}
        for chunk in (1024, 512):
            with rm.Qwen35GGUFResidentSession(
                args.model,
                max_sequence_length=args.length,
                backend="hip_gfx1151",
                prefill_config=PrefillConfig(
                    linear_chunk_size=chunk,
                    moe_chunk_size=chunk,
                ),
            ) as session:
                result = session.prefill(tokens, use_bulk=True, return_logits=True)
                session.runtime.device_synchronize()
                raw_logits = result.logits if hasattr(result, "logits") else result
                logits[chunk] = np.asarray(raw_logits, dtype=np.float32).reshape(-1)
    finally:
        rm._gguf_prefill_chunk_sizes_for = original_override

    p1024 = _softmax(logits[1024])
    p512 = _softmax(logits[512])
    kl = float((p1024 * (np.log(p1024 + 1e-12) - np.log(p512 + 1e-12))).sum())
    top1_1024 = int(np.argmax(logits[1024]))
    top1_512 = int(np.argmax(logits[512]))
    max_abs = float(np.max(np.abs(logits[1024] - logits[512])))

    print(f"KL(l1024||l512)={kl:.8f}")
    print(f"top1_1024={top1_1024} top1_512={top1_512}")
    print(f"max_abs_logit={max_abs:.8f}")


if __name__ == "__main__":
    main()
