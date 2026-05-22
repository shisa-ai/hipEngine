#!/usr/bin/env python3
"""Diagnostic GGUF layer-level AOTriton prefill threshold sweep."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_embedding import gguf_q6_k_embedding_bf16_out
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner, _FullStackScratch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
    parser.add_argument("--layer-id", type=int, default=3)
    parser.add_argument("--lengths", default="1,2,4", help="comma-separated prompt row counts")
    parser.add_argument("--tokens", default="760,4087,369,220", help="comma-separated token IDs used to build prefix rows")
    parser.add_argument("--attn-aotriton-min-tokens", type=int, default=3)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    lengths = _parse_ints(args.lengths, "--lengths")
    tokens = _parse_ints(args.tokens, "--tokens")
    if max(lengths) > len(tokens):
        raise ValueError("--tokens must contain at least max(--lengths) entries")
    runtime = get_hip_runtime()
    rows = []
    with Qwen35GGUFFullStackRunner(args.model) as runner:
        for length in lengths:
            hidden_rows = _prefix_hidden_rows(runner, tokens[:length], args.layer_id, runtime)
            start = time.perf_counter()
            result = runner.run_full_attention_prefill_layer(
                args.layer_id,
                hidden_rows,
                attn_aotriton_min_tokens=args.attn_aotriton_min_tokens,
            )
            seconds = time.perf_counter() - start
            rows.append(
                {
                    "rows": int(length),
                    "threshold": int(args.attn_aotriton_min_tokens),
                    "mode": result.mode,
                    "used_aotriton": bool(result.used_aotriton),
                    "seconds": seconds,
                }
            )
    payload = {
        "schema": 1,
        "model": str(args.model),
        "layer_id": int(args.layer_id),
        "tokens": tokens,
        "lengths": lengths,
        "attn_aotriton_min_tokens": int(args.attn_aotriton_min_tokens),
        "rows": rows,
    }
    text = json.dumps(payload, indent=2)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)


def _parse_ints(value: str, flag: str) -> list[int]:
    out = [int(part) for part in value.split(",") if part.strip()]
    if not out or any(item <= 0 for item in out):
        raise ValueError(f"{flag} must contain positive integers")
    return out


def _prefix_hidden_rows(runner: Qwen35GGUFFullStackRunner, token_ids: list[int], layer_id: int, runtime) -> np.ndarray:
    scratch = _FullStackScratch.allocate(runner, runtime=runtime)
    token_buf = malloc(np.dtype(np.int64).itemsize, runtime=runtime)
    hidden_a = malloc(runner.hidden_size * 2, runtime=runtime)
    hidden_b = malloc(runner.hidden_size * 2, runtime=runtime)
    hidden_rows = np.empty((len(token_ids), runner.hidden_size), dtype=np.uint16)
    try:
        scratch.zero_states(runtime)
        for position, token_id in enumerate(token_ids):
            scratch.set_full_attention_position(position, runtime)
            token_arr = np.asarray([int(token_id)], dtype=np.int64)
            copy_host_to_device(token_buf, host_array_ptr(token_arr), runtime=runtime)
            gguf_q6_k_embedding_bf16_out(
                token_buf.ptr,
                runner.weights.root("token_embedding").allocation().tensor.ptr,
                hidden_a.ptr,
                rows=1,
                hidden_size=runner.hidden_size,
                vocab_size=runner.vocab_size,
                runtime=runtime,
            )
            src = hidden_a
            dst = hidden_b
            for prev_layer_id in range(layer_id):
                if runner.weights.config.layer_types[prev_layer_id] != "linear_attention":
                    raise ValueError("sweep helper currently supports the first full-attention layer only")
                runner._run_linear_attention_layer(prev_layer_id, src.ptr, dst.ptr, scratch)
                src, dst = dst, src
            copy_device_to_host(host_array_ptr(hidden_rows[position : position + 1]), src, runtime=runtime)
    finally:
        for buffer in reversed((hidden_b, hidden_a, token_buf, *scratch.buffers)):
            free(buffer, runtime=runtime)
    return hidden_rows


if __name__ == "__main__":
    main()
