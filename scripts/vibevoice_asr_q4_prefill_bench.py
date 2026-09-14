#!/usr/bin/env python3
"""Reproduce the VibeVoice-ASR Q4_K_M prefill/decode comparison.

Measures the dense bf16 lane and the Q4_K_M lane under one protocol: same
fixture prompt, same process, one warmup then N repetitions of
``runtime.prefill_rows``, followed by a greedy 16-token chain in each lane to
confirm the two produce identical tokens.

Both lanes route prefill through ``prefill_rows``: the bf16 lane uses its
hipBLASLt fp16 GEMMs, the Q4 lane uses the raw-block Q4_K/Q6_K WMMA prefill
kernels. Decode is pack8 GEMV for Q4 and hipBLASLt for bf16.

Usage:
    python3 scripts/vibevoice_asr_q4_prefill_bench.py --reps 5 \\
        --gguf /tmp/vibevoice-asr-q4km.gguf \\
        --out benchmarks/results/vibevoice-q4-prefill-bench.json

Requires ROCm and both weight sets present locally. The GGUF is produced by
``scripts/vibevoice_asr_to_gguf.py`` plus llama-quantize; see
``docs/MODEL-VIBEVOICE-ASR.md``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "vibevoice_asr" / "vibevoice_asr_lm.npz"


def _load_prompt():
    """Prompt rows for the fixture clip, audio rows spliced in at their positions."""
    data = dict(np.load(FIXTURE))
    input_ids = np.asarray(data["input_ids"])[0]
    positions = np.asarray(data["audio_placeholder_positions"])
    audio = data["audio_embeds"].astype(np.float32)
    return input_ids, positions, audio


def _prompt_rows(runner, input_ids, positions, audio):
    rows = [runner.embed_row(int(t)) for t in input_ids]
    for position in positions:
        rows[position] = audio[position - positions[0]]
    return rows


def _bench_prefill(runner, rows_list, reps):
    from hipengine.core.memory import copy_host_array_to_device, free, malloc
    from hipengine.loading.vibevoice_layout import f32_to_bf16_bits

    rows = len(rows_list)
    buffer = malloc(rows * runner.spec.hidden_size * 2)
    try:
        copy_host_array_to_device(
            buffer, f32_to_bf16_bits(np.asarray(rows_list, dtype=np.float32)))
        # one warmup so JIT/build cost is outside the measured window
        runner.reset()
        runner.prefill_rows(buffer, rows, 0)
        runner.runtime.device_synchronize()
        start = time.perf_counter()
        for _ in range(reps):
            runner.reset()
            runner.prefill_rows(buffer, rows, 0)
        runner.runtime.device_synchronize()
        return (time.perf_counter() - start) / reps
    finally:
        free(buffer)


def _bench_greedy(runner, rows_list, tokens):
    from hipengine.runtime.vibevoice_qwen2 import greedy_generate

    runner.reset()
    start = time.perf_counter()
    generated = greedy_generate(runner, rows_list, max_new_tokens=tokens)
    runner.runtime.device_synchronize()
    return time.perf_counter() - start, generated


def _measure(runner, prompt, reps, tokens):
    rows = len(prompt)
    prefill = _bench_prefill(runner, prompt, reps)
    elapsed, generated = _bench_greedy(runner, prompt, tokens)
    return {
        "prompt_rows": rows,
        "prefill_s": prefill,
        "greedy_s": elapsed,
        "greedy_tokens": len(generated),
        "generated": generated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", default="/tmp/vibevoice-asr-q4km.gguf")
    parser.add_argument("--model", default="microsoft/VibeVoice-ASR-HF")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--greedy-tokens", type=int, default=16)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    input_ids, positions, audio = _load_prompt()

    from hipengine.loading.hf_cache import resolve_model_path

    model_path = resolve_model_path(args.model)

    from hipengine.loading.vibevoice_asr import load_vibevoice_qwen2
    from hipengine.runtime.vibevoice_qwen2 import VibevoiceQwen2Runtime

    bf16 = VibevoiceQwen2Runtime(load_vibevoice_qwen2(str(model_path)),
                                 max_context=1024, prefill_variant="hipblaslt")
    try:
        bf16_result = _measure(bf16, _prompt_rows(bf16, input_ids, positions, audio),
                               args.reps, args.greedy_tokens)
    finally:
        bf16.close()

    from hipengine.loading.vibevoice_asr_gguf import load_vibevoice_qwen2_q4
    from hipengine.runtime.vibevoice_qwen2_q4 import VibevoiceQwen2Q4Runtime

    weights = load_vibevoice_qwen2_q4(args.gguf)
    try:
        q4 = VibevoiceQwen2Q4Runtime(weights, max_context=1024)
        try:
            q4_result = _measure(q4, _prompt_rows(q4, input_ids, positions, audio),
                                 args.reps, args.greedy_tokens)
        finally:
            q4.close()
    finally:
        weights.close()

    tokens_match = bf16_result["generated"] == q4_result["generated"]
    summary = {
        "protocol": {
            "fixture": str(FIXTURE.relative_to(Path(__file__).resolve().parent.parent)),
            "reps": args.reps,
            "greedy_tokens": args.greedy_tokens,
            "warmup": "one prefill before the measured window",
            "gguf": args.gguf,
            "note": "both lanes batched through prefill_rows in one process",
        },
        "bf16": bf16_result,
        "q4": q4_result,
        "prefill_speedup": bf16_result["prefill_s"] / q4_result["prefill_s"],
        "greedy_speedup": bf16_result["greedy_s"] / q4_result["greedy_s"],
        "greedy_tokens_identical": tokens_match,
    }
    print(f"prefill {bf16_result['prompt_rows']} rows over {args.reps} reps:")
    print(f"  bf16 {bf16_result['prefill_s']:.3f} s | q4 {q4_result['prefill_s']:.3f} s"
          f"  -> {summary['prefill_speedup']:.2f}x")
    print(f"greedy {args.greedy_tokens} tokens:")
    print(f"  bf16 {bf16_result['greedy_s']:.3f} s | q4 {q4_result['greedy_s']:.3f} s"
          f"  -> {summary['greedy_speedup']:.2f}x")
    print(f"token chains identical: {tokens_match}")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
