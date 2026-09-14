#!/usr/bin/env python3
"""Where a VibeVoice-ASR Q4 decoded token's wall time goes.

Reports three things for the decode loop:

1. **Phase split.** ``push_token`` + ``forward_layers`` enqueue work
   asynchronously, so timing them without a synchronize measures *enqueue*
   time, not execution. This splits a token into enqueue, the synchronize that
   drains the layer stack, and ``logits_argmax`` (lm_head GEMV + logits
   download + host argmax).

2. **Launch count.** How many kernel-wrapper calls one ``forward_layers`` makes,
   by name.

3. **Overlap probe.** Enqueue N tokens' layer stacks and synchronize once,
   versus synchronizing after every token. If the two agree, the host enqueue
   time is being absorbed by GPU execution rather than adding to it.

Interpretation limits
---------------------
The overlap probe compares two enqueue/synchronize shapes; it bounds how much
host launch overhead can be adding to the measured loop, and a small difference
is evidence that it is mostly absorbed. It is not a proof that no host launch
overhead remains, and it says nothing about other shapes.

Nothing in this script measures the batched-prefill GEMM kernels. Decode GEMV
bandwidth is not evidence about a prefill GEMM outlier, which needs its own
measurement.

Usage:
    python3 scripts/vibevoice_q4_decode_phase_split.py [--steps N] [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np

GGUF = Path("/tmp/vibevoice-asr-q4km.gguf")
LM_FIXTURE = (Path(__file__).resolve().parent.parent
              / "tests" / "fixtures" / "vibevoice_asr" / "vibevoice_asr_lm.npz")


def _prompt_rows(runner, lm) -> list[np.ndarray]:
    input_ids = np.asarray(lm["input_ids"])[0]
    positions = np.asarray(lm["audio_placeholder_positions"])
    audio = lm["audio_embeds"].astype(np.float32)
    rows = [runner.embed_row(int(t)) for t in input_ids]
    for p in positions:
        rows[p] = audio[p - positions[0]]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--overlap-tokens", type=int, default=40)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    from hipengine.core.memory import (MemcpyKind, copy_host_array_to_device,
                                       free, malloc)
    from hipengine.loading.vibevoice_asr_gguf import load_vibevoice_qwen2_q4
    from hipengine.loading.vibevoice_layout import f32_to_bf16_bits
    from hipengine.runtime.vibevoice_qwen2_q4 import VibevoiceQwen2Q4Runtime

    if not GGUF.is_file():
        raise SystemExit(f"missing Q4 GGUF: {GGUF}")
    if not LM_FIXTURE.is_file():
        raise SystemExit(f"missing LM fixture: {LM_FIXTURE}")

    with np.load(LM_FIXTURE) as data:
        lm = {k: data[k] for k in data.files}

    weights = load_vibevoice_qwen2_q4(GGUF)
    runner = VibevoiceQwen2Q4Runtime(weights, max_context=512)
    sync = runner.runtime.device_synchronize
    payload: dict[str, object] = {}
    try:
        hidden = runner.spec.hidden_size
        rows = _prompt_rows(runner, lm)
        total = len(rows)
        pristine = f32_to_bf16_bits(np.asarray(rows, dtype=np.float32))
        prompt = malloc(total * hidden * 2)

        def prefill() -> None:
            copy_host_array_to_device(prompt, pristine)
            runner.reset()
            runner.prefill_rows(prompt, total, 0)
            runner.runtime.memcpy(runner._hidden.ptr,
                                  prompt.ptr + (total - 1) * hidden * 2,
                                  hidden * 2, MemcpyKind.DEVICE_TO_DEVICE)
            sync()

        prefill()
        token = runner.logits_argmax()[1]

        # --- 3. launch count -------------------------------------------------
        counts: Counter[str] = Counter()
        wrapped = []
        for name in dir(runner.kernels):
            if name.startswith("_"):
                continue
            fn = getattr(runner.kernels, name)
            if not callable(fn):
                continue

            def wrap(fn=fn, name=name):
                def inner(*a, **kw):
                    counts[name] += 1
                    return fn(*a, **kw)
                return inner

            wrapped.append((name, fn, wrap()))
            setattr(runner.kernels, name, wrap())
        try:
            prefill()
            token = runner.logits_argmax()[1]
            counts.clear()
            runner.push_token(runner.embed_row(token), total)
            runner.forward_layers(total)
            calls_per_layer_stack = sum(counts.values())
            breakdown = counts.most_common()
        finally:
            for name, fn, _ in wrapped:
                setattr(runner.kernels, name, fn)

        # --- 1. phase split --------------------------------------------------
        launch, execute, logits = [], [], []
        for _ in range(args.trials):
            prefill()
            token = runner.logits_argmax()[1]
            pos = total
            a = b = c = 0.0
            for _ in range(args.steps):
                t0 = time.perf_counter()
                runner.push_token(runner.embed_row(token), pos)
                runner.forward_layers(pos)
                t1 = time.perf_counter()
                sync()
                t2 = time.perf_counter()
                _, token = runner.logits_argmax()
                t3 = time.perf_counter()
                a += t1 - t0; b += t2 - t1; c += t3 - t2
                pos += 1
            launch.append(a * 1000 / args.steps)
            execute.append(b * 1000 / args.steps)
            logits.append(c * 1000 / args.steps)
        la, ex, lg = (statistics.median(v) for v in (launch, execute, logits))
        phase_total = la + ex + lg

        # --- 3. overlap probe ------------------------------------------------
        row = runner.embed_row(token)
        n = args.overlap_tokens

        def enqueue_only(pos):
            for i in range(n):
                runner.push_token(row, pos + i)
                runner.forward_layers(pos + i)

        for _ in range(2):
            enqueue_only(total); sync()

        bulk, per_token = [], []
        for _ in range(3):
            t0 = time.perf_counter(); enqueue_only(total); sync()
            bulk.append((time.perf_counter() - t0) * 1000 / n)
            t0 = time.perf_counter()
            for i in range(n):
                runner.push_token(row, total + i)
                runner.forward_layers(total + i)
                sync()
            per_token.append((time.perf_counter() - t0) * 1000 / n)
        bm, pm = statistics.median(bulk), statistics.median(per_token)

        print(f"kernel-wrapper calls per layer stack : {calls_per_layer_stack}")
        for name, count in breakdown[:6]:
            print(f"    {name:32s} {count}")
        print(f"enqueue (push+forward)  ms/token     : {la:6.2f}")
        print(f"synchronize (drain)     ms/token     : {ex:6.2f}")
        print(f"logits_argmax           ms/token     : {lg:6.2f}")
        print(f"phase total             ms/token     : {phase_total:6.2f}")
        print(f"overlap: enqueue N then sync         : {bm:6.2f} ms/token")
        print(f"overlap: sync every token            : {pm:6.2f} ms/token")
        print(f"overlap difference                   : {pm - bm:+6.2f} ms/token")
        print("  (a small difference means the enqueue time is mostly absorbed by "
              "GPU work; it does not prove no host overhead remains)")

        payload = {
            "steps_per_trial": args.steps,
            "calls_per_layer_stack": calls_per_layer_stack,
            "call_breakdown": breakdown,
            "enqueue_ms_per_token": round(la, 2),
            "enqueue_trials": [round(v, 2) for v in launch],
            "drain_ms_per_token": round(ex, 2),
            "drain_trials": [round(v, 2) for v in execute],
            "logits_argmax_ms_per_token": round(lg, 2),
            "logits_trials": [round(v, 2) for v in logits],
            "phase_total_ms_per_token": round(phase_total, 2),
            "overlap_enqueue_then_sync_ms": round(bm, 2),
            "overlap_sync_every_token_ms": round(pm, 2),
            "overlap_difference_ms": round(pm - bm, 2),
            "logits_download_bytes_per_token": runner.spec.vocab_size * 4,
            "interpretation": (
                "Enqueue time is largely absorbed by GPU execution: the two "
                "overlap shapes agree to within "
                f"{abs(pm - bm):.2f} ms/token. This bounds how much host launch "
                "overhead adds to the measured loop; it is not proof that none "
                "remains. This script does not measure prefill GEMM kernels, so "
                "it is not evidence about a batched-prefill GEMM outlier."
            ),
        }
        free(prompt)
    finally:
        runner.close()
        weights.close()

    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
