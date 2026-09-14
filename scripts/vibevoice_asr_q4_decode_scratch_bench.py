#!/usr/bin/env python3
"""VibeVoice-ASR Q4_K_M decode A/B benchmark for caller-owned scratch.

Measures one decoded token's cost with the o_proj decode GEMV receiving a
persistent bf16 scratch versus allocating a fresh one on every layer. With
``--compare-bf16`` it also measures the dense bf16 backbone so the matched
Q4-vs-bf16 decode ratio can be restated from one session rather than compared
against a number recorded in an earlier session.

Every lane runs the identical loop, fixture and prompt rows in one process,
rotating lane order per trial so host drift cannot favour any lane. The Q4
lanes must produce the same token chain and bit-identical logits; the only
difference between them is where the o_proj input scratch comes from.

Usage:
    python3 scripts/vibevoice_asr_q4_decode_scratch_bench.py [--steps N] [--trials N]
    python3 scripts/vibevoice_asr_q4_decode_scratch_bench.py --compare-bf16
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

GGUF = Path("/tmp/vibevoice-asr-q4km.gguf")
LM_FIXTURE = (Path(__file__).resolve().parent.parent
              / "tests" / "fixtures" / "vibevoice_asr" / "vibevoice_asr_lm.npz")
PINNED_HF_MODEL_ID = "microsoft/VibeVoice-ASR-HF"


def _prompt_rows(runner, lm) -> list[np.ndarray]:
    input_ids = np.asarray(lm["input_ids"])[0]
    positions = np.asarray(lm["audio_placeholder_positions"])
    audio = lm["audio_embeds"].astype(np.float32)
    rows = [runner.embed_row(int(t)) for t in input_ids]
    for p in positions:
        rows[p] = audio[p - positions[0]]
    return rows


def _prefill(runner, rows) -> None:
    """Batched prefill, then stage the final prompt row as the decode input."""
    from hipengine.core.memory import MemcpyKind, free
    from hipengine.loading.vibevoice_layout import f32_to_bf16_bits
    from hipengine.runtime.vibevoice_qwen2 import _upload

    total = len(rows)
    hidden = runner.spec.hidden_size
    prompt_buf = _upload(f32_to_bf16_bits(np.asarray(rows, dtype=np.float32)))
    try:
        runner.prefill_rows(prompt_buf, total, 0)
        runner.runtime.memcpy(
            runner._hidden.ptr,
            prompt_buf.ptr + (total - 1) * hidden * 2,
            hidden * 2,
            MemcpyKind.DEVICE_TO_DEVICE,
        )
    finally:
        free(prompt_buf)


def _run(runner, rows, steps: int, scratch) -> tuple[float, list[int], float]:
    """Prefill the prompt, then decode ``steps`` tokens.

    ``scratch`` is the caller-owned o_proj bf16 buffer to pass, or None to
    make the wrapper allocate one per layer. Returns (ms per decode step,
    generated token ids, logits checksum).
    """
    runner._o_proj_x_bf16 = scratch
    runner.reset()
    _prefill(runner, rows)

    tokens: list[int] = []
    checksum = 0.0
    position = len(rows)
    # First token comes from the prompt's own logits, outside the timed region.
    _, token = runner.logits_argmax()

    start = time.perf_counter()
    for _ in range(steps):
        runner.push_token(runner.embed_row(token), position)
        runner.forward_layers(position)
        logits, token = runner.logits_argmax()
        checksum += float(logits.sum())
        tokens.append(token)
        position += 1
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / steps, tokens, checksum


def _stats(values: list[float]) -> dict[str, object]:
    return {
        "median": round(statistics.median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "trials": [round(v, 2) for v in values],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--compare-bf16", action="store_true",
                        help="also measure the dense bf16 backbone in this process")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    from hipengine.loading.vibevoice_asr_gguf import load_vibevoice_qwen2_q4
    from hipengine.runtime.vibevoice_qwen2_q4 import VibevoiceQwen2Q4Runtime

    if not GGUF.is_file():
        raise SystemExit(f"missing Q4 GGUF: {GGUF}")
    if not LM_FIXTURE.is_file():
        raise SystemExit(f"missing LM fixture: {LM_FIXTURE}")

    with np.load(LM_FIXTURE) as data:
        lm = {k: data[k] for k in data.files}

    q4_weights = load_vibevoice_qwen2_q4(GGUF)
    q4 = VibevoiceQwen2Q4Runtime(q4_weights, max_context=512)
    bf16_weights = None
    bf16 = None
    try:
        q4_rows = _prompt_rows(q4, lm)
        scratch = q4._o_proj_x_bf16
        if scratch is None:
            raise SystemExit("Q4 runner exposes no o_proj scratch buffer")
        scratch_bytes = scratch.nbytes
        layers = len(q4.layers)

        # label -> (runner, rows, scratch)
        lanes: dict[str, tuple[object, list[np.ndarray], object]] = {
            "q4_fresh_malloc": (q4, q4_rows, None),
            "q4_persistent_scratch": (q4, q4_rows, scratch),
        }
        if args.compare_bf16:
            from hipengine.loading.hf_cache import resolve_model_path
            from hipengine.loading.vibevoice_asr import load_vibevoice_qwen2
            from hipengine.runtime.vibevoice_qwen2 import VibevoiceQwen2Runtime

            bf16_weights = load_vibevoice_qwen2(str(resolve_model_path(PINNED_HF_MODEL_ID)))
            bf16 = VibevoiceQwen2Runtime(bf16_weights, max_context=512)
            lanes["bf16_dense"] = (bf16, _prompt_rows(bf16, lm), None)

        order = list(lanes)
        samples: dict[str, list[float]] = {k: [] for k in order}
        tokens_out: dict[str, list[int]] = {}
        checksums: dict[str, float] = {}

        for _ in range(args.warmup):
            for label in order:
                runner, rows, scratch_arg = lanes[label]
                _run(runner, rows, args.steps, scratch_arg)

        for trial in range(args.trials):
            rotation = order[trial % len(order):] + order[:trial % len(order)]
            for label in rotation:
                runner, rows, scratch_arg = lanes[label]
                ms, tokens, checksum = _run(runner, rows, args.steps, scratch_arg)
                samples[label].append(ms)
                tokens_out[label] = tokens
                checksums[label] = checksum
    finally:
        if bf16 is not None:
            bf16.close()
        if bf16_weights is not None:
            # The dense bf16 loader returns a plain handle with no close().
            closer = getattr(bf16_weights, "close", None)
            if closer is not None:
                closer()
        q4.close()
        q4_weights.close()

    results = {label: _stats(v) for label, v in samples.items()}
    for label, st in results.items():
        print(f"{label:24s}: {st['median']:.2f} ms/token  {st['trials']}")

    fresh = results["q4_fresh_malloc"]["median"]
    reused = results["q4_persistent_scratch"]["median"]
    speedup = fresh / reused
    print(f"{'scratch delta':24s}: {fresh - reused:+.2f} ms/token "
          f"({(speedup - 1.0) * 100:+.2f}%)  speedup {speedup:.4f}x")

    identical = tokens_out["q4_fresh_malloc"] == tokens_out["q4_persistent_scratch"]
    rel_checksum = (abs(checksums["q4_fresh_malloc"] - checksums["q4_persistent_scratch"])
                    / max(abs(checksums["q4_fresh_malloc"]), 1e-9))
    print(f"{'q4 token chain identical':24s}: {identical}")
    print(f"{'q4 logits checksum rel':24s}: {rel_checksum:.3e}")

    payload = {
        "steps_per_trial": args.steps,
        "trials": args.trials,
        "warmup_runs_per_lane": args.warmup,
        "prompt_rows": len(q4_rows),
        "q4_layers": layers,
        "o_proj_scratch_bytes": scratch_bytes,
        "lanes": results,
        "scratch_delta_ms_per_token": round(fresh - reused, 2),
        "scratch_speedup": round(speedup, 4),
        "q4_token_chain_identical": identical,
        "q4_logits_checksum_rel_diff": rel_checksum,
    }
    if args.compare_bf16:
        bf16_med = results["bf16_dense"]["median"]
        payload["q4_vs_bf16_decode_speedup"] = round(bf16_med / reused, 4)
        payload["q4_fresh_vs_bf16_decode_speedup"] = round(bf16_med / fresh, 4)
        print(f"{'q4 scratch vs bf16':24s}: {bf16_med / reused:.4f}x "
              f"(was {bf16_med / fresh:.4f}x with per-layer malloc)")

    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {args.json}")

    if not identical:
        raise SystemExit("FAIL: the Q4 lanes produced different token chains")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
