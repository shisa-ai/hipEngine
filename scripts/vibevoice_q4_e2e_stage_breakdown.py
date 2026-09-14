#!/usr/bin/env python3
"""VibeVoice-ASR Q4_K_M end-to-end stage breakdown and real-time factor.

Replays the production Q4 path on frozen requests from the matched-request
cache (the same arrays the WER gate and the bf16 bench consume), timing each
stage separately so the real-time factor can be attributed:

  1. front-end        both causal encoders + connectors (+ PCM staging)
  2. prompt assembly  embed_row per token, audio scatter, f32->bf16, H2D upload
  3. prefill          batched prefill_rows + the first logits read
  4. decode           per-token push_token/forward_layers/logits_argmax

Real-time factor is total wall time / audio duration, so lower is better and
1.0 means it transcribes at the speed of playback. It is reported per request
and pooled; the pooled value is total time over total audio, which is the
duration-weighted mean.

Timing boundaries: each stage is bracketed by device synchronizes, so the
measured value is execution wall time rather than enqueue time. Weight loading,
tokenization and request preparation are outside every timer.

Usage:
    python3 scripts/vibevoice_q4_e2e_stage_breakdown.py --num-clips 20 \
        --request-cache /tmp/librispeech-requests --gguf /tmp/vibevoice-asr-q4km.gguf
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

AUDIO_TOKEN_ID = 151654
IM_END_ID = 151645
BENCH_SCRIPT = Path(__file__).resolve().parent / "vibevoice_asr_bench.py"
WER_SCRIPT = Path(__file__).resolve().parent / "vibevoice_asr_wer.py"


def _load_wer_module():
    spec = importlib.util.spec_from_file_location("wer", WER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["wer"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-clips", type=int, default=20)
    parser.add_argument("--clip-cache", type=Path, default=Path("/tmp/librispeech-clean-spread"),
                        help="24 kHz PCM cache read by scripts/vibevoice_asr_wer.py")
    parser.add_argument("--gguf", type=Path, default=Path("/tmp/vibevoice-asr-q4km.gguf"))
    parser.add_argument("--model", default="microsoft/VibeVoice-ASR-HF")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    from hipengine.core.memory import (MemcpyKind, copy_host_array_to_device,
                                       free, malloc)
    from hipengine.loading.hf_cache import resolve_model_path
    from hipengine.loading.vibevoice_asr import (load_vibevoice_connector,
                                                 load_vibevoice_encoder)
    from hipengine.loading.vibevoice_asr_gguf import load_vibevoice_qwen2_q4
    from hipengine.loading.vibevoice_layout import f32_to_bf16_bits
    from hipengine.runtime.vibevoice_encoder import VibevoiceFrontendRuntime
    from hipengine.runtime.vibevoice_qwen2_q4 import VibevoiceQwen2Q4Runtime
    from scripts.vibevoice_asr_bench import _read_request

    if not args.gguf.is_file():
        raise SystemExit(f"missing Q4 GGUF: {args.gguf}")

    # Requests are regenerated per clip through the bench's prepare lane, the
    # same matched-request protocol the WER gate uses. The on-disk request
    # cache under /tmp is not trusted: its stored array hashes do not validate.
    wer = _load_wer_module()
    clips = wer._load_clips(args.num_clips, args.clip_cache)
    if not clips:
        raise SystemExit(f"no clips under {args.clip_cache}")
    print(f"loaded {len(clips)} clips, "
          f"{sum(c['seconds'] for c in clips):.1f} s audio")

    model_path = resolve_model_path(args.model)
    specs = {k: load_vibevoice_encoder(str(model_path), k) for k in ("acoustic", "semantic")}
    frontend = VibevoiceFrontendRuntime(
        *specs["acoustic"], *specs["semantic"],
        load_vibevoice_connector(str(model_path), "acoustic"),
        load_vibevoice_connector(str(model_path), "semantic"),
    )
    weights = load_vibevoice_qwen2_q4(args.gguf)
    runner = VibevoiceQwen2Q4Runtime(weights, max_context=1024)
    sync = runner.runtime.device_synchronize
    hidden = runner.spec.hidden_size

    def transcribe(arrays) -> tuple[dict, int, int]:
        input_ids = arrays["input_ids"][0].tolist()
        pcm = arrays["pcm"]

        sync()
        t0 = time.perf_counter()
        embeds = frontend.forward(pcm, noise=arrays["noise"][0],
                                  noise_scale=arrays["scale"][0])
        sync()
        t1 = time.perf_counter()

        rows = [runner.embed_row(int(t)) for t in input_ids]
        positions = [j for j, t in enumerate(input_ids) if t == AUDIO_TOKEN_ID]
        for j, row in zip(positions, embeds):
            rows[j] = row
        total = len(rows)
        buf = malloc(total * hidden * 2)
        copy_host_array_to_device(
            buf, f32_to_bf16_bits(np.asarray(rows, dtype=np.float32)))
        sync()
        t2 = time.perf_counter()

        runner.reset()
        runner.prefill_rows(buf, total, 0)
        runner.runtime.memcpy(runner._hidden.ptr,
                              buf.ptr + (total - 1) * hidden * 2,
                              hidden * 2, MemcpyKind.DEVICE_TO_DEVICE)
        _, token = runner.logits_argmax()
        sync()
        t3 = time.perf_counter()
        free(buf)

        generated = 0
        for step in range(args.max_new_tokens):
            generated += 1
            if step + 1 == args.max_new_tokens or token == IM_END_ID:
                break
            pos = total + step
            runner.push_token(runner.embed_row(token), pos)
            runner.forward_layers(pos)
            _, token = runner.logits_argmax()
        sync()
        t4 = time.perf_counter()

        return ({"frontend": t1 - t0, "assembly": t2 - t1, "prefill": t3 - t2,
                 "decode": t4 - t3}, total, generated)

    def prepare(clip) -> tuple[dict, dict]:
        pcm = np.load(clip["wav"])
        with tempfile.TemporaryDirectory(prefix="vv-stage-") as tmp:
            root = Path(tmp)
            np.save(root / "pcm.npy", pcm)
            subprocess.run(
                [sys.executable, str(BENCH_SCRIPT), "--model", args.model,
                 "--pcm-npy", str(root / "pcm.npy"),
                 "--request", str(root / "request.npz"),
                 "--output", str(root / "out.json"), "--seed", str(args.seed),
                 "--max-new-tokens", str(args.max_new_tokens), "--lane", "prepare"],
                check=True, capture_output=True,
            )
            return _read_request(root / "request.npz")

    # Warm up once so lazy state (JIT, arena growth) is outside the sample.
    arrays0, _ = prepare(clips[0])
    transcribe(arrays0)

    rows_out = []
    for clip in clips:
        arrays, meta = prepare(clip)
        stages, prompt_tokens, generated = transcribe(arrays)
        audio = float(meta.get("audio_seconds") or clip["seconds"])
        total_s = sum(stages.values())
        rows_out.append({
            "clip_id": clip["clip_id"],
            "audio_seconds": round(audio, 3),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated,
            "frontend_ms": round(stages["frontend"] * 1e3, 1),
            "assembly_ms": round(stages["assembly"] * 1e3, 1),
            "prefill_ms": round(stages["prefill"] * 1e3, 1),
            "decode_ms": round(stages["decode"] * 1e3, 1),
            "total_s": round(total_s, 3),
            "rtf": round(total_s / audio, 4),
            "decode_ms_per_token": round(stages["decode"] * 1e3 / max(generated, 1), 2),
        })
        print(f"[{len(rows_out):3d}/{len(clips)}] {audio:6.2f}s audio "
              f"{prompt_tokens:4d}+{generated:3d} tok  total {total_s:5.2f}s  "
              f"RTF {total_s / audio:5.3f}")

    frontend.close()
    runner.close()
    weights.close()

    def med(key):
        return statistics.median(r[key] for r in rows_out)

    audio_total = sum(r["audio_seconds"] for r in rows_out)
    time_total = sum(r["total_s"] for r in rows_out)
    stage_totals = {k: sum(r[f"{k}_ms"] for r in rows_out)
                    for k in ("frontend", "assembly", "prefill", "decode")}
    gen_total = sum(r["generated_tokens"] for r in rows_out)

    print(f"\nclips {len(rows_out)}, {audio_total:.1f} s audio, "
          f"{gen_total} generated tokens")
    print(f"{'stage':12s} {'total ms':>10s} {'share':>7s} {'median ms':>10s}")
    for k, v in stage_totals.items():
        print(f"{k:12s} {v:10.1f} {v / (time_total * 1e3) * 100:6.1f}% "
              f"{med(k + '_ms'):10.1f}")
    print(f"{'total':12s} {time_total * 1e3:10.1f} {100.0:6.1f}% "
          f"{med('total_s') * 1e3:10.1f}")
    print(f"\nRTF pooled (total time / total audio) : {time_total / audio_total:.4f}")
    print(f"RTF median of per-clip RTF           : "
          f"{statistics.median(r['rtf'] for r in rows_out):.4f}")
    print(f"decode ms/token (median)             : {med('decode_ms_per_token'):.2f}")
    print(f"decode tokens/s (median)             : "
          f"{1000.0 / med('decode_ms_per_token'):.2f}")
    print(f"prompt tokens (median)               : {med('prompt_tokens'):.0f}")
    print(f"generated tokens (median)            : {med('generated_tokens'):.0f}")

    payload = {
        "model": args.model,
        "quant": "Q4_K_M GGUF backbone (raw blocks), bf16 front-end",
        "gguf": str(args.gguf),
        "clips": len(rows_out),
        "audio_seconds_total": round(audio_total, 1),
        "generated_tokens_total": gen_total,
        "stage_totals_ms": {k: round(v, 1) for k, v in stage_totals.items()},
        "stage_share_pct": {k: round(v / (time_total * 1e3) * 100, 1)
                            for k, v in stage_totals.items()},
        "total_ms": round(time_total * 1e3, 1),
        "rtf_pooled": round(time_total / audio_total, 4),
        "rtf_median_per_clip": round(statistics.median(r["rtf"] for r in rows_out), 4),
        "rtf_min": min(r["rtf"] for r in rows_out),
        "rtf_max": max(r["rtf"] for r in rows_out),
        "decode_ms_per_token_median": round(med("decode_ms_per_token"), 2),
        "decode_tokens_per_s_median": round(1000.0 / med("decode_ms_per_token"), 2),
        "per_clip": rows_out,
    }
    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
