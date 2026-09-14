#!/usr/bin/env python3
"""hipEngine-Q4 lane of the VibeVoice WER gate.

Uses the same matched frozen-request protocol as the bench: per clip the
prepare subprocess (HF processor, fixed seed) writes request.npz, and
this process runs the torch-free front-end + Q4_K_M backbone runner on
the identical arrays. Scores WER exactly like vibevoice_asr_wer.py
(Whisper EnglishTextNormalizer + jiwer over the parsed Content).

Usage:
    python3 scripts/vibevoice_asr_wer_q4.py --num-clips 50 \
        --cache-dir /tmp/librispeech-clean-spread --gguf /tmp/vibevoice-asr-q4km.gguf \
        --out /tmp/vibevoice-wer-50clips-q4.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

WER_SCRIPT = Path(__file__).resolve().parent / "vibevoice_asr_wer.py"
BENCH_SCRIPT = Path(__file__).resolve().parent / "vibevoice_asr_bench.py"


def _load_wer_module():
    spec = importlib.util.spec_from_file_location("wer", WER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["wer"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-clips", type=int, default=50)
    parser.add_argument("--cache-dir", default="/tmp/librispeech-clean-spread")
    parser.add_argument("--gguf", default="/tmp/vibevoice-asr-q4km.gguf")
    parser.add_argument("--model", default="microsoft/VibeVoice-ASR-HF")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    wer = _load_wer_module()
    clips = wer._load_clips(args.num_clips, Path(args.cache_dir))
    print(f"loaded {len(clips)} clips, {sum(c['seconds'] for c in clips):.1f} s audio")

    from scripts.vibevoice_asr_e2e import AUDIO_TOKEN_ID, IM_END_ID
    from hipengine.loading.hf_cache import resolve_model_path
    from hipengine.loading.vibevoice_asr import (
        load_vibevoice_connector,
        load_vibevoice_encoder,
    )
    from hipengine.runtime.vibevoice_encoder import VibevoiceFrontendRuntime
    from hipengine.runtime.vibevoice_qwen2 import greedy_generate
    from hipengine.runtime.vibevoice_qwen2_q4 import VibevoiceQwen2Q4Runtime
    from hipengine.loading.vibevoice_asr_gguf import load_vibevoice_qwen2_q4
    from tokenizers import Tokenizer

    model_path = resolve_model_path(args.model)
    specs = {k: load_vibevoice_encoder(str(model_path), k) for k in ("acoustic", "semantic")}
    frontend = VibevoiceFrontendRuntime(
        *specs["acoustic"], *specs["semantic"],
        load_vibevoice_connector(str(model_path), "acoustic"),
        load_vibevoice_connector(str(model_path), "semantic"),
    )
    weights = load_vibevoice_qwen2_q4(args.gguf)
    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    # one runner for all clips (weight buffers are owned once; reset per clip)
    runner = VibevoiceQwen2Q4Runtime(weights, max_context=1024)

    hyps = []
    timings = []
    for i, clip in enumerate(clips):
        pcm = np.load(clip["wav"])
        with tempfile.TemporaryDirectory(prefix="vv-q4-") as tmp:
            root = Path(tmp)
            np.save(root / "pcm.npy", pcm)
            subprocess.run(
                [sys.executable, str(BENCH_SCRIPT), "--model", args.model,
                 "--pcm-npy", str(root / "pcm.npy"), "--request", str(root / "request.npz"),
                 "--output", str(root / "out.json"), "--seed", str(args.seed),
                 "--max-new-tokens", str(args.max_new_tokens), "--lane", "prepare"],
                check=True, capture_output=True,
            )
            from scripts.vibevoice_asr_bench import _read_request
            arrays, meta = _read_request(root / "request.npz")
            input_ids = arrays["input_ids"][0].tolist()
            if len(input_ids) + args.max_new_tokens > runner.max_context:
                raise ValueError("clip exceeds runner max_context")
            t0 = time.perf_counter()
            embeds = frontend.forward(
                arrays["pcm"], noise=arrays["noise"][0],
                noise_scale=arrays["scale"][0],
            )
            rows = [runner.embed_row(t) for t in input_ids]
            positions = [j for j, t in enumerate(input_ids) if t == AUDIO_TOKEN_ID]
            if len(positions) != len(embeds):
                raise ValueError("audio placeholder count mismatch")
            for j, row in zip(positions, embeds):
                rows[j] = row
            ids = greedy_generate(runner, rows, max_new_tokens=args.max_new_tokens,
                                  eos_token_id=IM_END_ID)
            elapsed = time.perf_counter() - t0
        text = tokenizer.decode(ids, skip_special_tokens=True).strip()
        timings.append(elapsed)
        hyps.append(wer._transcription_only(text))
        print(f"[q4 {i+1}/{len(clips)}] {elapsed:.2f}s {hyps[-1][:60]!r}")

    runner.close()
    frontend.close()
    refs = [c["text"] for c in clips]
    total_wer = wer._wer(refs, hyps)
    print(f"hipEngine-Q4 WER: {total_wer:.2f}% "
          f"(mean {np.mean(timings):.2f} s/clip)")

    out = {
        "systems": {
            "hip_q4": {
                "wer": total_wer,
                "hypotheses": hyps,
                "per_clip": [
                    {"clip_id": c["clip_id"],
                     "wer": wer._wer([c["text"]], [h]),
                     "seconds": elapsed}
                    for c, h, elapsed in zip(clips, hyps, timings)
                ],
            }
        },
        "clips": len(clips),
        "clip_ids": [c["clip_id"] for c in clips],
        "ref_texts": refs,
        "protocol": {
            "gguf": args.gguf,
            "matched_request": "bench prepare subprocess, fixed seed",
            "frontend": "bf16 dense (unchanged)",
        },
    }
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
