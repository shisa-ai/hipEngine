#!/usr/bin/env python3
"""Comparative full-benchmark driver: LibriSpeech test-clean, persistent lanes.

Scales the 50-clip gate to a statistically meaningful clip count with
model/persistent lanes (one process per lane, one prepare pass), because
the bench's per-clip subprocess protocol reloads models each clip.

Lanes (choose any subset):
- torch    : transformers bf16 eager GPU (reference)
- hip      : hipEngine bf16 backbone (hipblaslt batched prefill)
- hipq4    : hipEngine Q4_K_M backbone (incremental prefill, naive gemv)

Per clip the driver reproduces the bench's matched-request protocol
in-process: identical processor call, seed, noise and scale arrays, and
hash verification. Per-stage timings are recorded for every clip so the
same run doubles as the matched-protocol speed comparison.

Usage:
    python3 scripts/vibevoice_asr_wer_full.py --num-clips 500 \
        --lanes torch hip hipq4 --out /tmp/vibevoice-wer-full.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

WER_SCRIPT = Path(__file__).resolve().parent / "vibevoice_asr_wer.py"


def _load_wer():
    spec = importlib.util.spec_from_file_location("wer", WER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["wer"] = module
    spec.loader.exec_module(module)
    return module


def array_hash(array: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def prepare_all(clips, model_path, seed, max_new_tokens, cache_dir):
    """One processor pass over all clips -> request.npz per clip (hash-checked)."""
    import torch
    from transformers import VibeVoiceAsrProcessor
    from scripts.vibevoice_asr_e2e import IM_END_ID  # noqa: F401

    processor = VibeVoiceAsrProcessor.from_pretrained(str(model_path))
    config = json.loads((model_path / "config.json").read_text())
    vae_std = config["acoustic_tokenizer_encoder_config"]["vae_std"]
    width = config["acoustic_tokenizer_encoder_config"]["hidden_size"]
    requests = {}
    for i, clip in enumerate(clips):
        out_path = cache_dir / f"{clip['clip_id']}.request.npz"
        if out_path.is_file():
            requests[clip["clip_id"]] = out_path
            continue
        raw = np.load(clip["wav"])
        inputs = processor.apply_transcription_request(audio=raw, prompt=None)
        pcm = inputs["input_values"].to(torch.bfloat16).float().numpy().reshape(-1)
        frames = (raw.size + 3199) // 3200
        rng = np.random.default_rng(seed)
        noise = torch.tensor(rng.standard_normal((1, frames, width)), dtype=torch.bfloat16)
        base_scale = torch.tensor(rng.standard_normal(1), dtype=torch.bfloat16)
        scale = base_scale * vae_std
        arrays = dict(pcm=pcm, input_ids=inputs["input_ids"].numpy(),
                      padding_mask=inputs["padding_mask"].numpy(),
                      noise=noise.float().numpy(),
                      base_scale=base_scale.float().numpy(), scale=scale.float().numpy())
        np.savez(out_path, **arrays)
        Path(str(out_path).replace(".npz", ".json")).write_text(json.dumps({
            "seed": seed, "audio_seconds": raw.size / 24000,
            "hashes": {k: array_hash(v) for k, v in arrays.items()},
        }))
        requests[clip["clip_id"]] = out_path
        if (i + 1) % 50 == 0:
            print(f"prepared {i+1}/{len(clips)}")
    return requests


def read_request(path):
    arrays = dict(np.load(path))
    meta = json.loads(Path(str(path).replace(".npz", ".json")).read_text())
    for k, v in arrays.items():
        if array_hash(v) != meta["hashes"][k]:
            raise ValueError(f"request hash mismatch for {k}")
    return arrays, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-clips", type=int, default=500)
    parser.add_argument("--lanes", nargs="+", default=["torch", "hip", "hipq4"])
    parser.add_argument("--cache-dir", default="/tmp/librispeech-clean-spread")
    parser.add_argument("--request-cache", default="/tmp/librispeech-requests")
    parser.add_argument("--model", default="microsoft/VibeVoice-ASR-HF")
    parser.add_argument("--gguf", default="/tmp/vibevoice-asr-q4km.gguf")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    wer = _load_wer()
    clips = wer._load_clips(args.num_clips, Path(args.cache_dir))
    print(f"loaded {len(clips)} clips, {sum(c['seconds'] for c in clips):.1f} s audio")

    from hipengine.loading.hf_cache import resolve_model_path
    model_path = resolve_model_path(args.model)
    request_cache = Path(args.request_cache)
    request_cache.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    requests = prepare_all(clips, model_path, args.seed, args.max_new_tokens, request_cache)
    print(f"prepare done in {time.perf_counter()-t0:.0f}s")

    results = {"clips": len(clips), "clip_ids": [c["clip_id"] for c in clips],
               "ref_texts": [c["text"] for c in clips], "systems": {}}

    from scripts.vibevoice_asr_e2e import AUDIO_TOKEN_ID, IM_END_ID

    for lane in args.lanes:
        print(f"=== lane {lane}")
        hyps, timings = [], []
        if lane == "torch":
            import torch
            from transformers import VibeVoiceAsrForConditionalGeneration
            from scripts.vibevoice_asr_bench import recorded_noise
            torch.set_grad_enabled(False)
            model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
                str(model_path), torch_dtype=torch.bfloat16,
                device_map="cuda", attn_implementation="eager").eval()
            from transformers import VibeVoiceAsrProcessor
            proc = VibeVoiceAsrProcessor.from_pretrained(str(model_path))
            for i, clip in enumerate(clips):
                arrays, meta = read_request(requests[clip["clip_id"]])
                input_ids = arrays["input_ids"][0].tolist()
                noise = torch.from_numpy(arrays["noise"])
                base_scale = torch.from_numpy(arrays["base_scale"])
                t0 = time.perf_counter()
                ids = torch.tensor([input_ids], device="cuda")
                pcm = torch.from_numpy(arrays["pcm"]).reshape(1, 1, -1).to(device="cuda", dtype=torch.bfloat16)
                mask = torch.from_numpy(arrays["padding_mask"]).to("cuda")
                with recorded_noise(torch, noise, base_scale):
                    out = model.generate(inputs=ids, input_values=pcm, padding_mask=mask,
                                         max_new_tokens=args.max_new_tokens,
                                         do_sample=False, eos_token_id=IM_END_ID)
                torch.cuda.synchronize()
                gen = out[0, len(input_ids):].tolist()
                timings.append({"inference_s": time.perf_counter() - t0, "tokens": len(gen)})
                hyps.append(proc.decode(gen, skip_special_tokens=True).strip())
                print(f"[torch {i+1}/{len(clips)}] {timings[-1]['inference_s']:.2f}s {hyps[-1][:50]!r}")
            del model
            torch.cuda.empty_cache()
        elif lane in ("hip", "hipq4"):
            from tokenizers import Tokenizer
            from hipengine.loading.vibevoice_asr import (
                load_vibevoice_connector, load_vibevoice_encoder)
            from hipengine.runtime.vibevoice_encoder import VibevoiceFrontendRuntime
            from hipengine.runtime.vibevoice_qwen2 import (
                VibevoiceQwen2Runtime, greedy_generate)
            specs = {k: load_vibevoice_encoder(str(model_path), k) for k in ("acoustic", "semantic")}
            frontend = VibevoiceFrontendRuntime(
                *specs["acoustic"], *specs["semantic"],
                load_vibevoice_connector(str(model_path), "acoustic"),
                load_vibevoice_connector(str(model_path), "semantic"))
            tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
            if lane == "hip":
                from hipengine.loading.vibevoice_asr import load_vibevoice_qwen2
                runner = VibevoiceQwen2Runtime(
                    load_vibevoice_qwen2(str(model_path)), max_context=1024,
                    prefill_variant="hipblaslt")
            else:
                from hipengine.runtime.vibevoice_qwen2_q4 import VibevoiceQwen2Q4Runtime
                from hipengine.loading.vibevoice_asr_gguf import load_vibevoice_qwen2_q4
                runner = VibevoiceQwen2Q4Runtime(
                    load_vibevoice_qwen2_q4(args.gguf), max_context=1024)
            for i, clip in enumerate(clips):
                arrays, meta = read_request(requests[clip["clip_id"]])
                input_ids = arrays["input_ids"][0].tolist()
                t0 = time.perf_counter()
                embeds = frontend.forward(arrays["pcm"], noise=arrays["noise"][0],
                                          noise_scale=arrays["scale"][0])
                t_front = time.perf_counter() - t0
                rows = [runner.embed_row(t) for t in input_ids]
                positions = [j for j, t in enumerate(input_ids) if t == AUDIO_TOKEN_ID]
                for j, row in zip(positions, embeds):
                    rows[j] = row
                gen = greedy_generate(runner, rows, max_new_tokens=args.max_new_tokens,
                                      eos_token_id=IM_END_ID)
                timings.append({"frontend_s": t_front,
                                "rest_s": time.perf_counter() - t0 - t_front,
                                "tokens": len(gen)})
                hyps.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
                print(f"[{lane} {i+1}/{len(clips)}] {time.perf_counter()-t0:.2f}s {hyps[-1][:50]!r}")
            runner.close()
            frontend.close()
        malformed = [c["clip_id"] for c, h in zip(clips, hyps)
                     if wer.parse_transcript(h)[1] != "ok"]
        results["systems"][lane] = {
            "hypotheses": hyps,
            "timings": timings,
            "malformed_transcripts": malformed,
            "wer_fraction": wer._wer(results["ref_texts"], hyps),
            "wer_pct": wer._wer_pct(results["ref_texts"], hyps),
        }

    for lane, data in results["systems"].items():
        print(f"{lane} WER: {data['wer_pct']:.3f}%"
              + (f"  [{len(data['malformed_transcripts'])} malformed]"
                 if data["malformed_transcripts"] else ""))
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
