#!/usr/bin/env python3
"""Comparative full-benchmark driver: LibriSpeech test-clean, isolated lanes.

Scales the 50-clip WER gate to a statistically meaningful clip count. The
bench's per-clip subprocess protocol reloads the model for every clip, which
makes a full-set run impractical (the bf16 lane would need ~26 h), so this
driver prepares every request once and loads each lane's model once.

Lanes (choose any subset):
- torch    : transformers bf16 eager GPU (reference)
- hip      : hipEngine bf16 backbone (hipBLASLt batched prefill)
- hipq4    : hipEngine Q4_K_M backbone

Each lane runs in its **own subprocess**. That is not cosmetic: the bench
protocol requires the HIP lane to be torch-free, and the Q4 lane reuses
device state that a torch context can disturb. A single-process driver
would silently violate both.

Requests are cached per clip and revalidated against the current seed,
model, prompt length and audio hash; a stale cache is regenerated rather
than silently reused. The first ``--warmup`` clips are still scored but are
excluded from the reported timing summary.

Usage:
    python3 scripts/vibevoice_asr_wer_full.py --num-clips 500 \
        --lanes torch hip hipq4 --out /tmp/vibevoice-wer-full.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
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
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _cache_key(seed: int, model: str, max_new_tokens: int, audio_hash: str) -> dict:
    return {"seed": seed, "model": str(model), "prompt": None,
            "max_new_tokens": max_new_tokens, "audio_sha256": audio_hash}


def _cache_is_current(sidecar: Path, key: dict) -> bool:
    try:
        recorded = json.loads(sidecar.read_text())
    except Exception:
        return False
    return all(recorded.get(k) == v for k, v in key.items())


def prepare_all(clips, model_path, seed, max_new_tokens, cache_dir, force=False):
    """One processor pass over all clips -> request.npz per clip (hash-checked).

    A cached request is reused only when its recorded seed, model, prompt
    length and audio hash all match this run.
    """
    import torch
    from transformers import VibeVoiceAsrProcessor

    processor = VibeVoiceAsrProcessor.from_pretrained(str(model_path))
    config = json.loads((model_path / "config.json").read_text())
    vae_std = config["acoustic_tokenizer_encoder_config"]["vae_std"]
    width = config["acoustic_tokenizer_encoder_config"]["hidden_size"]
    requests = {}
    prepared = reused = 0
    for i, clip in enumerate(clips):
        raw = np.load(clip["wav"])
        key = _cache_key(seed, model_path, max_new_tokens, array_hash(raw))
        out_path = cache_dir / f"{clip['clip_id']}.request.npz"
        sidecar = Path(str(out_path).replace(".npz", ".json"))
        if not force and out_path.is_file() and _cache_is_current(sidecar, key):
            requests[clip["clip_id"]] = out_path
            reused += 1
            continue
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
        sidecar.write_text(json.dumps(dict(key, audio_seconds=raw.size / 24000,
                                          hashes={k: array_hash(v) for k, v in arrays.items()})))
        requests[clip["clip_id"]] = out_path
        prepared += 1
        if (i + 1) % 50 == 0:
            print(f"prepared {i+1}/{len(clips)}")
    print(f"requests: {prepared} prepared, {reused} reused from cache")
    return requests


def read_request(path):
    arrays = dict(np.load(path))
    meta = json.loads(Path(str(path).replace(".npz", ".json")).read_text())
    for k, v in arrays.items():
        if array_hash(v) != meta["hashes"][k]:
            raise ValueError(f"request hash mismatch for {k}")
    return arrays, meta


def run_lane_worker(args, wer) -> int:
    """Run one lane in this process and write its records to --lane-out."""
    clips = wer._load_clips(args.num_clips, Path(args.cache_dir))
    request_cache = Path(args.request_cache)
    lane = args.lane_worker
    hyps, timings = [], []

    if lane == "torch":
        import torch
        from transformers import VibeVoiceAsrForConditionalGeneration, VibeVoiceAsrProcessor
        from scripts.vibevoice_asr_bench import recorded_noise

        torch.set_grad_enabled(False)
        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            str(args.model), torch_dtype=torch.bfloat16,
            device_map="cuda", attn_implementation="eager").eval()
        proc = VibeVoiceAsrProcessor.from_pretrained(str(args.model))
        for i, clip in enumerate(clips):
            arrays, _ = read_request(request_cache / f"{clip['clip_id']}.request.npz")
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
                                     do_sample=False, eos_token_id=args.im_end_id)
            torch.cuda.synchronize()
            gen = out[0, len(input_ids):].tolist()
            timings.append({"seconds": time.perf_counter() - t0, "tokens": len(gen)})
            hyps.append(proc.decode(gen, skip_special_tokens=True).strip())
            print(f"[torch {i+1}/{len(clips)}] {timings[-1]['seconds']:.2f}s", flush=True)
        del model
    else:
        from tokenizers import Tokenizer
        from hipengine.loading.vibevoice_asr import (
            load_vibevoice_connector, load_vibevoice_encoder)
        from hipengine.runtime.vibevoice_encoder import VibevoiceFrontendRuntime
        from hipengine.runtime.vibevoice_qwen2 import greedy_generate

        specs = {k: load_vibevoice_encoder(str(args.model), k) for k in ("acoustic", "semantic")}
        frontend = VibevoiceFrontendRuntime(
            *specs["acoustic"], *specs["semantic"],
            load_vibevoice_connector(str(args.model), "acoustic"),
            load_vibevoice_connector(str(args.model), "semantic"))
        tokenizer = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
        weights = None
        if lane == "hip":
            from hipengine.loading.vibevoice_asr import load_vibevoice_qwen2
            from hipengine.runtime.vibevoice_qwen2 import VibevoiceQwen2Runtime

            runner = VibevoiceQwen2Runtime(load_vibevoice_qwen2(str(args.model)),
                                           max_context=1024, prefill_variant="hipblaslt")
        else:
            from hipengine.loading.vibevoice_asr_gguf import load_vibevoice_qwen2_q4
            from hipengine.runtime.vibevoice_qwen2_q4 import VibevoiceQwen2Q4Runtime

            weights = load_vibevoice_qwen2_q4(args.gguf)
            runner = VibevoiceQwen2Q4Runtime(weights, max_context=1024)
        try:
            for i, clip in enumerate(clips):
                arrays, _ = read_request(request_cache / f"{clip['clip_id']}.request.npz")
                input_ids = arrays["input_ids"][0].tolist()
                t0 = time.perf_counter()
                embeds = frontend.forward(arrays["pcm"], noise=arrays["noise"][0],
                                          noise_scale=arrays["scale"][0])
                t_front = time.perf_counter() - t0
                rows = [runner.embed_row(t) for t in input_ids]
                positions = [j for j, t in enumerate(input_ids) if t == args.audio_token_id]
                for j, row in zip(positions, embeds):
                    rows[j] = row
                gen = greedy_generate(runner, rows, max_new_tokens=args.max_new_tokens,
                                      eos_token_id=args.im_end_id)
                timings.append({"frontend_s": t_front,
                                "seconds": time.perf_counter() - t0,
                                "tokens": len(gen)})
                hyps.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
                print(f"[{lane} {i+1}/{len(clips)}] {timings[-1]['seconds']:.2f}s", flush=True)
        finally:
            runner.close()
            if weights is not None:
                weights.close()
            frontend.close()

    if lane.startswith("hip") and "torch" in sys.modules:
        raise RuntimeError("torch was imported in the HIP lane")

    malformed = [c["clip_id"] for c, h in zip(clips, hyps)
                 if wer.parse_transcript(h)[1] != "ok"]
    refs = [c["text"] for c in clips]
    # A schema failure must not be scored as an ordinary transcription error,
    # so malformed generations are excluded from WER and reported separately
    # (the gate is "malformed == 0", not "WER looks fine"). Excluding them
    # must never be silent, or a lane that emits garbage would score better.
    ok_pairs = [(c["text"], h) for c, h in zip(clips, hyps)
                if wer.parse_transcript(h)[1] == "ok"]
    ok_refs = [r for r, _ in ok_pairs]
    ok_hyps = [h for _, h in ok_pairs]
    scored = timings[args.warmup:]
    record = {
        "hypotheses": hyps,
        "timings": timings,
        "warmup_clips": args.warmup,
        "malformed_transcripts": malformed,
        "clips_scored_for_wer": len(ok_hyps),
        "wer_excludes_malformed": bool(malformed),
        "wer_fraction": wer._wer(ok_refs, ok_hyps) if ok_hyps else None,
        "wer_pct": wer._wer_pct(ok_refs, ok_hyps) if ok_hyps else None,
        "mean_seconds_excl_warmup": (sum(t["seconds"] for t in scored) / len(scored)
                                     if scored else None),
        "clips_scored_for_timing": len(scored),
    }
    Path(args.lane_out).write_text(json.dumps(record, indent=2))
    print(f"{lane} WER: {record['wer_pct']:.3f}% over {len(ok_hyps)}/{len(clips)} clips"
          + (f"  [{len(malformed)} malformed excluded: {malformed}]" if malformed else ""))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-clips", type=int, default=500)
    parser.add_argument("--lanes", nargs="+", default=["torch", "hip", "hipq4"],
                        choices=["torch", "hip", "hipq4"])
    parser.add_argument("--cache-dir", default="/tmp/librispeech-clean-spread")
    parser.add_argument("--request-cache", default="/tmp/librispeech-requests")
    parser.add_argument("--model", default="microsoft/VibeVoice-ASR-HF")
    parser.add_argument("--gguf", default="/tmp/vibevoice-asr-q4km.gguf")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2,
                        help="leading clips scored but excluded from timing summary")
    parser.add_argument("--force-prepare", action="store_true",
                        help="regenerate the request cache even if it looks current")
    parser.add_argument("--out", required=False)
    # internal worker arguments
    parser.add_argument("--lane-worker", choices=["torch", "hip", "hipq4"], default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--lane-out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    wer = _load_wer()
    if args.lane_worker:
        from scripts.vibevoice_asr_e2e import AUDIO_TOKEN_ID, IM_END_ID

        args.audio_token_id, args.im_end_id = AUDIO_TOKEN_ID, IM_END_ID
        return run_lane_worker(args, wer)

    if not args.out:
        parser.error("--out is required for a driver run")

    clips = wer._load_clips(args.num_clips, Path(args.cache_dir))
    print(f"loaded {len(clips)} clips, {sum(c['seconds'] for c in clips):.1f} s audio")

    from hipengine.loading.hf_cache import resolve_model_path

    model_path = resolve_model_path(args.model)
    request_cache = Path(args.request_cache)
    request_cache.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    prepare_all(clips, model_path, args.seed, args.max_new_tokens, request_cache,
                force=args.force_prepare)
    print(f"prepare done in {time.perf_counter()-t0:.0f}s")

    results = {"clips": len(clips), "clip_ids": [c["clip_id"] for c in clips],
               "ref_texts": [c["text"] for c in clips],
               "audio_seconds": sum(c["seconds"] for c in clips),
               "protocol": {"seed": args.seed, "model": str(model_path),
                            "max_new_tokens": args.max_new_tokens,
                            "warmup_clips": args.warmup,
                            "isolation": "one subprocess per lane",
                            "request_cache": str(request_cache)},
               "systems": {}}

    for lane in args.lanes:
        lane_out = request_cache / f"lane-{lane}.json"
        print(f"=== lane {lane} (subprocess)")
        subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        "--lane-worker", lane, "--lane-out", str(lane_out),
                        "--num-clips", str(args.num_clips),
                        "--cache-dir", args.cache_dir,
                        "--request-cache", str(request_cache),
                        "--model", str(model_path), "--gguf", args.gguf,
                        "--seed", str(args.seed),
                        "--max-new-tokens", str(args.max_new_tokens),
                        "--warmup", str(args.warmup)], check=True)
        results["systems"][lane] = json.loads(lane_out.read_text())

    for lane, data in results["systems"].items():
        n_ok = data.get("clips_scored_for_wer")
        mal = data.get("malformed_transcripts") or []
        wer_txt = ("n/a" if data["wer_pct"] is None
                   else f"{data['wer_pct']:.3f}%")
        print(f"{lane} WER: {wer_txt} over {n_ok}/{results['clips']} clips  "
              f"mean {data['mean_seconds_excl_warmup']:.2f} s/clip "
              f"(excl. {data['warmup_clips']} warmup)"
              + (f"  [{len(mal)} malformed excluded]" if mal else ""))
    total_malformed = sum(len(d.get("malformed_transcripts") or [])
                          for d in results["systems"].values())
    results["malformed_transcripts_total"] = total_malformed
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}  (malformed transcripts across lanes: {total_malformed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
