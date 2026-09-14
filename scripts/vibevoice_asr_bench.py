"""Same-host VibeVoice-ASR speed comparison: hipEngine vs torch GPU.

Times both stacks end-to-end on identical PCM (feature-extraction
normalization applied identically): hipEngine front-end -> incremental
Qwen2 greedy, vs transformers VibeVoice-ASR on cuda bf16 greedy.
Reports per-phase and total wall time plus transcript equality.

Usage:
    python3 scripts/vibevoice_asr_bench.py [--pcm-file FILE] [--seconds 11]
        [--repeats 3]
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

IM_END_ID = 151645


def _load_pcm(args) -> tuple[np.ndarray, float]:
    if args.pcm_file is not None:
        import wave

        with wave.open(str(args.pcm_file), "rb") as fh:
            pcm = np.frombuffer(fh.readframes(fh.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        return pcm, len(pcm) / fh.getframerate()
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from vibevoice_asr_e2e import synth_pcm

    pcm = synth_pcm(args.seconds, 0)
    return pcm, args.seconds


def _normalize(pcm: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    if rms > 0:
        pcm = pcm * (10 ** (-25.0 / 20)) / (rms + 1e-8)
        peak = float(np.abs(pcm).max())
        if peak > 1.0:
            pcm = pcm / (peak + 1e-8)
    if len(pcm) % 3200:
        pcm = np.pad(pcm, (0, 3200 - len(pcm) % 3200))
    return pcm


def _host_identity() -> str:
    try:
        gpu = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=20
        ).stdout
        name = next(
            (ln.split(":", 1)[1].strip() for ln in gpu.splitlines() if "Marketing Name:" in ln and "AMD" in ln),
            "unknown",
        )
    except Exception:
        name = "unknown"
    return f"{socket.gethostname()} / {name} / {platform.machine()}"


def bench_hipengine(pcm: np.ndarray, duration: float, args) -> dict:
    from hipengine.loading.hf_cache import resolve_model_path
    from hipengine.loading.vibevoice_asr import (
        load_vibevoice_connector,
        load_vibevoice_encoder,
        load_vibevoice_qwen2,
    )
    from hipengine.runtime.vibevoice_encoder import VibevoiceFrontendRuntime
    from hipengine.runtime.vibevoice_qwen2 import VibevoiceQwen2Runtime

    from vibevoice_asr_e2e import AUDIO_TOKEN_ID, build_prompt

    specs, conns = {}, {}
    for tok in ("acoustic", "semantic"):
        specs[tok] = load_vibevoice_encoder(args.weights, tok)
        conns[tok] = load_vibevoice_connector(args.weights, tok)
    frontend = VibevoiceFrontendRuntime(
        specs["acoustic"][0], specs["acoustic"][1],
        specs["semantic"][0], specs["semantic"][1],
        conns["acoustic"], conns["semantic"],
    )
    lm_path = resolve_model_path(args.model)
    lm_weights = load_vibevoice_qwen2(str(lm_path))
    runner = VibevoiceQwen2Runtime(
        lm_weights, max_context=max(int(len(pcm) / 3200) + 64 + args.max_new_tokens, 1024)
    )
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(str(lm_path)) / "tokenizer.json"))

    frames = specs["acoustic"][0].frame_count(len(pcm))
    rng = np.random.default_rng(20260914)
    noise = rng.standard_normal((1, frames, specs["acoustic"][0].hidden_size)).astype(np.float32)
    scale = (specs["acoustic"][0].vae_std * rng.standard_normal(1)).astype(np.float32)

    prompt = build_prompt(duration, frames)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False).ids

    result = {"engine": "hipEngine (torch-free)"}
    timings = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        audio_embeds = frontend.forward(pcm, noise=noise[0], noise_scale=scale[0])
        t1 = time.perf_counter()

        rows = [runner.embed_row(int(t)) for t in input_ids]
        placeholder_idx = [i for i, t in enumerate(input_ids) if t == AUDIO_TOKEN_ID]
        if len(placeholder_idx) != audio_embeds.shape[0]:
            raise RuntimeError(
                f"template placeholders {len(placeholder_idx)} != front-end frames {audio_embeds.shape[0]}"
            )
        for j, i in enumerate(placeholder_idx):
            rows[i] = audio_embeds[j].astype(np.float32)
        from hipengine.runtime.vibevoice_qwen2 import greedy_generate as _greedy

        t2_pre = time.perf_counter()
        gen = _greedy(runner, rows, max_new_tokens=args.max_new_tokens, eos_token_id=IM_END_ID)
        t3 = time.perf_counter()
        # split prefill vs decode: rerun decode timing is included above; use
        # the greedy call's single timing for prompt+decode combined and the
        # token count for reporting
        timings.append(
            {"frontend_s": t1 - t0, "prompt_decode_s": t3 - t2_pre, "total_s": t3 - t0, "tokens": len(gen)}
        )
    result["timings"] = timings
    result["tokens"] = gen
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    result["text"] = text
    frontend.close()
    runner.close()
    return result


def bench_torch_gpu(pcm: np.ndarray, duration: float, args) -> dict:
    import torch

    from transformers import VibeVoiceAsrForConditionalGeneration, VibeVoiceAsrProcessor

    lm_path = str(resolve_model_path_torch(args))
    processor = VibeVoiceAsrProcessor.from_pretrained(lm_path)
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        lm_path, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    ).eval()
    torch.set_grad_enabled(False)

    pcm_f32 = pcm.astype(np.float32)
    result = {"engine": "torch GPU (cuda bf16, sdpa)"}
    timings = []
    text = ""
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        inputs = processor.apply_transcription_request(audio=pcm_f32)
        input_values = inputs["input_values"].to(device="cuda", dtype=torch.bfloat16)
        if input_values.ndim == 2:
            input_values = input_values.reshape(1, 1, -1)
        input_ids = inputs["input_ids"].to("cuda")
        padding_mask = inputs.get("padding_mask")
        if padding_mask is not None:
            padding_mask = padding_mask.to("cuda")
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        generated = model.generate(
            inputs=input_ids,
            input_values=input_values,
            padding_mask=padding_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        timings.append(
            {"preprocess_s": t1 - t0, "model_s": t2 - t1, "total_s": t2 - t0, "tokens": int(generated.shape[1] - inputs["input_ids"].shape[1])}
        )
    text = processor.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    result["timings"] = timings
    result["text"] = text
    return result


def resolve_model_path_torch(args) -> str:
    from hipengine.loading.hf_cache import resolve_model_path

    return str(resolve_model_path(args.model))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcm-file", type=Path, default=None)
    parser.add_argument("--seconds", type=float, default=11.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default="microsoft/VibeVoice-ASR-HF")
    parser.add_argument("--weights", default="microsoft/VibeVoice-ASR")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--skip-torch", action="store_true")
    args = parser.parse_args()

    pcm_raw, duration = _load_pcm(args)
    pcm = _normalize(pcm_raw)
    print(f"host: {_host_identity()}")
    print(f"audio: {len(pcm)} samples ({duration:.2f} s), frames {len(pcm)//3200}")

    hip = bench_hipengine(pcm, duration, args)
    best = min(hip["timings"], key=lambda t: t["total_s"])
    print(f"hipEngine: frontend {best['frontend_s']:.3f}s prompt+decode {best['prompt_decode_s']:.3f}s "
          f"({best['tokens']} tok) total {best['total_s']:.3f}s")
    print("hipEngine text:", hip["text"][:120])

    if not args.skip_torch:
        tor = bench_torch_gpu(pcm, duration, args)
        best_t = min(tor["timings"], key=lambda t: t["total_s"])
        print(f"torch GPU: model {best_t['model_s']:.3f}s total {best_t['total_s']:.3f}s")
        print("torch text:", tor["text"][:120])
        print(f"transcripts equal: {hip['text'].strip() == tor['text'].strip()}")
        ratio = best_t["total_s"] / best["total_s"]
        print(f"hipEngine/torch total ratio: {best['total_s']/best_t['total_s']:.2f}x (lower is better for hipEngine)")
    out = {"host": _host_identity(), "audio_seconds": duration, "hip": {k: hip[k] for k in ('timings','text')}}
    if not args.skip_torch:
        out["torch"] = {k: tor[k] for k in ('timings','text')}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
