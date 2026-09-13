"""Torch-free E2E VibeVoice-ASR transcription harness (gfx1100/gfx1151).

Pipeline: raw mono 24 kHz PCM -> hipEngine front-end (both causal encoders +
connectors, numpy-RNG acoustic sampling) -> chat-template prompt tokens ->
hipEngine incremental Qwen2 greedy decode -> JSON transcript. The tokenizer
is loaded from the HF artifact's ``tokenizer.json`` via the ``tokenizers``
library (boundary-only dependency, torch-free).

Usage:
    python3 scripts/vibevoice_asr_e2e.py [--seconds 4] [--seed 0]
        [--model microsoft/VibeVoice-ASR-HF]
        [--pcm-file out.wav (optional; synthetic speech-like PCM otherwise)]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

AUDIO_TOKEN = "<|box_start|>"
AUDIO_BOS = "<|object_ref_start|>"
AUDIO_EOS = "<|object_ref_end|>"
AUDIO_TOKEN_ID = 151648
IM_END_ID = 151645
SYSTEM_PROMPT = (
    "You are a helpful assistant that transcribes audio input into text "
    "output in JSON format."
)


def synth_pcm(seconds: float, seed: int) -> np.ndarray:
    """Deterministic speech-like excitation (same family as the oracle)."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds * 24_000))
    t = np.arange(n, dtype=np.float64) / 24_000
    f0 = 120.0 + 40.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / 24_000
    sig = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    sig *= 0.6 + 0.4 * np.sin(2 * np.pi * 3.1 * t)
    sig += 0.05 * rng.standard_normal(n).astype(np.float64)
    return sig.astype(np.float32)


def build_prompt(duration: float, frames: int, *, context: str | None = None) -> str:
    """Hand-rolled chat template matching the processor's transcription request."""
    audio_block = f"{AUDIO_BOS}{AUDIO_TOKEN * frames}{AUDIO_EOS}\n"
    if context:
        info = (
            f"This is a {duration:.2f} seconds audio, with extra info: {context}\n\n"
            "Please transcribe it with these keys: Start time, End time, Speaker ID, Content"
        )
    else:
        info = (
            f"This is a {duration:.2f} seconds audio, please transcribe it with "
            "these keys: Start time, End time, Speaker ID, Content"
        )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{audio_block}{info}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def parse_transcript(text: str) -> list[dict] | None:
    stripped = text.strip()
    if stripped.startswith("assistant"):
        stripped = stripped[len("assistant"):].strip()
    if not stripped.startswith("["):
        return None
    try:
        segments = json.loads(stripped)
    except json.JSONDecodeError:
        # tolerate truncation: keep the last complete objects
        depth = 0
        for i, ch in enumerate(stripped):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0 and i + 1 < len(stripped):
                    try:
                        return json.loads(stripped[: i + 1])
                    except json.JSONDecodeError:
                        continue
        return None
    if not isinstance(segments, list):
        return None
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="microsoft/VibeVoice-ASR-HF")
    parser.add_argument("--weights", default="microsoft/VibeVoice-ASR", help="front-end weights artifact")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--noise-seed", type=int, default=None, help="acoustic sampling seed")
    parser.add_argument("--pcm-file", type=Path, default=None, help="mono 24 kHz wav")
    args = parser.parse_args()

    from tokenizers import Tokenizer

    from hipengine.loading.vibevoice_asr import (
        load_vibevoice_connector,
        load_vibevoice_encoder,
        load_vibevoice_qwen2,
    )
    from hipengine.loading.hf_cache import resolve_model_path
    from hipengine.runtime.vibevoice_encoder import VibevoiceFrontendRuntime
    from hipengine.runtime.vibevoice_qwen2 import VibevoiceQwen2Runtime, greedy_generate

    if args.pcm_file is not None:
        import wave

        with wave.open(str(args.pcm_file), "rb") as fh:
            if fh.getframerate() != 24_000 or fh.getnchannels() != 1:
                raise SystemExit("pcm file must be mono 24 kHz")
            pcm = np.frombuffer(fh.readframes(fh.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        duration = len(pcm) / 24_000
    else:
        pcm = synth_pcm(args.seconds, args.seed)
        duration = args.seconds
    # processor contract: RMS-normalize then clip (feature extractor)
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    if rms > 0:
        pcm = pcm * (10 ** (-25.0 / 20)) / (rms + 1e-8)
        peak = float(np.abs(pcm).max())
        if peak > 1.0:
            pcm = pcm / (peak + 1e-8)
    # mandatory: pad to a multiple of the 3200 hop
    if len(pcm) % 3200:
        pcm = np.pad(pcm, (0, 3200 - len(pcm) % 3200))

    print("loading front-end weights ...")
    specs, conns = {}, {}
    for tok in ("acoustic", "semantic"):
        specs[tok] = load_vibevoice_encoder(args.weights, tok)
        conns[tok] = load_vibevoice_connector(args.weights, tok)
    frontend = VibevoiceFrontendRuntime(
        specs["acoustic"][0], specs["acoustic"][1],
        specs["semantic"][0], specs["semantic"][1],
        conns["acoustic"], conns["semantic"],
    )
    frames = specs["acoustic"][0].frame_count(len(pcm))
    rng = np.random.default_rng(args.noise_seed if args.noise_seed is not None else 20260914)
    noise = rng.standard_normal((1, frames, specs["acoustic"][0].hidden_size)).astype(np.float32)
    scale = (specs["acoustic"][0].vae_std * rng.standard_normal(1)).astype(np.float32)
    audio_embeds = frontend.forward(pcm, noise=noise[0], noise_scale=scale[0])
    frontend.close()
    print(f"audio frames: {frames}, embeds {audio_embeds.shape}")

    print("loading Qwen2 backbone ...")
    lm_path = resolve_model_path(args.model)
    lm_weights = load_vibevoice_qwen2(str(lm_path))
    runner = VibevoiceQwen2Runtime(lm_weights, max_context=max(frames + 64 + args.max_new_tokens, 1024))

    tok_path = Path(str(lm_path)) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_path))
    prompt = build_prompt(duration, frames)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = ids.ids

    rows = [runner.embed_row(int(t)) for t in input_ids]
    placeholder_positions = [i for i, t in enumerate(input_ids) if t == AUDIO_TOKEN_ID]
    if len(placeholder_positions) != frames:
        raise SystemExit(f"template frames {len(placeholder_positions)} != front-end frames {frames}")
    for j, p in enumerate(placeholder_positions):
        rows[p] = audio_embeds[j].astype(np.float32)

    print(f"prompt tokens: {len(input_ids)}, decoding ...")
    generated = greedy_generate(runner, rows, max_new_tokens=args.max_new_tokens, eos_token_id=IM_END_ID)
    out_ids = input_ids + generated
    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    # strip everything before the assistant turn
    if "assistant" in text:
        text = text[text.rindex("assistant") + len("assistant"):].strip()
    runner.close()

    print("--- raw ---")
    print(text)
    print("--- parsed ---")
    segments = parse_transcript(text)
    if segments is None:
        print("(unparseable output)")
    else:
        for seg in segments:
            print(f"[{seg.get('Start')}-{seg.get('End')}] speaker {seg.get('Speaker')}: {seg.get('Content')}")


if __name__ == "__main__":
    main()
