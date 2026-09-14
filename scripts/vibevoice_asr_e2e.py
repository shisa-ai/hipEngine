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
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hipengine.generation.vibevoice_protocol import (
    AUDIO_TOKEN_ID, IM_END_ID, build_prompt, parse_transcript, preprocess_audio,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="microsoft/VibeVoice-ASR-HF")
    parser.add_argument("--weights", default=None, help="front-end artifact (defaults to --model)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--noise-seed", type=int, default=None, help="acoustic sampling seed")
    parser.add_argument("--pcm-file", type=Path, default=None, help="mono 24 kHz wav")
    args = parser.parse_args()

    if args.pcm_file is not None:
        import wave

        with wave.open(str(args.pcm_file), "rb") as fh:
            if (fh.getframerate(), fh.getnchannels(), fh.getsampwidth()) != (24000, 1, 2):
                raise SystemExit("pcm file must be mono 24 kHz")
            pcm = np.frombuffer(fh.readframes(fh.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        duration = len(pcm) / 24_000
    else:
        pcm = synth_pcm(args.seconds, args.seed)
        duration = args.seconds
    from hipengine import LLM
    if args.weights is not None and args.weights != args.model:
        raise SystemExit("use one HF checkpoint for the complete pipeline (--model)")
    engine = LLM(args.model, max_sequence_length=4096)
    try:
        result = engine.transcribe(pcm, max_new_tokens=args.max_new_tokens,
            seed=args.noise_seed if args.noise_seed is not None else 20260914)
    finally:
        engine.close()
    text = result.text

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
