#!/usr/bin/env python3
"""Validate the CPU-reference VibeVoice codec decoder against the torch oracle.

Runs the pinned fork's ``VibeVoiceAcousticTokenizerModel`` decoder in float32
on a fixed latent and compares, per stage, against
``hipengine.kernels.cpu_reference.vibevoice_codec``:

1. full-sequence decode vs the oracle's non-streaming decode;
2. frame-by-frame streaming decode vs the oracle's streaming decode;
3. streaming vs full equality inside the numpy reference itself.

Writes a compact JSON artifact with max/mean absolute and relative errors.
Run inside the oracle venv (torch), from the repository root:

  ~/venvs/vibevoice-oracle/bin/python scripts/vibevoice_codec_cpu_reference_check.py \
      --model /models/vibevoice/VibeVoice-1.5B \
      --json benchmarks/results/2026-09-15-vibevoice-codec-cpu-reference.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="/models/vibevoice/VibeVoice-1.5B")
    parser.add_argument("--fork", type=str, default="/home/lhl/VibeVoice-community")
    parser.add_argument("--frames", type=int, default=6, help="latent frames to decode")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--json", type=str, required=True)
    args = parser.parse_args()

    import torch

    sys.path.insert(0, args.fork)
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )

    from vibevoice.modular.modular_vibevoice_tokenizer import (
        VibeVoiceTokenizerStreamingCache,
    )

    from hipengine.loading.safetensors import load_weight_index, read_tensor_storage_bytes
    from hipengine.kernels.cpu_reference import vibevoice_codec as ref

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    latent = rng.standard_normal((args.frames, 64)).astype(np.float32) * 0.5

    # Load the full checkpoint (the acoustic tokenizer weights live inside it);
    # constructing the tokenizer standalone would leave random init weights.
    full = VibeVoiceForConditionalGenerationInference.from_pretrained(
        args.model, torch_dtype=torch.float32
    ).eval()
    model = full.model.acoustic_tokenizer

    index = load_weight_index(args.model)

    def read_tensor(full_name: str) -> bytes:
        info = index.require([full_name])[0]
        return read_tensor_storage_bytes(info)

    weights = ref.load_decoder_weights(read_tensor)

    latents_t = torch.from_numpy(latent.T[None])  # [1, 64, T]
    with torch.no_grad():
        oracle_full = model.decode(latents_t.clone(), use_cache=False)[0, 0].numpy()
        cache = VibeVoiceTokenizerStreamingCache()
        chunks = []
        for frame in range(args.frames):
            chunk = model.decode(
                latents_t[:, :, frame : frame + 1],
                cache=cache,
                sample_indices=torch.zeros(1, dtype=torch.long),
                use_cache=True,
            )
            chunks.append(chunk[0, 0].numpy())
        oracle_stream = np.concatenate(chunks)

    # 1. numpy full vs oracle full.
    ours_full = ref.decode_full_sequence(latent, weights)
    # 2. numpy streaming (frame by frame) vs oracle streaming.
    streams = ref.DecoderStreams()
    ours_stream_chunks = [
        ref.decode_latent_frames(latent[i : i + 1], weights, streams)
        for i in range(args.frames)
    ]
    ours_stream = np.concatenate(ours_stream_chunks)
    # 3. numpy streaming vs numpy full.
    full_vs_stream = np.abs(ours_full - ours_stream).max()

    def err(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        scale = np.abs(b.astype(np.float64)).max() or 1.0
        return {
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "max_rel_to_peak": float(diff.max() / scale),
        }

    report = {
        "frames": args.frames,
        "samples_per_frame": ref.DecoderGeometry().hop_length,
        "full_vs_oracle_full": err(ours_full, oracle_full),
        "stream_vs_oracle_stream": err(ours_stream, oracle_stream),
        "stream_vs_full_within_reference": {
            "max_abs": float(full_vs_stream),
            "bit_exact": bool(full_vs_stream == 0.0),
        },
        "oracle_stream_vs_oracle_full": err(oracle_stream, oracle_full),
        "waveform_peak": {
            "oracle_full": float(np.abs(oracle_full).max()),
            "reference_full": float(np.abs(ours_full).max()),
        },
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
