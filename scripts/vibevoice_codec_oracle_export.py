#!/usr/bin/env python3
"""Export the pinned torch oracle's codec decodes for cross-venv comparison.

Runs the community fork's acoustic decoder in float32 on the seeded latent and
saves the latent plus the oracle's full-sequence and frame-by-frame streaming
waveforms to an .npz. The GPU-side check
(``scripts/vibevoice_codec_gpu_reference_check.py``) runs in the project venv
and compares against this artifact; keeping torch and the gfx1100 decoder in
separate processes avoids the oracle venv's loaded ROCm-torch runtime
interfering with the module path.

Run inside the oracle venv (torch), from the repository root:

  ~/venvs/vibevoice-oracle/bin/python scripts/vibevoice_codec_oracle_export.py \
      --model /models/vibevoice/VibeVoice-1.5B \
      --npz benchmarks/results/2026-09-15-vibevoice-codec-gpu-oracle-ref.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="/models/vibevoice/VibeVoice-1.5B")
    parser.add_argument("--fork", type=str, default="/home/lhl/VibeVoice-community")
    parser.add_argument("--frames", type=int, default=6, help="latent frames to decode")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--npz", type=str, required=True)
    args = parser.parse_args()

    import torch

    sys.path.insert(0, args.fork)
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    from vibevoice.modular.modular_vibevoice_tokenizer import (
        VibeVoiceTokenizerStreamingCache,
    )

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    latent = rng.standard_normal((args.frames, 64)).astype(np.float32) * 0.5
    frames = args.frames

    full = VibeVoiceForConditionalGenerationInference.from_pretrained(
        args.model, torch_dtype=torch.float32
    ).eval()
    model = full.model.acoustic_tokenizer
    latents_t = torch.from_numpy(latent.T[None])  # [1, 64, T]
    with torch.no_grad():
        oracle_full = model.decode(latents_t.clone(), use_cache=False)[0, 0].numpy()
        cache = VibeVoiceTokenizerStreamingCache()
        chunks = []
        for frame in range(frames):
            chunk = model.decode(
                latents_t[:, :, frame : frame + 1],
                cache=cache,
                sample_indices=torch.zeros(1, dtype=torch.long),
                use_cache=True,
            )
            chunks.append(chunk[0, 0].numpy())
        oracle_stream = np.concatenate(chunks)

    out = Path(args.npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        latent=latent,
        oracle_full=oracle_full,
        oracle_stream=oracle_stream,
    )
    print(f"saved {out}: latent {latent.shape}, oracle_full peak"
          f" {np.abs(oracle_full).max():.6e}, oracle_stream peak"
          f" {np.abs(oracle_stream).max():.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
