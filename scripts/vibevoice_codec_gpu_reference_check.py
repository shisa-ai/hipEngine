#!/usr/bin/env python3
"""Validate the device-resident VibeVoice codec decoder on the pinned checkpoint.

Runs the gfx1100 kernel decoder (``hipengine.models.vibevoice_codec``) in the
project venv against the CPU reference and, when the exported oracle artifact
is present (``scripts/vibevoice_codec_oracle_export.py``, run in the oracle
venv — torch and the gfx1100 decoder stay in separate processes because the
oracle venv's loaded ROCm-torch runtime interferes with the module path):

1. GPU full-sequence decode vs the CPU reference's full decode;
2. GPU frame-by-frame streaming decode vs the CPU reference's streaming decode;
3. GPU streaming vs GPU full (expected bit-identical: same windows, same
   kernel code path per element);
4. GPU chunked (2-frame) and post-reset streaming vs the reference streaming;
5. both GPU paths vs the exported torch oracle waveforms.

Writes a compact JSON artifact:

  uv run python scripts/vibevoice_codec_gpu_reference_check.py \
      --model /models/vibevoice/VibeVoice-1.5B \
      --npz benchmarks/results/2026-09-15-vibevoice-codec-gpu-oracle-ref.npz \
      --json benchmarks/results/2026-09-15-vibevoice-codec-gpu-reference.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="/models/vibevoice/VibeVoice-1.5B")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--npz", type=str, default="",
                        help="oracle export from vibevoice_codec_oracle_export.py")
    parser.add_argument("--json", type=str, required=True)
    args = parser.parse_args()

    from hipengine.kernels.cpu_reference import vibevoice_codec as ref
    from hipengine.models.vibevoice_codec import VibeVoiceCodecDecoderDevice

    if args.npz:
        exported = np.load(args.npz)
        latent = exported["latent"]
        oracle_full = exported["oracle_full"]
        oracle_stream = exported["oracle_stream"]
    else:
        rng = np.random.default_rng(20260915)
        latent = rng.standard_normal((args.frames, 64)).astype(np.float32) * 0.5
        oracle_full = None
        oracle_stream = None
    frames = latent.shape[0]

    # --- CPU reference ---
    from hipengine.loading.safetensors import load_weight_index, read_tensor_storage_bytes

    index = load_weight_index(args.model)

    def read_tensor(name: str) -> bytes:
        return read_tensor_storage_bytes(index.require([name])[0])

    weights = ref.load_decoder_weights(read_tensor)
    ref_full = ref.decode_full_sequence(latent, weights)
    streams = ref.DecoderStreams()
    ref_stream = np.concatenate(
        [ref.decode_latent_frames(latent[i : i + 1], weights, streams)
         for i in range(frames)]
    )

    # --- GPU decoder (W7900) ---
    decoder = VibeVoiceCodecDecoderDevice.from_checkpoint(args.model)
    try:
        gpu_full = decoder.decode_full(latent)
        gpu_stream = np.concatenate(
            [decoder.decode_chunk(latent[i : i + 1]) for i in range(frames)]
        )
        half = frames // 2
        decoder.reset()  # the chunked phase needs fresh-utterance cache state
        chunk_sizes = [2] * (frames // 2) + ([1] if frames % 2 else [])
        gpu_chunked = np.concatenate(
            [
                decoder.decode_chunk(
                    latent[sum(chunk_sizes[:i]) : sum(chunk_sizes[: i + 1])]
                )
                for i in range(len(chunk_sizes))
            ]
        )
        print("chunked done", np.abs(gpu_chunked).max(), flush=True)
        decoder.reset()
        gpu_reset_stream = np.concatenate(
            [decoder.decode_chunk(latent[i : i + 1]) for i in range(frames)]
        )
    finally:
        decoder.close()

    def err(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        scale = np.abs(b.astype(np.float64)).max() or 1.0
        return {
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "max_rel_to_peak": float(diff.max() / scale),
        }

    report = {
        "frames": frames,
        "samples_per_frame": ref.DecoderGeometry().hop_length,
        "gpu_full_vs_ref_full": err(gpu_full, ref_full),
        "gpu_stream_vs_ref_stream": err(gpu_stream, ref_stream),
        "gpu_stream_vs_gpu_full": {
            "max_abs": float(np.abs(gpu_stream - gpu_full).max()),
            "bit_exact": bool(np.array_equal(gpu_stream, gpu_full)),
        },
        "gpu_chunked_vs_ref_stream": err(gpu_chunked, ref_stream),
        "gpu_reset_stream_vs_ref_stream": err(gpu_reset_stream, ref_stream),
        "waveform_peak": {
            "reference_full": float(np.abs(ref_full).max()),
            "gpu_full": float(np.abs(gpu_full).max()),
        },
    }
    if oracle_full is not None:
        report["gpu_full_vs_oracle_full"] = err(gpu_full, oracle_full)
        report["gpu_stream_vs_oracle_stream"] = err(gpu_stream, oracle_stream)
        report["ref_full_vs_oracle_full"] = err(ref_full, oracle_full)
        report["waveform_peak"]["oracle_full"] = float(np.abs(oracle_full).max())

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
