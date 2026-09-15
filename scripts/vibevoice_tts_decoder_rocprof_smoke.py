#!/usr/bin/env python3
"""Cached-only rocprof smoke for the VibeVoice-TTS decoder HIP kernels.

Prebuild the three DSOs outside rocprof (``vibevoice_tts_decoder.so``,
``vibevoice_encoder.so``, ``dense_gemv.so``), then run this child under
``rocprofv3 --kernel-trace``. The child loads the exact cached libraries with
``require_cached=True`` and decodes a few fixture frames through the
torch-free runtime. No compiler subprocess can be launched from the profiled
process.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.kernels.hip_gfx1100.linear import dense_gemv
from hipengine.kernels.hip_gfx1100.vibevoice import decoder as tts_decoder
from hipengine.kernels.hip_gfx1100.vibevoice import encoder as vv_enc
from hipengine.runtime.vibevoice_tts_decoder import VibevoiceTTSDecoderGPU

_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "vibevoice_tts" / "single_diffusion.npz"
_FRAMES = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    version_path = args.compiler_version_file.expanduser().resolve()
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(version_path)
    compiler_version = version_path.read_text(encoding="utf-8").strip()

    for build, label in (
        (tts_decoder.build_vibevoice_decoder, "vibevoice_tts_decoder"),
        (vv_enc.build_vibevoice_encoder, "vibevoice_encoder"),
        (dense_gemv.build_dense_gemv, "dense_gemv"),
    ):
        library = build(load=True, require_cached=True, compiler_version=compiler_version)
        if library is None:
            raise RuntimeError(f"{label} cached build unavailable; prebuild outside the profiler")

    from hipengine.loading.hf_cache import resolve_model_path
    from hipengine.loading.vibevoice_tts import load_vibevoice_tts_decoder

    spec, weights, _, _ = load_vibevoice_tts_decoder(resolve_model_path("microsoft/VibeVoice-1.5B"))
    data = np.load(_FIXTURE)
    n = int(data["num_calls_recorded"])
    lat = np.stack([data[f"call{i}_scaled_latent"].reshape(-1) for i in range(min(_FRAMES, n))])

    runner = VibevoiceTTSDecoderGPU(spec, weights)
    pcm = runner.decode_bulk(lat)
    runner.close()
    print(f"decoded {pcm.shape} pcm, absmax {np.abs(pcm).max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
