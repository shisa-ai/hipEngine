#!/usr/bin/env python3
"""Minimal rocprofv3 driver for the VibeVoice ASR front-end forward.

Prebuild the .so caches outside the profiler (run warm once), then invoke
with ``rocprofv3 --kernel-trace``. Requires the original
``microsoft/VibeVoice-ASR`` snapshot in the local HF cache.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from hipengine.kernels.hip_gfx1100.vibevoice.encoder import build_vibevoice_encoder  # noqa: E402
from hipengine.loading.vibevoice_asr import load_vibevoice_connector, load_vibevoice_encoder  # noqa: E402
from hipengine.runtime.vibevoice_encoder import VibevoiceFrontendRuntime  # noqa: E402

os.environ.setdefault("HIPENGINE_VIBEVOICE_REQUIRE_CACHED", "1")

specs, conns = {}, {}
for tok in ("acoustic", "semantic"):
    specs[tok] = load_vibevoice_encoder("microsoft/VibeVoice-ASR", tok)
    conns[tok] = load_vibevoice_connector("microsoft/VibeVoice-ASR", tok)

if "--warm" in sys.argv:
    lib = build_vibevoice_encoder(require_cached=False)
    print("warm: built", lib is not None)
    raise SystemExit(0)

lib = build_vibevoice_encoder(require_cached=True)
assert lib is not None

runtime = VibevoiceFrontendRuntime(
    specs["acoustic"][0], specs["acoustic"][1],
    specs["semantic"][0], specs["semantic"][1],
    conns["acoustic"], conns["semantic"],
    library=lib,
)
rng = np.random.default_rng(0)
pcm = (0.1 * rng.standard_normal(48000)).astype(np.float32)
emb = runtime.forward(pcm)
runtime.close()
print("front-end ok:", emb.shape, float(np.abs(emb).max()))
