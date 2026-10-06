#!/usr/bin/env python3
"""Cached-only rocprofv3 smoke for the YuE2 VAE decoder kernels.

The profiled process must not compile: every library is opened with
``require_cached=True`` against a precomputed compiler-version file, so
``hipcc``/clang never runs under ``rocprofv3``. Prebuild outside the profiler,
then:

    hipcc --version > /tmp/hipengine-hipcc-version.txt
    rocprofv3 --kernel-trace --output-format csv -d /tmp/yue2-vae-trace -- \
      python3 scripts/yue2_vae_rocprof_smoke.py \
        --compiler-version-file /tmp/hipengine-hipcc-version.txt

It drives one small full decode plus one tiled decode, so every VAE kernel
(conv1d, conv_transpose1d, snake_beta, add) appears in the trace.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _vae_dir() -> str:
    cache = Path.home() / ".cache/huggingface/hub"
    for directory in sorted(cache.glob("models--m-a-p--YuE2-Vae/snapshots/*")):
        if (directory / "model.safetensors").is_file():
            return str(directory)
    raise SystemExit("YuE2-Vae checkpoint not found; set --vae-dir")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--vae-dir", default="")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--core-frames", type=int, default=2)
    parser.add_argument("--halo-frames", type=int, default=16)
    args = parser.parse_args()

    version_path = args.compiler_version_file.expanduser().resolve()
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(version_path)
    compiler_version = version_path.read_text(encoding="utf-8").strip()

    from hipengine.kernels.hip_gfx1100.yue2 import vae as vae_kernels

    # The decoder uses only its own family: conv1d, conv_transpose1d, snake_beta
    # and the residual add all live in ``vae.hip``.
    library = vae_kernels.build_yue2_vae(
        load=True, require_cached=True, compiler_version=compiler_version
    )
    if library is None:
        raise RuntimeError("yue2_vae cached build unavailable; prebuild outside the profiler")

    from hipengine.loading.yue2 import load_yue2_vae_decoder
    from hipengine.runtime.yue2_vae import Yue2VaeRuntime

    runtime = Yue2VaeRuntime(load_yue2_vae_decoder(args.vae_dir or _vae_dir()))
    rng = np.random.default_rng(20260916)
    latent = rng.standard_normal((1, 64, int(args.frames))).astype(np.float32)
    full = runtime.decode(latent)
    tiled = runtime.decode_tiled(
        latent, core_frames=int(args.core_frames), halo_frames=int(args.halo_frames)
    )
    print(
        f"vae full shape={full.shape} rms={float(np.sqrt((full**2).mean())):.6f} "
        f"peak={float(np.abs(full).max()):.6f}"
    )
    print(
        f"vae tiled shape={tiled.shape} bit_exact_vs_full="
        f"{bool(np.array_equal(tiled, full))} required_halo={runtime.required_halo(int(args.core_frames))}"
    )
    runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
