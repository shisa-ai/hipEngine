#!/usr/bin/env python3
"""Cached-only rocprof smoke for the YuE2 NAR HIP kernels.

Prebuild every DSO outside ``rocprofv3`` (``yue2_nar``, ``vibevoice_encoder``,
``paro_silu``, ``dense_gemv``, ``cast``), then run this child under
``rocprofv3 --kernel-trace``. The child loads the exact cached libraries with
``require_cached=True`` and drives one NAR velocity evaluation through the
torch-free runtime, so no compiler subprocess can be launched from the profiled
process.

Usage:
    hipcc --version > /tmp/hipengine-hipcc-version.txt
    rocprofv3 --kernel-trace --output-format csv -d /tmp/yue2-nar-trace -- \\
        python3 scripts/yue2_nar_rocprof_smoke.py \\
            --compiler-version-file /tmp/hipengine-hipcc-version.txt
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hipengine.kernels.hip_gfx1100.convert import cast  # noqa: E402
from hipengine.kernels.hip_gfx1100.fused import paro_silu  # noqa: E402
from hipengine.kernels.hip_gfx1100.linear import dense_gemv  # noqa: E402
from hipengine.kernels.hip_gfx1100.vibevoice import encoder as vv_encoder  # noqa: E402
from hipengine.kernels.hip_gfx1100.yue2 import nar  # noqa: E402

FIXTURE = REPO / "tests/fixtures/yue2/nar/chunk0.npz"


def _cached_model_dir() -> str:
    cache = Path.home() / ".cache/huggingface/hub"
    for directory in sorted(cache.glob("models--m-a-p--YuE2-3B/snapshots/*")):
        if (directory / "model.safetensors").is_file():
            return str(directory)
    raise SystemExit("YuE2-3B checkpoint not found; set --model-dir")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--raw-t", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()

    version_path = args.compiler_version_file.expanduser().resolve()
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(version_path)
    compiler_version = version_path.read_text(encoding="utf-8").strip()

    for build, label in (
        (nar.build_yue2_nar, "yue2_nar"),
        (vv_encoder.build_vibevoice_encoder, "vibevoice_encoder"),
        (paro_silu.build_paro_silu, "paro_silu"),
        (dense_gemv.build_dense_gemv, "dense_gemv"),
        (cast.build_cast, "cast"),
    ):
        library = build(load=True, require_cached=True, compiler_version=compiler_version)
        if library is None:
            raise RuntimeError(f"{label} cached build unavailable; prebuild outside the profiler")

    from hipengine.loading.yue2 import load_yue2_weights
    from hipengine.runtime.yue2_ar import Yue2ArRuntime, bf16_bits_to_f32
    from hipengine.runtime.yue2_nar import Yue2NarRuntime, song_chunks

    arrays = np.load(FIXTURE)
    weights = load_yue2_weights(args.model_dir or _cached_model_dir())
    ar = Yue2ArRuntime(weights, branches=1)
    runtime = Yue2NarRuntime(weights, ar)
    chunk = song_chunks(
        [int(v) for v in arrays["prefix"]],
        [int(v) for v in arrays["codec"]],
        int(arrays.get("seed", 1234)) if "seed" in arrays.files else 1234,
        noise=np.asarray(arrays["noise"], dtype=np.float32),
    )[0]
    runtime.condition(chunk)
    velocity = bf16_bits_to_f32(runtime.velocity_bits(float(args.raw_t)))
    # Also drive the midpoint solver so ``yue2_nar_state_update_bf16`` appears in
    # the trace: the velocity path alone never launches it.
    latents = runtime.solve(int(args.steps))
    print(f"nar solve steps={int(args.steps)} norm={float(np.linalg.norm(latents)):.4f}")
    print(
        f"nar velocity rows={chunk.nar_length} ar={chunk.ar_length} "
        f"norm={float(np.linalg.norm(velocity)):.4f} absmax={float(np.abs(velocity).max()):.4f}"
    )
    runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
