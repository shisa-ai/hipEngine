#!/usr/bin/env python3
"""Minimal rocprofv3 driver for one full Surya OCR GPU request.

Runs preprocessing, the HIP vision tower, prefill, and a few decode steps on
the gfx11 Surya runner so ``rocprofv3 --kernel-trace`` sees the whole Surya
kernel family (`surya_*`, EVIE vision, GDN conv/recurrent, rocBLAS GEMM).

Prebuild the JIT libraries outside the profiler and run this driver
cache-only:

    hipcc --version > /tmp/hipengine-hipcc-version.txt
    HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/surya_gpu_rocprof_driver.py
    rocprofv3 --kernel-trace --output-format csv -d /tmp/surya-rocprof -- \\
      env HIPENGINE_HIP_ARCH=gfx1151 \\
          HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \\
          HIPENGINE_REQUIRE_CACHED_BUILD=1 \\
          python3 scripts/surya_gpu_rocprof_driver.py

The driver fails loudly if a JIT build is not a cache hit, so a trace never
includes hipcc compile time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "surya"
MODEL_ID = "datalab-to/surya-ocr-2"
PROMPT = "Transcribe this page."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--page", type=Path, default=FIXTURES / "page_small.png")
    args = parser.parse_args()

    from PIL import Image

    from hipengine.kernels.cpu_reference.surya import SuryaSpec, SuryaWeights
    from hipengine.loading.surya import (
        SuryaTokenizer,
        compute_mrope_positions,
        preprocess_image_surya,
        render_chat_prompt,
        resolve_surya_path,
    )
    from hipengine.runtime.surya import SuryaGpuRunner

    model_dir = resolve_surya_path(MODEL_ID)
    spec = SuryaSpec()
    weights = SuryaWeights.load(str(model_dir / "model.safetensors"))
    tokenizer = SuryaTokenizer(model_dir)
    page = Image.open(args.page).convert("RGB")

    runner = SuryaGpuRunner(weights, spec)
    try:
        pixel_rows, grid = preprocess_image_surya(page)
        merged = runner.vision_forward(pixel_rows, [grid])
        n_img = (grid[1] // 2) * (grid[2] // 2)
        ids, mm = render_chat_prompt(tokenizer, PROMPT, n_img)
        pos = compute_mrope_positions(mm, grid, spec.vision_spatial_merge_size)
        logits = runner.prefill(np.asarray(ids, dtype=np.int64), pos,
                                visual_features=merged)
        generated: list[int] = []
        p = int(pos[:, -1].max())
        for step in range(args.decode_steps):
            nxt = int(np.argmax(logits))
            if nxt == spec.eos_token_id:
                break
            generated.append(nxt)
            if step + 1 >= args.decode_steps:
                break
            logits = runner.decode_step(nxt, p + 1 + step)
    finally:
        runner.close()

    print(json.dumps({
        "page": str(args.page),
        "grid": list(grid),
        "vision_tokens": int(merged.shape[0]),
        "prompt_tokens": int(len(ids)),
        "decode_steps": len(generated),
        "first_ids": generated[:8],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
