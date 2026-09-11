#!/usr/bin/env python3
"""rocprofv3 driver for the Surya HIP pipeline, one phase at a time.

Runs preprocessing, the HIP vision tower, prefill, and decode on the gfx11
Surya runner so ``rocprofv3 --kernel-trace`` sees the whole Surya kernel family
(`surya_*`, EVIE vision, GDN conv/recurrent, rocBLAS GEMM).

Phases exist because a single trace of a whole request cannot attribute time to
vision vs prefill vs decode: the kernels overlap in name and the counts are not
knowable from the CSV alone. Each phase is run in its own process, repeated
enough times that per-phase cost is the kernel total divided by the number of
calls the driver reports having made (warmup included, so the division is
exact).

Prebuild the JIT libraries outside the profiler and run this driver
cache-only:

    hipcc --version > /tmp/hipengine-hipcc-version.txt
    HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/surya_gpu_rocprof_driver.py --phase decode
    rocprofv3 --kernel-trace --memory-copy-trace --output-format csv \\
      -d /tmp/surya-rocprof -- \\
      env HIPENGINE_HIP_ARCH=gfx1151 \\
          HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \\
          HIPENGINE_REQUIRE_CACHED_BUILD=1 \\
          python3 scripts/surya_gpu_rocprof_driver.py --phase decode --repeat 200

The driver fails loudly if a JIT build is not a cache hit, so a trace never
includes hipcc compile time.

The driver prints a JSON line to stdout describing what it actually executed.
``scripts/surya_profile_report.py`` consumes that plus the rocprofv3 CSVs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "surya"
MODEL_ID = "datalab-to/surya-ocr-2"
PROMPT = "Transcribe this page."

PHASES = ("vision", "prefill", "decode", "request")


def _build(page_path: Path):
    from PIL import Image

    from hipengine.loading.surya import (
        SuryaTokenizer,
        compute_mrope_positions,
        load_surya_spec,
        load_surya_weights,
        preprocess_image_surya,
        render_chat_prompt,
        resolve_surya_path,
    )
    from hipengine.runtime.surya import SuryaGpuRunner

    model_dir = resolve_surya_path(MODEL_ID)
    spec = load_surya_spec(model_dir)
    weights = load_surya_weights(model_dir)
    tokenizer = SuryaTokenizer(model_dir)
    page = Image.open(page_path).convert("RGB")
    runner = SuryaGpuRunner(weights, spec)

    rows, grid = preprocess_image_surya(page)
    n_img = (grid[1] // 2) * (grid[2] // 2)
    ids, mm = render_chat_prompt(tokenizer, PROMPT, n_img)
    pos = compute_mrope_positions(mm, grid, spec.vision_spatial_merge_size)
    return runner, spec, rows, grid, ids, pos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, default="request")
    parser.add_argument("--repeat", type=int, default=1,
                        help="phase repeats (vision/prefill calls, decode steps)")
    parser.add_argument("--decode-steps", type=int, default=8,
                        help="decode steps for --phase request")
    parser.add_argument("--page", type=Path, default=FIXTURES / "page_small.png")
    args = parser.parse_args()

    runner, spec, rows, grid, ids, pos = _build(args.page)
    sync = runner.runtime.device_synchronize
    id_array = np.asarray(ids, dtype=np.int64)
    calls = 0
    detail: dict[str, object] = {}

    try:
        if args.phase == "vision":
            # one warmup, then the counted calls
            runner.vision_forward(rows, [grid])
            calls = 1
            for _ in range(args.repeat):
                runner.vision_forward(rows, [grid])
                calls += 1
            sync()
            detail["vision_tokens"] = int(
                (grid[1] // 2) * (grid[2] // 2)
            )
        elif args.phase == "prefill":
            runner.prefill(id_array, pos, visual_features=None)
            calls = 1
            for _ in range(args.repeat):
                runner.prefill(id_array, pos, visual_features=None)
                calls += 1
            sync()
            detail["prompt_tokens"] = int(len(ids))
        elif args.phase == "decode":
            merged = runner.vision_forward(rows, [grid])
            logits = runner.prefill(id_array, pos, visual_features=merged)
            p_last = int(pos[:, -1].max())
            sync()
            step = 0
            token = int(np.argmax(logits))
            while step < args.repeat:
                token = int(np.argmax(logits))
                if token == spec.eos_token_id:
                    token = int(np.argmax(logits[1:]) + 1) if logits.size > 1 else token
                logits = runner.decode_step(token, p_last + 1 + step)
                step += 1
            sync()
            calls = step
            detail["prompt_tokens"] = int(len(ids))
            detail["decode_steps"] = step
        else:  # request: one whole OCR request, the end-to-end picture
            t0 = time.perf_counter()
            merged = runner.vision_forward(rows, [grid])
            logits = runner.prefill(id_array, pos, visual_features=merged)
            p_last = int(pos[:, -1].max())
            generated: list[int] = []
            for step in range(args.decode_steps):
                token = int(np.argmax(logits))
                if token == spec.eos_token_id:
                    break
                generated.append(token)
                if step + 1 >= args.decode_steps:
                    break
                logits = runner.decode_step(token, p_last + 1 + step)
            sync()
            detail["wall_s"] = time.perf_counter() - t0
            detail["decode_steps"] = len(generated)
            detail["first_ids"] = generated[:8]
            calls = 1
    finally:
        runner.close()

    print(json.dumps({
        "phase": args.phase,
        "page": str(args.page),
        "grid": [int(v) for v in grid],
        "calls_executed": calls,
        **detail,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
