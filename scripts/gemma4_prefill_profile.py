#!/usr/bin/env python3
"""Prefill-only driver for a kernel census under rocprofv3.

Loads the public Gemma 4 surface, warms the JIT cache with short prefills, then
runs ``--tokens``-token prefills (``--repeats`` of them) with a device
synchronize after each. Nothing but prefill runs, so a kernel trace of this
process is a prefill census.

Run the warmup outside the profiler first (``--warm-only``) so the profiled
process never spawns ``hipcc``; see docs/OPTIMIZATION.md section 7.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = Path(
    "/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warm-only", action="store_true")
    parser.add_argument("--expect-gpu", default=None)
    args = parser.parse_args()

    import hipengine
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = __import__("ctypes").CDLL("libamdhip64.so")
    name = __import__("ctypes").create_string_buffer(256)
    library.hipDeviceGetName(name, len(name), 0)
    device = name.value.decode()
    print(f"[prefill_profile] device0={device}", flush=True)
    if args.expect_gpu and args.expect_gpu not in device:
        print(f"ERROR: device0 is {device!r}, expected {args.expect_gpu!r}", file=sys.stderr)
        return 2

    llm = hipengine.LLM(model=str(args.artifact))
    generator = llm._get_text_generator()
    generator.context_length = int(args.context)
    runner = generator._ensure_runner()

    prompt = [9707] * args.tokens
    runner.reset()
    runner.forward(prompt[:64])
    runtime.device_synchronize()
    runner.reset()
    runner.forward(prompt)
    runtime.device_synchronize()
    if args.warm_only:
        print("[prefill_profile] warm-only complete", flush=True)
        return 0

    for index in range(args.repeats):
        runner.reset()
        runtime.device_synchronize()
        start = time.perf_counter()
        runner.forward(prompt)
        runtime.device_synchronize()
        elapsed = time.perf_counter() - start
        print(
            f"[prefill_profile] repeat={index} tokens={args.tokens} "
            f"prefill_s={elapsed:.6f} prefill_tps={args.tokens / elapsed:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
