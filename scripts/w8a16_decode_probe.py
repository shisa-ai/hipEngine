#!/usr/bin/env python3
"""Synthetic W8A16 decode-shape kernel probe.

This is a profiler helper for the D5.2 W8A16 decode audit.  It launches the
same raw-pointer kernels used by Qwen3.5/PARO decode at representative shapes:

* lm-head BF16->FP32: hidden=2048, out=248320
* legacy shared-expert gate/up FP16 lowp: hidden=2048, out=1024
* legacy shared-expert down FP16 lowp: hidden=512, out=2048

Run it under ``rocprofv3 --kernel-trace`` and group rows by kernel name,
workgroup size, and grid/workgroup block count.
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

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.quant.w8a16_linear import (
    build_w8a16_linear,
    w8a16_linear_bf16_f32_out,
    w8a16_linear_fp16_lowp_out,
    w8a16_shared_gate_up_silu_fp16,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--threads", default="64,128,256,512")
    parser.add_argument("--lm-reps", type=int, default=6)
    parser.add_argument("--shared-reps", type=int, default=80)
    parser.add_argument(
        "--include-fused-shared",
        action="store_true",
        help="Also probe the existing fused shared gate/up+SiLU helper for c=1 context.",
    )
    args = parser.parse_args()

    if args.lm_reps < 0 or args.shared_reps < 0:
        raise ValueError("rep counts must be non-negative")
    threads_list = [int(value) for value in args.threads.split(",") if value]
    if not threads_list:
        raise ValueError("--threads must list at least one value")

    compiler_version = args.compiler_version_file.read_text() if args.compiler_version_file is not None else None
    library = build_w8a16_linear(
        load=True,
        compiler_version=compiler_version,
        require_cached=args.require_cached_build,
    )
    runtime = get_hip_runtime()

    hidden = 2048
    vocab = 248_320
    shared_int = 512

    # Values do not matter for the timing/resource audit.  Dense non-zero inputs
    # avoid any special handling from memory tooling while keeping host setup
    # deterministic and cheap.
    x_bf16 = np.full((1, hidden), 0x3F80, dtype=np.uint16)  # BF16 1.0
    x_fp16 = np.ones((1, hidden), dtype=np.float16)
    x_shared = np.ones((1, shared_int), dtype=np.float16)
    lm_weight = np.ones((vocab, hidden), dtype=np.int8)
    lm_scale = np.ones((vocab,), dtype=np.float32)
    gate_up_weight = np.ones((2 * shared_int, hidden), dtype=np.int8)
    gate_up_scale = np.ones((2 * shared_int,), dtype=np.float32)
    down_weight = np.ones((hidden, shared_int), dtype=np.int8)
    down_scale = np.ones((hidden,), dtype=np.float32)

    x_bf16_d = _to_device(x_bf16, runtime)
    x_fp16_d = _to_device(x_fp16, runtime)
    x_shared_d = _to_device(x_shared, runtime)
    lm_weight_d = _to_device(lm_weight, runtime)
    lm_scale_d = _to_device(lm_scale, runtime)
    gate_up_weight_d = _to_device(gate_up_weight, runtime)
    gate_up_scale_d = _to_device(gate_up_scale, runtime)
    down_weight_d = _to_device(down_weight, runtime)
    down_scale_d = _to_device(down_scale, runtime)
    lm_out_d = malloc(vocab * np.dtype(np.float32).itemsize, runtime=runtime)
    gate_up_out_d = malloc((2 * shared_int) * np.dtype(np.float16).itemsize, runtime=runtime)
    down_out_d = malloc(hidden * np.dtype(np.float16).itemsize, runtime=runtime)
    fused_out_d = malloc(shared_int * np.dtype(np.float16).itemsize, runtime=runtime)

    cases: list[dict[str, object]] = []
    for threads in threads_list:
        w8a16_linear_bf16_f32_out(
            x_bf16_d.ptr,
            lm_weight_d.ptr,
            lm_scale_d.ptr,
            lm_out_d.ptr,
            1,
            hidden,
            vocab,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        w8a16_linear_fp16_lowp_out(
            x_fp16_d.ptr,
            gate_up_weight_d.ptr,
            gate_up_scale_d.ptr,
            gate_up_out_d.ptr,
            1,
            hidden,
            2 * shared_int,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        w8a16_linear_fp16_lowp_out(
            x_shared_d.ptr,
            down_weight_d.ptr,
            down_scale_d.ptr,
            down_out_d.ptr,
            1,
            shared_int,
            hidden,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        if args.include_fused_shared:
            w8a16_shared_gate_up_silu_fp16(
                x_fp16_d.ptr,
                gate_up_weight_d.ptr,
                gate_up_scale_d.ptr,
                fused_out_d.ptr,
                1,
                hidden,
                shared_int,
                threads=threads,
                library=library,
                runtime=runtime,
            )
    runtime.device_synchronize()

    start = time.perf_counter()
    for threads in threads_list:
        for _ in range(args.lm_reps):
            w8a16_linear_bf16_f32_out(
                x_bf16_d.ptr,
                lm_weight_d.ptr,
                lm_scale_d.ptr,
                lm_out_d.ptr,
                1,
                hidden,
                vocab,
                threads=threads,
                library=library,
                runtime=runtime,
            )
        cases.append({"case": "lm_head_bf16_f32", "threads": threads, "reps": args.lm_reps, "hidden": hidden, "out_features": vocab})

        for _ in range(args.shared_reps):
            w8a16_linear_fp16_lowp_out(
                x_fp16_d.ptr,
                gate_up_weight_d.ptr,
                gate_up_scale_d.ptr,
                gate_up_out_d.ptr,
                1,
                hidden,
                2 * shared_int,
                threads=threads,
                library=library,
                runtime=runtime,
            )
        cases.append({"case": "shared_gate_up_fp16_lowp", "threads": threads, "reps": args.shared_reps, "hidden": hidden, "out_features": 2 * shared_int})

        for _ in range(args.shared_reps):
            w8a16_linear_fp16_lowp_out(
                x_shared_d.ptr,
                down_weight_d.ptr,
                down_scale_d.ptr,
                down_out_d.ptr,
                1,
                shared_int,
                hidden,
                threads=threads,
                library=library,
                runtime=runtime,
            )
        cases.append({"case": "shared_down_fp16_lowp", "threads": threads, "reps": args.shared_reps, "hidden": shared_int, "out_features": hidden})

        if args.include_fused_shared:
            for _ in range(args.shared_reps):
                w8a16_shared_gate_up_silu_fp16(
                    x_fp16_d.ptr,
                    gate_up_weight_d.ptr,
                    gate_up_scale_d.ptr,
                    fused_out_d.ptr,
                    1,
                    hidden,
                    shared_int,
                    threads=threads,
                    library=library,
                    runtime=runtime,
                )
            cases.append({"case": "shared_gate_up_silu_fp16", "threads": threads, "reps": args.shared_reps, "hidden": hidden, "out_features": shared_int})
    runtime.device_synchronize()
    elapsed = time.perf_counter() - start

    print(json.dumps({"elapsed_s": elapsed, "cases": cases}, indent=2))
    return 0


def _to_device(array: np.ndarray, runtime) -> object:
    arr = np.ascontiguousarray(array)
    buf = malloc(arr.nbytes, runtime=runtime)
    copy_host_to_device(buf, host_array_ptr(arr), runtime=runtime)
    return buf


if __name__ == "__main__":
    raise SystemExit(main())
