#!/usr/bin/env python3
"""Kernel-trace smoke for the rows 2-4 local32 IQ verifier sibling.

Launches the sibling on real UD verifier shapes and synchronizes, so a
`rocprofv3 --kernel-trace` run around it shows the expected kernel names. The
extension is prebuilt outside the profiler (the repo's JIT-cache rule).

CPU/GPU: needs a GPU.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import (
    build_gguf_iq_dense, launch_local32_rows)

# Real UD-Q4_K_M verifier shapes: (quant, K, N, rows).
SHAPES = (
    ("gguf_iq4_xs", 5120, 17408, 4),
    ("gguf_iq4_nl", 5120, 5120, 4),
    ("gguf_iq3_s", 5120, 17408, 4),
    ("gguf_iq3_xxs", 5120, 5120, 4),
    ("gguf_iq2_s", 5120, 5120, 4),
    ("gguf_iq2_xs", 5120, 5120, 4),
)

COL_BYTES = {
    "gguf_iq4_xs": lambda k: (k // 32) * 18,
    "gguf_iq4_nl": lambda k: (k // 32) * 18,
    "gguf_iq3_s": lambda k: (k // 256) * 110,
    "gguf_iq3_xxs": lambda k: (k // 256) * 98,
    "gguf_iq2_s": lambda k: (k // 256) * 82,
    "gguf_iq2_xs": lambda k: (k // 256) * 74,
}


def bf16(x):
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compiler-version-file', type=Path, default=None)
    ap.add_argument('--require-cached-build', action='store_true')
    args = ap.parse_args()
    lib = build_gguf_iq_dense(
        compiler_version=(args.compiler_version_file.read_text()
                          if args.compiler_version_file else None),
        require_cached=args.require_cached_build)
    hip = get_hip_runtime()
    rng = np.random.default_rng(7)
    bufs = []
    try:
        for quant, k, n, rows in SHAPES:
            x = bf16(rng.normal(0, 0.1, (rows, k)))
            # Zero weights keep the reads in bounds without a real tensor; the
            # trace only needs the launch geometry and the kernel name.
            wraw = np.zeros(n * COL_BYTES[quant](k), dtype=np.uint8)
            out = np.zeros((rows, n), dtype=np.uint16)
            x_b = malloc(x.nbytes); bufs.append(x_b)
            w_b = malloc(wraw.nbytes); bufs.append(w_b)
            o_b = malloc(out.nbytes); bufs.append(o_b)
            copy_host_to_device(x_b, host_array_ptr(x), x.nbytes)
            copy_host_to_device(w_b, host_array_ptr(wraw), wraw.nbytes)
            for _ in range(3):
                launch_local32_rows(x_b.ptr, w_b.ptr, o_b.ptr, rows, k, n,
                                    quant=quant, output='bf16', library=lib)
            hip.device_synchronize()
            print(f'launched {quant} rows={rows} K={k} N={n}', flush=True)
    finally:
        for b in reversed(bufs):
            free(b)
    print('trace smoke done')


if __name__ == '__main__':
    main()
