#!/usr/bin/env python3
"""Screen the 2D-tiled router logits against the shipped token-tile-4 kernel."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.hip_gfx1100.moe.router import (
    qwen35_router_logits_f32_f32w_token_tile4_dense_exact,
)

TOKENS = 1024
HIDDEN = 2560
EXPERTS = 512
REPS = 30
WARMUP = 5
SO = Path("/tmp/qwen4exp-scope/router_tile2d.so")


def build() -> ctypes.CDLL:
    hipcc = os.environ.get("HIPCC", "hipcc")
    cmd = [hipcc, "-O3", "--offload-arch=gfx1151", "-shared", "-fPIC",
           str(SO.with_suffix(".hip")), "-o", str(SO)]
    subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(SO))
    lib.router_logits_tile2d.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3 + [ctypes.c_void_p]
    lib.router_logits_tile2d.restype = ctypes.c_int
    return lib


def main() -> None:
    lib = build()
    rt = get_hip_runtime()
    rng = np.random.default_rng(7)

    hidden = rng.normal(0, 0.1, (TOKENS, HIDDEN)).astype(np.float32)
    weight = rng.normal(0, 0.1, (EXPERTS, HIDDEN)).astype(np.float32)

    d_hidden = malloc(hidden.nbytes)
    d_weight = malloc(weight.nbytes)
    d_out = malloc(TOKENS * EXPERTS * 4)
    copy_host_to_device(d_hidden, host_array_ptr(hidden), hidden.nbytes)
    copy_host_to_device(d_weight, host_array_ptr(weight), weight.nbytes)

    ref = np.zeros((TOKENS, EXPERTS), dtype=np.float32)
    got = np.zeros((TOKENS, EXPERTS), dtype=np.float32)

    def run_shipped() -> None:
        qwen35_router_logits_f32_f32w_token_tile4_dense_exact(
            d_hidden.ptr, d_weight.ptr, d_out.ptr, TOKENS, HIDDEN, EXPERTS)

    def run_tiled() -> None:
        err = lib.router_logits_tile2d(
            ctypes.c_void_p(d_hidden.ptr), ctypes.c_void_p(d_weight.ptr),
            ctypes.c_void_p(d_out.ptr), TOKENS, HIDDEN, EXPERTS, None)
        if err:
            raise RuntimeError(f"tile2d launch failed: {err}")

    run_shipped()
    copy_device_to_host(host_array_ptr(ref), d_out, ref.nbytes)
    run_tiled()
    copy_device_to_host(host_array_ptr(got), d_out, got.nbytes)

    def time(fn) -> float:
        for _ in range(WARMUP):
            fn()
        rt.device_synchronize()
        start, stop = rt.event_create(), rt.event_create()
        rt.event_record(start)
        for _ in range(REPS):
            fn()
        rt.event_record(stop)
        rt.device_synchronize()
        ms = rt.event_elapsed_time_ms(start, stop) / REPS
        rt.event_destroy(start)
        rt.event_destroy(stop)
        return ms

    t_shipped = time(run_shipped)
    t_tiled = time(run_tiled)

    # Traffic model: 5.369 GB * (1/T_EXP + 1/T_TOK), tokens*experts*hidden*4 bytes
    elem = TOKENS * EXPERTS * HIDDEN * 4
    shipped_traffic = elem * (1.0 + 1.0 / 4.0)
    tiled_traffic = elem * (1.0 / 16.0 + 1.0 / 16.0)
    flops = 2.0 * TOKENS * EXPERTS * HIDDEN

    out = {
        "shape": {"tokens": TOKENS, "hidden": HIDDEN, "experts": EXPERTS},
        "shipped": {
            "tile": "4 tokens x 1 expert", "grid": [EXPERTS, TOKENS // 4],
            "block": 256, "median_ms": round(t_shipped, 4),
            "traffic_gb": round(shipped_traffic / 1e9, 3),
            "effective_gb_s": round(shipped_traffic / 1e6 / t_shipped, 1),
        },
        "tile2d": {
            "tile": "16 tokens x 16 experts", "grid": [EXPERTS // 16, TOKENS // 16],
            "block": 256, "median_ms": round(t_tiled, 4),
            "traffic_gb": round(tiled_traffic / 1e9, 3),
            "effective_gb_s": round(tiled_traffic / 1e6 / t_tiled, 1),
        },
        "speedup": round(t_shipped / t_tiled, 3),
        "traffic_ratio": round(shipped_traffic / tiled_traffic, 2),
        "flops_gflop": round(flops / 1e9, 3),
        "fma_floor_ms": round(flops / 16.4e12 * 1e3, 4),
        "max_abs_diff": float(np.abs(ref - got).max()),
        "max_rel_diff": float((np.abs(ref - got) / np.maximum(np.abs(ref), 1e-6)).max()),
    }
    print("RESULT " + json.dumps(out, indent=1))

    free(d_hidden); free(d_weight); free(d_out)


if __name__ == "__main__":
    main()
