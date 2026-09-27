#!/usr/bin/env python3
"""What limits the shipped router logits kernel? Screen tile width and reduction shape."""

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
SRC = Path("/tmp/qwen4exp-scope/router_variants.hip")
SO = Path("/tmp/qwen4exp-scope/router_variants.so")

NAMES = ["router_token_tile4", "router_token_tile8", "router_token_tile16",
         "router_token_tile32", "router_warp_token"]


def build() -> ctypes.CDLL:
    subprocess.run(
        [os.environ.get("HIPCC", "hipcc"), "-O3", "--offload-arch=gfx1151",
         "-shared", "-fPIC", str(SRC), "-o", str(SO)], check=True)
    lib = ctypes.CDLL(str(SO))
    for n in NAMES:
        fn = getattr(lib, n)
        fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3 + [ctypes.c_void_p]
        fn.restype = ctypes.c_int
    return lib


def main() -> None:
    lib = build()
    rt = get_hip_runtime()
    rng = np.random.default_rng(7)
    hidden = rng.normal(0, 0.1, (TOKENS, HIDDEN)).astype(np.float32)
    weight = rng.normal(0, 0.1, (EXPERTS, HIDDEN)).astype(np.float32)

    d_h = malloc(hidden.nbytes)
    d_w = malloc(weight.nbytes)
    d_o = malloc(TOKENS * EXPERTS * 4)
    copy_host_to_device(d_h, host_array_ptr(hidden), hidden.nbytes)
    copy_host_to_device(d_w, host_array_ptr(weight), weight.nbytes)

    ref = np.zeros((TOKENS, EXPERTS), dtype=np.float32)
    qwen35_router_logits_f32_f32w_token_tile4_dense_exact(
        d_h.ptr, d_w.ptr, d_o.ptr, TOKENS, HIDDEN, EXPERTS)
    copy_device_to_host(host_array_ptr(ref), d_o, ref.nbytes)

    def runner(name):
        def go():
            err = getattr(lib, name)(
                ctypes.c_void_p(d_h.ptr), ctypes.c_void_p(d_w.ptr),
                ctypes.c_void_p(d_o.ptr), TOKENS, HIDDEN, EXPERTS, None)
            if err:
                raise RuntimeError(f"{name} failed: {err}")
        return go

    def time(fn) -> float:
        for _ in range(WARMUP):
            fn()
        rt.device_synchronize()
        a, b = rt.event_create(), rt.event_create()
        rt.event_record(a)
        for _ in range(REPS):
            fn()
        rt.event_record(b)
        rt.device_synchronize()
        ms = rt.event_elapsed_time_ms(a, b) / REPS
        rt.event_destroy(a); rt.event_destroy(b)
        return ms

    results = {}
    for name in NAMES:
        go = runner(name)
        go()
        got = np.zeros_like(ref)
        copy_device_to_host(host_array_ptr(got), d_o, got.nbytes)
        results[name] = {
            "median_ms": round(time(go), 4),
            "max_abs_diff_vs_shipped": float(np.abs(ref - got).max()),
        }

    # Baselines for context.
    def shipped():
        qwen35_router_logits_f32_f32w_token_tile4_dense_exact(
            d_h.ptr, d_w.ptr, d_o.ptr, TOKENS, HIDDEN, EXPERTS)
    t_shipped = time(shipped)
    flops = 2.0 * TOKENS * EXPERTS * HIDDEN
    out = {
        "shape": {"tokens": TOKENS, "hidden": HIDDEN, "experts": EXPERTS},
        "shipped_library_ms": round(t_shipped, 4),
        "flops_gflop": round(flops / 1e9, 3),
        "fma_floor_ms": round(flops / 16.4e12 * 1e3, 4),
        "achieved_gflop_s": round(flops / (t_shipped * 1e-3) / 1e9, 1),
        "variants": results,
    }
    print("RESULT " + json.dumps(out, indent=1))
    free(d_h); free(d_w); free(d_o)


if __name__ == "__main__":
    main()
