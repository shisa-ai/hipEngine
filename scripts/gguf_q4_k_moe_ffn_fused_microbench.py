"""B2 A/B microbench: fused MoE FFN megakernel vs the unfused raw Q4_K chain.

Decode shape (1 token, top_k selected experts). Times both paths writing the same
``moe_down_out[rows, hidden]`` and reports per-call latency, launch count, and the
fused-vs-unfused KL (both paths should agree within bf16 since they compute the
same selected-expert FFN). This is the honest per-layer measurement behind the
B2 verdict; it does not load the full model.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    lsb = (u32 >> 16) & 1
    return (((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16)).reshape(f32.shape)


def _bf16_u16_to_f32(arr: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(arr, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(u16.shape).copy()


def _softmax_kl(ref: np.ndarray, cand: np.ndarray) -> float:
    ref = ref.astype(np.float64); cand = cand.astype(np.float64)
    def logsm(x):
        s = x - x.max(axis=-1, keepdims=True)
        return s - np.log(np.exp(s).sum(axis=-1, keepdims=True))
    p = np.exp(logsm(ref))
    return float(np.mean(np.sum(p * (logsm(ref) - logsm(cand)), axis=-1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiler-version-file", type=Path, default=None)
    ap.add_argument("--require-cached-build", action="store_true")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--ffn-len", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.fused.paro_silu import silu_mul_separate_out_bf16
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        gguf_q4_k_selected_dual_gemv_bf16_bf16_out,
        gguf_q4_k_selected_gemv_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_moe_ffn_fused import (
        build_gguf_q4_k_moe_ffn_fused,
        gguf_q4_k_selected_ffn_fused_bf16_bf16_out,
    )
    from tests._gguf_synthetic_weights import make_q4_k_weight

    build_gguf_q4_k_moe_ffn_fused(load=True, require_cached=args.require_cached_build)
    rt = get_hip_runtime()
    H, F, K, E = args.hidden, args.ffn_len, args.top_k, args.experts
    rows = K

    rng = np.random.default_rng(11)
    x = _f32_to_bf16_u16((rng.standard_normal((1, H)) * 1e-3).astype(np.float32))
    selected = np.ascontiguousarray((np.arange(rows) % E).astype(np.int64))
    gate = np.ascontiguousarray(np.stack([np.roll(make_q4_k_weight(F, H), e + 1, 0) for e in range(E)]))
    up = np.ascontiguousarray(np.stack([np.roll(make_q4_k_weight(F, H), e + 5, 0) for e in range(E)]))
    down = np.ascontiguousarray(np.stack([np.roll(make_q4_k_weight(H, F), e + 9, 0) for e in range(E)]))

    bufs = []
    def dev(arr):
        b = malloc(arr.nbytes); copy_host_to_device(b, host_array_ptr(arr), arr.nbytes); bufs.append(b); return b
    try:
        xb, sb, gb, ub, db = dev(x), dev(selected), dev(gate), dev(up), dev(down)
        down_fused = np.zeros((rows, H), np.uint16); ofb = malloc(down_fused.nbytes); bufs.append(ofb)
        down_unf = np.zeros((rows, H), np.uint16); oub = malloc(down_unf.nbytes); bufs.append(oub)
        gate_o = malloc(rows * F * 2); bufs.append(gate_o)
        up_o = malloc(rows * F * 2); bufs.append(up_o)
        inter_o = malloc(rows * F * 2); bufs.append(inter_o)

        def run_fused():
            gguf_q4_k_selected_ffn_fused_bf16_bf16_out(
                xb.ptr, sb.ptr, gb.ptr, ub.ptr, db.ptr, ofb.ptr, 1, rows, E, H, F, threads=256)

        def run_unfused():
            gguf_q4_k_selected_dual_gemv_bf16_bf16_out(
                xb.ptr, sb.ptr, gb.ptr, ub.ptr, gate_o.ptr, up_o.ptr, 1, rows, E, H, F, threads=256)
            silu_mul_separate_out_bf16(gate_o.ptr, up_o.ptr, inter_o.ptr, rows, F)
            gguf_q4_k_selected_gemv_bf16_bf16_out(
                inter_o.ptr, sb.ptr, db.ptr, oub.ptr, rows, rows, E, F, H, threads=256)

        def bench(fn):
            for _ in range(args.warmup):
                fn()
            rt.device_synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                fn()
            rt.device_synchronize()
            return (time.perf_counter() - t0) / args.iters * 1e3  # ms/call

        f_ms = bench(run_fused)
        u_ms = bench(run_unfused)
        run_fused(); run_unfused(); rt.device_synchronize()
        copy_device_to_host(host_array_ptr(down_fused), ofb, down_fused.nbytes)
        copy_device_to_host(host_array_ptr(down_unf), oub, down_unf.nbytes)
        kl = _softmax_kl(_bf16_u16_to_f32(down_unf), _bf16_u16_to_f32(down_fused))

        print(f"shape: 1 token x top_k={K} selected, hidden={H}, ffn_len={F}, E={E}")
        print(f"fused   : {f_ms:.4f} ms/call   (1 launch/layer)")
        print(f"unfused : {u_ms:.4f} ms/call   (3 launches/layer: dual_gemv + silu + down)")
        print(f"speedup (unfused/fused): {u_ms / f_ms:.3f}x  ({'fused faster' if f_ms < u_ms else 'unfused faster'})")
        print(f"per-token FFN (x40 layers): fused {f_ms*40:.2f} ms  unfused {u_ms*40:.2f} ms")
        print(f"fused-vs-unfused KL: {kl:.3e}  (both compute the same selected FFN)")
    finally:
        for b in bufs:
            free(b)


if __name__ == "__main__":
    main()
