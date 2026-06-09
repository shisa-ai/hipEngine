"""B5 A/B microbench: fused MoE FFN megakernel vs unfused raw chain, swept over
batch size c (concurrent decode tokens).

Decode shape per c: x_rows=c, rows=c*top_k selected (token t -> experts
selected[t*top_k:(t+1)*top_k]). The fused kernel launches c*top_k blocks; the
unfused chain parallelizes across output columns (out_features*rows blocks).
This measures whether more tokens lift the fused kernel out of the single-token
occupancy wall. Hot-cache A/B (fair for both); reports per-call ms, fused block
count, fused-vs-unfused KL, and the crossover (if any).
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
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--c", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--json", type=Path, default=None)
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

    rng = np.random.default_rng(11)
    gate = np.ascontiguousarray(np.stack([np.roll(make_q4_k_weight(F, H), e + 1, 0) for e in range(E)]))
    up = np.ascontiguousarray(np.stack([np.roll(make_q4_k_weight(F, H), e + 5, 0) for e in range(E)]))
    down = np.ascontiguousarray(np.stack([np.roll(make_q4_k_weight(H, F), e + 9, 0) for e in range(E)]))

    bufs_static = []
    def dev(arr, store):
        b = malloc(arr.nbytes); copy_host_to_device(b, host_array_ptr(arr), arr.nbytes); store.append(b); return b
    gb = dev(gate, bufs_static); ub = dev(up, bufs_static); db = dev(down, bufs_static)

    print(f"hidden={H} ffn_len={F} top_k={K} experts={E}  CUs=48 (W7900)")
    print(f"{'c':>3} {'rows':>5} {'fused_blocks':>12} {'fused_ms':>9} {'unfused_ms':>11} {'speedup':>8} {'verdict':>14} {'kl':>10}")
    rows_out = []
    for c in args.c:
        rows = c * K
        x = _f32_to_bf16_u16((rng.standard_normal((c, H)) * 1e-3).astype(np.float32))
        selected = np.ascontiguousarray((np.arange(rows) % E).astype(np.int64))
        bufs = []
        try:
            xb = dev(x, bufs); sb = dev(selected, bufs)
            ofb = malloc(rows * H * 2); bufs.append(ofb)
            oub = malloc(rows * H * 2); bufs.append(oub)
            gate_o = malloc(rows * F * 2); bufs.append(gate_o)
            up_o = malloc(rows * F * 2); bufs.append(up_o)
            inter_o = malloc(rows * F * 2); bufs.append(inter_o)

            def run_fused():
                gguf_q4_k_selected_ffn_fused_bf16_bf16_out(
                    xb.ptr, sb.ptr, gb.ptr, ub.ptr, db.ptr, ofb.ptr, c, rows, E, H, F, threads=256)

            def run_unfused():
                gguf_q4_k_selected_dual_gemv_bf16_bf16_out(
                    xb.ptr, sb.ptr, gb.ptr, ub.ptr, gate_o.ptr, up_o.ptr, c, rows, E, H, F, threads=256)
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
                return (time.perf_counter() - t0) / args.iters * 1e3

            f_ms = bench(run_fused); u_ms = bench(run_unfused)
            df = np.zeros((rows, H), np.uint16); du = np.zeros((rows, H), np.uint16)
            run_fused(); run_unfused(); rt.device_synchronize()
            copy_device_to_host(host_array_ptr(df), ofb, df.nbytes)
            copy_device_to_host(host_array_ptr(du), oub, du.nbytes)
            kl = _softmax_kl(_bf16_u16_to_f32(du), _bf16_u16_to_f32(df))
            speed = u_ms / f_ms
            verdict = "fused faster" if f_ms < u_ms else "unfused faster"
            print(f"{c:>3} {rows:>5} {rows:>12} {f_ms:>9.4f} {u_ms:>11.4f} {speed:>7.3f}x {verdict:>14} {kl:>10.2e}")
            rows_out.append({"c": c, "rows": rows, "fused_blocks": rows, "fused_ms": f_ms,
                             "unfused_ms": u_ms, "speedup": speed, "fused_faster": bool(f_ms < u_ms), "kl": kl})
        finally:
            for b in bufs:
                free(b)
    for b in bufs_static:
        free(b)
    if args.json is not None:
        import json
        args.json.write_text(json.dumps({"hidden": H, "ffn_len": F, "top_k": K, "experts": E, "sweep": rows_out}, indent=2) + "\n")


if __name__ == "__main__":
    main()
