"""B3/B4 A/B microbench: fused PARO selected-FFN megakernel vs the unfused PARO
chain, swept over decode batch size c (concurrent verify/decode tokens).

Verify shape per c: x_rows=c decode tokens, rows=c*top_k selected (token t ->
experts selected[t*top_k:(t+1)*top_k]). The fused kernel launches c*top_k blocks
(one per (token, expert)); the unfused chain is the production sequence
``paro_rotate1(hidden) -> gemv_awq_selected_dual_pack8 (gate+up) ->
silu_mul_dual_rotate_out (silu*mul + down-rotate) -> gemv_awq_selected_pack8
(down)``. Hot-cache A/B (fair for both); reports per-call ms, fused block count,
fused-vs-unfused KL, and the crossover.

This is the kernel-time half of the C_B<=2 campaign (MEGAKERNEL.md B3->B4): the
megakernel must beat the unfused chain it replaces at the verify shape (default
c=4 for B=3) before it can help C_B. ``performance_claim=false`` -- diagnostic
A/B used to drive the megakernel optimize loop, not a retained speed row.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

GROUP_SIZE = 128


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    lsb = (u32 >> 16) & 1
    return (((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16)).reshape(f32.shape)


def _bf16_u16_to_f32(arr: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(arr, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(u16.shape).copy()


def _softmax_kl(ref: np.ndarray, cand: np.ndarray) -> float:
    ref = ref.astype(np.float64)
    cand = cand.astype(np.float64)

    def logsm(x):
        s = x - x.max(axis=-1, keepdims=True)
        return s - np.log(np.exp(s).sum(axis=-1, keepdims=True))

    p = np.exp(logsm(ref))
    return float(np.mean(np.sum(p * (logsm(ref) - logsm(cand)), axis=-1)))


def _awq_stack(rng, out_f: int, in_f: int, E: int):
    out_packed = out_f // 8
    groups = in_f // GROUP_SIZE
    qw = rng.integers(0, 2**32, size=(E, out_packed, in_f), dtype=np.uint64).astype(np.uint32).view(np.int32)
    qz = rng.integers(0, 2**32, size=(E, groups, out_packed), dtype=np.uint64).astype(np.uint32).view(np.int32)
    sc = rng.uniform(0.001, 0.04, size=(E, groups, out_f)).astype(np.float32)
    return (np.ascontiguousarray(qw), np.ascontiguousarray(qz), _f32_to_bf16_u16(sc))


def _calib(rng, dim: int, krot: int):
    half = GROUP_SIZE // 2
    pairs = np.zeros((krot, dim), np.int16)
    for r in range(krot):
        for g in range(dim // GROUP_SIZE):
            for lane in range(half):
                pairs[r, g * GROUP_SIZE + 2 * lane] = 2 * lane
                pairs[r, g * GROUP_SIZE + 2 * lane + 1] = 2 * lane + 1
    theta = _f32_to_bf16_u16(rng.uniform(-1, 1, (krot, dim // 2)).astype(np.float32))
    cscale = _f32_to_bf16_u16(rng.uniform(0.5, 1.5, dim).astype(np.float32))
    return np.ascontiguousarray(pairs), theta, cscale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiler-version-file", type=Path, default=None)
    ap.add_argument("--require-cached-build", action="store_true")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--ffn-len", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--krot", type=int, default=1)
    ap.add_argument("--c", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--metric-c", type=int, default=4, help="c whose fused_ms is the loop metric")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.fused.paro_silu import silu_mul_dual_rotate_out_bf16
    from hipengine.kernels.hip_gfx1100.quant.paro_awq_gemv import (
        gemv_awq_selected_dual_pack8_transposed_bf16,
        gemv_awq_selected_pack8_transposed_bf16,
    )
    from hipengine.kernels.hip_gfx1100.quant.paro_moe_ffn_fused import (
        build_paro_moe_ffn_fused,
        paro_selected_ffn_fused_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.rotary.paro_rotate import paro_rotate1_bf16

    library = build_paro_moe_ffn_fused(load=True, require_cached=args.require_cached_build)
    rt = get_hip_runtime()
    H, F, K, E, kr = args.hidden, args.ffn_len, args.top_k, args.experts, args.krot
    fp = F // 8  # gate/up out_packed
    hp = H // 8  # down out_packed

    rng = np.random.default_rng(11)
    gqw, gqz, gsc = _awq_stack(rng, F, H, E)
    uqw, uqz, usc = _awq_stack(rng, F, H, E)
    dqw, dqz, dsc = _awq_stack(rng, H, F, E)
    r1p, r1t, r1s = _calib(rng, H, kr)
    drp, drt, drs = _calib(rng, F, kr)

    static = []

    def dev(arr, store=static):
        a = np.ascontiguousarray(arr)
        b = malloc(a.nbytes)
        copy_host_to_device(b, host_array_ptr(a), a.nbytes)
        store.append((b, a))
        return b.ptr

    g = (dev(gqw), dev(gqz), dev(gsc))
    u = (dev(uqw), dev(uqz), dev(usc))
    d = (dev(dqw), dev(dqz), dev(dsc))
    R1 = (dev(r1p), dev(r1t), dev(r1s))
    DR = (dev(drp), dev(drt), dev(drs))

    print(f"hidden={H} ffn_len={F} top_k={K} experts={E} krot={kr}  CUs=48 (W7900)")
    print(f"{'c':>3} {'rows':>5} {'blocks':>7} {'fused_ms':>9} {'unfused_ms':>11} {'speedup':>8} {'verdict':>14} {'kl':>10}")
    rows_out = []
    metric_fused_ms = None
    for c in args.c:
        rows = c * K
        x = _f32_to_bf16_u16((rng.standard_normal((c, H)) * 0.1).astype(np.float32))
        selected = np.ascontiguousarray((np.arange(rows) % E).astype(np.int64))
        local = []
        try:
            xb = dev(x, local)
            sb = dev(selected, local)
            of = malloc(rows * H * 2); local.append((of, None))           # fused out
            ou = malloc(rows * H * 2); local.append((ou, None))           # unfused out
            xrot = malloc(c * H * 2); local.append((xrot, None))          # rotated x
            gu = malloc(rows * 2 * F * 2); local.append((gu, None))       # gate_up concat
            inter = malloc(rows * F * 2); local.append((inter, None))     # silu*mul + rotate

            def run_fused():
                paro_selected_ffn_fused_bf16_bf16_out(
                    xb, sb, g[0], g[1], g[2], u[0], u[1], u[2], d[0], d[1], d[2],
                    R1[0], R1[1], R1[2], DR[0], DR[1], DR[2], of.ptr,
                    c, rows, E, H, F, GROUP_SIZE, kr, threads=256, library=library)

            def run_unfused():
                paro_rotate1_bf16(xb, xrot.ptr, R1[0], R1[1], R1[2], c, H, GROUP_SIZE, kr)
                gemv_awq_selected_dual_pack8_transposed_bf16(
                    xrot.ptr, sb, g[0], g[1], g[2], u[0], u[1], u[2], gu.ptr,
                    c, rows, H, fp, fp, E, GROUP_SIZE, threads=128)
                silu_mul_dual_rotate_out_bf16(gu.ptr, DR[0], DR[1], DR[2], inter.ptr, rows, F, GROUP_SIZE, kr)
                gemv_awq_selected_pack8_transposed_bf16(
                    inter.ptr, sb, d[0], d[1], d[2], ou.ptr, rows, F, hp, E, GROUP_SIZE, threads=128)

            def bench(fn):
                for _ in range(args.warmup):
                    fn()
                rt.device_synchronize()
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    fn()
                rt.device_synchronize()
                return (time.perf_counter() - t0) / args.iters * 1e3

            f_ms = bench(run_fused)
            u_ms = bench(run_unfused)
            df = np.zeros((rows, H), np.uint16)
            du = np.zeros((rows, H), np.uint16)
            run_fused(); run_unfused(); rt.device_synchronize()
            copy_device_to_host(host_array_ptr(df), of, df.nbytes)
            copy_device_to_host(host_array_ptr(du), ou, du.nbytes)
            kl = _softmax_kl(_bf16_u16_to_f32(du), _bf16_u16_to_f32(df))
            speed = u_ms / f_ms
            verdict = "fused faster" if f_ms < u_ms else "unfused faster"
            print(f"{c:>3} {rows:>5} {rows:>7} {f_ms:>9.4f} {u_ms:>11.4f} {speed:>7.3f}x {verdict:>14} {kl:>10.2e}")
            rows_out.append({"c": c, "rows": rows, "blocks": rows, "fused_ms": f_ms,
                             "unfused_ms": u_ms, "speedup": speed, "fused_faster": bool(f_ms < u_ms), "kl": kl})
            if c == args.metric_c:
                metric_fused_ms = f_ms
        finally:
            for b, _ in local:
                free(b)

    for b, _ in static:
        free(b)

    if metric_fused_ms is not None:
        row = next(r for r in rows_out if r["c"] == args.metric_c)
        print(f"METRIC fused_ms={metric_fused_ms:.6f} c={args.metric_c} unfused_ms={row['unfused_ms']:.6f} "
              f"speedup={row['speedup']:.4f} kl={row['kl']:.3e}")
    if args.json is not None:
        import json
        args.json.write_text(json.dumps(
            {"hidden": H, "ffn_len": F, "top_k": K, "experts": E, "krot": kr,
             "metric_c": args.metric_c, "performance_claim": False, "sweep": rows_out}, indent=2) + "\n")


if __name__ == "__main__":
    main()
