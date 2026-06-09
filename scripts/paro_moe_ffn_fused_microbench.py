"""B3/B4 A/B microbench: fused PARO selected-FFN megakernel vs the PRODUCTION
selected-expert FFN chain, swept over decode batch size c (verify/decode tokens).

Verify shape per c: x_rows=c decode tokens, rows=c*top_k selected (token t ->
experts selected[t*top_k:(t+1)*top_k]). The fused megakernel launches c*top_k
blocks (one per (token, expert)) and computes the whole expert FFN on-chip.

BASELINE CORRECTNESS (B4 fix, 2026-06-09): the original baseline compared the
megakernel against the NAIVE non-staged chain at threads=128, which rocprof
showed is an ~8x strawman (~678 us) versus the DEPLOYED selected FFN (~81.6 us:
rotate1 4.8 + gate_up dual 41.7 + silu+rotate+down staged 35.2, two wide
GPU-filling kernels). This microbench now defaults to fp16 (the deployed dtype),
krot=8 (the deployed rotation count), and a production-class in-process baseline
(paro_rotate1 -> gemv_awq_selected_dual_pack8_transposed -> silu_mul_dual_rotate
-> gemv_awq_selected_pack8_transposed) at the production verifier thread count
(64). It reports both streamed throughput and single-launch latency, and the
block count vs the 48-CU W7900 so occupancy starvation is visible. The deployed
rocprof anchor (~81.6 us/layer) is printed for reference; --baseline naive keeps
the old strawman for comparison.

This is the kernel-time half of the C_B<=2 campaign (MEGAKERNEL.md B3->B4): the
megakernel must beat the PRODUCTION selected FFN it replaces at the verify shape
to help C_B. ``performance_claim=false`` -- diagnostic A/B, not a retained row.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

GROUP_SIZE = 128
CUS = 48  # W7900 gfx1100
# rocprof-measured deployed selected FFN per layer (batched B=3 verify, fp16):
#   paro_rotate1 4.8 us + gemv_awq_selected_dual_pack8 41.7 us
#   + gemv_awq_selected_pack8_silu_rotate_staged 35.2 us = 81.6 us
ROCPROF_DEPLOYED_US = 81.6


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


class _Cast:
    """16-bit float conversion helpers for the chosen activation dtype."""

    def __init__(self, dtype: str) -> None:
        self.dtype = dtype

    def to_dev(self, arr: np.ndarray) -> np.ndarray:
        if self.dtype == "fp16":
            return np.ascontiguousarray(arr, dtype=np.float16)
        return _f32_to_bf16_u16(arr)

    def from_dev(self, arr: np.ndarray) -> np.ndarray:
        if self.dtype == "fp16":
            return np.ascontiguousarray(arr, dtype=np.float16).astype(np.float32)
        return _bf16_u16_to_f32(arr)

    @property
    def np_dtype(self):
        return np.float16 if self.dtype == "fp16" else np.uint16

    @property
    def itemsize(self) -> int:
        return 2


def _awq_stack(rng, out_f: int, in_f: int, E: int, cast: _Cast):
    out_packed = out_f // 8
    groups = in_f // GROUP_SIZE
    qw = rng.integers(0, 2**32, size=(E, out_packed, in_f), dtype=np.uint64).astype(np.uint32).view(np.int32)
    qz = rng.integers(0, 2**32, size=(E, groups, out_packed), dtype=np.uint64).astype(np.uint32).view(np.int32)
    sc = rng.uniform(0.001, 0.04, size=(E, groups, out_f)).astype(np.float32)
    return (np.ascontiguousarray(qw), np.ascontiguousarray(qz), cast.to_dev(sc))


def _calib(rng, dim: int, krot: int, cast: _Cast):
    half = GROUP_SIZE // 2
    pairs = np.zeros((krot, dim), np.int16)
    for r in range(krot):
        for g in range(dim // GROUP_SIZE):
            for lane in range(half):
                pairs[r, g * GROUP_SIZE + 2 * lane] = 2 * lane
                pairs[r, g * GROUP_SIZE + 2 * lane + 1] = 2 * lane + 1
    theta = cast.to_dev(rng.uniform(-1, 1, (krot, dim // 2)).astype(np.float32))
    cscale = cast.to_dev(rng.uniform(0.5, 1.5, dim).astype(np.float32))
    return np.ascontiguousarray(pairs), theta, cscale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiler-version-file", type=Path, default=None)
    ap.add_argument("--require-cached-build", action="store_true")
    ap.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16",
                    help="activation/scale dtype; fp16 is the deployed path")
    ap.add_argument("--baseline", choices=("production", "naive"), default="production",
                    help="production = deployed-class chain at threads=64; naive = old strawman")
    ap.add_argument("--sel-threads", type=int, default=64,
                    help="threads for the baseline selected GEMVs (deployed verifier profile = 64)")
    ap.add_argument("--fused-threads", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--ffn-len", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--krot", type=int, default=8, help="deployed model rotation count = 8")
    ap.add_argument("--c", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--metric-c", type=int, default=4, help="c whose fused_ms is the loop metric")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    cast = _Cast(args.dtype)

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.quant.paro_moe_ffn_fused import build_paro_moe_ffn_fused

    if args.dtype == "fp16":
        from hipengine.kernels.hip_gfx1100.fused.paro_silu import silu_mul_dual_rotate_out_fp16 as silu_fn
        from hipengine.kernels.hip_gfx1100.quant.paro_awq_gemv import (
            gemv_awq_selected_dual_pack8_transposed_fp16 as dual_fn,
            gemv_awq_selected_pack8_transposed_fp16 as down_fn,
        )
        from hipengine.kernels.hip_gfx1100.quant.paro_moe_ffn_fused import (
            paro_selected_ffn_fused_fp16_fp16_out as fused_fn,
        )
        from hipengine.kernels.hip_gfx1100.rotary.paro_rotate import paro_rotate1_fp16 as rotate_fn
    else:
        from hipengine.kernels.hip_gfx1100.fused.paro_silu import silu_mul_dual_rotate_out_bf16 as silu_fn
        from hipengine.kernels.hip_gfx1100.quant.paro_awq_gemv import (
            gemv_awq_selected_dual_pack8_transposed_bf16 as dual_fn,
            gemv_awq_selected_pack8_transposed_bf16 as down_fn,
        )
        from hipengine.kernels.hip_gfx1100.quant.paro_moe_ffn_fused import (
            paro_selected_ffn_fused_bf16_bf16_out as fused_fn,
        )
        from hipengine.kernels.hip_gfx1100.rotary.paro_rotate import paro_rotate1_bf16 as rotate_fn

    library = build_paro_moe_ffn_fused(load=True, require_cached=args.require_cached_build)
    rt = get_hip_runtime()
    H, F, K, E, kr = args.hidden, args.ffn_len, args.top_k, args.experts, args.krot
    fp = F // 8  # gate/up out_packed
    hp = H // 8  # down out_packed
    st = args.sel_threads

    rng = np.random.default_rng(11)
    gqw, gqz, gsc = _awq_stack(rng, F, H, E, cast)
    uqw, uqz, usc = _awq_stack(rng, F, H, E, cast)
    dqw, dqz, dsc = _awq_stack(rng, H, F, E, cast)
    r1p, r1t, r1s = _calib(rng, H, kr, cast)
    drp, drt, drs = _calib(rng, F, kr, cast)

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

    print(f"dtype={args.dtype} baseline={args.baseline} hidden={H} ffn_len={F} top_k={K} "
          f"experts={E} krot={kr} sel_threads={st}  CUs={CUS} (W7900)")
    print(f"rocprof deployed selected-FFN anchor = {ROCPROF_DEPLOYED_US:.1f} us/layer "
          f"(rotate1 4.8 + dual 41.7 + silu_rotate_down_staged 35.2)")
    print("GPU time via HIP graph replay (launch-overhead-free); us = us/replay")
    print(f"{'c':>3} {'rows':>5} {'blk':>4} {'CUfill':>7} {'fused_us':>10} {'base_us':>10} "
          f"{'speedup':>8} {'verdict':>14} {'kl':>10}")
    rows_out = []
    metric_fused_ms = None
    for c in args.c:
        rows = c * K
        x = cast.to_dev((rng.standard_normal((c, H)) * 0.1).astype(np.float32))
        selected = np.ascontiguousarray((np.arange(rows) % E).astype(np.int64))
        sz = cast.itemsize
        local = []
        try:
            xb = dev(x, local)
            sb = dev(selected, local)
            of = malloc(rows * H * sz); local.append((of, None))          # fused out
            ou = malloc(rows * H * sz); local.append((ou, None))          # baseline out
            xrot = malloc(c * H * sz); local.append((xrot, None))         # rotated x
            gu = malloc(rows * 2 * F * sz); local.append((gu, None))      # gate_up concat
            inter = malloc(rows * F * sz); local.append((inter, None))    # silu*mul + rotate

            def run_fused(stream=0):
                fused_fn(
                    xb, sb, g[0], g[1], g[2], u[0], u[1], u[2], d[0], d[1], d[2],
                    R1[0], R1[1], R1[2], DR[0], DR[1], DR[2], of.ptr,
                    c, rows, E, H, F, GROUP_SIZE, kr, threads=args.fused_threads,
                    stream=stream, library=library)

            def run_baseline(stream=0):
                rotate_fn(xb, xrot.ptr, R1[0], R1[1], R1[2], c, H, GROUP_SIZE, kr, stream=stream)
                dual_fn(
                    xrot.ptr, sb, g[0], g[1], g[2], u[0], u[1], u[2], gu.ptr,
                    c, rows, H, fp, fp, E, GROUP_SIZE, threads=st, stream=stream)
                silu_fn(gu.ptr, DR[0], DR[1], DR[2], inter.ptr, rows, F, GROUP_SIZE, kr, stream=stream)
                down_fn(
                    inter.ptr, sb, d[0], d[1], d[2], ou.ptr, rows, F, hp, E, GROUP_SIZE,
                    threads=st, stream=stream)

            def graph_us(enqueue):
                # Capture the kernel sequence into a HIP graph and replay it.
                # One host launch per replay (not one per kernel), so the
                # measurement reflects GPU time -- NOT Python ctypes launch
                # overhead, which dominates a per-call loop at these tiny
                # shapes and is what made the old microbench mispredict.
                cap = rt.stream_create()
                enqueue(0); rt.device_synchronize()  # warm modules before capture
                rt.stream_begin_capture(cap)
                enqueue(cap)
                graph = rt.stream_end_capture(cap)
                gexec = rt.graph_instantiate(graph)
                try:
                    for _ in range(args.warmup):
                        rt.graph_launch(gexec, cap)
                    rt.stream_synchronize(cap)
                    t0 = time.perf_counter()
                    for _ in range(args.iters):
                        rt.graph_launch(gexec, cap)
                    rt.stream_synchronize(cap)
                    return (time.perf_counter() - t0) / args.iters * 1e6
                finally:
                    rt.graph_exec_destroy(gexec)
                    rt.graph_destroy(graph)
                    rt.stream_destroy(cap)

            f_us = graph_us(run_fused)
            b_us = graph_us(run_baseline)
            f_ms = f_us / 1e3
            b_ms = b_us / 1e3
            df = np.zeros((rows, H), cast.np_dtype)
            du = np.zeros((rows, H), cast.np_dtype)
            run_fused(); run_baseline(); rt.device_synchronize()
            copy_device_to_host(host_array_ptr(df), of, df.nbytes)
            copy_device_to_host(host_array_ptr(du), ou, du.nbytes)
            kl = _softmax_kl(cast.from_dev(du), cast.from_dev(df))
            speed = b_us / f_us
            verdict = "fused faster" if f_us < b_us else "BASE faster"
            cufill = f"{min(rows, CUS)}/{CUS}"
            print(f"{c:>3} {rows:>5} {rows:>4} {cufill:>7} {f_us:>10.2f} {b_us:>10.2f} "
                  f"{speed:>7.3f}x {verdict:>14} {kl:>10.2e}")
            rows_out.append({"c": c, "rows": rows, "blocks": rows, "cu_fill": min(rows, CUS) / CUS,
                             "fused_ms": f_ms, "fused_gpu_us": f_us,
                             "baseline_ms": b_ms, "baseline_gpu_us": b_us,
                             "speedup_vs_baseline": speed, "fused_faster": bool(f_us < b_us), "kl": kl})
            if c == args.metric_c:
                metric_fused_ms = f_ms
        finally:
            for b, _ in local:
                free(b)

    for b, _ in static:
        free(b)

    if metric_fused_ms is not None:
        row = next(r for r in rows_out if r["c"] == args.metric_c)
        print(f"METRIC fused_ms={metric_fused_ms:.6f} c={args.metric_c} baseline_ms={row['baseline_ms']:.6f} "
              f"speedup_vs_baseline={row['speedup_vs_baseline']:.4f} kl={row['kl']:.3e}")
    if args.json is not None:
        import json
        args.json.write_text(json.dumps(
            {"dtype": args.dtype, "baseline": args.baseline, "hidden": H, "ffn_len": F, "top_k": K,
             "experts": E, "krot": kr, "sel_threads": st, "cus": CUS,
             "rocprof_deployed_us": ROCPROF_DEPLOYED_US,
             "metric_c": args.metric_c, "performance_claim": False, "sweep": rows_out}, indent=2) + "\n")


if __name__ == "__main__":
    main()
