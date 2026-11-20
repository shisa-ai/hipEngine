#!/usr/bin/env python3
"""Micro-probe for the llama.cpp PR #29510 review (branch unified-kv-multiseq).

Diagnostic only: no production default changes, no promotion claim.

Arm A (pool-size invariance): the production split-K batched decode wrapper
runs c4 equal 8K contexts against a 32K-token physical pool and against a
512K-token physical pool. hipEngine's attention iterates per-row live_counts
through per-row page tables, so kernel time must be flat in pool size. This
is the measured confirmation that llama.cpp's unified-cache mask penalty
(attending the whole shared cache and masking other sequences away) does not
exist here.

Arm B (mixed-length empty splits): the production launch formula
num_splits = ceil(max_context_len / 256) makes short rows in a mixed-length
batch launch empty split blocks that only zero partial buffers. Arms B3/B4
isolate that cost: identical short-row valid work, num_splits 128 vs 4.

Kernel under test: the exact wrapper the GGUF resident runner calls at
hipengine/runtime/qwen35_gguf_runner.py:6990
(qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_batch_spans), with the
runner's geometry arguments (gate_stride1=head_dim, gate_stride2=1,
scale=head_dim**-0.5) and the runner's num_splits formula.

Run inside the therock ROCm environment, e.g.:
  ENV_PREFIX=~/miniforge3/envs/therock
  SITE=$ENV_PREFIX/lib/python3.12/site-packages
  env LD_LIBRARY_PATH="$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_devel/lib:$SITE/_rocm_sdk_libraries/lib" \
      HIPENGINE_HIP_ARCH=gfx1151 PYTHONPATH=. \
      $ENV_PREFIX/bin/python scripts/probe_paged_decode_pool_invariance.py \
      --json-out benchmarks/results/<dated>.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import socket
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    build_qwen35_paged_attn_decode,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_batch_spans,
)
from hipengine.kvcache.spans import KVLiveSpans

BLOCK = 256
Q_HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
SCALE = HEAD_DIM ** -0.5


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round float32 to bf16 bit patterns (uint16), numpy-only."""
    f32 = np.asarray(x, dtype=np.float32)
    u32 = f32.view(np.uint32)
    # round-to-nearest-even via the standard bias trick
    lsb = (u32 >> 16) & 1
    rounded = u32 + 0x7FFF + lsb
    return (rounded >> 16).astype(np.uint16)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


class Arm:
    def __init__(self, runtime, device, library, *, name, contexts, page_maps,
                 pool_pages, num_splits, pool_ptr_k, pool_ptr_v):
        self.name = name
        self.contexts = list(contexts)
        self.rows = len(contexts)
        self.num_splits = num_splits
        self.pool_pages = pool_pages
        self.runtime = runtime
        self.device = device
        self.library = library
        self.max_ctx = max(self.contexts)
        self.table_len = max(num_splits, (self.max_ctx + BLOCK - 1) // BLOCK)

        table = np.full((self.rows, self.table_len), -1, dtype=np.int32)
        for r, pages in enumerate(page_maps):
            table[r, :len(pages)] = pages
        live = np.asarray(self.contexts, dtype=np.int64)

        self.ptrs = {}
        def alloc(key, nbytes):
            self.ptrs[key] = runtime.malloc(nbytes)
            return self.ptrs[key]

        def upload(key, arr):
            ptr = alloc(key, arr.nbytes)
            runtime.memcpy(ptr, arr.ctypes.data, arr.nbytes, 1)  # H2D
            return ptr

        upload("table", table)
        upload("live", live)
        rng = np.random.default_rng(1234)
        upload("query", rng.normal(0, 0.1, (self.rows, Q_HEADS, HEAD_DIM)).astype(np.float32))
        upload("gate", bf16_bits(rng.normal(0, 1.0, (self.rows, Q_HEADS, HEAD_DIM))))
        alloc("out", self.rows * Q_HEADS * HEAD_DIM * 2)
        alloc("partial_out", self.rows * Q_HEADS * num_splits * HEAD_DIM * 4)
        alloc("partial_m", self.rows * Q_HEADS * num_splits * 4)
        alloc("partial_l", self.rows * Q_HEADS * num_splits * 4)

        self.spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(self.ptrs["table"], (self.rows, self.table_len), DType.INT32, device),
            live_counts=Tensor.from_handle(self.ptrs["live"], (self.rows,), DType.INT64, device),
            max_live_count=self.max_ctx,
            storage_dtype=DType.BF16,
            span_role="decode",
        )
        self.pool_k = pool_ptr_k
        self.pool_v = pool_ptr_v

        valid_splits = sum((c + BLOCK - 1) // BLOCK for c in self.contexts)
        self.total_blocks = KV_HEADS * num_splits * self.rows
        self.empty_blocks = KV_HEADS * (num_splits * self.rows - valid_splits)

    def call(self):
        qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_batch_spans(
            self.ptrs["query"], self.pool_k, self.pool_v,
            self.ptrs["gate"], self.ptrs["out"],
            self.ptrs["partial_out"], self.ptrs["partial_m"], self.ptrs["partial_l"],
            self.spans, self.rows, BLOCK, self.num_splits, BLOCK,
            Q_HEADS, KV_HEADS, HEAD_DIM, HEAD_DIM, 1, SCALE,
            stream=0, library=self.library, runtime=self.runtime,
        )

    def free(self):
        for ptr in self.ptrs.values():
            self.runtime.free(ptr)


def measure(arm, *, warmup, batches, iters):
    rt = arm.runtime
    for _ in range(warmup):
        arm.call()
    rt.device_synchronize() if hasattr(rt, "device_synchronize") else None
    start = rt.event_create()
    stop = rt.event_create()
    samples = []
    for _ in range(batches):
        rt.event_record(start, 0)
        for _ in range(iters):
            arm.call()
        rt.event_record(stop, 0)
        rt.event_synchronize(stop)
        samples.append(rt.event_elapsed_time_ms(start, stop) * 1000.0 / iters)
    rt.event_destroy(start)
    rt.event_destroy(stop)
    return {
        "per_call_us_median": statistics.median(samples),
        "per_call_us_mean": statistics.mean(samples),
        "per_call_us_min": min(samples),
        "batch_samples_us": samples,
    }


def oracle_check(runtime, device, library):
    """Small random case vs numpy reference: merged split softmax * sigmoid(gate)."""
    rng = np.random.default_rng(7)
    contexts = [512, 256]
    rows = 2
    pages = [[0, 1], [2]]
    pool_pages = 3
    k = rng.normal(0, 0.1, (pool_pages * BLOCK, KV_HEADS, HEAD_DIM)).astype(np.float32)
    v = rng.normal(0, 0.1, (pool_pages * BLOCK, KV_HEADS, HEAD_DIM)).astype(np.float32)
    q = rng.normal(0, 0.1, (rows, Q_HEADS, HEAD_DIM)).astype(np.float32)
    g = rng.normal(0, 1.0, (rows, Q_HEADS, HEAD_DIM)).astype(np.float32)
    k_bits, v_bits, g_bits = bf16_bits(k), bf16_bits(v), bf16_bits(g)
    k_f, v_f = bf16_to_f32(k_bits), bf16_to_f32(v_bits)

    ptrs = {}
    def upload(name, arr):
        ptr = runtime.malloc(arr.nbytes)
        runtime.memcpy(ptr, arr.ctypes.data, arr.nbytes, 1)
        ptrs[name] = ptr
    upload("k", k_bits)
    upload("v", v_bits)
    upload("q", q)
    upload("gate", g_bits)
    table = np.full((rows, 2), -1, dtype=np.int32)
    for r, p in enumerate(pages):
        table[r, :len(p)] = p
    upload("table", table)
    upload("live", np.asarray(contexts, dtype=np.int64))
    for name, nbytes in (
        ("out", rows * Q_HEADS * HEAD_DIM * 2),
        ("partial_out", rows * Q_HEADS * 2 * HEAD_DIM * 4),
        ("partial_m", rows * Q_HEADS * 2 * 4),
        ("partial_l", rows * Q_HEADS * 2 * 4),
    ):
        ptrs[name] = runtime.malloc(nbytes)
    spans = KVLiveSpans.paged_uniform(
        block_table=Tensor.from_handle(ptrs["table"], (rows, 2), DType.INT32, device),
        live_counts=Tensor.from_handle(ptrs["live"], (rows,), DType.INT64, device),
        max_live_count=512, storage_dtype=DType.BF16, span_role="decode",
    )
    qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_batch_spans(
        ptrs["q"], ptrs["k"], ptrs["v"], ptrs["gate"], ptrs["out"],
        ptrs["partial_out"], ptrs["partial_m"], ptrs["partial_l"],
        spans, rows, BLOCK, 2, BLOCK, Q_HEADS, KV_HEADS, HEAD_DIM, HEAD_DIM, 1, SCALE,
        stream=0, library=library, runtime=runtime,
    )
    runtime.stream_synchronize(0)
    out_bits = np.empty(rows * Q_HEADS * HEAD_DIM, dtype=np.uint16)
    runtime.memcpy(out_bits.ctypes.data, ptrs["out"], out_bits.nbytes, 2)  # D2H
    got = bf16_to_f32(out_bits).reshape(rows, Q_HEADS, HEAD_DIM)

    ref = np.empty_like(got)
    for r in range(rows):
        toks = np.concatenate([k_f[p * BLOCK:(p + 1) * BLOCK] for p in pages[r]])[:contexts[r]]
        toks_v = np.concatenate([v_f[p * BLOCK:(p + 1) * BLOCK] for p in pages[r]])[:contexts[r]]
        for h in range(Q_HEADS):
            kvh = h // (Q_HEADS // KV_HEADS)
            scores = (toks[:, kvh] @ q[r, h]) * SCALE
            w = np.exp(scores - scores.max())
            w /= w.sum()
            ref[r, h] = (w @ toks_v[:, kvh]) * (1.0 / (1.0 + np.exp(-g[r, h])))
    ref_bf16 = bf16_to_f32(bf16_bits(ref))
    max_diff = float(np.max(np.abs(got - ref_bf16)))
    finite = bool(np.isfinite(got).all())
    for ptr in ptrs.values():
        runtime.free(ptr)
    return {"max_abs_diff_vs_bf16_numpy": max_diff, "all_finite": finite,
            "pass": finite and max_diff <= 0.02}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    runtime = get_hip_runtime()
    device = Device("hip", 0)
    name_buf = ctypes.create_string_buffer(256)
    runtime.check(runtime.library.hipDeviceGetName(name_buf, ctypes.c_int(256), ctypes.c_int(0)))
    gpu_name = name_buf.value.decode(errors="replace")
    free_b, total_b = runtime.mem_get_info()
    hip_ver = ctypes.c_int()
    runtime.library.hipRuntimeGetVersion(ctypes.byref(hip_ver))
    try:
        import importlib.metadata as md
        rocm_ver = md.version("rocm")
    except Exception:
        rocm_ver = "unknown"
    identity = {
        "hostname": socket.gethostname(),
        "kernel": platform.release(),
        "gpu": gpu_name,
        "pci_bus_id": runtime.device_pci_bus_id(0),
        "hip_runtime_version": hip_ver.value,
        "rocm_sdk_package": rocm_ver,
        "vram_free_gib": round(free_b / 2**30, 3),
        "vram_total_gib": round(total_b / 2**30, 3),
    }
    print("identity:", json.dumps(identity, indent=2))

    t0 = time.time()
    library = build_qwen35_paged_attn_decode(load=True)
    print(f"kernel library ready in {time.time() - t0:.1f}s")

    oracle = oracle_check(runtime, device, library)
    print("oracle:", json.dumps(oracle))
    if not oracle["pass"]:
        print("ORACLE FAILED — aborting before timing", file=sys.stderr)
        return 1

    # Physical pools: 128 pages = 32K tokens; 2048 pages = 512K tokens.
    pool_elems = lambda pages: pages * BLOCK * KV_HEADS * HEAD_DIM
    small_k = runtime.malloc(pool_elems(128) * 2)
    small_v = runtime.malloc(pool_elems(128) * 2)
    large_k = runtime.malloc(pool_elems(2048) * 2)
    large_v = runtime.malloc(pool_elems(2048) * 2)
    for ptr, n in ((small_k, 128), (small_v, 128), (large_k, 2048), (large_v, 2048)):
        runtime.memset(ptr, 0, pool_elems(n) * 2)
    runtime.stream_synchronize(0)

    def pages(start, count):
        return list(range(start, start + count))

    arm_specs = [
        # A: equal 4x8K, pool 32K vs 512K
        ("A1_equal8K_pool32K", [8192] * 4, [pages(r * 32, 32) for r in range(4)], 128, 32, small_k, small_v),
        ("A2_equal8K_pool512K", [8192] * 4, [pages(r * 512, 32) for r in range(4)], 2048, 32, large_k, large_v),
        # B: mixed-length vs controls
        ("B1_mixed_32K_1K_1K_1K", [32768, 1024, 1024, 1024],
         [pages(0, 128), pages(256, 4), pages(512, 4), pages(768, 4)], 2048, 128, large_k, large_v),
        ("B2_equal_4x8960_same_total", [8960] * 4, [pages(r * 64, 35) for r in range(4)], 2048, 35, large_k, large_v),
        ("B3_short4x1K_splits128", [1024] * 4, [pages(r * 64, 4) for r in range(4)], 2048, 128, large_k, large_v),
        ("B4_short4x1K_splits4", [1024] * 4, [pages(r * 64, 4) for r in range(4)], 2048, 4, large_k, large_v),
        ("B5_long1x32K_splits128", [32768], [pages(0, 128)], 2048, 128, large_k, large_v),
        ("B6_mixed_128K_1K_1K_1K", [131072, 1024, 1024, 1024],
         [pages(0, 512), pages(640, 4), pages(704, 4), pages(768, 4)], 2048, 512, large_k, large_v),
        ("B7_short4x1K_splits512", [1024] * 4, [pages(r * 64, 4) for r in range(4)], 2048, 512, large_k, large_v),
        ("B8_long1x128K_splits512", [131072], [pages(0, 512)], 2048, 512, large_k, large_v),
    ]
    results = {}
    for name, contexts, page_maps, pool_pages, splits, pk, pv in arm_specs:
        arm = Arm(runtime, device, library, name=name, contexts=contexts,
                  page_maps=page_maps, pool_pages=pool_pages, num_splits=splits,
                  pool_ptr_k=pk, pool_ptr_v=pv)
        stats = measure(arm, warmup=args.warmup, batches=args.batches, iters=args.iters)
        results[name] = {
            "contexts": contexts,
            "pool_tokens": pool_pages * BLOCK,
            "num_splits": splits,
            "grid_blocks": arm.total_blocks,
            "empty_split_blocks": arm.empty_blocks,
            **stats,
        }
        print(f"{name}: median {stats['per_call_us_median']:.1f} us "
              f"(min {stats['per_call_us_min']:.1f}), blocks {arm.total_blocks}, "
              f"empty {arm.empty_blocks}")
        arm.free()

    a_delta = (results["A2_equal8K_pool512K"]["per_call_us_median"]
               / results["A1_equal8K_pool32K"]["per_call_us_median"] - 1.0)
    b_empty_us = (results["B3_short4x1K_splits128"]["per_call_us_median"]
                  - results["B4_short4x1K_splits4"]["per_call_us_median"])
    verdict = {
        "arm_A_pool512K_vs_pool32K_delta_pct": round(a_delta * 100, 2),
        "arm_A_flat_within_5pct": abs(a_delta) <= 0.05,
        "arm_B_empty_split_overhead_us_B3_minus_B4": round(b_empty_us, 2),
        "arm_B_empty_overhead_pct_of_B3": round(
            b_empty_us / results["B3_short4x1K_splits128"]["per_call_us_median"] * 100, 2),
        "arm_B_mixed_vs_equal_total_pct": round(
            (results["B1_mixed_32K_1K_1K_1K"]["per_call_us_median"]
             / results["B2_equal_4x8960_same_total"]["per_call_us_median"] - 1.0) * 100, 2),
        "arm_B_empty_split_overhead_us_at512_B7_minus_B4": round(
            results["B7_short4x1K_splits512"]["per_call_us_median"]
            - results["B4_short4x1K_splits4"]["per_call_us_median"], 2),
        "arm_B_mixed128K_vs_long_only128K_pct": round(
            (results["B6_mixed_128K_1K_1K_1K"]["per_call_us_median"]
             / results["B8_long1x128K_splits512"]["per_call_us_median"] - 1.0) * 100, 2),
    }
    print("verdict:", json.dumps(verdict, indent=2))

    artifact = {
        "probe": "pr29510-paged-decode-pool-invariance",
        "date": "2026-11-08",
        "status": "diagnostic",
        "branch": "unified-kv-multiseq",
        "purpose": ("Confirm hipEngine paged decode attention is invariant to shared-pool size "
                    "(no llama.cpp-style unified-KV mask penalty) and quantify mixed-length "
                    "empty-split overhead in the production split-K batch wrapper."),
        "identity": identity,
        "kernel_wrapper": "qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_batch_spans",
        "kernel_symbols": [
            "hipengine_qwen35_paged_full_attn_decode_split_k_gqa_context_batch_bf16_spans",
            "hipengine_qwen35_paged_full_attn_decode_split_k_reduce_gate_bf16_batch",
        ],
        "production_call_site": "hipengine/runtime/qwen35_gguf_runner.py:6990 (kv_live_spans_batch_split_gqa)",
        "geometry": {"q_heads": Q_HEADS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM,
                     "block_size": BLOCK, "model_class": "Qwen3.8-27B (24Q/4KV/head-256)"},
        "method": {"warmup": args.warmup, "batches": args.batches, "iters_per_batch": args.iters,
                   "timing": "HIP events around back-to-back wrapper calls (context kernel + gated reduce), per-call us = batch elapsed / iters",
                   "num_splits_formula": "ceil(max_context_len / 256), as at the production call site"},
        "oracle": oracle,
        "arms": results,
        "verdict": verdict,
        "command": " ".join(sys.argv),
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2) + "\n")
        print("artifact:", out)
    for ptr in (small_k, small_v, large_k, large_v):
        runtime.free(ptr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())