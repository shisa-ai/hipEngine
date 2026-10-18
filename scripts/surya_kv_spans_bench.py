#!/usr/bin/env python3
"""Surya decode attention: rocBLAS parent vs fused ``KVLiveSpans`` kernel.

Isolated, correctness-gated comparison of the two decode-attention routes at a
fixed context length.  Both routes run on the same device buffers with the same
inputs, so the only difference is the kernel chain:

  parent: batched SGEMM (QK^T) + scale + row softmax + batched SGEMM (AV)
          over a materialized ``(nq, max_seq)`` fp32 score row.
  fused:  ``hipengine_surya_full_attn_decode_split_k_f32_spans`` (GQA-fused
          split-K producer) + ``..._reduce_f32``, reading ``KVLiveSpans``.

The fused route reads each K/V plane once per KV head instead of once per query
head, so it moves ``q_per_kv`` times less KV traffic; it also never
materializes the score row.  The parent route ignores span metadata by
construction, so parity is only checked on the identity dense fill (page table
``b -> b``, ``token_positions = arange``, empty eviction mask) where the two
agree; the span-honouring behaviour is gated by ``tests/test_gpu_surya_kv_spans.py``.

Usage:
    python3 scripts/surya_kv_spans_bench.py --max-seq 8580 --max-seq 16384
    python3 scripts/surya_kv_spans_bench.py --json benchmarks/results/x.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.core.memory import (  # noqa: E402
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.rocblas import Rocblas  # noqa: E402
from hipengine.kernels.hip_gfx1100.evie.evie_ops import build_evie_ops  # noqa: E402
from hipengine.kernels.hip_gfx1100.surya.surya_ops import (  # noqa: E402
    build_surya_ops,
    plan_surya_dense_spans,
    surya_full_attn_decode_f32_spans,
)
from hipengine.kvcache import KVLiveSpans  # noqa: E402
from hipengine.core.device import Device  # noqa: E402
from hipengine.core.tensor import Tensor  # noqa: E402
from hipengine.runtime.surya import attention_decode_rocblas_f32  # noqa: E402

NQ = 8
NK = 2
HD = 256
FULL_ATTN_LAYERS = 6
PARENT_PARITY_ATOL = 5.0e-5


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _git(*argv: str) -> str:
    try:
        return subprocess.run(
            ["git", *argv], capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except Exception:  # pragma: no cover - host probing is best effort
        return ""


def _provenance(argv: list[str]) -> dict:
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "command": " ".join(argv),
        "git_revision": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }


def _host() -> str:
    try:
        name = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=30
        ).stdout
        gfx = next(
            (line.split()[1] for line in name.splitlines() if "gfx" in line), "unknown"
        )
        marketing = next(
            (
                line.split(":", 1)[1].strip()
                for line in name.splitlines()
                if "Marketing Name" in line and "CPU" not in line
            ),
            "unknown",
        )
    except Exception:  # pragma: no cover - host probing is best effort
        gfx, marketing = "unknown", "unknown"
    return f"{gfx} ({marketing}), {platform.machine()}, {platform.platform()}"


def _timed(fn, *, warmup: int, runs: int, runtime) -> float:
    for _ in range(warmup):
        fn()
    runtime.device_synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    runtime.device_synchronize()
    return (time.perf_counter() - start) / runs


def _measure(max_seq: int, *, runs: int, warmup: int, chunk_size: int | None = None) -> dict:
    runtime = get_hip_runtime()
    rocblas = Rocblas.load()
    rocblas.set_workspace(0, 0)
    evie = build_evie_ops(load=True)
    surya = build_surya_ops(load=True)

    plan = plan_surya_dense_spans(max_seq, chunk_size=chunk_size)
    rng = np.random.default_rng(0x5A17)
    query = rng.standard_normal((NQ, HD)).astype(np.float32)
    key_plane = rng.standard_normal((NK, max_seq, HD)).astype(np.float32)
    value_plane = rng.standard_normal((NK, max_seq, HD)).astype(np.float32)
    page_table = plan.page_table
    token_positions = plan.token_positions
    evict_mask = plan.evict_mask
    live_count = max_seq
    row_position = max_seq - 1

    allocations = []

    def alloc(nbytes: int):
        buf = malloc(nbytes, runtime=runtime)
        allocations.append(buf)
        return buf

    query_buf = alloc(query.nbytes)
    key_buf = alloc(key_plane.nbytes)
    value_buf = alloc(value_plane.nbytes)
    fused_out = alloc(query.nbytes)
    parent_out = alloc(query.nbytes)
    scores_buf = alloc(NQ * max_seq * 4)
    partial_out = alloc(NQ * plan.num_splits * HD * 4)
    partial_m = alloc(NQ * plan.num_splits * 4)
    partial_l = alloc(NQ * plan.num_splits * 4)
    page_buf = alloc(page_table.nbytes)
    pos_buf = alloc(token_positions.nbytes)
    mask_buf = alloc(evict_mask.nbytes)
    live_buf = alloc(16)

    try:
        for buffer, array in (
            (query_buf, query),
            (key_buf, key_plane),
            (value_buf, value_plane),
            (page_buf, page_table),
            (pos_buf, token_positions),
            (mask_buf, evict_mask),
        ):
            copy_host_to_device(
                buffer, host_array_ptr(array), array.nbytes, runtime=runtime
            )
        scalars = np.array([live_count, row_position], dtype=np.int64)
        copy_host_to_device(
            live_buf, host_array_ptr(scalars), scalars.nbytes, runtime=runtime
        )
        runtime.device_synchronize()

        spans = KVLiveSpans.paged_dense(
            block_table=_tensor(page_buf.ptr, (plan.block_table_len,), "int32"),
            live_counts=_tensor(live_buf.ptr, (1,), "int64"),
            token_positions=_tensor(pos_buf.ptr, (max_seq,), "int64"),
            evict_mask=_tensor(mask_buf.ptr, (max_seq,), "bool"),
            row_positions=_tensor(live_buf.ptr + 8, (1,), "int64"),
            capacity=max_seq,
            block_size=plan.block_size,
            storage_dtype="fp32",
        )

        ptr_cache: dict[tuple, object] = {}

        def ptr_array(ptrs):
            key = tuple(ptrs)
            if key not in ptr_cache:
                host = np.asarray(ptrs, dtype=np.uint64)
                buf = alloc(host.nbytes)
                copy_host_to_device(
                    buf, host_array_ptr(host), host.nbytes, runtime=runtime
                )
                ptr_cache[key] = buf
            return ptr_cache[key]

        def run_parent() -> None:
            attention_decode_rocblas_f32(
                rocblas, evie, runtime, ptr_array,
                q_ptr=query_buf.ptr, k_cache_ptr=key_buf.ptr,
                v_cache_ptr=value_buf.ptr, out_ptr=parent_out.ptr,
                scores_ptr=scores_buf.ptr, total=live_count,
                num_q_heads=NQ, num_key_value_heads=NK, head_dim=HD,
                max_seq=max_seq,
            )

        def run_fused() -> None:
            surya_full_attn_decode_f32_spans(
                query_buf.ptr, key_buf.ptr, value_buf.ptr, fused_out.ptr,
                partial_out.ptr, partial_m.ptr, partial_l.ptr, spans,
                plan.block_size, NQ, NK, HD, HD**-0.5,
                chunk_size=plan.chunk_size,
                library=surya, runtime=runtime,
            )

        parent_us = _timed(run_parent, warmup=warmup, runs=runs, runtime=runtime) * 1e6
        fused_us = _timed(run_fused, warmup=warmup, runs=runs, runtime=runtime) * 1e6

        parent_host = np.empty_like(query)
        fused_host = np.empty_like(query)
        copy_device_to_host(
            host_array_ptr(parent_host), parent_out, parent_host.nbytes, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(fused_host), fused_out, fused_host.nbytes, runtime=runtime
        )
        max_abs = float(np.max(np.abs(parent_host - fused_host)))
        parity = max_abs <= PARENT_PARITY_ATOL

        return {
            "max_seq": max_seq,
            "live_count": live_count,
            "num_q_heads": NQ,
            "num_kv_heads": NK,
            "head_dim": HD,
            "gqa_repeat": NQ // NK,
            "chunk_size": plan.chunk_size,
            "num_splits": plan.num_splits,
            "parent_us_per_layer": parent_us,
            "fused_us_per_layer": fused_us,
            "parent_us_per_token": parent_us * FULL_ATTN_LAYERS,
            "fused_us_per_token": fused_us * FULL_ATTN_LAYERS,
            "speedup": parent_us / fused_us,
            "parent_max_abs_vs_fused": max_abs,
            "parent_parity_atol": PARENT_PARITY_ATOL,
            "correctness": "PASS" if parity else "FAIL",
            "runs": runs,
            "warmup": warmup,
        }
    finally:
        for buffer in allocations:
            free(buffer, runtime=runtime)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-seq", type=int, action="append", default=None,
        help="context length to measure (repeatable)",
    )
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="override the split-K chunk (default: plan_surya_decode_splits)",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    lengths = args.max_seq or [8580, 16384]
    ctypes.CDLL("libamdhip64.so")
    host = _host()
    print(f"host: {host}")
    results = []
    for max_seq in lengths:
        row = _measure(
            max_seq, runs=args.runs, warmup=args.warmup, chunk_size=args.chunk_size
        )
        results.append(row)
        print(
            f"max_seq={row['max_seq']:>6}  "
            f"parent {row['parent_us_per_layer']:8.1f} us/layer  "
            f"fused {row['fused_us_per_layer']:8.1f} us/layer  "
            f"speedup {row['speedup']:5.2f}x  "
            f"(6 layers: {row['parent_us_per_token']/1000:6.2f} -> "
            f"{row['fused_us_per_token']/1000:6.2f} ms/token)  "
            f"max|diff| {row['parent_max_abs_vs_fused']:.2e}  "
            f"{row['correctness']}"
        )

    if args.json:
        payload = {
            "date": time.strftime("%Y-%m-%d"),
            "host": host,
            "provenance": _provenance(sys.argv),
            "model": "datalab-to/surya-ocr-2 (text decoder attention only, fp32)",
            "quant": "fp32",
            "workload": {
                "kind": "isolated decode attention, single query row",
                "num_q_heads": NQ,
                "num_kv_heads": NK,
                "head_dim": HD,
                "full_attention_layers": FULL_ATTN_LAYERS,
                "context_lengths": lengths,
            },
            "protocol": (
                "Same device buffers and inputs for both routes. Timed with "
                "hipDeviceSynchronize on both sides of each region, after "
                "discarded warmups; the reported number is the mean of --runs "
                "iterations. The parent route is the four-dispatch rocBLAS "
                "chain; the fused route is the spans-aware split-K producer "
                "plus its reduce. Correctness gate: max absolute difference "
                "over the (8, 256) output row against the parent on the "
                "identity dense fill, atol 5e-5."
            ),
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
