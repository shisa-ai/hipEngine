#!/usr/bin/env python3
"""Price the QSA head-tile width on the shipped wave-row kernel.

`qsa_sparse_attention_h256_wave_rows_f32` amortizes each selected K/V load over
``HEADS`` query heads (the shipped setting is HEADS=4, "quad"). The QSA layers
have 24 query heads and 2 KV heads, so 12 query heads share each KV head and
HEADS=4 captures only a third of the available reuse. The three registered
variants differ *only* in that tile width and produce identical arithmetic, so
timing them on one geometry measures exactly what the reuse is worth - the same
question a row-tiling or LDS-staging rewrite would answer, without writing one.

Read the result as "what does K/V amortization buy on this kernel", not as a
prefill rate: the K/V contents are synthetic and only the geometry, the selected
counts and the access pattern are representative.

Example:
    python3 scripts/qwen4exp_qsa_head_tile_microbench.py \
        --rows 256 --stride 2051 --repetitions 20 \
        --output benchmarks/results/<dir>/head-tile.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.attention import qwen4_exp_qsa as qsa
from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
    qwen4_exp_qsa_sparse_attention_paged_bf16_h256_head_pair_rows_f32,
    qwen4_exp_qsa_sparse_attention_paged_bf16_h256_head_quad_rows_f32,
    qwen4_exp_qsa_sparse_attention_paged_bf16_h256_page256_wave_rows_f32,
)
from hipengine.kvcache import KVLiveSpans

import numpy as np

QUERY_HEADS = 24
KV_HEADS = 2
HEAD_DIM = 256
BLOCK_SIZE = 256
CAPACITY = 4352

VARIANTS = (
    ("heads_1_page256", qwen4_exp_qsa_sparse_attention_paged_bf16_h256_page256_wave_rows_f32, 1),
    ("heads_2_head_pair", qwen4_exp_qsa_sparse_attention_paged_bf16_h256_head_pair_rows_f32, 2),
    ("heads_4_head_quad", qwen4_exp_qsa_sparse_attention_paged_bf16_h256_head_quad_rows_f32, 4),
)


def bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round fp32 to BF16 and return the raw 16-bit pattern as uint16."""

    as_f32 = values.astype(np.float32)
    bits = as_f32.view(np.uint32)
    rounded = ((bits + 0x8000) & 0xFFFF0000).astype(np.uint32)
    return (rounded >> 16).astype(np.uint16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--stride", type=int, default=2051,
                        help="selected_stride; the shipped cap is 2051")
    parser.add_argument("--counts", type=int, default=None,
                        help="selected positions per row; defaults to stride - 1")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=913)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = args.rows
    stride = args.stride
    counts = args.counts if args.counts is not None else stride - 1
    if counts > stride:
        raise SystemExit("counts cannot exceed stride")

    runtime = get_hip_runtime()
    rng = np.random.default_rng(args.seed)
    allocations: list = []

    def upload(values: np.ndarray):
        values = np.ascontiguousarray(values)
        device = malloc(values.nbytes, runtime=runtime)
        allocations.append(device)
        copy_host_to_device(device, host_array_ptr(values), runtime=runtime)
        return device

    query = rng.normal(0, 0.4, (rows, QUERY_HEADS, HEAD_DIM)).astype(np.float32)
    keys = bf16_bits(rng.normal(0, 0.4, (CAPACITY, KV_HEADS, HEAD_DIM)))
    values = bf16_bits(rng.normal(0, 0.4, (CAPACITY, KV_HEADS, HEAD_DIM)))
    tables = np.stack([rng.permutation(17) for _ in range(rows)]).astype(np.int32)
    # The shipped selection is the compressed block starts plus the tail; the
    # kernel only walks the first `count` entries, so a sorted sample over the
    # capacity reproduces the access pattern without reproducing the indexer.
    selected = np.stack([
        np.sort(rng.choice(CAPACITY, size=stride, replace=False)) for _ in range(rows)
    ]).astype(np.int64)
    live = np.full(rows, counts, dtype=np.int32)

    try:
        dq = upload(query)
        dk = upload(keys)
        dv = upload(values)
        ds = upload(selected)
        dc = upload(live)
        dt = upload(tables)
        dl = upload(np.full(rows, CAPACITY - 1, np.int64))
        spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(dt.ptr, tables.shape, DType.INT32, Device("hip", 0)),
            live_counts=Tensor.from_handle(dl.ptr, (rows,), DType.INT64, Device("hip", 0)),
            max_live_count=CAPACITY - 1,
            storage_dtype=DType.BF16,
        )
        library = qsa.build_qwen4_exp_qsa(load=True)
        outputs = {name: upload(np.zeros_like(query)) for name, _, _ in VARIANTS}

        def launch(name: str, fn) -> None:
            fn(dq.ptr, dk.ptr, dv.ptr, ds.ptr, dc.ptr, outputs[name].ptr, spans,
               rows=rows, selected_stride=stride, block_size=BLOCK_SIZE,
               query_heads=QUERY_HEADS, kv_heads=KV_HEADS, head_dim=HEAD_DIM,
               library=library, runtime=runtime)

        results = {}
        reference = None
        for name, fn, heads in VARIANTS:
            for _ in range(args.warmup):
                launch(name, fn)
            runtime.device_synchronize()
            import time
            samples = []
            for _ in range(args.repetitions):
                t0 = time.perf_counter()
                launch(name, fn)
                runtime.device_synchronize()
                samples.append((time.perf_counter() - t0) * 1e3)
            median_ms = statistics.median(samples)
            # One chunk's K/V footprint per block: counts x 2 (K,V) x head_dim x 2 B
            groups = QUERY_HEADS // heads
            kv_bytes = groups * rows * counts * 2 * HEAD_DIM * 2
            results[name] = {
                "heads": heads,
                "head_group_blocks_per_row": groups,
                "median_ms": median_ms,
                "min_ms": min(samples),
                "max_ms": max(samples),
                "repetitions": len(samples),
                "kv_read_gb_per_launch": kv_bytes / 1e9,
                "effective_kv_gb_per_s": kv_bytes / (median_ms / 1e3) / 1e9,
                "output_ptr": outputs[name].ptr,
            }

        # The three variants must be bit-identical: only the tile width changes.
        out_arrays = {}
        for name in outputs:
            host = np.empty_like(query)
            copy_device_to_host(host_array_ptr(host), outputs[name], runtime=runtime)
            out_arrays[name] = host
        names = [n for n, _, _ in VARIANTS]
        identical = all(
            np.array_equal(out_arrays[names[0]].view(np.uint32),
                           out_arrays[other].view(np.uint32))
            for other in names[1:]
        )
        baseline = results[names[-1]]["median_ms"]
        for name in names:
            results[name]["speedup_vs_heads_4"] = baseline / results[name]["median_ms"]

        report = {
            "schema": 1,
            "kind": "qwen4exp_qsa_head_tile_microbench",
            "performance_claim": False,
            "question": (
                "What is K/V amortization worth on the shipped QSA wave-row "
                "kernel? The three registered variants differ only in how many "
                "query heads share one selected K/V load."),
            "geometry": {
                "rows": rows,
                "selected_stride": stride,
                "selected_counts": counts,
                "query_heads": QUERY_HEADS,
                "kv_heads": KV_HEADS,
                "head_dim": HEAD_DIM,
                "block_size": BLOCK_SIZE,
                "query_heads_per_kv_head": QUERY_HEADS // KV_HEADS,
                "note": (
                    "Synthetic K/V contents; geometry, counts and access pattern "
                    "are representative, the values are not."),
            },
            "variants": results,
            "bit_identical_across_variants": bool(identical),
            "reading": (
                "A speedup at HEADS=1 or 2 over HEADS=4 means the kernel is "
                "K/V-traffic limited and the shipped quad tile is leaving reuse "
                "on the table; a flat result means the cost is the scalar FMA "
                "issue rate and only tensor cores or a different reduction "
                "structure can move it."),
        }
        text = json.dumps(report, indent=1)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")
        print(text)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
