#!/usr/bin/env python3
"""Qwen4Exp QSA c1 decode pass-breakdown probe (no production change).

Reproduces the production sparse decode geometry (rows=1, 24 query heads,
2 KV heads, head_dim=256, paged BF16 KV block 256, capacity 4352) and times:

* ``strict_spans``   - the single-kernel strict parent (one CTA per q head);
* ``strict_ordered_three_pass_spans`` - the production decode route
  (scores -> serial online-softmax coefficients -> weighted-V recurrence);
* the dense full-attention decode kernel at the same context, as a
  timing-only reference (its arithmetic is not the sparse contract).

Run under ``rocprofv3 --kernel-trace`` to attribute the ordered route's
cost to its three kernels; plain invocation reports wall medians only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans
from hipengine.kernels.hip_gfx1100.attention import qwen4_exp_qsa as qsa
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    qwen35_paged_full_attn_decode_context_bf16_spans,
)
from hipengine.loading.materialize import float_array_to_bf16_bits
from scripts.qwen4exp_canonical_ar_bench import _git_metadata, _host_metadata


class DecodeFixture:
    """c1 decode fixture: one row, scattered selected positions, permuted pages."""

    def __init__(self, selected: int, context: int, capacity: int = 4352):
        self.selected, self.context = selected, context
        self.runtime = get_hip_runtime()
        self.allocations = []
        rng = np.random.default_rng(506)
        self.query = rng.normal(0, 0.4, (1, 24, 256)).astype(np.float32)
        self.key = float_array_to_bf16_bits(
            rng.normal(0, 0.4, (capacity, 2, 256)).astype(np.float32))
        self.value = float_array_to_bf16_bits(
            rng.normal(0, 0.4, (capacity, 2, 256)).astype(np.float32))
        pages = (context + 255) // 256
        self.tables = np.stack([rng.permutation(pages)]).astype(np.int32)
        # Selected positions: sorted unique, biased toward the live context tail
        # like a real top-k selection, but deterministic and reproducible.
        pool = np.arange(context)
        weights = np.linspace(1.0, 2.0, context)
        pick = rng.choice(pool, size=min(selected, context), replace=False, p=weights / weights.sum())
        self.selected_positions = np.sort(pick).astype(np.int64)
        self.dq, self.dk, self.dv = [self.upload(v) for v in (self.query, self.key, self.value)]
        self.ds = self.upload(self.selected_positions)
        dt = self.upload(self.tables)
        dl = self.upload(np.array([context], dtype=np.int64))
        self.spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(
                dt.ptr, self.tables.shape, DType.INT32, Device("hip", 0)),
            live_counts=Tensor.from_handle(
                dl.ptr, (1,), DType.INT64, Device("hip", 0)),
            max_live_count=context, storage_dtype=DType.BF16)
        self.block_table_len = self.spans.base_offsets.numel
        self.ordered_output = self.upload(np.full(self.query.shape, 23.0, np.float32))
        self.strict_output = self.upload(np.full(self.query.shape, 23.0, np.float32))
        self.dense_output = self.upload(np.full(self.query.shape, 23.0, np.float32))
        self.scores = self.upload(np.zeros((24, len(self.selected_positions)), np.float32))
        # coefficients: two planes of query_heads x selected_count
        self.coefficients = self.upload(
            np.zeros((2, 24, len(self.selected_positions)), np.float32))
        self.counts = self.upload(np.array([len(self.selected_positions)], np.int32))
        self.wave_output = self.upload(np.full(self.query.shape, 23.0, np.float32))
        self.quad_output = self.upload(np.full(self.query.shape, 23.0, np.float32))
        self.v2_output = self.upload(np.full(self.query.shape, 23.0, np.float32))
        self.library = qsa.build_qwen4_exp_qsa(load=True)

    def upload(self, values):
        values = np.ascontiguousarray(values)
        p = malloc(values.nbytes, runtime=self.runtime)
        self.allocations.append(p)
        copy_host_to_device(p, host_array_ptr(values), runtime=self.runtime)
        return p

    def run_ordered(self):
        qsa.qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_f32(
            self.dq.ptr, self.dk.ptr, self.dv.ptr, self.ds.ptr,
            self.scores.ptr, self.coefficients.ptr, self.ordered_output.ptr,
            self.spans,
            selected_count=len(self.selected_positions), block_size=256,
            query_heads=24, kv_heads=2, head_dim=256,
            library=self.library, runtime=self.runtime)
        self.runtime.device_synchronize()

    def run_ordered_v2(self):
        qsa.qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_v2_f32(
            self.dq.ptr, self.dk.ptr, self.dv.ptr, self.ds.ptr,
            self.scores.ptr, self.coefficients.ptr, self.v2_output.ptr,
            self.spans,
            selected_count=len(self.selected_positions), block_size=256,
            query_heads=24, kv_heads=2, head_dim=256,
            library=self.library, runtime=self.runtime)
        self.runtime.device_synchronize()

    def run_strict(self):
        qsa.qwen4_exp_qsa_sparse_attention_paged_bf16_f32(
            self.dq.ptr, self.dk.ptr, self.dv.ptr, self.ds.ptr,
            self.strict_output.ptr, self.spans,
            selected_count=len(self.selected_positions), block_size=256,
            query_heads=24, kv_heads=2, head_dim=256,
            library=self.library, runtime=self.runtime)
        self.runtime.device_synchronize()

    def run_dense(self):
        qwen35_paged_full_attn_decode_context_bf16_spans(
            self.dq.ptr, self.dk.ptr, self.dv.ptr, self.dense_output.ptr,
            self.spans,
            max_context_len=self.context, block_size=256,
            num_q_heads=24, num_kv_heads=2, head_dim=256,
            scale=256 ** -0.5,
            runtime=self.runtime)
        self.runtime.device_synchronize()

    def run_wave(self):
        qsa.qwen4_exp_qsa_sparse_attention_paged_bf16_h256_page256_wave_rows_f32(
            self.dq.ptr, self.dk.ptr, self.dv.ptr, self.ds.ptr, self.counts.ptr,
            self.wave_output.ptr, self.spans,
            rows=1, selected_stride=len(self.selected_positions), block_size=256,
            query_heads=24, kv_heads=2, head_dim=256,
            library=self.library, runtime=self.runtime)
        self.runtime.device_synchronize()

    def run_quad(self):
        qsa.qwen4_exp_qsa_sparse_attention_paged_bf16_h256_head_quad_rows_f32(
            self.dq.ptr, self.dk.ptr, self.dv.ptr, self.ds.ptr, self.counts.ptr,
            self.quad_output.ptr, self.spans,
            rows=1, selected_stride=len(self.selected_positions), block_size=256,
            query_heads=24, kv_heads=2, head_dim=256,
            library=self.library, runtime=self.runtime)
        self.runtime.device_synchronize()

    def download_which(self, which):
        out = np.empty_like(self.query)
        ptr = {"ordered": self.ordered_output, "strict": self.strict_output,
               "dense": self.dense_output, "wave": self.wave_output,
               "quad": self.quad_output, "v2": self.v2_output}[which]
        copy_device_to_host(host_array_ptr(out), ptr, runtime=self.runtime)
        return out

    def download(self, which):
        return self.download_which(which)

    def close(self):
        for p in reversed(self.allocations):
            free(p, runtime=self.runtime)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=int, default=2051)
    parser.add_argument("--context", type=int, default=4097)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--reps", type=int, default=60)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.reps <= 0:
        raise ValueError("warmup must be >= 0 and reps positive")
    if args.selected <= 0 or args.selected > args.context:
        raise ValueError("selected must be positive and <= context")
    if args.compiler_version_file is not None:
        import os
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    if args.require_cached_build:
        import os
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"

    fixture = DecodeFixture(args.selected, args.context)
    report = {
        "schema": 1,
        "kind": "qwen4exp_qsa_decode_pass_probe",
        "command": sys.argv,
        "source": _git_metadata(REPO_ROOT),
        "host": _host_metadata(),
        "geometry": {
            "rows": 1, "query_heads": 24, "kv_heads": 2, "head_dim": 256,
            "block_size": 256, "capacity": 4352,
            "selected": args.selected, "context": args.context,
        },
        "exactness": {},
        "timing_ms": {},
        "note": ("leaf-only probe, no production change; rocprofv3 --kernel-trace "
                 "attributeion intended for the ordered route's three kernels"),
    }
    try:
        fixture.run_strict()
        fixture.run_ordered()
        ordered = fixture.download("ordered")
        strict = fixture.download("strict")
        bit_equal = bool(np.array_equal(ordered.view(np.uint32), strict.view(np.uint32)))
        report["exactness"]["ordered_vs_strict_bit_equal"] = bit_equal
        fixture.run_ordered_v2()
        v2 = fixture.download_which("v2")
        report["exactness"]["v2_vs_ordered_bit_equal"] = bool(
            np.array_equal(v2.view(np.uint32), ordered.view(np.uint32)))
        fixture.run_wave()
        wave = fixture.download_which("wave")
        report["exactness"]["wave_vs_ordered_bit_equal"] = bool(
            np.array_equal(wave.view(np.uint32), ordered.view(np.uint32)))
        fixture.run_quad()
        quad = fixture.download_which("quad")
        report["exactness"]["quad_vs_ordered_bit_equal"] = bool(
            np.array_equal(quad.view(np.uint32), ordered.view(np.uint32)))
        fixture.run_dense()
        dense = fixture.download("dense")
        report["exactness"]["dense_max_abs_diff_vs_ordered"] = float(
            np.max(np.abs(dense - ordered))) if bit_equal else None
        for name, runner in (("ordered", fixture.run_ordered),
                             ("ordered_v2", fixture.run_ordered_v2),
                             ("strict", fixture.run_strict),
                             ("wave", fixture.run_wave),
                             ("quad", fixture.run_quad),
                             ("dense", fixture.run_dense)):
            for _ in range(args.warmup):
                runner()
            samples = []
            for _ in range(args.reps):
                start = time.perf_counter()
                runner()
                samples.append((time.perf_counter() - start) * 1e3)
            report["timing_ms"][name] = {
                "median": statistics.median(samples),
                "mean": statistics.fmean(samples),
                "min": min(samples),
                "max": max(samples),
                "samples": samples,
            }
    finally:
        fixture.close()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "exactness": report["exactness"],
        "timing_ms": {k: round(v["median"], 4) for k, v in report["timing_ms"].items()},
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
