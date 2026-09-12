#!/usr/bin/env python3
"""Actual-weight exact Q5_K iu8-risk+repair vs row4 MoE gate/up A/B screen.

#15 Q5_K bundle KL-shaving variant. Runs the strict row4 parent (the
layer-2 Q5_K production owner) against the candidate chain: compact WMMA
tile map + risk-collecting iu8-WMMA kernel + sparse exact repair, then the
SiLU in both arms (dual vs separate layout). Actual layer-2 gate/up
weights, synthetic uniform or skewed routing, all pairs must be
bit-identical. Reports risk counts, repair share, and operation-complete
medians for both arm orders.
"""

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
    silu_mul_dual_out_bf16,
    silu_mul_separate_out_bf16,
)
from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
    qwen35_moe_wmma_tile_map,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    gguf_q5_k_selected_grouped_row4_gemv_bf16_bf16_out as row4_parent,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q5_k_q8_1_selected_prefill import (
    build_gguf_q5_k_q8_1_selected_prefill,
    gguf_q5_k_selected_dual_sparse_exact_repair_bf16 as sparse_repair,
    gguf_q5_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out as iu8_risk,
)
from scripts.qwen4exp_canonical_ar_bench import _host_metadata, _git_metadata
from tests.test_gpu_qwen4_exp_pf3_moe_schedules import (
    _upload, _alloc, _download, _make_activation,
)


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def _tile_arrays(counts: np.ndarray):
    starts = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    tiles_per_expert = (counts + 15) // 16
    total_tiles = int(tiles_per_expert.sum())
    tile_expert = np.full(total_tiles, -1, dtype=np.int64)
    wmma_start = np.zeros(counts.shape[0] + 1, dtype=np.int64)
    tile = 0
    for e, n in enumerate(tiles_per_expert):
        wmma_start[e] = tile * 16
        if n:
            tile_expert[tile:tile + n] = e
            tile += int(n)
    wmma_start[-1] = tile * 16
    return starts, wmma_start, tile * 16, tile_expert


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--rows", type=int, nargs="+", default=[512, 1024, 4096])
    p.add_argument("--pairs", type=int, default=10)
    p.add_argument("--layer", type=int, default=2)
    p.add_argument("--routing", choices=("uniform", "skewed"), default="uniform")
    p.add_argument("--multiplier", type=float, nargs="+", default=[16.0])
    p.add_argument("--compiler-version-file", type=Path, required=True)
    p.add_argument("--require-cached-build", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not hip_available():
        p.error("HIP runtime unavailable")
    if a.pairs < 1 or any(r < 2 for r in a.rows):
        p.error("positive pairs and rows>=2 required")

    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(a.compiler_version_file)
    if a.require_cached_build:
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"

    runtime = get_hip_runtime()
    library = build_gguf_q5_k_q8_1_selected_prefill(load=True)
    readers = [GGUFReader(path) for path in discover_gguf_files(a.model_root)]
    weights = []
    identities = []
    for name in (f"blk.{a.layer}.ffn_gate_exps.weight",
                 f"blk.{a.layer}.ffn_up_exps.weight"):
        reader = next(r for r in readers if any(t.name == name for t in r.info.tensors))
        info = reader.tensor_info(name)
        assert info.shape == (512, 640, 2560) and info.ggml_type_name == "Q5_K", info
        raw = reader.tensor_data(name)
        identities.append({
            "tensor": name, "path": str(reader.path), "shape": info.shape,
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
        weights.append(raw)

    report = {
        "schema": 1,
        "kind": "qwen4exp_q5_k_iu8_exact_screen",
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": sys.argv,
        "model": "Qwen3.8-Flash-Next UD-Q4_K_XL",
        "weights": identities,
        "arithmetic_class": "T0-exact-candidate",
        "runtime_default_changed": False,
        "boundary": ("grouped Q5_K gate/up plus identical BF16 SiLU; tile map + "
                     "iu8-risk + sparse exact repair versus the row4 parent; no "
                     "routing/down/combine"),
        "parent_variant": "gguf_q5_k_selected_grouped_row4_gemv_bf16_bf16_out",
        "candidate_variants": [
            "gguf_q5_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out",
            "gguf_q5_k_selected_dual_sparse_exact_repair_bf16",
        ],
        "routing": a.routing,
        "layer": a.layer,
        "cases": [],
    }

    allocations = []
    try:
        wa, wb = [_upload(w, runtime, allocations) for w in weights]
        for rows in a.rows:
            rng = np.random.default_rng(1788 + rows)
            scores = rng.random((rows, 512))
            if a.routing == "skewed":
                priorities = np.exp(rng.normal(0, 1.5, 512))
                scores = -np.log(np.maximum(scores, np.finfo(np.float64).tiny)) / priorities
            selected = np.argsort(scores, axis=1)[:, :10]
            counts = np.bincount(selected.reshape(-1), minlength=512)
            starts, wmma_start, wmma_total_rows, tile_expert = _tile_arrays(counts)
            compact = rows * 10
            capacity = compact * 1280
            active = min(compact, 512)
            static_rows = (active + (compact - active) // 16) * 16
            static_tiles = static_rows // 16
            x, _ = _make_activation(compact, 2560, 3456 + rows)
            dx, ds, dws, dte = [
                _upload(v, runtime, allocations)
                for v in (x, starts, wmma_start, tile_expert)
            ]
            # scheduler-side buffers for the GPU tile map (candidate arm)
            d_group_wmma = _alloc(513, np.int64, runtime, allocations)
            d_group_total = _alloc(1, np.int64, runtime, allocations)
            d_tile_out = _alloc(static_tiles, np.int64, runtime, allocations)
            out_row4_gate = _alloc(compact * 640, np.uint16, runtime, allocations)
            out_row4_up = _alloc(compact * 640, np.uint16, runtime, allocations)
            out_row4_silu = _alloc(compact * 640, np.uint16, runtime, allocations)
            out_iu8 = _alloc(compact * 1280, np.uint16, runtime, allocations)
            out_iu8_silu = _alloc(compact * 640, np.uint16, runtime, allocations)
            d_risk_count = _alloc(1, np.int32, runtime, allocations)
            d_risk_indices = _alloc(capacity, np.int32, runtime, allocations)
            zero = np.zeros(1, dtype=np.int32)
            wmma_total_host = np.empty(1, dtype=np.int64)

            def run_parent() -> None:
                row4_parent(
                    dx.ptr, ds.ptr, None, wa.ptr, out_row4_gate.ptr,
                    compact, compact, 512, 2560, 640,
                    library=None, runtime=runtime)
                row4_parent(
                    dx.ptr, ds.ptr, None, wb.ptr, out_row4_up.ptr,
                    compact, compact, 512, 2560, 640,
                    library=None, runtime=runtime)
                silu_mul_separate_out_bf16(
                    out_row4_gate.ptr, out_row4_up.ptr, out_row4_silu.ptr,
                    compact, 640, runtime=runtime)
                runtime.device_synchronize()

            def run_candidate(multiplier: float) -> int:
                from hipengine.core.memory import copy_host_to_device, host_array_ptr
                copy_host_to_device(d_risk_count, host_array_ptr(zero), 4,
                                    runtime=runtime)
                qwen35_moe_wmma_tile_map(
                    ds.ptr, d_group_wmma.ptr, d_tile_out.ptr, d_group_total.ptr,
                    512, tile_capacity=static_tiles,
                    runtime=runtime)
                from hipengine.core.memory import copy_device_to_host
                copy_device_to_host(host_array_ptr(wmma_total_host), d_group_total,
                                    8, runtime=runtime)
                total_rows = int(wmma_total_host[0])
                if total_rows <= 0 or total_rows > static_tiles * 16:
                    raise RuntimeError("tile row count invalid")
                iu8_risk(
                    dx.ptr, ds.ptr, d_group_wmma.ptr, d_tile_out.ptr,
                    wa.ptr, wb.ptr, out_iu8.ptr,
                    d_risk_count.ptr, d_risk_indices.ptr, capacity,
                    multiplier, compact, 2560, 640, 640, 512, total_rows,
                    library=library, runtime=runtime)
                sparse_repair(
                    dx.ptr, ds.ptr, wa.ptr, wb.ptr, out_iu8.ptr,
                    d_risk_count.ptr, d_risk_indices.ptr, capacity,
                    compact, 2560, 640, 640, 512,
                    library=library, runtime=runtime)
                silu_mul_dual_out_bf16(
                    out_iu8.ptr, out_iu8_silu.ptr, rows=compact, features=640,
                    runtime=runtime)
                runtime.device_synchronize()
                return int(_download(d_risk_count, (1,), np.int32, runtime)[0])

            for multiplier in a.multiplier:
                pair_records = []
                parent_bits = None
                candidate_bits = None
                risk_counts = []
                for pair in range(a.pairs):
                    for order in ("parent-first", "candidate-first"):
                        if order == "parent-first":
                            t0 = time.perf_counter()
                            run_parent()
                            parent_ms = (time.perf_counter() - t0) * 1000.0
                            t0 = time.perf_counter()
                            risk_count = run_candidate(multiplier)
                            candidate_ms = (time.perf_counter() - t0) * 1000.0
                        else:
                            t0 = time.perf_counter()
                            risk_count = run_candidate(multiplier)
                            candidate_ms = (time.perf_counter() - t0) * 1000.0
                            t0 = time.perf_counter()
                            run_parent()
                            parent_ms = (time.perf_counter() - t0) * 1000.0
                        if pair == 0 and order == "parent-first":
                            parent_bits = _download(
                                out_row4_silu, (compact, 640), np.uint16, runtime)
                            candidate_bits = _download(
                                out_iu8_silu, (compact, 640), np.uint16, runtime)
                        risk_counts.append(risk_count)
                        pair_records.append({
                            "pair": pair, "order": order,
                            "parent_ms": round(parent_ms, 3),
                            "candidate_ms": round(candidate_ms, 3),
                            "risk_count": risk_count,
                        })
                    if pair == 0:
                        exact = bool(np.array_equal(parent_bits, candidate_bits))
                        if not exact:
                            diff = int(np.count_nonzero(parent_bits != candidate_bits))
                            raise AssertionError(
                                f"rows={rows} multiplier={multiplier}: {diff} of "
                                f"{parent_bits.size} SiLU outputs differ")
                parent_times = [r["parent_ms"] for r in pair_records]
                candidate_times = [r["candidate_ms"] for r in pair_records]
                case = {
                    "rows": rows,
                    "multiplier": multiplier,
                    "compact": compact,
                    "pairs": a.pairs,
                    "exact_silu": True,
                    "risk_count_median": statistics.median(risk_counts),
                    "risk_count_max": max(risk_counts),
                    "repair_share": round(
                        statistics.median(risk_counts) / (compact * 640), 6),
                    "parent_ms_median": round(statistics.median(parent_times), 3),
                    "candidate_ms_median": round(statistics.median(candidate_times), 3),
                    "speedup": round(
                        statistics.median(parent_times)
                        / statistics.median(candidate_times), 4),
                    "records": pair_records,
                }
                report["cases"].append(case)
                print(
                    f"rows={rows} mult={multiplier}: parent "
                    f"{case['parent_ms_median']} ms candidate "
                    f"{case['candidate_ms_median']} ms speedup "
                    f"{case['speedup']}x risks {case['risk_count_median']} "
                    f"({case['repair_share']*100:.3f}%) exact",
                    flush=True,
                )
    finally:
        for device in allocations:
            free(device)

    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
