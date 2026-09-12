#!/usr/bin/env python3
"""Feasibility probe: exact pair2 vs Q5_1 MMQ ds4 grouped MoE down.

Measures, on actual layer down weights with synthetic routing, the
operation-complete speed ratio and the BF16 flip fraction of the default-off
weight-exact Q5_1 DP4A MMQ chain (ds4 pack + consumer) against the exact
production pair2 (expertgrid64 m1 row-publish) parent. The flip fraction
drives a future risk-collected sparse exact repair candidate; this probe
changes no runtime default.
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
from hipengine.kernels.hip_gfx1100.quant import gguf_q5_1_mmq_selected_prefill as q51mmq
from hipengine.kernels.hip_gfx1100.quant import qwen4_exp_q5_1 as q51
from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
    qwen4_exp_q5_1_selected_wmma_iu8_risk_prefill_bf16_bf16_out,
    qwen4_exp_q5_1_selected_sparse_exact_repair_row_publish_bf16,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
    gguf_q8_1_mmq_ds4_pack_bf16_d4x3,
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


def _expert_starts(counts: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(counts))).astype(np.int64)


def _tile_map(counts: np.ndarray):
    """Host-side compact tile map: padded 16-row tiles per active expert."""
    experts = counts.shape[0]
    tiles_per = (counts + 15) // 16
    total_tiles = int(tiles_per.sum())
    tile_expert = np.full(total_tiles, -1, dtype=np.int64)
    wmma_start = np.zeros(experts + 1, dtype=np.int64)
    tile = 0
    for e in range(experts):
        wmma_start[e] = tile * 16
        if counts[e] > 0:
            tile_expert[tile:tile + tiles_per[e]] = e
            tile += int(tiles_per[e])
    wmma_start[experts] = tile * 16
    return wmma_start, tile * 16, tile_expert


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--rows", type=int, nargs="+", default=[512, 1024])
    p.add_argument("--pairs", type=int, default=10)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--routing", choices=("uniform", "skewed"), default="uniform")
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
    mmq_library = q51mmq.build_gguf_q5_1_mmq_selected_prefill(load=True)
    parent_library = q51.build_qwen4_exp_q5_1(load=True)
    readers = [GGUFReader(path) for path in discover_gguf_files(a.model_root)]
    name = f"blk.{a.layer}.ffn_down_exps.weight"
    reader = next(r for r in readers if any(t.name == name for t in r.info.tensors))
    info = reader.tensor_info(name)
    assert info.ggml_type_name == "Q5_1", info
    experts, out_features, in_features = info.shape
    assert (out_features, in_features) == (2560, 640), info
    raw = reader.tensor_data(name)
    identity = {
        "tensor": name, "path": str(reader.path), "shape": info.shape,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }

    report = {
        "schema": 1,
        "kind": "qwen4exp_q51_mmq_flip_probe",
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": sys.argv,
        "model": "Qwen3.8-Flash-Next UD-Q4_K_XL",
        "weights": [identity],
        "arithmetic_class": "diagnostic",
        "runtime_default_changed": False,
        "boundary": ("grouped Q5_1 down only: exact pair2 (expertgrid64 m1 "
                     "row-publish) parent versus default-off weight-exact "
                     "ds4 pack + DP4A MMQ chain; no routing/silu/combine"),
        "parent_variant": "selected_grouped_prefill_pair2_row_publish_bf16_bf16_out",
        "candidate_variant": "gguf_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out",
        "routing": a.routing,
        "layer": a.layer,
        "cases": [],
    }

    allocations = []
    try:
        wd = _upload(raw, runtime, allocations)
        planes = 3
        for rows in a.rows:
            mark = len(allocations)
            rng = np.random.default_rng(2899 + rows)
            scores = rng.random((rows, experts))
            if a.routing == "skewed":
                priorities = np.exp(rng.normal(0, 1.5, experts))
                scores = -np.log(np.maximum(scores, np.finfo(np.float64).tiny)) / priorities
            selected = np.argsort(scores, axis=1)[:, :10]
            counts = np.bincount(selected.reshape(-1), minlength=experts)
            starts = _expert_starts(counts)
            wmma_start, wmma_total_rows, tile_expert = _tile_map(counts)
            compact = rows * 10
            x, _ = _make_activation(compact, in_features, 4321 + rows)
            dx = _upload(x, runtime, allocations)
            ds = _upload(starts, runtime, allocations)
            dws = _upload(wmma_start, runtime, allocations)
            dte = _upload(tile_expert, runtime, allocations)
            out_parent = _alloc((compact, out_features), np.uint16, runtime, allocations)
            out_mmq = _alloc((compact, out_features), np.uint16, runtime, allocations)
            out_iu8 = _alloc((compact, out_features), np.uint16, runtime, allocations)
            risk_capacity = compact * out_features
            d_risk_count = _alloc(1, np.int32, runtime, allocations)
            d_risk_indices = _alloc(risk_capacity, np.int32, runtime, allocations)
            zero = np.zeros(1, dtype=np.int32)
            from hipengine.core.memory import copy_host_to_device, host_array_ptr
            ds4_blocks = in_features // 128
            workspace_bytes = compact * planes * ds4_blocks * 144
            d_ws = _alloc(1, np.uint8, runtime, allocations)
            from hipengine.core.memory import malloc
            ws = malloc(workspace_bytes, runtime=runtime)
            allocations.append(ws)

            def run_parent() -> None:
                q51.qwen4_exp_q5_1_selected_grouped_prefill_pair2_row_publish_bf16_bf16_out(
                    dx.ptr, ds.ptr, wd.ptr, out_parent.ptr,
                    compact, experts, in_features, out_features,
                    library=parent_library, runtime=runtime)
                runtime.device_synchronize()

            def run_mmq() -> None:
                gguf_q8_1_mmq_ds4_pack_bf16_d4x3(
                    dx.ptr, ws.ptr, compact, in_features,
                    runtime=runtime)
                q51mmq.gguf_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out(
                    ws.ptr, ds.ptr, wd.ptr, out_mmq.ptr,
                    compact, experts, in_features, out_features, planes,
                    library=mmq_library, runtime=runtime)
                runtime.device_synchronize()

            def run_iu8(mult: float) -> None:
                copy_host_to_device(
                    d_risk_count, host_array_ptr(zero), runtime=runtime)
                qwen4_exp_q5_1_selected_wmma_iu8_risk_prefill_bf16_bf16_out(
                    dx.ptr, ds.ptr, dws.ptr, dte.ptr, wd.ptr,
                    out_iu8.ptr, d_risk_count.ptr, d_risk_indices.ptr,
                    risk_capacity, mult, compact, in_features, out_features,
                    experts, wmma_total_rows,
                    library=parent_library, runtime=runtime)
                qwen4_exp_q5_1_selected_sparse_exact_repair_row_publish_bf16(
                    dx.ptr, ds.ptr, wd.ptr, out_iu8.ptr, d_risk_count.ptr,
                    d_risk_indices.ptr, risk_capacity, compact, in_features,
                    out_features, experts,
                    library=parent_library, runtime=runtime)
                runtime.device_synchronize()

            run_parent()
            run_mmq()
            run_iu8(16.0)
            ref = _download(out_parent, (compact, out_features), np.uint16, runtime)
            cand = _download(out_mmq, (compact, out_features), np.uint16, runtime)
            iu8 = _download(out_iu8, (compact, out_features), np.uint16, runtime)
            risks = int(_download(d_risk_count, (1,), np.int32, runtime)[0])
            flips = int(np.count_nonzero(cand != ref))
            flips_iu8 = int(np.count_nonzero(iu8 != ref))
            diff = cand.astype(np.int32) - ref.astype(np.int32)
            ulp = {str(int(v)): int(c) for v, c in
                   zip(*np.unique(diff[diff != 0], return_counts=True))} \
                if np.any(diff != 0) else {}

            times = {"pair2": [], "mmq": [], "iu8": []}
            for pair in range(a.pairs):
                if pair % 2 == 0:
                    order = (("pair2", run_parent), ("mmq", run_mmq),
                             ("iu8", lambda: run_iu8(16.0)))
                else:
                    order = (("iu8", lambda: run_iu8(16.0)), ("mmq", run_mmq),
                             ("pair2", run_parent))
                for label, fn in order:
                    start = time.perf_counter()
                    fn()
                    times[label].append(time.perf_counter() - start)

            case = {
                "tokens": rows,
                "compact_rows": compact,
                "active_experts": int(np.count_nonzero(counts)),
                "median_active_expert_rows": float(np.median(counts[counts > 0])),
                "max_expert_rows": int(counts.max()),
                "flips": flips,
                "total_outputs": int(cand.size),
                "flip_fraction": flips / int(cand.size),
                "flip_ulp_distribution": ulp,
                "iu8_flips": flips_iu8,
                "iu8_flip_fraction": flips_iu8 / int(cand.size),
                "iu8_queued_risks": risks,
                "iu8_risk_fraction": risks / int(cand.size),
                "seconds": times,
                "pair2_median_ms": statistics.median(times["pair2"]) * 1e3,
                "mmq_median_ms": statistics.median(times["mmq"]) * 1e3,
                "iu8_median_ms": statistics.median(times["iu8"]) * 1e3,
                "speedup": (statistics.median(times["pair2"]) /
                            statistics.median(times["mmq"])),
                "iu8_speedup": (statistics.median(times["pair2"]) /
                                statistics.median(times["iu8"])),
            }
            report["cases"].append(case)
            print(json.dumps(case))
            for ptr in reversed(allocations[mark:]):
                free(ptr, runtime=runtime)
            del allocations[mark:]
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
