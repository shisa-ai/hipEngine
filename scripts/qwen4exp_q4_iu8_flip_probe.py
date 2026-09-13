#!/usr/bin/env python3
"""Feasibility probe: exact pair2 vs production iu8-WMMA MoE gate/up.

Measures, on actual layer gate/up weights with synthetic or captured
routing, the operation-complete speed ratio and the BF16 flip fraction of
the production iu8-WMMA kernel against the exact grouped pair2 parent.
The flip fraction is the driver for a future risk-collected sparse exact
repair candidate; this probe changes no runtime default.
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
from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_selected_prefill as q4
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


def _tile_map(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Host-side compact tile map: padded 16-row tiles per active expert."""

    experts = counts.shape[0]
    starts = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    tiles_per_expert = (counts + 15) // 16
    total_tiles = int(tiles_per_expert.sum())
    tile_expert = np.full(total_tiles, -1, dtype=np.int64)
    wmma_start = np.zeros(experts + 1, dtype=np.int64)
    tile = 0
    for e in range(experts):
        wmma_start[e] = tile * 16
        if counts[e] > 0:
            tile_expert[tile:tile + tiles_per_expert[e]] = e
            tile += int(tiles_per_expert[e])
    wmma_start[experts] = tile * 16
    return starts, wmma_start, tile * 16, tile_expert


def _static_upper_bound(selected_rows: int, num_experts: int) -> int:
    active = min(selected_rows, num_experts)
    return (active + (selected_rows - active) // 16) * 16


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--rows", type=int, nargs="+", default=[512, 1024])
    p.add_argument("--pairs", type=int, default=10)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--routing", choices=("uniform", "skewed"), default="uniform")
    p.add_argument("--compiler-version-file", type=Path, required=True)
    p.add_argument("--require-cached-build", action="store_true")
    p.add_argument("--dump-prefix", type=Path, default=None,
                   help="optional prefix to save parent/candidate outputs as .npy")
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
    library = q4.build_gguf_q4_k_selected_prefill(load=True)
    readers = [GGUFReader(path) for path in discover_gguf_files(a.model_root)]
    weights = []
    identities = []
    for name in (f"blk.{a.layer}.ffn_gate_exps.weight",
                 f"blk.{a.layer}.ffn_up_exps.weight"):
        reader = next(r for r in readers if any(t.name == name for t in r.info.tensors))
        info = reader.tensor_info(name)
        assert info.shape == (512, 640, 2560) and info.ggml_type_name == "Q4_K", info
        raw = reader.tensor_data(name)
        identities.append({
            "tensor": name, "path": str(reader.path), "shape": info.shape,
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
        weights.append(raw)

    report = {
        "schema": 1,
        "kind": "qwen4exp_q4_iu8_flip_probe",
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": sys.argv,
        "model": "Qwen3.8-Flash-Next UD-Q4_K_XL",
        "weights": identities,
        "arithmetic_class": "diagnostic",
        "runtime_default_changed": False,
        "boundary": ("grouped Q4_K gate/up only: exact pair2 parent versus "
                     "production iu8-WMMA; no routing/silu/down/combine"),
        "parent_variant": "gguf_q4_k_selected_dual_grouped_pair2_bf16_bf16_out",
        "candidate_variant": "gguf_q4_k_selected_dual_wmma_iu8_prefill_bf16_bf16_out",
        "routing": a.routing,
        "layer": a.layer,
        "cases": [],
    }

    allocations = []
    try:
        wa, wb = [_upload(w, runtime, allocations) for w in weights]
        for rows in a.rows:
            mark = len(allocations)
            rng = np.random.default_rng(1788 + rows)
            scores = rng.random((rows, 512))
            if a.routing == "skewed":
                priorities = np.exp(rng.normal(0, 1.5, 512))
                scores = -np.log(np.maximum(scores, np.finfo(np.float64).tiny)) / priorities
            selected = np.argsort(scores, axis=1)[:, :10]
            counts = np.bincount(selected.reshape(-1), minlength=512)
            starts, wmma_start, wmma_total_rows, tile_expert = _tile_map(counts)
            static_rows = _static_upper_bound(rows * 10, 512)
            compact = rows * 10
            x, _ = _make_activation(compact, 2560, 3456 + rows)
            dx, ds, dws, dte = [
                _upload(v, runtime, allocations)
                for v in (x, starts, wmma_start, tile_expert)
            ]
            out_pair = [_alloc((compact, 640), np.uint16, runtime, allocations)
                        for _ in range(2)]
            out_iu8 = _alloc((compact, 1280), np.uint16, runtime, allocations)

            def run_pair() -> None:
                q4.gguf_q4_k_selected_dual_grouped_pair2_bf16_bf16_out(
                    dx.ptr, ds.ptr, wa.ptr, wb.ptr,
                    out_pair[0].ptr, out_pair[1].ptr,
                    compact, 512, 2560, 640,
                    library=library, runtime=runtime)
                runtime.device_synchronize()

            def run_iu8(total_rows: int) -> None:
                q4.gguf_q4_k_selected_dual_wmma_iu8_prefill_bf16_bf16_out(
                    dx.ptr, ds.ptr, dws.ptr, dte.ptr, wa.ptr, wb.ptr,
                    out_iu8.ptr, compact, 2560, 640, 640, 512, total_rows,
                    library=library, runtime=runtime)
                runtime.device_synchronize()

            run_pair()
            run_iu8(wmma_total_rows)
            ref_gate = _download(out_pair[0], (compact, 640), np.uint16, runtime)
            ref_up = _download(out_pair[1], (compact, 640), np.uint16, runtime)
            cand = _download(out_iu8, (compact, 1280), np.uint16, runtime)
            if a.dump_prefix is not None:
                np.save(f"{a.dump_prefix}-parent-gate.npy", ref_gate)
                np.save(f"{a.dump_prefix}-parent-up.npy", ref_up)
                np.save(f"{a.dump_prefix}-cand-gate.npy", cand[:, :640].copy())
                np.save(f"{a.dump_prefix}-cand-up.npy", cand[:, 640:].copy())
            flips = {
                "gate": int(np.count_nonzero(cand[:, :640] != ref_gate)),
                "up": int(np.count_nonzero(cand[:, 640:] != ref_up)),
            }
            total_outputs = compact * 1280
            # ULP distance of flipped outputs (bf16 bit distance).
            ulp = {}
            for label, ref in (("gate", ref_gate), ("up", ref_up)):
                cols = slice(0, 640) if label == "gate" else slice(640, 1280)
                diff = cand[:, cols].astype(np.int32) - ref.astype(np.int32)
                ulp[label] = {
                    str(int(v)): int(c) for v, c in
                    zip(*np.unique(diff[diff != 0], return_counts=True))
                } if np.any(diff != 0) else {}

            times = {"pair2": [], "iu8": []}
            for pair in range(a.pairs):
                if pair % 2 == 0:
                    order = (("pair2", run_pair),
                             ("iu8", lambda: run_iu8(wmma_total_rows)))
                else:
                    order = (("iu8", lambda: run_iu8(wmma_total_rows)),
                             ("pair2", run_pair))
                for name, fn in order:
                    start = time.perf_counter()
                    fn()
                    times[name].append(time.perf_counter() - start)

            case = {
                "tokens": rows,
                "compact_rows": compact,
                "active_experts": int(np.count_nonzero(counts)),
                "median_active_expert_rows": float(np.median(counts[counts > 0])),
                "max_expert_rows": int(counts.max()),
                "wmma_total_rows": int(wmma_total_rows),
                "static_upper_bound_rows": int(static_rows),
                "flips": flips,
                "flip_fraction": (flips["gate"] + flips["up"]) / total_outputs,
                "flip_ulp_distribution": ulp,
                "seconds": times,
                "pair2_median_ms": statistics.median(times["pair2"]) * 1e3,
                "iu8_median_ms": statistics.median(times["iu8"]) * 1e3,
                "speedup": (statistics.median(times["pair2"]) /
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
