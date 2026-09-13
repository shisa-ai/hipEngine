#!/usr/bin/env python3
"""Actual-weight exact iu8-risk+repair vs pair2 MoE gate/up A/B screen.

Runs the exact grouped pair2 parent (production early-layer owner) against
the candidate chain: compact WMMA tile map + risk-collecting iu8-WMMA
kernel + sparse exact repair, followed by the same SiLU in both arms.
Actual layer gate/up weights, synthetic uniform or skewed routing, all
pairs must be bit-identical. Reports risk counts, repair share, and
operation-complete medians for both arm orders.
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
    p.add_argument("--rows", type=int, nargs="+", default=[512, 1024])
    p.add_argument("--pairs", type=int, default=10)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--routing", choices=("uniform", "skewed"), default="uniform")
    p.add_argument("--multiplier", type=float, nargs="+", default=[32.0])
    p.add_argument("--routing-capture", type=Path)
    p.add_argument("--routing-case")
    p.add_argument("--routing-chunk", type=int, default=0)
    p.add_argument("--compiler-version-file", type=Path, required=True)
    p.add_argument("--require-cached-build", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not hip_available():
        p.error("HIP runtime unavailable")
    if a.pairs < 1 or any(r < 2 for r in a.rows):
        p.error("positive pairs and rows>=2 required")
    capture = None
    if a.routing_capture:
        from scripts.qwen4exp_routing_capture import (
            select_routing, validate_replay_identity,
        )
        from scripts.qwen4exp_canonical_ar_bench import DEFAULT_FIXTURE, load_fixture
        from scripts.qwen4exp_framework_family_refresh import check_host, model_identity
        if not a.routing_case:
            p.error("capture requires --routing-case")
        check_host()
        model_identity(a.model_root)
        capture = json.loads(a.routing_capture.read_text())
        validate_replay_identity(
            capture, fixture_sha256=load_fixture(DEFAULT_FIXTURE)[1])

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
        "kind": "qwen4exp_q4_iu8_exact_screen",
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": sys.argv,
        "model": "Qwen3.8-Flash-Next UD-Q4_K_XL",
        "weights": identities,
        "arithmetic_class": "T0-exact-candidate",
        "runtime_default_changed": False,
        "boundary": ("grouped Q4_K gate/up plus identical BF16 SiLU; tile map + "
                     "iu8-risk + sparse exact repair versus pair2 parent; no "
                     "routing/down/combine"),
        "parent_variant": "gguf_q4_k_selected_dual_grouped_pair2_bf16_bf16_out",
        "candidate_variants": [
            "gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out",
            "gguf_q4_k_selected_dual_sparse_exact_repair_bf16",
        ],
        "routing": a.routing,
        "layer": a.layer,
        "cases": [],
    }

    allocations = []
    try:
        wa, wb = [_upload(w, runtime, allocations) for w in weights]
        for rows in a.rows:
            if capture is not None:
                counts = select_routing(
                    capture, case_id=a.routing_case, layer=a.layer,
                    chunk=a.routing_chunk, tokens=rows)
                if counts.shape != (512,):
                    raise ValueError("replay requires the model's 512 experts")
            else:
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
            # Production-style static upper bound: avoids the device read.
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
            out_pair_gate = _alloc(compact * 640, np.uint16, runtime, allocations)
            out_pair_up = _alloc(compact * 640, np.uint16, runtime, allocations)
            out_silu = _alloc(compact * 640, np.uint16, runtime, allocations)
            out_iu8 = _alloc(compact * 1280, np.uint16, runtime, allocations)
            out_iu8_silu = _alloc(compact * 640, np.uint16, runtime, allocations)
            d_risk_count = _alloc(1, np.int32, runtime, allocations)
            d_risk_indices = _alloc(capacity, np.int32, runtime, allocations)
            zero = np.zeros(1, dtype=np.int32)
            wmma_total_host = np.empty(1, dtype=np.int64)

            def run_pair() -> None:
                q4.gguf_q4_k_selected_dual_grouped_pair2_bf16_bf16_out(
                    dx.ptr, ds.ptr, wa.ptr, wb.ptr,
                    out_pair_gate.ptr, out_pair_up.ptr,
                    compact, 512, 2560, 640,
                    library=library, runtime=runtime)
                silu_mul_separate_out_bf16(
                    out_pair_gate.ptr, out_pair_up.ptr, out_silu.ptr,
                    compact, 640, runtime=runtime)
                runtime.device_synchronize()

            def run_candidate(multiplier: float) -> None:
                # GPU tile map with the static upper bound, exactly as the
                # production compact WMMA path avoids the device read.
                qwen35_moe_wmma_tile_map(
                    ds.ptr, d_group_wmma.ptr, d_tile_out.ptr, d_group_total.ptr,
                    512, tile_capacity=static_tiles,
                    stream=0, runtime=runtime)
                from hipengine.core.memory import (
                    copy_device_to_host, copy_host_to_device, host_array_ptr,
                )
                copy_host_to_device(d_risk_count, host_array_ptr(zero), 4,
                                    runtime=runtime)
                q4.gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out(
                    dx.ptr, ds.ptr, d_group_wmma.ptr, d_tile_out.ptr,
                    wa.ptr, wb.ptr, out_iu8.ptr,
                    d_risk_count.ptr, d_risk_indices.ptr, capacity, multiplier,
                    compact, 2560, 640, 640, 512, static_rows,
                    library=library, runtime=runtime)
                q4.gguf_q4_k_selected_dual_sparse_exact_repair_bf16(
                    dx.ptr, ds.ptr, wa.ptr, wb.ptr, out_iu8.ptr,
                    d_risk_count.ptr, d_risk_indices.ptr, capacity,
                    compact, 2560, 640, 640, 512,
                    library=library, runtime=runtime)
                silu_mul_dual_out_bf16(
                    out_iu8.ptr, out_iu8_silu.ptr, compact, 640,
                    runtime=runtime)
                runtime.device_synchronize()

            for multiplier in a.multiplier:
                run_pair()
                run_candidate(multiplier)
                ref_gate = _download(out_pair_gate, (compact, 640), np.uint16, runtime)
                ref_up = _download(out_pair_up, (compact, 640), np.uint16, runtime)
                cand = _download(out_iu8, (compact, 1280), np.uint16, runtime)
                gate_flips = int(np.count_nonzero(cand[:, :640] != ref_gate))
                up_flips = int(np.count_nonzero(cand[:, 640:] != ref_up))
                risks = int(_download(d_risk_count, (1,), np.int32, runtime)[0])
                # SiLU must also be bit-identical downstream of the repair.
                ref_silu = _download(out_silu, (compact, 640), np.uint16, runtime)
                cand_silu = _download(out_iu8_silu, (compact, 640), np.uint16, runtime)
                silu_flips = int(np.count_nonzero(cand_silu != ref_silu))

                times = {"pair2": [], "candidate": []}
                for pair in range(a.pairs):
                    order = ((run_pair, "pair2"),
                             (lambda: run_candidate(multiplier), "candidate"))
                    if pair % 2 == 1:
                        order = order[::-1]
                    for fn, name in order:
                        start = time.perf_counter()
                        fn()
                        times[name].append(time.perf_counter() - start)
                    # every pair re-verifies exactness
                    g = _download(out_pair_gate, (compact, 640), np.uint16, runtime)
                    c = _download(out_iu8, (compact, 1280), np.uint16, runtime)
                    np.testing.assert_array_equal(c[:, :640], g)
                    np.testing.assert_array_equal(
                        c[:, 640:], _download(out_pair_up, (compact, 640), np.uint16, runtime))

                case = {
                    "tokens": rows,
                    "risk_multiplier": multiplier,
                    "routing": "captured" if capture is not None else a.routing,
                    "compact_rows": compact,
                    "active_experts": int(np.count_nonzero(counts)),
                    "median_active_expert_rows": float(np.median(counts[counts > 0])),
                    "max_expert_rows": int(counts.max()),
                    "wmma_total_rows": int(wmma_total_rows),
                    "static_upper_bound_rows": int(static_rows),
                    "risk_count": risks,
                    "risk_fraction": risks / (compact * 1280),
                    "pre_repair_gate_flips": gate_flips,
                    "pre_repair_up_flips": up_flips,
                    "silu_flips": silu_flips,
                    "all_pairs_exact": True,
                    "seconds": times,
                    "pair2_median_ms": statistics.median(times["pair2"]) * 1e3,
                    "candidate_median_ms": statistics.median(times["candidate"]) * 1e3,
                    "speedup": (statistics.median(times["pair2"]) /
                                statistics.median(times["candidate"])),
                }
                report["cases"].append(case)
                print(json.dumps(case))
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
