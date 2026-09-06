#!/usr/bin/env python3
"""Actual-weight Q5_K gate/up A/B including the candidate's device group map."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as gemv
from hipengine.kernels.hip_gfx1100.moe import group_scatter as group
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from scripts.qwen4exp_canonical_ar_bench import _git_metadata, _host_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[64, 512])
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--require-cached-build", action="store_true")
    args = parser.parse_args()
    if args.pairs < 1 or any(r < 1 for r in args.rows):
        parser.error("positive rows and pairs required")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    if args.require_cached_build:
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    runtime = get_hip_runtime()
    library = gemv.build_gguf_k_gemv(load=True)
    group_library = group.build_qwen35_moe_group_scatter(load=True)
    readers = [GGUFReader(p) for p in discover_gguf_files(args.model_root)]
    weights = []
    identities = []
    for name in ("blk.2.ffn_gate_exps.weight", "blk.2.ffn_up_exps.weight"):
        reader = next(r for r in readers if any(t.name == name for t in r.info.tensors))
        info = reader.tensor_info(name)
        if info.shape != (512, 640, 2560) or info.ggml_type_name != "Q5_K":
            raise ValueError(f"unexpected binding tensor: {info}")
        raw = reader.tensor_data(name)
        identities.append({
            "path": str(reader.path), "tensor": name, "shape": info.shape,
            "bytes": raw.nbytes, "sha256": hashlib.sha256(raw).hexdigest(),
        })
        weights.append(raw)
    allocations = []

    def upload(array):
        array = np.ascontiguousarray(array)
        result = malloc(array.nbytes, runtime=runtime)
        allocations.append(result)
        copy_host_to_device(result, host_array_ptr(array), runtime=runtime)
        return result

    report = {
        "kind": "qwen4exp_q5k_grouped_row4_gate_up_screen", "schema": 1,
        "command": sys.argv, "source": _git_metadata(ROOT), "host": _host_metadata(),
        "model": "Qwen3.8-Flash-Next UD-Q4_K_XL", "weights": identities,
        "arithmetic_class": "T0", "runtime_default_changed": False,
        "boundary": "two Q5_K selected projections; candidate includes GPU count/prefix/scatter and zeroing",
        "routing_fixture": "seeded distinct top-10 per token, broad and uneven experts; not prompt tuned",
        "cases": [],
    }
    try:
        dw = [upload(w) for w in weights]
        for rows in args.rows:
            mark = len(allocations)
            rng = np.random.default_rng(9021 + rows)
            # Sampling without replacement preserves the real top-k constraint.
            selected = np.argsort(rng.random((rows, 512)), axis=1)[:, :10].astype(np.int64)
            compact = selected.size
            x = rng.normal(0, 0.25, (rows, 2560)).astype(np.float32).view(np.uint32)
            x = ((x + 0x7FFF + ((x >> 16) & 1)) >> 16).astype(np.uint16)
            dx, ds = upload(x), upload(selected)
            counts = upload(np.zeros(512, np.int32))
            padded = upload(np.zeros(512, np.int32))
            starts = upload(np.zeros(513, np.int64))
            total = upload(np.zeros(1, np.int64))
            offsets = upload(np.zeros(512, np.int32))
            lanes = upload(np.zeros(compact, np.int64))
            sorted_experts = upload(np.zeros(compact, np.int64))
            routing = upload(np.ones(compact, np.float32))
            sorted_routing = upload(np.zeros(compact, np.float32))
            parent_out = [upload(np.zeros((compact, 640), np.uint16)) for _ in dw]
            candidate_out = [upload(np.zeros((compact, 640), np.uint16)) for _ in dw]

            def run(mode):
                if mode == "candidate":
                    runtime.memset(counts.ptr, 0, counts.nbytes)
                    group.qwen35_moe_group_count(
                        ds.ptr, counts.ptr, compact, 512, library=group_library, runtime=runtime)
                    group.qwen35_moe_group_prefix(
                        counts.ptr, padded.ptr, starts.ptr, total.ptr, 512, 1,
                        library=group_library, runtime=runtime)
                    runtime.memset(offsets.ptr, 0, offsets.nbytes)
                    group.qwen35_moe_group_scatter(
                        ds.ptr, routing.ptr, starts.ptr, offsets.ptr, lanes.ptr,
                        sorted_experts.ptr, sorted_routing.ptr, compact, 512,
                        library=group_library, runtime=runtime)
                for weight, parent, candidate in zip(dw, parent_out, candidate_out):
                    if mode == "parent":
                        gemv.gguf_q5_k_selected_gemv_bf16_bf16_out(
                            dx.ptr, ds.ptr, weight.ptr, parent.ptr, rows, compact,
                            512, 2560, 640, library=library, runtime=runtime)
                    else:
                        gemv.gguf_q5_k_selected_grouped_row4_gemv_bf16_bf16_out(
                            dx.ptr, starts.ptr, lanes.ptr, weight.ptr, candidate.ptr,
                            rows, compact, 512, 2560, 640, library=library, runtime=runtime)
                runtime.device_synchronize()

            run("parent")
            run("candidate")
            timing = {"parent": [], "candidate": []}
            for pair in range(args.pairs):
                for mode in (("parent", "candidate") if pair % 2 == 0 else ("candidate", "parent")):
                    t0 = time.perf_counter()
                    run(mode)
                    timing[mode].append(time.perf_counter() - t0)
                for parent, candidate in zip(parent_out, candidate_out):
                    a = np.empty((compact, 640), np.uint16)
                    b = np.empty_like(a)
                    copy_device_to_host(host_array_ptr(a), parent, runtime=runtime)
                    copy_device_to_host(host_array_ptr(b), candidate, runtime=runtime)
                    np.testing.assert_array_equal(a, b)
            report["cases"].append({
                "rows": rows, "experts": 512, "topk": 10,
                "active_experts": int(np.unique(selected).size),
                "selected_sha256": hashlib.sha256(selected).hexdigest(),
                "seconds": timing, "all_pairs_exact": True,
                "speedup": statistics.median(timing["parent"]) / statistics.median(timing["candidate"]),
            })
            for ptr in reversed(allocations[mark:]):
                free(ptr, runtime=runtime)
            del allocations[mark:]
    finally:
        runtime.device_synchronize()
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
