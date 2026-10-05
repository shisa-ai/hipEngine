#!/usr/bin/env python3
"""Counterbalanced real-weight tile4 router DPP screen; no model-rate claim."""
from __future__ import annotations

import argparse
import fcntl
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

from scripts.qwen4exp_canonical_ar_bench import _git_metadata, _host_metadata
from scripts.qwen4exp_framework_family_refresh import model_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--compiler-version-file", type=Path, required=True)
    p.add_argument("--require-cached-build", action="store_true")
    p.add_argument("--rows", type=int, nargs="+", default=[512, 1024, 4096])
    p.add_argument("--layers", type=int, nargs="+", default=[0, 22, 47])
    p.add_argument("--pairs", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.pairs < 2 or args.pairs % 2 or any(r < 1 for r in args.rows):
        p.error("positive rows and even pairs>=2 required")
    os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (malloc, free, copy_host_to_device,
                                       copy_device_to_host, host_array_ptr, memory_stats)
    from hipengine.kernels.hip_gfx1100.moe import router
    from hipengine.loading.gguf import GGUFReader, discover_gguf_files

    readers = [GGUFReader(file) for file in discover_gguf_files(args.model_root)]
    report = {
        "schema": 1, "kind": "qwen4exp_router_dpp_operation_screen",
        "performance_claim": False, "default_changed": False,
        "source": _git_metadata(ROOT), "host": _host_metadata(),
        "model": model_identity(args.model_root), "command": sys.argv,
        "boundary": "F32 router logits plus unchanged top10 selection; synchronized wall and HIP events, D2H excluded",
        "inputs": "real F32 weights, synthetic F32 activations; no natural-input/model qualification inferred",
        "cases": [],
    }
    with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rt = get_hip_runtime()
        lib = router.build_qwen35_router(require_cached=args.require_cached_build)
        report["library"] = {"path": lib._name,
                             "sha256": hashlib.sha256(Path(lib._name).read_bytes()).hexdigest()}
        for layer in args.layers:
            name = f"blk.{layer}.ffn_gate_inp.weight"
            reader = next(r for r in readers if any(t.name == name for t in r.info.tensors))
            weight = np.array(reader.tensor_data(name), dtype=np.float32, copy=True)
            n, k = weight.shape
            for rows in args.rows:
                x = np.random.default_rng(10005 + rows + layer).normal(0, .1, (rows, k)).astype(np.float32)
                out = np.empty((rows, n), np.float32)
                ids = np.empty((rows, 10), np.int64)
                routing = np.empty((rows, 10), np.float32)
                arrays = [x, weight, out, ids, routing]
                buffers = []
                start, stop = rt.event_create(), rt.event_create()
                try:
                    for arr in arrays:
                        buffers.append(malloc(arr.nbytes))
                    for arr, buf in zip(arrays[:2], buffers[:2], strict=True):
                        copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes)
                    fns = {"parent": router.qwen35_router_logits_f32_f32w_token_tile4_dense_exact,
                           "dpp": router.qwen35_router_logits_f32_f32w_token_tile4_dpp_exact}
                    def launch(arm):
                        fns[arm](buffers[0].ptr, buffers[1].ptr, buffers[2].ptr,
                                 rows, k, n, library=lib, runtime=rt)
                        router.qwen35_router_select(buffers[2].ptr, buffers[3].ptr,
                                                   buffers[4].ptr, rows, n, n, 10,
                                                   threads=128, library=lib, runtime=rt)
                    hashes = {}
                    for arm in fns:
                        launch(arm)
                        rt.device_synchronize()
                        for arr, buf in zip(arrays[2:], buffers[2:], strict=True):
                            copy_device_to_host(host_array_ptr(arr), buf, arr.nbytes)
                        hashes[arm] = hashlib.sha256(b"".join(a.tobytes() for a in arrays[2:])).hexdigest()
                    if len(set(hashes.values())) != 1:
                        raise RuntimeError(f"candidate changed logits/selection/routing at {name}/{rows}")
                    samples = []
                    for pair in range(args.pairs):
                        order = ("parent", "dpp") if pair % 2 == 0 else ("dpp", "parent")
                        for arm in order:
                            t0 = time.perf_counter()
                            rt.event_record(start)
                            launch(arm)
                            rt.event_record(stop)
                            rt.event_synchronize(stop)
                            wall_ms = 1000 * (time.perf_counter() - t0)
                            samples.append({"arm": arm, "pair": pair, "order": list(order),
                                            "event_ms": rt.event_elapsed_time_ms(start, stop),
                                            "wall_ms": wall_ms})
                    medians = {arm: {key: statistics.median(s[key] for s in samples if s["arm"] == arm)
                                     for key in ("event_ms", "wall_ms")} for arm in fns}
                    row = {"tensor": name, "rows": rows, "k": k, "n": n,
                           "weight_sha256": hashlib.sha256(weight.tobytes()).hexdigest(),
                           "output_hashes": hashes, "bit_exact_logits_ids_routing": True,
                           "median": medians, "samples": samples,
                           "event_speedup": medians["parent"]["event_ms"] / medians["dpp"]["event_ms"],
                           "wall_speedup": medians["parent"]["wall_ms"] / medians["dpp"]["wall_ms"]}
                    report["cases"].append(row)
                    print(name, rows, medians, row["wall_speedup"], flush=True)
                finally:
                    rt.event_destroy(stop)
                    rt.event_destroy(start)
                    for buf in reversed(buffers):
                        free(buf)
        report["memory_after_close"] = memory_stats()
        assert report["memory_after_close"]["active_allocations"] == 0
        assert report["memory_after_close"]["current_allocated_bytes"] == 0
    report["status"] = "completed_operation_screen"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
