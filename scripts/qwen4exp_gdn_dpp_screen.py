"""Counterbalanced complete GDN recurrence/norm/gate DPP screen."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_canonical_ar_bench import _host_metadata, _git_metadata
from scripts.qwen4exp_framework_family_refresh import check_host


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[16, 17, 64, 512, 1024, 4096])
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_host()
    if args.pairs < 2 or any(r < 16 for r in args.rows):
        raise ValueError("at least two pairs and rows>=16 required")
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_host_to_device, host_array_ptr, free
    from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
        build_qwen4_exp_gdn, build_qwen4_exp_gdn_dpp, qwen4_exp_gdn_prefill_tiled16_f32,
    )
    from tests.test_gpu_qwen4_exp_gdn_tiled16_prefill import _upload, _alloc, _download

    with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime = get_hip_runtime()
        libs = {
            "parent": build_qwen4_exp_gdn(require_cached=args.require_cached_build),
            "dpp": build_qwen4_exp_gdn_dpp(require_cached=args.require_cached_build),
        }
        report = {
            "kind": "qwen4exp_gdn_dpp_screen", "status": "running", "performance_claim": False,
            "host": _host_metadata(), "source": _git_metadata(ROOT), "command": sys.argv,
            "arithmetic": "F32 output and recurrent state; synthetic Hk16/Hv48/D128",
            "boundary": "recurrence plus output norm/gate; state reset outside events",
            "libraries": {name: {"path": lib._name, "sha256": hashlib.sha256(Path(lib._name).read_bytes()).hexdigest()}
                          for name, lib in libs.items()},
            "cases": [],
        }
        for rows in args.rows:
            rng = np.random.default_rng(83194 + rows)
            arrays = [
                rng.normal(0, .05, (rows, 10240)).astype(np.float32),
                rng.normal(0, .5, (rows, 6144)).astype(np.float32),
                rng.normal(-.2, .1, (rows, 48)).astype(np.float32),
                rng.normal(0, .2, (rows, 48)).astype(np.float32),
                rng.normal(-1, .1, 48).astype(np.float32),
                -np.exp(rng.normal(-.5, .1, 48)).astype(np.float32),
                rng.normal(1, .05, 128).astype(np.float32),
            ]
            initial = rng.normal(0, .01, (48, 128, 128)).astype(np.float32)
            allocations = []
            try:
                inputs = [_upload(a, runtime, allocations) for a in arrays]
                state = _upload(initial, runtime, allocations)
                out = _alloc((rows, 6144), np.float32, runtime, allocations)

                def launch(name):
                    qwen4_exp_gdn_prefill_tiled16_f32(
                        *(a.ptr for a in inputs), state.ptr, out.ptr,
                        rows, 16, 48, 128, 128, library=libs[name], runtime=runtime,
                    )

                gold = None
                for name in libs:
                    copy_host_to_device(state, host_array_ptr(initial), runtime=runtime)
                    launch(name)
                    runtime.device_synchronize()
                    result = (_download(out, (rows, 6144), np.float32, runtime).tobytes(),
                              _download(state, initial.shape, np.float32, runtime).tobytes())
                    if gold is not None and gold != result:
                        raise ValueError(f"DPP changed outputs/state at rows{rows}")
                    gold = result
                samples = []
                for pair in range(args.pairs):
                    for name in (("parent", "dpp") if pair % 2 == 0 else ("dpp", "parent")):
                        copy_host_to_device(state, host_array_ptr(initial), runtime=runtime)
                        start, stop = runtime.event_create(), runtime.event_create()
                        try:
                            runtime.event_record(start)
                            launch(name)
                            runtime.event_record(stop)
                            runtime.event_synchronize(stop)
                            samples.append({"arm": name, "pair": pair,
                                            "ms": runtime.event_elapsed_time_ms(start, stop)})
                        finally:
                            runtime.event_destroy(stop)
                            runtime.event_destroy(start)
                medians = {name: statistics.median(s["ms"] for s in samples if s["arm"] == name)
                           for name in libs}
                row = {"rows": rows, "bit_exact_outputs_and_state": True,
                       "median_ms": medians, "samples": samples,
                       "parent_over_dpp": medians["parent"] / medians["dpp"]}
                report["cases"].append(row)
                print(rows, medians, row["parent_over_dpp"], flush=True)
            finally:
                for allocation in reversed(allocations):
                    free(allocation, runtime=runtime)
        report["status"] = "completed"
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
