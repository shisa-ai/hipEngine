"""Same-host actual Q8 weight projection screen, parent versus uniform scale loads."""

import argparse
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
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as q8
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from scripts.qwen4exp_canonical_ar_bench import _git_metadata, _host_metadata
from tests.test_qwen4_exp_pf3_moe_schedules import _alloc, _download, _upload
from tests.test_qwen4exp_q8_wave_scale import CANDIDATE, PARENT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--compiler-version-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows", type=int, nargs="+", default=[64, 512])
    p.add_argument("--pairs", type=int, default=20)
    p.add_argument("--tensor", action="append")
    p.add_argument("--pressure-mib", type=int, default=0)
    p.add_argument("--gr-composite", action="store_true")
    args = p.parse_args()
    if args.pairs < 1 or args.pressure_mib < 0 or any(rows < 1 for rows in args.rows):
        p.error("positive rows and pair count required")
    if args.pairs > 1 and args.pairs % 2:
        p.error("use an even pair count to balance first/second arm order")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    runtime = get_hip_runtime()
    library = q8.build_gguf_k_gemv(load=True, require_cached=True)
    parent_name, candidate_name = PARENT, CANDIDATE
    if args.gr_composite:
        from tests.test_qwen4exp_gr_wave_scale import CANDIDATE as GR_CANDIDATE
        from tests.test_qwen4exp_gr_wave_scale import PARENT as GR_PARENT

        parent_name, candidate_name = GR_PARENT, GR_CANDIDATE
    readers = [GGUFReader(path) for path in discover_gguf_files(args.model_root)]
    report = {
        "schema": 1,
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": sys.argv,
        "cases": [],
        "model": "Qwen3.8-Flash-Next UD-Q4_K_XL",
        "contract": "T0 F32 output bits",
        "default_changed": False,
        "library_prebound": True,
        "order_balanced": args.pairs % 2 == 0,
        "device_fill_precondition_mib": args.pressure_mib,
        "boundary": "GR up+sigmoid+branch mean" if args.gr_composite else "Q8 projection",
    }
    defaults = (
        ["blk.0.hc_attn_up.weight", "blk.0.hc_ffn_up.weight"]
        if args.gr_composite
        else ["blk.0.attn_gate.weight", "blk.0.ffn_down_shexp.weight"]
    )
    for name in args.tensor or defaults:
        reader = next(r for r in readers if any(t.name == name for t in r.info.tensors))
        info = reader.tensor_info(name)
        assert info.ggml_type_name == "Q8_0"
        n, k = info.shape
        raw = reader.tensor_data(name)
        for rows in args.rows:
            allocations = []
            try:
                x = np.random.default_rng(5486 + rows).normal(0, 0.1, (rows, k)).astype(np.float32)
                dx, dw = [_upload(v, runtime, allocations) for v in (x, raw)]
                outputs = [_alloc((rows, n), np.float32, runtime, allocations) for _ in range(2)]
                normalized = None
                mixed = []
                if args.gr_composite:
                    assert n % 8 == 0
                    normalized = _upload(
                        np.random.default_rng(7813 + rows)
                        .normal(0, 0.2, (rows, n))
                        .astype(np.float32),
                        runtime,
                        allocations,
                    )
                    mixed = [
                        _alloc((rows, n // 4), np.float32, runtime, allocations) for _ in range(2)
                    ]
                pressure = (
                    _alloc((args.pressure_mib * 1024 * 1024,), np.uint8, runtime, allocations)
                    if args.pressure_mib
                    else None
                )

                def run(
                    candidate,
                    dx=dx,
                    dw=dw,
                    outputs=outputs,
                    rows=rows,
                    k=k,
                    n=n,
                    normalized=normalized,
                    mixed=mixed,
                ):
                    fn = getattr(q8, candidate_name if candidate else parent_name)
                    if normalized is not None:
                        fn(
                            dx.ptr,
                            dw.ptr,
                            normalized.ptr,
                            outputs[int(candidate)].ptr,
                            mixed[int(candidate)].ptr,
                            rows,
                            k,
                            4,
                            n // 4,
                            runtime=runtime,
                            library=library,
                        )
                    else:
                        fn(
                            dx.ptr,
                            dw.ptr,
                            outputs[int(candidate)].ptr,
                            rows,
                            k,
                            n,
                            runtime=runtime,
                            library=library,
                        )
                    runtime.device_synchronize()

                run(False)
                run(True)
                samples = [[], []]
                for pair in range(args.pairs):
                    for candidate in (False, True) if pair % 2 == 0 else (True, False):
                        if pressure is not None:
                            runtime.memset(pressure.ptr, pair % 256, pressure.nbytes)
                            runtime.device_synchronize()
                        start = time.perf_counter()
                        run(candidate)
                        samples[int(candidate)].append((time.perf_counter() - start) * 1000)
                    a, b = [_download(o, (rows, n), np.float32, runtime) for o in outputs]
                    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
                    if mixed:
                        ma, mb = [_download(o, (rows, n // 4), np.float32, runtime) for o in mixed]
                        np.testing.assert_array_equal(ma.view(np.uint32), mb.view(np.uint32))
                medians = [statistics.median(s) for s in samples]
                row = {
                    "tensor": name,
                    "shape": [rows, k, n],
                    "weight_sha256": hashlib.sha256(raw).hexdigest(),
                    "parent_ms": medians[0],
                    "candidate_ms": medians[1],
                    "speedup": medians[0] / medians[1],
                    "mean_speedup": statistics.mean(samples[0]) / statistics.mean(samples[1]),
                    "speedup_by_order": {
                        order: statistics.median(samples[0][offset::2])
                        / statistics.median(samples[1][offset::2])
                        for order, offset in (("parent_first", 0), ("candidate_first", 1))
                        if len(samples[0]) > offset
                    },
                    "samples_ms": samples,
                    "exact": True,
                }
                report["cases"].append(row)
                print(json.dumps(row), flush=True)
            finally:
                for ptr in reversed(allocations):
                    free(ptr, runtime=runtime)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
