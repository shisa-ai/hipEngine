#!/usr/bin/env python3
"""Persistent-barrier microbench driver (task #105, Phase 2 step 1).

Builds scripts/persistent_barrier_microbench.hip and compares, for identical
memory-bound work, N separate HIP launches vs ONE persistent cooperative kernel
with N in-kernel grid.sync() barriers.  Verdict gates the persistent-forward
(3-5x) program: persistent is viable iff the per-stage in-kernel barrier costs
materially less than a kernel-launch dispatch boundary.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.build import build_hip
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import DeviceBuffer, free, malloc

_SOURCE = Path(__file__).with_name("persistent_barrier_microbench.hip")


def _lib() -> ctypes.CDLL:
    return build_hip(sources=[_SOURCE], family="diag", profile="baseline",
                     output_name="persistent_barrier_microbench.so", load=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    lib = _lib()
    rt = get_hip_runtime()
    block = int(args.block)

    lib.hipengine_pbm_max_grid.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.hipengine_pbm_max_grid.restype = ctypes.c_int
    max_grid = ctypes.c_int(0); cus = ctypes.c_int(0)
    err = lib.hipengine_pbm_max_grid(block, ctypes.byref(max_grid), ctypes.byref(cus))
    if err != 0:
        raise RuntimeError(f"hipengine_pbm_max_grid failed: hip err {err}")
    coop_grid = int(max_grid.value)  # cooperative launch ceiling (resident blocks)

    for name in ("hipengine_pbm_run_nlaunch", "hipengine_pbm_run_persistent"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                       ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p]
        fn.restype = ctypes.c_int

    def bench(fn, buf_ptr, n, n_stages, grid_x, reps):
        ms = ctypes.c_float(0.0)
        # warmup
        getattr(lib, fn)(ctypes.c_void_p(buf_ptr), n, n_stages, block, grid_x, 1.0000001, 1e-9,
                         ctypes.byref(ms), None)
        out = []
        for _ in range(reps):
            r = getattr(lib, fn)(ctypes.c_void_p(buf_ptr), n, n_stages, block, grid_x, 1.0000001, 1e-9,
                                 ctypes.byref(ms), None)
            if r != 0:
                return None, r
            out.append(float(ms.value))
        return out, 0

    results = []
    bytes_per_elem = 4  # float32
    for mb in [float(x) for x in str(args.stage_mb).split(",") if x.strip()]:
        n = int(mb * 1024 * 1024 / bytes_per_elem)
        buf = malloc(n * bytes_per_elem, runtime=rt)
        try:
            for n_stages in [int(x) for x in str(args.stages).split(",") if x.strip()]:
                # N-launch uses the SAME grid as cooperative for fairness.
                nl, e1 = bench("hipengine_pbm_run_nlaunch", buf.ptr, n, n_stages, coop_grid, int(args.reps))
                pl, e2 = bench("hipengine_pbm_run_persistent", buf.ptr, n, n_stages, coop_grid, int(args.reps))
                row = {"stage_mb": mb, "n": n, "n_stages": n_stages, "grid": coop_grid, "block": block}
                if nl is None or pl is None:
                    row["error"] = {"nlaunch_err": e1, "persistent_err": e2}
                    results.append(row); continue
                nl_ms = statistics.median(nl); pl_ms = statistics.median(pl)
                # rw bytes/stage = 2 (read+write) * n * 4
                rw_gb = 2 * n * bytes_per_elem / 1e9
                row.update({
                    "nlaunch_ms": round(nl_ms, 4), "persistent_ms": round(pl_ms, 4),
                    "nlaunch_us_per_stage": round(nl_ms * 1000 / n_stages, 3),
                    "persistent_us_per_stage": round(pl_ms * 1000 / n_stages, 3),
                    "per_stage_saving_us": round((nl_ms - pl_ms) * 1000 / n_stages, 3),
                    "speedup": round(nl_ms / pl_ms, 3) if pl_ms > 0 else None,
                    "nlaunch_GBps": round(rw_gb * n_stages / (nl_ms / 1000), 1),
                    "persistent_GBps": round(rw_gb * n_stages / (pl_ms / 1000), 1),
                })
                results.append(row)
        finally:
            free(buf, runtime=rt)

    # AR-faithful distinct-slice: big buffer, each stage reads a fresh HBM slice.
    distinct = []
    if float(args.distinct_total_mb) > 0:
        fnd = lib.hipengine_pbm_run_distinct
        fnd.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int,
                        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p]
        fnd.restype = ctypes.c_int
        total_n = int(float(args.distinct_total_mb) * 1024 * 1024 / bytes_per_elem)
        bufd = malloc(total_n * bytes_per_elem, runtime=rt)
        n_stages_d = int(str(args.distinct_stages))
        try:
            for slice_mb in [float(x) for x in str(args.distinct_slice_mb).split(chr(44)) if x.strip()]:
                slice_n = int(slice_mb * 1024 * 1024 / bytes_per_elem)

                def b2(persistent):
                    ms = ctypes.c_float(0.0)
                    fnd(ctypes.c_void_p(bufd.ptr), total_n, slice_n, n_stages_d, block, coop_grid,
                        1.0000001, 1e-9, persistent, ctypes.byref(ms), None)
                    vals = []
                    for _ in range(int(args.reps)):
                        r = fnd(ctypes.c_void_p(bufd.ptr), total_n, slice_n, n_stages_d, block, coop_grid,
                                1.0000001, 1e-9, persistent, ctypes.byref(ms), None)
                        if r != 0:
                            return None
                        vals.append(float(ms.value))
                    return statistics.median(vals)

                nl_ms = b2(0); pl_ms = b2(1)
                rw_gb = 2 * slice_n * bytes_per_elem / 1e9
                distinct.append({
                    "slice_mb": slice_mb, "slice_n": slice_n, "n_stages": n_stages_d,
                    "total_mb": float(args.distinct_total_mb), "grid": coop_grid,
                    "nlaunch_us_per_stage": round(nl_ms * 1000 / n_stages_d, 3) if nl_ms else None,
                    "persistent_us_per_stage": round(pl_ms * 1000 / n_stages_d, 3) if pl_ms else None,
                    "speedup": round(nl_ms / pl_ms, 3) if (nl_ms and pl_ms) else None,
                    "nlaunch_GBps": round(rw_gb * n_stages_d / (nl_ms / 1000), 1) if nl_ms else None,
                    "persistent_GBps": round(rw_gb * n_stages_d / (pl_ms / 1000), 1) if pl_ms else None,
                })
        finally:
            free(bufd, runtime=rt)

    return {"status": "passed", "kind": "diagnostic-persistent-barrier-microbench",
            "coop_grid_max_blocks": coop_grid, "device_cus": int(cus.value),
            "block": block, "reps": int(args.reps), "results": results, "distinct": distinct}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage-mb", default="2,6,12", help="working-set MB per stage (read+write)")
    p.add_argument("--stages", default="40,920", help="stage counts (40=per-layer, 920=per-token AR)")
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--distinct-total-mb", default="0", help="big buffer MB for AR-faithful distinct-slice mode (0=skip)")
    p.add_argument("--distinct-slice-mb", default="6", help="comma-separated per-stage slice MB for distinct mode")
    p.add_argument("--distinct-stages", default="160", help="stage count for distinct mode")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    res = run(args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"coop_grid={res['coop_grid_max_blocks']} CUs={res['device_cus']} block={res['block']}")
    print(f"{'stage_mb':>8}{'stages':>7}{'nl_us/st':>9}{'pers_us/st':>11}{'save_us':>8}{'speedup':>8}{'nl_GBps':>8}{'pe_GBps':>8}")
    for r in res["results"]:
        if "error" in r:
            print(f"{r['stage_mb']:>8}{r['n_stages']:>7}  ERROR {r['error']}")
        else:
            print(f"{r['stage_mb']:>8}{r['n_stages']:>7}{r['nlaunch_us_per_stage']:>9}{r['persistent_us_per_stage']:>11}{r['per_stage_saving_us']:>8}{r['speedup']:>8}{r['nlaunch_GBps']:>8}{r['persistent_GBps']:>8}")
    if res.get("distinct"):
        print("-- AR-faithful distinct fresh-HBM-slice per stage --")
        print(f"{'slice_mb':>8}{'stages':>7}{'nl_us/st':>9}{'pers_us/st':>11}{'speedup':>8}{'nl_GBps':>8}{'pe_GBps':>8}")
        for r in res["distinct"]:
            print(f"{r['slice_mb']:>8}{r['n_stages']:>7}{r['nlaunch_us_per_stage']:>9}{r['persistent_us_per_stage']:>11}{r['speedup']:>8}{r['nlaunch_GBps']:>8}{r['persistent_GBps']:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
