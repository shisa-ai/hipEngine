#!/usr/bin/env python3
"""RT-4 follow-up: is the KV reset's 220 GB/s the hardware limit or the memset?

RT-4's attribution found `Qwen4ExpGGUFResidentModelRunner.reset` clears the full K/V
capacity at an effective **~220 GB/s** (6 GiB in 29.182 ms at 128K capacity). That is low
for a pure memset on this part, and it matters because it changes the cheapest fix:

  - If 220 GB/s is the hardware limit, the only lever is clearing less (RT-4's proposed
    contract change, with its poison/short-after-long/rollback/c2 correctness burden).
  - If the hardware can do substantially more, the cheaper fix is to make the memset
    itself faster - same effect, **no contract change, no correctness burden**.

This script measures the achievable device memset rate at several sizes and compares it
against RT-4's recorded figure. It loads no model.

Diagnostic only. No runtime change, no promotion.
"""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# RT-4's recorded reset figures, so the comparison is against the published numbers
# rather than a re-derivation of them.
RT4_BYTES_PER_TOKEN = 24_576
RT4_CAPACITY_TOKENS = 131_072


def load_hip():
    lib = ctypes.CDLL("libamdhip64.so")
    lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.hipMalloc.restype = ctypes.c_int
    lib.hipFree.argtypes = [ctypes.c_void_p]
    lib.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    lib.hipMemset.restype = ctypes.c_int
    lib.hipDeviceSynchronize.restype = ctypes.c_int
    return lib


def timed_memset(lib, ptr, nbytes, *, repeats, warmups):
    for _ in range(warmups):
        lib.hipMemset(ptr, 0, nbytes)
    lib.hipDeviceSynchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        rc = lib.hipMemset(ptr, 0, nbytes)
        lib.hipDeviceSynchronize()
        samples.append(time.perf_counter() - start)
        if rc != 0:
            raise RuntimeError(f"hipMemset failed rc={rc}")
    median = statistics.median(samples)
    return {
        "nbytes": nbytes,
        "gib": nbytes / 2**30,
        "median_ms": median * 1e3,
        "gbs": nbytes / median / 1e9,
        "samples_ms": [s * 1e3 for s in samples],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lib = load_hip()
    report: dict = {
        "schema": 1,
        "kind": "qwen4exp_kv_reset_memset_bandwidth",
        "performance_claim": False,
        "promotion_claim": False,
        "purpose": (
            "Determine whether RT-4's ~220 GB/s reset rate is the hardware memset limit "
            "or a property of the memset call, which decides between a contract change "
            "and a cheaper memset fix."
        ),
        "command": sys.argv,
        "rt4_reference": {
            "bytes_per_token": RT4_BYTES_PER_TOKEN,
            "capacity_tokens": RT4_CAPACITY_TOKENS,
            "bytes_cleared_per_reset": RT4_BYTES_PER_TOKEN * RT4_CAPACITY_TOKENS * 2,
            "recorded_reset_ms": 29.182,
            "recorded_effective_gbs": (
                RT4_BYTES_PER_TOKEN * RT4_CAPACITY_TOKENS * 2 / (29.182 / 1e3) / 1e9),
        },
        "cases": [],
    }

    # Ladder around the two sizes RT-4 measured, plus a large one to expose any
    # size-dependence in the achieved rate.
    ladder = [1 << 20, 64 << 20, 768 << 20, 6 * (1 << 30), 12 * (1 << 30)]
    allocated = []
    try:
        for nbytes in ladder:
            ptr = ctypes.c_void_p()
            rc = lib.hipMalloc(ctypes.byref(ptr), nbytes)
            if rc != 0:
                report["cases"].append({"nbytes": nbytes, "skipped": f"hipMalloc rc={rc}"})
                continue
            allocated.append(ptr)
            entry = timed_memset(
                lib, ptr, nbytes, repeats=args.repeats, warmups=args.warmups)
            entry["ratio_vs_rt4"] = entry["gbs"] / report["rt4_reference"][
                "recorded_effective_gbs"]
            report["cases"].append(entry)
            print(json.dumps(entry))
    finally:
        for ptr in allocated:
            lib.hipFree(ptr)

    good = [c for c in report["cases"] if "gbs" in c]
    if good:
        # Compare like with like: the 6 GiB case is the same volume RT-4 clears at 128K.
        six = [c for c in good if abs(c["gib"] - 6) < 0.5]
        report["best_gbs"] = max(c["gbs"] for c in good)
        report["six_gib_gbs"] = six[0]["gbs"] if six else None
        ref = report["rt4_reference"]["recorded_effective_gbs"]
        report["verdict"] = {
            "rt4_effective_gbs": ref,
            "best_measured_gbs": report["best_gbs"],
            "headroom_factor": report["best_gbs"] / ref,
            "is_hardware_limit": report["best_gbs"] < ref * 1.3,
        }
        if six:
            report["verdict"]["six_gib_ms_if_at_best"] = (
                six[0]["nbytes"] / (report["best_gbs"] * 1e9) * 1e3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report.get("verdict", {}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
