"""Serial two-branch projections against one two-row call, at the AR's own shapes.

The AR decode runs two CFG branches over the same weights. `rowtile2` issues the
weight stream once for both rows instead of once per branch. This measures that
swap on the production shapes from `hipengine/runtime/yue2_ar.py` (hidden 2048, q/o
2048, k/v 1024, gate/up 8192+4096, down 2048, LM head 184704x2048) and reports
effective weight bandwidth, which is what decides the memory-bound shapes.

Each branch keeps its own row: the benchmark never shares KV spans or positions
between branches, it only shares the weight read.

    python3 scripts/yue2_ar_rowtile_bench.py [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hipengine.core.memory import (  # noqa: E402
    copy_host_array_to_device,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.linear import dense_gemv  # noqa: E402
from hipengine.runtime.yue2_nar import to_bf16_bits  # noqa: E402

HIDDEN = 2048
SHAPES = [
    ("q", HIDDEN, HIDDEN),
    ("k", HIDDEN, 1024),
    ("v", HIDDEN, 1024),
    ("o", HIDDEN, HIDDEN),
    ("lm_head", HIDDEN, 184704),
]


def _upload(array: np.ndarray):
    host = np.ascontiguousarray(array)
    buffer = malloc(max(host.nbytes, 8))
    copy_host_array_to_device(buffer, host)
    return buffer


def _timed(call, *, repeats: int, warmup: int) -> float:
    """Milliseconds per call, best of `repeats` after `warmup`."""

    for _ in range(warmup):
        call()
    dense_gemv.get_hip_runtime().device_synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        dense_gemv.get_hip_runtime().device_synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return min(samples), statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    rng = np.random.default_rng(7)
    x = to_bf16_bits(rng.standard_normal((2, HIDDEN)).astype(np.float32) * 0.5)
    x_buf = _upload(x)
    out_buf = _upload(np.zeros((2, 184704), dtype=np.float32))

    rows = []
    print(f"{'shape':<10} {'serial ms':>10} {'two-row ms':>11} {'speedup':>8} "
          f"{'serial GB/s':>12} {'two-row GB/s':>13}")
    for name, in_features, out_features in SHAPES:
        weight = to_bf16_bits(
            rng.standard_normal((out_features, in_features)).astype(np.float32) * 0.05
        )
        w_buf = _upload(weight)
        weight_bytes = weight.nbytes
        out = out_buf.ptr

        def serial():
            dense_gemv.dense_gemv_bf16_f32_out(
                x_buf.ptr, w_buf.ptr, out, 1, in_features, out_features)
            dense_gemv.dense_gemv_bf16_f32_out(
                x_buf.ptr + HIDDEN * 2, w_buf.ptr, out + out_features * 4,
                1, in_features, out_features)

        def paired():
            dense_gemv.dense_gemv_bf16_f32_out_rowtile2(
                x_buf.ptr, w_buf.ptr, out, 2, in_features, out_features)

        serial_ms, serial_med = _timed(serial, repeats=args.repeats, warmup=args.warmup)
        paired_ms, paired_med = _timed(paired, repeats=args.repeats, warmup=args.warmup)
        rows.append({
            "shape": name,
            "in_features": in_features,
            "out_features": out_features,
            "weight_mib": weight_bytes / 2**20,
            "serial_ms": serial_ms,
            "serial_median_ms": serial_med,
            "two_row_ms": paired_ms,
            "two_row_median_ms": paired_med,
            "speedup": serial_ms / paired_ms,
            "serial_gbs": weight_bytes / (serial_ms * 1e-3) / 1e9,
            "two_row_gbs": weight_bytes / (paired_ms * 1e-3) / 1e9,
        })
        print(f"{name:<10} {serial_ms:>10.3f} {paired_ms:>11.3f} {serial_ms/paired_ms:>7.2f}x "
              f"{rows[-1]['serial_gbs']:>12.1f} {rows[-1]['two_row_gbs']:>13.1f}")

    per_step = sum(r["serial_ms"] for r in rows) * 24
    per_step_paired = sum(r["two_row_ms"] for r in rows) * 24
    print()
    print(f"projection families above, per step (24 layers, 2 branches): "
          f"{per_step:.2f} ms serial against {per_step_paired:.2f} ms two-row "
          f"({per_step - per_step_paired:.2f} ms per step saved)")
    print(f"head alone, per step (2 branches): "
          f"{[r for r in rows if r['shape'] == 'lm_head'][0]['serial_ms']:.2f} ms serial against "
          f"{[r for r in rows if r['shape'] == 'lm_head'][0]['two_row_ms']:.2f} ms two-row")

    if args.json:
        payload = {
            "benchmark": "yue2_ar_rowtile_bench",
            "date": time.strftime("%Y-%m-%d"),
            "host": Path("/etc/hostname").read_text().strip(),
            "hidden_size": HIDDEN,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "rows": rows,
            "per_step_ms": {"serial": per_step, "two_row": per_step_paired},
            "notes": [
                "Two rows per call: the two CFG branches, each with its own row of "
                "activations and its own output row. No KV span or position sharing.",
                "Timed with a synchronize per call, so these are per-call latencies, "
                "not a pipeline rate.",
                "rowtile2 is bit-identical per row to the single-row kernel "
                "(tests/test_unit_yue2_ar_gemv_rowtile2.py).",
            ],
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
