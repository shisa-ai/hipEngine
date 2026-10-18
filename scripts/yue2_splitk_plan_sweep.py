#!/usr/bin/env python3
"""Sweep the split-K decode attention plan on this device, kernel-only.

``plan_split`` picks a split count from the context and the device's compute-unit
count. This times the kernel across a range of split counts at one decode context, so
the policy's choice can be compared with the neighbours it did not pick, and prints
the grid each one launches. The plan's own choice is marked in the output.

    python3 scripts/yue2_splitk_plan_sweep.py --context 998
    python3 scripts/yue2_splitk_plan_sweep.py --context 4098 --iters 200 --json out.json

Kernel-only timings are per call: a decode step runs one call per layer per branch,
so multiply by that count to compare against a step-time measurement.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hipengine.core.device import Device
from hipengine.core.hip import device_compute_units
from hipengine.core.memory import (
    copy_host_array_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.vibevoice import attention_spans_splitk
from hipengine.kvcache.spans import KVLiveSpans

Q_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128


def _bf16_bits(values) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((raw + 0x7FFF + ((raw >> 16) & 1)) >> 16).astype(np.uint16)


def _upload(array):
    host = np.ascontiguousarray(array)
    buffer = malloc(max(host.nbytes, 8))
    copy_host_array_to_device(buffer, host)
    return buffer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=998)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--splits", default="")
    parser.add_argument("--json", default="", help="write the sweep to this path")
    args = parser.parse_args()

    context = args.context
    capacity = min(context + 8, 16000)
    rng = np.random.default_rng(0)
    query = rng.standard_normal((1, Q_HEADS, HEAD_DIM)).astype(np.float32)
    keys = _bf16_bits(rng.standard_normal((capacity, KV_HEADS, HEAD_DIM)))
    values = _bf16_bits(rng.standard_normal((capacity, KV_HEADS, HEAD_DIM)))
    scale = 1.0 / float(np.sqrt(HEAD_DIM))

    query_buf = _upload(query)
    key_buf = _upload(keys)
    value_buf = _upload(values)
    out_buf = _upload(np.zeros((1, Q_HEADS, HEAD_DIM), np.float32))
    workspace = malloc(
        attention_spans_splitk.workspace_floats(1, Q_HEADS, HEAD_DIM, attention_spans_splitk.MAX_SPLITS)
        * 4
    )
    base = _upload(np.arange(capacity, dtype=np.int32))
    counts = _upload(np.array([context], dtype=np.int64))
    positions = _upload(np.arange(capacity, dtype=np.int64))
    evicted = _upload(np.zeros(capacity, dtype=np.bool_))
    row_position = _upload(np.array([context - 1], dtype=np.int64))
    spans = KVLiveSpans(
        Tensor.from_handle(base.ptr, (capacity,), "int32", Device("hip")),
        Tensor.from_handle(counts.ptr, (1,), "int64", Device("hip")),
        capacity,
        Tensor.from_handle(positions.ptr, (capacity,), "int64", Device("hip")),
        Tensor.from_handle(evicted.ptr, (capacity,), "bool", Device("hip")),
        "bf16",
        row_positions=Tensor.from_handle(row_position.ptr, (1,), "int64", Device("hip")),
    )

    units = device_compute_units()
    policy = attention_spans_splitk.plan_split(context, KV_HEADS, compute_units=units)
    print(f"device compute units: {units}")
    print(f"policy plan: splits={policy.splits} chunk={policy.chunk} "
          f"blocks={policy.splits * KV_HEADS}")

    if args.splits:
        sweep = [int(value) for value in args.splits.split(",")]
    else:
        sweep = sorted({8, 12, 16, 24, 32, 40, 48, 56, 64, policy.splits})

    # A plan whose chunk exceeds MAX_CHUNK cannot cover this context, and the wrapper
    # refuses it; skip those instead of tripping the guard in the sweep.
    floor_splits = -(-context // attention_spans_splitk.MAX_CHUNK)
    sweep = [splits for splits in sweep if splits >= floor_splits]
    if not sweep:
        print(f"no plan covers {context} tokens inside MAX_CHUNK")
        return 1

    def run(plan):
        attention_spans_splitk.vv_attention_spans_splitk(
            query_buf.ptr, key_buf.ptr, value_buf.ptr, out_buf.ptr, workspace.ptr, spans,
            1, Q_HEADS, KV_HEADS, HEAD_DIM, scale, context, plan=plan,
        )

    print(f"\n{'splits':>6} {'chunk':>5} {'blocks':>6} {'blocks/CU':>9} {'ms/call':>8}")
    timings = {}
    for splits in sweep:
        plan = attention_spans_splitk.SplitPlan(splits=splits, chunk=-(-context // splits))
        for _ in range(10):
            run(plan)
        ctypes.CDLL("libamdhip64.so").hipDeviceSynchronize()
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            for _ in range(args.iters):
                run(plan)
            ctypes.CDLL("libamdhip64.so").hipDeviceSynchronize()
            best = min(best, (time.perf_counter() - start) / args.iters * 1e3)
        timings[splits] = best
        marker = "  <-- policy" if splits == policy.splits else ""
        print(f"{splits:>6} {plan.chunk:>5} {splits * KV_HEADS:>6} "
              f"{splits * KV_HEADS / units:>9.2f} {best:>8.3f}{marker}")

    floor = min(timings.values())
    if policy.splits in timings:
        print(f"\npolicy {timings[policy.splits]:.3f} ms, best {floor:.3f} ms "
              f"(+{(timings[policy.splits] / floor - 1) * 100:.1f}% over best)")
    if args.json:
        payload = {
            "device": {"backend_compute_units": units},
            "context": context,
            "iterations": args.iters,
            "policy": {"splits": policy.splits, "chunk": policy.chunk},
            "sweep": [
                {
                    "splits": splits,
                    "chunk": -(-context // splits),
                    "blocks": splits * KV_HEADS,
                    "ms_per_call": round(timings[splits], 4),
                }
                for splits in sweep
                if splits in timings
            ],
            "floor_ms_per_call": round(floor, 4),
        }
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json}")

    for buffer in (query_buf, key_buf, value_buf, out_buf, workspace, base, counts,
                   positions, evicted, row_position):
        free(buffer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
