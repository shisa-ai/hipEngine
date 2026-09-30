#!/usr/bin/env python3
"""D7 screen: does column-tiling the weighted accumulate cut its 17.8 us?

The decode launch of `gemma4_moe_weighted_accumulate_bf16` is one block per
token -- at `tokens == 1` a single 256-thread block doing 8 lanes x 2816
columns of dependent loads, measured at 17.8 us/launch against a ~2 us floor
for the neighbouring single-block kernels (lane_to_row 1.9, gather 2.5, gelu
2.0). The hypothesis is latency, not work: splitting the column dimension
across `gridDim.y` gives the memory pipeline parallel blocks to hide behind.

The tiling is scheduling-only -- each column's slot-order sum stays inside one
thread -- and
`test_weighted_accumulate_is_bitwise_grid_independent` proves the outputs
bitwise equal across tile counts against the slot-order reference. This screen
measures the time and re-checks the bits in-process at the production
geometry (hidden 2816, top_k 8), plus a tokens=8 form for the prefill-shaped
launch. The gate is absolute: the D5 pack kernel taught that a new launch shape
can carry an unexplained fixed cost, so the saving must be a real reduction,
not a reshuffled overhead.

Run: python3 scripts/gemma4_accumulate_grid_screen.py --gpu 1
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=2816)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "2026-09-30-gemma4-accumulate-tile-screen.json",
    )
    args = parser.parse_args()

    os.environ.setdefault("HIP_VISIBLE_DEVICES", str(args.gpu))
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_moe import (
        build_gemma4_moe,
        gemma4_moe_lane_to_row_i32,
        gemma4_moe_weighted_accumulate_bf16,
    )

    rt = get_hip_runtime()
    build_gemma4_moe(load=True)
    hidden, top_k = args.hidden, args.top_k
    rng = np.random.default_rng(78)

    _buffers = []

    def dev(array) -> int:
        host = np.ascontiguousarray(array)
        buffer = malloc(host.nbytes)
        copy_host_to_device(buffer, host_array_ptr(host), host.nbytes)
        _buffers.append(buffer)
        return buffer.ptr

    def f32_to_bf16_u16(arr) -> "np.ndarray":
        bits = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        return (rounded >> 16).astype(np.uint16)

    def run(tokens: int, col_tiles: int) -> None:
        p = prepared[tokens]
        gemma4_moe_weighted_accumulate_bf16(
            p["expert"], p["l2r"], weights_ptr, p["out"],
            tokens, hidden, top_k, col_tiles=col_tiles,
        )

    # One geometry set per token count; buffers are reused across arms.
    prepared = {}
    for tokens in (1, 8):
        lanes = tokens * top_k
        sorted_lanes = rng.permutation(lanes).astype(np.int64)
        weights = rng.random(lanes).astype(np.float32)
        expert = rng.standard_normal((lanes, hidden)).astype(np.float32)
        lanes_ptr = dev(sorted_lanes)
        l2r = dev(np.zeros((lanes,), dtype=np.int32))
        gemma4_moe_lane_to_row_i32(lanes_ptr, l2r, lanes)
        prepared[tokens] = {
            "expert": dev(f32_to_bf16_u16(expert)),
            "l2r": l2r,
            "out": dev(np.zeros((tokens, hidden), dtype=np.uint16)),
            "expected": None,
        }
    weights_ptr = dev(rng.random(8 * args.top_k + 64).astype(np.float32))

    def bench(fn) -> float:
        for _ in range(args.warmup):
            fn()
        rt.device_synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            fn()
        rt.device_synchronize()
        return (time.perf_counter() - t0) / args.iters * 1e6  # us/call

    arms = {}
    for tokens in (1, 8):
        arms[f"tiles1_t{tokens}"] = (
            lambda t=tokens: run(t, 1)
        )
        arms[f"auto_t{tokens}"] = (
            lambda t=tokens: run(t, 0)
        )
        arms[f"tiles11_t{tokens}"] = (
            lambda t=tokens: run(t, 11)
        )

    results = {name: round(bench(fn), 3) for name, fn in arms.items()}

    # Bitwise re-check at the screen geometry: tile forms against tiles=1.
    def read(tokens) -> "np.ndarray":
        p = prepared[tokens]
        buf = np.empty((tokens, hidden), dtype=np.uint16)
        copy_device_to_host(
            host_array_ptr(buf), DeviceBuffer(ptr=p["out"], nbytes=buf.nbytes), buf.nbytes
        )
        return buf

    bit_equal = {}
    for tokens in (1, 8):
        run(tokens, 1)
        base = np.array(read(tokens))
        equal = True
        for tiles in (2, 7, 0, 11):
            run(tokens, tiles)
            if not np.array_equal(np.array(read(tokens)), base):
                equal = False
        bit_equal[str(tokens)] = bool(equal)

    delta_t1 = results["tiles1_t1"] - results["auto_t1"]
    probe_overhead = results["auto_t8"] - results["tiles1_t8"]
    wins = delta_t1 > 1.0  # at least 1 us saved on the decode launch
    bits_ok = all(bit_equal.values())
    verdict = "wire" if (wins and bits_ok) else "record"

    artifact = {
        "schema": "gemma4-accumulate-tile-screen/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": f"python3 scripts/gemma4_accumulate_grid_screen.py --gpu {args.gpu} --hidden {hidden} --iters {args.iters}",
        "git_commit": _git_head(),
        "arch": platform.machine(),
        "gpu_env": {"HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "")},
        "geometry": {"hidden": hidden, "top_k": top_k, "iters": args.iters, "warmup": args.warmup},
        "arms_us_per_call": results,
        "bit_equal": bit_equal,
        "delta_us_decode": round(delta_t1, 3),
        "tokens8_auto_minus_tiles1_us": round(probe_overhead, 3),
        "verdict": verdict,
        "evidence_note": (
            "tiles1 = the pre-change one-block-per-token geometry, auto = the new "
            "shipped default (hidden/256 tiles), tiles11 = explicit tile count. "
            "Bitwise equality is re-checked in-process per token count; the "
            "standing proof is test_weighted_accumulate_is_bitwise_grid_independent. "
            "The shipped-path claim is the same-lane census A/B, not this projection."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))

    for buffer in _buffers:
        free(buffer)
    return 0 if verdict == "wire" else 2


if __name__ == "__main__":
    sys.exit(main())