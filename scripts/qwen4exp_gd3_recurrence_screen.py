"""GD-3 screen: is the GDN prefill recurrence latency-bound or traffic-bound?

The ledger row's first step is "measure stall cycles in our register-state
kernel first; if latency-bound, add the windowed prefetch while keeping our
per-row reduction order".  A direct counter read needs PMC access, which is not
always available, so this screen establishes the same thing from the outside:

* absolute wall time per 1K tokens (the ledger quotes gufo ~65 ms against
  "our 205" for the register-state route), and
* the distance from the two hard floors - bytes moved and FLOPs issued.

If the kernel sits orders of magnitude above the traffic floor it is not
traffic-bound, and the only remaining explanations are latency, occupancy, or
issue.  That is what makes a prefetch (a pure scheduling change, exact by
construction) the right experiment rather than a wider tile.

    python3 scripts/qwen4exp_gd3_recurrence_screen.py --tokens 1024 --reps 5
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [str(REPO_ROOT), *(e for e in sys.path if e != str(REPO_ROOT))]

import numpy as np  # noqa: E402

from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.core.memory import (  # noqa: E402
    copy_host_to_device,
    free,
    host_array_ptr,
)
from hipengine.kernels.hip_gfx1100.linear_attn import qwen4_exp_gdn as gdn  # noqa: E402

K_HEADS = 16
V_HEADS = 48
DIM = 128
CONV_WIDTH = 2 * K_HEADS * DIM + V_HEADS * DIM
CORE_WIDTH = V_HEADS * DIM

# W7900 (gfx1100) single-precision peaks, for the floor comparison.
PEAK_F32_TFLOPS = 61.0
PEAK_BW_GBPS = 221.0

# The recurrence routes worth comparing.  ``register_state`` and ``serial_lds``
# are the two the ledger row names; ``columnwarps`` and ``tiled16`` are what the
# gfx1151 production profile actually selects for GDN layers.
ROUTES = {
    "register_state": gdn.qwen4_exp_gdn_register_prefill_f32,
    "serial_lds": gdn.qwen4_exp_gdn_prefill_f32,
    "columnwarps": gdn.qwen4_exp_gdn_prefill_columnwarps_f32,
    "tiled16": gdn.qwen4_exp_gdn_prefill_tiled16_f32,
}


def _alloc(shape, dtype, runtime, allocations):
    from hipengine.core.memory import malloc

    device = malloc(int(np.prod(shape)) * np.dtype(dtype).itemsize, runtime=runtime)
    allocations.append(device)
    return device


def _upload(array, runtime, allocations):
    ptr = _alloc(array.shape, array.dtype, runtime, allocations)
    copy_host_to_device(ptr, host_array_ptr(np.ascontiguousarray(array)), runtime=runtime)
    return ptr


class Fixture:
    """Same shape and launch pattern as tests/test_gpu_qwen4exp_gdn_register.py."""

    def __init__(self, tokens):
        self.runtime = get_hip_runtime()
        self.allocations = []
        self.tokens = tokens
        rng = np.random.default_rng(8305)
        q = rng.normal(0, 0.05, (tokens, K_HEADS, DIM)).astype(np.float32)
        k = rng.normal(0, 0.05, (tokens, K_HEADS, DIM)).astype(np.float32)
        q[:, 0] = 0
        k[:, 0] = 0
        v = rng.normal(0, 0.05, (tokens, V_HEADS, DIM)).astype(np.float32)
        conv = np.concatenate(
            (q.reshape(tokens, -1), k.reshape(tokens, -1), v.reshape(tokens, -1)), axis=1
        )
        self.gate = rng.normal(0, 0.5, (tokens, V_HEADS, DIM)).astype(np.float32)
        self.alpha = rng.normal(-0.2, 0.1, (tokens, V_HEADS)).astype(np.float32)
        self.beta = rng.normal(0, 0.2, (tokens, V_HEADS)).astype(np.float32)
        self.dt = rng.normal(-1, 0.1, V_HEADS).astype(np.float32)
        self.a = -np.exp(rng.normal(-0.5, 0.1, V_HEADS)).astype(np.float32)
        self.norm = rng.normal(1, 0.05, DIM).astype(np.float32)
        self.state = rng.normal(0, 0.01, (V_HEADS, DIM, DIM)).astype(np.float32)
        self.inputs = [
            _upload(x, self.runtime, self.allocations)
            for x in (conv, self.gate, self.alpha, self.beta, self.dt, self.a, self.norm)
        ]
        self.states = [_upload(self.state, self.runtime, self.allocations) for _ in range(2)]
        self.outputs = [
            _alloc((tokens, V_HEADS, DIM), np.float32, self.runtime, self.allocations)
            for _ in range(2)
        ]
        self.library = gdn.build_qwen4_exp_gdn(load=True)

    def launch(self, route: str):
        idx = 0 if route == "register_state" else 1
        copy_host_to_device(
            self.states[idx], host_array_ptr(self.state), runtime=self.runtime
        )
        fn = ROUTES[route]
        fn(
            *[x.ptr for x in self.inputs],
            self.states[idx].ptr,
            self.outputs[idx].ptr,
            self.tokens,
            K_HEADS,
            V_HEADS,
            DIM,
            DIM,
            library=self.library,
            runtime=self.runtime,
        )

    def close(self):
        for p in reversed(self.allocations):
            free(p, runtime=self.runtime)


def bytes_moved(tokens: int) -> int:
    """Lower bound on unique bytes this recurrence must touch per pass."""
    conv = tokens * CONV_WIDTH * 4
    gate = tokens * CORE_WIDTH * 4
    scalars = tokens * V_HEADS * 2 * 4  # alpha, beta
    output = tokens * CORE_WIDTH * 4
    state = V_HEADS * DIM * DIM * 4 * 2  # read once, written once
    return conv + gate + scalars + output + state


def flops_issued(tokens: int) -> int:
    """Per token, per v-head, per lane: 2 FMAs over DIM columns, twice."""
    return tokens * V_HEADS * DIM * DIM * 2 * 2


def timed(fixture, route, reps, warmup=1):
    for _ in range(warmup):
        fixture.launch(route)
        fixture.runtime.device_synchronize()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fixture.launch(route)
        fixture.runtime.device_synchronize()
        samples.append(time.perf_counter() - start)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1024])
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        print("HIP unavailable", file=sys.stderr)
        return 2

    payload = {"schema": 1, "kind": "qwen4exp-gd3-recurrence-screen", "results": []}
    for tokens in args.tokens:
        fixture = Fixture(tokens)
        try:
            b = bytes_moved(tokens)
            f = flops_issued(tokens)
            bw_floor_ms = b / (PEAK_BW_GBPS * 1e9) * 1e3
            flop_floor_ms = f / (PEAK_F32_TFLOPS * 1e12) * 1e3
            row = {
                "tokens": tokens,
                "bytes_moved": b,
                "flops": f,
                "bandwidth_floor_ms": bw_floor_ms,
                "flop_floor_ms": flop_floor_ms,
                "arms": {},
            }
            for name, fn in ROUTES.items():
                samples = timed(fixture, name, args.reps)
                median = statistics.median(samples)  # seconds
                achieved_bw = b / median / 1e9
                achieved_tf = f / median / 1e12
                row["arms"][name] = {
                    "median_ms": median * 1e3,
                    "min_ms": min(samples) * 1e3,
                    "max_ms": max(samples) * 1e3,
                    "samples_ms": [s * 1e3 for s in samples],
                    "achieved_gbps": achieved_bw,
                    "achieved_tflops": achieved_tf,
                    "x_above_bandwidth_floor": (median * 1e3) / bw_floor_ms,
                    "pct_of_bw_roof": 100.0 * achieved_bw / PEAK_BW_GBPS,
                    "pct_of_f32_roof": 100.0 * achieved_tf / PEAK_F32_TFLOPS,
                }
                a = row["arms"][name]
                print(
                    f"tokens={tokens:<6} {name:<15} "
                    f"median={a['median_ms']:8.3f} ms  "
                    f"({a['achieved_gbps']:6.1f} GB/s = {a['pct_of_bw_roof']:5.1f}% of roof, "
                    f"{a['achieved_tflops']:6.3f} TFLOP/s = {a['pct_of_f32_roof']:5.1f}% of roof)  "
                    f"x{a['x_above_bandwidth_floor']:.0f} above traffic floor"
                )
            print(
                f"  floors: traffic {bw_floor_ms:.4f} ms ({b / 1e6:.1f} MB), "
                f"flops {flop_floor_ms:.4f} ms ({f / 1e9:.2f} GFLOP)"
            )
            payload["results"].append(row)
        finally:
            fixture.close()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
