#!/usr/bin/env python3
"""Sweep T16 dense-prefill rows to expose the 256-row shared-B tile cost.

Why this exists: `rocprofv3` shows the W7900 AR first-token wall is owned by
`gguf_q4_t16_dense_wmma_prefill_shared_b_bf16_kernel` (240 launches, 0.623 ms
median, 58% of the 241.8 ms busy wall at rows 35, zero pack8 launches;
`benchmarks/results/2026-08-30-w7900-q4km-ar-prefill-kernel-owner-trace.json`).
That kernel is launched on a `Q4_DENSE_TILE_ROWS = 4*4*16 = 256` row tile, and
the repo already documents the consequence for tiny cycles in
`hipengine/kernels/hip_gfx1100/__init__.py`: ragged 2-5-row verify cycles
"ride the shared-B 256-row padded tile at a measured ~5.1x cycle cost on
W7900". If the same flat 256-row cost applies at prefill rows 33-67, then every
published C1-C8 prompt pays 256-row work for 35-67 rows, and the cheap
registered siblings (single-wave, small-M, rows6 rowtile) are the fix.

This harness measures the claim directly, per row, on the real repacked T16
payload, for the FFN gate/up geometry (in_features=5120, out_features=17408)
that owns the wall. It does not change dispatch: it launches registered leaves
directly and reports cost curves. A win here is a *candidate*, and promoting it
still requires the declared correctness gates plus a same-protocol C1 packet.

Arms (all registered leaves, same tiles/x/out buffers):
  shared_b        the shipping owner, 256-row tile, B staged in shared memory
  single_wave     gguf_q4_k_t16_wmma_prefill_bf16_bf16_out
  smallm          gguf_q4_k_t16_wmma_prefill_smallm_bf16_bf16_out
  rows6_rowtile   gguf_q4_k_t16_physical_c1_rowtile_gfx1100_bf16_bf16_out

Ratios are always reference/candidate, so > 1.0 means the candidate is faster.
Each row is timed with a counterbalanced forward/reverse schedule. Repeated
order pairs also reverse pair order (forward/reverse, then reverse/forward) so
a candidate must be non-regressive in both arm orders rather than benefit from
a fixed position or a best-of-two selection.

Usage:
    .venv/bin/python scripts/gguf_t16_prefill_row_sweep.py \
        --rows 6,32,33,35,36,48,64,65,96,128,192,255,256,257 --output /tmp/he-t16/sweep.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

QK_K = 256
DEFAULT_ROWS = "6,32,33,35,36,39,43,46,48,60,64,65,67,96,128,192,255,256,257"
DEFAULT_IN_FEATURES = 5120
DEFAULT_OUT_FEATURES = 17408
# Dense shapes that still fall through to the 256-row shared-B tile at rows > 6.
OTHER_SHAPES = (
    (17_408, 5_120),
    (5_120, 6_144),
    (6_144, 5_120),
    (5_120, 10_240),
    (5_120, 12_288),
    (5_120, 1_024),
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip()


def _tile16_payload(out_features: int, in_features: int, seed: int) -> np.ndarray:
    """Canonical T16 tiles built by the shipping repacker.

    Random tile-shaped bytes are not admissible: a kernel reading the wrong
    layout times something that is not the owner (an earlier pack8 harness
    measured ~18 GB/s that way), so every arm consumes
    ``repack_gguf_q4_k_tile16`` output over a valid Q4_K block image. Bit
    exactness against the raw-Q4_K CPU dequantization stays covered by
    ``tests/test_gguf_q4_k_wmma_prefill.py``.
    """

    from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16

    rng = np.random.default_rng(seed)
    blocks = in_features // QK_K
    block_bytes = 144
    raw = np.empty((1, out_features, blocks, block_bytes), dtype=np.uint8)
    raw[:, :, :, 4:] = rng.integers(
        0, 256, size=(1, out_features, blocks, block_bytes - 4), dtype=np.uint8
    )
    scales = (
        rng.uniform(0.002, 0.02, size=(1, out_features, blocks))
        .astype(np.float16)
        .view(np.uint8)
        .reshape(1, out_features, blocks, 2)
    )
    mins = (
        rng.uniform(-0.002, 0.002, size=(1, out_features, blocks))
        .astype(np.float16)
        .view(np.uint8)
        .reshape(1, out_features, blocks, 2)
    )
    raw[:, :, :, 0:2] = scales
    raw[:, :, :, 2:4] = mins
    packed = repack_gguf_q4_k_tile16(
        raw.reshape(1, out_features, blocks * block_bytes)
    )
    return np.ascontiguousarray(packed.tiles.reshape(-1))


def _activations(rows: int, in_features: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 0.35, size=(rows, in_features)).astype(np.float32)
    bits = values.view(np.uint32)
    lsb = (bits >> 16) & np.uint32(1)
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)


class _Buffers:
    """One weight upload plus per-row activation/output buffers."""

    def __init__(self, tiles: np.ndarray) -> None:
        from hipengine.core.hip import get_hip_runtime
        from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc

        self.runtime = get_hip_runtime()
        self.malloc = malloc
        self.copy_h2d = copy_host_to_device
        self.host_array_ptr = host_array_ptr
        self._keep: list[object] = []
        self.tiles_buf = malloc(tiles.nbytes)
        self.copy_h2d(self.tiles_buf, host_array_ptr(tiles), tiles.nbytes)
        self.tiles_ptr = int(self.tiles_buf.ptr)
        self.runtime.device_synchronize()

    def upload(self, host: np.ndarray) -> int:
        buf = self.malloc(host.nbytes)
        self.copy_h2d(buf, self.host_array_ptr(host), host.nbytes)
        self.runtime.device_synchronize()
        self._keep.append(buf)
        return int(buf.ptr)

    def sync(self) -> None:
        self.runtime.device_synchronize()


def _resolve_arms() -> dict[str, object]:
    from hipengine.kernels.hip_gfx1100.quant import gguf_k_t16_selected_prefill as P

    arms: dict[str, object] = {
        "shared_b": P.gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out,
        "shared_b_row64": P.gguf_q4_k_t16_wmma_prefill_shared_b_row64_bf16_bf16_out,
        "single_wave": P.gguf_q4_k_t16_wmma_prefill_bf16_bf16_out,
        "smallm": P.gguf_q4_k_t16_wmma_prefill_smallm_bf16_bf16_out,
        "c1_rowtile_dispatch": P.gguf_q4_k_t16_physical_c1_rowtile_gfx1100_bf16_bf16_out,
    }
    # The sidecar leaves that the verifier / native-batch-decode scope selects are
    # not exported from the prefill module, and the prefill dispatcher above only
    # reaches the rowtile kernel at row 6. Measure the real leaves directly, or a
    # shape's rowtile option stays unmeasured (as it was for (17408, 5120)).
    try:
        from hipengine.kernels.hip_gfx1100.quant import gguf_t16_selected_gemv as G
    except Exception:  # pragma: no cover - module always present in-tree
        return arms
    for name, symbol in (
        ("dense_rowtile", "gguf_q4_k_t16_dense_rowtile_bf16_bf16_out"),
        ("rowtile_col4", "gguf_q4_k_t16_dense_rowtile_col4_bf16_bf16_out"),
    ):
        fn = getattr(G, symbol, None) or getattr(P, symbol, None)
        if fn is not None:
            arms[name] = fn
    return arms


def _counterbalanced_schedule(
    arm_names: tuple[str, ...], *, order_pairs: int
) -> list[dict[str, object]]:
    """Return balanced forward/reverse passes, including pair-order reversal."""

    if order_pairs < 1:
        raise ValueError("order_pairs must be at least 1")
    forward = tuple(arm_names)
    reverse = tuple(reversed(arm_names))
    schedule: list[dict[str, object]] = []
    pass_index = 0
    for pair_index in range(order_pairs):
        directions = (
            (("forward", forward), ("reverse", reverse))
            if pair_index % 2 == 0
            else (("reverse", reverse), ("forward", forward))
        )
        for direction, names in directions:
            schedule.append(
                {
                    "pass_index": pass_index,
                    "pair_index": pair_index,
                    "direction": direction,
                    "arms": names,
                }
            )
            pass_index += 1
    return schedule


def _timed(
    fn: object,
    buffers: _Buffers,
    x_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    reps: int,
    warmup: int,
    *,
    capture: bool = False,
    out_buf: object = None,
) -> tuple[dict[str, float], np.ndarray | None]:
    for _ in range(warmup):
        fn(x_ptr, buffers.tiles_ptr, out_ptr, rows, in_features, out_features)
        buffers.sync()
    samples: list[float] = []
    for _ in range(reps):
        start = time.perf_counter()
        fn(x_ptr, buffers.tiles_ptr, out_ptr, rows, in_features, out_features)
        buffers.sync()
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    out = {
        "median_ms": statistics.median(samples),
        "min_ms": samples[0],
        "p90_ms": samples[-max(1, len(samples) // 10)],
    }
    # Capture after timing and return an owned copy. Keeping the snapshot on the
    # shared buffers object lets the next arm silently overwrite provenance.
    captured = (
        _read_out(buffers, out_buf, rows, out_features).copy() if capture else None
    )
    return out, captured


def _read_out(buffers: _Buffers, out_buf: object, rows: int, out_features: int) -> np.ndarray:
    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    host = np.zeros(rows * out_features, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(host), out_buf, host.nbytes)
    buffers.sync()
    return host


def _bf16_finite(bits: np.ndarray) -> bool:
    f32 = (bits.astype(np.uint32) << 16).view(np.float32)
    return bool(np.all(np.isfinite(f32)))


def _ulp_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Worst BF16 ULP distance between two outputs of the same shape."""

    if a.shape != b.shape:
        return -1
    same = bool(np.array_equal(a, b))
    af = (a.astype(np.uint32) << 16).view(np.int32).astype(np.int64)
    bf = (b.astype(np.uint32) << 16).view(np.int32).astype(np.int64)
    # Sign-magnitude BF16 mapped to a monotonic integer axis before differencing.
    a_ord = np.where(af < 0, np.int64(0x80000000) - af, af)
    b_ord = np.where(bf < 0, np.int64(0x80000000) - bf, bf)
    return 0 if same else int(np.max(np.abs(a_ord - b_ord)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", default=DEFAULT_ROWS)
    parser.add_argument("--in-features", type=int, default=DEFAULT_IN_FEATURES)
    parser.add_argument("--out-features", type=int, default=DEFAULT_OUT_FEATURES)
    parser.add_argument(
        "--all-shapes",
        action="store_true",
        help="also sweep the shapes that still fall through to shared-B",
    )
    parser.add_argument("--reps", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument(
        "--order-pairs",
        type=int,
        default=1,
        help="number of counterbalanced forward/reverse arm-order pairs",
    )
    parser.add_argument(
        "--arms",
        default="",
        help="optional comma-separated subset of registered arm names",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.reps < 1:
        parser.error("--reps must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.order_pairs < 1:
        parser.error("--order-pairs must be at least 1")
    rows_list = [int(v) for v in str(args.rows).split(",") if v.strip()]
    if not rows_list or any(rows < 1 for rows in rows_list):
        parser.error("--rows must contain positive integers")
    arms = _resolve_arms()
    requested_arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    if requested_arms:
        unknown = [name for name in requested_arms if name not in arms]
        if unknown:
            parser.error(f"unknown --arms: {','.join(unknown)}")
        if "shared_b" not in requested_arms:
            parser.error("--arms must include shared_b as the timing/correctness reference")
        arms = {name: arms[name] for name in requested_arms}
    shapes: list[tuple[int, int]] = [(int(args.in_features), int(args.out_features))]
    if args.all_shapes:
        shapes = shapes + [tuple(s) for s in OTHER_SHAPES]  # type: ignore[arg-type]

    shape_blocks: list[dict[str, object]] = []
    for in_features, out_features in shapes:
        tiles = _tile16_payload(out_features, in_features, args.seed)
        weight_bytes = int(tiles.nbytes)
        buffers = _Buffers(tiles)
        per_row: list[dict[str, object]] = []
        for rows in rows_list:
            x_host = _activations(rows, in_features, args.seed + rows)
            x_ptr = buffers.upload(x_host)
            out_buf = buffers.malloc(rows * out_features * 2)
            out_ptr = int(out_buf.ptr)
            results: dict[str, object] = {}
            measurements: dict[str, list[dict[str, object]]] = {
                name: [] for name in arms
            }
            captures: dict[str, dict[int, np.ndarray]] = {name: {} for name in arms}
            errors: dict[str, str] = {}
            schedule = _counterbalanced_schedule(
                tuple(arms), order_pairs=args.order_pairs
            )
            for sweep_pass in schedule:
                pass_index = int(sweep_pass["pass_index"])
                for position, name in enumerate(sweep_pass["arms"]):
                    if name in errors:
                        continue
                    try:
                        timing, captured = _timed(
                            arms[name],
                            buffers,
                            x_ptr,
                            out_ptr,
                            rows,
                            in_features,
                            out_features,
                            args.reps,
                            args.warmup,
                            capture=True,
                            out_buf=out_buf,
                        )
                    except Exception as exc:  # leaf may reject a shape/tile combination
                        errors[name] = f"{type(exc).__name__}: {exc}"[:160]
                        continue
                    measurements[name].append(
                        {
                            "pass_index": pass_index,
                            "pair_index": int(sweep_pass["pair_index"]),
                            "direction": str(sweep_pass["direction"]),
                            "position": position,
                            **timing,
                        }
                    )
                    if captured is not None:
                        captures[name][pass_index] = captured

            aggregate_ms: dict[str, float] = {}
            for name in arms:
                if name in errors:
                    results[name] = {"error": errors[name]}
                    continue
                passes = measurements[name]
                if len(passes) != len(schedule):
                    results[name] = {
                        "error": f"incomplete schedule: {len(passes)}/{len(schedule)} passes"
                    }
                    continue
                pass_medians = [float(item["median_ms"]) for item in passes]
                counterbalanced_median = statistics.median(pass_medians)
                direction_medians = {
                    direction: statistics.median(
                        float(item["median_ms"])
                        for item in passes
                        if item["direction"] == direction
                    )
                    for direction in ("forward", "reverse")
                }
                entry: dict[str, object] = {
                    "pass_measurements": passes,
                    "finite": all(
                        _bf16_finite(captured)
                        for captured in captures[name].values()
                    ),
                    "counterbalanced_median_ms": counterbalanced_median,
                    "direction_medians_ms": direction_medians,
                    "spread_pct": (
                        (max(pass_medians) - min(pass_medians))
                        * 100.0
                        / max(counterbalanced_median, 1e-9)
                    ),
                    "ms_per_row": counterbalanced_median / rows,
                    "effective_gbps": weight_bytes
                    / (counterbalanced_median * 1e-3)
                    / 1e9,
                }
                results[name] = entry
                aggregate_ms[name] = counterbalanced_median

            reference_captures = captures.get("shared_b", {})
            for name, value in results.items():
                if not isinstance(value, dict) or "error" in value:
                    continue
                if name == "shared_b":
                    value["ulp_vs_shared_b"] = 0
                    continue
                common_passes = sorted(
                    set(reference_captures).intersection(captures.get(name, {}))
                )
                if common_passes:
                    value["ulp_vs_shared_b"] = max(
                        _ulp_distance(reference_captures[index], captures[name][index])
                        for index in common_passes
                    )
                    value["output_capture_passes"] = common_passes

            baseline = aggregate_ms.get("shared_b")
            baseline_entry = results.get("shared_b")
            for name, value in results.items():
                if (
                    isinstance(value, dict)
                    and "counterbalanced_median_ms" in value
                    and baseline
                ):
                    value["shared_b_over_candidate"] = (
                        baseline / float(value["counterbalanced_median_ms"])
                    )
                    if name != "shared_b" and isinstance(baseline_entry, dict):
                        baseline_directions = baseline_entry["direction_medians_ms"]
                        candidate_directions = value["direction_medians_ms"]
                        order_ratios = {
                            direction: float(baseline_directions[direction])
                            / float(candidate_directions[direction])
                            for direction in ("forward", "reverse")
                        }
                        value["direction_shared_b_over_candidate"] = order_ratios
                        value["non_regressive_in_both_orders"] = all(
                            ratio >= 1.0 for ratio in order_ratios.values()
                        )
                # A leaf can time fast while reading buffers this harness does not
                # supply, which is not a speedup. Mark the divergence so a ratio can
                # never be read as a win: (17408, 5120) dense_rowtile timed 14.3x at
                # 0.071 ms while disagreeing with the owner by ~1e7 BF16 ULPs.
                if isinstance(value, dict) and value.get("ulp_vs_shared_b") not in (None, 0):
                    value["output_diverges_from_owner"] = True
                    value.pop("shared_b_over_candidate", None)
                    value.pop("direction_shared_b_over_candidate", None)
                    value.pop("non_regressive_in_both_orders", None)
            successful_entries = [
                value
                for value in results.values()
                if isinstance(value, dict) and "error" not in value
            ]
            per_row.append(
                {
                    "rows": rows,
                    "arms": results,
                    "all_successful_arm_outputs_finite": bool(successful_entries)
                    and all(bool(value.get("finite")) for value in successful_entries),
                }
            )
            line = f"shape=({in_features},{out_features}) rows={rows:>4}"
            for name in list(arms):
                value = results.get(name, {})
                if isinstance(value, dict) and "counterbalanced_median_ms" in value:
                    line += f"  {name}={value['counterbalanced_median_ms']:>8.3f}ms"
                    if name != "shared_b":
                        if value.get("output_diverges_from_owner"):
                            line += "(DIV ulp=%s)" % value.get("ulp_vs_shared_b")
                        else:
                            line += f"({value['shared_b_over_candidate']:.2f}x)"
                else:
                    line += f"  {name}=err"
            print(line, flush=True)
        shape_blocks.append(
            {
                "in_features": in_features,
                "out_features": out_features,
                "weight_tile_bytes": weight_bytes,
                "rows": per_row,
            }
        )

    payload = {
        "schema": 3,
        "kind": "gguf_t16_prefill_row_sweep",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "host": {"hostname": platform.node(), "machine": platform.machine(), "python": platform.python_version()},
        "shapes": shape_blocks,
        "protocol": {
            "reps": args.reps,
            "warmup": args.warmup,
            "order_pairs": args.order_pairs,
            "arm_names": list(arms),
            "schedule": _counterbalanced_schedule(
                tuple(arms), order_pairs=args.order_pairs
            ),
            "seed": args.seed,
            "timing": "wall around one launch plus device_synchronize; counterbalanced median over forward/reverse passes; ratios are shared_b/candidate so >1.0 means faster; no best-of selection",
            "payload": "repack_gguf_q4_k_tile16 output over a valid Q4_K block image",
            "ulp": "worst BF16 ULP distance against the shared-B output, an upper bound where signs differ",
        },
        "kernel_tile_rows": {"shared_b": 256},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
