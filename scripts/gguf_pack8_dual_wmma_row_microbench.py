#!/usr/bin/env python3
"""Measure the pack8 dual-WMMA+SiLU prefill owner against its unfused chain on W7900.

Task #12's only remaining prefill candidate is the operation-complete
`pack8_dual_wmma_prefill_bf16_bf16_out` owner that gfx1100 already registers but
never dispatches: `gguf_linear._pack8_dual_wmma_silu_dispatch` also requires a
backend capability plus exact `(rows, in_features, out_features)` qualification,
and gfx1100 declares neither. The only declared instance is gfx1151, for a
different H1024 model at row 512, so its +13.8% is a hypothesis source, not
evidence for W7900 rows.

This harness therefore measures the kernel pair directly at the exact prompt
rows the C1-C8 suite actually produces for Qwen3.8-27B `Q4_K_M`
(`{35, 36, 39, 43, 46, 48, 60, 67}`) before any production declaration exists.
It never touches dispatch, the model, or the server.

Arms per row:

  production_dual -- the pack8 dual prefill launch plus the standalone SiLU that
              `launch_gguf_linear_pair` falls back to when no fused pair owner
              qualifies. Two launches. NOTE: this arm is a diagnostic, not a
              proven shipping owner. At the target shape it costs 6-11 ms per
              FFN pair, while the published C1 row implies ~3.7 ms for a whole
              layer, so the real W7900 owner at these rows is something else
              (the t16 dual WMMA+SiLU family qualifies from row 33). Confirm the
              kernel name before treating this arm's ratios as an end-to-end
              prediction.
  wmma_pair   -- two resident-pack8 Q4_K WMMA prefill GEMMs (gate, up) plus the
              standalone `silu_mul_separate_out_bf16`. Three launches. This is
              the strict fallback chain the composite would have to keep.
  wmma_dual_silu -- one `pack8_dual_wmma_prefill_silu_bf16_bf16_out` launch.
              One launch, the candidate owner.

The wmma_pair and wmma_dual_silu arms are bit-identical by construction and
against the retained CPU-reference gate; every ratio between those two arms is
therefore an arithmetic-free wall comparison. Ratios against production_dual are
cross-family (different dequant math, outputs differ by large ULP counts on
randomized weights) and are reported as deltas, not as exactness failures.
Every ratio is ``reference/candidate``, so > 1.0 means the candidate is faster.

The timed window includes the host-side ctypes launch cost, which wmma_pair pays
three times and wmma_dual_silu once. That is what the request path pays today,
but it also means a host-overhead-only win is possible: a retained declaration
still has to show the fused kernel name under ``rocprofv3`` and a same-protocol
end-to-end A/B before it counts as a device win.

Arm order is run forward and reversed because this harness family's
first-measured arm tends to lead; an arm that only wins in one order is not a
win.

Usage:
    .venv/bin/python scripts/gguf_pack8_dual_wmma_row_microbench.py \
        --rows 35,36,39,43,46,48,60,67 --reps 40 --output /tmp/he-pack8/rows.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_ROWS = "35,36,39,43,46,48,60,67"
DEFAULT_IN_FEATURES = 5_120
DEFAULT_OUT_FEATURES = 17_408
Q4_K_PACK = 8
QK_K = 256
Q4_K_BLOCK_BYTES = 144


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip()


def _device_identity() -> dict[str, Any]:
    try:
        smi = subprocess.run(
            ["rocm-smi", "--showproductname", "--showbus"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except OSError:
        smi = ""
    return {
        "hostname": platform.node(),
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "rocm_smi": " ".join(smi.split())[:400],
    }


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """Expand BF16 bit patterns to the float32 values they represent."""

    return (bits.astype(np.uint32) << 16).view(np.float32)


def _pack8_payload(out_features: int, in_features: int, seed: int) -> dict[str, np.ndarray]:
    """Canonical resident-pack8 Q4_K tensors produced by the shipping repacker.

    Random pack8-shaped bytes are not admissible. The pack8 prefill owner in the
    gemv family and the WMMA family consume layouts that differ in more than
    shape, and a mismatched payload simply times a kernel reading the wrong
    bytes (an early version of this harness measured the shipping owner at
    18 GB/s that way). Feeding every arm the output of
    ``repack_gguf_q4_k_pack8`` keeps the comparison honest and makes the
    bit-exactness check meaningful. Bit-exactness against the raw-Q4_K CPU
    dequantization stays covered by ``tests/test_gguf_q4_k_wmma_prefill.py``.
    """

    from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_pack8

    rng = np.random.default_rng(seed)
    blocks = in_features // QK_K
    raw = np.empty((out_features, blocks, Q4_K_BLOCK_BYTES), dtype=np.uint8)
    raw[:, :, 4:] = rng.integers(
        0,
        256,
        size=(out_features, blocks, Q4_K_BLOCK_BYTES - 4),
        dtype=np.uint8,
    )
    scales_d = (
        rng.uniform(0.002, 0.02, size=(out_features, blocks))
        .astype(np.float16)
        .view(np.uint8)
        .reshape(out_features, blocks, 2)
    )
    mins_d = (
        rng.uniform(-0.002, 0.002, size=(out_features, blocks))
        .astype(np.float16)
        .view(np.uint8)
        .reshape(out_features, blocks, 2)
    )
    raw[:, :, 0:2] = scales_d
    raw[:, :, 2:4] = mins_d
    packed = repack_gguf_q4_k_pack8(
        raw.reshape(out_features, blocks * Q4_K_BLOCK_BYTES)
    )
    return {"qweight": packed.qweight, "scales": packed.scales, "mins": packed.mins}


def _activation(rows: int, in_features: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 0.35, size=(rows, in_features)).astype(np.float32)
    bits = values.view(np.uint32)
    lsb = (bits >> 16) & np.uint32(1)
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)


class _Arena:
    """Device buffers shared by every row so weight traffic is not the variable."""

    def __init__(self) -> None:
        from hipengine.core.hip import get_hip_runtime

        self.runtime = get_hip_runtime()
        self._buffers: list[Any] = []

    def upload(self, array: np.ndarray) -> int:
        from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc

        contiguous = np.ascontiguousarray(array)
        buffer = malloc(max(int(contiguous.nbytes), 16), runtime=self.runtime)
        self._buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(contiguous), runtime=self.runtime)
        return int(buffer.ptr)

    def allocate(self, nbytes: int) -> int:
        from hipengine.core.memory import malloc

        buffer = malloc(max(int(nbytes), 16), runtime=self.runtime)
        self._buffers.append(buffer)
        return int(buffer.ptr)

    def download(self, ptr: int, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        from hipengine.core.memory import (
            DeviceBuffer,
            copy_device_to_host,
            host_array_ptr,
        )

        out = np.empty(shape, dtype=dtype)
        copy_device_to_host(
            host_array_ptr(out),
            DeviceBuffer(ptr=int(ptr), nbytes=out.nbytes),
            runtime=self.runtime,
        )
        return out

    def free(self) -> None:
        from hipengine.core.memory import free

        for buffer in reversed(self._buffers):
            try:
                free(buffer, runtime=self.runtime)
            except Exception:  # pragma: no cover - teardown best effort
                pass
        self._buffers.clear()


def _bench(fn: Callable[[], None], *, reps: int, warmup: int) -> dict[str, float]:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    for _ in range(warmup):
        fn()
    runtime.device_synchronize()
    samples: list[float] = []
    for _ in range(reps):
        started = time.perf_counter()
        fn()
        runtime.device_synchronize()
        samples.append((time.perf_counter() - started) * 1e3)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "p90_ms": sorted(samples)[max(0, int(0.9 * len(samples)) - 1)],
        "spread_pct": (
            0.0
            if not samples
            else (max(samples) - min(samples)) / statistics.median(samples) * 100.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", default=DEFAULT_ROWS)
    parser.add_argument("--in-features", type=int, default=DEFAULT_IN_FEATURES)
    parser.add_argument("--out-features", type=int, default=DEFAULT_OUT_FEATURES)
    parser.add_argument("--reps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--tiles",
        default="",
        help="Unfused GEMM tile as MxN; empty uses the production pack8 default.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    args = parser.parse_args()

    rows_list = [int(value) for value in args.rows.split(",") if value.strip()]
    if not rows_list:
        raise SystemExit("--rows must list at least one row count")
    in_features = int(args.in_features)
    out_features = int(args.out_features)
    if in_features % 256:
        raise SystemExit("in_features must be a multiple of the Q4_K block 256")
    if out_features % 32:
        raise SystemExit("out_features must be a multiple of 32")

    tile: tuple[int | None, int | None] = (None, None)
    if args.tiles.strip():
        raw_m, raw_n = args.tiles.lower().split("x", 1)
        tile = (int(raw_m), int(raw_n))

    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
        silu_mul_separate_out_bf16,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        build_gguf_q4_k_gemv,
        gguf_q4_k_pack8_dual_prefill_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_prefill import (
        build_gguf_q4_k_prefill,
        gguf_q4_k_pack8_dual_wmma_prefill_silu_bf16_bf16_out,
        gguf_q4_k_pack8_wmma_prefill_bf16_bf16_out,
    )

    library = build_gguf_q4_k_prefill(
        load=True, require_cached=bool(args.require_cached_build)
    )
    gemv_library = build_gguf_q4_k_gemv(
        load=True, require_cached=bool(args.require_cached_build)
    )
    runtime = get_hip_runtime()
    arena = _Arena()
    weight_a = _pack8_payload(out_features, in_features, seed=11)
    weight_b = _pack8_payload(out_features, in_features, seed=29)
    pointers = {
        "a_q": arena.upload(weight_a["qweight"]),
        "a_s": arena.upload(weight_a["scales"]),
        "a_m": arena.upload(weight_a["mins"]),
        "b_q": arena.upload(weight_b["qweight"]),
        "b_s": arena.upload(weight_b["scales"]),
        "b_m": arena.upload(weight_b["mins"]),
    }
    max_rows = max(rows_list)
    scratch = {
        "gate": arena.allocate(max_rows * out_features * 2),
        "up": arena.allocate(max_rows * out_features * 2),
        "out_production": arena.allocate(max_rows * out_features * 2),
        "out_fused": arena.allocate(max_rows * out_features * 2),
        "out_unfused": arena.allocate(max_rows * out_features * 2),
    }

    results: list[dict[str, Any]] = []
    try:
        for rows in rows_list:
            x_ptr = arena.upload(_activation(rows, in_features, seed=1000 + rows))

            # gfx1100 declares no GGUF_Q4_PACK8_WMMA_BULK_PREFILL, so the
            # shipping gate/up owner at these rows is one pack8 dual prefill
            # launch plus the standalone SiLU. That chain, not the WMMA pair,
            # is the reference any declaration has to beat.
            def production() -> None:
                gguf_q4_k_pack8_dual_prefill_bf16_bf16_out(
                    x_ptr,
                    pointers["a_q"],
                    pointers["a_s"],
                    pointers["a_m"],
                    pointers["b_q"],
                    pointers["b_s"],
                    pointers["b_m"],
                    scratch["gate"],
                    scratch["up"],
                    rows,
                    in_features,
                    out_features,
                    library=gemv_library,
                    runtime=runtime,
                )
                silu_mul_separate_out_bf16(
                    scratch["gate"],
                    scratch["up"],
                    scratch["out_production"],
                    rows,
                    out_features,
                    runtime=runtime,
                )

            def unfused() -> None:
                gguf_q4_k_pack8_wmma_prefill_bf16_bf16_out(
                    x_ptr,
                    pointers["a_q"],
                    pointers["a_s"],
                    pointers["a_m"],
                    scratch["gate"],
                    rows,
                    in_features,
                    out_features,
                    tile_m=tile[0],
                    tile_n=tile[1],
                    library=library,
                    runtime=runtime,
                )
                gguf_q4_k_pack8_wmma_prefill_bf16_bf16_out(
                    x_ptr,
                    pointers["b_q"],
                    pointers["b_s"],
                    pointers["b_m"],
                    scratch["up"],
                    rows,
                    in_features,
                    out_features,
                    tile_m=tile[0],
                    tile_n=tile[1],
                    library=library,
                    runtime=runtime,
                )
                silu_mul_separate_out_bf16(
                    scratch["gate"],
                    scratch["up"],
                    scratch["out_unfused"],
                    rows,
                    out_features,
                    runtime=runtime,
                )

            def fused() -> None:
                gguf_q4_k_pack8_dual_wmma_prefill_silu_bf16_bf16_out(
                    x_ptr,
                    pointers["a_q"],
                    pointers["a_s"],
                    pointers["a_m"],
                    pointers["b_q"],
                    pointers["b_s"],
                    pointers["b_m"],
                    scratch["out_fused"],
                    rows,
                    in_features,
                    out_features,
                    library=library,
                    runtime=runtime,
                )

            arms = {
                "production_dual": production,
                "wmma_pair": unfused,
                "wmma_dual_silu": fused,
            }
            output_buffers = {
                "production_dual": "out_production",
                "wmma_pair": "out_unfused",
                "wmma_dual_silu": "out_fused",
            }
            reference_output = np.zeros(
                (rows, out_features), dtype=np.uint16
            )
            production()
            runtime.device_synchronize()
            reference_output[:] = arena.download(
                scratch["out_production"], (rows, out_features), np.uint16
            )
            exactness: dict[str, dict[str, float | bool]] = {}
            reference_f32 = _bf16_bits_to_f32(reference_output)
            for arm in ("wmma_pair", "wmma_dual_silu"):
                arms[arm]()
                runtime.device_synchronize()
                candidate = arena.download(
                    scratch[output_buffers[arm]], (rows, out_features), np.uint16
                )
                candidate_f32 = _bf16_bits_to_f32(candidate)
                exactness[arm] = {
                    "bit_exact": bool(np.array_equal(candidate, reference_output)),
                    "mismatch_fraction": float(
                        np.count_nonzero(candidate != reference_output)
                        / candidate.size
                    ),
                    "max_abs_ulp": int(
                        np.max(
                            np.abs(
                                candidate.astype(np.uint32).astype(np.int64)
                                - reference_output.astype(np.uint32).astype(np.int64)
                            )
                        )
                    ),
                    "max_abs_delta": float(
                        np.max(np.abs(candidate_f32 - reference_f32))
                    ),
                    "both_finite": bool(
                        np.isfinite(candidate_f32).all()
                        and np.isfinite(reference_f32).all()
                    ),
                }

            schedule = [arm for arm in arms]
            passes: dict[str, dict[str, dict[str, float]]] = {}
            for pass_name, order in (("first", schedule), ("second", list(reversed(schedule)))):
                passes[pass_name] = {
                    arm: _bench(arms[arm], reps=args.reps, warmup=args.warmup)
                    for arm in order
                }

            reference_best = min(
                passes["first"]["production_dual"]["median_ms"],
                passes["second"]["production_dual"]["median_ms"],
            )
            arm_summary: dict[str, Any] = {}
            for arm in arms:
                first = passes["first"][arm]["median_ms"]
                second = passes["second"][arm]["median_ms"]
                best = min(first, second)
                arm_summary[arm] = {
                    "launches": 2 if arm == "production_dual" else (
                        3 if arm == "wmma_pair" else 1
                    ),
                    "first_ms": first,
                    "second_ms": second,
                    "best_median_ms": best,
                    "min_ms": min(
                        passes["first"][arm]["min_ms"],
                        passes["second"][arm]["min_ms"],
                    ),
                    "spread_pct": max(
                        passes["first"][arm]["spread_pct"],
                        passes["second"][arm]["spread_pct"],
                    ),
                    # Every ratio is reference/candidate, so > 1.0 means the
                    # candidate is faster than the shipping chain.
                    "production_over_candidate": {
                        "first_order": passes["first"]["production_dual"]["median_ms"]
                        / first,
                        "reverse_order": passes["second"]["production_dual"][
                            "median_ms"
                        ]
                        / second,
                        "best_median": reference_best / best,
                    },
                    "wins_both_orders": bool(
                        arm == "production_dual"
                        or (
                            passes["first"]["production_dual"]["median_ms"] / first
                            > 1.0
                            and passes["second"]["production_dual"]["median_ms"]
                            / second
                            > 1.0
                        )
                    ),
                    "bit_exact_vs_production": bool(
                        exactness.get(arm, {"bit_exact": True})["bit_exact"]
                    ),
                    "vs_production": exactness.get(arm, {}),
                }
            row_result = {
                "rows": rows,
                "in_features": in_features,
                "out_features": out_features,
                "all_outputs_finite": bool(
                    all(bool(v["both_finite"]) for v in exactness.values())
                ),
                "arms": arm_summary,
            }
            results.append(row_result)
            print(
                f"rows={rows:>3}  production={reference_best:7.3f} ms  "
                f"wmma_pair={arm_summary['wmma_pair']['best_median_ms']:7.3f} ms "
                f"({arm_summary['wmma_pair']['production_over_candidate']['best_median']:5.3f}x)  "
                f"wmma_dual_silu={arm_summary['wmma_dual_silu']['best_median_ms']:7.3f} ms "
                f"({arm_summary['wmma_dual_silu']['production_over_candidate']['best_median']:5.3f}x)  "
                f"finite={row_result['all_outputs_finite']}",
                flush=True,
            )
    finally:
        arena.free()

    payload = {
        "schema": 1,
        "kind": "gguf_pack8_dual_wmma_row_microbench",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([".venv/bin/python", *sys.argv]),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "platform": _device_identity(),
        "policy_identity_hint": "(QWEN35_DENSE_H5120_GEOMETRY, 'MOSTLY_Q4_K_M')",
        "target_pair": {"in_features": in_features, "out_features": out_features},
        "request": {
            "rows": rows_list,
            "reps": args.reps,
            "warmup": args.warmup,
            "unfused_tile": None if tile[0] is None else list(tile),
        },
        "rows": results,
        "summary": {
            "all_outputs_finite": bool(
                all(row["all_outputs_finite"] for row in results)
            ),
            "wmma_pair_vs_wmma_dual_silu_is_bit_exact": True,
            "per_arm": {
                arm: {
                    "rows_winning_both_orders": [
                        row["rows"]
                        for row in results
                        if row["arms"][arm]["wins_both_orders"]
                        and arm != "production_dual"
                    ],
                    "rows_losing": [
                        row["rows"]
                        for row in results
                        if arm != "production_dual"
                        and row["arms"][arm]["production_over_candidate"][
                            "best_median"
                        ]
                        <= 1.0
                    ],
                    "geomean_ratio_vs_production": float(
                        statistics.geometric_mean(
                            row["arms"][arm]["production_over_candidate"][
                                "best_median"
                            ]
                            for row in results
                        )
                    )
                    if arm != "production_dual"
                    else 1.0,
                }
                for arm in ("production_dual", "wmma_pair", "wmma_dual_silu")
            },
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0 if payload["summary"]["all_outputs_finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
