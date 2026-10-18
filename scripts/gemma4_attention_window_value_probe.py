#!/usr/bin/env python3
"""What does the sliding window actually buy in Gemma 4's exact attention kernel?

Direct measurement at the prefill block shape, with no head_dim inference. The
same geometry is timed twice: once with the pure causal mask the 5 global layers
build, and once with the causal-and-windowed mask the 25 sliding layers build.
The difference is exactly what the window buys in the kernel that serves every
prefill block past the 1024-token window.

Why this exists. The punchlist's first estimate of that difference assumed the
kernel scales linearly with head_dim, and therefore read the sliding/global
per-layer cost ratio (59.1 / 122.1 ms at 4096) as "the window buys about 3%".
That assumption is false: doubling head_dim costs only about 1.28x, because the
kernel is dominated by its per-key walk rather than by per-key ALU work. The
window buys about 30%, not 3%. ``--head-dim-ratio`` reproduces the refutation
directly; see ``docs/campaigns/GEMMA4-26B-A4B-PUNCHLIST.md`` V3.

``--mode block`` times one block shape (the default: rows 512, keys 4096, which
is the last and most expensive block of a 4096-token prefill). ``--mode sweep``
walks all eight blocks of a 4096-token prefill and reports the sums, which are
the numbers comparable to the per-layer costs in the campaign's family census.

Absolute times here are inflated relative to a profiler's device-busy sum,
because each launch is timed in isolation and therefore includes launch latency
that a back-to-back stream never pays. Compare ratios, not absolute values: the
causal arm runs about 1.27x the profiled cost and the shorter windowed arm about
1.44x.

Recipe (from the gemma4 worktree; ``N`` is the physical GPU)::

    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=N PYTHONPATH=. \\
      .venv/bin/python scripts/gemma4_attention_window_value_probe.py \\
      --json benchmarks/results/<artifact>.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The prefill block shape: Gemma 4 26B-A4B's sliding geometry at max_block 512.
DEFAULT_ROWS = 512
DEFAULT_KEYS = 4096
DEFAULT_HEADS = 16
DEFAULT_WINDOW = 1024
# The two real geometries. Sliding layers are head_dim 256 with 8 KV heads;
# global layers are head_dim 512 with 2.
SLIDING = {"num_kv_heads": 8, "head_dim": 256}
GLOBAL = {"num_kv_heads": 2, "head_dim": 512}
# head_dim isolation holds the KV head count fixed so only head_dim moves.
ISOLATION = {"num_kv_heads": 8, "head_dim": 512}


def bf16(values: np.ndarray) -> np.ndarray:
    """Round-trip through float16 and keep the raw 16-bit words."""

    return np.ascontiguousarray(
        values.astype(np.float32).astype(np.float16).view(np.uint16)
    )


def keep_mask(start: int, rows: int, keys: int, window: int | None) -> np.ndarray:
    """The ``(rows, keys)`` uint8 keep-mask, matching ``_keep_mask``."""

    queries = np.arange(start, start + rows, dtype=np.int64)[:, None]
    key_positions = np.arange(keys, dtype=np.int64)[None, :]
    keep = key_positions <= queries
    if window is not None:
        keep &= (queries - key_positions) < int(window)
    return np.ascontiguousarray(keep.astype(np.uint8))


def time_block(
    runtime: Any,
    mask: np.ndarray,
    *,
    rows: int,
    keys: int,
    heads: int,
    num_kv_heads: int,
    head_dim: int,
    warmup: int,
    reps: int,
    seed: int = 5,
) -> float:
    """Median milliseconds for one launch of the exact prefill kernel."""

    from hipengine.core.memory import (
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        Gemma4AttentionScratch,
        gemma4_attention_prefill_bf16,
    )

    rng = np.random.default_rng(seed)
    query = bf16(rng.normal(0.0, 0.5, (rows, heads, head_dim)))
    key = bf16(rng.normal(0.0, 0.5, (keys, num_kv_heads, head_dim)))
    value = bf16(rng.normal(0.0, 0.5, (keys, num_kv_heads, head_dim)))
    out_host = np.zeros((rows, heads, head_dim), dtype=np.uint16)

    owned = []
    scratch = Gemma4AttentionScratch()

    def upload(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        owned.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), array.nbytes, runtime=runtime)
        return buffer

    q_buf = upload(query)
    k_buf = upload(key)
    v_buf = upload(value)
    m_buf = upload(mask)
    o_buf = upload(out_host)

    def once() -> None:
        gemma4_attention_prefill_bf16(
            q_buf.ptr,
            k_buf.ptr,
            v_buf.ptr,
            m_buf.ptr,
            o_buf.ptr,
            tokens=rows,
            keys=keys,
            num_heads=heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=1.0,
            scratch=scratch,
            runtime=runtime,
        )

    try:
        for _ in range(warmup):
            once()
        runtime.device_synchronize()
        samples = []
        for _ in range(reps):
            runtime.device_synchronize()
            started = time.perf_counter()
            once()
            runtime.device_synchronize()
            samples.append((time.perf_counter() - started) * 1e3)
        return float(statistics.median(samples))
    finally:
        scratch.close()
        for buffer in owned:
            free(buffer, runtime=runtime)


def sweep(runtime: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Time every block of a ``keys``-token prefill, causal and windowed."""

    rows = args.rows
    blocks = []
    totals = {"causal_sliding": 0.0, "windowed_sliding": 0.0, "causal_global": 0.0}
    for index in range(args.keys // rows):
        start = index * rows
        keys = start + rows
        causal = time_block(
            runtime, keep_mask(start, rows, keys, None), rows=rows, keys=keys,
            heads=args.heads, warmup=args.warmup, reps=args.reps, **SLIDING,
        )
        windowed = time_block(
            runtime, keep_mask(start, rows, keys, args.window), rows=rows, keys=keys,
            heads=args.heads, warmup=args.warmup, reps=args.reps, **SLIDING,
        )
        glob = time_block(
            runtime, keep_mask(start, rows, keys, None), rows=rows, keys=keys,
            heads=args.heads, warmup=args.warmup, reps=args.reps, **GLOBAL,
        )
        totals["causal_sliding"] += causal
        totals["windowed_sliding"] += windowed
        totals["causal_global"] += glob
        blocks.append(
            {"start": start, "keys": keys, "causal_sliding_ms": causal,
             "windowed_sliding_ms": windowed, "causal_global_ms": glob}
        )
        print(
            f"  keys {keys:5d}  causal hd256 {causal:8.3f}  "
            f"windowed hd256 {windowed:8.3f}  causal hd512 {glob:8.3f}",
            flush=True,
        )

    causal = totals["causal_sliding"]
    windowed = totals["windowed_sliding"]
    result = {
        "mode": "sweep",
        "shape": {"rows": rows, "keys": args.keys, "heads": args.heads,
                  "window": args.window},
        "blocks": blocks,
        "totals": totals,
        "window_saving_pct": 100.0 * (causal - windowed) / causal,
        "window_speedup": causal / windowed,
    }
    print()
    print(f"  sum causal   hd256 (all blocks) = {causal:8.2f} ms")
    print(f"  sum windowed hd256 (all blocks) = {windowed:8.2f} ms")
    print(f"  sum causal   hd512 (all blocks) = {totals['causal_global']:8.2f} ms")
    print(f"  window saving = {result['window_saving_pct']:+.1f}%  "
          f"({result['window_speedup']:.3f}x)")
    return result


def head_dim_ratio(runtime: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Refute (or confirm) the head_dim-linearity assumption, block by block."""

    rows = args.rows
    points = []
    ratios = []
    for index in range(args.keys // rows):
        start = index * rows
        keys = start + rows
        mask = keep_mask(start, rows, keys, None)
        small = time_block(
            runtime, mask, rows=rows, keys=keys, heads=args.heads,
            warmup=args.warmup, reps=args.reps, **SLIDING,
        )
        large = time_block(
            runtime, mask, rows=rows, keys=keys, heads=args.heads,
            warmup=args.warmup, reps=args.reps, **ISOLATION,
        )
        ratios.append(large / small)
        points.append({"keys": keys, "hd256_ms": small, "hd512_ms": large,
                       "ratio": large / small})
        print(f"  keys {keys:5d}  hd256 {small:8.3f}  hd512 {large:8.3f}  "
              f"ratio {large / small:.3f}", flush=True)

    mean = float(statistics.fmean(ratios))
    result = {
        "mode": "head-dim-ratio",
        "points": points,
        "ratios": ratios,
        "mean_ratio": mean,
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
        "linear_assumption": 2.0,
    }
    print()
    print(f"  head_dim 512/256 at fixed kv heads: mean {mean:.3f}x "
          f"(range {min(ratios):.3f}-{max(ratios):.3f}); "
          f"the linear assumption would be 2.000x")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("block", "sweep", "head-dim-ratio"),
                        default="sweep")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--keys", type=int, default=DEFAULT_KEYS)
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--label", default="gemma4-attention-window-value")
    args = parser.parse_args(argv)

    if args.rows <= 0 or args.keys <= 0 or args.keys % args.rows:
        parser.error("require rows > 0 and keys to be a positive multiple of rows")

    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    print(f"{args.label}: mode={args.mode} rows={args.rows} keys={args.keys} "
          f"heads={args.heads} window={args.window}", flush=True)

    if args.mode == "sweep":
        result = sweep(runtime, args)
    elif args.mode == "head-dim-ratio":
        result = head_dim_ratio(runtime, args)
    else:
        causal = time_block(
            runtime, keep_mask(args.keys - args.rows, args.rows, args.keys, None),
            rows=args.rows, keys=args.keys, heads=args.heads,
            warmup=args.warmup, reps=args.reps, **SLIDING,
        )
        windowed = time_block(
            runtime, keep_mask(args.keys - args.rows, args.rows, args.keys, args.window),
            rows=args.rows, keys=args.keys, heads=args.heads,
            warmup=args.warmup, reps=args.reps, **SLIDING,
        )
        result = {
            "mode": "block",
            "shape": {"rows": args.rows, "keys": args.keys, "heads": args.heads,
                      "window": args.window},
            "causal_ms": causal,
            "windowed_ms": windowed,
            "window_saving_pct": 100.0 * (causal - windowed) / causal,
            "window_speedup": causal / windowed,
        }
        print(f"  causal {causal:.3f} ms -> windowed {windowed:.3f} ms = "
              f"{result['window_saving_pct']:+.1f}% ({result['window_speedup']:.3f}x)")

    if args.json is not None:
        payload = {
            "label": args.label,
            "performance_claim": False,
            "command": " ".join(["python3", "scripts/gemma4_attention_window_value_probe.py"] + sys.argv[1:]),
            "warmup": args.warmup,
            "reps": args.reps,
            "result": result,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"  artifact={args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
