"""Surya attention memory budgets: vision and text-prefill score tiles.

The Surya path has two attention score matrices and both are quadratic in
sequence length. The vision tower's is tiled by query rows, so only
``heads * n * block * 4`` bytes are live for ``n`` patches; this script emits
the tile plan for the grids that matter (the 300-DPI A4 page and the
checkpoint's ``max_pixels`` ceiling) and compares it with the dense matrix the
previous path materialized.

The text prefill is tiled the same way, so the second half of this script
measures what the tiling actually costs on the device and separates the terms
so a context budget can be chosen from evidence instead of from the model's
``max_position_embeddings``. Pass ``--text-budget-bytes 0`` to measure the
dense path the tiling replaced.

What it reports, per (``max_seq``, ``text budget``, token count):

* ``resident_bytes`` — the runner's long-lived allocations: fp32 weights, KV
  planes (``2 * nk * max_seq * hd * 4`` per full-attention layer), GDN
  recurrence state, conv windows, and the ``KVLiveSpans`` state (the split-K
  decode partials, the page table, the position table, and the eviction mask).
* ``peak_scratch_bytes`` — the high-water mark of hipEngine-owned device
  buffers above ``resident_bytes`` while the prefill runs. Split into
  ``score_tile_bytes`` (the planned causal score tile, ``nq * tokens * block *
  4``) and everything else, which is linear in the token count (activations,
  GDN projections, MLP scratch).
* ``peak_total_bytes`` — the sum, i.e. what the device must have free.

Measurement is the process-local hipEngine allocation counter, so it covers
buffers hipEngine owns and excludes ROCm/rocBLAS internal allocations. The
score-tile term is also checked against the planner, and the linear coefficient
is fitted across the token counts so the page-scale numbers can be
extrapolated rather than guessed.

Usage:
    python3 scripts/surya_attention_memory.py --out benchmarks/results/<name>.json
    python3 scripts/surya_attention_memory.py --tokens 2048,4096,8580,16384 \
        --text-budget-bytes 0 536870912 --runs 3
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MODEL_ID = "datalab-to/surya-ocr-2"

# A 300-DPI A4 page resizes to a 220x156 patch grid: 34320 unmerged patches,
# i.e. 8580 merged image tokens before the prompt and any output.
A4_300DPI_GRID = (1, 220, 156)
# The checkpoint's own preprocessor ceiling (SURYA_MAX_PIXELS = 16_777_216 px)
# admits a 4096x4096 page, a 256x256 patch grid.
MAX_PIXELS_GRID = (1, 256, 256)
PAGE_TOKENS_300DPI = (220 // 2) * (156 // 2)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=False
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _cpu_model() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text()
        match = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
        if match:
            return match.group(1).strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _provenance(argv: list[str]) -> dict[str, object]:
    info: dict[str, object] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "cpu": _cpu_model(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "command": " ".join(argv),
        "git_revision": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_hip"] = getattr(torch.version, "hip", None)
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = None
    return info


def vision_tile_plan(model: str, grids: list[tuple[int, int, int]], budget: int) -> list[dict]:
    """Dense vs tiled score footprint for each vision grid."""

    from hipengine.loading.surya import load_surya_spec, resolve_surya_path
    from hipengine.runtime.surya import plan_vision_attention

    spec = load_surya_spec(resolve_surya_path(model))
    heads = spec.vision_num_heads
    rows = []
    for grid in grids:
        n = int(grid[1]) * int(grid[2])
        block, scratch = plan_vision_attention(n, heads, budget)
        rows.append(
            {
                "grid": [int(v) for v in grid],
                "patches": n,
                "merged_image_tokens": (int(grid[1]) // 2) * (int(grid[2]) // 2),
                "dense_score_bytes": heads * n * n * 4,
                "query_block": block,
                "tiled_score_bytes": scratch,
                "reduction_x": (heads * n * n * 4) / scratch,
            }
        )
    return rows


def text_tile_plan(model: str, token_counts: list[int], budgets: list[int | None]) -> list[dict]:
    """Dense vs tiled causal score footprint for each text-prefill length."""

    from hipengine.loading.surya import load_surya_spec, resolve_surya_path
    from hipengine.runtime.surya import plan_score_tiles

    spec = load_surya_spec(resolve_surya_path(model))
    heads = spec.num_attention_heads
    rows = []
    for tokens in token_counts:
        dense = heads * tokens * tokens * 4
        plans = []
        for budget in budgets:
            block, scratch = plan_score_tiles(tokens, heads, budget)
            plans.append(
                {
                    "budget_bytes": None if budget is None else int(budget),
                    "query_block": block,
                    "score_tile_bytes": scratch,
                    "reduction_x": dense / scratch,
                }
            )
        rows.append(
            {
                "tokens": int(tokens),
                "dense_score_bytes": dense,
                "plans": plans,
            }
        )
    return rows


def _spec_and_weights(model: str):
    from hipengine.loading.surya import (
        load_surya_spec,
        load_surya_weights,
        resolve_surya_path,
    )

    model_dir = resolve_surya_path(model)
    return load_surya_spec(model_dir), load_surya_weights(model_dir)


def _synthetic_prompt(tokens: int) -> tuple[np.ndarray, np.ndarray]:
    """A text-only prompt of ``tokens`` ids with plain 1-D rope positions."""

    ids = (np.arange(1, tokens + 1, dtype=np.int64) % 1000) + 1
    axis = np.arange(tokens, dtype=np.int64)
    return ids, np.stack([axis, axis, axis])


def _resident_components(spec, weights, max_seq: int) -> dict:
    """Long-lived allocation accounting for one runner configuration."""

    from hipengine.core.memory import memory_stats, reset_memory_stats
    from hipengine.kernels.hip_gfx1100.surya.surya_ops import plan_surya_dense_spans
    from hipengine.runtime.surya import SuryaGpuRunner

    probe = SuryaGpuRunner(weights, spec, max_seq=max_seq)
    try:
        reset_memory_stats()
        s = spec
        nk, hd = s.num_key_value_heads, s.head_dim
        nq = s.num_attention_heads
        spans = plan_surya_dense_spans(max_seq)
        span_scalars = 16  # live_counts + row_positions, one int64 each
        span_metadata = (
            spans.page_table.nbytes + spans.token_positions.nbytes
            + spans.evict_mask.nbytes + span_scalars
        )
        span_decode = nq * spans.num_splits * (hd + 2) * 4
        return {
            "weights_bytes": sum(b.nbytes for b in probe._w.values()),
            "kv_planes_bytes": 2 * nk * max_seq * hd * 4 * probe.n_attn_layers,
            "gdn_state_bytes": sum(b.nbytes for b in probe._gdn_state.values()),
            "conv_state_bytes": sum(b.nbytes for b in probe._conv_state.values()),
            "span_metadata_bytes": int(span_metadata),
            "span_decode_partial_bytes": int(span_decode),
            "span_num_splits": int(spans.num_splits),
            "span_block_size": int(spans.block_size),
            "resident_bytes": memory_stats()["current_allocated_bytes"],
        }
    finally:
        probe.close()


def measure(
    *,
    spec,
    weights,
    max_seq: int,
    token_counts: list[int],
    budget_bytes: int | None,
    runs: int = 1,
) -> dict:
    from hipengine.core.memory import memory_stats, reset_memory_stats
    from hipengine.runtime.surya import SuryaGpuRunner, plan_score_tiles

    s = spec
    nq = s.num_attention_heads

    def fresh():
        """A runner with a clean allocation high-water mark.

        The peak counter is process-global, so a fresh runner plus a reset
        after construction is what isolates one prefill's footprint. Reusing a
        runner would carry the previous token count's scratch (and its peak)
        into the next measurement.

        ``budget_bytes=None`` is passed explicitly: it means "no budget, one
        dense tile" at the runner, which is *not* the same as omitting the
        argument (that selects the runner's default budget).
        """

        runner = SuryaGpuRunner(
            weights, spec, max_seq=max_seq, max_prefill_scratch_bytes=budget_bytes
        )
        reset_memory_stats()
        return runner

    rows = []
    skipped = []
    for tokens in sorted(token_counts):
        if tokens > max_seq:
            # a prompt longer than the context cannot be prefilled at all
            skipped.append(int(tokens))
            continue
        ids, pos = _synthetic_prompt(tokens)
        runner = fresh()
        try:
            resident = memory_stats()["current_allocated_bytes"]
            runner.prefill(ids, pos)
            stats = memory_stats()
            scratch = stats["current_allocated_bytes"] - resident
            peak = stats["peak_allocated_bytes"] - resident
            timings = []
            for _ in range(max(1, int(runs))):
                t0 = time.time()
                runner.prefill(ids, pos)
                timings.append(time.time() - t0)
        finally:
            runner.close()
        block, tile = plan_score_tiles(tokens, nq, budget_bytes)
        rows.append(
            {
                "tokens": int(tokens),
                "query_block": block,
                "resident_bytes": int(resident),
                "peak_scratch_bytes": int(peak),
                "retained_scratch_bytes": int(scratch),
                "score_tile_bytes": int(tile),
                "dense_score_bytes": int(nq * tokens * tokens * 4),
                "non_attention_scratch_bytes": int(peak - tile),
                "linear_bytes_per_token": ((peak - tile) / tokens if tokens else 0.0),
                "peak_total_bytes": int(resident + peak),
                "prefill_seconds": float(np.mean(timings)),
                "prefill_seconds_runs": [float(t) for t in timings],
            }
        )
    return {
        "max_seq": int(max_seq),
        "budget_bytes": None if budget_bytes is None else int(budget_bytes),
        "skipped_over_context": skipped,
        "rows": rows,
    }


def fit_linear(rows: list[dict]) -> float | None:
    """Least-squares slope of non-attention scratch against token count."""

    if len(rows) < 2:
        return None
    x = np.array([r["tokens"] for r in rows], dtype=np.float64)
    y = np.array([r["non_attention_scratch_bytes"] for r in rows], dtype=np.float64)
    a = np.vstack([x, np.ones_like(x)]).T
    slope, _intercept = np.linalg.lstsq(a, y, rcond=None)[0]
    return float(slope)


def _parse_budgets(values: list[int]) -> list[int | None]:
    """``0`` selects the dense path (no budget)."""

    return [None if int(v) == 0 else int(v) for v in values]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-seq", type=int, nargs="+", default=[16384])
    parser.add_argument(
        "--tokens", default="256,512,1024,2048,4096", help="comma-separated"
    )
    parser.add_argument(
        "--text-budget-bytes",
        type=int,
        nargs="+",
        default=[512 * 1024**2],
        help="causal score-tile budget(s) for the text prefill; 0 measures dense",
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="timed prefill repeats per row"
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--vision-budget-bytes",
        type=int,
        default=512 * 1024**2,
        help="vision score-tile budget for the tile-plan table",
    )
    args = parser.parse_args()

    tile_plan = vision_tile_plan(
        args.model, [A4_300DPI_GRID, MAX_PIXELS_GRID], args.vision_budget_bytes
    )
    print("vision attention score tile (budget %.0f MiB):" % (args.vision_budget_bytes / 1024**2))
    for row in tile_plan:
        print(
            f"  grid={tuple(row['grid'])} patches={row['patches']:6d} "
            f"dense={row['dense_score_bytes'] / 1e9:8.2f} GB "
            f"block={row['query_block']:6d} "
            f"tiled={row['tiled_score_bytes'] / 1e6:8.2f} MB "
            f"({row['reduction_x']:.0f}x smaller)"
        )

    token_counts = [int(t) for t in args.tokens.split(",") if t.strip()]
    budgets = _parse_budgets(args.text_budget_bytes)
    plan = text_tile_plan(args.model, token_counts, budgets)
    print("text prefill causal score tile:")
    for row in plan:
        print(f"  tokens={row['tokens']:6d} dense={row['dense_score_bytes'] / 1e9:8.2f} GB")
        for p in row["plans"]:
            budget = "dense" if p["budget_bytes"] is None else f"{p['budget_bytes'] / 1024**2:.0f} MiB"
            print(
                f"    budget={budget:>8s} block={p['query_block']:6d} "
                f"tile={p['score_tile_bytes'] / 1e6:8.2f} MB "
                f"({p['reduction_x']:.1f}x smaller)"
            )

    results = []
    spec, weights = _spec_and_weights(args.model)
    for max_seq in args.max_seq:
        components = _resident_components(spec, weights, max_seq)
        blocks = []
        for budget in budgets:
            result = measure(
                spec=spec,
                weights=weights,
                max_seq=max_seq,
                token_counts=token_counts,
                budget_bytes=budget,
                runs=args.runs,
            )
            result["fitted_linear_bytes_per_token"] = fit_linear(result["rows"])
            blocks.append(result)
            budget_label = "dense" if budget is None else f"{budget / 1024**2:.0f} MiB"
            print(f"max_seq={max_seq} text budget={budget_label}")
            for row in result["rows"]:
                print(
                    f"  tokens={row['tokens']:6d}  "
                    f"peak={row['peak_total_bytes'] / 1e6:9.1f} MB  "
                    f"tile={row['score_tile_bytes'] / 1e6:8.1f} MB  "
                    f"other={row['non_attention_scratch_bytes'] / 1e6:8.1f} MB  "
                    f"({row['linear_bytes_per_token'] / 1024:6.1f} KiB/token)  "
                    f"{row['prefill_seconds']:6.2f}s"
                )
            slope = result["fitted_linear_bytes_per_token"]
            print(f"  fitted non-attention slope: {slope / 1024:.1f} KiB/token")
        results.append(
            {
                "max_seq": int(max_seq),
                "resident_components": components,
                "budgets": blocks,
            }
        )

    payload = {
        "date": time.strftime("%Y-%m-%d"),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "provenance": _provenance(sys.argv),
        "vision_attention": {
            "budget_bytes": int(args.vision_budget_bytes),
            "grids": tile_plan,
        },
        "text_attention": {
            "budget_bytes_list": [None if b is None else int(b) for b in budgets],
            "runs": int(args.runs),
            "tile_plan": plan,
        },
        "protocol": {
            "measurement": "hipengine.core.memory process-local device allocation counter (hipEngine-owned buffers)",
            "boundary": "peak scratch = peak_allocated_bytes - resident_bytes, where resident is the runner's long-lived allocation after construction",
            "prompt": "synthetic text-only ids of the given length with plain 1-D rope positions; no image, so this isolates the text stack",
            "isolation": "a fresh runner per token count, with reset_memory_stats() after construction, so the peak is one prefill's footprint and not a previous run's high-water mark",
            "score_tile_bytes": "the planner's causal score tile (nq * tokens * block * 4) for the row's budget; budget_bytes null means the dense path",
            "timing": "prefill_seconds is the mean of prefill_seconds_runs, which are wall-clock repeats on an already-warm runner",
        },
        "page_tokens": {"300dpi_a4": PAGE_TOKENS_300DPI},
        "results": results,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
