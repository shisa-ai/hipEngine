"""Wall-clock cost of the Surya vision-attention query-row tiling.

The tiling exists to make page-scale vision grids admissible at all: the dense
score matrix is ``vision_num_heads * n^2 * 4`` bytes, which is 56.5 GB for a
300-DPI A4 page (34320 patches) and 206 GB at the checkpoint's ``SURYA_MAX_PIXELS``
ceiling (65536). Those grids cannot run densely, so they have no dense baseline.

This measures the grids that *can* run both ways, so the tiling's cost is known
rather than assumed: ``None`` disables the budget and runs one dense tile, and a
byte budget selects the largest query block whose tile fits *and* that the
planner's shape envelope admits. ``--blocks`` measures named shapes directly,
lifting the envelope so shapes the planner would not choose are still on the
curve; byte-budget rows always keep it, because those rows are the production
plan. One runner per shape, warm-up call discarded, median of ``--reps``.

Run this with no other GPU job. The sweep is sequential per shape, so a
concurrent job inflates whichever shape happens to be running, and it inflates
every sample of that row -- the row's own spread does not show it. ``--passes
2`` re-runs the plans in reverse order, which is the check for that.

When varying the shape from outside this script, lift the planner's envelope
(``hipengine.runtime.surya.SHAPE_TILE_ROWS`` / ``SHAPE_TILE_DIVISOR`` /
``SHAPE_TILE_MULTIPLE``) or the production cap silently overrides the requested
block. A re-check of 4096-patch shapes that forgot to do that measured a
128-row tile for every shape it asked for and reported a flat, fast curve --
which reads like "the budget-derived shape is fine" rather than like a bug.
``--blocks`` does the lift, and every row records ``envelope_lifted`` so the
artifact says which mode it was measured in.

Usage::

    python3 scripts/surya_vision_tiling_cost.py --out benchmarks/results/....json
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

FIXTURES = Path("tests/fixtures/surya")
MODEL = "datalab-to/surya-ocr-2"
MIB = 1024 * 1024

# A second pass in reverse order is compared against this: a relative tolerance
# for the big rows and an absolute floor for the 27-30 ms ones, where 5% is
# 1.4 ms of ordinary run-to-run noise and would flag every pass.
PASS_TOLERANCE = 0.10
PASS_TOLERANCE_MS = 2.0


def _production_envelope() -> tuple[int, int, int]:
    """The planner's shipped shape envelope, read before anything lifts it."""

    from hipengine.runtime.surya import (
        SHAPE_TILE_DIVISOR,
        SHAPE_TILE_MULTIPLE,
        SHAPE_TILE_ROWS,
    )

    return (
        int(SHAPE_TILE_ROWS),
        int(SHAPE_TILE_DIVISOR),
        int(SHAPE_TILE_MULTIPLE),
    )


_ENVELOPE = _production_envelope()

# (fixture, patch count). 1024 patches is a 512x512 page; 4096 is the 1024x1024
# Japanese page at grid 1x64x64; 6400 is the 1024x1600 long page at 1x100x64;
# 34320 is the 300-DPI A4 page at 220x156, the grid the memory plan is written
# against and the only one of the four that cannot run densely. The first three
# give a dense baseline and span the range where the optimal block moves.
CASES = (
    ("page_small.png", 256),
    ("page_dense.png", 1024),
    ("page_ja.png", 4096),
    ("page_long.png", 6400),
    ("page_a4.png", 34320),
)

# A page-scale grid costs ~30-60 s per forward, so the default page set is the
# three that have a dense baseline; the A4 grid is opt-in.
DEFAULT_PAGES: tuple[str, ...] = ("page_dense.png", "page_ja.png", "page_long.png")

# None = dense (one tile). The rest are tile budgets, smallest last so the
# printed curve reads largest-tile-first.
BUDGETS: tuple[int | None, ...] = (None, 512 * MIB, 64 * MIB, 8 * MIB)

# Explicit query-block sweep. The budget is an awkward knob for shape work:
# ``block = budget // (heads * rows * 4)``, so the interesting shapes (a block
# that divides the grid, a block one row either side of a divisor) need the
# budget solved backwards. ``--blocks`` does that, so the same runner and the
# same timed region are used for every shape.
SWEEP_BLOCKS: tuple[int, ...] = (
    16, 21, 22, 32, 42, 48, 56, 64, 72, 80, 85, 96, 100, 112, 128, 144, 160,
    170, 171, 192, 200, 256, 341, 342, 512, 683, 1024, 1365, 2048, 2730, 2731,
    4096,
)


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


def _shape_envelope(enable: bool) -> None:
    """Turn the planner's shape envelope on or off for the next measurement.

    Byte-budget rows are the production plan, so they keep the envelope and the
    wavefront rounding. Shape rows measure the landscape, so they lift both: a
    width the planner would never choose still has to be measured to know what
    it costs, and the envelope's own constants were chosen from widths outside
    it — including the non-multiples of 32 that justify the rounding.
    """

    import hipengine.runtime.surya as surya

    if enable:
        (
            surya.SHAPE_TILE_ROWS,
            surya.SHAPE_TILE_DIVISOR,
            surya.SHAPE_TILE_MULTIPLE,
        ) = _ENVELOPE
    else:
        surya.SHAPE_TILE_ROWS = 1 << 40
        surya.SHAPE_TILE_DIVISOR = 1 << 40
        surya.SHAPE_TILE_MULTIPLE = 1


def _timings(page: str, budget: int | None, reps: int,
             block: int | None = None) -> dict[str, object]:
    """Time ``vision_forward`` once per repeat under one tile shape.

    ``block`` names a query-block shape and solves the budget backwards
    (``heads * rows * block * 4``); ``budget`` is the byte form used when the
    shape is not what is being varied.
    """

    from PIL import Image

    from hipengine.loading.surya import (
        load_surya_spec,
        load_surya_weights,
        preprocess_image_surya,
        resolve_surya_path,
    )
    from hipengine.runtime.surya import SuryaGpuRunner

    model_dir = resolve_surya_path(MODEL)
    spec = load_surya_spec(model_dir)
    weights = load_surya_weights(model_dir)
    pixel_rows, grid = preprocess_image_surya(
        Image.open(FIXTURES / page).convert("RGB")
    )
    patches = int(grid[1]) * int(grid[2])

    if block is not None:
        if not 1 <= int(block) <= patches:
            raise SystemExit(
                f"block {block} is outside 1..{patches} for {page}"
            )
        budget = int(spec.vision_num_heads) * patches * int(block) * 4

    runner = SuryaGpuRunner(
        weights, spec, max_seq=4096, max_vision_scratch_bytes=budget
    )
    try:
        runner.vision_forward(pixel_rows, [grid])  # warm: JIT + scratch alloc
        samples: list[float] = []
        for _ in range(reps):
            start = time.perf_counter()
            runner.vision_forward(pixel_rows, [grid])
            samples.append(time.perf_counter() - start)
        planned = int(runner.vision_block([grid]))
        scratch = int(runner.vision_scratch_bytes([grid]))
    finally:
        runner.close()

    if block is not None and planned != int(block):
        raise SystemExit(
            f"asked for block {block} on {page} but the planner returned "
            f"{planned}"
        )

    dense_bytes = int(spec.vision_num_heads) * patches * patches * 4
    tiles = -(-patches // planned)
    return {
        "page": page,
        "grid": [int(v) for v in grid],
        "patches": patches,
        "budget_bytes": budget,
        "query_block": planned,
        "tiles": tiles,
        "tail_rows": patches - (tiles - 1) * planned,
        "even": patches % planned == 0,
        "scratch_bytes": scratch,
        "scratch_mib": round(scratch / MIB, 1),
        "dense_score_bytes": dense_bytes,
        "median_ms": round(statistics.median(samples) * 1000.0, 2),
        "min_ms": round(min(samples) * 1000.0, 2),
        "max_ms": round(max(samples) * 1000.0, 2),
        "samples_ms": [round(value * 1000.0, 2) for value in samples],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument(
        "--blocks", default=None,
        help="comma-separated explicit query-block shapes; the budget is "
             "solved backwards for each, so this is the shape sweep",
    )
    parser.add_argument(
        "--pages", default=None,
        help="comma-separated fixture names; default is the three grids that "
             "also have a dense baseline",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("benchmarks/results/2026-09-12-gfx1151-surya-vision-tiling-cost.json"),
    )
    parser.add_argument(
        "--with-budgets", action="store_true",
        help="with --blocks, run the byte-budget plans first so one artifact "
             "holds both the default path and the shape sweep",
    )
    parser.add_argument(
        "--budgets", default=None,
        help="comma-separated tile budgets in MiB for the byte-budget plans, "
             "or 'dense' for the unbounded tile; default is the full set "
             "(dense, 512, 64, 8 MiB). A grid that cannot run densely "
             "(page_a4.png) must not include 'dense', and a budget far below "
             "the grid makes its row very slow, so both ends are selectable",
    )
    parser.add_argument(
        "--passes", type=int, default=1, choices=(1, 2),
        help="run the plans twice, the second time in reverse order, and "
             "record pass_ratio per row; the only check that catches a "
             "steady disturbance from a concurrent GPU job",
    )
    args = parser.parse_args()

    import sys


    selected = (
        tuple(value.strip() for value in args.pages.split(",") if value.strip())
        if args.pages else DEFAULT_PAGES
    )
    known = {page for page, _ in CASES}
    unknown = [page for page in selected if page not in known]
    if unknown:
        raise SystemExit(f"unknown page(s) {unknown}; known: {sorted(known)}")
    blocks = (
        [int(value) for value in args.blocks.split(",") if value.strip()]
        if args.blocks else None
    )
    if args.budgets is None:
        budget_plans: list[dict[str, object]] = [
            {"budget": budget, "block": None} for budget in BUDGETS
        ]
    else:
        budget_plans = []
        for value in args.budgets.split(","):
            value = value.strip().lower()
            if not value:
                continue
            if value in ("dense", "none"):
                budget_plans.append({"budget": None, "block": None})
            else:
                budget_plans.append(
                    {"budget": int(float(value) * MIB), "block": None}
                )
        if not budget_plans:
            raise SystemExit("--budgets selected no plans")
    rows: list[dict[str, object]] = []
    keys: list[tuple[str, int | None, int | None]] = []
    second: dict[tuple[str, int | None, int | None], dict[str, object]] = {}
    for page, expected in CASES:
        if page not in selected:
            continue
        if not (FIXTURES / page).exists():
            raise SystemExit(f"missing {FIXTURES / page}")
        if blocks is None:
            plans: list[dict[str, object]] = list(budget_plans)
        else:
            plans = [
                {"budget": None, "block": block}
                for block in blocks
                if block <= expected  # a block larger than the grid is dense
            ]
            if args.with_budgets:
                plans = [dict(plan) for plan in budget_plans] + plans
        for pass_index in range(args.passes):
            # the second pass reverses the order, so a steady disturbance from
            # a concurrent job cannot land on the same shape twice
            ordered = plans if pass_index == 0 else list(reversed(plans))
            for plan in ordered:
                budget, block = plan["budget"], plan["block"]
                # budget rows are the production plan; named shapes lift the
                # envelope so the whole landscape is on the curve
                _shape_envelope(block is None)
                row = _timings(page, budget, args.reps, block=block)
                row["envelope_lifted"] = block is not None
                if row["patches"] != expected:
                    raise SystemExit(
                        f"{page} produced {row['patches']} patches, expected {expected}"
                    )
                if pass_index == 1:
                    second[(page, block, budget)] = row
                    continue
                rows.append(row)
                keys.append((page, block, budget))
                label = "dense" if budget is None and block is None else (
                    f"block={block}" if block is not None else f"{budget // MIB} MiB"
                )
                print(
                    f"{page:18s} {label:>12s} block={row['query_block']:5d} "
                    f"tiles={row['tiles']:4d} tail={row['tail_rows']:5d} "
                    f"scratch={row['scratch_mib']:7.1f} MiB "
                    f"median={row['median_ms']:9.2f} ms"
                )

    for row, key in zip(rows, keys):
        twin = second.pop(key, None)
        if twin is None:
            continue
        row["median_ms_pass2"] = twin["median_ms"]
        row["samples_ms_pass2"] = twin["samples_ms"]
        row["pass_ratio"] = round(
            float(row["median_ms"]) / float(twin["median_ms"]), 3
        )
    if second:
        raise SystemExit(
            f"{len(second)} pass-2 rows did not match a pass-1 row; "
            "the plan list is not stable across passes"
        )
    noisy = [
        row for row in rows
        if abs(float(row.get("median_ms", 0.0))
               - float(row.get("median_ms_pass2", row.get("median_ms", 0.0))))
        > max(PASS_TOLERANCE * float(row.get("median_ms", 0.0)), PASS_TOLERANCE_MS)
    ]
    for row in noisy:
        print(
            f"WARNING pass disagreement {row['page']} "
            f"block={row['query_block']} budget={row['budget_bytes']}: "
            f"{row['median_ms']} vs {row['median_ms_pass2']} ms "
            f"({row['pass_ratio']}x) -- re-measure on a quiet GPU"
        )
    if args.passes > 1:
        print(
            f"pass check: {len(rows) - len(noisy)}/{len(rows)} rows within "
            f"max({PASS_TOLERANCE:.0%}, {PASS_TOLERANCE_MS:g} ms)"
        )

    # Dense is the baseline wherever it is runnable, so the cost of a tile
    # budget is reported against it per page rather than across pages.
    by_page: dict[str, dict[str, object]] = {}
    for row in rows:
        page = str(row["page"])
        if row["budget_bytes"] is None and row["query_block"] == row["patches"]:
            by_page[page] = {"dense_median_ms": row["median_ms"]}
    for row in rows:
        dense = by_page.get(str(row["page"]), {}).get("dense_median_ms")
        if dense:
            row["vs_dense"] = round(float(dense) / float(row["median_ms"]), 3)

    # Best measured shape per page, so a rule can be scored against the shape
    # the sweep actually found rather than against the dense tile alone.
    best: dict[str, dict[str, object]] = {}
    for row in rows:
        page = str(row["page"])
        if page not in best or row["median_ms"] < best[page]["median_ms"]:
            best[page] = {"query_block": row["query_block"],
                          "median_ms": row["median_ms"]}
    for row in rows:
        found = best.get(str(row["page"]))
        if found:
            row["vs_best_shape"] = round(
                float(found["median_ms"]) / float(row["median_ms"]), 3
            )

    artifact = {
        "date": time.strftime("%Y-%m-%d", time.gmtime()),
        "model": MODEL,
        "provenance": _provenance(sys.argv),
        "protocol": (
            "one runner per shape; one discarded warm-up vision_forward; "
            "median of --reps wall-clock timings around vision_forward only "
            "(preprocessing and scratch allocation excluded)"
        ),
        "boundary": (
            "None disables the tile budget and runs one dense tile; a byte "
            "budget selects the largest query block whose "
            "vision_num_heads * n * block * 4 tile fits; --blocks names the "
            "block directly and solves the budget backwards so the shapes can "
            "be swept without the budget's rounding"
        ),
        "shape_envelope": (
            "byte-budget rows are the production plan (envelope and wavefront "
            "rounding respected); --blocks rows lift both so shapes the "
            "planner would not choose are still measured"
        ),
        "note": (
            "Grids that cannot run densely have no dense baseline: the dense "
            "score matrix is 56.5 GB at 34320 patches and 206 GB at 65536, "
            "which is why the tiling exists. These rows are the grids that run "
            "both ways."
        ),
        "reps": args.reps,
        "passes": args.passes,
        "pass_check": (
            "no second pass: run with --passes 2 to detect a steady disturbance "
            "from a concurrent GPU job, which a row's own spread cannot see"
            if args.passes < 2 else
            f"second pass in reverse order; {len(rows) - len(noisy)}/{len(rows)} "
            f"rows agree with pass 1 within max({PASS_TOLERANCE:.0%}, "
            f"{PASS_TOLERANCE_MS:g} ms), pass_ratio = pass1/pass2 per row"
        ),
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
