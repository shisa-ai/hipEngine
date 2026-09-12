"""Wall-clock cost of the Surya vision-attention query-row tiling.

The tiling exists to make page-scale vision grids admissible at all: the dense
score matrix is ``vision_num_heads * n^2 * 4`` bytes, which is 56.5 GB for a
300-DPI A4 page (34320 patches) and 206 GB at the checkpoint's ``SURYA_MAX_PIXELS``
ceiling (65536). Those grids cannot run densely, so they have no dense baseline.

This measures the grids that *can* run both ways, so the tiling's cost is known
rather than assumed: ``None`` disables the budget and runs one dense tile, and a
byte budget selects the largest query block whose tile fits. One runner per
budget, warm-up call discarded, median of ``--reps``.

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

# (fixture, patch count). 1024 patches is a 512x512 page; 4096 is the 1024x1024
# Japanese page at grid 1x64x64. Both are admissible densely.
CASES = (("page_dense.png", 1024), ("page_ja.png", 4096))

# None = dense (one tile). The rest are tile budgets, smallest last so the
# printed curve reads largest-tile-first.
BUDGETS: tuple[int | None, ...] = (None, 512 * MIB, 64 * MIB, 8 * MIB)


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


def _timings(page: str, budget: int | None, reps: int) -> dict[str, object]:
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
        block = int(runner.vision_block([grid]))
        scratch = int(runner.vision_scratch_bytes([grid]))
    finally:
        runner.close()

    dense_bytes = int(spec.vision_num_heads) * patches * patches * 4
    return {
        "page": page,
        "grid": [int(v) for v in grid],
        "patches": patches,
        "budget_bytes": budget,
        "query_block": block,
        "tiles": -(-patches // block),
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
        "--out", type=Path,
        default=Path("benchmarks/results/2026-09-12-gfx1151-surya-vision-tiling-cost.json"),
    )
    args = parser.parse_args()

    import sys

    rows: list[dict[str, object]] = []
    for page, expected in CASES:
        if not (FIXTURES / page).exists():
            raise SystemExit(f"missing {FIXTURES / page}")
        for budget in BUDGETS:
            row = _timings(page, budget, args.reps)
            if row["patches"] != expected:
                raise SystemExit(
                    f"{page} produced {row['patches']} patches, expected {expected}"
                )
            rows.append(row)
            label = "dense" if budget is None else f"{budget // MIB} MiB"
            print(
                f"{page:18s} {label:>8s} block={row['query_block']:5d} "
                f"tiles={row['tiles']:4d} scratch={row['scratch_mib']:7.1f} MiB "
                f"median={row['median_ms']:9.2f} ms"
            )

    # Dense is the baseline wherever it is runnable, so the cost of a tile
    # budget is reported against it per page rather than across pages.
    by_page: dict[str, dict[str, object]] = {}
    for row in rows:
        page = str(row["page"])
        if row["budget_bytes"] is None:
            by_page[page] = {"dense_median_ms": row["median_ms"]}
    for row in rows:
        dense = by_page.get(str(row["page"]), {}).get("dense_median_ms")
        if dense:
            row["vs_dense"] = round(float(dense) / float(row["median_ms"]), 3)

    artifact = {
        "date": time.strftime("%Y-%m-%d", time.gmtime()),
        "model": MODEL,
        "provenance": _provenance(sys.argv),
        "protocol": (
            "one runner per budget; one discarded warm-up vision_forward; "
            "median of --reps wall-clock timings around vision_forward only "
            "(preprocessing and scratch allocation excluded)"
        ),
        "boundary": (
            "None disables the tile budget and runs one dense tile; a byte "
            "budget selects the largest query block whose "
            "vision_num_heads * n * block * 4 tile fits"
        ),
        "note": (
            "Grids that cannot run densely have no dense baseline: the dense "
            "score matrix is 56.5 GB at 34320 patches and 206 GB at 65536, "
            "which is why the tiling exists. These rows are the grids that run "
            "both ways."
        ),
        "reps": args.reps,
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
