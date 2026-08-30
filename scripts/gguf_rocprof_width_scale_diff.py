#!/usr/bin/env python3
"""Diff two kernel traces taken at different wave widths and name the kernels that scale.

Why this exists: decode per-step cost on the W7900 GGUF path rises from ~33.2 ms at width 1 to
~50.4 ms at width 2, where a weight-bound batched step should stay near 33 ms, so some decode work
is per-row.
Finding it from `rocprofv3 --kernel-trace` output needs a per-kernel comparison, because the two
possible signatures look different in the data:

  * ``per_row_launches`` - the kernel is launched once per row, so launch count scales with width;
  * ``per_row_inside_launch`` - launch count is flat but each launch gets longer with width.

Kernels that appear in only one arm are listed rather than dropped: an arm that silently started or
stopped running a kernel family (for example an MTP verifier that engages only at width >= 2) would
otherwise be misread as row scaling. Loading and CSV-column handling come from
``gguf_kernel_family_rocprof_summary.py`` so trace parsing is not duplicated.

Usage:
    .venv/bin/python scripts/gguf_rocprof_width_scale_diff.py \
        --base-dir /tmp/he-rows/trace-c1 --candidate-dir /tmp/he-rows/trace-c2 \
        --rows-base 1 --rows-candidate 2 --json /tmp/he-rows/scale-diff.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SUMMARY_SCRIPT = REPO / "scripts" / "gguf_kernel_family_rocprof_summary.py"

# Ratio at or beyond this fraction of the width ratio counts as scaling with rows.
SCALL_FRACTION = 0.75
FLAT_CEILING = 1.2


def _load_summary_module():
    spec = importlib.util.spec_from_file_location("roctx_summary_for_scale_diff", SUMMARY_SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - repository layout error
        raise RuntimeError(f"cannot load {SUMMARY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rows_for_dir(trace_dir: Path, region: re.Pattern | None) -> list[dict]:
    if not trace_dir.is_dir():
        raise FileNotFoundError(f"trace directory does not exist: {trace_dir}")
    summary = _load_summary_module()
    csv_files = sorted(trace_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"no rocprofv3 CSV under {trace_dir}")
    rows: list[dict] = []
    for csv_path in csv_files:
        rows.extend(summary.load_kernels(csv_path, region))
    if not rows:
        raise FileNotFoundError(f"no kernel rows parsed from {trace_dir} ({len(csv_files)} CSVs)")
    return rows


def _per_kernel(rows: list[dict]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"launches": 0, "total_ns": 0.0, "durations": [], "grids": set()}
    )
    for row in rows:
        entry = grouped[row["name"]]
        entry["launches"] += 1
        duration = row.get("duration_ns")
        if duration is not None:
            entry["total_ns"] += float(duration)
            entry["durations"].append(float(duration))
        grid = tuple(row.get(f"grid_{axis}") for axis in "xyz")
        if any(value is not None for value in grid):
            entry["grids"].add(grid)
    out: dict[str, dict[str, Any]] = {}
    for name, entry in grouped.items():
        durations = entry["durations"]
        out[name] = {
            "launches": entry["launches"],
            "total_ms": round(entry["total_ns"] / 1e6, 3),
            "mean_us": round(statistics.mean(durations) / 1e3, 2) if durations else None,
            "grids": sorted(str(grid) for grid in entry["grids"]),
        }
    return out


def _ratio(candidate: float | int | None, base: float | int | None) -> float | None:
    if base is None or candidate is None or base == 0:
        return None
    return round(float(candidate) / float(base), 3)


def _classify(
    launch_ratio: float | None, mean_ratio: float | None, expected_scale: float
) -> str:
    threshold = expected_scale * SCALL_FRACTION
    if launch_ratio is not None and launch_ratio >= threshold:
        return "per_row_launches"
    if mean_ratio is not None and mean_ratio >= threshold:
        return "per_row_inside_launch"
    if (launch_ratio is None or launch_ratio < FLAT_CEILING) and (
        mean_ratio is None or mean_ratio < FLAT_CEILING
    ):
        return "flat"
    return "partial"


def diff_dirs(
    base_dir: Path,
    candidate_dir: Path,
    *,
    rows_base: int,
    rows_candidate: int,
    region: str | None = None,
    min_total_ms: float = 0.0,
) -> dict[str, Any]:
    """Compare per-kernel launch counts and durations between two traced widths."""
    if rows_base <= 0 or rows_candidate <= 0:
        raise ValueError(f"row counts must be positive, got {rows_base} and {rows_candidate}")
    if rows_candidate < rows_base:
        raise ValueError(
            "candidate must be the wider run so ratios read as growth; swap the arguments "
            f"(rows_base={rows_base}, rows_candidate={rows_candidate})"
        )
    pattern = re.compile(region) if region else None
    base = _per_kernel(_rows_for_dir(Path(base_dir), pattern))
    candidate = _per_kernel(_rows_for_dir(Path(candidate_dir), pattern))
    expected_scale = rows_candidate / rows_base

    kernels: dict[str, dict[str, Any]] = {}
    for name in sorted(set(base) & set(candidate)):
        b, c = base[name], candidate[name]
        if b["total_ms"] < min_total_ms and c["total_ms"] < min_total_ms:
            continue
        launch_ratio = _ratio(c["launches"], b["launches"])
        mean_ratio = _ratio(c["mean_us"], b["mean_us"])
        kernels[name] = {
            "launches_base": b["launches"],
            "launches_candidate": c["launches"],
            "launch_ratio": launch_ratio,
            "mean_us_base": b["mean_us"],
            "mean_us_candidate": c["mean_us"],
            "mean_ratio": mean_ratio,
            "total_ms_base": b["total_ms"],
            "total_ms_candidate": c["total_ms"],
            "total_ratio": _ratio(c["total_ms"], b["total_ms"]),
            "grids_base": b["grids"],
            "grids_candidate": c["grids"],
            "grid_changed": sorted(b["grids"]) != sorted(c["grids"]),
            "classification": _classify(launch_ratio, mean_ratio, expected_scale),
        }
    counts: dict[str, int] = defaultdict(int)
    for entry in kernels.values():
        counts[entry["classification"]] += 1
    return {
        "schema": "gguf_rocprof_width_scale_diff.v1",
        "base_dir": str(base_dir),
        "candidate_dir": str(candidate_dir),
        "rows_base": rows_base,
        "rows_candidate": rows_candidate,
        "expected_scale": round(expected_scale, 3),
        "flat_ceiling": FLAT_CEILING,
        "scale_fraction": SCALL_FRACTION,
        "total_ms_base": round(sum(item["total_ms"] for item in base.values()), 3),
        "total_ms_candidate": round(sum(item["total_ms"] for item in candidate.values()), 3),
        "launches_base": sum(item["launches"] for item in base.values()),
        "launches_candidate": sum(item["launches"] for item in candidate.values()),
        "classifications": dict(sorted(counts.items())),
        "only_in_base": sorted(set(base) - set(candidate)),
        "only_in_candidate": sorted(set(candidate) - set(base)),
        "kernels": kernels,
        "ranked_by_candidate_ms": [entry_key for entry_key in _rank_names(kernels)],
    }


def _rank_names(kernels: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(kernels, key=lambda name: -kernels[name]["total_ms_candidate"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-dir", type=Path, required=True, help="narrower (base) trace dir")
    parser.add_argument("--candidate-dir", type=Path, required=True, help="wider trace dir")
    parser.add_argument("--rows-base", type=int, default=1)
    parser.add_argument("--rows-candidate", type=int, default=2)
    parser.add_argument("--region", default=None, help="regex filter on kernel name")
    parser.add_argument("--min-total-ms", type=float, default=0.0)
    parser.add_argument("--top", type=int, default=18)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    report = diff_dirs(
        args.base_dir,
        args.candidate_dir,
        rows_base=args.rows_base,
        rows_candidate=args.rows_candidate,
        region=args.region,
        min_total_ms=args.min_total_ms,
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        f"rows {report['rows_base']} -> {report['rows_candidate']} "
        f"(expected scale {report['expected_scale']}), "
        f"total {report['total_ms_base']} ms -> {report['total_ms_candidate']} ms, "
        f"launches {report['launches_base']} -> {report['launches_candidate']}"
    )
    print(f"classifications: {report['classifications']}")
    if report["only_in_base"] or report["only_in_candidate"]:
        print(
            f"only_in_base={len(report['only_in_base'])} "
            f"only_in_candidate={len(report['only_in_candidate'])} "
            "(kernels present in one arm only - not row scaling)"
        )
    header = (
        f"{'kernel':<58} {'launches':>16} {'mean us':>18} "
        f"{'total ms':>18}  classification"
    )
    print(header)
    for name in report["ranked_by_candidate_ms"][: args.top]:
        entry = report["kernels"][name]
        print(
            f"{name[:58]:<58} "
            f"{str(entry['launches_base']) + '->' + str(entry['launches_candidate']):>16} "
            f"{str(entry['mean_us_base']) + '->' + str(entry['mean_us_candidate']):>18} "
            f"{str(entry['total_ms_base']) + '->' + str(entry['total_ms_candidate']):>18}  "
            f"{entry['classification']}"
            + ("  [grid changed]" if entry["grid_changed"] else "")
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
