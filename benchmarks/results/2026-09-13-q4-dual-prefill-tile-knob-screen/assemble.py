#!/usr/bin/env python3
"""Rebuild the reviewed tile screen from unchanged original measurement files.

No source-level instruction count or logical byte rate is a hardware counter.
The original resource_table.json contains obsolete derived occupancy fields;
only its compiler counts are consumed and the register ceiling is recalculated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.gguf_prefill_kernel_resources import derived_occupancy

DUAL_TEMPLATE_RE = re.compile(
    r"dual_wmma_prefill_silu_bf16_kernelILb0ELb0ELi(\d+)ELi(\d+)EE"
)
OWNER_BY_ROWS = {32: "row32", 48: "row48", 64: "row64", 128: "row128", 256: "parent"}
GEOMETRIES = {
    "shared_b": (3, 4, 4),
    "shared_b2w4": (2, 4, 4),
    "shared_b2w2": (2, 2, 4),
    "shared_b3w8r3": (3, 8, 3),
    "default": (3, 1, 4),
    "lowvgpr": (1, 1, 2),
    "lowvgpr48": (1, 1, 3),
}


def dual_ladder(resources: dict) -> list[dict]:
    rows = []
    for entry in resources["kernels"]:
        match = DUAL_TEMPLATE_RE.search(entry["kernel"])
        if not match:
            continue
        row_tiles, active_waves = map(int, match.groups())
        token_rows = 16 * row_tiles * active_waves
        row = {key: entry[key] for key in (
            "kernel", "vgpr", "sgpr", "lds_bytes", "private_bytes",
        )}
        row.update(
            owner=OWNER_BY_ROWS[token_rows], rows_per_block=token_rows,
            columns_per_block=32, row_tiles_per_wave=row_tiles,
            active_waves_per_block=active_waves, launched_waves_per_block=4,
            threads_per_block=128,
        )
        row.update(derived_occupancy(entry["vgpr"], entry["lds_bytes"], 4))
        rows.append(row)
    if {row["owner"] for row in rows} != set(OWNER_BY_ROWS.values()) or len(rows) != 5:
        raise ValueError("expected five distinct dual geometry owners")
    return sorted(rows, key=lambda row: row["rows_per_block"])


def row_tile_sweep(sweep: dict, ladder: list[dict]) -> list[dict]:
    cells = []
    columns = sweep["out_features"]
    for entry in sweep["rows"]:
        variants = {}
        for owner in ladder:
            name = owner["owner"]
            exact = name == "parent" or entry[f"{name}_bit_equal"] is True
            if not exact:
                raise ValueError(f"row screen parity failed: {name} {entry['rows']}")
            ms = entry[f"{name}_ms"]
            blocks = ((columns + 31) // 32) * (
                (entry["rows"] + owner["rows_per_block"] - 1) // owner["rows_per_block"]
            )
            variants[name] = {
                "median_ms": ms,
                "vs_parent": round(entry["parent_ms"] / ms, 4),
                "bit_equal_vs_parent": exact,
                "blocks": blocks,
                # Concurrent execution makes duration / blocks an amortized
                # quantity, not a measurement of any block's latency.
                "amortized_us_per_block": round(ms * 1000 / blocks, 4),
            }
        cells.append({"rows": entry["rows"], "variants": variants})
    return cells


def column_tile_sweep(sweep: dict) -> list[dict]:
    cells = []
    for entry in sweep["cells"]:
        owners = {}
        for name, timing in entry["owners"].items():
            if name != "shared_b" and timing.get("bit_exact_vs_shared_b") is not True:
                raise ValueError(f"column screen parity failed: {name} {entry['rows']}")
            out_tiles, waves, row_tiles = GEOMETRIES[name]
            owners[name] = {
                "columns_per_block": 16 * out_tiles,
                "rows_per_block": 16 * waves * row_tiles,
                "threads_per_block": 32 * waves,
                "median_ms": round(timing["median_s"] * 1000, 4),
                "bit_exact_vs_shared_b": timing.get("bit_exact_vs_shared_b"),
            }
        cells.append({"rows": entry["rows"], "owners": owners})
    return cells


def build(resources_path: Path, row_path: Path, col_path: Path) -> dict:
    resources = json.loads(resources_path.read_text())
    row = json.loads(row_path.read_text())
    col = json.loads(col_path.read_text())
    ladder = dual_ladder(resources)
    return {
        "schema": 2,
        "date": "2026-09-13",
        "status": "diagnostic",
        "kind": "q4_dual_wmma_prefill_tile_knob_screen",
        "performance_claim": False,
        "review_correction": "Six-finding review; original raw measurements unchanged.",
        "hardware": {
            "host_name": "gfx1151",
            "device_name": row["device"],
            "target_arch": resources["provenance"]["arch"],
            "physical_vgpr_per_simd_wave32": 1536,
            "vgpr_allocation_granule_wave32": 24,
            "occupancy_measured": False,
        },
        "protocol": {
            "model": row["model"],
            "shape": [row["in_features"], row["out_features"]],
            "burst": row["burst"], "warmups": row["warmups"],
            "repetitions": row["repetitions"],
            "column_sweep_config": col["config"],
        },
        "dual_ladder_resources": ladder,
        "dual_row_tile_sweep": row_tile_sweep(row, ladder),
        "column_tile_sweep": column_tile_sweep(col),
        "findings": [
            "The measured row32/48/64/128 variants lose to the 256-row parent "
            "on these prefill shapes. This does not settle all possible geometries.",
            "The single-matrix column sweep changes other geometry axes too; "
            "it is not an isolated dual-kernel column ablation.",
            "Raw VGPR/SGPR/LDS counts are compiler metadata. Register-only "
            "wave ceilings are not actual occupancy and do not rule out "
            "a residency change from a smaller LDS slab.",
            "Kernel duration divided by grid size is an amortized cost, not "
            "a block latency. Source operation counts do not isolate issue stalls.",
            "The subsequent two col16 arms lost and were removed. The original "
            "instruction-count ranking and later doubled-decode-time explanation "
            "are withdrawn; neither isolates the measured slowdown.",
        ],
        "candidate_directions": [
            {"id": "decode_restructure", "status": "unmeasured",
             "prerequisite": "Inspect ISA, load layout, resource changes and complete-owner cost."},
            {"id": "wider_columns", "status": "unmeasured",
             "prerequisite": "Redesign decode coverage and assess LDS/epilogue resources."},
            {"id": "more_token_rows", "status": "unmeasured",
             "prerequisite": "Assess the 64-KiB epilogue union or a new register-local epilogue."},
            {"id": "residency_retune", "status": "unmeasured",
             "prerequisite": "Use corrected architecture/mode limits and actual resource evidence."},
        ],
        "gate": (
            "Exact ownership and the declared production numerical/task gates, "
            "registered strict fallback, complete-owner screen and same-host "
            "whole-model validation before promotion."
        ),
        "provenance": {
            "resource_capture": resources["provenance"],
            "source_commit": col.get("source_commit"),
            "input_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (resources_path, row_path, col_path)
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, default=HERE / "resource_table.json")
    parser.add_argument("--rowtile", type=Path, default=HERE / "rowtile_prefill.json")
    parser.add_argument("--coltile", type=Path, default=HERE / "col_tile_owners.json")
    parser.add_argument("--out", type=Path, default=HERE / "artifact.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    encoded = json.dumps(build(args.resources, args.rowtile, args.coltile), indent=1) + "\n"
    if args.check:
        if args.out.read_text() != encoded:
            raise SystemExit("artifact differs from deterministic assembly")
    else:
        args.out.write_text(encoded)
    print(f"verified {args.out}" if args.check else f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
