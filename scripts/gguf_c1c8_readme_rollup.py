"""Roll a C1-C8 server-bench packet into the README complete-wall matrix rows.

Published aggregate convention (matches benchmarks/README.md and the cross-engine
artifact): per width, mean(generated_tokens) / mean(wall_seconds) per arm, three
decimals. Prints the refreshed hipEngine rows, the delta against the values
currently printed in the README, and the bold/winner assignment across the three
comparable rows for each arm.

Usage: gguf_c1c8_readme_rollup.py PACKET [--widths 1,2,...,8]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

# A point-in-time SNAPSHOT, not the baseline: the baseline is parsed from benchmarks/README.md at
# runtime (see `read_readme_rows`). This exists only so the tool still runs if a label is renamed.
# It drifted once already - it still says C8 AR 44.338 while the README has published 78.667 since
# the grouped-prefill promotion, and a delta computed against it invented +76% that never happened.
SNAPSHOT_ROWS = {
    "ar": {
        "hipengine": [21.871, 30.455, 35.625, 39.231, 41.124, 42.490, 43.570, 44.338],
        "llama_current": [21.720, 35.440, 30.787, 27.760, 36.390, 45.529, 51.914, 58.744],
        "llama_laurent": [21.463, 35.100, 30.635, 27.667, 36.473, 45.826, 52.537, 59.348],
    },
    "k3": {
        "hipengine": [31.146, 29.639, 30.547, 31.033, 30.755, 30.248, 30.596, 30.845],
        "llama_current": [32.553, 41.042, 45.324, 49.977, 59.644, 72.195, 75.354, 94.735],
        "llama_laurent": [32.733, 40.808, 45.947, 51.054, 61.013, 74.628, 78.281, 101.072],
    },
}
# Back-compat name kept for any caller that imports this module directly.
README_ROWS = SNAPSHOT_ROWS
LABELS = {"ar": "hipEngine AR", "k3": "hipEngine explicit K3"}
README_PATH = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "README.md"
# README row label -> (arm, role) in the row-table shape above.
ROW_LABELS = {
    "hipEngine AR": ("ar", "hipengine"),
    "llama.cpp current HIP AR": ("ar", "llama_current"),
    "llama.cpp Laurent HIP AR": ("ar", "llama_laurent"),
    "hipEngine explicit K3": ("k3", "hipengine"),
    "llama.cpp current HIP K3": ("k3", "llama_current"),
    "llama.cpp Laurent HIP K3": ("k3", "llama_laurent"),
}


def parse_markdown_row(line: str) -> list[float] | None:
    """Parse one markdown table row into floats, or None if it is not a numeric row.

    Handles the published bold convention (`**78.667**`) and ignores separator/label-only rows.
    """

    parts = [p.strip() for p in line.strip().strip("|").split("|")]
    if len(parts) < 3:
        return None
    values: list[float] = []
    for cell in parts[1:]:
        text = cell.strip().strip("*").strip()
        try:
            values.append(float(text))
        except ValueError:
            return None
    return values


def read_readme_rows(path: pathlib.Path | None = None) -> dict[str, dict[str, list[float]]]:
    """Read the C1-C8 complete-wall rows straight out of benchmarks/README.md."""

    target = README_PATH if path is None else pathlib.Path(path)
    rows: dict[str, dict[str, list[float]]] = {
        arm: {role: [] for role in ("hipengine", "llama_current", "llama_laurent")}
        for arm in ("ar", "k3")
    }
    if not target.is_file():
        return rows
    for line in target.read_text().splitlines():
        if not line.lstrip().startswith("|"):
            continue
        label = line.lstrip().strip().strip("|").split("|")[0].strip()
        mapped = ROW_LABELS.get(label)
        if mapped is None:
            continue
        parsed = parse_markdown_row(line)
        if parsed and len(parsed) == 8:
            arm, role = mapped
            rows[arm][role] = parsed
    return rows


def baseline_rows(*, use_snapshot: bool = False) -> tuple[dict, str]:
    """Return (rows, source_label); falls back to the snapshot only with a loud warning."""

    if not use_snapshot:
        parsed = read_readme_rows()
        missing = [
            f"{arm}/{role}"
            for arm, roles in parsed.items()
            for role, v in roles.items()
            if not v
        ]
        if not missing:
            return parsed, "benchmarks/README.md (parsed live)"
        print(f"WARNING: could not parse {missing} from benchmarks/README.md; "
              "falling back to the frozen SNAPSHOT_ROWS, whose deltas may be stale.")
    return SNAPSHOT_ROWS, "frozen snapshot (may be stale)"


def aggregate(packet: dict, width: int, arm: str) -> float:
    cells = [c for c in packet["cells"] if c["width"] == width]
    if not cells:
        raise SystemExit(f"packet has no cells for width {width}")
    tokens = statistics.mean(c[arm]["generated_tokens"] for c in cells)
    wall = statistics.mean(c[arm]["wall_seconds"] for c in cells)
    return tokens / wall


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("packet")
    ap.add_argument("--widths", default="1,2,3,4,5,6,7,8")
    ap.add_argument(
        "--baseline",
        choices=("readme", "snapshot"),
        default="readme",
        help="what to diff against; README is parsed live so deltas are real",
    )
    args = ap.parse_args()
    widths = [int(w) for w in args.widths.split(",")]
    packet = json.load(open(args.packet))
    README_ROWS, baseline_source = baseline_rows(use_snapshot=args.baseline == "snapshot")
    print(f"packet status={packet.get('status')} passed={packet.get('passed')}")
    print(f"baseline: {baseline_source}")
    for arm, key in (("ar", "ar"), ("mtp", "k3")):
        new = [round(aggregate(packet, w, arm), 3) for w in widths]
        old = [README_ROWS[key]["hipengine"][w - 1] for w in widths]
        print(f"\n{LABELS[key]}")
        for width, before, after in zip(widths, old, new):
            others = {
                row: README_ROWS[key][row][width - 1]
                for row in ("llama_current", "llama_laurent")
            }
            winner = max([("hipEngine", after)] + list(others.items()), key=lambda kv: kv[1])[0]
            delta = (after - before) / before * 100
            print(
                f"  C{width}: {before:8.3f} -> {after:8.3f}  {delta:+6.2f}%   winner: {winner}"
            )
        def cell(idx: int, value: float) -> str:
            width = widths[idx]
            peers = [README_ROWS[key][r][width - 1] for r in ("llama_current", "llama_laurent")]
            return f"**{value:.3f}**" if value >= max(peers) else f"{value:.3f}"

        print(f"  | {LABELS[key]} | " + " | ".join(cell(i, v) for i, v in enumerate(new)) + " |")
        # Full comparable block with bold recomputed across all three rows, because
        # a refreshed hipEngine value can move a column's bold to or from a peer row.
        rows = {"hipEngine AR" if key == "ar" else "hipEngine explicit K3": list(new)}
        for row, label in (
            ("llama_current", "llama.cpp current HIP " + key.upper()),
            ("llama_laurent", "llama.cpp Laurent HIP " + key.upper()),
        ):
            rows[label] = [README_ROWS[key][row][w - 1] for w in widths]
        print(f"  --- full {key.upper()} block (bold recomputed) ---")
        for label, values in rows.items():
            out = []
            for idx, value in enumerate(values):
                best = max(row_values[idx] for row_values in rows.values())
                out.append(f"**{value:.3f}**" if value >= best else f"{value:.3f}")
            print(f"  | {label} | " + " | ".join(out) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
