#!/usr/bin/env python3
"""Decode or watch a qwen35 GGUF persistent prefill flight-recorder file."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.runtime.prefill_flight_recorder import read_prefill_flight_recorder


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Binary recorder path")
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=0.0,
        help="Poll interval; zero prints one snapshot and exits",
    )
    parser.add_argument(
        "--entries",
        type=int,
        default=8,
        help="Number of most recent ring entries to include",
    )
    return parser.parse_args(argv)


def compact_snapshot(path: Path, *, entries: int) -> dict[str, object]:
    if entries < 0:
        raise ValueError("--entries must be non-negative")
    snapshot = read_prefill_flight_recorder(path)
    ring = list(snapshot.get("entries", []))
    snapshot["entries"] = ring[-entries:] if entries else []
    return snapshot


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.watch_seconds < 0:
        raise ValueError("--watch-seconds must be non-negative")
    previous_cursor: tuple[int, int] | None = None
    while True:
        snapshot = compact_snapshot(args.path, entries=args.entries)
        cursor = (
            int(snapshot["submitted_sequence"]),
            int(snapshot["completed_sequence"]),
        )
        if args.watch_seconds <= 0:
            print(json.dumps(snapshot, indent=2, sort_keys=True))
            return 0
        if cursor != previous_cursor:
            print(json.dumps(snapshot, separators=(",", ":"), sort_keys=True), flush=True)
            previous_cursor = cursor
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
