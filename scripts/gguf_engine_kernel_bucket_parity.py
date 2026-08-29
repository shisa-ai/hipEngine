#!/usr/bin/env python3
"""Compare rocprofv3 kernel busy time per generated token across engines.

The published C1-C8 matrix reports one rate per cell, and the wave decomposition
splits that rate into admission and decode. Neither says *which device work* the
losing term is made of. This tool reads `rocprofv3 --kernel-trace` CSV output for
one or more engines and reports, per engine, the device busy time attributed to
each kernel family and the busy time per generated token, optionally restricted
to one wall-clock window (for example one prefill phase).

Read the aggregate honestly: total device busy time per token is comparable
across engines because both run the same workload and denominator. Individual
kernel-family rows are *not* one-to-one comparable, because two engines implement
the same math with differently-named and differently-fused kernels. Use the
per-engine rows to find where one engine spends time it should not, not to claim
"kernel X is slower than kernel Y".

Usage:
    uv run python scripts/gguf_engine_kernel_bucket_parity.py \
        --engine he=/tmp/he_trace/he_kernel_trace.csv \
        --engine current=/tmp/cur/cur_kernel_trace.csv \
        --generated-tokens he=24 --generated-tokens current=24 \
        --window he=START_NS:END_NS --top 25 --output /tmp/parity.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def _short(name: str) -> str:
    return re.sub(r"\s*\(.*$", "", name.split("<")[0]).strip()[:96]


def _load_trace(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            try:
                start = int(float(row["Start_Timestamp"]))
                end = int(float(row["End_Timestamp"]))
            except (KeyError, ValueError, TypeError):
                continue
            name = (row.get("Kernel_Name") or "").strip()
            if not name or end < start:
                continue
            rows.append({"name": name, "start_ns": start, "end_ns": end, "duration_ns": end - start})
    return rows


def _window(raw: str | None) -> tuple[int, int] | None:
    if not raw:
        return None
    start, _, end = raw.partition(":")
    if not start or not end:
        raise ValueError(f"bad --window {raw!r}; expected START_NS:END_NS")
    return int(float(start)), int(float(end))


def _summarize(rows: list[dict], window: tuple[int, int] | None) -> dict:
    selected = [
        row
        for row in rows
        if window is None or (row["start_ns"] >= window[0] and row["end_ns"] <= window[1])
    ]
    busy: dict[str, int] = defaultdict(int)
    calls: dict[str, int] = defaultdict(int)
    for row in selected:
        key = _short(row["name"])
        busy[key] += row["duration_ns"]
        calls[key] += 1
    total = sum(busy.values())
    span = (
        max(row["end_ns"] for row in selected) - min(row["start_ns"] for row in selected)
        if selected
        else 0
    )
    return {
        "launches": len(selected),
        "busy_ns": total,
        "span_ns": span,
        "device_utilization": round(total / span, 6) if span else None,
        "busy_s": round(total / 1e9, 6),
        "families_ms": {name: round(value / 1e6, 4) for name, value in busy.items()},
        "family_launches": dict(calls),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--engine",
        action="append",
        required=True,
        help="label=trace.csv (repeatable); one rocprofv3 kernel-trace CSV per engine",
    )
    parser.add_argument(
        "--generated-tokens",
        action="append",
        default=[],
        help="label=N denominator for busy-ns-per-token; use the generated tokens whose "
        "request window the --window covers",
    )
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        help="label=START_NS:END_NS keep only launches fully inside this window",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    traces = dict(spec.split("=", 1) for spec in args.engine)
    tokens = dict((s.split("=", 1)[0], int(s.split("=", 1)[1])) for s in args.generated_tokens)
    windows = dict((s.split("=", 1)[0], s.split("=", 1)[1]) for s in args.window)

    engines: dict[str, dict] = {}
    for label, raw in traces.items():
        rows = _load_trace(Path(raw))
        summary = _summarize(rows, _window(windows.get(label)))
        denominator = tokens.get(label)
        summary["generated_tokens"] = denominator
        summary["busy_ms_per_token"] = (
            round(summary["busy_ns"] / 1e6 / denominator, 6) if denominator else None
        )
        top = sorted(summary["families_ms"].items(), key=lambda kv: -kv[1])[: int(args.top)]
        summary["top_families_ms"] = dict(top)
        engines[label] = summary
        print(
            f"== {label}: launches={summary['launches']} busy={summary['busy_s']:.3f}s "
            f"span={summary['span_ns'] / 1e9:.3f}s util={summary['device_utilization']}"
            + (
                f" busy_ms_per_token={summary['busy_ms_per_token']:.3f}"
                if summary["busy_ms_per_token"] is not None
                else ""
            )
        )
        for name, ms in top:
            launches = summary["family_launches"].get(name, 0)
            print(f"   {ms:10.3f} ms  {launches:7d} launches  {name}")

    payload = {
        "schema": 1,
        "comparability_note": (
            "busy time per generated token is comparable across engines; individual "
            "kernel-family rows are not, because engines name and fuse kernels differently"
        ),
        "engines": engines,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
