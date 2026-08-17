#!/usr/bin/env python3
"""PN3 evidence: reproduce the c1 decode stage ranking from GPU timestamps.

Computes nested-exclusive GPU-visible wall per decode-stage role from an
existing ``rocprofv3`` marker trace (the PN2 ``c1-profile-raw/trace``), using
the ROCTX range markers' Start/End GPU timestamps (which ARE populated even
though kernel durations report 0 on this gfx1151/ROCm combo).

The PN2 "same_stream_device_wall_clock_stages" authority over-attributes the
``decode_linear_attn_qkv_gate`` stage (7.74 ms/token host wall) because host-
wall stage windows include pipeline bubbles and loose enclosing scopes. The
GPU-timestamped nested-exclusive walls give the trustworthy fresh ranking that
PN3 must select a candidate from.

Nesting: ROCTX ranges nest (a parent like ``gdn_attention_core`` encloses the
input/decay/output projection ranges). Exclusive wall per range = its span
minus its direct children spans (grandchildren are not double-subtracted).
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

DECODE_ROLE_PREFIX = "hipengine_gguf_decode_role:"
DECODE_STEP_PREFIX = "hipengine_gguf_eager_decode_step_"


def _direct_parents(ranges: list[tuple[int, int, str]]) -> list[int | None]:
    """Return the direct-parent index for each range (proper nesting)."""
    parents: list[int | None] = [None] * len(ranges)
    stack: list[tuple[int, int]] = []  # (end, idx)
    for i, (st, en, _role) in enumerate(ranges):
        while stack and stack[-1][0] <= st:
            stack.pop()
        if stack:
            parents[i] = stack[-1][1]
        stack.append((en, i))
    return parents


def exclusive_walls(
    ranges: list[tuple[int, int, str]],
) -> tuple[dict[str, float], dict[str, int], dict[str, float]]:
    """Return (exclusive_us_by_role, count_by_role, per_range_us_by_role)."""
    ordered = sorted(ranges, key=lambda x: (x[0], -x[1]))
    parents = _direct_parents(ordered)
    children: dict[int, list[int]] = collections.defaultdict(list)
    for i, p in enumerate(parents):
        if p is not None:
            children[p].append(i)
    excl: dict[str, float] = collections.defaultdict(float)
    cnt: dict[str, int] = collections.defaultdict(int)
    per: dict[str, list[float]] = collections.defaultdict(list)
    for i, (st, en, role) in enumerate(ordered):
        e = en - st
        for c in children.get(i, []):
            e -= ordered[c][1] - ordered[c][0]
        e = max(0, e)
        excl[role] += e
        cnt[role] += 1
        per[role].append(e)
    return excl, cnt, per


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path(
            "/tmp/hipengine-zbook-production-numerics/"
            "20260817T040401Z-53be617b835b/c1-profile-raw/trace"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/pn3-stage-ranking.json"))
    parser.add_argument("--tokens", type=int, default=24)
    args = parser.parse_args()

    marker_csv = next(args.trace_dir.glob("*_marker_api_trace.csv"))
    rows = list(csv.DictReader(open(marker_csv)))

    steps = [r for r in rows if r["Function"].startswith(DECODE_STEP_PREFIX)]
    if not steps:
        raise SystemExit("no eager_decode_step markers found")
    t0 = min(int(r["Start_Timestamp"]) for r in steps)

    ranges: list[tuple[int, int, str]] = []
    for r in rows:
        if not r["Function"].startswith(DECODE_ROLE_PREFIX):
            continue
        st = int(r["Start_Timestamp"])
        if st < t0:
            continue
        ranges.append((st, int(r["End_Timestamp"]), r["Function"][len(DECODE_ROLE_PREFIX):]))

    excl, cnt, per = exclusive_walls(ranges)
    tokens = args.tokens
    ranking = []
    for role in sorted(excl, key=lambda r: -excl[r]):
        # excl[role] is summed nanoseconds of GPU-timestamped range walls.
        ranking.append(
            {
                "role": role,
                "markers": cnt[role],
                "exclusive_ms_total": excl[role] / 1e6,
                "ms_per_token": excl[role] / 1e6 / tokens,
                "median_us_per_marker": sorted(per[role])[len(per[role]) // 2] / 1e3,
            }
        )
    result = {
        "schema": 1,
        "source_trace": str(marker_csv),
        "decode_tokens": tokens,
        "t0_timestamp": t0,
        "method": "nested-exclusive GPU-visible wall from ROCTX range markers",
        "ranking": ranking,
    }
    text = json.dumps(result, indent=2)
    print(text)
    args.output.write_text(text + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
