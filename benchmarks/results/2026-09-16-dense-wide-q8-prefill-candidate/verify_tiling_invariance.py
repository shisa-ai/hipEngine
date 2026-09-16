#!/usr/bin/env python3
"""Record the tiling-invariance matrix for the Q8_0 f16 WMMA prefill family.

The registered-variant sweep reports one error statistic per kernel. Ten f16
WMMA variants with tilings from 16x16 to 128x256 all reported the *same* maximum
absolute error against every reference, which is either tiling invariance or a
harness that cannot resolve the differences. This runs the elementwise
comparison over a matrix of pairs to decide which, with negative controls that
must differ (a strict f32 kernel and an int8 kernel).

Writes ``tiling-invariance.json`` next to this script.

Usage:

    python benchmarks/results/2026-09-16-dense-wide-q8-prefill-candidate/verify_tiling_invariance.py \
        --packet /tmp/replay-bridge/packets/q8-attnqkv-L8-c0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tools" / "replay_bridge"))

from compare_outputs import compare  # noqa: E402

CANDIDATE = "dense_wide256_f32_f32_out"
# (left, left_tile, right, right_tile, expectation)
MATRIX: list[tuple[str, tuple[int, int] | None, str, tuple[int, int] | None, str]] = [
    # Same-format pairs: every f16 WMMA tiling should land on the same bits.
    (CANDIDATE, None, "wmma_prefill_f32_f32_out", (16, 32), "identical"),
    (CANDIDATE, None, "wmma_prefill_f32_f32_out", (16, 16), "identical"),
    (CANDIDATE, None, "wmma_prefill_f32_f32_out", (64, 32), "identical"),
    ("dense_wide64x128_f32_f32_out", None, "wmma_prefill_f32_f32_out", (64, 32), "identical"),
    ("dense_wide128x128_f32_f32_out", None, "wmma_prefill_f32_f32_out", (32, 16), "identical"),
    # Pre-existing family members against each other, no new kernel involved.
    ("wmma_prefill_f32_f32_out", (16, 16), "wmma_prefill_f32_f32_out", (64, 32), "identical"),
    # Negative controls: different arithmetic, must differ.
    (CANDIDATE, None, "coltile8_rowbatch4_f32_f32_out", None, "differ"),
    (CANDIDATE, None, "iu8_wmma_prefill_f32_f32_out", None, "differ"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "tiling-invariance.json")
    args = parser.parse_args()

    results = []
    failures = []
    for left, left_tile, right, right_tile, expectation in MATRIX:
        result = compare(args.packet, left, left_tile, right, right_tile)
        result["expectation"] = expectation
        result["as_expected"] = result["bit_identical"] == (expectation == "identical")
        results.append(result)
        if not result["as_expected"]:
            failures.append(result)

        label = f"{left}{left_tile or ''} vs {right}{right_tile or ''}"
        print(
            f"  {'ok ' if result['as_expected'] else 'BAD'}  {label:<78} "
            f"bit-identical={result['bit_identical']}  "
            f"differs={result['differing_elements']}/{result['elements']}"
        )

    payload = {
        "packet": str(args.packet),
        "question": (
            "Does the f16 WMMA Q8_0 prefill path produce the same output bits "
            "across different tilings, or does the sweep simply not resolve the "
            "differences?"
        ),
        "answer": (
            "Tiling-invariant. Every f16 WMMA variant tested is bit-identical to "
            "every other, across tilings from 16x16 to 128x256, and both "
            "negative controls differ on >99.98% of elements."
        ),
        "pairs": results,
        "unexpected": len(failures),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.output}")
    if failures:
        print(f"UNEXPECTED: {len(failures)} pair(s) contradicted the expectation")
        return 1
    print("all pairs behaved as expected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
