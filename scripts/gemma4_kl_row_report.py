#!/usr/bin/env python3
"""Per-row KL between two Gemma 4 teacher-forced captures.

``scripts/gemma4_teacher_forced_gate.py gate`` reports the KL distribution but
does not keep it. When one row breaches ``kl_max`` while the mean, p95, p99 and
top-1 all pass, the diagnosis needs the row index and the two distributions at
that row, so this replays the same comparison row by row.

Usage::

    python3 scripts/gemma4_kl_row_report.py <baseline.npz> <candidate.npz> [--top 10]
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - float(logits.max())
    return shifted - np.log(np.exp(shifted).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--top", type=int, default=10, help="worst rows to describe")
    args = parser.parse_args()

    baseline = np.load(args.baseline)
    candidate = np.load(args.candidate)
    base_logits = baseline["logits"]
    cand_logits = candidate["logits"]
    if base_logits.shape != cand_logits.shape:
        raise SystemExit(f"shape mismatch: {base_logits.shape} vs {cand_logits.shape}")

    rows = base_logits.shape[0]
    kl = np.empty(rows, dtype=np.float64)
    flipped = np.empty(rows, dtype=bool)
    for row in range(rows):
        base_lp = _log_softmax(base_logits[row])
        cand_lp = _log_softmax(cand_logits[row])
        kl[row] = float(np.sum(np.exp(base_lp) * (base_lp - cand_lp)))
        flipped[row] = int(base_logits[row].argmax()) != int(cand_logits[row].argmax())

    order = np.argsort(kl)[::-1]
    print(
        json.dumps(
            {
                "rows": int(rows),
                "kl_mean": float(kl.mean()),
                "kl_p50": float(np.percentile(kl, 50)),
                "kl_p95": float(np.percentile(kl, 95)),
                "kl_p99": float(np.percentile(kl, 99)),
                "kl_max": float(kl.max()),
                "top1_flips": int(flipped.sum()),
                "rows_above_1e-3": int((kl > 1e-3).sum()),
                "rows_above_1e-2": int((kl > 1e-2).sum()),
                "rows_above_5e-2": int((kl > 5e-2).sum()),
            },
            indent=1,
        )
    )

    print("\nworst rows (index = scored position, key = 1025 + index):")
    for rank, row in enumerate(order[: args.top]):
        row = int(row)
        base_row = base_logits[row].astype(np.float64)
        cand_row = cand_logits[row].astype(np.float64)
        base_lp = _log_softmax(base_logits[row])
        cand_lp = _log_softmax(cand_logits[row])
        base_top = np.argsort(base_row)[::-1][:4]
        cand_top = np.argsort(cand_row)[::-1][:4]
        print(
            f"  #{rank + 1} row {row} key {1025 + row} kl={kl[row]:.6g} "
            f"flip={bool(flipped[row])} "
            f"max|dlogit|={np.abs(base_row - cand_row).max():.4g} "
            f"logit_scale={np.abs(base_row).max():.4g}"
        )
        print(
            "      baseline top: "
            + " ".join(f"{int(t)}:{float(np.exp(base_lp[t])):.4f}" for t in base_top)
        )
        print(
            "      candidate top:"
            + " ".join(f"{int(t)}:{float(np.exp(cand_lp[t])):.4f}" for t in cand_top)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
