#!/usr/bin/env python3
"""Compare the risk-queue depth the repair pass actually sees, across multipliers.

The repair kernels' measured time is flat across the risk multiplier (4808.702 ms
at 4.0, 4844.580 ms at 1.0) while their grid-stride loop bound is
``min(risk_count, max_risks)``, so the work should scale.  The missing number is
``risk_count`` itself.  ``scripts/qwen4exp_profile_gap.py --risk-diagnostics``
records it per (route, role) and summarizes it under ``risk_diagnostics``.

Usage:
  python3 scripts/qwen4exp_risk_count_compare.py /tmp/risk-counts/risk-4.0.json \
      /tmp/risk-counts/risk-1.0.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def collect(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    report = json.loads(path.read_text())
    summary = report.get("risk_diagnostics") or {}
    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, bucket in (summary.get("by_role") or {}).items():
        route, role = (key.split("/", 1) + [""])[:2] if isinstance(key, str) else key
        out[(route, role)] = {
            "calls": float(bucket.get("calls") or 0),
            "outputs": float(bucket.get("outputs") or 0),
            "risk": float(bucket.get("risk") or 0),
            "max_repair_rate": float(bucket.get("max_repair_rate") or 0.0),
        }
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    arms = [(Path(p).stem, collect(Path(p))) for p in argv[1:]]
    keys = sorted({k for _, data in arms for k in data})
    header = f"{'route/role':46s}" + "".join(f"{name:>22s}" for name, _ in arms)
    print(header)
    for key in keys:
        label = f"{key[0]}/{key[1]}"
        row = f"{label:46s}"
        values = []
        for _, data in arms:
            bucket = data.get(key)
            if bucket is None:
                row += f"{'-':>22s}"
                values.append(None)
            else:
                row += f"{int(bucket['risk']):>22d}"
                values.append(bucket)
        print(row)
        if len(values) == 2 and all(values):
            a, b = values
            if b["risk"] > 0:
                print(f"{'  ratio ' + arms[0][0] + '/' + arms[1][0]:46s}"
                      f"{a['risk'] / b['risk']:>21.2f}x")
            print(f"{'  outputs 4.0 / 1.0':46s}"
                  f"{int(a['outputs']):>10d} / {int(b['outputs']):<10d}")
            print(f"{'  repair rate 4.0 / 1.0':46s}"
                  f"{a['max_repair_rate']:>10.5f} / {b['max_repair_rate']:<10.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
