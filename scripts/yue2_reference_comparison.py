#!/usr/bin/env python3
"""Compare the hipEngine YuE2 product path against the pinned upstream reference.

The reference numbers come from the oracle's own recorded runs of the twelve
production cases (``artifacts/yue2/oracle/cases/<case>/result.json``), which were
taken on the same host as the hipEngine runs. Both sides decode the same latent
frame counts, so the decoder column is a fully matched comparison; the solver
column is not, because the reference always solves at its own step count while a
hipEngine run may use fewer, so it is reported raw and per step.

Usage:
    python3 scripts/yue2_e2e_gate.py --steps 2 --skip-live --json /tmp/hip.json
    python3 scripts/yue2_reference_comparison.py /tmp/hip.json \
        --steps 2 --json benchmarks/results/yue2_reference_comparison.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ORACLE = REPO / "artifacts/yue2/oracle/cases"
REFERENCE_STEPS = 32


def load_reference(oracle: Path) -> dict:
    reference = {}
    for path in sorted(oracle.glob("*/result.json")):
        result = json.loads(path.read_text())
        timing = result["timing"]
        reference[path.parent.name] = {
            "ar_seconds": float(timing["semantic"]["seconds"]),
            "ar_tokens": int(timing["semantic"]["output_tokens"]),
            "nar_seconds": float(timing["nar_seconds"]),
            "vae_seconds": float(timing["vae_seconds"]),
            "e2e_seconds": float(timing["e2e_seconds"]),
        }
    return reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate_json", type=Path)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--steps", type=int, required=True,
                        help="ODE steps the hipEngine gate ran with")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    gate = json.loads(args.gate_json.read_text())
    reference = load_reference(args.oracle_dir)
    if not reference:
        raise SystemExit(f"no reference results under {args.oracle_dir}")

    missing = [
        name for name, case in gate["cases"].items()
        if "nar_seconds" not in case or "vae_seconds" not in case
    ]
    if missing:
        raise SystemExit(
            "gate JSON has no per-stage timing for "
            f"{len(missing)} case(s) (first: {missing[0]}); re-run "
            "scripts/yue2_e2e_gate.py, which records nar_seconds and vae_seconds"
        )

    rows = {}
    for name in sorted(gate["cases"]):
        case = gate["cases"][name]
        ref = reference.get(name)
        if ref is None:
            continue
        hip_nar = float(case["nar_seconds"])
        hip_vae = float(case["vae_seconds"])
        rows[name] = {
            "frames": int(case["frames"]),
            "hip_nar_seconds": round(hip_nar, 2),
            "hip_vae_seconds": round(hip_vae, 2),
            "ref_nar_seconds": round(ref["nar_seconds"], 2),
            "ref_vae_seconds": round(ref["vae_seconds"], 2),
            "ref_ar_seconds": round(ref["ar_seconds"], 2),
            # The decoder is step-independent, so this ratio is matched.
            "vae_speedup": round(ref["vae_seconds"] / hip_vae, 2) if hip_vae else None,
            # The solver is not: report raw and per-step.
            "nar_ratio_raw": round(hip_nar / ref["nar_seconds"], 2),
            "nar_ratio_per_step": round(
                (hip_nar / args.steps) / (ref["nar_seconds"] / REFERENCE_STEPS), 2
            ),
        }

    total = {
        key: round(sum(row[key] for row in rows.values()), 2)
        for key in ("hip_nar_seconds", "hip_vae_seconds", "ref_nar_seconds", "ref_vae_seconds")
    }
    total["vae_speedup"] = round(total["ref_vae_seconds"] / total["hip_vae_seconds"], 2)
    total["nar_ratio_raw"] = round(total["hip_nar_seconds"] / total["ref_nar_seconds"], 2)
    total["nar_ratio_per_step"] = round(
        (total["hip_nar_seconds"] / args.steps) / (total["ref_nar_seconds"] / REFERENCE_STEPS), 2
    )

    print(f"hipEngine {args.steps} ODE steps vs the pinned upstream reference "
          f"({REFERENCE_STEPS} steps), same host")
    print(f"{'case':26s} {'frames':>7s} {'hip VAE':>9s} {'ref VAE':>9s} {'x':>6s} "
          f"{'hip NAR':>9s} {'ref NAR':>9s} {'raw':>7s} {'/step':>7s}")
    for name, row in sorted(rows.items(), key=lambda kv: kv[1]["frames"]):
        print(
            f"{name:26s} {row['frames']:7d} {row['hip_vae_seconds']:9.2f} "
            f"{row['ref_vae_seconds']:9.2f} {row['vae_speedup']:6.2f} "
            f"{row['hip_nar_seconds']:9.2f} {row['ref_nar_seconds']:9.2f} "
            f"{row['nar_ratio_raw']:7.2f} {row['nar_ratio_per_step']:7.2f}"
        )
    print(
        f"{'TOTAL':26s} {sum(r['frames'] for r in rows.values()):7d} "
        f"{total['hip_vae_seconds']:9.2f} {total['ref_vae_seconds']:9.2f} "
        f"{total['vae_speedup']:6.2f} {total['hip_nar_seconds']:9.2f} "
        f"{total['ref_nar_seconds']:9.2f} {total['nar_ratio_raw']:7.2f} "
        f"{total['nar_ratio_per_step']:7.2f}"
    )

    if args.json:
        payload = {
            "provenance": {
                "command_line": " ".join(sys.argv),
                "gate_json": str(args.gate_json),
                "oracle_dir": str(args.oracle_dir),
                "hipengine_steps": args.steps,
                "reference_steps": REFERENCE_STEPS,
                "revision": subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=REPO,
                    capture_output=True, text=True, check=False,
                ).stdout.strip(),
                "note": "decoder ratios are matched (step-independent); solver ratios are "
                        "reported raw and per step because the step counts differ",
            },
            "cases": rows,
            "total": total,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"[compare] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
