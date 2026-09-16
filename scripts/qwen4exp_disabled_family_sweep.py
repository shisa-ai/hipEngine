#!/usr/bin/env python3
"""Measure recoverable time for each disabled Qwen4Exp prefill family.

For every family in ``PRODUCTION_ARITHMETIC_RECOVERY_FLAGS`` that the current
default leaves off, run the same production prefill twice: once as the current
default (the fallback) and once with that family's fast path enabled. Record the
wall time and, where the family routes through the exact iu8 risk+repair
instrument, the repair rate the fast path would have to repair.

The ranking is by seconds recoverable on the measured prefill, not by any
historical headline gain. A family that is fast but numerically inadmissible
still appears here with its speedup, because the speedup is what a future
corrected arithmetic would be worth; admissibility is a separate axis and is
recorded from the project's own rejection evidence rather than re-derived.

This is a diagnostic. Its runs are not rate rows and it does not claim a
performance result.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Each entry: family name, the env key that enables it, the value that enables
# it, and any companion key/value the family needs.
FAMILIES: tuple[dict, ...] = (
    {
        "family": "PRODUCTION_MOE_PREFILL",
        "env": {"HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL": "1"},
    },
    {
        "family": "Q4_IU8_PREFILL",
        "env": {
            "HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL": "1",
            "HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS": ",".join(map(str, range(35, 48))),
        },
    },
    {
        "family": "Q8_MMQ_PREFILL",
        "env": {"HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL": "1"},
    },
    {
        "family": "Q8_IU8_WMM",
        "env": {"HIPENGINE_QWEN4_EXP_Q8_IU8_WMM": "1"},
    },
    # The f16 WMMA dense Q8_0 selector is a layer list rather than a flag, and
    # it is not one of the recovery flags, so it was absent from the original
    # ranking. Its numerical envelope is gated in
    # benchmarks/results/2026-09-16-q8-wmma-dense-prefill-layers-gate/ at both
    # scopes; these arms supply the matching performance half.
    {
        "family": "Q8_WMMA_LAYERS_32_47",
        "env": {
            "HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS": ",".join(map(str, range(32, 48)))
        },
    },
    {
        "family": "Q8_WMMA_LAYERS_0_47",
        "env": {
            "HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS": ",".join(map(str, range(0, 48)))
        },
    },
    {
        "family": "GR_IU8",
        "env": {"HIPENGINE_QWEN4_EXP_GR_IU8": "1"},
    },
    {
        "family": "GR_IU8_DOWN",
        "env": {"HIPENGINE_QWEN4_EXP_GR_IU8_DOWN": "1"},
    },
    {
        "family": "GDN_PEER_PREFILL",
        "env": {
            "HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL": "1",
            "HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL_LAYERS": ",".join(map(str, range(35, 48))),
        },
    },
    {
        "family": "GDN_COLWARPS_PREFILL",
        "env": {
            "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL": "1",
            "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS": ",".join(map(str, range(27, 48))),
        },
    },
    {
        "family": "Q4_DP4A64",
        "env": {
            "HIPENGINE_QWEN4_EXP_Q4_DP4A64": "1",
            "HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS": ",".join(
                map(str, (0, 2, 5, 6, 8, 9, 10, 11) + tuple(range(13, 48)))
            ),
        },
    },
)


def _run(
    python: str,
    model_root: Path,
    fixture: Path,
    case_id: str,
    repetitions: int,
    overrides: dict[str, str],
    output: Path,
    reuse: bool = False,
) -> dict:
    if reuse and output.is_file():
        payload = json.loads(output.read_text())
        return _summarize(payload, ["reused", str(output)])
    command = [
        python,
        str(ROOT / "scripts" / "qwen4exp_profile_gap.py"),
        "--model-root", str(model_root),
        "--mode", "prefill",
        "--fixture", str(fixture),
        "--case-id", case_id,
        "--repetitions", str(repetitions),
        "--risk-diagnostics",
        "--output", str(output),
    ]
    for key, value in overrides.items():
        command += ["--override", f"{key}={value}"]
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=3600
    )
    if completed.returncode != 0:
        return {
            "ok": False,
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-1500:],
            "command": command,
        }
    return _summarize(json.loads(output.read_text()), command)


def _summarize(payload: dict, command) -> dict:
    risk = payload.get("risk_diagnostics") or {}
    # The instrument reports per role and per layer; there is no precomputed
    # total, so one is derived from the outputs and risk counts it records.
    by_role = risk.get("by_role") or {}
    outputs = sum(int(v.get("outputs") or 0) for v in by_role.values())
    risky = sum(int(v.get("risk") or 0) for v in by_role.values())
    max_rate = max(
        (float(v["max_repair_rate"]) for v in by_role.values()
         if v.get("max_repair_rate") is not None),
        default=None,
    )
    over_capacity = sum(
        int(v.get("over_capacity_calls") or 0) for v in by_role.values()
    )
    return {
        "ok": True,
        "command": command,
        "wall_summary": payload.get("wall_summary"),
        "wall_seconds": payload.get("wall_seconds"),
        "route_env": payload.get("route_env"),
        "configuration_class": payload.get("configuration_class"),
        "fell_back_to_strict": payload.get("fell_back_to_strict"),
        "named_profile_intact": payload.get("named_profile_intact"),
        "risk_calls": risk.get("calls"),
        "risk_outputs": outputs,
        "risk_risky": risky,
        "aggregate_repair_rate": (risky / outputs) if outputs else None,
        "max_repair_rate": max_rate,
        "over_capacity_calls": over_capacity,
        "risk_by_role": {k: v.get("aggregate_repair_rate") for k, v in by_role.items()},
        "logits_sha256": payload.get("logits_sha256"),
        "token_id": payload.get("token_id"),
    }


def _median_wall(result: dict) -> float | None:
    if not result.get("ok"):
        return None
    summary = result.get("wall_summary") or {}
    value = summary.get("median")
    return float(value) if value is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--case-id", default="code-p4096")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--only", action="append", default=None,
                        help="restrict to these family names")
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument(
        "--reuse-scratch",
        action="store_true",
        help=(
            "Aggregate the runs already present in --scratch instead of "
            "re-measuring. The per-family JSONs are the measurement; this "
            "only rebuilds the ranking from them."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.scratch.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only) if args.only else None
    families = [f for f in FAMILIES if wanted is None or f["family"] in wanted]

    print("=== fallback (current default) ===", flush=True)
    fallback = _run(
        args.python, args.model_root, args.fixture, args.case_id,
        args.repetitions, {}, args.scratch / "fallback.json",
        reuse=args.reuse_scratch,
    )
    fallback_wall = _median_wall(fallback)
    print(f"  median prefill {fallback_wall} s  ok={fallback.get('ok')}", flush=True)

    rows = []
    for entry in families:
        name = entry["family"]
        print(f"=== {name} ===", flush=True)
        result = _run(
            args.python, args.model_root, args.fixture, args.case_id,
            args.repetitions, entry["env"], args.scratch / f"{name}.json",
            reuse=args.reuse_scratch,
        )
        wall = _median_wall(result)
        if wall is None or fallback_wall is None:
            print(f"  FAILED: {json.dumps(result)[:300]}", flush=True)
            rows.append({
                "family": name,
                "env": entry["env"],
                "measured": False,
                "failure": result.get("stderr_tail") or result.get("returncode"),
            })
            continue
        saved = fallback_wall - wall
        rows.append({
            "family": name,
            "env": entry["env"],
            "measured": True,
            "candidate_median_s": round(wall, 4),
            "fallback_median_s": round(fallback_wall, 4),
            "seconds_recoverable": round(saved, 4),
            "speedup": round(fallback_wall / wall, 4),
            "aggregate_repair_rate": result.get("aggregate_repair_rate"),
            "max_repair_rate": result.get("max_repair_rate"),
            "over_capacity_calls": result.get("over_capacity_calls"),
            "logits_sha256": result.get("logits_sha256"),
            "same_logits_as_fallback": (
                result.get("logits_sha256") == fallback.get("logits_sha256")
            ),
            "named_profile_intact": result.get("named_profile_intact"),
            "fell_back_to_strict": result.get("fell_back_to_strict"),
        })
        print(
            f"  median {wall:.3f} s  saved {saved:+.3f} s  "
            f"speedup {fallback_wall / wall:.3f}x  "
            f"repair {result.get('aggregate_repair_rate')}",
            flush=True,
        )

    measured = [r for r in rows if r["measured"]]
    ranked = sorted(measured, key=lambda r: -r["seconds_recoverable"])
    artifact = {
        "schema": 1,
        "kind": "qwen4exp_disabled_family_recoverable_time",
        "performance_claim": False,
        "numerics_evaluated": False,
        "why": (
            "Each disabled family is worth whatever time its fast path saves on a "
            "real production prefill, but only if a corrected arithmetic can be "
            "made admissible. This measures the first half; admissibility is a "
            "separate axis recorded from the project's own rejection evidence."
        ),
        "model_root": str(args.model_root),
        "fixture": str(args.fixture),
        "case_id": args.case_id,
        "repetitions": args.repetitions,
        "fallback": fallback,
        "fallback_median_s": fallback_wall,
        "families": rows,
        "ranked_by_seconds_recoverable": [
            {"family": r["family"], "seconds_recoverable": r["seconds_recoverable"],
             "speedup": r["speedup"]}
            for r in ranked
        ],
        "total_seconds_recoverable": round(
            sum(r["seconds_recoverable"] for r in ranked), 4
        ),
        "all_candidates_bit_identical": all(
            r["same_logits_as_fallback"] for r in ranked
        ) if ranked else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=1) + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
